"""
Gemini 2.5 Pro API Model Adapter for lmms-eval

This module provides integration with Gemini 2.5 Pro.

Usage:
    python -m lmms_eval \
        --model gemini25_pro \
        --model_args model_version=aihub/gemini-2.5-pro,max_frames_num=2,skip_audio=True \
        --tasks omni_test_local \
        --batch_size 1

Environment Variables (optional, can also be passed via model_args):
    GEMINI25_PRO_API_KEY: API key for Gemini 2.5 Pro
    GEMINI_BASE_URL: Base URL for the API
"""

import base64
import io
import json
import os
import pathlib
import time
from io import BytesIO
from typing import Dict, List, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.env import get_env_value

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai.types import HttpOptions
except ImportError as e:
    eval_logger.warning(f"Error importing google.genai: {str(e)}. Please install with: pip install google-genai")
    genai = None
    genai_types = None

try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("decord not available, video processing may not work")
    VideoReader = None
    cpu = None

NUM_SECONDS_TO_SLEEP = 10


@register_model("gemini25_pro")
class Gemini25Pro(lmms):
    """
    Gemini 2.5 Pro model adapter using google.genai SDK.
    
    Supports multi-modal inputs: images, videos, audio, and text.
    """
    
    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages (system prompt)
    
    def __init__(
        self,
        model_version: str = "aihub/gemini-2.5-pro",
        api_key: str = None,
        base_url: str = None,
        api_version: str = "v1/beta/google/gemini",
        timeout: int = 120,
        max_retries: int = 5,
        max_frames_num: int = 4,
        skip_audio: bool = False,  # Pro supports audio
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        streaming: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize Gemini 2.5 Pro model.
        
        Args:
            model_version: Model name (default: aihub/gemini-2.5-pro)
            api_key: API key (default: from model_args or env)
            base_url: API base URL (default: from model_args or env)
            api_version: API version path
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            max_frames_num: Maximum frames to extract from videos
            skip_audio: Whether to skip audio content
            continual_mode: Enable response caching
            response_persistent_folder: Folder for cached responses
            streaming: Whether to use streaming mode (not recommended for eval)
        """
        super().__init__()
        
        if genai is None:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_frames_num = max_frames_num
        self.skip_audio = skip_audio
        self.streaming = streaming
        self.continual_mode = continual_mode
        
        # API configuration
        self.api_key = api_key or get_env_value(
            ["GEMINI25_PRO_API_KEY", "GEMINI_API_KEY", "GEMINI3_API_KEY", "GOOGLE_API_KEY"],
            required=True,
            description="Gemini 2.5 Pro API key",
        )
        self.base_url = base_url or get_env_value(
            ["GEMINI_BASE_URL", "GEMINI3_BASE_URL"],
            required=True,
            description="Gemini base URL",
        )
        self.api_version = api_version
        
        # Initialize client
        http_options = HttpOptions(base_url=self.base_url, api_version=self.api_version)
        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        
        eval_logger.info(f"Initialized Gemini25Pro with model: {self.model_version}")
        eval_logger.info(f"API Base URL: {self.base_url}")
        
        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/gemini25_pro_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, 
                f"{self.model_version.replace('/', '_')}_response.json"
            )
            
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"
        
        # Distributed setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        
        self.device = self.accelerator.device

    def encode_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """Load and convert image to PIL Image."""
        if isinstance(image, str):
            if not os.path.exists(image):
                eval_logger.warning(f"Image file not found: {image}, returning placeholder")
                # Return a small placeholder image instead of crashing
                return Image.new("RGB", (100, 100), (128, 128, 128))
            return Image.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    def encode_image_to_base64(self, image: Union[Image.Image, str]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()
        
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        return base64.b64encode(byte_data).decode("utf-8")

    def encode_video_to_frames(self, video_path: str, num_frames: int = None) -> List[Image.Image]:
        """Extract frames from video file."""
        if VideoReader is None:
            eval_logger.warning("decord not available, skipping video")
            return []
        
        if num_frames is None:
            num_frames = self.max_frames_num
            
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            
            if total_frame_num == 0:
                return []
            
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
            
            pil_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                pil_frames.append(img)
            
            return pil_frames
        except Exception as e:
            eval_logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    def load_audio(self, audio_path: str):
        """Load audio file and return as genai Part object."""
        try:
            audio_data = pathlib.Path(audio_path).read_bytes()
            
            # Determine MIME type based on extension
            ext = audio_path.lower().split('.')[-1]
            mime_map = {
                'wav': 'audio/wav',
                'mp3': 'audio/mp3',
                'm4a': 'audio/mp4',
                'flac': 'audio/flac',
                'ogg': 'audio/ogg',
            }
            mime_type = mime_map.get(ext, 'audio/wav')
            
            # Use genai_types.Part.from_bytes to create proper Part object
            return genai_types.Part.from_bytes(data=audio_data, mime_type=mime_type)
        except Exception as e:
            eval_logger.warning(f"Error loading audio {audio_path}: {e}")
            return None

    def build_gemini_content(self, contexts: str, visuals: List) -> List:
        """
        Build content list for Gemini API.
        
        Gemini expects a flat list of content parts: [image, image, audio, text, ...]
        """
        content_parts = []
        
        # Process visuals first
        for visual in visuals:
            if isinstance(visual, str):
                # File path - could be image, video, or audio
                visual_lower = visual.lower()
                
                if any(ext in visual_lower for ext in ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv', '.webm']):
                    # Video - extract frames
                    frames = self.encode_video_to_frames(visual)
                    for frame in frames:
                        content_parts.append(frame)
                        
                elif any(ext in visual_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                    # Image
                    img = self.encode_image(visual)
                    content_parts.append(img)
                    
                elif any(ext in visual_lower for ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']):
                    # Audio
                    if not self.skip_audio:
                        audio_content = self.load_audio(visual)
                        if audio_content:
                            content_parts.append(audio_content)
                            
            elif isinstance(visual, Image.Image):
                content_parts.append(self.encode_image(visual))
                
            elif isinstance(visual, dict):
                # Could be audio dict format from dataset
                if "path" in visual:
                    # Audio path in dict format: {"path": "/path/to/audio.wav"}
                    if not self.skip_audio:
                        audio_content = self.load_audio(visual["path"])
                        if audio_content:
                            content_parts.append(audio_content)
                elif "sampling_rate" in visual and "array" in visual:
                    # Audio array format
                    if not self.skip_audio:
                        try:
                            import soundfile as sf
                            audio_io = io.BytesIO()
                            sf.write(audio_io, visual["array"], visual["sampling_rate"], format="WAV")
                            audio_io.seek(0)
                            # Use genai_types.Part.from_bytes to create proper Part object
                            audio_part = genai_types.Part.from_bytes(data=audio_io.read(), mime_type="audio/wav")
                            content_parts.append(audio_part)
                        except Exception as e:
                            eval_logger.warning(f"Error processing audio array: {e}")
        
        # Add text at the end
        content_parts.append(contexts)
        
        return content_parts

    def flatten(self, input_list):
        """Flatten nested list."""
        new_list = []
        for i in input_list:
            if isinstance(i, list):
                for j in i:
                    new_list.append(j)
            else:
                new_list.append(i)
        return new_list

    def _extract_response_text(self, response) -> str:
        """
        Extract text from a Gemini response, properly handling thinking models.
        
        Thinking models may return parts with thought=True (internal reasoning)
        and thought_signature. We only want the non-thought text parts.
        If response.text works (SDK auto-concatenates non-thought text), use it.
        Otherwise fall back to manual part extraction, skipping thought parts.
        """
        # Try response.text first — SDK concatenates non-thought text parts
        try:
            text = response.text
            if text:
                return text
        except Exception:
            pass
        
        # Fallback: manually extract non-thought text parts
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts or []
                text_parts = []
                for part in parts:
                    # Skip thought parts (internal reasoning)
                    if getattr(part, 'thought', False):
                        continue
                    # Skip thought_signature parts
                    if getattr(part, 'thought_signature', None) and not getattr(part, 'text', None):
                        continue
                    # Collect actual text
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return "\n".join(text_parts)
        
        return ""

    def _parse_messages(self, messages: List[Dict]) -> Tuple[str, str, List]:
        """
        Parse messages format to extract system prompt, user text, and visuals.
        
        System role messages are extracted separately for native API support
        (Gemini's system_instruction parameter).
        
        Returns:
            Tuple of (system_text, combined_user_text, visuals_list)
        """
        system_texts = []
        all_texts = []
        all_visuals = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])
            
            if isinstance(content, str):
                if role == "system":
                    system_texts.append(content)
                else:
                    all_texts.append(f"[{role.capitalize()}]: {content}")
                continue
            
            if not isinstance(content, list):
                continue
                
            message_texts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                    
                item_type = item.get("type", "")
                
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        message_texts.append(text)
                elif item_type == "image":
                    url = item.get("url", "")
                    if url:
                        if isinstance(url, str):
                            all_visuals.append(url)
                        elif isinstance(url, Image.Image):
                            all_visuals.append(url)
                elif item_type == "video":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(url)
                elif item_type == "audio":
                    url = item.get("url", "")
                    if url:
                        if isinstance(url, str):
                            all_visuals.append({"path": url})
                        elif isinstance(url, dict):
                            all_visuals.append(url)
            
            if message_texts:
                if role == "system":
                    system_texts.append(' '.join(message_texts))
                elif role == "user":
                    all_texts.append(f"[User]: {' '.join(message_texts)}")
                elif role == "assistant":
                    all_texts.append(f"[Assistant]: {' '.join(message_texts)}")
                else:
                    all_texts.append(' '.join(message_texts))
        
        system_text = "\n\n".join(system_texts) if system_texts else ""
        combined_text = "\n\n".join(all_texts)
        return system_text, combined_text, all_visuals

    def generate_until(self, requests) -> List[str]:
        """
        Generate responses for a list of requests.
        
        Supports two modes:
        1. Simple mode: (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
        2. Messages mode: (ctx, doc_to_messages, gen_kwargs, doc_id, task, split)
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Gemini25Pro Responding")
        
        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"
        
        for request in requests:
            args = request.args
            
            # Detect mode based on argument types
            system_prompt = None  # Will be set if messages mode provides system role
            if len(args) >= 6:
                if callable(args[1]):
                    # Messages mode: (ctx, doc_to_messages, gen_kwargs, doc_id, task, split)
                    ctx, doc_to_messages_func, gen_kwargs, doc_id, task, split = args
                    messages = doc_to_messages_func(self.task_dict[task][split][doc_id])
                    system_prompt, contexts, visuals = self._parse_messages(messages)
                else:
                    # Simple mode: (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
                    contexts, gen_kwargs, doc_to_visual, doc_id, task, split = args
                    visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                    visuals = self.flatten(visuals)
                    if None in visuals:
                        visuals = [v for v in visuals if v is not None]
            else:
                eval_logger.error(f"Unexpected args format: {args}")
                res.append("")
                pbar.update(1)
                continue
            
            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue
            
            # Get visuals (for simple mode they're already set above)
            # Build content for Gemini
            content_parts = self.build_gemini_content(contexts, visuals)
            
            # CRITICAL: Prepend system prompt directly into content_parts
            # AI Hub proxy may ignore system_instruction in config,
            # so we embed the system prompt as the first text element
            if system_prompt:
                content_parts.insert(0, f"[System Instructions]\n{system_prompt}\n[End System Instructions]\n")
            
            # Generation config
            gen_config = {}
            if "max_new_tokens" in gen_kwargs:
                gen_config["max_output_tokens"] = gen_kwargs["max_new_tokens"]
            else:
                gen_config["max_output_tokens"] = 1024
            if "temperature" in gen_kwargs:
                gen_config["temperature"] = gen_kwargs["temperature"]
            else:
                gen_config["temperature"] = 0
            
            # Add response_mime_type for JSON mode if specified
            if gen_kwargs.get("response_format") == "json":
                gen_config["response_mime_type"] = "application/json"
            
            # Also try system_instruction in config (may be ignored by proxy)
            if system_prompt:
                gen_config["system_instruction"] = system_prompt
            
            # Retry logic with empty-response retry
            # Thinking models may sometimes produce only thought parts without
            # text output, so we retry on empty responses with escalating temperature.
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    # On retry for empty response, bump temperature to break the pattern
                    retry_config = gen_config.copy() if gen_config else {}
                    if attempt > 0:
                        retry_config["temperature"] = min(0.3 * attempt, 1.0)
                    
                    if self.streaming:
                        response = self.client.models.generate_content_stream(
                            model=self.model_version,
                            contents=content_parts,
                            config=retry_config if retry_config else None,
                        )
                        response_text = ""
                        for chunk in response:
                            if hasattr(chunk, 'text') and chunk.text:
                                response_text += chunk.text
                    else:
                        response = self.client.models.generate_content(
                            model=self.model_version,
                            contents=content_parts,
                            config=retry_config if retry_config else None,
                        )
                        # Extract response text from thinking model
                        response_text = self._extract_response_text(response)
                    
                    # Retry on empty response (thinking model sometimes returns empty)
                    if not response_text.strip():
                        eval_logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: empty response, retrying with higher temperature...")
                        if attempt < self.max_retries - 1:
                            time.sleep(NUM_SECONDS_TO_SLEEP)
                            continue
                    
                    break
                    
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {str(e)}\"")
                        response_text = ""
            
            res.append(response_text)
            pbar.update(1)
            
            # Cache response
            if self.continual_mode:
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)
        
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round generation not implemented yet."""
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini 2.5 Pro")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Log likelihood not supported for Gemini API."""
        raise NotImplementedError("Gemini 2.5 Pro API does not support loglikelihood")
