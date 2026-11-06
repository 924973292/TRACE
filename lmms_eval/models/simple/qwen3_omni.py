"""
Qwen3-Omni API Model Adapter for lmms-eval

This module provides integration with Qwen3-Omni via API (OpenAI compatible).

Usage:
    python -m lmms_eval \
        --model qwen3_omni \
        --model_args model_version=Qwen3-Omni-30B-A3B-Instruct,max_frames_num=4 \
        --tasks task_0126_batch5 \
        --batch_size 1

Environment Variables (optional, can also be passed via model_args):
    QWEN3_API_KEY: API key for the OpenAI-compatible endpoint
    QWEN3_BASE_URL: Base URL for the API
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
    from openai import OpenAI
except ImportError as e:
    eval_logger.warning(f"Error importing openai: {str(e)}. Please install with: pip install openai")
    OpenAI = None

try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("decord not available, video processing may not work")
    VideoReader = None
    cpu = None

NUM_SECONDS_TO_SLEEP = 5


@register_model("qwen3_omni")
class Qwen3Omni(lmms):
    """
    Qwen3-Omni model adapter using OpenAI-compatible API.
    
    Supports multi-modal inputs: images, videos, audio, and text.
    Uses inference API endpoint.
    """
    
    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages (system prompt)
    
    def __init__(
        self,
        model_version: str = "Qwen3-Omni-30B-A3B-Instruct",
        api_key: str = None,
        base_url: str = None,
        timeout: int = 120,
        max_retries: int = 5,
        max_frames_num: int = 4,
        skip_audio: bool = False,
        max_new_tokens: int = 4096,
        temperature: float = 0.01,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize Qwen3-Omni model.
        
        Args:
            model_version: Model name (default: Qwen3-Omni-30B-A3B-Instruct)
            api_key: API key (default: from model_args or env)
            base_url: API base URL (default: from model_args or env)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            max_frames_num: Maximum frames to extract from videos
            skip_audio: Whether to skip audio content
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            continual_mode: Enable response caching
            response_persistent_folder: Folder for cached responses
        """
        super().__init__()
        
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_frames_num = max_frames_num
        self.skip_audio = skip_audio
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.continual_mode = continual_mode
        
        # API configuration
        self.api_key = api_key or get_env_value(
            ["QWEN3_API_KEY", "OPENAI_API_KEY"],
            default="EMPTY",
        )
        self.base_url = base_url or get_env_value(
            ["QWEN3_BASE_URL", "QWEN3_API_URL"],
            required=True,
            description="Qwen3-Omni base URL",
        )
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        
        eval_logger.info(f"Initialized Qwen3Omni with model: {self.model_version}")
        eval_logger.info(f"API Base URL: {self.base_url}")
        eval_logger.info(f"skip_audio: {self.skip_audio}, max_frames_num: {self.max_frames_num}")
        
        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/qwen3_omni_persistent_folder"
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

    def encode_file_base64(self, file_path: str) -> str:
        """Encode file to base64 string."""
        if not os.path.exists(file_path):
            eval_logger.warning(f"File not found: {file_path}")
            return None
        with open(file_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def encode_image_to_base64(self, image: Union[Image.Image, str]) -> str:
        """Encode image to base64 string with proper data URL format."""
        if isinstance(image, str):
            # File path
            if not os.path.exists(image):
                eval_logger.warning(f"Image file not found: {image}")
                return None
            ext = image.lower().split('.')[-1]
            mime_map = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'bmp': 'image/bmp',
            }
            mime_type = mime_map.get(ext, 'image/png')
            b64 = self.encode_file_base64(image)
            if b64 is None:
                return None
            return f"data:{mime_type};base64,{b64}"
        else:
            # PIL Image
            output_buffer = BytesIO()
            image_rgb = image.convert("RGB") if image.mode != "RGB" else image
            image_rgb.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            b64 = base64.standard_b64encode(byte_data).decode("utf-8")
            return f"data:image/png;base64,{b64}"

    def encode_video_to_frames(self, video_path: str, num_frames: int = None) -> List[str]:
        """Extract frames from video file and return as base64 data URLs."""
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
            
            frame_urls = []
            for frame in frames:
                img = Image.fromarray(frame)
                frame_url = self.encode_image_to_base64(img)
                frame_urls.append(frame_url)
            
            return frame_urls
        except Exception as e:
            eval_logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    def encode_audio_to_base64(self, audio_path: str) -> str:
        """Encode audio file to base64 data URL."""
        if not os.path.exists(audio_path):
            eval_logger.warning(f"Audio file not found: {audio_path}")
            return None
        ext = audio_path.lower().split('.')[-1]
        mime_map = {
            'wav': 'audio/wav',
            'mp3': 'audio/mp3',
            'm4a': 'audio/mp4',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
        }
        mime_type = mime_map.get(ext, 'audio/wav')
        b64 = self.encode_file_base64(audio_path)
        if b64 is None:
            return None
        return f"data:{mime_type};base64,{b64}"

    def build_openai_content(self, contexts: str, visuals: List) -> List[Dict]:
        """
        Build content list for OpenAI-compatible API (Qwen3-Omni format).
        
        Qwen3-Omni expects content in format:
        [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
            {"type": "text", "text": "..."}
        ]
        """
        content_parts = []
        
        # Process visuals first
        for visual in visuals:
            if isinstance(visual, str):
                # File path - could be image, video, or audio
                visual_lower = visual.lower()
                
                if any(ext in visual_lower for ext in ['.mp4', '.avi', '.mov', '.flv', '.wmv', '.mkv', '.webm']):
                    # Video - extract frames as images
                    frame_urls = self.encode_video_to_frames(visual)
                    for frame_url in frame_urls:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": frame_url}
                        })
                        
                elif any(ext in visual_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                    # Image
                    img_url = self.encode_image_to_base64(visual)
                    if img_url:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": img_url}
                        })
                    
                elif any(ext in visual_lower for ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']):
                    # Audio
                    if not self.skip_audio:
                        audio_url = self.encode_audio_to_base64(visual)
                        if audio_url:
                            content_parts.append({
                                "type": "audio_url",
                                "audio_url": {"url": audio_url}
                            })
                        
            elif isinstance(visual, Image.Image):
                img_url = self.encode_image_to_base64(visual)
                if img_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    })
                
            elif isinstance(visual, dict):
                # Could be audio dict format from dataset
                if "path" in visual:
                    # Audio path in dict format: {"path": "/path/to/audio.wav"}
                    if not self.skip_audio:
                        audio_url = self.encode_audio_to_base64(visual["path"])
                        if audio_url:
                            content_parts.append({
                                "type": "audio_url",
                                "audio_url": {"url": audio_url}
                            })
                elif "sampling_rate" in visual and "array" in visual:
                    # Audio array format
                    if not self.skip_audio:
                        try:
                            import soundfile as sf
                            audio_io = io.BytesIO()
                            sf.write(audio_io, visual["array"], visual["sampling_rate"], format="WAV")
                            audio_io.seek(0)
                            b64 = base64.standard_b64encode(audio_io.read()).decode("utf-8")
                            content_parts.append({
                                "type": "audio_url",
                                "audio_url": {"url": f"data:audio/wav;base64,{b64}"}
                            })
                        except Exception as e:
                            eval_logger.warning(f"Error processing audio array: {e}")
        
        # Add text at the end
        content_parts.append({"type": "text", "text": contexts})
        
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

    def _parse_messages(self, messages: List[Dict]) -> Tuple[str, str, List]:
        """
        Parse messages format to extract system prompt, user text, and visuals.
        
        System role messages are extracted separately for native API support.
        
        Messages format:
        [
            {"role": "system", "content": [{"type": "text", "text": "..."}]},
            {"role": "user", "content": [
                {"type": "video", "url": "/path/to/video.mp4"},
                {"type": "audio", "url": {"path": "/path/to/audio.wav"}},
                {"type": "image", "url": "/path/to/image.png"},
                {"type": "text", "text": "..."}
            ]},
            ...
        ]
        
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
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Qwen3Omni Responding")
        
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
            
            # Build content for Qwen3-Omni (OpenAI format)
            content_parts = self.build_openai_content(contexts, visuals)
            
            # CRITICAL: Embed system prompt directly in user content
            # so we prepend the system prompt as the first text element in user content
            if system_prompt:
                content_parts.insert(0, {"type": "text", "text": f"[System Instructions]\n{system_prompt}\n[End System Instructions]\n"})
            
            # Log content info for debugging
            eval_logger.debug(f"Content parts: {len(content_parts)} items")
            for i, part in enumerate(content_parts):
                if part.get("type") == "image_url":
                    eval_logger.debug(f"  [{i}] Image")
                elif part.get("type") == "audio_url":
                    eval_logger.debug(f"  [{i}] Audio")
                elif part.get("type") == "text":
                    eval_logger.debug(f"  [{i}] Text: {len(part.get('text', ''))} chars")
            
            # Generation config
            max_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
            temperature = gen_kwargs.get("temperature", self.temperature)
            
            # Build messages for OpenAI API
            # Keep system role message AND embed in content (double insurance)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_parts})
            
            # Retry logic
            response_text = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_version,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    response_text = response.choices[0].message.content if response.choices else ""
                    
                    # Log successful response
                    if response_text:
                        eval_logger.debug(f"Got response: {response_text[:200]}...")
                    break
                    
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {str(e)}")
                        response_text = ""
            
            # Ensure we never return None
            if response_text is None:
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
        raise NotImplementedError("TODO: Implement multi-round generation for Qwen3-Omni")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Log likelihood not supported for API."""
        raise NotImplementedError("Qwen3-Omni API does not support loglikelihood")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
