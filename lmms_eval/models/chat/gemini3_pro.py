"""
Gemini 3 Pro Chat API Model Adapter for lmms-eval

This module provides chat-style integration with Gemini 3 Pro.
It supports doc_to_messages format with multi-turn conversations.

Usage:
    python -m lmms_eval \
        --model gemini3_pro_chat \
        --model_args model_version=aihub/gemini-3-pro-preview,max_frames_num=2,skip_audio=True \
        --tasks omni_test_local \
        --batch_size 1

Environment Variables (optional):
    GEMINI3_API_KEY: API key for the AI Hub
    GEMINI3_BASE_URL: Base URL for the API
"""

import base64
import io
import json
import os
import pathlib
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

try:
    from google import genai
    from google.genai.types import HttpOptions, Part, Blob
except ImportError as e:
    eval_logger.warning(f"Error importing google.genai: {str(e)}. Please install with: pip install google-genai")
    genai = None
    Part = None
    Blob = None

try:
    from decord import VideoReader, cpu
except ImportError:
    eval_logger.warning("decord not available, video processing may not work")
    VideoReader = None
    cpu = None

NUM_SECONDS_TO_SLEEP = 10


def setup_io_logger(log_file: str = None):
    """Setup a dedicated logger for input/output recording."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"./results/gemini3_io_{timestamp}.log"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    return log_file


@register_model("gemini3_pro_chat")
class Gemini3ProChat(lmms):
    """
    Gemini 3 Pro Chat model adapter using google.genai SDK.
    
    This version supports doc_to_messages format for multi-turn conversations
    with multi-modal inputs: images, videos, and text.
    """
    
    is_simple = False  # Indicates this model uses doc_to_messages format
    
    def __init__(
        self,
        model_version: str = "aihub/gemini-3-pro-preview",
        api_key: str = None,
        base_url: str = None,
        api_version: str = "v1/beta/google/gemini",
        timeout: int = 120,
        max_retries: int = 5,
        max_frames_num: int = 2,
        skip_audio: bool = True,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        streaming: bool = False,
        io_log_file: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize Gemini 3 Pro Chat model.
        
        Args:
            model_version: Model name (default: aihub/gemini-3-pro-preview)
            api_key: API key (default: from GEMINI3_API_KEY env var or hardcoded)
            base_url: API base URL (default: from GEMINI3_BASE_URL env var or hardcoded)
            api_version: API version path
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            max_frames_num: Maximum frames to extract from videos
            skip_audio: Whether to skip audio content
            continual_mode: Enable response caching
            response_persistent_folder: Folder for cached responses
            streaming: Whether to use streaming mode
            io_log_file: Path to input/output log file (optional)
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
        
        # Setup I/O logging
        self.io_log_file = setup_io_logger(io_log_file)
        self.step_counter = 0
        
        # API configuration
        self.api_key = api_key or os.getenv("GEMINI3_API_KEY", "<YOUR_API_KEY>")
        self.base_url = base_url or os.getenv("GEMINI3_BASE_URL", "<YOUR_BASE_URL>")
        self.api_version = api_version
        
        # Initialize client
        http_options = HttpOptions(base_url=self.base_url, api_version=self.api_version)
        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        
        eval_logger.info(f"Initialized Gemini3ProChat with model: {self.model_version}")
        eval_logger.info(f"API Base URL: {self.base_url}")
        eval_logger.info(f"I/O Log File: {self.io_log_file}")
        
        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/gemini3_chat_persistent_folder"
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

    def log_io(self, step: int, input_summary: Dict, output: str, latency: float = 0):
        """
        Log input and output for each step to the I/O log file.
        
        Args:
            step: Step number
            input_summary: Summary of input (text prompts, media file paths)
            output: Model output text
            latency: API response time in seconds
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "latency_seconds": round(latency, 2),
            "input": input_summary,
            "output": output
        }
        
        with open(self.io_log_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Step {step} | {timestamp} | Latency: {latency:.2f}s\n")
            f.write("-" * 80 + "\n")
            f.write("INPUT:\n")
            f.write(json.dumps(input_summary, ensure_ascii=False, indent=2) + "\n")
            f.write("-" * 80 + "\n")
            f.write("OUTPUT:\n")
            f.write(output + "\n")
            f.write("=" * 80 + "\n\n")

    def summarize_contents(self, contents: List) -> Dict:
        """
        Create a summary of contents for logging (without binary data).
        
        Args:
            contents: List of content parts (strings and Part objects)
            
        Returns:
            Dict with summarized content info
        """
        summary = {
            "total_parts": len(contents),
            "parts": []
        }
        
        for i, part in enumerate(contents):
            if isinstance(part, str):
                # Truncate long text for logging
                text = part[:500] + "..." if len(part) > 500 else part
                summary["parts"].append({
                    "index": i,
                    "type": "text",
                    "length": len(part),
                    "content": text
                })
            else:
                # Part object with inline_data
                try:
                    if hasattr(part, 'inline_data'):
                        data = part.inline_data
                        mime = getattr(data, 'mime_type', 'unknown')
                        size = len(getattr(data, 'data', b''))
                        summary["parts"].append({
                            "index": i,
                            "type": "media",
                            "mime_type": mime,
                            "size_bytes": size
                        })
                    else:
                        summary["parts"].append({
                            "index": i,
                            "type": "unknown",
                            "repr": str(type(part))
                        })
                except Exception as e:
                    summary["parts"].append({
                        "index": i,
                        "type": "error",
                        "error": str(e)
                    })
        
        return summary

    def encode_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """Load and convert image to PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB") if image.mode != "RGB" else image

    def extract_all_frames_to_pil(self, video_path: str) -> List[Image.Image]:
        """
        Extract ALL frames from video file (no frame limit).
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of PIL Images (all frames)
        """
        if VideoReader is None:
            eval_logger.warning("decord not available, skipping video")
            return []
        
        if not os.path.exists(video_path):
            eval_logger.warning(f"Video file not found: {video_path}")
            return []
            
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            
            if total_frame_num == 0:
                return []
            
            # Extract all frames
            frame_idx = list(range(total_frame_num))
            frames = vr.get_batch(frame_idx).asnumpy()
            
            pil_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                pil_frames.append(img)
            
            eval_logger.debug(f"Extracted {len(pil_frames)} frames from {video_path}")
            return pil_frames
        except Exception as e:
            eval_logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return []

    def load_audio_bytes(self, audio_path: str) -> Optional[Dict]:
        """
        Load audio file and return as Gemini-compatible dict.
        
        Args:
            audio_path: Path to audio file (wav/mp3)
            
        Returns:
            Dict with mime_type and data bytes, or None if failed
        """
        if not os.path.exists(audio_path):
            eval_logger.warning(f"Audio file not found: {audio_path}")
            return None
            
        try:
            audio_data = pathlib.Path(audio_path).read_bytes()
            
            # Determine mime type
            if audio_path.endswith('.wav'):
                mime_type = "audio/wav"
            elif audio_path.endswith('.mp3'):
                mime_type = "audio/mpeg"
            elif audio_path.endswith('.m4a'):
                mime_type = "audio/mp4"
            else:
                mime_type = "audio/wav"  # Default
            
            eval_logger.debug(f"Loaded audio {audio_path}, size: {len(audio_data)} bytes")
            return {"mime_type": mime_type, "data": audio_data}
        except Exception as e:
            eval_logger.error(f"Error loading audio {audio_path}: {str(e)}")
            return None

    def _image_to_part(self, image: Union[Image.Image, str]) -> Part:
        """
        Convert image to Gemini Part with inlineData.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Part object with inline image data
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image
        
        # Encode to PNG bytes
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        
        return Part(inlineData=Blob(data=img_bytes, mimeType="image/png"))

    def _video_to_part(self, video_path: str) -> Optional[Part]:
        """
        Load video file and convert to Gemini Part.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Part object with inline video data, or None if failed
        """
        if not os.path.exists(video_path):
            eval_logger.warning(f"Video file not found: {video_path}")
            return None
        
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
            
            # Determine mime type
            if video_path.endswith('.mp4'):
                mime_type = "video/mp4"
            elif video_path.endswith('.webm'):
                mime_type = "video/webm"
            elif video_path.endswith('.mov'):
                mime_type = "video/quicktime"
            else:
                mime_type = "video/mp4"  # Default
            
            eval_logger.debug(f"Loaded video {video_path}, size: {len(video_data)} bytes")
            return Part(inlineData=Blob(data=video_data, mimeType=mime_type))
        except Exception as e:
            eval_logger.error(f"Error loading video {video_path}: {str(e)}")
            return None

    def _audio_to_part(self, audio_path: str) -> Optional[Part]:
        """
        Load audio file and convert to Gemini Part.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Part object with inline audio data, or None if failed
        """
        if not os.path.exists(audio_path):
            eval_logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            # Determine mime type
            if audio_path.endswith('.wav'):
                mime_type = "audio/wav"
            elif audio_path.endswith('.mp3'):
                mime_type = "audio/mpeg"
            elif audio_path.endswith('.m4a'):
                mime_type = "audio/mp4"
            elif audio_path.endswith('.ogg'):
                mime_type = "audio/ogg"
            else:
                mime_type = "audio/wav"  # Default
            
            eval_logger.debug(f"Loaded audio {audio_path}, size: {len(audio_data)} bytes")
            return Part(inlineData=Blob(data=audio_data, mimeType=mime_type))
        except Exception as e:
            eval_logger.error(f"Error loading audio {audio_path}: {str(e)}")
            return None

    def flatten_messages_to_contents(self, messages: List[Dict]) -> List:
        """
        Flatten multi-turn messages into a single contents list for Gemini.
        
        Uses Gemini's Part(inlineData=Blob(...)) format for media files.
        Video files are sent as whole files (not extracted frames).
        
        Args:
            messages: List of message dicts with role and content
            
        Returns:
            Flat list of content parts (strings and Part objects)
        """
        contents = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])
            
            # Add role marker for context
            if role == "assistant":
                contents.append("[Assistant Response]:")
            
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        contents.append(item)
                    elif isinstance(item, dict):
                        item_type = item.get("type", "")
                        
                        if item_type == "text":
                            text_content = item.get("text", "")
                            if text_content:
                                contents.append(text_content)
                        
                        elif item_type == "image":
                            # Image path or PIL Image - convert to Part
                            url = item.get("url") or item.get("image")
                            if url:
                                try:
                                    part = self._image_to_part(url)
                                    contents.append(part)
                                except Exception as e:
                                    eval_logger.warning(f"Error loading image: {e}")
                        
                        elif item_type == "video":
                            # Video path - load as whole file
                            video_path = item.get("url") or item.get("video")
                            if video_path:
                                part = self._video_to_part(video_path)
                                if part:
                                    contents.append(part)
                        
                        elif item_type == "audio":
                            # Audio path - load as bytes
                            if not self.skip_audio:
                                audio_url = item.get("url")
                                # Handle both string path and dict with path
                                if isinstance(audio_url, dict):
                                    audio_path = audio_url.get("path", "")
                                else:
                                    audio_path = audio_url
                                
                                if audio_path:
                                    part = self._audio_to_part(audio_path)
                                    if part:
                                        contents.append(part)
        
        return contents
        
        return contents

    def encode_image_to_bytes(self, image: Union[Image.Image, str]) -> bytes:
        """Encode image to bytes."""
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()
        
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

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

    def convert_messages_to_gemini_format(self, messages: List[Dict], video_kwargs: Dict = None) -> List[Dict]:
        """
        Convert ChatMessages format to Gemini API format.
        
        Gemini uses a different format:
        [
            {"role": "user", "parts": [{"text": "..."}, {"inline_data": {...}}]},
            {"role": "model", "parts": [{"text": "..."}]},
            ...
        ]
        
        Note: Gemini uses "model" instead of "assistant"
        """
        if video_kwargs is None:
            video_kwargs = {"nframes": self.max_frames_num}
        
        nframes = video_kwargs.get("nframes", self.max_frames_num)
        gemini_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])
            
            # Map role names
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Gemini doesn't have system role, prepend to first user message
                role = "user"
            
            parts = []
            
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        item_type = item.get("type", "")
                        
                        if item_type == "text":
                            text_content = item.get("text", "")
                            if text_content:
                                parts.append({"text": text_content})
                        
                        elif item_type == "image":
                            url = item.get("url") or item.get("image")
                            if url:
                                try:
                                    img_bytes = self.encode_image_to_bytes(url)
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": base64.b64encode(img_bytes).decode("utf-8")
                                        }
                                    })
                                except Exception as e:
                                    eval_logger.warning(f"Error encoding image: {e}")
                        
                        elif item_type == "image_url":
                            # OpenAI format
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")
                            if url.startswith("data:image"):
                                # Already base64 encoded
                                try:
                                    # Extract base64 data
                                    _, base64_data = url.split(",", 1)
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": "image/png",
                                            "data": base64_data
                                        }
                                    })
                                except Exception as e:
                                    eval_logger.warning(f"Error parsing base64 image: {e}")
                        
                        elif item_type == "video":
                            video_path = item.get("url") or item.get("video")
                            if video_path:
                                frames = self.encode_video_to_frames(video_path, nframes)
                                for frame in frames:
                                    try:
                                        img_bytes = self.encode_image_to_bytes(frame)
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": "image/png",
                                                "data": base64.b64encode(img_bytes).decode("utf-8")
                                            }
                                        })
                                    except Exception as e:
                                        eval_logger.warning(f"Error encoding video frame: {e}")
                        
                        elif item_type == "audio":
                            if not self.skip_audio:
                                audio_path = item.get("url") or item.get("audio")
                                if audio_path:
                                    try:
                                        audio_data = pathlib.Path(audio_path).read_bytes()
                                        mime_type = "audio/wav" if audio_path.endswith('.wav') else "audio/mpeg"
                                        parts.append({
                                            "inline_data": {
                                                "mime_type": mime_type,
                                                "data": base64.b64encode(audio_data).decode("utf-8")
                                            }
                                        })
                                    except Exception as e:
                                        eval_logger.warning(f"Error loading audio: {e}")
            
            if parts:  # Only add message if it has content
                gemini_messages.append({
                    "role": role,
                    "parts": parts
                })
        
        return gemini_messages

    def generate_until(self, requests) -> List[str]:
        """
        Generate responses for chat-style requests.
        
        This version handles doc_to_messages format and flattens all
        multi-turn messages into a single contents list for Gemini.
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Gemini3ProChat Responding")
        
        e2e_latency = 0
        total_tokens = 0
        
        for ctx, doc_to_messages, gen_kwargs, doc_id, task, split in [reg.args for reg in requests]:
            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue
            
            # Get messages from doc (raw format with video/audio/image paths)
            chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
            
            # Flatten all messages into single contents list
            # This handles video frame extraction and audio loading directly
            contents = self.flatten_messages_to_contents(chat_messages)
            
            # Create input summary for logging
            input_summary = self.summarize_contents(contents)
            input_summary["doc_id"] = doc_id
            input_summary["task"] = task
            
            # Retry logic
            response_text = ""
            step_latency = 0
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    if self.streaming:
                        response = self.client.models.generate_content_stream(
                            model=self.model_version,
                            contents=contents,
                        )
                        response_text = ""
                        for chunk in response:
                            if hasattr(chunk, 'text') and chunk.text:
                                response_text += chunk.text
                    else:
                        response = self.client.models.generate_content(
                            model=self.model_version,
                            contents=contents,
                        )
                        
                        # Get response text (simple approach matching user's example)
                        response_text = response.text if response.text else ""
                    
                    end_time = time.time()
                    step_latency = end_time - start_time
                    e2e_latency += step_latency
                    
                    # Approximate token count (handle empty response)
                    if response_text:
                        total_tokens += len(response_text.split())
                    
                    break
                    
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {str(e)}")
                        response_text = ""
            
            # Log input/output for this step
            self.step_counter += 1
            self.log_io(self.step_counter, input_summary, response_text, step_latency)
            
            res.append(response_text)
            pbar.update(1)
            
            # Cache response
            if self.continual_mode:
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)
        
        # Log metrics
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)
        
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round generation not implemented yet."""
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini 3 Pro Chat")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Log likelihood not supported for Gemini API."""
        raise NotImplementedError("Gemini 3 Pro API does not support loglikelihood")
