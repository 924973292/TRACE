"""
VITA API Model Adapter for lmms-eval

This module provides integration with VITA model via its custom FastAPI server.
The VITA server exposes a /generate endpoint that accepts multipart form data
with question text and optional image/video/audio files.

Usage:
    python -m lmms_eval \
        --model vita_api \
        --model_args api_url=http://172.16.1.134:8000/generate,max_frames_num=4 \
        --tasks task_v2_vita \
        --batch_size 1
"""

import io
import json
import os
import time
from typing import Dict, List, Tuple, Union

import requests as http_requests
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5


@register_model("vita_api")
class VitaAPI(lmms):
    """
    VITA model adapter using custom FastAPI HTTP endpoint.

    The VITA server accepts multipart form data:
    - question: str (text prompt)
    - image: UploadFile (optional)
    - video: UploadFile (optional)
    - audio: UploadFile (optional)
    - temperature: float
    - model_type: str
    - conv_mode: str

    Returns JSON: {"response": str, "inference_time": float}
    Response may start with "☜" and end with "<|im_end|>" which need to be stripped.
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages

    def __init__(
        self,
        api_url: str = "http://172.16.8.164:8000/generate",
        timeout: int = 180,
        max_retries: int = 5,
        model_type: str = "qwen2p5_instruct",
        conv_mode: str = "qwen2p5_instruct",
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_type = model_type
        self.conv_mode = conv_mode
        self.continual_mode = continual_mode

        eval_logger.info(f"Initialized VitaAPI with endpoint: {self.api_url}")

        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/vita_api_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                "vita_api_response.json",
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
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

    # ------------------------------------------------------------------
    # Message parsing (same logic as gemini adapter)
    # ------------------------------------------------------------------
    def _parse_messages(self, messages: List[Dict]) -> Tuple[str, str, List]:
        """
        Parse messages format to extract system prompt, user text, and visuals.

        Returns:
            Tuple of (system_text, combined_user_text, all_visuals)
            where all_visuals is a list of:
              - str (image or video path)
              - dict {"path": str} (audio path)
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
                    if url and isinstance(url, str):
                        all_visuals.append(("image", url))

                elif item_type == "video":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(("video", url))

                elif item_type == "audio":
                    url = item.get("url", "")
                    if url:
                        if isinstance(url, str):
                            all_visuals.append(("audio", url))
                        elif isinstance(url, dict) and "path" in url:
                            all_visuals.append(("audio", url["path"]))

            if message_texts:
                if role == "system":
                    system_texts.append(" ".join(message_texts))
                elif role == "user":
                    all_texts.append(f"[User]: {' '.join(message_texts)}")
                elif role == "assistant":
                    all_texts.append(f"[Assistant]: {' '.join(message_texts)}")
                else:
                    all_texts.append(" ".join(message_texts))

        system_text = "\n\n".join(system_texts) if system_texts else ""
        combined_text = "\n\n".join(all_texts)
        return system_text, combined_text, all_visuals

    # ------------------------------------------------------------------
    # Clean response
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_response(text: str) -> str:
        """Strip VITA-specific artifacts from response."""
        if not text:
            return ""
        # Strip leading "☜" character
        text = text.lstrip("☜").strip()
        # Strip trailing <|im_end|>
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")].strip()
        return text

    # ------------------------------------------------------------------
    # Determine mime type from extension
    # ------------------------------------------------------------------
    @staticmethod
    def _get_mime_type(path: str) -> str:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
            "mp4": "video/mp4",
            "avi": "video/avi",
            "mov": "video/quicktime",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }
        return mime_map.get(ext, "application/octet-stream")

    # ------------------------------------------------------------------
    # Build multipart request
    # ------------------------------------------------------------------
    def _build_request(
        self, question: str, visuals: List[Tuple[str, str]], temperature: float = 0.0
    ) -> Tuple[dict, dict]:
        """
        Build multipart form data for the VITA /generate endpoint.

        VITA server accepts separate fields: question, image, video, audio.
        It can handle one file per type simultaneously.

        Args:
            question: The text prompt (system + user combined)
            visuals: List of (type, path) tuples
            temperature: Generation temperature

        Returns:
            Tuple of (data dict for form fields, files dict for uploads)
        """
        data = {
            "question": question,
            "temperature": str(temperature),
            "model_type": self.model_type,
            "conv_mode": self.conv_mode,
        }
        files = {}

        # Collect the LAST file of each type (in case there are multiples,
        # the last is usually the most important — the primary screenshot/video/audio)
        image_path = None
        video_path = None
        audio_path = None

        for vtype, vpath in visuals:
            if vtype == "image":
                image_path = vpath
            elif vtype == "video":
                video_path = vpath
            elif vtype == "audio":
                audio_path = vpath

        if image_path and os.path.exists(image_path):
            mime = self._get_mime_type(image_path)
            files["image"] = (os.path.basename(image_path), open(image_path, "rb"), mime)

        if video_path and os.path.exists(video_path):
            mime = self._get_mime_type(video_path)
            files["video"] = (os.path.basename(video_path), open(video_path, "rb"), mime)

        if audio_path and os.path.exists(audio_path):
            mime = self._get_mime_type(audio_path)
            files["audio"] = (os.path.basename(audio_path), open(audio_path, "rb"), mime)

        return data, files

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="VITA API Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for request in requests:
            args = request.args

            # Parse request arguments
            if len(args) >= 6 and callable(args[1]):
                # Messages mode
                ctx, doc_to_messages_func, gen_kwargs, doc_id, task, split = args
                messages = doc_to_messages_func(self.task_dict[task][split][doc_id])
                system_prompt, user_text, visuals = self._parse_messages(messages)
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

            # Combine system prompt and user text as the question
            question_parts = []
            if system_prompt:
                question_parts.append(f"[System Instructions]\n{system_prompt}\n[End System Instructions]")
            if user_text:
                question_parts.append(user_text)
            question = "\n\n".join(question_parts)

            temperature = gen_kwargs.get("temperature", 0)

            # Build multipart request
            data, files = self._build_request(question, visuals, temperature)

            # Retry logic
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    # Need to re-open files on retry (file handles may be consumed)
                    if attempt > 0:
                        # Close previous file handles
                        for key in files:
                            try:
                                files[key][1].close()
                            except Exception:
                                pass
                        data, files = self._build_request(question, visuals, temperature)

                    resp = http_requests.post(
                        self.api_url,
                        data=data,
                        files=files if files else None,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    response_text = result.get("response", "")
                    response_text = self._clean_response(response_text)

                    if response_text:
                        eval_logger.debug(
                            f"VITA response (attempt {attempt+1}): {response_text[:200]}..."
                        )
                        break
                    else:
                        eval_logger.warning(
                            f"Attempt {attempt+1}/{self.max_retries}: empty response, retrying..."
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(NUM_SECONDS_TO_SLEEP)

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(
                            f"All {self.max_retries} attempts failed. Last error: {str(e)}"
                        )
                        response_text = ""
                finally:
                    # Close file handles
                    for key in list(files.keys()):
                        try:
                            files[key][1].close()
                        except Exception:
                            pass

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
        raise NotImplementedError("Multi-round generation not supported for VITA API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("VITA API does not support loglikelihood")
