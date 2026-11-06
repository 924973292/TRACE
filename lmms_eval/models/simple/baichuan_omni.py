"""
Baichuan-Omni-1.5 API Model Adapter for lmms-eval

This module provides integration with Baichuan-Omni-1.5 model via its custom FastAPI server.
The server exposes a /chat endpoint that accepts multipart form data
with instruction text, input_type, and an optional file upload.

Note: Baichuan server only accepts ONE file per request.
When multiple media types are present, we prioritize: video > image > audio,
since video contains the most information (visual + temporal).

Usage:
    python -m lmms_eval \
        --model baichuan_omni \
        --model_args api_url=$BAICHUAN_OMNI_API_URL \
        --tasks task_v2_baichuan_omni \
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
from lmms_eval.models.model_utils.env import get_env_value

NUM_SECONDS_TO_SLEEP = 5


@register_model("baichuan_omni")
class BaichuanOmni(lmms):
    """
    Baichuan-Omni-1.5 model adapter using custom FastAPI HTTP endpoint.

    The server accepts multipart form data:
    - instruction: str (text prompt)
    - input_type: str (one of: "text", "audio", "image", "video")
    - file: UploadFile (optional, single file)

    Returns JSON: {"response": str, "audio_base64": str|null}
    Response may end with "<|endoftext|>" which needs to be stripped.

    Important: Only ONE file is supported per request. When the message
    contains multiple media types, priority is: video > image > audio.
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages

    def __init__(
        self,
        api_url: str = None,
        timeout: int = 180,
        max_retries: int = 5,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.api_url = api_url or get_env_value(
            ["BAICHUAN_OMNI_API_URL", "BAICHUAN_OMNI_BASE_URL"],
            required=True,
            description="Baichuan-Omni API URL",
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.continual_mode = continual_mode

        eval_logger.info(f"Initialized BaichuanOmni with endpoint: {self.api_url}")

        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/baichuan_omni_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                "baichuan_omni_response.json",
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
            where all_visuals is a list of (type, path) tuples:
              type is one of: "image", "video", "audio"
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
        """Strip Baichuan-specific artifacts from response."""
        if not text:
            return ""
        # Strip trailing <|endoftext|>
        if text.endswith("<|endoftext|>"):
            text = text[: -len("<|endoftext|>")].strip()
        return text.strip()

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
    # Select primary media (Baichuan only supports ONE file per request)
    # ------------------------------------------------------------------
    @staticmethod
    def _select_primary_media(visuals: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Select the most informative single media file.

        Priority: video > image > audio
        For each type, take the LAST occurrence (most recent/primary).

        Returns:
            Tuple of (input_type, file_path) or ("text", "") if no media
        """
        video_path = None
        image_path = None
        audio_path = None

        for vtype, vpath in visuals:
            if vtype == "video":
                video_path = vpath
            elif vtype == "image":
                image_path = vpath
            elif vtype == "audio":
                audio_path = vpath

        # Priority: video > image > audio
        if video_path and os.path.exists(video_path):
            return "video", video_path
        if image_path and os.path.exists(image_path):
            return "image", image_path
        if audio_path and os.path.exists(audio_path):
            return "audio", audio_path

        return "text", ""

    # ------------------------------------------------------------------
    # Build multipart request
    # ------------------------------------------------------------------
    def _build_request(
        self, instruction: str, visuals: List[Tuple[str, str]]
    ) -> Tuple[dict, dict, str]:
        """
        Build multipart form data for the Baichuan /v1/chat/multimodal endpoint.

        Args:
            instruction: The text prompt (system + user combined)
            visuals: List of (type, path) tuples

        Returns:
            Tuple of (data dict, files dict, selected_input_type)
        """
        input_type, file_path = self._select_primary_media(visuals)

        data = {
            "text_query": instruction,
            "generate_audio": "false",
        }
        files = {}

        if file_path:
            mime = self._get_mime_type(file_path)
            # Map input_type to the correct form field name
            file_field_map = {
                "video": "video_file",
                "image": "image_file",
                "audio": "audio_file",
            }
            field_name = file_field_map.get(input_type, "video_file")
            files[field_name] = (os.path.basename(file_path), open(file_path, "rb"), mime)

        return data, files, input_type

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Baichuan-Omni Responding",
        )

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

            # Combine system prompt and user text as the instruction
            instruction_parts = []
            if system_prompt:
                instruction_parts.append(
                    f"[System Instructions]\n{system_prompt}\n[End System Instructions]"
                )
            if user_text:
                instruction_parts.append(user_text)
            instruction = "\n\n".join(instruction_parts)

            # Build multipart request
            data, files, input_type = self._build_request(instruction, visuals)

            # Retry logic
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    # Need to re-open files on retry
                    if attempt > 0:
                        for key in files:
                            try:
                                files[key][1].close()
                            except Exception:
                                pass
                        data, files, input_type = self._build_request(instruction, visuals)

                    resp = http_requests.post(
                        self.api_url,
                        data=data,
                        files=files if files else None,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    # Support both old {"response": "..."} and new {"text": "..."} format
                    response_text = result.get("text", "") or result.get("response", "")
                    response_text = self._clean_response(response_text)

                    if response_text:
                        eval_logger.debug(
                            f"Baichuan response (attempt {attempt+1}): {response_text[:200]}..."
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
        raise NotImplementedError("Multi-round generation not supported for Baichuan-Omni")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Baichuan-Omni API does not support loglikelihood")
