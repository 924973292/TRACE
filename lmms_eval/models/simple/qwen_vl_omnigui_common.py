"""Shared helpers for Qwen-VL GUI wrappers (GUI-Owl, UI-TARS, MAI-UI).

ConfigurableMessagesTask passes doc_to_messages(doc) from utils.py unchanged.
These helpers only normalize media paths and Qwen-VL processor keys — they must
NOT inject or rewrite system prompts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.models.geopact_vl_hook import scope_requires_episode_order


def resolve_media_path(path: Union[str, object]) -> Union[str, object]:
    """Turn benchmark-relative paths into absolute local paths for qwen_vl_utils."""
    if not isinstance(path, str) or not path:
        return path
    if path.startswith(("data:", "http://", "https://")):
        return path
    local = path[7:] if path.startswith("file://") else path
    if os.path.isabs(local):
        return local
    return os.path.abspath(local)


def read_processor_vision_bounds(processor) -> dict:
    """Log-friendly snapshot of vision resize policy (matches qwen3_omni_chat init)."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return {}
    out = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(ip).__name__,
    }
    for key in ("min_pixels", "max_pixels", "patch_size", "merge_size", "size"):
        val = getattr(ip, key, None)
        if val is not None:
            out[key] = val
    return out


def convert_messages_to_qwen(
    messages: List[Dict],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> List[Dict]:
    """url/image_url -> image; optional min/max_pixels (OmniGUI Qwen-Omni uses none)."""
    qwen_messages: List[Dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        qwen_content = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                if text:
                    qwen_content.append({"type": "text", "text": text})
            elif item_type == "image":
                image = item.get("image", item.get("url", item.get("image_url")))
                if image:
                    image_item: Dict = {
                        "type": "image",
                        "image": resolve_media_path(image),
                    }
                    if min_pixels is not None:
                        image_item["min_pixels"] = min_pixels
                    if max_pixels is not None:
                        image_item["max_pixels"] = max_pixels
                    qwen_content.append(image_item)
            elif item_type == "video":
                video = item.get("video", item.get("url"))
                if video:
                    qwen_content.append({
                        "type": "video",
                        "video": resolve_media_path(video),
                    })

        if qwen_content:
            qwen_messages.append({"role": role, "content": qwen_content})
    return qwen_messages


def log_messages_debug(tag: str, raw_messages: List[Dict], qwen_messages: List[Dict]) -> None:
    if os.environ.get("GUIOWL_DEBUG_MESSAGES", "").lower() not in {"1", "true", "yes"}:
        return
    sys_prompt = ""
    for msg in raw_messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if isinstance(c, str):
                sys_prompt = c[:240]
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        sys_prompt = str(part.get("text", ""))[:240]
                        break
    image_paths = []
    for msg in qwen_messages:
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image":
                image_paths.append(part.get("image"))
    eval_logger.info(
        f"[{tag}] system_prompt_prefix={sys_prompt!r} n_images={len(image_paths)} "
        f"image_paths={image_paths[:4]}"
    )


def extract_qwen_image_paths(qwen_messages: List[Dict]) -> List[str]:
    """Chronological image paths from Qwen-VL messages (matches token_ranges order)."""
    paths: List[str] = []
    for msg in qwen_messages:
        for part in msg.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            url = part.get("image", "")
            if isinstance(url, str) and url:
                paths.append(_norm_geopact_path(url))
    return paths


def _flatten_images(image_inputs: Any) -> List[Any]:
    if image_inputs is None:
        return []
    if isinstance(image_inputs, (list, tuple)):
        out: List[Any] = []
        for item in image_inputs:
            if isinstance(item, (list, tuple)):
                out.extend(_flatten_images(item))
            else:
                out.append(item)
        return out
    return [image_inputs]


def _image_size(image: Any) -> tuple[Optional[int], Optional[int]]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    return None, None


def _path_image_size(path: str) -> tuple[Optional[int], Optional[int]]:
    try:
        if not path or str(path).startswith("data:"):
            return None, None
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _grid_resized_size(grid_row: Any, processor: Any = None) -> tuple[Optional[int], Optional[int], Optional[list]]:
    if grid_row is None:
        return None, None, None
    if hasattr(grid_row, "tolist"):
        vals = grid_row.tolist()
    else:
        vals = list(grid_row)
    if len(vals) < 3:
        return None, None, vals
    ip = getattr(processor, "image_processor", None)
    patch = int(getattr(ip, "patch_size", 14) or 14)
    return int(vals[2]) * patch, int(vals[1]) * patch, [int(v) for v in vals[:3]]


def derive_qwen_image_geometries(
    qwen_messages: List[Dict],
    image_inputs: Any = None,
    image_grid_thw: Any = None,
    processor: Any = None,
) -> List[Dict[str, Any]]:
    """Return per-image raw and Qwen-resized geometry in message image order."""
    paths = extract_qwen_image_paths(qwen_messages)
    flat_images = _flatten_images(image_inputs)
    grid_rows = []
    if image_grid_thw is not None:
        if hasattr(image_grid_thw, "detach"):
            image_grid_thw = image_grid_thw.detach().cpu()
        try:
            grid_rows = [image_grid_thw[i] for i in range(int(image_grid_thw.shape[0]))]
        except Exception:
            grid_rows = []

    out: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        raw_w, raw_h = _path_image_size(path)
        resized_w, resized_h = _image_size(flat_images[idx]) if idx < len(flat_images) else (None, None)
        grid_w, grid_h, grid = _grid_resized_size(grid_rows[idx], processor) if idx < len(grid_rows) else (None, None, None)
        qwen_w = grid_w or resized_w or raw_w
        qwen_h = grid_h or resized_h or raw_h
        out.append(
            {
                "index": idx,
                "image_path": path,
                "raw_width": raw_w,
                "raw_height": raw_h,
                "resized_width": resized_w,
                "resized_height": resized_h,
                "qwen_width": qwen_w,
                "qwen_height": qwen_h,
                "image_grid_thw": grid,
            }
        )
    return out


def _scale_coord_1000_to_resized(coord: Sequence[Any], geom: Dict[str, Any]) -> List[int]:
    w = max(int(geom.get("qwen_width") or geom.get("raw_width") or 1), 1)
    h = max(int(geom.get("qwen_height") or geom.get("raw_height") or 1), 1)
    return [int(round(float(coord[0]) / 1000.0 * w)), int(round(float(coord[1]) / 1000.0 * h))]


def _action_coords_to_resized(action: Dict[str, Any], geom: Dict[str, Any]) -> Dict[str, Any]:
    action = dict(action)
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        val = action.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            action[key] = _scale_coord_1000_to_resized(val, geom)
    return action


def _rewrite_native_assistant_to_resized(text: str, geom: Dict[str, Any]) -> str:
    try:
        from lmms_eval.tasks.agentcpm_gui.gui_owl_native_messages import format_native_assistant_turn
        from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action

        action = parse_omni_action(text)
        if not action or "action_type" not in action:
            return text
        return format_native_assistant_turn(_action_coords_to_resized(action, geom))
    except Exception:
        return text


def apply_resized_pixel_protocol_to_qwen_messages(
    qwen_messages: List[Dict],
    geometries: List[Dict[str, Any]],
) -> List[Dict]:
    """Patch native GUI-Owl messages to use Qwen-resized pixel coordinates."""
    try:
        from lmms_eval.tasks.agentcpm_gui.gui_owl_native_messages import (
            GUI_OWL_NATIVE_RESIZED_PIXEL_SYSTEM_PROMPT,
            GUI_OWL_NATIVE_SYSTEM_PROMPT,
        )
    except Exception:
        return qwen_messages

    image_idx = 0
    pending_image_idx: Optional[int] = None
    for msg in qwen_messages:
        role = msg.get("role")
        content = msg.get("content") or []
        for part in content:
            if not isinstance(part, dict):
                continue
            if role == "system" and part.get("type") == "text":
                text = str(part.get("text", ""))
                if text == GUI_OWL_NATIVE_SYSTEM_PROMPT or "normalized 1000x1000 coordinate space" in text:
                    part["text"] = GUI_OWL_NATIVE_RESIZED_PIXEL_SYSTEM_PROMPT
            elif part.get("type") == "image":
                pending_image_idx = image_idx
                image_idx += 1
        if role == "assistant" and pending_image_idx is not None and pending_image_idx < len(geometries):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = _rewrite_native_assistant_to_resized(str(part.get("text", "")), geometries[pending_image_idx])
            pending_image_idx = None

    if geometries:
        current = geometries[-1]
        current_w = int(current.get("qwen_width") or current.get("raw_width") or 0)
        current_h = int(current.get("qwen_height") or current.get("raw_height") or 0)
        note = (
            "Coordinate protocol for the current screenshot: the image provided "
            f"to the vision encoder is {current_w}x{current_h} pixels. "
            "Return click, long_press, and swipe coordinates in this resized "
            "image pixel space, using [x,y] with origin at the top-left. "
            "For a target element, click near its center; scoring is correct "
            "when the predicted point falls inside the resized ground-truth box."
        )
        for msg in reversed(qwen_messages):
            if msg.get("role") == "user":
                content = msg.setdefault("content", [])
                insert_at = 0
                content.insert(insert_at, {"type": "text", "text": note})
                break
    return qwen_messages


def _norm_geopact_path(path: str) -> str:
    if path.startswith("file://"):
        path = path[7:]
    return os.path.normpath(os.path.abspath(path))


def _geopact_episode_key_from_doc(doc: Dict) -> tuple[str, str, int]:
    subset = str(doc.get("subset", "unknown") or "unknown")
    episode_id = str(doc.get("episode_id", "unknown") or "unknown")
    episode_key = f"{subset}-{episode_id}"
    step_id = int(doc.get("step_id", doc.get("step_index", 0)) or 0)
    return episode_key, episode_id, step_id


def _reset_geopact_sticky_on_episode_change(model, episode_key: str, target=None) -> None:
    # `target` overrides the default `model.model` descent for carriers whose
    # patched modeling file reads `_geopact_*` off a different object (the
    # qwen2_5_vl patch reads off the ForConditionalGeneration itself).
    inner = target if target is not None else getattr(model, "model", model)
    prev_key = getattr(inner, "_geopact_sticky_episode_key", None)
    if prev_key == episode_key:
        return
    inner._geopact_sticky_masks = {}
    inner._geopact_nap_orders = {}
    inner._geopact_compact_frame_cache = {}
    inner._geopact_compact_frame_cache_hits = set()
    inner._geopact_sticky_relevance_query = None
    inner._geopact_sticky_relevance_query_source = None
    inner._geopact_turn_instr_by_path = {}
    inner._geopact_turn_query_cache = {}
    inner._geopact_sticky_episode_key = episode_key
    try:
        from lmms_eval.models.geopact_vl_hook import clear_geopact_mrope_caches

        clear_geopact_mrope_caches(model)
    except Exception as exc:
        eval_logger.warning(f"GeoPACT MRoPE cache clear failed: {exc}")
    try:
        from lmms_eval.models.geopact_session_kv import clear_session_kv

        clear_session_kv(model)
    except Exception as exc:
        inner._geopact_session_kv_state = None
        inner._geopact_session_full_prefill_backup = None
        eval_logger.warning(f"GeoPACT session KV clear failed: {exc}")


def extract_episode_image_paths_from_doc(
    doc: Dict,
    lmms_eval_specific_kwargs: Optional[Dict] = None,
) -> List[str]:
    """Chronological episode image paths without building full Qwen messages."""
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    include_history = bool(lmms_eval_specific_kwargs.get("qwen_include_history", True))

    def resolve_path(path: str) -> str:
        if path and media_dir and not os.path.isabs(path):
            return os.path.join(media_dir, path)
        return path

    paths: List[str] = []
    history = doc.get("history", [])
    if not history and doc.get("action_history"):
        from lmms_eval.tasks.agentcpm_gui.utils import _merged_action_history_to_qwen_history

        history = _merged_action_history_to_qwen_history(doc)

    from lmms_eval.tasks.history_window import (
        LOCKED_EVAL_HISTORY_N,
        effective_history_slice_n,
        slice_action_history_list,
    )

    if os.environ.get("GEOPACT_EVAL_HISTORY_N", "").strip():
        max_history = effective_history_slice_n()
    else:
        max_history = int(
            lmms_eval_specific_kwargs.get("max_history_turns", LOCKED_EVAL_HISTORY_N)
        )
    hist_steps = slice_action_history_list(history, n=max_history) if include_history else []
    for hist_step in hist_steps:
        hist_image = hist_step.get("image_path") or hist_step.get("image")
        if hist_image:
            paths.append(_norm_geopact_path(resolve_path(str(hist_image))))

    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        paths.append(_norm_geopact_path(resolve_path(str(current_image))))
    return paths


def extract_turn_instructions_from_doc(
    doc: Dict,
    lmms_eval_specific_kwargs: Optional[Dict] = None,
) -> List[str]:
    """Per-frame pure instruction texts, chronological, aligned with
    ``extract_episode_image_paths_from_doc`` (history frames with an image,
    then the current frame).

    Each entry is the same text the ``rich_turn`` protocol renders into that
    turn's ``Instruction:`` line (per-step instruction, falling back to the
    episode task goal) — no template boilerplate.
    """
    from lmms_eval.tasks.agentcpm_gui.utils import (
        _qwen_step_instruction,
        _qwen_task_description,
    )

    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    include_history = bool(lmms_eval_specific_kwargs.get("qwen_include_history", True))
    task_description = _qwen_task_description(doc)

    history = doc.get("history", [])
    if not history and doc.get("action_history"):
        from lmms_eval.tasks.agentcpm_gui.utils import _merged_action_history_to_qwen_history

        history = _merged_action_history_to_qwen_history(doc)

    from lmms_eval.tasks.history_window import (
        LOCKED_EVAL_HISTORY_N,
        effective_history_slice_n,
        slice_action_history_list,
    )

    if os.environ.get("GEOPACT_EVAL_HISTORY_N", "").strip():
        max_history = effective_history_slice_n()
    else:
        max_history = int(
            lmms_eval_specific_kwargs.get("max_history_turns", LOCKED_EVAL_HISTORY_N)
        )
    hist_steps = slice_action_history_list(history, n=max_history) if include_history else []
    instrs: List[str] = []
    for hist_step in hist_steps:
        if hist_step.get("image_path") or hist_step.get("image"):
            instrs.append(
                (_qwen_step_instruction(doc, hist_step) or task_description or "").strip()
            )

    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if current_image:
        instrs.append((_qwen_step_instruction(doc) or task_description or "").strip())
    return instrs


def set_geopact_episode_image_context(
    model,
    doc: Dict,
    image_paths: List[str],
    lmms_eval_specific_kwargs: Optional[Dict] = None,
    target=None,
) -> None:
    """Set episode/image-path context without rebuilding full messages.

    `target` overrides the default `model.model` descent (see
    `_reset_geopact_sticky_on_episode_change`).
    """
    inner = target if target is not None else getattr(model, "model", model)
    scope = str(getattr(inner, "_geopact_prune_scope", "") or "").lower()
    sticky = scope_requires_episode_order(scope)
    episode_key, episode_id, step_id = _geopact_episode_key_from_doc(doc)
    _reset_geopact_sticky_on_episode_change(model, episode_key, target=target)
    inner._geopact_episode_key = episode_key
    inner._geopact_image_paths = list(image_paths)
    inner._geopact_step_index = step_id
    from lmms_eval.tasks.agentcpm_gui.utils import _qwen_task_description

    inner._geopact_task_instruction_text = _qwen_task_description(doc)

    # Frame -> per-turn pure instruction (turn t's frame must be pruned with
    # turn t's instruction). Aligned by chronological order with image_paths;
    # on any mismatch keep only the current-frame entry and let older frames
    # fall back to the episode-task query.
    turn_map: Dict[str, str] = {}
    try:
        instrs = extract_turn_instructions_from_doc(doc, lmms_eval_specific_kwargs)
        if instrs and len(instrs) == len(image_paths):
            for path, instr in zip(image_paths, instrs):
                if instr:
                    turn_map[_norm_geopact_path(str(path))] = instr
        elif instrs and image_paths:
            turn_map[_norm_geopact_path(str(image_paths[-1]))] = instrs[-1]
    except Exception as exc:
        eval_logger.warning(f"GeoPACT turn-instruction map build failed: {exc}")
    inner._geopact_turn_instr_by_path = turn_map

    if sticky:
        if episode_id in {"", "unknown", "None"}:
            from lmms_eval.tasks.history_window import task_supports_session_kv

            if task_supports_session_kv(lmms_eval_specific_kwargs):
                raise RuntimeError(
                    f"GeoPACT sticky: missing episode_id on doc (subset={doc.get('subset')}, step={step_id})"
                )
            # Single-step benchmarks (screenspot / mmbench / ac-high) carry no
            # episode structure: sticky degrades to per-image keys under the
            # constant "unknown" episode — the behavior every validated
            # single-step run used. Only multi-step tasks must fail loud.
        if not image_paths:
            raise RuntimeError(
                f"GeoPACT sticky: no image paths for episode_key={episode_key}"
            )


def prepare_geopact_episode_context(
    model,
    doc: Dict,
    qwen_messages: List[Dict],
    lmms_eval_specific_kwargs: Optional[Dict] = None,
    target=None,
) -> None:
    """Inject episode/image-path context for image_sticky pruning."""
    episode_key, episode_id, step_id = _geopact_episode_key_from_doc(doc)
    _reset_geopact_sticky_on_episode_change(model, episode_key, target=target)
    paths = extract_qwen_image_paths(qwen_messages)
    if not paths:
        paths = extract_episode_image_paths_from_doc(doc, lmms_eval_specific_kwargs)
    set_geopact_episode_image_context(
        model, doc, paths, lmms_eval_specific_kwargs, target=target
    )


@dataclass
class ModuleAPrewarmHandle:
    """Background Module A worker prewarm (overlaps processor/tokenizer init)."""

    started_at: float
    thread: Optional[threading.Thread] = None
    done_at: Optional[float] = None
    prewarm_ms: Optional[float] = None
    error: Optional[str] = None
    joined: bool = False


def _module_a_subprocess_prewarm_wanted() -> bool:
    if os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_PREWARM", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    if os.environ.get("GEOPACT_MODULE_A_SUBPROCESS", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return True


def module_a_prewarm_async_enabled() -> bool:
    if not _module_a_subprocess_prewarm_wanted():
        return False
    return os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_PREWARM_ASYNC", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def start_module_a_prewarm_background() -> Optional[ModuleAPrewarmHandle]:
    """Spawn worker prewarm thread (call right after Qwen ``from_pretrained``)."""
    if not module_a_prewarm_async_enabled():
        return None

    handle = ModuleAPrewarmHandle(started_at=time.perf_counter())

    def _run() -> None:
        try:
            from lmms_eval.models.geopact_module_a_worker import prewarm_module_a_worker

            t0 = time.perf_counter()
            if prewarm_module_a_worker():
                handle.prewarm_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            handle.error = repr(exc)
        finally:
            handle.done_at = time.perf_counter()

    handle.thread = threading.Thread(target=_run, name="module_a_prewarm", daemon=True)
    handle.thread.start()
    eval_logger.info("GeoPACT: Module A worker prewarm started (background)")
    return handle


def join_module_a_prewarm(
    handle: Optional[ModuleAPrewarmHandle],
    *,
    where: str = "init",
) -> Optional[float]:
    """Wait for background prewarm; return worker wall ms if started."""
    if handle is None or handle.joined:
        return None
    join_t0 = time.perf_counter()
    if handle.thread is None:
        return None
    handle.thread.join()
    join_wait_ms = (time.perf_counter() - join_t0) * 1000.0
    handle.joined = True
    if handle.error:
        eval_logger.warning(
            "GeoPACT: Module A worker prewarm failed ({}): {}",
            where,
            handle.error,
        )
        return None
    if handle.prewarm_ms is not None:
        overlap_ms = max(0.0, float(handle.prewarm_ms) - join_wait_ms)
        eval_logger.info(
            "GeoPACT: Module A worker prewarm done ({}): worker_ms={:.0f} "
            "join_wait_ms={:.0f} overlap_saved_ms={:.0f}",
            where,
            handle.prewarm_ms,
            join_wait_ms,
            overlap_ms,
        )
        return float(handle.prewarm_ms)
    return None


def prewarm_module_a_subprocess_in_eval_process() -> Optional[float]:
    """Start pooled Module A worker in the current eval process (after Qwen load).

    Shell ``python -c prewarm`` in a separate process does not help lmms_eval;
    this must run in the same process that calls generate().
    Returns prewarm wall ms when worker started, else None.
    """
    if not _module_a_subprocess_prewarm_wanted():
        return None
    try:
        from lmms_eval.models.geopact_module_a_worker import prewarm_module_a_worker

        t0 = time.perf_counter()
        if not prewarm_module_a_worker():
            return None
        ms = (time.perf_counter() - t0) * 1000.0
        eval_logger.info(
            "GeoPACT: Module A subprocess worker prewarmed in eval process ({:.0f} ms)",
            ms,
        )
        return ms
    except Exception as exc:
        eval_logger.warning("GeoPACT: Module A worker prewarm failed: {!r}", exc)
        return None


def cache_geopact_image(model, qwen_messages: List[Dict], target=None) -> None:
    """Store the current (last) screenshot on inner model for GeoPACT MIU detection.

    `target` overrides the default descent to `model.model` for carriers whose
    patched modeling file reads `_geopact_*` off a different object (e.g. the
    qwen2_5_vl patch reads off the ForConditionalGeneration itself).
    """
    inner = target if target is not None else getattr(model, "model", model)
    last_img = None
    for msg in qwen_messages:
        for part in msg.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            url = part.get("image", "")
            if not isinstance(url, str) or not url:
                continue
            if url.startswith("data:"):
                import base64
                from io import BytesIO

                _, b64data = url.split(",", 1)
                last_img = Image.open(BytesIO(base64.b64decode(b64data))).convert("RGB")
            else:
                last_img = Image.open(url).convert("RGB")
    if last_img is not None:
        inner._geopact_last_image = last_img
    elif getattr(inner, "_geopact_projector_prune_hook", None) is not None:
        eval_logger.warning("GeoPACT: no PIL image loaded for current frame (MIU detection may fail)")


USAGE_MARKER_START = "<MINICPM_O_USAGE_JSON>"
USAGE_MARKER_END = "</MINICPM_O_USAGE_JSON>"


def format_response_with_usage(text: str, usage: Dict[str, Any]) -> str:
    if not usage:
        return text
    return (
        f"{text}{USAGE_MARKER_START}"
        f"{json.dumps(usage, ensure_ascii=False)}"
        f"{USAGE_MARKER_END}"
    )


def canonicalize_omni_action_output(raw: str) -> tuple[str, Optional[str]]:
    """Normalize verbose GUI-Owl text to compact Omni JSON for eval/logging.

    Uses the same ``parse_omni_action`` as ``omni_process_results``. When the
    model wraps JSON in natural language, returns only the action JSON string
    and optionally the original raw text (for debugging in usage stats).
    """
    raw_stripped = (raw or "").strip()
    if not raw_stripped:
        return raw_stripped, None
    try:
        from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action
    except Exception:
        return raw_stripped, None

    action = parse_omni_action(raw_stripped)
    if not action or "action_type" not in action:
        return raw_stripped, None

    payload: Dict[str, Any] = {"action_type": int(action["action_type"])}
    # Keep source-space metadata so evaluate_omni_step can apply the same
    # conservative equivalence rules as raw GUI-Owl mobile_use output
    # (e.g. open(app) on launcher step 0, answer/terminate completion).
    for key in ("_tool_call_name", "_mobile_use_action"):
        val = action.get(key)
        if val is not None and val != "":
            payload[key] = val
    for key in (
        "coordinate",
        "start_coordinate",
        "end_coordinate",
        "text",
        "duration",
        "button",
        "status",
    ):
        val = action.get(key)
        if val is not None and val != "":
            payload[key] = val

    canonical = json.dumps(payload, ensure_ascii=False)
    if canonical == raw_stripped:
        return raw_stripped, None
    return canonical, raw_stripped


def count_qwen_vision_tokens(input_ids: Any, model: Any) -> int:
    """Count image + video placeholder tokens in the prompt."""
    if input_ids is None:
        return 0
    inner = getattr(model, "model", model)
    config = getattr(inner, "config", None)
    if config is None:
        return 0
    ids = input_ids[0] if hasattr(input_ids, "shape") and len(input_ids.shape) > 1 else input_ids
    if hasattr(ids, "tolist"):
        ids_list = ids.tolist()
    else:
        ids_list = list(ids)
    token_ids: set[int] = set()
    for attr in ("image_token_id", "video_token_id"):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val >= 0:
            token_ids.add(val)
    if not token_ids:
        return 0
    return sum(1 for tok in ids_list if int(tok) in token_ids)


def _gpu_env_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "batch_size": 1,
        "precision": os.environ.get("GEOPACT_PRECISION", "bf16"),
        "kv_cache_enabled": os.environ.get("GEOPACT_VISION_EMBED_CACHE", "1").strip().lower()
        not in {"0", "false", "off", "no"},
    }
    try:
        import torch

        if torch.cuda.is_available():
            meta["gpu_model"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return meta


def build_qwen_vl_usage_stats(
    *,
    step_t0: float,
    preprocess_ms: float,
    generate_t0: float,
    gen_timing: Dict[str, float],
    prompt_len: int,
    output_tokens: int,
    peak_mem_mb: float = 0.0,
    avg_mem_mb: float = 0.0,
    compact_stats: Optional[Dict[str, Any]] = None,
    vision_tokens: int = 0,
    history_tokens: int = 0,
    processor_prompt_len: int = 0,
    extra_llm_prefill_ms: float = 0.0,
    exclusive_timings: Optional[Dict[str, float]] = None,
    timing_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build latency payload embedded in model output for omni_process_results.

    vlm_prefill_ms = vision encoder (+ merger inside the visual module).
    llm_prefill_forward_ms = LM prefill on the prompt (generate hooks + any
    external incremental prefill such as dialogue KV batch suffix forward).
    prefill_ms ≈ vision_prefill_ms + llm_prefill_ms (sequential stages).

    ttft_ms = wall time from step start through first generated token, including
    processor / GeoPACT prepare / external dialogue batch prefill before generate().
    """
    from lmms_eval.models.geopact_step_timing import (
        llm_prefill_from_exclusive,
        vision_prefill_from_exclusive,
    )

    compact_stats = compact_stats or {}
    generate_ms = float(gen_timing.get("generation_ms", 0.0))
    ttft_from_generate = float(gen_timing.get("ttft_ms", 0.0))
    step_ttft_ms = (generate_t0 - step_t0) * 1000.0 + ttft_from_generate

    vlm_ms = float(
        gen_timing.get(
            "source_image_feature_prefill_ms",
            gen_timing.get("vlm_prefill_ms", 0.0),
        )
    )
    llm_prefill_ms = float(
        gen_timing.get(
            "llm_prefill_forward_ms",
            gen_timing.get("prefill_forward_ms", 0.0),
        )
    ) + float(extra_llm_prefill_ms or 0.0)
    decode_ms = float(
        gen_timing.get(
            "llm_decode_forward_ms",
            gen_timing.get("decode_total_ms", gen_timing.get("decode_forward_ms", 0.0)),
        )
    )
    geopact_prepare_ms = float(
        compact_stats.get("geopact_prepare_ms", gen_timing.get("geopact_prepare_ms", 0.0))
    )
    pruning_ms = float(compact_stats.get("pruning_ms", gen_timing.get("pruning_ms", 0.0)))
    exclusive = dict(exclusive_timings or {})
    if exclusive:
        vision_prefill_ms = vision_prefill_from_exclusive(exclusive)
        llm_prefill_ms = llm_prefill_from_exclusive(exclusive)
        decode_ms = float(exclusive.get("llm_decode_ms", decode_ms))
        vlm_ms = float(exclusive.get("vlm_prefill_ms", vlm_ms))
        prefill_ms = vision_prefill_ms + llm_prefill_ms
    else:
        llm_prefill_ms = float(
            gen_timing.get(
                "llm_prefill_forward_ms",
                gen_timing.get("prefill_forward_ms", 0.0),
            )
        ) + float(extra_llm_prefill_ms or 0.0)
        prefill_ms = float(gen_timing.get("prefill_ms", vlm_ms + llm_prefill_ms))
        if extra_llm_prefill_ms:
            prefill_ms = max(prefill_ms, vlm_ms + llm_prefill_ms)
        vision_prefill_ms = preprocess_ms + max(geopact_prepare_ms - pruning_ms, 0.0) + vlm_ms
    decode_per_token = decode_ms / output_tokens if output_tokens > 0 else 0.0
    throughput = output_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
    ledger_e2e = None
    if exclusive:
        ledger_e2e = exclusive.get("timing_e2e_ms")
        if ledger_e2e is None and isinstance(exclusive.get("timings_exclusive_ms"), dict):
            ledger_e2e = exclusive["timings_exclusive_ms"].get("timing_e2e_ms")
    e2e_ms = float(ledger_e2e) if ledger_e2e is not None else (time.perf_counter() - step_t0) * 1000.0

    orig_seq_len = int(
        compact_stats.get("orig_seq_len", processor_prompt_len or prompt_len)
    )
    compact_seq_len = int(compact_stats.get("compact_seq_len", prompt_len))
    tokens_before = int(compact_stats.get("tokens_before_pruning", orig_seq_len or prompt_len))
    tokens_after = int(compact_stats.get("tokens_after_pruning", compact_seq_len or prompt_len))
    if tokens_before <= 0:
        tokens_before = processor_prompt_len or prompt_len
    if tokens_after <= 0:
        tokens_after = prompt_len
    token_reduction_ratio = (
        (tokens_before - tokens_after) / tokens_before if tokens_before > 0 else 0.0
    )

    if vision_tokens <= 0:
        vision_tokens = int(
            compact_stats.get("n_visual_kept")
            or compact_stats.get("visual_tokens_kept")
            or compact_stats.get("n_visual_total")
            or compact_stats.get("visual_tokens_total")
            or 0
        )
    if history_tokens < 0:
        history_tokens = 0
    if history_tokens == 0 and prompt_len > vision_tokens:
        history_tokens = max(prompt_len - vision_tokens, 0)

    peak_mb = peak_mem_mb
    avg_mb = avg_mem_mb if avg_mem_mb > 0 else peak_mem_mb

    usage: Dict[str, Any] = {
        "input_tokens": int(prompt_len),
        "output_tokens": int(output_tokens),
        "total_tokens": int(prompt_len + output_tokens),
        "preprocess_ms": round(preprocess_ms, 2),
        "generation_ms": round(generate_ms, 2),
        "e2e_latency_ms": round(e2e_ms, 2),
        "timing_e2e_ms": round(e2e_ms, 2),
        "ttft_ms": round(step_ttft_ms, 2),
        "prefill_ms": round(prefill_ms, 2),
        "vlm_prefill_ms": round(vlm_ms, 2),
        "vision_prefill_ms": round(vision_prefill_ms, 2),
        "geopact_prepare_ms": round(geopact_prepare_ms, 2),
        "pruning_ms": round(pruning_ms, 2),
        "sticky_keep_ms": round(
            float((exclusive or {}).get("sticky_keep_ms", pruning_ms)),
            2,
        ),
        "llm_prefill_forward_ms": round(llm_prefill_ms, 2),
        "llm_prefill_ms": round(llm_prefill_ms, 2),
        "decode_total_ms": round(decode_ms, 2),
        "decode_latency_ms": round(decode_ms, 2),
        "decode_per_token_ms": round(decode_per_token, 4),
        "throughput_tokens_per_s": round(throughput, 2),
        "forward_calls": float(gen_timing.get("forward_calls", 0.0)),
        "decode_forward_calls": float(gen_timing.get("decode_forward_calls", 0.0)),
        "gpu_memory_gb": round(peak_mb / 1024.0, 4) if peak_mb > 0 else 0.0,
        "peak_gpu_memory_mb": round(peak_mb, 2),
        "average_gpu_memory_mb": round(avg_mb, 2),
        "vision_tokens": int(vision_tokens),
        "history_tokens": int(history_tokens),
        "tokens_before_pruning": tokens_before,
        "tokens_after_pruning": tokens_after,
        "token_reduction_ratio": round(token_reduction_ratio, 6),
        "orig_seq_len": orig_seq_len,
        "compact_seq_len": compact_seq_len,
        "processor_prompt_tokens": int(processor_prompt_len or prompt_len),
    }
    if exclusive:
        usage["timings_exclusive_ms"] = {k: round(float(v), 2) for k, v in exclusive.items()}
        usage["timing_exclusive_sum_ms"] = round(
            sum(float(exclusive.get(k, 0.0) or 0.0) for k in exclusive if k != "misc_ms"),
            2,
        )
        usage["misc_ms"] = round(float(exclusive.get("misc_ms", 0.0)), 2)
        usage["kv_finalize_ms"] = round(
            float(exclusive.get("dialogue_kv_finalize_ms", 0.0) or 0.0),
            2,
        )
    if timing_audit:
        usage["timing_audit"] = dict(timing_audit)
        for key, value in timing_audit.items():
            if key not in usage:
                usage[key] = value
    tokens_pruned = max(tokens_before - tokens_after, 0)
    usage["tokens_pruned"] = int(tokens_pruned)
    usage["llm_tokens"] = max(int(tokens_after) - int(vision_tokens), 0)
    usage["keep_ratio"] = round(tokens_after / tokens_before, 6) if tokens_before > 0 else 1.0
    usage.update(_gpu_env_metadata())
    return usage


def log_vision_preprocess_debug(
    tag: str,
    doc: Dict,
    inputs: object,
    *,
    explicit_min_pixels: Optional[int] = None,
    explicit_max_pixels: Optional[int] = None,
) -> None:
    """Log screenshot vs vision-tensor geometry when GUIOWL_DEBUG_VISION=1."""
    if os.environ.get("GUIOWL_DEBUG_VISION", "").lower() not in {"1", "true", "yes"}:
        return
    grid = None
    pixel_values = None
    if hasattr(inputs, "image_grid_thw"):
        grid = inputs.image_grid_thw
        pixel_values = getattr(inputs, "pixel_values", None)
    elif isinstance(inputs, dict):
        grid = inputs.get("image_grid_thw")
        pixel_values = inputs.get("pixel_values")
    grid_list = grid.tolist() if grid is not None and hasattr(grid, "tolist") else grid
    pv_shape = tuple(pixel_values.shape) if pixel_values is not None and hasattr(pixel_values, "shape") else None
    eval_logger.info(
        f"[{tag}] doc_image={doc.get('image_width')}x{doc.get('image_height')} "
        f"image_grid_thw={grid_list} pixel_values_shape={pv_shape} "
        f"explicit_bounds=min={explicit_min_pixels} max={explicit_max_pixels}"
    )


def assert_messages_request(request_args: tuple, model_name: str) -> None:
    """ConfigurableMessagesTask must pass doc_to_messages as args[1]."""
    if len(request_args) < 2 or not callable(request_args[1]):
        raise RuntimeError(
            f"{model_name} requires ConfigurableMessagesTask (is_simple=False). "
            f"Got request args layout {len(request_args)}-tuple; index 1 is not callable. "
            "Check task_type=chat and yaml doc_to_messages."
        )
