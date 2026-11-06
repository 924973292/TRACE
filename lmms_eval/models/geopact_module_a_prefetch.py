"""Async Module A prefetch: overlap subprocess detect with ViT encode."""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="module_a_prefetch",
            )
        return _EXECUTOR


def module_a_prefetch_async_enabled() -> bool:
    if os.environ.get("GEOPACT_MODULE_A_PREFETCH_ASYNC", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    from lmms_eval.models.geopact_module_a_worker import module_a_subprocess_pool_enabled
    from lmms_eval.models.geopact_vl_hook import (
        _module_a_prefetch_enabled,
        _module_a_subprocess_enabled,
    )

    return (
        _module_a_prefetch_enabled()
        and _module_a_subprocess_enabled()
        and module_a_subprocess_pool_enabled()
    )


def n_img_from_image_grid_thw(image_grid_thw) -> int:
    from lmms_eval.models.geopact_vl_hook import (
        _flat_image_token_ranges,
        _llm_tokens_per_image,
        _spatial_merge_size,
    )

    if image_grid_thw is None or getattr(image_grid_thw, "numel", lambda: 0)() < 3:
        return 0
    merge = _spatial_merge_size()
    total = 0
    for i in range(int(image_grid_thw.shape[0])):
        t, h, w = (int(v) for v in image_grid_thw[i].tolist())
        total += _llm_tokens_per_image(t, h, w, merge)
    if total > 0:
        return total
    ranges = _flat_image_token_ranges(image_grid_thw, max(total, 1))
    if ranges:
        return ranges[-1][1]
    return 0


def prefetch_grid_context(model, image_grid_thw) -> Optional[Tuple[int, int]]:
    from lmms_eval.models.geopact_vl_hook import (
        _flat_image_token_ranges,
        _frame_grid,
        _set_module_a_grid_context,
    )

    n_img = n_img_from_image_grid_thw(image_grid_thw)
    if n_img <= 0:
        return None
    token_ranges = _flat_image_token_ranges(image_grid_thw, n_img)
    if not token_ranges:
        return None
    range_idx = len(token_ranges) - 1
    start, end = token_ranges[range_idx]
    n_frame = end - start
    n_frames = len(token_ranges)
    frame_grid = _frame_grid(image_grid_thw, range_idx, n_frames)
    _set_module_a_grid_context(model, frame_grid, n_frame)
    return getattr(model, "_geopact_module_a_grid_hw", None)


def cancel_module_a_prefetch(model) -> None:
    if model is None:
        return
    handle = getattr(model, "_geopact_module_a_async", None)
    if not isinstance(handle, dict):
        return
    future: Optional[Future] = handle.get("future")
    if future is not None and not future.done():
        try:
            future.result(timeout=int(os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_TIMEOUT", "300")))
        except Exception:
            pass
    model._geopact_module_a_async = None


def discard_module_a_prefetch(model) -> None:
    """Drop any pending prefetch handle without joining it (episode boundary)."""
    if model is None:
        return
    if getattr(model, "_geopact_module_a_async", None) is not None:
        model._geopact_module_a_async = None


def _detect_job(
    pil_image,
    detector: str,
    policy: str,
    grid_hw: Optional[Tuple[int, int]],
) -> Tuple[List[dict], Dict[str, Any]]:
    from lmms_eval.models.geopact_module_a_worker import get_module_a_worker

    return get_module_a_worker().detect(pil_image, detector, policy, grid_hw=grid_hw)


def _image_identity(pil_image, image_id: Optional[str]) -> Optional[str]:
    """Stable identity for a prefetch target: prefer resolved path, else size."""
    if image_id:
        return str(image_id)
    if pil_image is None:
        return None
    try:
        w, h = pil_image.size
        return f"size:{w}x{h}"
    except Exception:
        return None


def start_module_a_prefetch(
    model,
    pil_image,
    detector: str,
    policy: str,
    image_grid_thw,
    *,
    image_id: Optional[str] = None,
    source: str = "same_step",
) -> bool:
    if model is None or pil_image is None:
        return False
    cancel_module_a_prefetch(model)
    grid_hw = prefetch_grid_context(model, image_grid_thw)
    started_at = time.perf_counter()
    future = _executor().submit(_detect_job, pil_image, detector, policy, grid_hw)
    model._geopact_module_a_async = {
        "future": future,
        "started_at": started_at,
        "vit_done_at": None,
        "detector": detector,
        "policy": policy,
        "image_id": _image_identity(pil_image, image_id),
        "grid_hw": grid_hw,
        "source": source,
    }
    return True


def mark_vit_done_for_module_a_prefetch(model) -> None:
    if model is None:
        return
    handle = getattr(model, "_geopact_module_a_async", None)
    if isinstance(handle, dict) and handle.get("vit_done_at") is None:
        handle["vit_done_at"] = time.perf_counter()


def _record_prefetch_timing(model, handle: Dict[str, Any], meta: Dict[str, Any]) -> None:
    done_at = time.perf_counter()
    started_at = float(handle.get("started_at") or done_at)
    vit_done_at = handle.get("vit_done_at")
    wall_ms = (done_at - started_at) * 1000.0
    overlap_ms = 0.0
    join_ms = wall_ms
    if vit_done_at is not None:
        vit_ms = (float(vit_done_at) - started_at) * 1000.0
        overlap_ms = max(0.0, min(wall_ms, vit_ms))
        join_ms = max(0.0, (done_at - float(vit_done_at)) * 1000.0)
    source = str(handle.get("source") or "same_step")
    reused = source == "cross_step"
    timings = dict(meta.get("timings_ms") or {})
    timings["prefetch_async_wall_ms"] = round(wall_ms, 2)
    timings["prefetch_async_overlap_ms"] = round(overlap_ms, 2)
    timings["prefetch_async_join_ms"] = round(join_ms, 2)
    meta["timings_ms"] = timings
    model._geopact_module_a_prefetch_async_join_ms = join_ms
    model._geopact_module_a_prefetch_async_overlap_ms = overlap_ms
    diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
    diag["module_a_prefetch_async"] = True
    diag["module_a_overlap_ms"] = round(overlap_ms, 2)
    diag["module_a_join_ms"] = round(join_ms, 2)
    diag["prefetch_reused"] = reused
    diag["prefetch_source"] = source
    diag["prefetch_grid_match"] = bool(handle.get("grid_match", False))
    model._geopact_last_prune_diag = diag


def _clear_prefetch_timing(model) -> None:
    """Reset the per-step join/overlap attrs on any non-consume path.

    They are written only on successful consume; without this reset a
    prefetch miss (sync re-detect) would let `_attach_module_a_prefetch_diag`
    report the PREVIOUS step's join/overlap as current-step evidence.
    """
    model._geopact_module_a_prefetch_async_join_ms = None
    model._geopact_module_a_prefetch_async_overlap_ms = None


def try_consume_module_a_prefetch(
    model,
    *,
    expected_image_id: Optional[str] = None,
) -> Optional[Tuple[List[dict], Dict[str, Any]]]:
    if model is None:
        return None
    handle = getattr(model, "_geopact_module_a_async", None)
    if not isinstance(handle, dict):
        _clear_prefetch_timing(model)
        return None
    future: Optional[Future] = handle.get("future")
    if future is None:
        model._geopact_module_a_async = None
        _clear_prefetch_timing(model)
        return None
    handle_image_id = str(handle.get("image_id") or "")
    if expected_image_id and handle_image_id and str(expected_image_id) != handle_image_id:
        # Stale prefetch for a different frame (e.g. last frame of the previous
        # episode): never splice its MIUs onto this frame; drop it and re-detect.
        cancel_module_a_prefetch(model)
        _clear_prefetch_timing(model)
        return None
    mius, meta = future.result(
        timeout=int(os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_TIMEOUT", "300"))
    )
    if str(handle.get("source") or "") == "cross_step" and not handle.get("grid_match", False):
        # Cross-step prefetch runs on a placeholder grid (raw PIL // 28 // merge);
        # if it was never refreshed to the real smart-resize grid, per-MIU token
        # counts and info_density are computed on the wrong grid. Fail loud so a
        # silent P_j corruption can never enter an experiment unnoticed.
        _log.warning(
            "module_a cross_step prefetch consumed with grid_match=False "
            "(placeholder grid %s); info_density/P_j may be inconsistent with the "
            "real vision grid for image_id=%s",
            handle.get("grid_hw"),
            handle.get("image_id"),
        )
    model._geopact_module_a_async = None
    _record_prefetch_timing(model, handle, meta)
    model._geopact_last_module_a_meta = meta
    model._geopact_module_a_subprocess_isolated = True
    model._geopact_module_a_subprocess_pooled = True
    cache_meta: Dict[str, Any] = {
        "module_a_subprocess": True,
        "module_a_subprocess_pool": True,
        "module_a_cache_hit": False,
        "module_a_prefetch_async": True,
    }
    return list(mius), cache_meta


def _current_image_id(model) -> Optional[str]:
    paths = getattr(model, "_geopact_image_paths", None) or []
    if paths:
        return str(paths[-1])
    return None


def _reuse_cross_step_prefetch(
    model,
    image_grid_thw,
    detector: str,
    policy: str,
    cur_image_id: Optional[str],
) -> bool:
    """Keep an in-flight/finished cross-step prefetch if it targets this image.

    Returns True when the existing handle is reused (no cancel + re-detect).
    Updates the real grid context so consume-time uses the correct grid_hw.
    """
    if os.environ.get("GEOPACT_MODULE_A_PREFETCH_REUSE", "0").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    handle = getattr(model, "_geopact_module_a_async", None)
    if not isinstance(handle, dict):
        return False
    if str(handle.get("source") or "") != "cross_step":
        return False
    if handle.get("future") is None:
        return False
    if str(handle.get("detector") or "") != detector or str(handle.get("policy") or "") != policy:
        return False
    if cur_image_id is None or str(handle.get("image_id") or "") != cur_image_id:
        return False
    # Refresh grid context with the real grid so consumers scale coords correctly.
    real_grid_hw = prefetch_grid_context(model, image_grid_thw)
    placeholder_hw = handle.get("grid_hw")
    handle["grid_hw"] = real_grid_hw
    handle["grid_match"] = bool(
        real_grid_hw is not None and placeholder_hw is not None and real_grid_hw == placeholder_hw
    )
    return True


def prefetch_module_a_for_history_capture(model, cap: Dict[str, Any]) -> bool:
    """Overlap Module A detect for a captured history frame (next-step rectify)."""
    if not module_a_prefetch_async_enabled():
        return False
    if not isinstance(cap, dict):
        return False
    try:
        from lmms_eval.models.geopact_nap import nap_current_prune_active

        # NAP stage 2 is a prefix contraction: no A/B re-run at rectify, so a
        # history-capture detect job would be pure wasted CPU.
        if nap_current_prune_active(model):
            return False
    except Exception:
        pass
    pil_image = cap.get("pil_image")
    grid_row = cap.get("grid_row")
    if pil_image is None or grid_row is None:
        return False
    from lmms_eval.models.geopact_vl_hook import _get_detector, _policy_is_token_baseline

    inner = getattr(model, "model", model)
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if _policy_is_token_baseline(policy):
        return False
    detector = _get_detector(
        policy,
        str(getattr(inner, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
    )
    if not hasattr(grid_row, "dim"):
        return False
    image_grid_thw = grid_row.unsqueeze(0) if int(grid_row.dim()) == 1 else grid_row
    image_id = str(cap.get("img_path") or "") or None
    return start_module_a_prefetch(
        inner,
        pil_image,
        detector,
        policy,
        image_grid_thw,
        image_id=image_id,
        source="history_capture",
    )


def _predict_image_grid_thw(inner, pil_image):
    """Predict the processor's ``image_grid_thw`` for a raw PIL frame.

    Applies the same ``smart_resize`` the qwen_vl_utils pipeline runs, using
    the patch size stashed by the wrapper and the run's explicit min/max
    pixel bounds when set. W57: the previous-step grid is stale on
    episode-first frames (different resolution → prepare-time grid check
    fails → the early submit is cancelled and the detector joins the critical
    path for ~50ms). Predicting the grid from the image itself makes the
    episode-first runway as long as every other step's. A wrong prediction is
    caught by ``_reuse_early_prefetch`` (mismatch → cancel + fresh same-step
    submit), so it can only cost latency, never selection identity.
    """
    patch = int(getattr(inner, "_geopact_image_patch_size", 0) or 0)
    if patch <= 0 or pil_image is None:
        return None
    try:
        import importlib

        vp = importlib.import_module("qwen_vl_utils.vision_process")
        smart_resize = vp.smart_resize
        min_default = int(getattr(vp, "MIN_PIXELS", 4 * 28 * 28))
        max_default = int(getattr(vp, "MAX_PIXELS", 16384 * 28 * 28))
        w, h = pil_image.size
        from lmms_eval.models.geopact_vl_hook import _spatial_merge_size

        factor = patch * _spatial_merge_size()
        min_px = getattr(inner, "_geopact_image_min_pixels", None)
        max_px = getattr(inner, "_geopact_image_max_pixels", None)
        h_bar, w_bar = smart_resize(
            h,
            w,
            factor=factor,
            min_pixels=int(min_px) if min_px else min_default,
            max_pixels=int(max_px) if max_px else max_default,
        )
    except Exception:
        return None
    import torch

    return torch.tensor([[1, h_bar // patch, w_bar // patch]], dtype=torch.long)


def early_start_module_a_prefetch(model) -> bool:
    """Submit the current frame's detect at PIL-cache time (suffix step head).

    The same-step submit in ``prepare_dialogue_pruned_embed_inputs`` only
    overlaps the ViT (~45ms), leaving ~50ms of detector join inside the
    selector bucket (W-L). Submitting here adds the chat-template render +
    tokenize + preprocess runway (~60-100ms) in front of the ViT. The grid is
    predicted from the frame itself via smart_resize (W57), falling back to
    the previous step's ``_geopact_last_image_grid_thw``; the consume path
    re-validates against the real grid and falls back to a fresh same-step
    submit on any mismatch, so the MIUs consumed are always computed on the
    true grid.
    """
    if not module_a_prefetch_async_enabled():
        return False
    from lmms_eval.models.geopact_qd_selector import bam_skip_module_a
    from lmms_eval.models.geopact_vl_hook import _get_detector, _policy_is_token_baseline

    inner = getattr(model, "model", model)
    if bam_skip_module_a():
        return False
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if _policy_is_token_baseline(policy):
        return False
    pil_image = getattr(inner, "_geopact_last_image", None)
    if pil_image is None:
        return False
    cur_image_id = _current_image_id(inner)
    if cur_image_id is None:
        return False
    handle = getattr(inner, "_geopact_module_a_async", None)
    if isinstance(handle, dict) and str(handle.get("image_id") or "") == cur_image_id:
        return True
    grid_thw = _predict_image_grid_thw(inner, pil_image)
    if grid_thw is None:
        grid_thw = getattr(inner, "_geopact_last_image_grid_thw", None)
    if grid_thw is None:
        return False
    grid_hw = prefetch_grid_context(inner, grid_thw)
    if grid_hw is None:
        return False
    detector = _get_detector(
        policy,
        str(getattr(inner, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
    )
    cancel_module_a_prefetch(inner)
    inner._geopact_module_a_async = {
        "future": _executor().submit(_detect_job, pil_image, detector, policy, grid_hw),
        "started_at": time.perf_counter(),
        "vit_done_at": None,
        "detector": detector,
        "policy": policy,
        "image_id": cur_image_id,
        "grid_hw": grid_hw,
        "source": "early",
    }
    return True


def _reuse_early_prefetch(
    model,
    image_grid_thw,
    detector: str,
    policy: str,
    cur_image_id: Optional[str],
) -> bool:
    """Keep an early (PIL-cache time) submit iff its guessed grid was right."""
    handle = getattr(model, "_geopact_module_a_async", None)
    if not isinstance(handle, dict) or str(handle.get("source") or "") != "early":
        return False
    real_grid = prefetch_grid_context(model, image_grid_thw)
    if (
        str(handle.get("detector") or "") == detector
        and str(handle.get("policy") or "") == policy
        and cur_image_id is not None
        and str(handle.get("image_id") or "") == str(cur_image_id)
        and handle.get("grid_hw") is not None
        and handle.get("grid_hw") == real_grid
    ):
        handle["grid_match"] = True
        return True
    cancel_module_a_prefetch(model)
    return False


def maybe_start_module_a_prefetch(model, image_grid_thw) -> bool:
    if not module_a_prefetch_async_enabled():
        return False
    from lmms_eval.models.geopact_qd_selector import bam_skip_module_a
    from lmms_eval.models.geopact_vl_hook import _get_detector, _policy_is_token_baseline

    if bam_skip_module_a():
        # no-mass BAM never reads the prior: don't burn a detector worker
        return False
    policy = str(getattr(model, "_geopact_prune_policy", "") or "")
    if _policy_is_token_baseline(policy):
        return False
    pil_image = getattr(model, "_geopact_last_image", None)
    if pil_image is None:
        return False
    detector = _get_detector(
        policy,
        str(getattr(model, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
    )
    cur_image_id = _current_image_id(model)
    if _reuse_early_prefetch(model, image_grid_thw, detector, policy, cur_image_id):
        return True
    if _reuse_cross_step_prefetch(model, image_grid_thw, detector, policy, cur_image_id):
        return True
    return start_module_a_prefetch(
        model, pil_image, detector, policy, image_grid_thw, image_id=cur_image_id
    )


def _pil_and_path_from_doc_image(
    doc: Dict[str, Any],
    lmms_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    from PIL import Image

    from lmms_eval.models.simple.qwen_vl_omnigui_common import (
        extract_episode_image_paths_from_doc,
    )

    paths = extract_episode_image_paths_from_doc(doc, lmms_kwargs or {})
    if not paths:
        return None, None
    path = paths[-1]
    try:
        return Image.open(path).convert("RGB"), str(path)
    except Exception:
        return None, None


def _pil_from_doc_image(
    doc: Dict[str, Any],
    lmms_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    pil_image, _ = _pil_and_path_from_doc_image(doc, lmms_kwargs)
    return pil_image


def schedule_next_step_module_a_prefetch(
    model,
    next_doc: Dict[str, Any],
    session_kwargs: Dict[str, Any],
) -> bool:
    """Start Module A detect for the next episode step during current-step tail work."""
    if os.environ.get("GEOPACT_MODULE_A_CROSS_STEP_PREFETCH", "0").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    if not module_a_prefetch_async_enabled():
        return False
    from lmms_eval.models.geopact_vl_hook import _get_detector, _policy_is_token_baseline

    inner = getattr(model, "model", model)
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if _policy_is_token_baseline(policy):
        return False
    pil_image, next_image_path = _pil_and_path_from_doc_image(next_doc, session_kwargs)
    if pil_image is None:
        return False
    hist = next_doc.get("action_history") or next_doc.get("history") or []
    if not hist and int(next_doc.get("step_index", next_doc.get("step_id", 0)) or 0) > 0:
        cancel_module_a_prefetch(inner)
        return False
    detector = _get_detector(
        policy,
        str(getattr(inner, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
    )
    import torch

    w, h = pil_image.size
    merge = 2
    grid_h = max(1, h // 28 // merge)
    grid_w = max(1, w // 28 // merge)
    placeholder_grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return start_module_a_prefetch(
        inner,
        pil_image,
        detector,
        policy,
        placeholder_grid,
        image_id=next_image_path,
        source="cross_step",
    )
