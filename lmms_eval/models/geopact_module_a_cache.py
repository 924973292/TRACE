"""Content-keyed LRU cache for Module A (detect → prefilter → pentagonal → density)."""
from __future__ import annotations

import copy
import hashlib
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


# Hard-disable Module A LRU / PIL short-circuit when env says cache off.
def module_a_cache_enabled() -> bool:
    val = os.environ.get("GEOPACT_MODULE_A_CACHE", "1").strip().lower()
    if val in {"0", "false", "off", "no"}:
        return False
    # Prefetch + subprocess path owns Module A; never LRU-cache on top.
    if os.environ.get("GEOPACT_MODULE_A_PREFETCH", "").strip().lower() in {
        "cpu",
        "gpu",
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return True


def _cache_max_entries() -> int:
    try:
        return max(1, int(os.environ.get("GEOPACT_MODULE_A_CACHE_SIZE", "256")))
    except ValueError:
        return 256


def pil_short_circuit_enabled() -> bool:
    val = os.environ.get("GEOPACT_MODULE_A_PIL_SHORT", "1").strip().lower()
    return val not in {"0", "false", "off", "no"}


def _pil_content_hash(pil: Image.Image) -> str:
    """Hash downscaled detect view (aligned with FAST_MIU_MAX_EDGE preprocessing)."""
    from lmms_eval.models.gui_miu_detectors import _fast_downscale_for_detect

    _, image_np, proc_w, proc_h = _fast_downscale_for_detect(pil)
    payload = image_np.tobytes() + f"{proc_w}x{proc_h}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _detector_version_key(detector: str) -> str:
    if detector in ("omniparser", "omniparser_blob", "omniparser_yolo_blob", "yolo_blob"):
        suffix = ""
        if detector != "omniparser":
            suffix = ":blob_aux"
        if os.environ.get("GEOPACT_YOLO_BLOB_AUX", "0").strip().lower() in {"1", "true", "yes", "on"}:
            suffix = ":blob_aux"
        return os.environ.get(
            "OMNIPARSER_ICON_WEIGHTS",
            "omniparser_default",
        ) + suffix
    return detector


def _policy_hash(policy: str) -> str:
    normalized = " ".join(str(policy or "").strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def _module_a_content_key(pil: Image.Image, model: Any = None) -> str:
    """Fast cache identity: episode image path when set, else downscaled PIL hash."""
    if model is not None:
        paths = getattr(model, "_geopact_image_paths", None) or []
        if paths:
            path = str(paths[-1] or "").strip()
            if path:
                return f"path:{path}"
    return f"pil:{_pil_content_hash(pil)}"


def module_a_cache_key(
    pil: Image.Image,
    detector: str,
    policy: str,
    *,
    grid_hw: Optional[Tuple[int, int]] = None,
    n_frame: Optional[int] = None,
    model: Any = None,
) -> str:
    grid_part = "none"
    if grid_hw is not None:
        grid_part = f"{int(grid_hw[0])}x{int(grid_hw[1])}"
    frame_part = "none" if n_frame is None else str(int(n_frame))
    return ":".join(
        [
            _module_a_content_key(pil, model),
            _detector_version_key(detector),
            _policy_hash(policy),
            grid_part,
            frame_part,
        ]
    )


def _get_cache_store(model: Any) -> OrderedDict[str, List[dict]]:
    store = getattr(model, "_geopact_module_a_cache", None)
    if store is None:
        store = OrderedDict()
        model._geopact_module_a_cache = store
    return store


def _touch_cache_stats(model: Any, *, hit: bool) -> None:
    stats = getattr(model, "_geopact_module_a_cache_stats", None)
    if not isinstance(stats, dict):
        stats = {"hits": 0, "misses": 0}
        model._geopact_module_a_cache_stats = stats
    if hit:
        stats["hits"] = int(stats.get("hits", 0)) + 1
    else:
        stats["misses"] = int(stats.get("misses", 0)) + 1


def get_or_run_module_a(
    pil: Image.Image,
    detector: str,
    policy: str,
    run_fn: Callable[[], List[dict]],
    *,
    model: Any = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return (mius, cache_meta). ``run_fn`` performs uncached Module A work."""
    meta: Dict[str, Any] = {"module_a_cache_hit": False, "module_a_cache_key": None}
    if not module_a_cache_enabled() or pil is None:
        return run_fn(), meta

    grid_hw = None
    n_frame = None
    if model is not None:
        grid_hw = getattr(model, "_geopact_module_a_grid_hw", None)
        compact = getattr(model, "_geopact_last_compact_stats", None) or {}
        if compact.get("n_visual_total") is not None:
            n_frame = int(compact.get("n_visual_total"))

    key = module_a_cache_key(pil, detector, policy, grid_hw=grid_hw, n_frame=n_frame, model=model)
    meta["module_a_cache_key"] = key
    run_key = ":".join(
        [
            _module_a_content_key(pil, model),
            _detector_version_key(detector),
            _policy_hash(policy),
            f"{grid_hw[0]}x{grid_hw[1]}" if grid_hw else "none",
            str(n_frame) if n_frame is not None else "none",
        ]
    )

    if model is not None:
        store = _get_cache_store(model)
        if key in store:
            store.move_to_end(key)
            _touch_cache_stats(model, hit=True)
            meta["module_a_cache_hit"] = True
            return copy.deepcopy(store[key]), meta

        if pil_short_circuit_enabled() and pil is not None:
            content_id = _module_a_content_key(pil, model)
            last_content = getattr(model, "_geopact_module_a_last_content_key", None)
            last_run = getattr(model, "_geopact_module_a_last_run_key", None)
            snapshot = getattr(model, "_geopact_module_a_last_snapshot", None)
            if content_id == last_content and run_key == last_run and snapshot is not None:
                meta["module_a_pil_short_circuit"] = True
                _touch_cache_stats(model, hit=True)
                return copy.deepcopy(snapshot), meta

    mius = run_fn()
    if model is not None:
        store = _get_cache_store(model)
        store[key] = copy.deepcopy(mius)
        store.move_to_end(key)
        while len(store) > _cache_max_entries():
            store.popitem(last=False)
        _touch_cache_stats(model, hit=False)
        if pil is not None and pil_short_circuit_enabled():
            model._geopact_module_a_last_content_key = _module_a_content_key(pil, model)
            model._geopact_module_a_last_run_key = run_key
            model._geopact_module_a_last_snapshot = copy.deepcopy(mius)

    return mius, meta
