"""Module A pipeline — detect → [optional OAS] → prefilter → pentagonal → info_density.

Optimization layers (no Module B/C token compression):
  A0  cheap: content-hash LRU cache (geopact_module_a_cache)
  A1  reorder: prefilter before pentagonal
  A2  detect: YOLO FP16 (GEOPACT_YOLO_HALF), downscale for classical CV
  A3  pentagonal: batch entropy/contrast, single LAB pass; E = conf × (S+H+C+G+R)
  A4  density: grid-aware token count when grid_hw provided
  A5  prefilter: geometry-only (area, aspect, cover, parent-of-many); OAS off by default
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def module_a_profile_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_PROFILE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def _module_a_sideload_mode() -> str:
    return os.environ.get("GEOPACT_MODULE_A_SIDELOAD_MODE", "full").strip().lower()


def _module_a_sideload_active() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_SIDELOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_module_a(
    pil_image: Image.Image,
    detector: str,
    policy: str,
    *,
    grid_hw: Optional[Tuple[int, int]] = None,
    detect_raw_fn,
    apply_attribute_drops_fn,
    policy_is_token_baseline_fn,
    warn_skip_abc_fn,
    fail_fn,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Execute Module A stages; return (mius, meta with timings + prefilter)."""
    from lmms_eval.models.gui_miu_detectors import (
        _attach_pentagonal_interaction_energy,
        filter_redundant_image_mius,
    )
    from lmms_eval.models.miu_info_density import attach_module_a_info_density_proxy
    from lmms_eval.models.oas_miu import attach_oas_attributes, parse_oas_drop_set

    meta: Dict[str, Any] = {"detector": detector, "policy": policy}
    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    try:
        raw_mius, image_np = detect_raw_fn()
    except Exception as exc:
        fail_fn(f"detector={detector} raised {type(exc).__name__}: {exc}", policy)
        return [], meta
    timings["detect_ms"] = _ms_since(t0)
    from lmms_eval.models.gui_miu_detectors import pop_last_detect_timings

    detect_sub = pop_last_detect_timings()
    if detect_sub:
        timings.update(detect_sub)
    meta["n_raw"] = len(raw_mius or [])

    if _module_a_sideload_active() and _module_a_sideload_mode() == "yolo_only":
        mius = [dict(m) for m in (raw_mius or [])]
        meta["n_kept"] = len(mius)
        meta["sideload_mode"] = "yolo_only"
        meta["timings_ms"] = timings
        meta["total_ms"] = _ms_since(t_all)
        return mius, meta

    if not raw_mius and not policy_is_token_baseline_fn(policy):
        warn_skip_abc_fn(f"detector={detector} returned 0 MIUs (blank/icon-free frame)")
        meta["timings_ms"] = timings
        meta["total_ms"] = _ms_since(t_all)
        return [], meta

    oas_enabled = os.environ.get("GEOPACT_MODULE_A_OAS", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    t0 = time.perf_counter()
    if oas_enabled:
        mius = attach_oas_attributes([dict(m) for m in raw_mius], drop=parse_oas_drop_set(policy))
    else:
        mius = [dict(m) for m in raw_mius]
    timings["oas_ms"] = _ms_since(t0)
    meta["oas_enabled"] = oas_enabled

    t0 = time.perf_counter()
    mius, prefilter_meta = filter_redundant_image_mius(mius)
    timings["prefilter_ms"] = _ms_since(t0)
    meta["n_kept"] = len(mius)
    meta["prefilter"] = prefilter_meta

    if mius and image_np is not None and not policy_is_token_baseline_fn(policy):
        t0 = time.perf_counter()
        mius = _attach_pentagonal_interaction_energy(mius, image_np)
        mius = apply_attribute_drops_fn(mius, policy)
        timings["pentagonal_ms"] = _ms_since(t0)

    t0 = time.perf_counter()
    attach_module_a_info_density_proxy(mius, policy, grid_hw=grid_hw)
    timings["density_ms"] = _ms_since(t0)

    timings["total_ms"] = _ms_since(t_all)
    meta["timings_ms"] = timings
    meta["grid_hw"] = list(grid_hw) if grid_hw else None
    return mius, meta
