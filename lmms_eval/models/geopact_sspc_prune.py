"""SSPC Module A supplement: spatial-skeleton anchors within or beyond Module B budget."""
from __future__ import annotations

import os
import random
from typing import Optional, Set

from lmms_eval.models.geopact_sspc import resolve_frame_grid, sspc_axis_mode, sspc_k_grid


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def sspc_skeleton_enabled() -> bool:
    return _env_flag("GEOPACT_SSPC_SKELETON", "0")


def sspc_random_extra_enabled() -> bool:
    return _env_flag("GEOPACT_SSPC_RANDOM_EXTRA", "0")


def sspc_random_extra_k() -> int:
    try:
        k = int(os.environ.get("GEOPACT_SSPC_RANDOM_EXTRA_K", "0") or "0")
    except ValueError:
        k = 0
    if k > 0:
        return k
    return skeleton_anchor_count()


def sspc_budget_mode() -> str:
    return os.environ.get("GEOPACT_SSPC_BUDGET", "in_budget").strip().lower()


def sspc_in_budget() -> bool:
    return sspc_budget_mode() not in ("additive", "extra", "v2")


def skeleton_anchor_count(k_grid: Optional[int] = None) -> int:
    k = sspc_k_grid() if k_grid is None else max(1, int(k_grid))
    return k * k


def _axis_centers(span: int, k_grid: int, axis_mode: str) -> list[int]:
    if span <= 0:
        return [0]
    if k_grid <= 1:
        return [span // 2]
    mode = (axis_mode or "decile").strip().lower()
    if mode == "linspace" or k_grid > 3:
        fracs = [0.05 + (0.9 * i / max(k_grid - 1, 1)) for i in range(k_grid)]
    else:
        fracs = [0.1, 0.5, 0.9][:k_grid]
    return [
        max(0, min(span - 1, int(round((span - 1) * t))))
        for t in fracs
    ]


def skeleton_flat_local_indices(
    grid_h: int,
    grid_w: int,
    n_frame: int,
    *,
    k_grid: Optional[int] = None,
    axis_mode: Optional[str] = None,
) -> Set[int]:
    """k_grid^2 skeleton token indices in frame-local flat space (full-frame grid)."""
    if n_frame <= 0:
        return set()
    gh, gw, _ = resolve_frame_grid(grid_h, grid_w, n_frame)
    k = sspc_k_grid() if k_grid is None else max(1, int(k_grid))
    axis = sspc_axis_mode() if axis_mode is None else axis_mode
    rows = _axis_centers(gh, k, axis)
    cols = _axis_centers(gw, k, axis)
    out: Set[int] = set()
    for r in rows:
        for c in cols:
            flat = int(r) * gw + int(c)
            if 0 <= flat < n_frame:
                out.add(flat)
    return out


def random_extra_local_indices(
    n_frame: int,
    k: int,
    *,
    exclude: Set[int],
    seed: Optional[int] = None,
) -> Set[int]:
    """Random extra keep tokens (ablation: matched count vs skeleton)."""
    k = max(0, int(k))
    cands = [i for i in range(int(n_frame)) if i not in exclude]
    if not cands or k <= 0:
        return set()
    rng = random.Random(seed)
    rng.shuffle(cands)
    return set(cands[: min(k, len(cands))])


def compute_sspc_reserved_local(
    *,
    grid_h: int,
    grid_w: int,
    n_frame: int,
    seed: Optional[int] = None,
) -> tuple[Set[int], str]:
    """Reserved skeleton/random slots (frame-local indices)."""
    if sspc_random_extra_enabled():
        k = sspc_random_extra_k()
        if sspc_in_budget():
            return random_extra_local_indices(n_frame, k, exclude=set(), seed=seed), "random_extra"
        return set(), "random_extra_additive"
    if sspc_skeleton_enabled():
        return skeleton_flat_local_indices(grid_h, grid_w, n_frame), "skeleton"
    return set(), "none"


def plan_sspc_keep_budget(
    keep_k_total: int,
    *,
    grid_h: int,
    grid_w: int,
    n_frame: int,
    seed: Optional[int] = None,
) -> tuple[Set[int], int, dict]:
    """In-budget: reserve skeleton/random first; B fills the remainder."""
    keep_k_total = max(1, int(keep_k_total))
    k_grid = sspc_k_grid()
    gh, gw, grid_matched = resolve_frame_grid(grid_h, grid_w, n_frame)
    mode = sspc_budget_mode()
    if not sspc_in_budget():
        return set(), keep_k_total, {
            "sspc_budget_mode": mode,
            "sspc_keep_k_total": keep_k_total,
            "sspc_keep_k_b": keep_k_total,
            "sspc_reserved_n": 0,
            "sspc_k_grid": k_grid,
            "sspc_anchor_n": skeleton_anchor_count(k_grid),
        }
    reserved, extra_mode = compute_sspc_reserved_local(
        grid_h=grid_h, grid_w=grid_w, n_frame=n_frame, seed=seed,
    )
    reserved_n_raw = len(reserved)
    reserved_truncated = reserved_n_raw >= keep_k_total
    reserved_n = reserved_n_raw
    if reserved_truncated:
        reserved = set(sorted(reserved)[:keep_k_total])
        reserved_n = len(reserved)
    keep_k_b = max(0, keep_k_total - reserved_n)
    return reserved, keep_k_b, {
        "sspc_budget_mode": "in_budget",
        "sspc_extra_mode": extra_mode,
        "sspc_keep_k_total": keep_k_total,
        "sspc_keep_k_b": keep_k_b,
        "sspc_reserved_n": reserved_n,
        "sspc_reserved_truncated": reserved_truncated,
        "sspc_k_grid": k_grid,
        "sspc_anchor_n": skeleton_anchor_count(k_grid),
        "sspc_grid_h": gh,
        "sspc_grid_w": gw,
        "sspc_grid_matched": grid_matched,
    }


def apply_sspc_extra_keep(
    keep_local: Set[int],
    *,
    grid_h: int,
    grid_w: int,
    n_frame: int,
    seed: Optional[int] = None,
    keep_k_total: Optional[int] = None,
) -> tuple[Set[int], dict]:
    """Additive union (v2) or finalize audit for in-budget (already merged in hook)."""
    keep_b = set(keep_local)
    k_grid = sspc_k_grid()
    gh, gw, grid_matched = resolve_frame_grid(grid_h, grid_w, n_frame)
    if sspc_in_budget():
        keep_final = keep_b
        k_total = int(keep_k_total or len(keep_final))
        audit = {
            "sspc_budget_mode": "in_budget",
            "sspc_keep_b_n": len(keep_b),
            "sspc_extra_n": 0,
            "sspc_keep_final_n": len(keep_final),
            "sspc_keep_k_total": k_total,
            "sspc_k_grid": k_grid,
            "sspc_anchor_n": skeleton_anchor_count(k_grid),
            "sspc_grid_h": gh,
            "sspc_grid_w": gw,
            "sspc_grid_matched": grid_matched,
            "budget_overage": max(0, len(keep_final) - k_total),
        }
        return keep_final, audit
    extra: Set[int] = set()
    mode = "none"
    if sspc_random_extra_enabled():
        extra = random_extra_local_indices(
            n_frame,
            sspc_random_extra_k(),
            exclude=keep_b,
            seed=seed,
        )
        mode = "random_extra"
    elif sspc_skeleton_enabled():
        extra = skeleton_flat_local_indices(grid_h, grid_w, n_frame, k_grid=k_grid)
        mode = "skeleton"
    keep_final = keep_b | extra
    k_total = int(keep_k_total or 0)
    audit = {
        "sspc_budget_mode": "additive",
        "sspc_extra_mode": mode,
        "sspc_keep_b_n": len(keep_b),
        "sspc_extra_n": len(extra - keep_b),
        "sspc_keep_final_n": len(keep_final),
        "sspc_keep_k_total": k_total,
        "sspc_k_grid": k_grid,
        "sspc_anchor_n": skeleton_anchor_count(k_grid),
        "sspc_skeleton_n": len(extra) if mode == "skeleton" else 0,
        "sspc_grid_h": gh,
        "sspc_grid_w": gw,
        "sspc_grid_matched": grid_matched,
        "budget_overage": max(0, len(keep_final) - k_total) if k_total else 0,
    }
    return keep_final, audit
