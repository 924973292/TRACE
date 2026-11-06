"""GeoPACT Module C v5 — CRS Ghost-KV (multi-step operator).

When a frame's history
contraction (NAP stage 2) deletes cache rows, the deleted set D is
partitioned into G certified contiguous regions (minimax-SSE DP on the
measured value geometry, `coc_partition_dp` — brute-force-verified optimal)
and each region contributes one *appended* ghost KV entry per layer:

    k_hat_g = mean_{j in R_g} k_j,   v_hat_g = mean_{j in R_g} v_j

taken from the live pre-crop *contextualized* cache rows. One ghost repairs
both pruning distortions at first order (doc §1.2): the numerator regains the
region's value contribution and the partition function regains the region's
attention competition. Every falsified Module C path is avoided:

  * C1: no embedding-level aggregates — ghosts are KV rows, they never pass
    through the embedding layer or the forward;
  * C3: append-only — surviving rows' K/V/positions are bit-untouched;
  * C4: the certificate partition runs on measured value features, not grid
    distance; ghost keys are means of measured post-MRoPE keys (the linear
    term of the RoPE spread cancels at the mean; residual is second order in
    the region key diameter and is exactly what the DP minimises);
  * C5/C6: ghosts are ordinary cache rows — FA2-native, no attention bias,
    no per-layer decode-time compute.

Bookkeeping contract (geopact_session_kv.py):
  plan   at `sliding_rewrite_prepare_dialogue_kv` (pre-crop cache alive);
         ghost ids are inserted into pending["ids_after_rewrite"] right after
         the kept visual ids, so the capture boundary / finalize ids and the
         spliced cache stay 1:1;
  splice at `_prefill_merged_sliding_rewrite_before_generate` after the
         merged forward: physical insert at cache index past_len + n_kept.
         The suffix rows of the same step were computed without attending to
         the ghosts (one-step lag, doc §5.4) — all *future* steps attend.

Flags (arms, not knobs):
  GEOPACT_CRS_GHOST=1   master switch (off -> serving path bit-exact);
  GEOPACT_CRS_G         region count G (default 16, clamped to [8, 32]; the
                        lower clamp keeps the capture boundary unambiguous,
                        see doc §6.3);
  GEOPACT_CRS_MULTI     one|prop — ghosts per region (doc §2.5): `one` is a
                        single entry per region; `prop` splits
                        GEOPACT_CRS_TOTAL entries proportionally to region
                        mass (largest remainder, >=1 each);
  GEOPACT_CRS_TOTAL     total ghost rows for multi=prop (default 2*G).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "crs_ghost_enabled",
    "crs_ghost_g",
    "crs_ghost_multi",
    "crs_ghost_total",
    "plan_crs_ghosts",
    "apply_crs_ghosts",
]


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def crs_ghost_enabled() -> bool:
    """GEOPACT_CRS_GHOST=1: certified regional Ghost-KV (multi-step C)."""
    return _flag("GEOPACT_CRS_GHOST")


def crs_ghost_g() -> int:
    """GEOPACT_CRS_G: number of certified regions (default 16, in [8, 32])."""
    raw = os.environ.get("GEOPACT_CRS_G", "").strip()
    try:
        v = int(raw) if raw else 16
    except ValueError:
        v = 16
    return max(8, min(32, v))


def crs_ghost_multi() -> str:
    """GEOPACT_CRS_MULTI: 'one' (default) or 'prop' (doc §2.5)."""
    v = os.environ.get("GEOPACT_CRS_MULTI", "one").strip().lower()
    if v not in ("one", "prop"):
        raise ValueError(f"GEOPACT_CRS_MULTI must be one/prop, got {v!r}")
    return v


def crs_ghost_total() -> int:
    """GEOPACT_CRS_TOTAL: total ghost rows under multi=prop (default 2*G)."""
    raw = os.environ.get("GEOPACT_CRS_TOTAL", "").strip()
    try:
        return max(1, int(raw)) if raw else 2 * crs_ghost_g()
    except ValueError:
        return 2 * crs_ghost_g()


def _region_counts(sizes: Sequence[int], total: int) -> List[int]:
    """Largest-remainder allocation of `total` ghost rows proportional to
    region mass, each region guaranteed >= 1 (doc §2.5 piecewise-constant
    multiplicity approximation)."""
    g = len(sizes)
    total = max(g, int(total))
    mass = float(sum(sizes))
    raw = [total * s / mass for s in sizes]
    counts = [max(1, int(x)) for x in raw]
    rem = total - sum(counts)
    if rem > 0:
        order = sorted(range(g), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in range(rem):
            counts[order[i % g]] += 1
    elif rem < 0:
        order = sorted(range(g), key=lambda i: counts[i], reverse=True)
        i = 0
        while rem < 0:
            j = order[i % g]
            if counts[j] > 1:
                counts[j] -= 1
                rem += 1
            i += 1
    return counts


@torch.inference_mode()
def plan_crs_ghosts(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    vis_start: int,
    n_vis: int,
    local_keep: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Ghost K/V rows for one contracting frame, from the pre-crop cache.

    legacy_tensors: per-layer (keys, values), shapes [B, H_kv, S, d];
    vis_start:      absolute cache position of the frame segment;
    n_vis:          cached segment length (== len(S_cur) under NAP stage 2);
    local_keep:     kept rows (S_hist) in segment-local coordinates.

    Returns {"k_layers": [L x [T, H_kv, d]], "v_layers": ..., "g": T,
    "audit": {...}} in cache dtype, or None when degenerate. Regions are
    contiguous in the segment's raster order; features are the same
    scale-balanced value geometry COC uses (`coc_value_features`).
    """
    from lmms_eval.models.geopact_c_absorb import coc_partition_dp, coc_value_features

    t0 = time.perf_counter()
    if not legacy_tensors or n_vis <= 0:
        return None
    keep_set = {int(x) for x in local_keep}
    dropped = [r for r in range(int(n_vis)) if r not in keep_set]
    if not dropped:
        return None
    g = min(crs_ghost_g(), len(dropped))

    feats = coc_value_features(legacy_tensors, vis_start=vis_start, n_vis=n_vis)
    if feats is None:
        return None
    d_idx = torch.tensor(dropped, dtype=torch.long, device=feats.device)
    f_drop = feats.index_select(0, d_idx)
    # Stride-subsample the DP feature width (same convention as the
    # single-step operator's crs_feature_dim): the full 1k-dim prefix-sum DP
    # costs >100ms on CPU for no certificate benefit; 256 dims is <1ms.
    if f_drop.shape[1] > 256:
        # .contiguous() before the DP's device->CPU hop: a strided slice of a
        # CUDA tensor copies element-wise otherwise (>100ms wall, smoke G-0).
        f_drop = f_drop[:, :: max(1, f_drop.shape[1] // 256)].contiguous()
    # OMP oversubscription guard: the DP's CPU tensor ops are tiny ([~550,
    # 256] prefix sums) — with the default 40 intra-op threads on a shared
    # pod each op pays ~ms of spin-up (measured 118ms vs 2.3ms single-thread,
    # smoke G-0 wave).
    _nthreads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        blocks, dp_audit = coc_partition_dp(f_drop, g)
    finally:
        torch.set_num_threads(_nthreads)
    if len(blocks) != g:
        return None

    if crs_ghost_multi() == "prop":
        counts = _region_counts([size for _s, size in blocks], crs_ghost_total())
    else:
        counts = [1] * g

    # Per-region member rows as absolute cache positions; one flat gather per
    # layer, then segment means via index_add (regions are small, G<=32).
    member_abs: List[int] = []
    region_of: List[int] = []
    for gi, (start, size) in enumerate(blocks):
        for r in dropped[start : start + size]:
            member_abs.append(int(vis_start) + int(r))
            region_of.append(gi)
    device = legacy_tensors[0][0].device
    m_idx = torch.tensor(member_abs, dtype=torch.long, device=device)
    r_idx = torch.tensor(region_of, dtype=torch.long, device=device)
    sizes_t = torch.zeros(g, dtype=torch.float32, device=device)
    sizes_t.index_add_(0, r_idx, torch.ones_like(r_idx, dtype=torch.float32))

    rep_idx = torch.tensor(
        [gi for gi, c in enumerate(counts) for _ in range(c)],
        dtype=torch.long,
        device=device,
    )
    k_layers: List[torch.Tensor] = []
    v_layers: List[torch.Tensor] = []
    for k, v in legacy_tensors:
        out_pair = []
        for t in (k, v):
            rows = t[0].index_select(1, m_idx.to(t.device)).float()  # [H, M, d]
            acc = torch.zeros(
                g, rows.shape[0], rows.shape[2], dtype=torch.float32, device=rows.device
            )
            acc.index_add_(0, r_idx.to(rows.device), rows.permute(1, 0, 2).contiguous())
            mean = acc / sizes_t.to(rows.device).view(-1, 1, 1)  # [G, H, d]
            out_pair.append(
                mean.index_select(0, rep_idx.to(rows.device)).to(t.dtype)  # [T, H, d]
            )
        k_layers.append(out_pair[0])
        v_layers.append(out_pair[1])

    total = int(rep_idx.numel())
    audit = {
        "crs_ghost_g": g,
        "crs_ghost_rows": total,
        "crs_ghost_dropped": len(dropped),
        "crs_ghost_multi": crs_ghost_multi(),
        "crs_ghost_sse_max": dp_audit.get("coc_sse_max"),
        "crs_ghost_sse_mean": dp_audit.get("coc_sse_mean"),
        "crs_ghost_plan_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return {"k_layers": k_layers, "v_layers": v_layers, "g": total, "audit": audit}


@torch.inference_mode()
def apply_crs_ghosts(
    cache: Any,
    *,
    insert_at: int,
    k_layers: Sequence[torch.Tensor],
    v_layers: Sequence[torch.Tensor],
    config: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Return a NEW cache with ghost rows spliced at physical index
    `insert_at` in every layer. Surviving rows are bit-identical views/copies;
    the input cache must be exclusively owned by the caller (merged-prefill
    running_kv), mirroring the `_cache_from_legacy_tensors` convention."""
    from lmms_eval.models.geopact_session_kv import _legacy_cache_tensors

    t0 = time.perf_counter()
    legacy = _legacy_cache_tensors(cache)
    if not legacy or not k_layers or len(legacy) != len(k_layers):
        return cache, {"crs_ghost_applied": False}
    p = int(insert_at)
    seq = int(legacy[0][0].shape[-2])
    if p <= 0 or p > seq:
        return cache, {"crs_ghost_applied": False, "crs_ghost_reason": "bad_insert"}

    spliced = []
    for li, (k, v) in enumerate(legacy):
        gk = k_layers[li].to(device=k.device, dtype=k.dtype).permute(1, 0, 2).unsqueeze(0)
        gv = v_layers[li].to(device=v.device, dtype=v.dtype).permute(1, 0, 2).unsqueeze(0)
        nk = torch.cat([k[..., :p, :], gk, k[..., p:, :]], dim=-2)
        nv = torch.cat([v[..., :p, :], gv, v[..., p:, :]], dim=-2)
        spliced.append((nk, nv))

    try:
        from lmms_eval.models.geopact_block_kv_cache import make_kv_cache

        new_cache = make_kv_cache(config=config)
        for layer_idx, (nk, nv) in enumerate(spliced):
            new_cache.update(nk, nv, layer_idx)
    except Exception:
        new_cache = tuple(spliced)
    stats = {
        "crs_ghost_applied": True,
        "crs_ghost_apply_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return new_cache, stats
