"""GeoPACT Module C v5 — CRS (Certified Regional Skeletons), single-step
operator.

Motivation: pruning's dominant
damage on GUI grounding is layout-scaffold destruction (SSv2 r10 mass on/off
= 56.4 vs 36.5; 4B herding -0.88 vs uniform). CRS restores the *deleted*
set's spatial structure with reconstruction operators that avoid every
falsified Module C path:

  * no synthetic embedding aggregates (GAEC v1 = 11.67, entry merge SSv2
    48 vs 57) -> skeleton anchors are REAL tokens;
  * no host value overwrite (C-Absorb full-test ~0) -> selection-time only;
  * certificate built on measured feature geometry, not grid distance
    (post-MRoPE key mismatch 0.64 falsified spatial~key proximity).

Single-step operator (this file): after BAM picks `order` (budget k), keep
the lexicographic head (1-rho)k and re-spend the tail budget g = ceil(rho*k)
on regional medoids of the *complement* set: the complement is partitioned
into g contiguous raster blocks by the minimax-SSE DP (`coc_partition_dp`,
brute-force-verified optimal), and each block contributes its medoid (the
real token closest to the block's feature mean). Every certified region of
the deleted set then keeps one in-distribution representative — the
diversity term's soft coverage upgraded to a hard per-region guarantee,
in-budget (bit-identical sequence length).

Multi-step operator (CRS-N, `crs_nested_order`): the multi-step failure mode
is not the current step (the single-step operator already covers it) but the
NAP stage-2 contraction — the history prefix of the entry ordering. Ghost-KV
(value-level residue appended at contraction time) measured flat on OmniGUI
(-0.36 ~ +0.08 GP) while the single-step index-level operator is +2.2 ~ +2.5
★ on SSv2: what needs repairing is the ORDER, not the values. CRS-N therefore
injects the certificate once, at the entry selection, into the ordering
itself: the first k_hist positions are rebuilt as head (k_hist - g top-ranked
picks) + g certified regional medoids of everything else, so that the pure
prefix cut a frame suffers when it enters history keeps one in-distribution
representative per certified region — anytime-valid along the nesting chain,
zero extra tokens, zero cache surgery, zero later-step cost. Single-step is
the k_hist >= k_cur degenerate case (operator no-ops; the tail reserve above
already certifies the terminal budget).

Flags (arms, not knobs):
  GEOPACT_CRS=1            single-step master switch (off -> bit-exact);
  GEOPACT_CRS_NESTED=1     multi-step nested-certificate ordering (entry
                           injection; forces raw-prefix contraction via the
                           order record so the certificate survives verbatim);
  GEOPACT_CRS_SKEL_FRAC    skeleton budget share rho (default 0.10, shared
                           by both levels);
  GEOPACT_CRS_DIM          feature dim stride-subsample target (default 256;
                           keeps the DP prefix-sum cost O(n*256) on CPU);
  GEOPACT_CRS_REP          design ablations for the representative choice:
                           medoid (default) | mid (arbitrary member of each
                           certified region — is the medoid the active
                           ingredient, or any regional token?) | stride
                           (uniform raster coverage of the complement, no
                           feature geometry — the naive-coverage null).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "crs_enabled",
    "crs_nested_enabled",
    "crs_skel_frac",
    "crs_nested_frac",
    "crs_feature_dim",
    "crs_rep_mode",
    "crs_nested_rep_mode",
    "crs_partition_complement",
    "crs_skeleton_reserve",
    "crs_nested_order",
]


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _env_f(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def crs_enabled() -> bool:
    """GEOPACT_CRS=1: certified regional skeleton reserve (single-step C)."""
    return _flag("GEOPACT_CRS")


def crs_nested_enabled() -> bool:
    """GEOPACT_CRS_NESTED=1: nested-certificate entry ordering (multi-step C)."""
    return _flag("GEOPACT_CRS_NESTED")


def crs_skel_frac() -> float:
    """GEOPACT_CRS_SKEL_FRAC: share of keep budget re-spent on skeleton
    medoids (default 0.10). Production arms use (0, 0.5]; the W38 dose-curve
    sweep (E2) extends to 1.0 = the entire budget spent on coverage witnesses
    (head empty, pure-certificate limit), so the clamp admits [0, 1]."""
    v = _env_f("GEOPACT_CRS_SKEL_FRAC", 0.10)
    return min(1.0, max(0.0, v))


def crs_feature_dim() -> int:
    """GEOPACT_CRS_DIM: stride-subsampled feature width for the partition DP
    (default 256). The DP runs on CPU prefix sums; full 3.5k-dim embeddings
    would cost tens of ms per frame for no certificate benefit."""
    raw = os.environ.get("GEOPACT_CRS_DIM", "").strip()
    try:
        return max(16, int(raw)) if raw else 256
    except ValueError:
        return 256


_DP_THREADS_CLAMPED = False


def _clamp_dp_threads_once() -> None:
    """Pin torch CPU threads to 1 for the DP, once per process (W57).

    The eval loop's only multi-ms CPU-torch consumer is this partition DP;
    everything else hot runs on GPU or outside torch (tokenizers-rust, PIL).
    Leaving the pool at 1 thread avoids the per-frame OMP pool respawn that
    the old set/restore pattern paid twice per frame."""
    global _DP_THREADS_CLAMPED
    if _DP_THREADS_CLAMPED:
        return
    torch.set_num_threads(1)
    _DP_THREADS_CLAMPED = True


def _skeleton_features(e_norm: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """[len(rows), d'] fp32 features of the complement rows: deterministic
    stride slice of the normalized embedding dims (re-normalized so the DP's
    SSE stays a dispersion measure on the unit sphere)."""
    f = e_norm.index_select(0, rows.to(device=e_norm.device, dtype=torch.long))
    d = f.shape[1]
    target = crs_feature_dim()
    if d > target:
        stride = max(1, d // target)
        f = f[:, ::stride]
    return F.normalize(f.float(), dim=-1)


def crs_rep_mode() -> str:
    """GEOPACT_CRS_REP: representative-choice design ablation (single-step
    level). medoid (default) | mid (regional but arbitrary) | stride (no
    geometry). Production single-step operator = stride (SSv2/SSPro
    verdicts: spatial-uniform coverage of the deleted set is the active
    ingredient, +9.3/+2.6 star over medoid).

    Extra controls: ``stride_full`` anchors the g witnesses on the
    FULL frame raster instead of the complement quantiles (isolates the
    "complementary to the kept set" mechanism from "uniform anywhere");
    ``hilbert`` spaces the witnesses evenly along a Hilbert curve of the
    complement (locality-preserving space-filling order — if it matches
    raster stride, the active ingredient is spatial witnessing itself, not
    the raster parameterization)."""
    v = os.environ.get("GEOPACT_CRS_REP", "medoid").strip().lower()
    return v if v in ("medoid", "mid", "stride", "stride_full", "hilbert") else "medoid"


def crs_nested_rep_mode() -> str:
    """GEOPACT_CRS_NESTED_REP: representative choice for the multi-step
    (CRS-N) level, decoupled from the single-step flag. Default medoid —
    stride does NOT transfer to the nesting level (OmniGUI -1.05 vs medoid):
    the history certificate protects feature-distribution representativeness,
    not current-frame layout coverage. The two levels use their own geometry
    by design."""
    v = os.environ.get("GEOPACT_CRS_NESTED_REP", "medoid").strip().lower()
    return v if v in ("medoid", "mid", "stride") else "medoid"


def crs_nested_frac() -> float:
    """GEOPACT_CRS_NESTED_FRAC: certificate share for the CRS-N level.
    Falls back to GEOPACT_CRS_SKEL_FRAC (backward compatible with all
    2026-07-16/17 nested arms). Production = 0.05: the multi-step dose curve
    is monotone toward light (0.05 +1.92 star / 0.10 +1.19 / 0.20 +0.30) —
    certificate size scales with the deleted set, not the kept budget."""
    raw = os.environ.get("GEOPACT_CRS_NESTED_FRAC", "").strip()
    if not raw:
        return crs_skel_frac()
    try:
        return min(0.5, max(0.0, float(raw)))
    except ValueError:
        return crs_skel_frac()


def _hilbert_d(x: int, y: int, side: int) -> int:
    """Hilbert-curve distance of cell (x, y) on a side×side grid (side = 2^k).

    Standard iterative rot/reflect walk (Wikipedia xy2d); used only to ORDER
    complement cells, so non-square grids are embedded in the enclosing
    power-of-two square — ordering stays locality-preserving."""
    rx = ry = 0
    d = 0
    s = side // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def _hilbert_sorted_rows(
    complement_rows: torch.Tensor,
    frame_flat: torch.Tensor,
    grid_w: int,
) -> torch.Tensor:
    """Complement rows re-ordered along the Hilbert curve of the token grid."""
    flat = [int(frame_flat[i]) for i in range(int(complement_rows.numel()))]
    max_rc = max((max(f // grid_w, f % grid_w) for f in flat), default=0)
    side = 1
    while side <= max_rc:
        side *= 2
    keys = [_hilbert_d(f % grid_w, f // grid_w, side) for f in flat]
    perm = sorted(range(len(keys)), key=lambda i: keys[i])
    return complement_rows[torch.tensor(perm, dtype=torch.long, device=complement_rows.device)]


def crs_partition_complement(
    e_norm: torch.Tensor,
    complement_rows: torch.Tensor,
    g: int,
    *,
    rep: Optional[str] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Partition the raster-ordered complement into g certified blocks and
    return the per-block medoid rows (frame-local indices) + DP audit.

    Ablation nulls (GEOPACT_CRS_REP): `mid` keeps the certified regions but
    takes each region's middle member instead of its medoid (is the
    in-distribution *centre* the active ingredient, or any regional token?);
    `stride` skips the partition entirely and spends the budget on g
    evenly-spaced complement rows (naive uniform coverage — the "your
    skeleton is just subsampling" null). ``rep`` overrides the flag (the
    CRS-N caller passes its own level's mode)."""
    from lmms_eval.models.geopact_c_absorb import coc_partition_dp, coc_pick_hosts

    if rep is None:
        rep = crs_rep_mode()
    n_comp = int(complement_rows.numel())
    if rep == "stride":
        pos = torch.linspace(0, n_comp - 1, g).round().to(torch.long)
        pos = torch.unique(pos)
        reps = [int(complement_rows[int(p)]) for p in pos]
        return reps, {"crs_rep": "stride"}

    feats = _skeleton_features(e_norm, complement_rows)
    blocks, audit = coc_partition_dp(feats, g)
    if rep == "mid":
        host_rows = sorted(start + size // 2 for start, size in blocks if size > 0)
        audit["crs_rep"] = "mid"
    else:
        host_rows, _ = coc_pick_hosts(feats, blocks)
    medoids = [int(complement_rows[r]) for r in host_rows]
    return medoids, audit


def crs_skeleton_reserve(
    image_embeds: torch.Tensor,
    order: List[int],
    keep_k: int,
    *,
    flat_indices: Optional[torch.Tensor] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Re-spend the tail of a BAM pick order on certified regional medoids.

    order: local pick order (length <= keep_k) from bam_select_tokens.
    Returns (new order, audit). No-op (same list object) when disabled,
    degenerate, or the complement is empty.

    Raster order: local rows are raster-ordered within the frame; when the
    rows are a candidate subset, `flat_indices` restores the frame-raster
    order for the contiguity-constrained DP.
    """
    audit: Dict[str, Any] = {}
    if not crs_enabled():
        return order, audit
    n = int(image_embeds.shape[0])
    k = max(1, min(int(keep_k), n))
    rho = crs_skel_frac()
    g = int(math.ceil(rho * k)) if rho > 0.0 else 0
    if g < 1 or len(order) < k or n <= k:
        return order, audit

    head = [int(i) for i in order[: k - g]]
    head_set = set(head)
    device = image_embeds.device
    mask = torch.ones(n, dtype=torch.bool, device=device)
    if head_set:
        mask[torch.tensor(head, dtype=torch.long, device=device)] = False
    comp = torch.nonzero(mask, as_tuple=False).view(-1)
    if flat_indices is not None:
        flat = flat_indices.detach().to(device=device, dtype=torch.long)
        comp = comp[torch.argsort(flat.index_select(0, comp))]
    if int(comp.numel()) < g:
        return order, audit

    rep = crs_rep_mode()
    if rep == "stride_full":
        # E2 control: witnesses anchored on the FULL frame raster (not the
        # complement quantiles); a full-screen anchor landing on a kept head
        # token slides to the nearest complement row so the budget stays
        # exact — but the anchoring no longer adapts to what was kept.
        comp_list = [int(c) for c in comp.tolist()]
        comp_set = set(comp_list)
        n_pos = n
        picked: List[int] = []
        picked_set: set = set()
        for gi in range(g):
            anchor = int(round(gi * (n_pos - 1) / max(1, g - 1))) if g > 1 else n_pos // 2
            best_row = None
            for off in range(n_pos):
                for cand_row in (anchor - off, anchor + off):
                    if 0 <= cand_row < n_pos and cand_row in comp_set and cand_row not in picked_set:
                        best_row = cand_row
                        break
                if best_row is not None:
                    break
            if best_row is None:
                audit["crs_skipped"] = "stride_full_exhausted"
                return order, audit
            picked.append(best_row)
            picked_set.add(best_row)
        medoids, dp_audit = picked, {"crs_rep": "stride_full"}
    elif rep == "hilbert":
        # M6 control: evenly spaced witnesses along the Hilbert curve of the
        # complement (locality-preserving order; needs the frame grid).
        if grid_hw is None or int(grid_hw[1]) < 1:
            audit["crs_skipped"] = "hilbert_no_grid"
            return order, audit
        frame_flat = (
            flat_indices.detach().to(device=device, dtype=torch.long).index_select(0, comp)
            if flat_indices is not None
            else comp
        )
        comp_h = _hilbert_sorted_rows(comp, frame_flat, int(grid_hw[1]))
        pos = torch.linspace(0, int(comp_h.numel()) - 1, g).round().to(torch.long)
        pos = torch.unique(pos)
        medoids = [int(comp_h[int(p)]) for p in pos]
        dp_audit = {"crs_rep": "hilbert"}
    else:
        e_norm = F.normalize(image_embeds.detach().to(torch.float32), dim=-1)
        medoids, dp_audit = crs_partition_complement(e_norm, comp, g)
    if len(medoids) != g or any(mi in head_set for mi in medoids):
        # medoids come from the complement by construction; any violation
        # means the partition degenerated — keep the untouched order.
        audit["crs_skipped"] = "degenerate_partition"
        return order, audit

    new_order = head + medoids
    audit.update(
        {
            "crs_g": g,
            "crs_skel_frac": round(rho, 4),
            "crs_tail_replaced": int(len(order) - len(head)),
            "crs_sse_max": dp_audit.get("coc_sse_max"),
            "crs_sse_mean": dp_audit.get("coc_sse_mean"),
        }
    )
    return new_order, audit


def crs_nested_order(
    frame_embeds: torch.Tensor,
    order: List[int],
    *,
    n_full: int,
    k_hist: int,
) -> Tuple[List[int], bool, Dict[str, Any]]:
    """CRS-N: rebuild the entry ordering so its k_hist prefix is certified.

    frame_embeds: [n_full, h] full-frame vision rows (frame-local raster
    order); order: entry pick sequence over full-frame flat indices (length
    k_cur, best rank first) — the object NAP stage 2 prefix-cuts when the
    frame enters history.

    The first (k_hist - g) positions keep B's top-ranked picks verbatim; the
    next g positions become the medoids of the g certified regions of the
    complement (everything the history cut would delete). Medoids not already
    in S_cur are swapped in, evicting the lowest-ranked non-skeleton members
    (|S_cur| is preserved — the current step's budget is untouched). The rest
    of S_cur follows in B rank order, so every intermediate budget still reads
    as "certified skeleton + best-ranked fill".

    Returns (new_order, keep_changed, audit); keep_changed signals membership
    swaps so the caller refreshes its keep mask.
    """
    audit: Dict[str, Any] = {}
    if not crs_nested_enabled():
        return order, False, audit
    _t0 = time.perf_counter()
    n = int(n_full)
    k_cur = len(order)
    kh = int(k_hist)
    rho = crs_nested_frac()
    g = int(math.ceil(rho * kh)) if rho > 0.0 else 0
    # Degenerate: no contraction ahead (single-step case) or budget too tight.
    if g < 1 or kh >= k_cur or kh - g < 0 or n <= kh:
        audit["crs_nested_skipped"] = "degenerate_budget"
        return order, False, audit

    head = [int(i) for i in order[: kh - g]]
    head_set = set(head)
    device = frame_embeds.device
    mask = torch.ones(n, dtype=torch.bool, device=device)
    if head_set:
        mask[torch.tensor(head, dtype=torch.long, device=device)] = False
    comp = torch.nonzero(mask, as_tuple=False).view(-1)
    if int(comp.numel()) < g:
        audit["crs_nested_skipped"] = "small_complement"
        return order, False, audit

    e_norm = F.normalize(frame_embeds.detach().to(torch.float32), dim=-1)
    audit["crs_nested_prep_ms"] = round((time.perf_counter() - _t0) * 1000.0, 2)
    _t1 = time.perf_counter()
    # Shared-pod guard: the partition DP runs tiny CPU tensor ops; default
    # 40-thread pools oversubscribe and cost ~50ms (ghost-plan diagnosis).
    # W57: clamp ONCE per process instead of toggling around every call —
    # each set_num_threads bounce respawns the OMP pool and cost a constant
    # ~16-18ms/frame (bench: toggle 20.2/30.9ms vs clamp-once 2.3/12.6ms on
    # mobile/desktop frames). The DP always executes at 1 thread either way,
    # so the math (and therefore the certificate) is unchanged.
    _clamp_dp_threads_once()
    medoids, dp_audit = crs_partition_complement(
        e_norm, comp, g, rep=crs_nested_rep_mode()
    )
    audit["crs_nested_dp_ms"] = round((time.perf_counter() - _t1) * 1000.0, 2)
    if len(medoids) != g or any(mi in head_set for mi in medoids):
        audit["crs_nested_skipped"] = "degenerate_partition"
        return order, False, audit

    medoid_set = set(medoids)
    order_set = {int(i) for i in order}
    swap_in = [m for m in medoids if m not in order_set]
    # Remaining S_cur members (below the certified prefix), best rank first;
    # evict from the tail to make room for swapped-in medoids.
    rest = [int(t) for t in order[kh - g :] if int(t) not in medoid_set]
    if len(swap_in) > len(rest):
        audit["crs_nested_skipped"] = "no_eviction_room"
        return order, False, audit
    if swap_in:
        rest = rest[: len(rest) - len(swap_in)]

    new_order = head + medoids + rest
    audit.update(
        {
            "crs_nested_g": g,
            "crs_nested_k_hist": kh,
            "crs_nested_promoted": g - len(swap_in),
            "crs_nested_swapped_in": len(swap_in),
            "crs_nested_sse_max": dp_audit.get("coc_sse_max"),
            "crs_nested_sse_mean": dp_audit.get("coc_sse_mean"),
            "crs_nested_total_ms": round((time.perf_counter() - _t0) * 1000.0, 2),
        }
    )
    return new_order, bool(swap_in), audit
