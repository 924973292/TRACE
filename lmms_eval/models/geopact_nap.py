"""Nested Anticipative Pruning (NAP): two-stage nested masks from one ordering.

Paradigm (plan: 嵌套两阶段剪枝范式):
  Stage 1 (entry)  — the current screenshot is pruned once at budget
                     ``k_cur = round(n_full * GEOPACT_RATIO_CURRENT)`` before it
                     enters the LLM. The selector's pick sequence is stored.
  Stage 2 (slide)  — when the frame becomes history, no A/B re-run happens:
                     the stored ordering is contracted to its first
                     ``k_hist = round(n_full * PRUNE_RATIO)`` picks
                     (S_hist ⊂ S_cur by construction), and the KV rewrite
                     path shrinks the cached segment to that prefix.

BAM's matching-pursuit greedy makes any prefix of the pick sequence the greedy
solution at that budget, so the nested masks are free. Baselines nest too:
``random`` stores a deterministic permutation of S_cur (prefix = uniform random
subset of S_cur), ``uniform`` re-strides inside S_cur at stage 2.

Order records are keyed by the sticky mask key and live on the model next to
``_geopact_sticky_masks`` with the same episode lifecycle.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "two_stage_enabled",
    "ratio_current",
    "nap_two_stage_active",
    "nap_current_prune_active",
    "nap_order_kind",
    "nap_contract_mode",
    "nap_anticipate_enabled",
    "anticipate_nap_coverage",
    "build_nap_order",
    "contract_nap_order",
    "store_nap_order",
    "lookup_nap_order",
    "clear_nap_orders",
    "nap_full_token_count",
]


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def two_stage_enabled() -> bool:
    """GEOPACT_TWO_STAGE=1 turns on the nested two-stage protocol."""
    return os.environ.get("GEOPACT_TWO_STAGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def ratio_current() -> float:
    """Stage-1 keep ratio for the current screenshot (default 0.75).

    ``PRUNE_RATIO`` / ``visual_prune_ratio`` keeps its meaning: the stage-2
    history budget r_hist.
    """
    raw = os.environ.get("GEOPACT_RATIO_CURRENT", "").strip()
    if not raw:
        return 0.75
    return float(raw)


def nap_two_stage_active(model=None) -> bool:
    """Two-stage protocol applies: dialogue KV + history prune target + flag."""
    if not two_stage_enabled():
        return False
    from lmms_eval.models.geopact_session_kv import (
        dialogue_kv_enabled,
        resolve_prune_target,
    )

    if not dialogue_kv_enabled():
        return False
    return resolve_prune_target(model) == "history"


def nap_current_prune_active(model=None) -> bool:
    """Stage-1 entry pruning fires only for a real (0, 1) current ratio.

    ``GEOPACT_RATIO_CURRENT>=1.0`` must degrade bit-exactly to the current
    protocol (current frame enters full, stage 2 re-selects) — regression gate.
    """
    if not nap_two_stage_active(model):
        return False
    r = ratio_current()
    return 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# Order semantics
# ---------------------------------------------------------------------------


def nap_order_kind(policy: str) -> str:
    """Contraction semantics per selector family.

    greedy — matching-pursuit/herding pick sequence; prefix is the greedy
             solution at the smaller budget.
    random — deterministic permutation of S_cur; prefix is a uniform random
             subset of S_cur (the honest nested-random baseline).
    stride — raster order; stage 2 re-strides evenly inside S_cur (the honest
             nested-uniform baseline; a stride order has no meaningful prefix).
    """
    p = str(policy or "").lower()
    if "uniform" in p:
        return "stride"
    if "random" in p:
        return "random"
    return "greedy"


def build_nap_order(
    *,
    keep_local: Sequence[int],
    pick_order: Optional[Sequence[int]] = None,
    kind: str = "greedy",
    rng=None,
) -> List[int]:
    """Permutation of ``keep_local`` implementing the kind's ordering.

    ``pick_order`` (selector greedy sequence) may contain extras (dropped) or
    miss members (SSPC reserved slots, geo floors): it is filtered to the keep
    set with order preserved, and missing members are appended in raster order
    — those were kept for reasons outside the greedy score, so they are the
    least-certain picks under the prefix reading.
    """
    keep_set = {int(i) for i in keep_local}
    if kind == "stride":
        return sorted(keep_set)
    if kind == "random":
        out = sorted(keep_set)
        if rng is not None:
            rng.shuffle(out)
        return out
    order: List[int] = []
    seen = set()
    for t in pick_order or ():
        ti = int(t)
        if ti in keep_set and ti not in seen:
            order.append(ti)
            seen.add(ti)
    for ti in sorted(keep_set - seen):
        order.append(ti)
    return order


def _stride_subset(sorted_items: List[int], k: int) -> List[int]:
    """Evenly spaced ``k`` picks from a sorted list (endpoints included)."""
    n = len(sorted_items)
    if k >= n:
        return list(sorted_items)
    if k <= 1:
        return [sorted_items[n // 2]]
    step = (n - 1) / (k - 1)
    picked: List[int] = []
    used = set()
    for i in range(k):
        j = int(round(i * step))
        while j in used and j + 1 < n:
            j += 1
        while j in used and j - 1 >= 0:
            j -= 1
        used.add(j)
        picked.append(sorted_items[j])
    return picked


def nap_contract_mode() -> str:
    """Stage-2 contraction for the greedy kind: 'strata' (default) or 'prefix'.

    Round-1 diagnosis (500 steps, 352 frames): the raw greedy prefix leaves
    spatial holes ~8.7x the ideal even gap (stride: 3.1x) because BAM's
    orthogonal-residual diversity lives in feature space, not on the grid. At
    the entry budget that is irrelevant (everything valuable enters); at the
    history budget the uncovered regions cost layout context and GP fell below
    nested-uniform. 'strata' keeps the ordering as the priority object but
    bounds the maximal hole by construction; 'prefix' stays as the ablation.
    """
    raw = os.environ.get("GEOPACT_NAP_CONTRACT", "strata").strip().lower()
    return "prefix" if raw == "prefix" else "strata"


def _strata_contract(order: Sequence[int], n_full: int, k: int) -> List[int]:
    """Rank-stratified coverage contraction (parameter-free).

    Partition the full-frame flat index range [0, n_full) into ``k`` equal
    intervals; each interval keeps its best-ranked member of the ordering.
    Interval slots that have no S_cur member are refilled with the globally
    best-ranked unused tokens, so the budget is exactly ``k``. Single O(n)
    pass over the ordering => deterministic, and the rank-0 token is always
    selected (it wins its own interval).
    """
    n_items = len(order)
    if k >= n_items:
        return sorted(int(x) for x in order)
    width = float(n_full) / float(k)
    taken: List[int] = []
    filled = set()
    skipped: List[int] = []
    for t in order:
        ti = int(t)
        cell = min(int(ti / width), k - 1)
        if cell in filled:
            skipped.append(ti)
        else:
            filled.add(cell)
            taken.append(ti)
        if len(taken) == k:
            break
    for ti in skipped:
        if len(taken) == k:
            break
        taken.append(ti)
    return sorted(taken)


def nap_anticipate_enabled() -> bool:
    """Entry-stage coverage guarantee (default on; GEOPACT_NAP_ANTICIPATE=0 is
    the ablation arm)."""
    raw = os.environ.get("GEOPACT_NAP_ANTICIPATE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def anticipate_nap_coverage(
    order: Sequence[int], n_full: int, k_hist: int
) -> Tuple[List[int], Dict[str, Any]]:
    """Stage-1 coverage swap: every stage-2 stratum gets an S_cur member.

    Round-2 diagnosis: strata contraction bounds history holes only inside
    S_cur — when the entry greedy leaves a whole stage-2 stratum with no
    member (dropped runs cluster spatially), the refill comes from global
    rank and the hole survives into every future step's history. The nested
    constraint S_hist ⊂ S_cur is physical (only materialized KV rows can be
    kept), so the fix must happen at entry: for each empty stratum, swap in
    its raster-center token, evicting the lowest-ranked kept tokens from
    strata that hold >= 2 members. Pigeonhole guarantees enough victims
    (|S_cur| >= k_hist). The swap size is the number of empty strata —
    typically a few percent of the entry budget — and swapped-in tokens sit
    at the end of the ordering (least-certain picks), so stage-2 strata
    prefer genuinely ranked tokens wherever one exists. Parameter-free.
    """
    audit: Dict[str, Any] = {"nap_ac_swaps": 0}
    items = [int(t) for t in order]
    k = int(k_hist)
    if k <= 0 or not items or len(items) >= int(n_full) or len(items) < k:
        return items, audit
    width = float(n_full) / float(k)

    def _cell(ti: int) -> int:
        return min(int(ti / width), k - 1)

    counts: Dict[int, int] = {}
    for ti in items:
        c = _cell(ti)
        counts[c] = counts.get(c, 0) + 1
    empty = [c for c in range(k) if c not in counts]
    audit["nap_ac_empty_strata"] = len(empty)
    if not empty:
        return items, audit

    evicted: set = set()
    need = len(empty)
    for ti in reversed(items):
        if need == 0:
            break
        c = _cell(ti)
        if counts[c] >= 2:
            evicted.add(ti)
            counts[c] -= 1
            need -= 1
    if need > 0:
        # cannot happen when len(items) >= k (pigeonhole), guard anyway
        empty = empty[: len(empty) - need]

    kept_set = set(items)
    added: List[int] = []
    for c in empty:
        center = min(int(n_full) - 1, int((c + 0.5) * width))
        # stratum is empty so its raster-center cannot be a kept token,
        # but clamp collisions defensively by scanning the stratum
        if center in kept_set:
            lo = int(c * width)
            hi = min(int(n_full), int((c + 1) * width) + 1)
            center = next((t for t in range(lo, hi) if t not in kept_set), None)
            if center is None:
                continue
        kept_set.add(center)
        added.append(center)

    out = [t for t in items if t not in evicted] + added
    audit["nap_ac_swaps"] = len(added)
    return out, audit


def contract_nap_order(record: Dict[str, Any], k_hist: int) -> List[int]:
    """S_hist ⊂ S_cur at budget ``k_hist`` (full-frame flat indices).

    greedy → rank-stratified coverage subset (default) or raw ordering prefix
             (GEOPACT_NAP_CONTRACT=prefix ablation);
    random → ordering prefix (uniform random subset of S_cur);
    stride → even re-stride inside S_cur.
    """
    order = [int(x) for x in (record.get("order") or ())]
    if not order:
        return []
    k = min(max(1, int(k_hist)), len(order))
    kind = str(record.get("kind") or "greedy")
    if kind == "stride":
        return _stride_subset(sorted(order), k)
    if record.get("crs_nested"):
        # CRS-N (Module C v5 multi-step): the certificate lives in the raw
        # ordering — its k_hist prefix is head + certified regional medoids
        # by construction. Strata re-bucketing would scramble it, so the
        # order record itself pins the contraction semantics.
        return order[:k]
    if kind == "greedy" and nap_contract_mode() == "strata":
        n_full = int(record.get("n_full") or 0)
        if n_full <= 0:
            n_full = max(order) + 1
        return _strata_contract(order, n_full, k)
    return order[:k]


# ---------------------------------------------------------------------------
# Per-model order store (sticky-mask lifecycle)
# ---------------------------------------------------------------------------


def _order_store(model) -> Dict[Tuple, Dict[str, Any]]:
    store = getattr(model, "_geopact_nap_orders", None)
    if store is None:
        store = {}
        model._geopact_nap_orders = store
    return store


def store_nap_order(
    model,
    cache_key,
    *,
    order: Sequence[int],
    kind: str,
    n_full: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not cache_key:
        return
    rec: Dict[str, Any] = {
        "order": tuple(int(x) for x in order),
        "kind": str(kind),
        "n_full": int(n_full),
    }
    if extra:
        # e.g. C-Absorb: 'absorb_parents' (rank-monotone tree) + 'grid_w'
        rec.update(extra)
    _order_store(model)[cache_key] = rec


def lookup_nap_order(model, cache_key) -> Optional[Dict[str, Any]]:
    if not cache_key:
        return None
    store = getattr(model, "_geopact_nap_orders", None)
    if not store:
        return None
    return store.get(cache_key)


def clear_nap_orders(model) -> None:
    if hasattr(model, "_geopact_nap_orders"):
        model._geopact_nap_orders = None


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


def nap_full_token_count(grid_row, merge_size: int) -> int:
    """Full-frame LLM token count from a (t, h, w) vision grid row."""
    if grid_row is None:
        return 0
    row = grid_row
    if hasattr(row, "dim") and row.dim() > 1:
        row = row[-1]
    vals = [int(v) for v in row.tolist()]
    if len(vals) < 3:
        return 0
    t, h, w = vals[0], vals[1], vals[2]
    m = max(1, int(merge_size))
    return (t * h * w) // (m * m)
