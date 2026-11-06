"""C-Entry: single-step Module C variant — dropped-set summary tokens at entry.

Multi-step C (QASMR) compensates dropped tokens in KV space, because there the
dropped rows were actually materialized in the dialogue cache. Single-step
episodes never materialize the dropped tokens at all, but the same zeroth-order
object exists one level earlier: the dropped embeddings. C-Entry closes the
single-step gap with the identical construction, moved from KV space to
embedding space:

  1. Module B runs with its budget reduced by a reserve R (in-budget: the
     total kept token count stays exactly k).
  2. The dropped set is partitioned with the same GUI-constrained spatial
     partition the relay builder uses (diameter cap, certificate-driven
     agglomeration) — one shared geometric object across both C variants.
  3. Each cluster contributes one summary token: the cluster-mean embedding,
     carried at the cluster medoid's original sequence slot, so origpos MRoPE
     encodes the summary's true spatial location and FlashAttention sees
     nothing but ordinary tokens (no attention-path change at all).

Reserve rule (fixed constants, not knobs — same spirit as the relay cap):
  R = min(16, max(1, keep_k // 16))   (<= 6.25% of the keep budget)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

__all__ = [
    "c_entry_enabled",
    "c_entry_reserve",
    "build_c_entry_summaries",
]

_C_ENTRY_CAP = 16


def c_entry_enabled() -> bool:
    return os.environ.get("GEOPACT_C_ENTRY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def c_entry_reserve(keep_k: int) -> int:
    """Slots reserved from Module B's budget for summary tokens."""
    k = int(keep_k)
    if k <= 1:
        return 0
    return min(_C_ENTRY_CAP, max(1, k // 16))


def _grid_pos(indices: torch.Tensor, grid_w: int, grid_h: int) -> torch.Tensor:
    """Normalized (y, x) centers for flat row-major token indices."""
    w = max(1, int(grid_w))
    h = max(1, int(grid_h))
    y = (indices // w).to(torch.float32)
    x = (indices % w).to(torch.float32)
    return torch.stack([(y + 0.5) / h, (x + 0.5) / w], dim=-1)


def build_c_entry_summaries(
    select_embeds: torch.Tensor,
    keep_local: Set[int],
    n: int,
    grid_h: int,
    grid_w: int,
    reserve: int,
) -> Tuple[List[int], Optional[torch.Tensor], Dict[str, Any]]:
    """Summaries of the dropped set: (medoid local indices, [R, hidden] means, audit).

    Clusters beyond the reserve are discarded (largest-first keep), never
    merged: a summary is only trusted under the partition's diameter
    certificate. When the dropped set is smaller than the reserve, the
    unused slots are simply not spent (logged as shortfall).
    """
    audit: Dict[str, Any] = {"c_entry_reserve": int(reserve)}
    dropped = sorted(set(range(int(n))) - {int(i) for i in keep_local})
    audit["c_entry_dropped"] = len(dropped)
    if reserve <= 0 or not dropped:
        return [], None, audit

    from lmms_eval.models.geopact_spatial_relay import relay_diam, spatial_partition

    device = select_embeds.device
    drop_t = torch.tensor(dropped, dtype=torch.long, device=device)
    pos = _grid_pos(drop_t, grid_w, grid_h)
    feat = select_embeds.index_select(0, drop_t).to(torch.float32)

    budget = min(int(reserve), len(dropped))
    clusters, part_audit = spatial_partition(
        pos,
        budget=budget,
        diam_max=relay_diam(),
        feat=feat,
    )
    audit["c_entry_clusters_raw"] = len(clusters)
    if not clusters:
        return [], None, audit
    # spatial_partition may hand back CPU index tensors; align with pos/feat.
    clusters = [c.to(device=device, dtype=torch.long) for c in clusters]

    # Over-budget clusters are DISCARDED, not folded: a summary is only
    # meaningful under the partition's diameter certificate, and merging
    # distant clusters voids it (the r05 screen showed folded summaries
    # average ~300 tokens across unrelated regions and cost accuracy).
    # Largest clusters keep the most recovered mass, so they win the slots.
    if len(clusters) > budget:
        clusters = sorted(clusters, key=lambda c: -int(c.numel()))
        audit["c_entry_discarded"] = len(clusters) - budget
        clusters = clusters[:budget]

    medoids: List[int] = []
    means: List[torch.Tensor] = []
    for c in clusters:
        cpos = pos.index_select(0, c)
        center = cpos.mean(0, keepdim=True)
        medoid_row = int(c[int(torch.cdist(center, cpos).argmin())])
        medoids.append(int(drop_t[medoid_row]))
        means.append(feat.index_select(0, c).mean(0))
    audit["c_entry_tokens"] = len(medoids)
    audit["c_entry_shortfall"] = max(0, int(reserve) - len(medoids))
    audit["c_entry_mean_cluster"] = round(len(dropped) / max(1, len(medoids)), 2)
    return medoids, torch.stack(means), audit
