"""NAP-History Absorb (Module C v3): rank-monotone load-balanced value absorption.

The main
method is *history-only, key-frozen, contextualized-value, FA2-native, strict
budget*:

  Stage 1 (entry)   — BAM produces the ordered pick sequence for S_cur. A
                      rank-monotone parent tree over the ordering is built once
                      (``build_absorb_parents``) and serialized into the NAP
                      order record. parent(rank r) has rank < r; rank 0 is the
                      root and survives every contraction (strata keeps the
                      best-ranked member of its stratum; prefix keeps rank 0
                      trivially).
  Stage 2 (history) — when the frame's KV segment is contracted to S_hist, each
                      dropped row j resolves its host as the first surviving
                      ancestor on its parent chain (``resolve_hosts``); with
                      ``balanced`` assignment the loads are then capacity-
                      capped at ceil(|D|/|S|), which drives the multiplicity
                      ratio kappa -> 1. Donor *contextualized* value rows are
                      summed per host **before** the cache crop
                      (``collect_donor_value_sums``), and after the merged
                      re-prefill the surviving value rows are rewritten in
                      place (``apply_history_absorb``):

                          v~_i = (v_i + lam * S_i) / (1 + lam * m_i)

                      Keys, MRoPE positions, row count and the FA2 attention
                      path are untouched; sequence length stays exactly the
                      history budget.

Zero-order theory (spec v2): with equal per-host multiplicity (kappa = 1) and
zero key mismatch (delta = 0) the compressed attention readout equals the full
readout exactly — the multiplicity factor is a common constant and cancels in
softmax. For kappa > 1 the residual is bounded by TV(p, p^c) * value-diameter
with TV <= (sqrt(kappa)-1)/(sqrt(kappa)+1). Load balancing is therefore the
design principle, not an afterthought.

Mass semantics are *flat*: every donor contributes exactly one unit of mass and
one value row to its final host (chain resolution only routes; it never
compounds lambda). Sum over hosts of m_i == |D| — donors are neither lost nor
double-absorbed for any survivor set containing rank 0.

Env gates (all read lazily; default = feature off):
  - ``GEOPACT_C_ABSORB=1``           enable history absorb (greedy NAP only)
  - ``GEOPACT_C_LAMBDA_HIST``        lambda_hist (default 1.0); lam <= 0 is a
                                     hard early-return, pipeline bit-identical
  - ``GEOPACT_C_ABSORB_ASSIGN``      'balanced' (default) | 'nearest' (E3 arm)
                                     | 'exact' (floor/ceil block assignment,
                                     kappa == exactly (1+ceil)/(1+floor); skips
                                     the parent tree on the hot path)
  - ``GEOPACT_C_ABSORB_MODE``        'absolute' (default, eq. 6 of spec v2)
                                     | 'delta' (old-host-relative residual
                                     transport)
                                     | 'oldhost' (host uses pre-crop old value)
  - ``GEOPACT_C_LAMBDA_ENTRY``       entry embedding-merge extension (default 0
                                     = off). Independent ToMe-like arm: merges
                                     dropped *embeddings* into kept ones before
                                     the first forward. This changes keys and
                                     the whole downstream trajectory — it does
                                     NOT inherit the value-only theory above.

The deployed
formula mixes the *old* (pre-crop) donor values with the *new* (post-re-prefill)
host value, so Proposition 1 does not directly cover the absolute merge. The
'delta' mode transports the donor residual relative to the OLD host instead:

    v_transport = v_new + alpha_i * (mu_old_i - v_old_i),
    alpha_i     = lam * m_i / (1 + lam * m_i)

Limits: (v_new == v_old, lam=1) recovers the zero-order group mean exactly;
donors identical to the old host transport nothing (the re-prefill drift of the
host is fully preserved); the common context shift lives in v_new only. The
per-layer drift eta_v = ||v_new - v_old|| is audited on host rows either way.
"""
from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

__all__ = [
    "c_absorb_enabled",
    "absorb_lambda_hist",
    "absorb_tau",
    "absorb_assign_mode",
    "absorb_value_mode",
    "entry_merge_lambda",
    "build_absorb_parents",
    "resolve_hosts",
    "rebalance_hosts",
    "exact_assign_hosts",
    "random_exact_assign_hosts",
    "absorb_load_audit",
    "collect_donor_value_sums",
    "apply_history_absorb",
    "entry_merge_embeds",
    "coc_enabled",
    "coc_value_features",
    "coc_partition_dp",
    "coc_pick_hosts",
    "plan_coc_contract",
    "plan_from_host_map",
]


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def c_absorb_enabled() -> bool:
    return os.environ.get("GEOPACT_C_ABSORB", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def absorb_lambda_hist() -> float:
    raw = os.environ.get("GEOPACT_C_LAMBDA_HIST", "").strip()
    if not raw:
        return 1.0
    try:
        return float(raw)
    except ValueError:
        return 1.0


def absorb_assign_mode() -> str:
    raw = os.environ.get("GEOPACT_C_ABSORB_ASSIGN", "balanced").strip().lower()
    return raw if raw in {"nearest", "exact", "randexact"} else "balanced"


def absorb_value_mode() -> str:
    """Factorized value-transport modes (review round 2 §D, 2x2 design):

      'absolute' (default)  v~ = (1-a)v_new + a*mu_old         (gamma=1, beta=1)
      'delta'               v~ = v_new + a*(mu_old - v_old)    (gamma=0, beta=1)
      'drift'               v~ = v_new + a*(v_old - v_new)     (gamma=1, beta=0)
      'oldhost'             v~ = (1-a)v_old + a*mu_old         (counterfactual)

    with a = alpha_i = lam*m_i/(1+lam*m_i) (or a = tau when GEOPACT_C_TAU is
    set). absolute - delta == a*(v_old - v_new): the two differ ONLY by the
    host context-drift rollback term, so 'drift' is the missing host-only
    control that decides whether the gain comes from donor transport or from
    rollback.
    """
    raw = os.environ.get("GEOPACT_C_ABSORB_MODE", "absolute").strip().lower()
    return raw if raw in {"delta", "oldhost", "drift"} else "absolute"


def absorb_tau() -> Optional[float]:
    """GEOPACT_C_TAU in (0,1): budget-stable interpolation strength (review §E.3).

    When set, the per-host absorption coefficient is the constant a = tau
    instead of alpha_i = lam*m_i/(1+lam*m_i) — equivalently a per-host
    lambda_i = tau / (m_i*(1-tau)) — so the transported fraction does not
    grow with the donor count m_i and keeps one semantics across grids.
    """
    raw = os.environ.get("GEOPACT_C_TAU", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if 0.0 < val < 1.0 else None


def entry_merge_lambda() -> float:
    raw = os.environ.get("GEOPACT_C_LAMBDA_ENTRY", "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def absorb_sim_mode() -> str:
    """Parent-similarity source for the tree (E4 assignment diagnostics).

    'gram' (default) — embedding cosine; 'spatial' — grid proximity only;
    'random' — seeded random similarity (negative control).
    """
    raw = os.environ.get("GEOPACT_C_ABSORB_SIM", "gram").strip().lower()
    return raw if raw in {"gram", "spatial", "random"} else "gram"


# ---------------------------------------------------------------------------
# W-A1: rank-monotone parent tree
# ---------------------------------------------------------------------------

# Build-time cap on direct children per node. Prevents the degenerate star at
# rank 0 (which would concentrate all donor mass on one host in 'nearest'
# mode); the resolve-time balanced pass enforces the real capacity bound.
_BUILD_CHILD_CAP = 4


def build_absorb_parents(
    order: Sequence[int],
    *,
    embeds: Optional[torch.Tensor] = None,
    grid_h: int = 0,
    grid_w: int = 0,
) -> List[int]:
    """Parent tree over the BAM ordering. ``parents[r] < r`` for r >= 1.

    ``order`` holds full-frame flat indices in rank order (S_cur). ``embeds``,
    when given, are the *frame-local* embedding rows (``frame_embeds[order]``
    is taken internally); parent choice maximizes cosine similarity among
    earlier ranks under the child cap. Without embeds, spatial grid proximity
    of the flat indices is used. Deterministic; O(k^2) similarity + one pass.
    """
    k = len(order)
    if k == 0:
        return []
    if k == 1:
        return [-1]

    sim_mode = absorb_sim_mode()
    use_gram = (
        sim_mode == "gram"
        and embeds is not None
        and int(embeds.shape[0]) > max(int(t) for t in order)
    )
    if use_gram:
        idx = torch.tensor([int(t) for t in order], dtype=torch.long, device=embeds.device)
        rows = embeds.index_select(0, idx).float()
        rows = torch.nn.functional.normalize(rows, dim=-1)
        sims = rows @ rows.t()
    elif sim_mode == "random":
        gen = torch.Generator().manual_seed(int(order[0]) * 9973 + k)
        sims = torch.rand(k, k, generator=gen)
    else:
        gw = max(1, int(grid_w))
        flat = torch.tensor([int(t) for t in order], dtype=torch.float32)
        pos = torch.stack([flat // gw, flat % gw], dim=1)
        sims = -torch.cdist(pos, pos)

    # Restrict candidates to strictly earlier ranks (rank-monotonicity) and
    # keep only each row's top candidates — the sequential cap pass below only
    # ever needs a handful before falling back to min-load.
    r_idx = torch.arange(k, device=sims.device)
    sims = sims.masked_fill(r_idx.view(1, -1) >= r_idx.view(-1, 1), float("-inf"))
    n_top = min(k, max(_BUILD_CHILD_CAP * 2, 8))
    top = torch.topk(sims, n_top, dim=1)
    cand = top.indices.cpu().tolist()
    valid = (~torch.isinf(top.values)).cpu().tolist()

    parents = [-1] * k
    child_count = [0] * k
    for r in range(1, k):
        chosen = -1
        fallback = -1
        for ci, c in enumerate(cand[r]):
            if not valid[r][ci]:
                break
            if fallback < 0:
                fallback = c
            if child_count[c] < _BUILD_CHILD_CAP:
                chosen = c
                break
        if chosen < 0:
            # all top candidates at cap: min-load earlier rank (deterministic).
            best = min(range(r), key=lambda c: (child_count[c], c))
            chosen = best if fallback < 0 or child_count[fallback] > child_count[best] else fallback
        parents[r] = int(chosen)
        child_count[chosen] += 1
    return parents


def resolve_hosts(
    parents: Sequence[int],
    survivor_ranks: Set[int],
) -> Tuple[List[int], List[int]]:
    """First surviving ancestor for every dropped rank.

    Returns ``(donor_ranks, host_ranks)`` aligned lists. Requires rank 0 in
    ``survivor_ranks`` (guaranteed by both strata and prefix contraction);
    raises otherwise — a dangling host would silently corrupt mass accounting.
    Flat mass semantics: routing only, one unit per donor.
    """
    k = len(parents)
    if 0 not in survivor_ranks and k > 0:
        raise ValueError("absorb resolve: rank 0 not in survivor set")
    # memoized chain walk: host[r] for dropped r
    host_memo: Dict[int, int] = {}

    def _host(r: int) -> int:
        if r in survivor_ranks:
            return r
        if r in host_memo:
            return host_memo[r]
        chain = []
        cur = r
        while cur not in survivor_ranks:
            if cur in host_memo:
                cur = host_memo[cur]
                break
            chain.append(cur)
            cur = parents[cur]
            if cur < 0:
                raise ValueError("absorb resolve: dangling parent chain")
        for c in chain:
            host_memo[c] = cur
        return cur

    donor_ranks: List[int] = []
    host_ranks: List[int] = []
    for r in range(k):
        if r in survivor_ranks:
            continue
        donor_ranks.append(r)
        host_ranks.append(_host(r))
    return donor_ranks, host_ranks


def rebalance_hosts(
    donor_ranks: Sequence[int],
    host_ranks: Sequence[int],
    survivor_ranks: Set[int],
    *,
    order: Sequence[int],
    grid_w: int,
) -> List[int]:
    """Capacity-capped reassignment: loads <= ceil(|D|/|S|), kappa -> 1.

    Overloaded hosts shed their spatially farthest donors to the nearest
    survivor with spare capacity. Deterministic, conserves donor count.
    """
    n_d = len(donor_ranks)
    survivors = sorted(survivor_ranks)
    n_s = len(survivors)
    if n_d == 0 or n_s == 0:
        return list(host_ranks)
    cap = max(1, math.ceil(n_d / n_s))
    load: Dict[int, int] = {s: 0 for s in survivors}
    for h in host_ranks:
        load[h] = load.get(h, 0) + 1
    if max(load.values()) <= cap:
        return list(host_ranks)

    gw = max(1, int(grid_w))

    def _pos(rank: int) -> Tuple[float, float]:
        t = int(order[rank])
        return (t // gw, t % gw)

    def _dist(a: int, b: int) -> float:
        pa, pb = _pos(a), _pos(b)
        return (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2

    out = list(host_ranks)
    donors_of: Dict[int, List[int]] = {}
    for di, h in enumerate(out):
        donors_of.setdefault(h, []).append(di)
    under = [s for s in survivors if load[s] < cap]
    for h in sorted(donors_of.keys(), key=lambda x: -load[x]):
        while load[h] > cap:
            # shed the donor farthest from this host
            members = donors_of[h]
            far_i = max(members, key=lambda di: _dist(donor_ranks[di], h))
            # nearest under-capacity survivor
            best = None
            best_d = None
            for s in under:
                if load[s] >= cap:
                    continue
                d = _dist(donor_ranks[far_i], s)
                if best_d is None or d < best_d:
                    best_d = d
                    best = s
            if best is None:
                break  # everything at cap: cannot happen with cap=ceil(|D|/|S|)
            members.remove(far_i)
            out[far_i] = best
            donors_of.setdefault(best, []).append(far_i)
            load[h] -= 1
            load[best] += 1
    return out


def exact_assign_hosts(
    donor_full: Sequence[int],
    survivor_full: Sequence[int],
) -> List[int]:
    """Exact floor/ceil block assignment in raster order (review §5.2/§12.1).

    Donors and survivors are sorted by full-frame flat index; survivor i takes
    the next ``q+1`` donors for the first ``r`` survivors and ``q`` afterwards
    (``|D| = q*|S| + r``), so every load is in {floor, ceil} — the count-only
    kappa is exactly (1+ceil)/(1+floor), the minimum achievable. Spatially
    contiguous by construction; no parent tree, no similarity, O(n log n).

    Returns ``host_full`` aligned to the *input* donor order.
    """
    n_d, n_s = len(donor_full), len(survivor_full)
    if n_d == 0 or n_s == 0:
        return []
    d_sorted = sorted(range(n_d), key=lambda i: int(donor_full[i]))
    s_sorted = sorted(int(t) for t in survivor_full)
    q, r = divmod(n_d, n_s)
    out = [0] * n_d
    di = 0
    for si, host in enumerate(s_sorted):
        take = q + (1 if si < r else 0)
        for _ in range(take):
            out[d_sorted[di]] = host
            di += 1
    return out


def random_exact_assign_hosts(
    donor_full: Sequence[int],
    survivor_full: Sequence[int],
    seed: int,
) -> List[int]:
    """Load-matched random donor->host map (review §D.4 negative control).

    Identical floor/ceil per-host loads as ``exact_assign_hosts`` (same kappa,
    same TV bound) but the donor->host correspondence is a seeded uniform
    shuffle instead of raster blocks. If this arm matches spatial-exact, the
    absorb gain is value smoothing / old-state injection, not semantic
    donor-host affinity. Deterministic per seed.
    """
    n_d, n_s = len(donor_full), len(survivor_full)
    if n_d == 0 or n_s == 0:
        return []
    s_sorted = sorted(int(t) for t in survivor_full)
    q, r = divmod(n_d, n_s)
    slots: List[int] = []
    for si, host in enumerate(s_sorted):
        slots.extend([host] * (q + (1 if si < r else 0)))
    rng = random.Random(int(seed) & 0x7FFFFFFF)
    rng.shuffle(slots)
    return slots


def absorb_load_audit(
    host_ranks: Sequence[int], survivor_ranks: Set[int]
) -> Dict[str, Any]:
    """kappa / load stats over *all* survivors (empty hosts have mass 0)."""
    load: Dict[int, int] = {s: 0 for s in survivor_ranks}
    for h in host_ranks:
        load[h] = load.get(h, 0) + 1
    if not load:
        return {}
    masses = sorted(load.values())
    m_min, m_max = masses[0], masses[-1]
    kappa = float(1 + m_max) / float(1 + m_min)
    tv_bound = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    return {
        "absorb_m_min": int(m_min),
        "absorb_m_max": int(m_max),
        "absorb_kappa": round(kappa, 4),
        "absorb_tv_bound": round(tv_bound, 4),
        "absorb_hosts_used": int(sum(1 for v in load.values() if v > 0)),
        "absorb_donors": int(len(host_ranks)),
    }


# ---------------------------------------------------------------------------
# W-A2: donor value collection (pre-crop) + in-place host rewrite (post-prefill)
# ---------------------------------------------------------------------------


def collect_donor_value_sums(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    donor_seq: Sequence[int],
    host_pos: Sequence[int],
    n_hosts: int,
    host_seq: Optional[Sequence[int]] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """Per-layer host-wise sums of donor *contextualized* value rows.

    ``donor_seq`` are absolute cache sequence positions of the donor rows (must
    be sliced before the frame crop); ``host_pos`` maps each donor to its
    host's position inside the kept-row segment [0, n_hosts).

    Returns (sums_per_layer [n_hosts, H_kv, d] fp32, mass [n_hosts] fp32,
    audit). When ``host_seq`` is given (absolute pre-crop position of each
    donor's host), the OLD host value rows are additionally captured per layer
    into ``audit['host_old']`` ([n_hosts, H_kv, d] fp32, zeros for untouched
    hosts) — required by the 'delta' / 'oldhost' value modes and by the
    re-prefill drift audit. Also computes a donor-dispersion diagnostic.
    """
    audit: Dict[str, Any] = {}
    sums: List[torch.Tensor] = []
    if not legacy_tensors or not donor_seq:
        return sums, torch.zeros(0), audit
    device = legacy_tensors[0][1].device
    d_idx = torch.tensor(list(donor_seq), dtype=torch.long, device=device)
    h_idx = torch.tensor(list(host_pos), dtype=torch.long, device=device)
    mass = torch.zeros(int(n_hosts), dtype=torch.float32, device=device)
    mass.index_add_(0, h_idx, torch.ones_like(h_idx, dtype=torch.float32))

    # unique (kept-segment position -> pre-crop cache row) map for old hosts
    host_old_layers: Optional[List[torch.Tensor]] = None
    u_pos_t = u_seq_t = None
    if host_seq is not None and len(host_seq) == len(donor_seq):
        seq_of_pos: Dict[int, int] = {}
        for p, sq in zip(host_pos, host_seq):
            seq_of_pos.setdefault(int(p), int(sq))
        u_pos = sorted(seq_of_pos)
        u_pos_t = torch.tensor(u_pos, dtype=torch.long, device=device)
        u_seq_t = torch.tensor([seq_of_pos[p] for p in u_pos], dtype=torch.long, device=device)
        host_old_layers = []

    disp_sum = 0.0
    disp_max = 0.0
    n_layers = 0
    for _k, v in legacy_tensors:
        # v: [B, H_kv, S, d]
        dv = v[0].index_select(1, d_idx.to(v.device)).float()  # [H_kv, D, d]
        h_kv, n_d, dim = dv.shape
        s = torch.zeros(int(n_hosts), h_kv, dim, dtype=torch.float32, device=dv.device)
        s.index_add_(0, h_idx.to(dv.device), dv.permute(1, 0, 2).contiguous())
        sums.append(s)
        if host_old_layers is not None:
            ho = torch.zeros(
                int(n_hosts), h_kv, dim, dtype=torch.float32, device=dv.device
            )
            hv = v[0].index_select(1, u_seq_t.to(v.device)).float()  # [H_kv, U, d]
            ho.index_copy_(0, u_pos_t.to(dv.device), hv.permute(1, 0, 2).contiguous())
            host_old_layers.append(ho)
        # dispersion diagnostic: donor rows around their host-group mean
        cnt = mass.to(dv.device).clamp(min=1.0)
        mean = s / cnt.view(-1, 1, 1)
        diff = dv.permute(1, 0, 2) - mean.index_select(0, h_idx.to(dv.device))
        norms = diff.norm(dim=-1)  # [D, H_kv]
        disp_sum += float(norms.mean().item())
        disp_max = max(disp_max, float(norms.max().item()))
        n_layers += 1
    if n_layers:
        audit["absorb_value_disp_mean"] = round(disp_sum / n_layers, 4)
        audit["absorb_value_disp_max"] = round(disp_max, 4)
    if host_old_layers is not None:
        audit["host_old"] = host_old_layers
    return sums, mass, audit


def _cache_layer_kv(cache: Any, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Live (keys, values) tensors of a cache layer (in-place writable)."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[idx].keys, layers[idx].values
    return cache.key_cache[idx], cache.value_cache[idx]


def apply_history_absorb(
    cache: Any,
    *,
    seg_start: int,
    host_pos: Sequence[int],
    sums: Sequence[torch.Tensor],
    mass: torch.Tensor,
    lam: float,
    host_old: Optional[Sequence[torch.Tensor]] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """In-place value rewrite of surviving rows after the merged re-prefill.

    Modes (row at cache position ``seg_start + p``, S = donor sum, m = mass,
    mu = S/m, a = lam*m/(1+lam*m) or a = tau when GEOPACT_C_TAU is set):
      absolute  v~ = (v_new + lam*S) / (1 + lam*m)  ==  (1-a)v_new + a*mu
      delta     v~ = v_new + a * (mu - v_old)       (donor-only transport)
      drift     v~ = v_new + a * (v_old - v_new)    (host rollback only; the
                exact algebraic difference absolute - delta — review §D.3)
      oldhost   v~ = (v_old + lam*S) / (1 + lam*m)  (counterfactual old cache)
    'delta'/'oldhost'/'drift' require ``host_old``; missing rows fall back to
    'absolute' per call (audited). Keys untouched. An empty plan (or lam <= 0
    with no tau) returns immediately without touching the cache (bit-exact
    deletion). Always audits the re-prefill drift eta_v = ||v_new - v_old||
    on touched host rows when old host values are available (review §3.5).
    """
    tau = absorb_tau()
    if (lam <= 0.0 and tau is None) or not sums or not len(host_pos):
        return {"absorb_applied": False}
    n_hosts = int(mass.shape[0])
    touched = torch.nonzero(mass > 0, as_tuple=False).flatten()
    n_layers = len(sums)
    want_mode = mode or absorb_value_mode()
    have_old = host_old is not None and len(host_old) == n_layers
    eff_mode = want_mode if (want_mode == "absolute" or have_old) else "absolute"
    drift_sum = 0.0
    drift_rel_sum = 0.0
    drift_max = 0.0
    drift_layers = 0
    for li in range(n_layers):
        _keys, values = _cache_layer_kv(cache, li)
        dev = values.device
        t_idx = touched.to(dev)
        rows = t_idx + int(seg_start)
        v_rows = values[0].index_select(1, rows).float()  # [H_kv, T, d]
        s = sums[li].to(dev).index_select(0, t_idx).permute(1, 0, 2)  # [H_kv, T, d]
        m = mass.to(dev).index_select(0, t_idx).view(1, -1, 1)
        # absorption coefficient: tau is budget-stable (constant across hosts);
        # the lambda form grows with the donor count m_i.
        a = float(tau) if tau is not None else None
        if have_old:
            v_old = host_old[li].to(dev).index_select(0, t_idx).permute(1, 0, 2)
            diff_n = (v_rows - v_old).norm(dim=-1)  # [H_kv, T]
            old_n = v_old.norm(dim=-1).clamp(min=1e-6)
            drift_sum += float(diff_n.mean().item())
            drift_rel_sum += float((diff_n / old_n).mean().item())
            drift_max = max(drift_max, float(diff_n.max().item()))
            drift_layers += 1
        if eff_mode == "delta":
            if a is None:
                merged = v_rows + (lam / (1.0 + lam * m)) * (s - m * v_old)
            else:
                merged = v_rows + a * (s / m - v_old)
        elif eff_mode == "drift":
            a_t = a if a is not None else (lam * m) / (1.0 + lam * m)
            merged = v_rows + a_t * (v_old - v_rows)
        elif eff_mode == "oldhost":
            if a is None:
                merged = (v_old + lam * s) / (1.0 + lam * m)
            else:
                merged = (1.0 - a) * v_old + a * (s / m)
        else:
            if a is None:
                merged = (v_rows + lam * s) / (1.0 + lam * m)
            else:
                merged = (1.0 - a) * v_rows + a * (s / m)
        values[0].index_copy_(1, rows, merged.to(values.dtype))
    out: Dict[str, Any] = {
        "absorb_applied": True,
        "absorb_layers": int(n_layers),
        "absorb_hosts_touched": int(touched.numel()),
        "absorb_n_hosts": n_hosts,
        "absorb_lambda": float(lam),
        "absorb_value_mode": eff_mode,
    }
    if tau is not None:
        out["absorb_tau"] = float(tau)
    if want_mode != eff_mode:
        out["absorb_mode_fallback"] = f"{want_mode}->absolute (no host_old)"
    if drift_layers:
        out["absorb_drift_v_mean"] = round(drift_sum / drift_layers, 4)
        out["absorb_drift_v_rel"] = round(drift_rel_sum / drift_layers, 4)
        out["absorb_drift_v_max"] = round(drift_max, 4)
    return out


def key_audit_enabled() -> bool:
    """GEOPACT_C_ABSORB_KEY_AUDIT=1: per-frame key-mismatch calibration (W-A4).

    Measures, on real frames, the quantity the delta-term of the error chain
    depends on: per-layer post-MRoPE ||k_donor - k_host|| from the live cache
    rows, its size relative to host key norms, and the correlation between
    spatial grid distance and key distance (the review's minimum requirement
    before any locality claim). Diagnostic only; adds gather work per frame.
    """
    return os.environ.get("GEOPACT_C_ABSORB_KEY_AUDIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def key_mismatch_audit(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    donor_seq: Sequence[int],
    host_seq: Sequence[int],
    donor_full: Sequence[int],
    host_full: Sequence[int],
    grid_w: int,
) -> Dict[str, Any]:
    """Post-RoPE key mismatch stats between each donor and its host (W-A4)."""
    if not legacy_tensors or not donor_seq or len(donor_seq) != len(host_seq):
        return {}
    device = legacy_tensors[0][0].device
    d_idx = torch.tensor(list(donor_seq), dtype=torch.long, device=device)
    h_idx = torch.tensor(list(host_seq), dtype=torch.long, device=device)
    n_d = int(d_idx.numel())
    per_donor = torch.zeros(n_d, dtype=torch.float32, device=device)
    rel_sum = 0.0
    p95s: List[float] = []
    n_layers = 0
    for k, _v in legacy_tensors:
        kd = k[0].index_select(1, d_idx.to(k.device)).float()  # [H, D, d]
        kh = k[0].index_select(1, h_idx.to(k.device)).float()
        dist = (kd - kh).norm(dim=-1)  # [H, D]
        host_norm = kh.norm(dim=-1).clamp(min=1e-6)
        per_donor += dist.mean(dim=0).to(device)
        rel_sum += float((dist / host_norm).mean().item())
        p95s.append(float(dist.flatten().quantile(0.95).item()))
        n_layers += 1
    if n_layers == 0:
        return {}
    per_donor /= n_layers
    gw = max(1, int(grid_w))
    df = torch.tensor([float(t) for t in donor_full], device=device)
    hf = torch.tensor([float(t) for t in host_full], device=device)
    sp = (
        ((df // gw) - (hf // gw)) ** 2 + ((df % gw) - (hf % gw)) ** 2
    ).sqrt()
    corr = float("nan")
    if n_d >= 3 and float(sp.std()) > 0 and float(per_donor.std()) > 0:
        a = (sp - sp.mean()) / sp.std()
        b = (per_donor - per_donor.mean()) / per_donor.std()
        corr = float((a * b).mean().item())
    return {
        "absorb_key_dist_mean": round(float(per_donor.mean().item()), 4),
        "absorb_key_dist_p95": round(sum(p95s) / len(p95s), 4),
        "absorb_key_rel_mean": round(rel_sum / n_layers, 4),
        "absorb_key_spatial_corr": round(corr, 4) if corr == corr else "nan",
        "absorb_key_layers": int(n_layers),
    }


# ---------------------------------------------------------------------------
# W-A2 orchestration: plan resolution in rank space -> cache row coordinates
# ---------------------------------------------------------------------------


def plan_history_absorb(
    *,
    rec: Dict[str, Any],
    keep_full: Sequence[int],
    scur_sorted: Sequence[int],
    vis_start: int,
) -> Optional[Dict[str, Any]]:
    """Resolve the absorption plan against the actual S_hist survivor set.

    'exact' assignment (review §5.2) bypasses the parent tree entirely: donors
    are block-assigned to survivors in raster order with floor/ceil loads.
    'balanced'/'nearest' resolve the stored rank-monotone tree (fallback path,
    kept for ablation and for records without exact support).

    Returns ``{donor_seq, host_pos, n_hosts, audit}`` in cache coordinates, or
    None when the record carries no plan (non-greedy kinds, lost records).
    """
    parents = rec.get("absorb_parents")
    order = [int(x) for x in (rec.get("order") or ())]
    if not order or str(rec.get("kind") or "greedy") != "greedy":
        return None
    mode = absorb_assign_mode()
    tree_free = mode in ("exact", "randexact")
    # Tree modes need the stored rank-monotone parents; exact/randexact block
    # assignment is tree-free (review §5.2) and must not be silently disabled
    # when the record predates parent construction.
    if not tree_free and (not parents or len(parents) != len(order)):
        return None
    keep_set = {int(x) for x in keep_full}
    rank_of = {t: r for r, t in enumerate(order)}
    survivor_ranks = {rank_of[t] for t in keep_set if t in rank_of}
    if not survivor_ranks:
        return None
    # rank-0 is a tree invariant (resolve_hosts walks parent chains up to the
    # root); exact assignment has no such requirement.
    if not tree_free and 0 not in survivor_ranks:
        return None
    grid_w = int(rec.get("grid_w") or 0)
    if tree_free:
        donor_ranks = [r for r in range(len(order)) if r not in survivor_ranks]
        if not donor_ranks:
            return None
        d_full = [order[r] for r in donor_ranks]
        s_full = [order[r] for r in sorted(survivor_ranks)]
        if mode == "randexact":
            # deterministic per-frame seed from the stored order identity
            seed = (len(order) * 1000003) ^ (order[0] * 31 + order[-1])
            h_full = random_exact_assign_hosts(d_full, s_full, seed)
        else:
            h_full = exact_assign_hosts(d_full, s_full)
        rank_back = {t: r for r, t in enumerate(order)}
        host_ranks = [rank_back[t] for t in h_full]
    else:
        donor_ranks, host_ranks = resolve_hosts(list(parents), survivor_ranks)
        if not donor_ranks:
            return None
        if mode == "balanced":
            host_ranks = rebalance_hosts(
                donor_ranks,
                host_ranks,
                survivor_ranks,
                order=order,
                grid_w=grid_w if grid_w > 0 else 1,
            )
    audit = absorb_load_audit(host_ranks, survivor_ranks)
    audit["absorb_assign"] = mode

    # rank -> full index -> S_cur row (cache row) / kept-segment position
    row_of_full = {t: i for i, t in enumerate(scur_sorted)}
    kept_sorted = sorted(keep_set)
    kept_pos_of_full = {t: p for p, t in enumerate(kept_sorted)}
    donor_full = [order[r] for r in donor_ranks]
    host_full = [order[r] for r in host_ranks]
    donor_seq = [int(vis_start) + row_of_full[t] for t in donor_full]
    host_pos = [kept_pos_of_full[t] for t in host_full]
    return {
        "donor_seq": donor_seq,
        "host_pos": host_pos,
        "n_hosts": len(kept_sorted),
        "donor_full": donor_full,
        "host_full": host_full,
        "host_seq": [int(vis_start) + row_of_full[t] for t in host_full],
        "grid_w": grid_w,
        "audit": audit,
    }


# ---------------------------------------------------------------------------
# COC: Certificate-Optimal Contraction (C v4)
# ---------------------------------------------------------------------------
#
# Motivation (plan "C 重设计"): in the error chain
#     err <= TV(kappa) * diam_v(groups)
# exact assignment already minimizes kappa; the untouched degree of freedom is
# the grouping itself and WHO survives as host — currently decided by BAM rank
# strata, blind to the contextualized value geometry the bound depends on.
# COC takes over stage-2 contraction: partition the S_cur rows (raster order)
# into k_hist contiguous blocks with sizes in {floor, ceil} (kappa stays
# minimal) minimizing the max within-block value dispersion (1D boundary DP on
# prefix sums), and keep each block's value medoid as the surviving host.
# Selection and recovery are thereby optimized against the same certificate.
# FA2-compatible, key-frozen, no extra tokens; absorption strength stays the
# single tau/lambda knob of apply_history_absorb.


def coc_enabled() -> bool:
    return os.environ.get("GEOPACT_C_COC", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def coc_layer_stride() -> int:
    """GEOPACT_C_COC_LAYER_STRIDE: sample every s-th layer for the value
    geometry (default 4 — 9 layers on a 36-layer model; the bound is
    per-layer, the grouping uses a scale-balanced average as a proxy)."""
    raw = os.environ.get("GEOPACT_C_COC_LAYER_STRIDE", "").strip()
    try:
        v = int(raw) if raw else 4
    except ValueError:
        v = 4
    return max(1, v)


def coc_value_features(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    vis_start: int,
    n_vis: int,
    layer_stride: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """[n_vis, H_kv*d] fp32 value-geometry features of the frame's cache rows.

    Per sampled layer the rows are scale-balanced (divided by the layer's mean
    row norm) before averaging, so no layer dominates the dispersion metric.
    """
    if not legacy_tensors or n_vis <= 0:
        return None
    stride = layer_stride or coc_layer_stride()
    feats: Optional[torch.Tensor] = None
    count = 0
    for li in range(0, len(legacy_tensors), stride):
        v = legacy_tensors[li][1]
        rows = v[0][:, int(vis_start) : int(vis_start) + int(n_vis), :].float()
        if rows.shape[1] != n_vis:
            return None
        f = rows.permute(1, 0, 2).reshape(n_vis, -1)  # [n, H_kv*d]
        scale = f.norm(dim=-1).mean().clamp(min=1e-6)
        f = f / scale
        feats = f if feats is None else feats + f
        count += 1
    if feats is None or count == 0:
        return None
    return feats / float(count)


def coc_partition_dp(
    feats: torch.Tensor, k: int
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Minimax boundary DP: split n rows into k contiguous blocks with sizes
    in {floor(n/k), ceil(n/k)} minimizing the max within-block SSE.

    With sizes fixed to two values, choosing boundaries == choosing which r of
    the k blocks take the extra row (n = q*k + r), so block i's start is
    i*q + (#ceil blocks before i): states (i, j) with j <= r, O(k*r)
    transitions, each O(1) via prefix sums. Returns [(start, size)] and audit
    (max/mean block SSE, the uniform-boundary max SSE for comparison).
    """
    n = int(feats.shape[0])
    k = max(1, min(int(k), n))
    q, r = divmod(n, k)
    feats = feats.detach().float().cpu()
    s1 = torch.cat([torch.zeros(1, feats.shape[1]), feats.cumsum(0)])
    sq = (feats * feats).sum(dim=-1)
    s2 = torch.cat([torch.zeros(1), sq.cumsum(0)])

    # Vectorized per-start segment SSE for the two legal sizes: the DP loop
    # below is then pure float lookups (O(k*r) Python ops, no tensor churn).
    def _sse_all(size: int) -> List[float]:
        if size <= 0 or size > n:
            return []
        if size == 1:
            return [0.0] * n
        starts = torch.arange(0, n - size + 1)
        tot = s2[starts + size] - s2[starts]
        mean_sq = ((s1[starts + size] - s1[starts]) ** 2).sum(dim=-1) / float(size)
        return (tot - mean_sq).clamp(min=0.0).tolist()

    sse_q = _sse_all(q)
    sse_q1 = _sse_all(q + 1)

    def sse(start: int, size: int) -> float:
        if size <= 1:
            return 0.0
        return sse_q[start] if size == q else sse_q1[start]

    inf = float("inf")
    # dp[j] after i blocks: minimal achievable max-SSE using j ceil blocks
    dp = [inf] * (r + 1)
    dp[0] = 0.0
    parent: List[List[int]] = []
    for i in range(k):
        nxt = [inf] * (r + 1)
        par = [-1] * (r + 1)
        for j in range(min(i, r) + 1):
            if dp[j] == inf:
                continue
            start = i * q + j
            # floor-size block (always legal if remaining ceil quota fits)
            if (r - j) <= (k - i - 1):
                c = max(dp[j], sse(start, q)) if q > 0 else dp[j]
                if q > 0 and c < nxt[j]:
                    nxt[j] = c
                    par[j] = j
            # ceil-size block
            if j < r:
                c = max(dp[j], sse(start, q + 1))
                if c < nxt[j + 1]:
                    nxt[j + 1] = c
                    par[j + 1] = j
        dp = nxt
        parent.append(par)
    # backtrack
    blocks: List[Tuple[int, int]] = []
    j = r
    for i in range(k - 1, -1, -1):
        pj = parent[i][j]
        size = (q + 1) if j != pj else q
        start = i * q + pj
        blocks.append((start, size))
        j = pj
    blocks.reverse()
    # uniform-boundary reference (first r blocks take the extra row)
    uni_max = 0.0
    pos = 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        uni_max = max(uni_max, sse(pos, size))
        pos += size
    sses = [sse(s, z) for s, z in blocks]
    audit = {
        "coc_sse_max": round(max(sses) if sses else 0.0, 4),
        "coc_sse_mean": round(sum(sses) / max(len(sses), 1), 4),
        "coc_sse_max_uniform": round(uni_max, 4),
    }
    return blocks, audit


def coc_pick_hosts(
    feats: torch.Tensor, blocks: Sequence[Tuple[int, int]]
) -> Tuple[List[int], Dict[int, int]]:
    """Per block: host = row closest to the block's value mean (medoid
    approximation); every other member becomes the host's donor.

    Returns (host rows ascending, donor row -> host row map).
    """
    hosts: List[int] = []
    host_of: Dict[int, int] = {}
    for start, size in blocks:
        if size <= 0:
            continue
        seg = feats[start : start + size]
        mean = seg.mean(dim=0, keepdim=True)
        host = start + int((seg - mean).norm(dim=-1).argmin())
        hosts.append(host)
        for rrow in range(start, start + size):
            if rrow != host:
                host_of[rrow] = host
    return sorted(hosts), host_of


def plan_coc_contract(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    vis_start: int,
    n_vis: int,
    scur_sorted: Sequence[int],
    k_hist: int,
) -> Optional[Dict[str, Any]]:
    """Full COC stage-2 plan from the live cache rows of one frame span.

    Returns {keep_full, host_map (donor full -> host full), audit} in
    full-frame coordinates, or None (bad span / degenerate budget).
    """
    n_vis = int(n_vis)
    if n_vis <= 0 or len(scur_sorted) != n_vis or int(k_hist) >= n_vis:
        return None
    feats = coc_value_features(
        legacy_tensors, vis_start=vis_start, n_vis=n_vis
    )
    if feats is None:
        return None
    blocks, audit = coc_partition_dp(feats, int(k_hist))
    host_rows, host_of_row = coc_pick_hosts(feats, blocks)
    if len(host_rows) != int(k_hist):
        return None
    full = [int(x) for x in scur_sorted]
    keep_full = sorted(full[rrow] for rrow in host_rows)
    host_map = {full[d]: full[h] for d, h in host_of_row.items()}
    audit["coc_blocks"] = len(blocks)
    return {"keep_full": keep_full, "host_map": host_map, "audit": audit}


def plan_from_host_map(
    *,
    keep_full: Sequence[int],
    scur_sorted: Sequence[int],
    vis_start: int,
    host_map: Dict[int, int],
) -> Optional[Dict[str, Any]]:
    """Absorption plan (same schema as plan_history_absorb) from an explicit
    donor->host map in full-frame coordinates (COC path)."""
    keep_set = {int(x) for x in keep_full}
    donor_full = sorted(int(d) for d in host_map if int(d) not in keep_set)
    if not donor_full:
        return None
    host_full = [int(host_map[d]) for d in donor_full]
    if any(h not in keep_set for h in host_full):
        return None
    row_of_full = {t: i for i, t in enumerate(int(x) for x in scur_sorted)}
    if any(d not in row_of_full or h not in row_of_full for d, h in zip(donor_full, host_full)):
        return None
    kept_sorted = sorted(keep_set)
    kept_pos_of_full = {t: p for p, t in enumerate(kept_sorted)}
    load: Dict[int, int] = {t: 0 for t in kept_sorted}
    for h in host_full:
        load[h] += 1
    masses = sorted(load.values())
    kappa = (1.0 + masses[-1]) / (1.0 + masses[0]) if masses else 1.0
    audit = {
        "absorb_assign": "coc",
        "absorb_donors": len(donor_full),
        "absorb_m_min": masses[0] if masses else 0,
        "absorb_m_max": masses[-1] if masses else 0,
        "absorb_kappa": round(kappa, 4),
    }
    return {
        "donor_seq": [int(vis_start) + row_of_full[d] for d in donor_full],
        "host_pos": [kept_pos_of_full[h] for h in host_full],
        "n_hosts": len(kept_sorted),
        "donor_full": donor_full,
        "host_full": host_full,
        "host_seq": [int(vis_start) + row_of_full[h] for h in host_full],
        "grid_w": 0,
        "audit": audit,
    }


# ---------------------------------------------------------------------------
# W-A3: entry embedding merge (independent ToMe-like extension arm)
# ---------------------------------------------------------------------------


def entry_merge_embeds(
    frame_embeds: torch.Tensor,
    keep_local: Set[int],
    n_frame: int,
    lam: float,
) -> Dict[str, Any]:
    """Merge dropped token *embeddings* into kept ones before the first forward.

    e~_i = (e_i + lam * sum_{j->i} e_j) / (1 + lam * m_i), balanced assignment
    by embedding cosine with capacity ceil(|D|/|S|). In-place on
    ``frame_embeds``. This is an empirical token-merging extension: it changes
    keys and the full network trajectory; it does not inherit the history
    value-only theory (spec v2 makes the split explicit).
    """
    if lam <= 0.0:
        return {"entry_merge_applied": False}
    kept = sorted(int(i) for i in keep_local)
    dropped = [i for i in range(int(n_frame)) if i not in keep_local]
    if not kept or not dropped:
        return {"entry_merge_applied": False}
    device = frame_embeds.device
    k_idx = torch.tensor(kept, dtype=torch.long, device=device)
    d_idx = torch.tensor(dropped, dtype=torch.long, device=device)
    e = frame_embeds.float()
    e_n = torch.nn.functional.normalize(e, dim=-1)
    sims = e_n.index_select(0, d_idx) @ e_n.index_select(0, k_idx).t()  # [D, S]
    cap = max(1, math.ceil(len(dropped) / len(kept)))
    # greedy balanced: donors in descending best-sim confidence, each takes its
    # most similar host with spare capacity.
    best_sim, _ = sims.max(dim=1)
    donor_order = torch.argsort(best_sim, descending=True).tolist()
    load = [0] * len(kept)
    host_of = [0] * len(dropped)
    sim_rank = torch.argsort(sims, dim=1, descending=True).tolist()
    for di in donor_order:
        placed = False
        for hj in sim_rank[di]:
            if load[hj] < cap:
                host_of[di] = hj
                load[hj] += 1
                placed = True
                break
        if not placed:  # all at cap — impossible with cap=ceil(|D|/|S|)
            hj = min(range(len(kept)), key=lambda x: load[x])
            host_of[di] = hj
            load[hj] += 1
    h_t = torch.tensor(host_of, dtype=torch.long, device=device)
    sums = torch.zeros(len(kept), e.shape[-1], dtype=torch.float32, device=device)
    sums.index_add_(0, h_t, e.index_select(0, d_idx))
    mass = torch.zeros(len(kept), dtype=torch.float32, device=device)
    mass.index_add_(0, h_t, torch.ones_like(h_t, dtype=torch.float32))
    merged = (e.index_select(0, k_idx) + lam * sums) / (1.0 + lam * mass.view(-1, 1))
    frame_embeds.index_copy_(0, k_idx, merged.to(frame_embeds.dtype))
    m_min, m_max = int(mass.min().item()), int(mass.max().item())
    kappa = float(1 + m_max) / float(1 + m_min)
    return {
        "entry_merge_applied": True,
        "entry_merge_donors": len(dropped),
        "entry_merge_hosts": len(kept),
        "entry_merge_kappa": round(kappa, 4),
        "entry_merge_lambda": float(lam),
    }
