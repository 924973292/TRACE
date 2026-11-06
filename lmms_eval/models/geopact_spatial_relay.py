"""GeoPACT Module C — QASMR spatial relays (spec: docs/geopact_module_c_qasmr.md).

Core, model-independent pieces of the query-adaptive spatial moment relay:

  * shared spatial partition of deleted tokens (GUI-constrained agglomeration
    with a computable error certificate, diameter cap and MIU hard borders);
  * per-layer / per-KV-head first- and second-order moment aggregation from a
    real post-MRoPE frame KV snapshot (C-Full / QASMR);
  * balanced-OT embedding relays (C-Lite / BOMR);
  * relay lifecycle containers (`FrameRelays`, `RelayCtx`) shared with the
    attention side-car in `geopact_relay_attention.py`.

Everything here is torch-only and CPU-testable. All paths are gated by
`GEOPACT_C_RELAY` (default off) and must not perturb baseline behaviour when
disabled.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def relay_enabled() -> bool:
    return _flag("GEOPACT_C_RELAY", "0")


def relay_mode() -> str:
    mode = os.environ.get("GEOPACT_C_RELAY_MODE", "kv1").strip().lower()
    if mode not in ("lite", "kv1", "kv2"):
        raise ValueError(f"GEOPACT_C_RELAY_MODE must be lite/kv1/kv2, got {mode!r}")
    return mode


def relay_lite_active() -> bool:
    return relay_enabled() and relay_mode() == "lite"


def relay_kv_mode_active() -> bool:
    return relay_enabled() and relay_mode() in ("kv1", "kv2")


def relay_from_cache_enabled() -> bool:
    """C reconstruction: build relays from the dialogue cache's real KV rows.

    The dropped tokens of a compressed frame were actually prefilled (stage 1
    superset under NAP; full frame under the legacy protocol), so their
    contextualized KV rows sit in the dialogue cache until the rewrite crop.
    Slicing those rows at transition time replaces the per-step full-frame
    ``layer_kv`` snapshot clone — the relay becomes a lazy compression of
    already-materialized evidence instead of a snapshot hack.
    """
    return _flag("GEOPACT_C_RELAY_FROM_CACHE", "0")


def relay_kv_snapshot_needed() -> bool:
    override = os.environ.get("GEOPACT_C_RELAY_KV_SNAPSHOT", "").strip()
    if override:
        return _flag("GEOPACT_C_RELAY_KV_SNAPSHOT")
    if relay_from_cache_enabled():
        return False
    return relay_kv_mode_active()


def relay_budget() -> int:
    return max(1, int(os.environ.get("GEOPACT_C_RELAY_R", "16")))


def relay_diam() -> float:
    return max(1e-3, float(os.environ.get("GEOPACT_C_RELAY_DIAM", "0.35")))


def relay_moment() -> str:
    raw = os.environ.get("GEOPACT_C_RELAY_MOMENT", "").strip().lower()
    if not raw or raw == "auto":
        return "diag" if relay_mode() == "kv2" else "none"
    if raw not in ("none", "diag", "lowrank"):
        raise ValueError(f"GEOPACT_C_RELAY_MOMENT must be none/diag/lowrank, got {raw!r}")
    return raw


def relay_rank() -> int:
    return max(1, int(os.environ.get("GEOPACT_C_RELAY_RANK", "2")))


def relay_max_total() -> int:
    return max(1, int(os.environ.get("GEOPACT_C_RELAY_MAX_TOTAL", "64")))


def relay_amin() -> float:
    return max(1e-6, float(os.environ.get("GEOPACT_C_RELAY_AMIN", "1e-3")))


def relay_audit_enabled() -> bool:
    return _flag("GEOPACT_C_RELAY_AUDIT", "0")


def relay_residual_enabled() -> bool:
    """B->C residual handoff: allocate cluster budget per MIU proportional to
    the uncovered selection quality rho_m (master spec §2.6)."""
    return _flag("GEOPACT_C_RELAY_RESID", "0")


def relay_layer_active(layer_idx: int) -> bool:
    raw = os.environ.get("GEOPACT_C_RELAY_LAYERS", "").strip()
    if not raw:
        return True
    try:
        allowed = {int(x) for x in raw.split(",") if x.strip() != ""}
    except ValueError:
        return True
    return int(layer_idx) in allowed


def relay_strict() -> bool:
    return _flag("GEOPACT_STRICT", "0")


# ---------------------------------------------------------------------------
# Error certificates (spec §4–§5)
# ---------------------------------------------------------------------------


def certificate_terms(
    eta: torch.Tensor,
    r_v: torch.Tensor,
    vbar_norm: torch.Tensor,
    *,
    order: int,
) -> torch.Tensor:
    """Per-cluster normalized remainder bound (Lemma 1 / Lemma 2).

    order=1: eps = |v̄|·½e^η η² + e^η η r_v  (+ ½e^η η² for the denominator,
    folded in via the (1+|v̄|) factor used by callers that want a single
    scalar). Here we return the *combined* scalar used for budgeting:
      order 1: (1+|v̄|)·½ e^η η²  + e^η η r_v
      order 2: (1+|v̄|)·(1/6) e^η η³ + ½ e^η η² r_v
    """
    eta = eta.clamp(min=0.0)
    e = torch.exp(eta)
    if order <= 1:
        return (1.0 + vbar_norm) * 0.5 * e * eta.pow(2) + e * eta * r_v
    return (1.0 + vbar_norm) * (e * eta.pow(3) / 6.0) + 0.5 * e * eta.pow(2) * r_v


def output_error_certificate(
    z_hat: torch.Tensor,
    o_hat_norm: torch.Tensor,
    e_z: torch.Tensor,
    e_n: torch.Tensor,
) -> torch.Tensor:
    """Theorem 1: ||o - ô|| <= (E_N + ||ô|| E_Z) / (Ẑ - E_Z), valid when Ẑ > E_Z."""
    denom = (z_hat - e_z).clamp(min=1e-30)
    return (e_n + o_hat_norm * e_z) / denom


@dataclass
class _Group:
    """Sufficient statistics for one candidate cluster during agglomeration."""

    idx: List[int]
    pos_sum: torch.Tensor          # [2]
    pos_min: torch.Tensor          # [2]
    pos_max: torch.Tensor          # [2]
    feat_sum: Optional[torch.Tensor]   # [F]
    radius_ub: float               # recursive upper bound on max||f - mean||
    miu: int

    @property
    def m(self) -> int:
        return len(self.idx)

    def mean_pos(self) -> torch.Tensor:
        return self.pos_sum / max(self.m, 1)

    def mean_feat(self) -> Optional[torch.Tensor]:
        if self.feat_sum is None:
            return None
        return self.feat_sum / max(self.m, 1)


def _group_cert(g: _Group, *, bq: float, tau: float) -> float:
    """Query-independent certificate proxy for a group (spec §5).

    Uses the recursive feature-radius upper bound as r_k; value terms are not
    tracked at partition time (exact per-layer/head certificates are computed
    after the partition is frozen), so the proxy is (1+0)·½e^η η² weighted by
    mass, plus a spatial spread penalty that keeps clusters compact even when
    the feature proxy is degenerate.
    """
    eta = bq * g.radius_ub / max(tau, 1e-9)
    eta = min(eta, 30.0)
    cert = g.m * 0.5 * math.exp(eta) * eta * eta
    spread = float((g.pos_max - g.pos_min).norm().item())
    return cert + 0.05 * g.m * spread * spread


def _merge_seed_stats_tensorized(
    bucket_id: torch.Tensor,
    n_buckets: int,
    pos32: torch.Tensor,
    feat32: Optional[torch.Tensor],
    miu_t: torch.Tensor,
    *,
    budget: int,
    diam_max: float,
    bq: float,
    tau: float,
) -> Tuple[torch.Tensor, int]:
    """Greedy smallest-certificate-increment merging, fully on ``pos32.device``.

    Same objective as the original per-pair loop: at each step merge the
    admissible same-MIU pair (merged bbox diagonal <= diam_max) with the
    smallest certificate delta; stop at the budget or when no admissible pair
    remains. The merged radius upper bound uses the exact identity
      ||mean_a - mean_new|| = m_b/(m_a+m_b) * ||mean_a - mean_b||,
    so a [G, G] feature-mean distance matrix (maintained incrementally)
    replaces any [G, G, F] tensor, and dead groups are masked instead of
    reallocating state — one host sync per merge (the argmin readout).

    Returns (per-token final group id in [0, n_buckets), n_merges). Group ids
    of merged-away buckets are remapped to their surviving root.
    """
    dev = pos32.device
    inf = float("inf")
    tau_s = max(tau, 1e-9)
    G = int(n_buckets)

    ones = torch.ones_like(bucket_id, dtype=torch.float32)
    m = torch.zeros(G, dtype=torch.float32, device=dev).index_add_(0, bucket_id, ones)
    pmin = torch.full((G, 2), inf, dtype=torch.float32, device=dev)
    pmax = torch.full((G, 2), -inf, dtype=torch.float32, device=dev)
    bid2 = bucket_id.view(-1, 1).expand(-1, 2)
    pmin.scatter_reduce_(0, bid2, pos32, reduce="amin", include_self=True)
    pmax.scatter_reduce_(0, bid2, pos32, reduce="amax", include_self=True)
    # per-bucket MIU label (buckets never mix labels by construction)
    miu = torch.zeros(G, dtype=torch.long, device=dev)
    miu.scatter_(0, bucket_id, miu_t)

    if feat32 is not None:
        F = int(feat32.shape[1])
        fsum = torch.zeros(G, F, dtype=torch.float32, device=dev)
        fsum.index_add_(0, bucket_id, feat32)
        fmean = fsum / m.view(-1, 1)
        dev_norm = (feat32 - fmean.index_select(0, bucket_id)).norm(dim=-1)
        radius = torch.zeros(G, dtype=torch.float32, device=dev)
        radius.scatter_reduce_(0, bucket_id, dev_norm, reduce="amax", include_self=True)
        dmat = torch.cdist(fmean, fmean)
    else:
        fsum = fmean = None
        radius = torch.zeros(G, dtype=torch.float32, device=dev)
        dmat = torch.zeros(G, G, dtype=torch.float32, device=dev)

    def cert_vec(mm: torch.Tensor, rr: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        eta = (float(bq) * rr / tau_s).clamp(min=0.0, max=30.0)
        return mm * 0.5 * torch.exp(eta) * eta * eta + 0.05 * mm * ss * ss

    alive = torch.ones(G, dtype=torch.bool, device=dev)
    parent = torch.arange(G, dtype=torch.long, device=dev)
    static_invalid = (miu.view(-1, 1) != miu.view(1, -1)) | torch.ones(
        G, G, dtype=torch.bool, device=dev
    ).tril()
    merges = 0
    n_alive = G
    while n_alive > budget:
        mn = m.view(-1, 1) + m.view(1, -1)
        pmin_n = torch.minimum(pmin.unsqueeze(1), pmin.unsqueeze(0))
        pmax_n = torch.maximum(pmax.unsqueeze(1), pmax.unsqueeze(0))
        diam_n = (pmax_n - pmin_n).norm(dim=-1)
        ra = radius.view(-1, 1) + (m.view(1, -1) / mn) * dmat
        rb = radius.view(1, -1) + (m.view(-1, 1) / mn) * dmat
        rad_n = torch.maximum(ra, rb)
        spread = (pmax - pmin).norm(dim=-1)
        cert_1 = cert_vec(m, radius, spread)
        delta = cert_vec(mn, rad_n, diam_n) - cert_1.view(-1, 1) - cert_1.view(1, -1)
        dead = ~alive
        invalid = (
            static_invalid
            | (diam_n > diam_max)
            | dead.view(-1, 1)
            | dead.view(1, -1)
        )
        delta = delta.masked_fill(invalid, inf)
        flat = torch.argmin(delta.view(-1))
        best = float(delta.view(-1)[flat])          # single host sync per merge
        if not math.isfinite(best):
            break
        idx = int(flat)
        i, j = idx // G, idx % G

        m[i] = m[i] + m[j]
        pmin[i] = torch.minimum(pmin[i], pmin[j])
        pmax[i] = torch.maximum(pmax[i], pmax[j])
        radius[i] = rad_n[i, j]
        alive[j] = False
        parent[j] = i
        # park the dead row far away so it can never win again
        pmin[j] = inf
        pmax[j] = -inf
        m[j] = 1.0
        if fsum is not None and fmean is not None:
            fsum[i] = fsum[i] + fsum[j]
            fmean[i] = fsum[i] / m[i]
            row = torch.cdist(fmean[i : i + 1], fmean)[0]
            dmat[i, :] = row
            dmat[:, i] = row
        merges += 1
        n_alive -= 1

    # resolve parent chains (paths have length <= merges)
    root = parent
    for _ in range(max(1, merges)):
        nxt = parent.index_select(0, root)
        if bool(torch.equal(nxt, root)):
            break
        root = nxt
    return root.index_select(0, bucket_id), merges


def spatial_partition(
    pos: torch.Tensor,
    *,
    budget: int,
    diam_max: float,
    miu_labels: Optional[Sequence[int]] = None,
    feat: Optional[torch.Tensor] = None,
    bq: float = 1.0,
    tau: float = 1.0,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Shared GUI-constrained spatial partition of deleted tokens (spec §2, §5).

    pos: [n, 2] normalized (y, x); feat: optional [n, F] proxy features used by
    the certificate during agglomeration (e.g. probe-layer keys or embeddings).
    Returns (clusters as LongTensors of token indices into ``pos``, audit).

    Two-stage certificate-driven agglomeration:
      1. seed groups = spatial grid cells of side diam_max/√2, refined by MIU
         label (clusters never cross MIU hard borders and satisfy the diameter
         cap by construction);
      2a. while #groups > budget: merge the pair with the smallest certificate
          increment among same-MIU pairs whose merged diameter stays <= cap;
          if no admissible pair exists, allow up to 25% soft over-budget;
      2b. while #groups < min(budget, n): split the group with the largest
          certificate via 2-means on positions (budget allocation by
          certificate reduction).
    """
    n = int(pos.shape[0])
    audit: Dict[str, Any] = {"relay_partition_n": n}
    if n == 0:
        return [], audit
    # Device-native: all stages run on pos.device (callers pass the KV-cache
    # GPU). The original CPU path shared the node's saturated cores with the
    # concurrent eval jobs and cost seconds per frame (wave1: 3-7s/frame).
    dev = pos.device
    pos32 = pos.detach().to(torch.float32)
    feat32 = (
        feat.detach().to(device=dev, dtype=torch.float32) if feat is not None else None
    )
    if miu_labels is not None:
        if len(miu_labels) != n:
            raise ValueError(f"miu_labels length {len(miu_labels)} != n {n}")
        miu_t = torch.as_tensor(
            [int(x) for x in miu_labels], dtype=torch.long, device=dev
        )
    else:
        miu_t = torch.zeros(n, dtype=torch.long, device=dev)

    # --- 1. seed groups: grid cells of side diam_max/sqrt(2) x MIU label ----
    cell = max(diam_max / math.sqrt(2.0), 1e-6)
    cy = torch.div(pos32[:, 0], cell, rounding_mode="floor").to(torch.long)
    cx = torch.div(pos32[:, 1], cell, rounding_mode="floor").to(torch.long)
    seed_key = torch.stack([cy, cx, miu_t], dim=1)
    _, bucket_id = torch.unique(seed_key, dim=0, return_inverse=True)
    n_buckets = int(bucket_id.max().item()) + 1
    audit["relay_partition_seed_groups"] = n_buckets

    # --- 2a. certificate-driven merging down to the budget -----------------
    # Hard constraints (MIU borders, diameter cap) always win: when no
    # admissible pair remains, the partition stays over budget (audited);
    # callers enforce the hard per-frame relay cap after the partition.
    soft_cap = max(budget, int(math.ceil(budget * 1.25)))
    if n_buckets > budget:
        root_id, merges = _merge_seed_stats_tensorized(
            bucket_id,
            n_buckets,
            pos32,
            feat32,
            miu_t,
            budget=budget,
            diam_max=diam_max,
            bq=bq,
            tau=tau,
        )
    else:
        root_id, merges = bucket_id, 0
    audit["relay_partition_merges"] = merges

    # materialize clusters once (single device->host transfer)
    _, group_of = torch.unique(root_id, return_inverse=True)
    group_list = group_of.cpu().tolist()
    n_groups = max(group_list) + 1
    members: List[List[int]] = [[] for _ in range(n_groups)]
    for tok, g in enumerate(group_list):
        members[g].append(tok)
    members = [idx for idx in members if idx]
    audit["relay_partition_over_budget"] = max(0, len(members) - soft_cap)

    # --- 2b. certificate-driven splitting up to the budget ------------------
    # Rare (only when the seed partition lands under budget); this is the only
    # stage that touches CPU group objects, so the common path stays on-device.
    splits = 0
    if len(members) < min(budget, n):
        pos_cpu = pos32.cpu()
        feat_cpu = feat32.cpu() if feat32 is not None else None
        labels_cpu = miu_t.cpu().tolist()

        def make_group(idx: List[int]) -> _Group:
            p = pos_cpu[idx]
            f_sum = feat_cpu[idx].sum(0) if feat_cpu is not None else None
            radius = 0.0
            if feat_cpu is not None:
                mean = f_sum / len(idx)
                radius = float((feat_cpu[idx] - mean).norm(dim=-1).max().item())
            return _Group(
                idx=list(idx),
                pos_sum=p.sum(0),
                pos_min=p.min(0).values,
                pos_max=p.max(0).values,
                feat_sum=f_sum,
                radius_ub=radius,
                miu=labels_cpu[idx[0]],
            )

        groups: List[_Group] = [make_group(idx) for idx in members]
        while len(groups) < min(budget, n):
            order = sorted(
                range(len(groups)),
                key=lambda k: _group_cert(groups[k], bq=bq, tau=tau),
                reverse=True,
            )
            split_done = False
            for k in order:
                g = groups[k]
                if g.m < 2:
                    continue
                sub = _two_means_split(pos_cpu[g.idx])
                left = [g.idx[t] for t in range(g.m) if sub[t] == 0]
                right = [g.idx[t] for t in range(g.m) if sub[t] == 1]
                if not left or not right:
                    continue
                groups = [gg for kk, gg in enumerate(groups) if kk != k]
                groups.append(make_group(left))
                groups.append(make_group(right))
                splits += 1
                split_done = True
                break
            if not split_done:
                break
        members = [sorted(g.idx) for g in groups]
    audit["relay_partition_splits"] = splits
    audit["relay_partition_final_groups"] = len(members)

    clusters = [torch.tensor(sorted(idx), dtype=torch.long) for idx in members]
    clusters.sort(key=lambda t: int(t[0]))
    # max diameter audit on-device (single readout)
    gid = torch.cat(
        [
            torch.full((len(c),), ci, dtype=torch.long, device=dev)
            for ci, c in enumerate(clusters)
        ]
    )
    tok = torch.cat([c.to(dev) for c in clusters])
    p_sel = pos32.index_select(0, tok)
    gmin = torch.full((len(clusters), 2), float("inf"), dtype=torch.float32, device=dev)
    gmax = torch.full((len(clusters), 2), -float("inf"), dtype=torch.float32, device=dev)
    gid2 = gid.view(-1, 1).expand(-1, 2)
    gmin.scatter_reduce_(0, gid2, p_sel, reduce="amin", include_self=True)
    gmax.scatter_reduce_(0, gid2, p_sel, reduce="amax", include_self=True)
    audit["relay_partition_max_diam"] = float((gmax - gmin).norm(dim=-1).max())
    return clusters, audit


def _two_means_split(p: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """2-means on [m,2] positions; returns {0,1} assignment (deterministic)."""
    m = int(p.shape[0])
    if m < 2:
        return torch.zeros(m, dtype=torch.long)
    # deterministic init: the two mutually farthest points
    d = torch.cdist(p, p)
    flat = int(torch.argmax(d).item())
    c0, c1 = flat // m, flat % m
    centers = torch.stack([p[c0], p[c1]])
    assign = torch.zeros(m, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(p, centers).argmin(dim=1)
        if int(assign.sum()) in (0, m):
            break
        new_centers = torch.stack(
            [p[assign == 0].mean(0), p[assign == 1].mean(0)]
        )
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    if int(assign.sum()) in (0, m):
        assign = torch.zeros(m, dtype=torch.long)
        assign[c1] = 1
    return assign


# ---------------------------------------------------------------------------
# Frame KV snapshot + per-layer/head moment aggregation (C-Full)
# ---------------------------------------------------------------------------


def capture_frame_kv_snapshot(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    vis_start: int,
    vis_end: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Slice per-layer post-MRoPE (K, V) for the frame span from a prompt KV.

    legacy_tensors: per-layer (k, v) with shape [1, Hkv, seq, d].
    Returns {"k": [L, Hkv, n, d], "v": [L, Hkv, n, dv]} detached clones, or
    None when the cache is unavailable / span invalid.
    """
    if not legacy_tensors:
        return None
    vs, ve = int(vis_start), int(vis_end)
    if ve <= vs:
        return None
    seq = int(legacy_tensors[0][0].shape[-2])
    if ve > seq:
        return None
    ks, vals = [], []
    for k, v in legacy_tensors:
        ks.append(k[0, :, vs:ve, :].detach())
        vals.append(v[0, :, vs:ve, :].detach())
    return {"k": torch.stack(ks, dim=0).clone(), "v": torch.stack(vals, dim=0).clone()}


@dataclass
class FrameRelays:
    """Per-frame relay objects for the side-car attention stream.

    Tensor layout (L layers, H KV heads, R clusters):
      kbar   [L, H, R, d]    post-MRoPE key centroids (fp32)
      vbar   [L, H, R, dv]   value centroids (fp32)
      logm   [R]             log cluster mass (shared across layers/heads)
      kk_diag/vk_diag        optional [L, H, R, d] diagonal second moments
      vk_u   [L, H, R, dv, r] / vk_w [L, H, R, d, r]  optional low-rank Σ^{vk}
      cert   [L, H, R]       per-layer/head certified remainder (audit/budget)
    """

    img_path: str
    frame_seq: int
    kbar: torch.Tensor
    vbar: torch.Tensor
    logm: torch.Tensor
    kk_diag: Optional[torch.Tensor] = None
    vk_diag: Optional[torch.Tensor] = None
    vk_u: Optional[torch.Tensor] = None
    vk_w: Optional[torch.Tensor] = None
    cert: Optional[torch.Tensor] = None
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_relays(self) -> int:
        return int(self.kbar.shape[2])

    def to(self, device: torch.device) -> "FrameRelays":
        if self.kbar.device == device:
            return self
        moved = FrameRelays(
            img_path=self.img_path,
            frame_seq=self.frame_seq,
            kbar=self.kbar.to(device),
            vbar=self.vbar.to(device),
            logm=self.logm.to(device),
            kk_diag=self.kk_diag.to(device) if self.kk_diag is not None else None,
            vk_diag=self.vk_diag.to(device) if self.vk_diag is not None else None,
            vk_u=self.vk_u.to(device) if self.vk_u is not None else None,
            vk_w=self.vk_w.to(device) if self.vk_w is not None else None,
            cert=self.cert.to(device) if self.cert is not None else None,
            audit=self.audit,
        )
        return moved


@dataclass
class RelayCtx:
    """All active relays for the episode (installed on the language model)."""

    frames: List[FrameRelays] = field(default_factory=list)
    mode: str = "kv1"
    amin: float = 1e-3
    _layer_cache: Dict[Tuple[int, str], Dict[str, Optional[torch.Tensor]]] = field(
        default_factory=dict, repr=False
    )

    @property
    def total_relays(self) -> int:
        return sum(f.n_relays for f in self.frames)

    def invalidate(self) -> None:
        self._layer_cache.clear()

    def layer_tensors(
        self, layer_idx: int, device: torch.device, *, groups: int = 1
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Concatenated per-layer relay tensors across frames (cached).

        ``groups`` > 1 pre-broadcasts the KV-head tensors to the query heads
        (GQA repeat_interleave) once at build time so the per-token decode
        path spends zero kernels on broadcasting.
        """
        if not self.frames:
            return None
        key = (int(layer_idx), str(device), int(groups))
        hit = self._layer_cache.get(key)
        if hit is not None:
            return hit

        def cat(attr: str) -> Optional[torch.Tensor]:
            parts = []
            for f in self.frames:
                t = getattr(f, attr)
                if t is None:
                    return None
                parts.append(t[layer_idx].to(device))
            out = torch.cat(parts, dim=1)  # [Hkv, R_total, ...]
            if groups > 1:
                out = out.repeat_interleave(int(groups), dim=0)  # [Hq, ...]
            return out.contiguous()

        logm = torch.cat([f.logm.to(device) for f in self.frames], dim=0)
        kbar = cat("kbar")
        out: Dict[str, Optional[torch.Tensor]] = {
            "kbar": kbar,
            # decode fast path: k̄ᵀ pre-transposed for a single matmul logit
            "kbar_t": kbar.transpose(-1, -2).contiguous() if kbar is not None else None,
            "vbar": cat("vbar"),
            "logm": logm,
            "kk_diag": cat("kk_diag"),
            "vk_diag": cat("vk_diag"),
            "vk_u": cat("vk_u"),
            "vk_w": cat("vk_w"),
        }
        while len(self._layer_cache) > 256:
            self._layer_cache.pop(next(iter(self._layer_cache)))
        self._layer_cache[key] = out
        return out


def build_kv_relays(
    k_frame: torch.Tensor,
    v_frame: torch.Tensor,
    clusters: Sequence[torch.Tensor],
    *,
    moment: str = "none",
    rank: int = 2,
    bq: float = 1.0,
    img_path: str = "",
    frame_seq: int = -1,
    order: Optional[int] = None,
) -> FrameRelays:
    """Aggregate first/second moments per layer/head over a shared partition.

    k_frame/v_frame: [L, H, n, d] fp-any (converted to fp32).
    clusters: LongTensors of frame-local token indices (disjoint).
    """
    if moment not in ("none", "diag", "lowrank"):
        raise ValueError(f"moment must be none/diag/lowrank, got {moment!r}")
    k32 = k_frame.detach().to(torch.float32)
    v32 = v_frame.detach().to(torch.float32)
    L, H, n, d = k32.shape
    dv = int(v32.shape[-1])
    tau = math.sqrt(float(d))
    R = len(clusters)
    if R == 0:
        raise ValueError("build_kv_relays needs at least one cluster")
    dev = k32.device

    cert_order = int(order) if order is not None else (2 if moment != "none" else 1)
    sizes = [int(c.numel()) for c in clusters]
    if any(s == 0 for s in sizes):
        raise ValueError("build_kv_relays: empty cluster")

    # Vectorized segment aggregation (single pass, no per-cluster Python loop,
    # no host syncs): moments via index_add over a flat cluster-label vector,
    # per-cluster max deviation norms via scatter_reduce amax. The previous
    # per-cluster loop cost O(R) kernel launches * L*H tensors and dominated
    # the wave1 relay build time together with the partition merge loop.
    sel = torch.cat([c.to(dev) for c in clusters])           # [S]
    labels_r = torch.cat(
        [torch.full((s,), ci, dtype=torch.long, device=dev) for ci, s in enumerate(sizes)]
    )
    S = int(sel.numel())
    m_vec = torch.tensor(sizes, dtype=torch.float32, device=dev)
    k_sel = k32.index_select(2, sel)                          # [L,H,S,d]
    v_sel = v32.index_select(2, sel)                          # [L,H,S,dv]

    kbar = torch.zeros(L, H, R, d, dtype=torch.float32, device=dev)
    vbar = torch.zeros(L, H, R, dv, dtype=torch.float32, device=dev)
    kbar.index_add_(2, labels_r, k_sel)
    vbar.index_add_(2, labels_r, v_sel)
    kbar /= m_vec.view(1, 1, R, 1)
    vbar /= m_vec.view(1, 1, R, 1)
    logm = torch.log(m_vec)

    dk = k_sel - kbar.index_select(2, labels_r)               # centered [L,H,S,d]
    dvv = v_sel - vbar.index_select(2, labels_r)

    kk_diag = vk_diag = None
    vk_u = vk_w = None
    if moment in ("diag", "lowrank"):
        kk_diag = torch.zeros(L, H, R, d, dtype=torch.float32, device=dev)
        kk_diag.index_add_(2, labels_r, dk.pow(2))
        kk_diag /= m_vec.view(1, 1, R, 1)
    if moment == "diag":
        vk_diag = torch.zeros(L, H, R, d, dtype=torch.float32, device=dev)
        vk_diag.index_add_(2, labels_r, dvv * dk)
        vk_diag /= m_vec.view(1, 1, R, 1)
    elif moment == "lowrank":
        # Low-rank Sigma^{vk} needs a per-cluster SVD — kept as a loop (kv2
        # lowrank is an ablation-only mode, never on the wave production path).
        r = max(1, int(rank))
        vk_u = torch.zeros(L, H, R, dv, r, dtype=torch.float32, device=dev)
        vk_w = torch.zeros(L, H, R, d, r, dtype=torch.float32, device=dev)
        for c in range(R):
            rows = (labels_r == c).nonzero(as_tuple=True)[0]
            mc = int(rows.numel())
            if mc < 2:
                continue
            dv_c = dvv.index_select(2, rows)
            dk_c = dk.index_select(2, rows)
            sig = torch.einsum("lhmv,lhmk->lhvk", dv_c, dk_c) / float(mc)
            r_eff = min(int(rank), min(dv, d))
            u, s, vh = torch.linalg.svd(sig.reshape(L * H, dv, d), full_matrices=False)
            s_r = s[:, :r_eff].clamp(min=0.0).sqrt()
            u_r = u[:, :, :r_eff] * s_r.unsqueeze(1)
            w_r = vh[:, :r_eff, :].transpose(1, 2) * s_r.unsqueeze(1)
            vk_u[:, :, c, :, :r_eff] = u_r.reshape(L, H, dv, r_eff)
            vk_w[:, :, c, :, :r_eff] = w_r.reshape(L, H, d, r_eff)

    lab_lh = labels_r.view(1, 1, S).expand(L, H, S)
    r_k = torch.zeros(L, H, R, dtype=torch.float32, device=dev)
    r_v = torch.zeros(L, H, R, dtype=torch.float32, device=dev)
    r_k.scatter_reduce_(2, lab_lh, dk.norm(dim=-1), reduce="amax", include_self=False)
    r_v.scatter_reduce_(2, lab_lh, dvv.norm(dim=-1), reduce="amax", include_self=False)
    eta = (bq * r_k / tau).clamp(max=30.0)
    cert = m_vec.view(1, 1, R) * certificate_terms(
        eta, r_v, vbar.norm(dim=-1), order=cert_order
    )

    audit = {
        "relay_kv_n_relays": R,
        "relay_kv_moment": moment,
        "relay_kv_cert_sum": float(cert.sum().item()),
        "relay_kv_cert_per_layer": [float(x) for x in cert.sum(dim=(1, 2)).tolist()],
        # Certificate calibration diagnostics (H5): fraction of (layer, head,
        # cluster) cells whose eta hit the clamp (saturated bound carries no
        # ordering information) + the pre-clamp spread.
        "relay_cert_eta_sat_frac": round(float((eta >= 29.999).float().mean()), 4),
        "relay_cert_eta_mean": round(float(eta.mean()), 4),
        "relay_cert_eta_max": round(float(eta.max()), 4),
    }
    return FrameRelays(
        img_path=img_path,
        frame_seq=int(frame_seq),
        kbar=kbar,
        vbar=vbar,
        logm=logm,
        kk_diag=kk_diag,
        vk_diag=vk_diag,
        vk_u=vk_u,
        vk_w=vk_w,
        cert=cert,
        audit=audit,
    )


def allocate_relay_budget(certs: Sequence[float], budget: int) -> List[int]:
    """Greedy certified-error-per-relay allocation (spec §5, report utility).

    certs[i] is the current certificate of group i; each extra relay assigned
    to a group is modelled as halving its certificate (concave surrogate).
    Returns per-group relay counts summing to ``budget`` (each >= 1).
    """
    g = len(certs)
    if g == 0:
        return []
    counts = [1] * g
    remaining = max(0, int(budget) - g)
    cur = [float(c) for c in certs]
    for _ in range(remaining):
        best = max(range(g), key=lambda i: cur[i])
        cur[best] *= 0.5
        counts[best] += 1
    return counts


# ---------------------------------------------------------------------------
# C-Lite / BOMR: balanced-OT embedding relays (spec §7)
# ---------------------------------------------------------------------------


def build_lite_relays(
    embeds: torch.Tensor,
    pos: torch.Tensor,
    deleted_flat_idx: Sequence[int],
    occupied: Set[int],
    *,
    n_relays: int,
    deepstack: Optional[Sequence[torch.Tensor]] = None,
    lam_p: float = 4.0,
    lam_x: float = 1.0,
    outer_iters: int = 6,
    sinkhorn_iters: int = 30,
    eps: float = 0.05,
) -> Tuple[List[int], torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor, Dict[str, Any]]:
    """Balanced-OT relays over deleted embedding rows.

    embeds: [nD, hidden] deleted rows; pos: [nD, 2] normalized (y, x);
    deleted_flat_idx: original frame-local flat indices of the deleted rows;
    occupied: survivor flat indices (relay grid slots must avoid them).

    Returns (relay_flat_indices, relay_embeds [R,hidden],
    relay_deepstack or None, pi [nD,R] row-stochastic, audit).
    """
    nD = int(embeds.shape[0])
    if nD == 0:
        raise ValueError("build_lite_relays: no deleted tokens")
    R = max(1, min(int(n_relays), nD))
    dev = embeds.device
    x = embeds.detach().to(device=dev, dtype=torch.float32)
    p = pos.detach().to(device=dev, dtype=torch.float32)
    audit: Dict[str, Any] = {"relay_lite_nD": nD, "relay_lite_R": R}

    if R == nD:
        pi = torch.eye(nD, dtype=torch.float32, device=dev)
        flat = [int(i) for i in deleted_flat_idx]
        ds_out = [layer.detach().clone() for layer in deepstack] if deepstack else None
        audit["relay_lite_identity"] = True
        return flat, x.clone().to(embeds.dtype), ds_out, pi, audit

    # k-means++-style deterministic init on positions.
    centers_idx = [0]
    d2 = (p - p[0]).pow(2).sum(-1)
    for _ in range(R - 1):
        j = int(torch.argmax(d2).item())
        centers_idx.append(j)
        d2 = torch.minimum(d2, (p - p[j]).pow(2).sum(-1))
    b = p[centers_idx].clone()          # [R,2] spatial barycenters
    r_emb = x[centers_idx].clone()      # [R,hidden]

    x_n = torch.nn.functional.normalize(x, dim=-1)
    log_mu = torch.full((nD,), -math.log(nD), device=dev, dtype=torch.float32)
    log_nu = torch.full((R,), -math.log(R), device=dev, dtype=torch.float32)
    pi = torch.full((nD, R), 1.0 / (nD * R), device=dev, dtype=torch.float32)

    for _ in range(outer_iters):
        r_n = torch.nn.functional.normalize(r_emb, dim=-1)
        cost = lam_p * torch.cdist(p, b).pow(2) + lam_x * (1.0 - x_n @ r_n.t())
        logk = -cost / max(eps, 1e-6)
        f = torch.zeros(nD, device=dev, dtype=torch.float32)
        g = torch.zeros(R, device=dev, dtype=torch.float32)
        # End on the f (row) update: exact row marginals are what the
        # zeroth-order mean-preservation identity needs (each deleted token
        # distributes exactly 1/nD of mass); column marginals converge to
        # |D|/R approximately and are audited below.
        for _ in range(sinkhorn_iters):
            g = log_nu - torch.logsumexp(logk + f.unsqueeze(1), dim=0)
            f = log_mu - torch.logsumexp(logk + g.unsqueeze(0), dim=1)
        pi = torch.exp(f.unsqueeze(1) + logk + g.unsqueeze(0))  # sums to 1
        col = pi.sum(0).clamp(min=1e-12)
        b_new = (pi.t() @ p) / col.unsqueeze(1)
        r_new = (pi.t() @ x) / col.unsqueeze(1)
        if torch.allclose(b_new, b, atol=1e-6) and torch.allclose(r_new, r_emb, atol=1e-5):
            b, r_emb = b_new, r_new
            break
        b, r_emb = b_new, r_new

    # Row-stochastic transport (each deleted token distributes weight 1).
    pi_row = pi / pi.sum(1, keepdim=True).clamp(min=1e-12)
    col_mass = pi.sum(0) * nD  # expected |D|/R per column
    audit["relay_lite_col_mass_max_dev"] = float(
        (col_mass - nD / R).abs().max().item()
    )

    # Snap each barycenter to the nearest *free* deleted grid slot.
    flat_idx = [int(i) for i in deleted_flat_idx]
    taken: Set[int] = set(int(i) for i in occupied)
    relay_flat: List[int] = []
    d_slots = torch.cdist(b, p)  # [R, nD]
    for a in range(R):
        order = torch.argsort(d_slots[a]).tolist()
        chosen = None
        for j in order:
            cand = flat_idx[j]
            if cand not in taken:
                chosen = cand
                break
        if chosen is None:
            raise RuntimeError("build_lite_relays: no free grid slot for relay")
        taken.add(chosen)
        relay_flat.append(chosen)

    # Barycentric aggregation with column-normalized weights (zeroth-order
    # embedding mean preserved exactly: sum_a col_a * r_a == mean over D).
    w_col = pi / pi.sum(0, keepdim=True).clamp(min=1e-12)  # [nD,R], cols sum 1
    relay_embeds = (w_col.t() @ x).to(embeds.dtype)
    ds_out: Optional[List[torch.Tensor]] = None
    if deepstack:
        ds_out = []
        for layer in deepstack:
            l32 = layer.detach().to(torch.float32)
            ds_out.append((w_col.t() @ l32).to(layer.dtype))

    # Zeroth-order mean check: col-mass-weighted relay mean == mean over D
    # exactly (rows of pi sum to 1/nD by the balanced marginals).
    col = pi.sum(0)
    relay_mean = (col.unsqueeze(1) * (w_col.t() @ x)).sum(0) / col.sum().clamp(min=1e-12)
    audit["relay_lite_mean_preserved_err"] = float(
        (relay_mean - x.mean(0)).norm().item()
        / max(float(x.mean(0).norm().item()), 1e-9)
    )
    return relay_flat, relay_embeds, ds_out, pi_row, audit


# ---------------------------------------------------------------------------
# Lifecycle: capture -> build -> install -> rotate/clear
# ---------------------------------------------------------------------------


def _language_model(inner) -> Any:
    return getattr(inner, "language_model", inner)


def _residual_bucket_budgets(
    rho: Dict[int, float],
    bucket_sizes: Dict[int, int],
    budget: int,
) -> Dict[int, int]:
    """Largest-remainder split of the cluster budget across MIU buckets by
    uncovered quality rho_m; every non-empty bucket gets >= 1 cluster and no
    bucket gets more clusters than it has tokens."""
    buckets = sorted(bucket_sizes)
    n_b = len(buckets)
    if n_b == 0:
        return {}
    budget = max(n_b, int(budget))
    weights = {b: max(0.0, float(rho.get(b, 0.0))) for b in buckets}
    total = sum(weights.values())
    if total <= 0.0:
        weights = {b: float(bucket_sizes[b]) for b in buckets}
        total = sum(weights.values())
    raw = {b: budget * weights[b] / total for b in buckets}
    out = {b: max(1, int(raw[b])) for b in buckets}
    out = {b: min(out[b], int(bucket_sizes[b])) for b in buckets}
    remaining = budget - sum(out.values())
    if remaining > 0:
        order = sorted(buckets, key=lambda b: raw[b] - int(raw[b]), reverse=True)
        i = 0
        while remaining > 0 and any(out[b] < bucket_sizes[b] for b in buckets):
            b = order[i % len(order)]
            if out[b] < bucket_sizes[b]:
                out[b] += 1
                remaining -= 1
            i += 1
    return out


def build_frame_relays_from_capture(
    cap: Dict[str, Any],
    *,
    local_keep: Sequence[int],
    n_frame: int,
    grid_h: int,
    grid_w: int,
    miu_labels: Optional[Sequence[int]] = None,
) -> Optional[FrameRelays]:
    """Build C-Full relays for one compressed frame from its KV snapshot.

    When the frame was selected by the QD selector, ``cap['qd_export']``
    carries the B->C handoff (per-token MIU labels + per-MIU uncovered
    quality rho). Labels always sharpen the partition's MIU hard borders;
    with GEOPACT_C_RELAY_RESID=1 the cluster budget is additionally split
    across MIUs proportional to rho (master spec §2.6).
    """
    snap = cap.get("layer_kv")
    if snap is None:
        return None
    k_frame = snap["k"]
    v_frame = snap["v"]
    if int(k_frame.shape[2]) != int(n_frame):
        raise RuntimeError(
            "GeoPACT relay: KV snapshot span mismatch "
            f"snap={int(k_frame.shape[2])} n_frame={n_frame}"
        )
    keep = set(int(i) for i in local_keep)
    deleted = [i for i in range(int(n_frame)) if i not in keep]
    if not deleted:
        return None

    qd_export = cap.get("qd_export") or {}
    if miu_labels is None:
        qd_lab = qd_export.get("miu_labels")
        if qd_lab is not None and int(getattr(qd_lab, "numel", lambda: 0)()) == int(n_frame):
            miu_labels = [int(x) for x in qd_lab.tolist()]

    dev = k_frame.device
    del_dev = torch.tensor(deleted, dtype=torch.long, device=dev)
    pos = _grid_positions(n_frame, grid_h, grid_w).to(dev)[del_dev]
    # Proxy features for partition certificates: keys of a mid-stack probe
    # layer, flattened over heads (cheap, shared across layers by design).
    # Kept on the KV device — the partition is device-native end to end.
    L = int(k_frame.shape[0])
    probe = k_frame[min(L // 2, L - 1)].index_select(1, del_dev)
    feat = probe.permute(1, 0, 2).reshape(len(deleted), -1).to(torch.float32)
    labels = [miu_labels[i] for i in deleted] if miu_labels is not None else None

    clusters, part_audit, resid_audit, dropped_clusters, dropped_mass_frac = (
        _partition_deleted_rows(
            pos,
            feat,
            labels,
            qd_export.get("rho") if relay_residual_enabled() else None,
            tau=math.sqrt(float(k_frame.shape[-1])),
            dev=dev,
        )
    )
    if not clusters:
        return None

    del_t = torch.tensor(deleted, dtype=torch.long)
    global_clusters = [del_t[c] for c in clusters]

    fr = build_kv_relays(
        k_frame,
        v_frame,
        global_clusters,
        moment=relay_moment(),
        rank=relay_rank(),
        img_path=str(cap.get("img_path") or ""),
        frame_seq=int(cap.get("frame_seq", -1)),
    )
    fr.audit.update(part_audit)
    if resid_audit:
        fr.audit.update(resid_audit)
    fr.audit["relay_kv_deleted"] = len(deleted)
    fr.audit["relay_dropped_clusters"] = dropped_clusters
    fr.audit["relay_dropped_mass_frac"] = round(dropped_mass_frac, 4)
    return fr


def _partition_deleted_rows(
    pos: torch.Tensor,
    feat: torch.Tensor,
    labels: Optional[Sequence[int]],
    rho: Optional[Dict[int, float]],
    *,
    tau: float,
    dev: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, Any], Dict[str, Any], int, float]:
    """Shared deleted-row partition + hard relay cap (snapshot & cache paths).

    Returns (clusters as local row-index LongTensors, partition audit,
    residual audit, dropped cluster count, dropped mass fraction).
    """
    m = int(pos.shape[0])
    resid_audit: Dict[str, Any] = {}
    if rho and labels is not None:
        # Residual-weighted partition: per-MIU sub-partitions with budgets
        # proportional to the uncovered selection quality of each bucket.
        bucket_rows: Dict[int, List[int]] = {}
        for row, lab in enumerate(labels):
            bucket_rows.setdefault(int(lab), []).append(row)
        budgets = _residual_bucket_budgets(
            {int(k): float(v) for k, v in rho.items()},
            {b: len(rows) for b, rows in bucket_rows.items()},
            relay_budget(),
        )
        clusters: List[torch.Tensor] = []
        part_audit: Dict[str, Any] = {"relay_partition_n": m}
        for b in sorted(bucket_rows):
            rows = bucket_rows[b]
            rows_t = torch.tensor(rows, dtype=torch.long)
            rows_dev = rows_t.to(dev)
            sub_clusters, sub_audit = spatial_partition(
                pos.index_select(0, rows_dev),
                budget=int(budgets.get(b, 1)),
                diam_max=relay_diam(),
                miu_labels=[b] * len(rows),
                feat=feat.index_select(0, rows_dev),
                tau=tau,
            )
            clusters.extend(rows_t[c] for c in sub_clusters)
        part_audit["relay_partition_final_groups"] = len(clusters)
        resid_audit = {
            "relay_resid_buckets": len(bucket_rows),
            "relay_resid_budgets": {int(b): int(v) for b, v in budgets.items()},
        }
    else:
        clusters, part_audit = spatial_partition(
            pos,
            budget=relay_budget(),
            diam_max=relay_diam(),
            miu_labels=labels,
            feat=feat,
            tau=tau,
        )
    if not clusters:
        return [], part_audit, resid_audit, 0, 0.0

    # Hard per-frame relay cap (protocol freeze: R'=GEOPACT_C_RELAY_R). The
    # diameter/MIU hard constraints can leave the partition over budget
    # (wave1: mean 53 / max 113 clusters at R=16), which blows up both the
    # build cost and the decode-path relay stream. Keep the highest-mass
    # clusters (mass dominates the relay logit via log m); dropped clusters
    # simply stay unrelayed — the relay is best-effort compensation, so this
    # degrades coverage, never correctness. Dropped mass share is audited.
    cap_r = relay_budget()
    dropped_clusters = 0
    dropped_mass_frac = 0.0
    if len(clusters) > cap_r:
        sizes = [int(c.numel()) for c in clusters]
        order = sorted(
            range(len(clusters)),
            key=lambda k: (-sizes[k], int(clusters[k][0])),
        )
        kept_ids = sorted(order[:cap_r], key=lambda k: int(clusters[k][0]))
        total_mass = float(sum(sizes))
        kept_mass = float(sum(sizes[k] for k in kept_ids))
        dropped_clusters = len(clusters) - cap_r
        dropped_mass_frac = 1.0 - kept_mass / max(total_mass, 1.0)
        clusters = [clusters[k] for k in kept_ids]
    return clusters, part_audit, resid_audit, dropped_clusters, dropped_mass_frac


def build_frame_relays_from_cache_rows(
    legacy_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    seq_positions: Sequence[int],
    full_positions: Sequence[int],
    n_full: int,
    grid_h: int,
    grid_w: int,
    img_path: str = "",
    frame_seq: int = -1,
    qd_export: Optional[Dict[str, Any]] = None,
) -> Optional[FrameRelays]:
    """C reconstruction: relays from the dropped tokens' real cache KV rows.

    The dropped tokens were materialized in the dialogue cache by the stage-1
    prefill (NAP middle layer S_cur\\S_hist, or the full frame under the
    legacy protocol). At transition time — before the rewrite crop — their
    per-layer post-MRoPE (K, V) rows are sliced directly from the cache and
    aggregated into relay moments, replacing the full-frame snapshot clone.

    legacy_tensors: per-layer (k, v) with shape [1, Hkv, seq, d].
    seq_positions:  absolute cache positions of the dropped rows.
    full_positions: full-frame flat indices of the same rows (grid geometry).
    """
    if not legacy_tensors or not seq_positions:
        return None
    if len(seq_positions) != len(full_positions):
        raise RuntimeError(
            "GeoPACT relay cache rows: seq/full position length mismatch "
            f"{len(seq_positions)} vs {len(full_positions)}"
        )
    seq_len = int(legacy_tensors[0][0].shape[-2])
    if max(int(p) for p in seq_positions) >= seq_len:
        raise RuntimeError(
            "GeoPACT relay cache rows: position beyond cache length "
            f"max={max(int(p) for p in seq_positions)} seq={seq_len}"
        )
    dev = legacy_tensors[0][0].device
    sel = torch.tensor([int(p) for p in seq_positions], dtype=torch.long, device=dev)
    ks, vs = [], []
    for k, v in legacy_tensors:
        ks.append(k[0].index_select(1, sel.to(k.device)).detach())
        vs.append(v[0].index_select(1, sel.to(v.device)).detach())
    k_del = torch.stack(ks, dim=0)  # [L, Hkv, m, d]
    v_del = torch.stack(vs, dim=0)

    full_t = torch.tensor([int(p) for p in full_positions], dtype=torch.long)
    pos = _grid_positions(int(n_full), int(grid_h), int(grid_w)).to(dev)[
        full_t.to(dev)
    ]
    L = int(k_del.shape[0])
    probe = k_del[min(L // 2, L - 1)]
    feat = probe.permute(1, 0, 2).reshape(int(sel.numel()), -1).to(torch.float32)

    labels: Optional[List[int]] = None
    rho = None
    if qd_export:
        qd_lab = qd_export.get("miu_labels")
        if qd_lab is not None and int(getattr(qd_lab, "numel", lambda: 0)()) == int(
            n_full
        ):
            labels = [int(qd_lab[int(p)]) for p in full_positions]
        rho = qd_export.get("rho") if relay_residual_enabled() else None

    clusters, part_audit, resid_audit, dropped_clusters, dropped_mass_frac = (
        _partition_deleted_rows(
            pos,
            feat,
            labels,
            rho if labels is not None else None,
            tau=math.sqrt(float(k_del.shape[-1])),
            dev=dev,
        )
    )
    if not clusters:
        return None

    fr = build_kv_relays(
        k_del,
        v_del,
        clusters,
        moment=relay_moment(),
        rank=relay_rank(),
        img_path=str(img_path or ""),
        frame_seq=int(frame_seq),
    )
    fr.audit.update(part_audit)
    if resid_audit:
        fr.audit.update(resid_audit)
    fr.audit["relay_kv_deleted"] = int(sel.numel())
    fr.audit["relay_kv_source"] = "cache_rows"
    fr.audit["relay_dropped_clusters"] = dropped_clusters
    fr.audit["relay_dropped_mass_frac"] = round(dropped_mass_frac, 4)
    return fr


def _grid_positions(n_frame: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Normalized (y, x) centers for frame-local flat indices (row-major)."""
    from lmms_eval.models.geopact_sspc import resolve_frame_grid

    gh, gw, _ = resolve_frame_grid(int(grid_h), int(grid_w), int(n_frame))
    idx = torch.arange(int(n_frame), dtype=torch.long)
    r = torch.div(idx, gw, rounding_mode="floor").to(torch.float32)
    c = torch.remainder(idx, gw).to(torch.float32)
    return torch.stack([(r + 0.5) / gh, (c + 0.5) / gw], dim=1)


def store_frame_relays(inner, state: Dict[str, Any], fr: FrameRelays) -> None:
    """Append frame relays to dialogue state and (re)install the LM ctx."""
    frames: List[FrameRelays] = list(state.get("relay_frames") or [])
    frames.append(fr)
    # Enforce the global cross-frame cap by dropping the oldest frames.
    while sum(f.n_relays for f in frames) > relay_max_total() and len(frames) > 1:
        frames.pop(0)
    state["relay_frames"] = frames
    install_relay_ctx(inner, state)


def install_relay_ctx(inner, state: Optional[Dict[str, Any]]) -> None:
    lm = _language_model(inner)
    frames = list((state or {}).get("relay_frames") or [])
    if not frames:
        lm._geopact_relay_ctx = None
        return
    lm._geopact_relay_ctx = RelayCtx(frames=frames, mode=relay_mode(), amin=relay_amin())
    # The attention side-car reads the ctx through
    # ``self_attn._geopact_language_model_ref`` (modeling patch); that back-ref
    # is otherwise only installed by the vsas/parb/gsc/trp/correction setups.
    # Install it here too so relay-only arms actually engage the side-car.
    layers = getattr(lm, "layers", None)
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "_geopact_language_model_ref", None) is None:
                attn._geopact_language_model_ref = lm


def clear_relay_state(inner) -> None:
    lm = _language_model(inner)
    if hasattr(lm, "_geopact_relay_ctx"):
        lm._geopact_relay_ctx = None


def rotate_frame_relay_keys(fr: FrameRelays, delta: int, inv_freq: torch.Tensor) -> FrameRelays:
    """Shift relay post-RoPE key centroids by a constant position delta.

    Same composition identity as ``_rotate_post_rope_keys`` in
    geopact_session_kv (R(p+d) = R(d) R(p) per 2D pair); linearity of the
    rotation makes it commute with the within-cluster mean.
    """
    if int(delta) == 0:
        return fr
    k32 = fr.kbar.to(torch.float32)
    inv = inv_freq.to(device=k32.device, dtype=torch.float32).flatten()
    if int(inv.numel()) * 2 != int(k32.shape[-1]):
        raise RuntimeError(
            "GeoPACT relay rotate: inv_freq/head_dim mismatch "
            f"inv={int(inv.numel())} head_dim={int(k32.shape[-1])}"
        )
    ang = float(delta) * inv
    cos = torch.cos(ang).repeat(2).view(1, 1, 1, -1)
    sin = torch.sin(ang).repeat(2).view(1, 1, 1, -1)
    half = int(k32.shape[-1]) // 2
    rot = torch.cat([-k32[..., half:], k32[..., :half]], dim=-1)
    fr.kbar = (k32 * cos + rot * sin).to(fr.kbar.dtype)
    return fr


def drop_relays_on_slide(inner, state: Dict[str, Any]) -> None:
    """v1 conservative policy: sliding eviction invalidates relay positions
    frame-by-frame bookkeeping we don't track yet — clear and warn."""
    if state.get("relay_frames"):
        _log.warning(
            "GeoPACT relay: KV slide detected — dropping %d relay frame(s) "
            "(v1 does not track per-frame cache spans across slides)",
            len(state.get("relay_frames") or ()),
        )
        state["relay_frames"] = []
    clear_relay_state(inner)
