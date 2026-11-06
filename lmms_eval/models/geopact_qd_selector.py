"""GeoPACT Module B v2 — quality-diversity (QD) token selector.

Spec: docs/geopact_v2_master_spec.md §2. The core idea is unchanged from v1
CDCT-Herding (Module A density prior x per-step instruction relevance as
quality, iterative greedy selection maximizing dissimilarity to the picked
set); this module is the stronger instantiation:

  * quality fusion: rank-calibrated  log q_j = beta_r*r~_j + beta_P*P~_j
    (scale-free, no multiplicative annihilation at r_j = 0);
  * GUI-structured kernel  K = K_sem * K_geo  where K_geo encodes spatial
    proximity and same-MIU redundancy (PSD by Schur product, K_jj = 1);
  * selection = fast greedy MAP of the conditional DPP
    L = diag(q) K diag(q): each step picks
        argmax_j  2 log q_j + log d_j^2,
    d_j^2 the squared orthogonal residual of token j against the span of the
    picked set (incremental Cholesky, O(n t) per step, O(n k^2) total);
  * B->C coupling: per-MIU compensability discount kappa_m (clusters QASMR
    can provably relay are cheaper to delete) and per-MIU uncovered quality
    rho_m exported for relay budget allocation.

Everything is torch-only, deterministic and CPU-testable. Gated by
`GEOPACT_B_SELECTOR=qd` (default `herding` = v1 path bit-for-bit).
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _qd_stage_record(key: str, t0: float) -> float:
    """Forward to the hook's HES stage recorder (W57 latency audit)."""
    try:
        from lmms_eval.models.geopact_vl_hook import _hes_stage_record

        return _hes_stage_record(key, t0)
    except Exception:
        return time.perf_counter()

__all__ = [
    "qd_selector_kind",
    "qd_active",
    "qd_params",
    "rank01",
    "fuse_log_quality",
    "grid_positions",
    "gui_structured_kernel",
    "orthogonal_residual_greedy",
    "bam_attention_likelihood",
    "bam_mass",
    "bam_mass_alpha",
    "bam_alpha_cal_enabled",
    "bam_alpha_star",
    "bam_alpha_auto_enabled",
    "bam_alpha_auto_a0",
    "bam_alpha_auto_max",
    "bam_qk_enabled",
    "bam_qk_scores",
    "bam_gqc_enabled",
    "bam_gqc_beta",
    "bam_gqc_grounding",
    "bam_gqc_select",
    "register_bam_qk_from_model",
    "bam_select_tokens",
    "bam_fast_tail_frac",
    "bam_skip_module_a",
    "qd_select_tokens",
    "miu_residual_quality",
    "compensability_kappa",
    "set_qd_frame_context",
    "clear_qd_frame_context",
    "qd_frame_probe_keys",
    "record_qd_frame_export",
    "pop_qd_frame_export",
]

# Per-frame side channel between the rewrite/rectify caller and the selection
# path (avoids threading new arguments through the sticky-hot `_hes_select`
# signatures). Thread-local because rectify may run on the async prefetch
# thread concurrently with main-thread selections.
_FRAME_TLS = threading.local()


def _frame_ctx() -> Dict[str, Any]:
    ctx = getattr(_FRAME_TLS, "ctx", None)
    if ctx is None:
        ctx = {}
        _FRAME_TLS.ctx = ctx
    return ctx


def set_qd_frame_context(probe_keys: Optional[torch.Tensor] = None) -> None:
    """Install per-frame inputs for the next QD selection (probe-layer keys
    [H, n, d] from the frame KV snapshot enable the compensability discount)."""
    ctx = _frame_ctx()
    ctx.clear()
    if probe_keys is not None:
        ctx["probe_keys"] = probe_keys


def clear_qd_frame_context() -> None:
    _frame_ctx().clear()


def qd_frame_probe_keys() -> Optional[torch.Tensor]:
    return _frame_ctx().get("probe_keys")


def record_qd_frame_export(export: Dict[str, Any]) -> None:
    _frame_ctx()["export"] = export


def pop_qd_frame_export() -> Optional[Dict[str, Any]]:
    return _frame_ctx().pop("export", None)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def qd_selector_kind() -> str:
    """`herding` (v1, default), `qd` (v2 quality-diversity greedy) or `bam`
    (v3 parameter-free Bayes-attention matching pursuit)."""
    kind = os.environ.get("GEOPACT_B_SELECTOR", "herding").strip().lower()
    if kind not in ("herding", "qd", "bam"):
        raise ValueError(f"GEOPACT_B_SELECTOR must be herding/qd/bam, got {kind!r}")
    return kind


def qd_active() -> bool:
    """True for any selector routed through the QD dispatch (`qd` or `bam`)."""
    return qd_selector_kind() in ("qd", "bam")


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def qd_params() -> Dict[str, float]:
    return {
        "beta_r": max(0.0, _env_f("GEOPACT_QD_BETA_R", 2.0)),
        "beta_p": max(0.0, _env_f("GEOPACT_QD_BETA_P", 1.0)),
        "eps": min(1.0, max(0.0, _env_f("GEOPACT_QD_EPS", 0.15))),
        "sigma": max(1e-3, _env_f("GEOPACT_QD_SIGMA", 0.2)),
        "gamma": max(0.0, _env_f("GEOPACT_QD_GAMMA", 1.0)),
        "gamma_c": max(0.0, _env_f("GEOPACT_B_COMPENSABILITY", 0.0)),
    }


# ---------------------------------------------------------------------------
# Quality fusion (master spec §2.2)
# ---------------------------------------------------------------------------


def rank01(x: torch.Tensor) -> torch.Tensor:
    """Frame-internal fractional rank (midranks for ties) normalized to [0, 1].

    Tied values share the mean rank of their group — ordinal ranking would
    split ties by index and inject a spurious gradient (e.g. all r_j = 0
    tokens must contribute identically). Constant signals return zeros
    (no ordering information -> no contribution).
    """
    n = int(x.numel())
    if n <= 1:
        return torch.zeros_like(x, dtype=torch.float32)
    if float(x.max() - x.min()) <= 0.0:
        return torch.zeros_like(x, dtype=torch.float32)
    x32 = x.detach().float()
    order = torch.argsort(x32, stable=True)
    sorted_x = x32[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_x, return_inverse=True, return_counts=True
    )
    starts = torch.cumsum(counts, dim=0) - counts
    mid = starts.to(torch.float32) + (counts.to(torch.float32) - 1.0) / 2.0
    ranks_sorted = mid[inverse]
    ranks = torch.empty(n, dtype=torch.float32, device=x.device)
    ranks[order] = ranks_sorted
    return ranks / float(n - 1)


def fuse_log_quality(
    relevance: torch.Tensor,
    prior: Optional[torch.Tensor],
    *,
    beta_r: float,
    beta_p: float,
    kappa: Optional[torch.Tensor] = None,
    gamma_c: float = 0.0,
) -> torch.Tensor:
    """log q_j = beta_r*rank(r)_j + beta_P*rank(P)_j - log(1 + gamma_c*kappa_j).

    No annihilation: r_j = 0 tokens keep their prior-driven quality. When the
    prior is missing the P term is dropped entirely (no bias injected). The
    optional compensability discount (spec §2.5) enters in log space.
    """
    log_q = beta_r * rank01(relevance)
    if prior is not None and float(prior.abs().max()) > 0.0:
        log_q = log_q + beta_p * rank01(prior.to(log_q.device))
    if kappa is not None and gamma_c > 0.0:
        log_q = log_q - torch.log1p(gamma_c * kappa.to(log_q.device).clamp(min=0.0))
    return log_q


# ---------------------------------------------------------------------------
# GUI-structured kernel (master spec §2.3)
# ---------------------------------------------------------------------------


def grid_positions(n: int, grid_h: int, grid_w: int, device=None) -> torch.Tensor:
    """Normalized (y, x) cell centers for row-major flat token indices."""
    idx = torch.arange(int(n), dtype=torch.long, device=device)
    gh = max(1, int(grid_h))
    gw = max(1, int(grid_w))
    r = torch.div(idx, gw, rounding_mode="floor").to(torch.float32)
    c = torch.remainder(idx, gw).to(torch.float32)
    return torch.stack([(r + 0.5) / gh, (c + 0.5) / gw], dim=1)


def gui_structured_kernel(
    e_norm: torch.Tensor,
    pos: torch.Tensor,
    miu_labels: Optional[torch.Tensor],
    *,
    eps: float,
    sigma: float,
    gamma: float,
) -> torch.Tensor:
    """K = K_sem ∘ K_geo, PSD with unit diagonal (Proposition B1).

    e_norm: [n, h] L2-normalized embeddings; pos: [n, 2] normalized centers;
    miu_labels: [n] long, -1 = not covered by any MIU (the same-MIU indicator
    only fires for two tokens sharing a real label >= 0, plus the diagonal).
    """
    k_sem = e_norm @ e_norm.t()
    d2 = torch.cdist(pos, pos).pow(2)
    gauss = torch.exp(-d2 / (2.0 * sigma * sigma))
    if miu_labels is not None and gamma > 0.0:
        lab = miu_labels.view(-1, 1)
        same = (lab == lab.t()) & (lab >= 0)
        boost = torch.where(
            same, torch.full_like(gauss, 1.0 + gamma), torch.ones_like(gauss)
        )
    else:
        boost = torch.ones_like(gauss)
    k_geo = eps + (1.0 - eps) * gauss * boost / (1.0 + max(gamma, 0.0))
    # Diagonal: gauss=1 and (for label>=0) boost=1+gamma give exactly 1; tokens
    # with label -1 miss the boost, so renormalize the diagonal to keep
    # K_jj = 1 (scaling rows/cols by a positive diagonal preserves PSD).
    diag = k_geo.diagonal()
    scale = diag.clamp(min=1e-12).rsqrt()
    k_geo = k_geo * scale.view(-1, 1) * scale.view(1, -1)
    return k_sem * k_geo


# ---------------------------------------------------------------------------
# Orthogonal-residual greedy (conditional DPP MAP, master spec §2.4)
# ---------------------------------------------------------------------------


_QD_D2_FLOOR = 1e-10
_QD_WORKSPACE: Dict[Tuple, Dict[str, Any]] = {}
_QD_WORKSPACE_LIMIT = 8
# Quality sentinel for padding rows: far below any real score
# (2*log_q >= 2*log softmax > -2*log n ~ -20, plus log d2 >= log floor ~ -23),
# yet finite so argmax stays well-defined once all real rows are selected.
_QD_PAD_NEG = -1e30
_QD_BUCKET = 128


def _qd_bucket(x: int) -> int:
    """Round a workspace dimension up to the bucket grid.

    Per-frame smart-resize makes (n, k) effectively unique per screenshot; an
    exact-shape workspace/CUDA-graph cache therefore re-captures every frame
    (warmup + capture = ~3x the loop, plus fresh 40MB+ allocations), which
    round-1 surfaced as ~180-225ms entry pruning. Bucketing pads both dims so
    a handful of graphs serve the whole run; padding rows carry sentinel
    quality and zero cross-kernel mass, so the real-row math is unchanged.
    """
    return max(_QD_BUCKET, ((int(x) + _QD_BUCKET - 1) // _QD_BUCKET) * _QD_BUCKET)


def _qd_cuda_graph_enabled() -> bool:
    """CUDA-graph the QD greedy loop (default on): pure launch-overhead
    removal — the loop is fixed-shape and host-sync-free by construction."""
    return os.environ.get("GEOPACT_QD_CUDA_GRAPH", "1").strip().lower() not in {
        "0", "false", "off",
    }


def _qd_workspace(n: int, k: int, device: torch.device) -> Dict[str, Any]:
    key = (int(n), int(k), str(device))
    ws = _QD_WORKSPACE.get(key)
    if ws is not None:
        return ws
    while len(_QD_WORKSPACE) >= _QD_WORKSPACE_LIMIT:
        _QD_WORKSPACE.pop(next(iter(_QD_WORKSPACE)))
    ws = {
        "K": torch.empty(n, n, dtype=torch.float32, device=device),
        "log_q2": torch.empty(n, dtype=torch.float32, device=device),
        "log_q2_work": torch.empty(n, dtype=torch.float32, device=device),
        "d2": torch.empty(n, dtype=torch.float32, device=device),
        "scores": torch.empty(n, dtype=torch.float32, device=device),
        "chol": torch.zeros(n, k, dtype=torch.float32, device=device),
        "e": torch.empty(n, dtype=torch.float32, device=device),
        "pad_neg": torch.full((1,), _QD_PAD_NEG, dtype=torch.float32, device=device),
        "zero1": torch.zeros(1, dtype=torch.float32, device=device),
        "order_buf": torch.zeros(k, dtype=torch.long, device=device),
        "picked_d2": torch.zeros(k, dtype=torch.float32, device=device),
    }
    _QD_WORKSPACE[key] = ws
    return ws


def _qd_greedy_loop(ws: Dict[str, Any], keep_k: int) -> None:
    """Fixed-shape, host-sync-free greedy MAP loop (CUDA-graph capturable).

    Reads ws['K'] (fp32, unit-ish diagonal) and ws['log_q2'] (= 2 log q);
    writes ws['order_buf'][:keep_k] and ws['picked_d2'][:keep_k].
    Self-contained: resets its own mutable state at the head (log_q2 is
    snapshotted into log_q2_work) so CUDA-graph warmup/capture/replay all
    start from the caller's inputs. (The first W57 trim mutated ws['log_q2']
    directly; the capture-time warmup then poisoned the buffer the recorded
    replay reads, corrupting the first call per bucket key — caught by the
    keeps identity gate on exactly the four first-seen frame sizes.)

    W57 launch trims (argmax sequence provably unchanged):
      * selected-row suppression by poisoning log_q2_work with the pad
        sentinel (a poisoned score is <= -1e30 + log(floor) while any real
        row stays >= -2 log n + log(floor) > -1e2) — drops the n-sized
        masked_fill and the bool selected_mask at the cost of one n-sized
        copy at the head;
      * e = kj - C C[j] via torch.sub(kj, e, out=e) — one kernel instead of
        sub_ + neg_ (IEEE RN: b - a == -(a - b) bitwise);
      * gating with torch.where instead of bool->float mul (identical values:
        gate*x is 1.0*x or 0.0*x with x finite).
    """
    K = ws["K"]
    log_q2 = ws["log_q2"]
    log_q2_work = ws["log_q2_work"]
    d2 = ws["d2"]
    scores = ws["scores"]
    chol = ws["chol"]
    e = ws["e"]
    pad_neg = ws["pad_neg"]
    zero1 = ws["zero1"]
    order_buf = ws["order_buf"]
    picked_d2 = ws["picked_d2"]

    chol.zero_()
    d2.copy_(K.diagonal())
    log_q2_work.copy_(log_q2)

    for t in range(keep_k):
        # scores = log_q2 + log(max(d2, floor)); selected rows carry the pad
        # sentinel in log_q2_work and lose every argmax to any real row.
        torch.clamp(d2, min=_QD_D2_FLOOR, out=scores)
        scores.log_()
        scores.add_(log_q2_work)
        j = torch.argmax(scores).view(1)                       # [1] on device
        order_buf[t] = j[0]
        dj2 = d2.index_select(0, j)                            # [1]
        picked_d2[t] = dj2[0]
        # e = (K[:, j] - C C[j]^T) * live / sqrt(max(dj2, floor)); columns of C
        # beyond t-1 are zero, so the fixed-shape matvec equals the t-column
        # product. Exhausted picks (dj2 <= floor) write a zero column.
        kj = K.index_select(1, j).squeeze(1)                   # [n]
        cj = chol.index_select(0, j).squeeze(0)                # [k]
        torch.mv(chol, cj, out=e)
        torch.sub(kj, e, out=e)
        inv = torch.where(
            dj2 > _QD_D2_FLOOR,
            torch.rsqrt(dj2.clamp(min=_QD_D2_FLOOR)),
            zero1,
        )
        e.mul_(inv)
        chol[:, t] = e
        d2.addcmul_(e, e, value=-1.0)
        d2.clamp_(min=0.0)
        log_q2_work.index_copy_(0, j, pad_neg)


def _qd_run_greedy(ws: Dict[str, Any], keep_k: int, device: torch.device) -> None:
    """Run the greedy loop via a cached CUDA graph when possible (same
    launch-overhead-removal pattern as `_ickh_run_greedy` in the hook)."""
    if device.type != "cuda" or not _qd_cuda_graph_enabled():
        _qd_greedy_loop(ws, keep_k)
        return
    gkey = ("graph", int(keep_k))
    graph = ws.get(gkey)
    if graph is False:
        _qd_greedy_loop(ws, keep_k)
        return
    if graph is None:
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                _qd_greedy_loop(ws, keep_k)  # warmup (allocator)
            torch.cuda.current_stream(device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    _qd_greedy_loop(ws, keep_k)
            except Exception:
                try:
                    graph.reset()
                except Exception:
                    pass
                raise
            ws[gkey] = graph
        except Exception:
            ws[gkey] = False
            _qd_greedy_loop(ws, keep_k)
            return
    graph.replay()


def orthogonal_residual_greedy(
    kernel: torch.Tensor,
    log_q: torch.Tensor,
    keep_k: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy MAP of L = diag(q) K diag(q) via incremental Cholesky.

    Each step picks argmax_j 2*log_q_j + log d_j^2 where d_j^2 is the squared
    residual of phi_j against span{phi_s : s picked}. When the span exhausts
    the feature space (residual <= floor) the score reduces to quality order
    (log(floor) is a shared constant), so selection degrades gracefully to
    quality top-k; exhausted picks contribute a zeroed Cholesky column so the
    factorization stays finite.

    O(n t) per step (one matvec), O(n k^2) total; fully device-side with the
    single host sync at the final order readout. On CUDA the fixed-shape loop
    is replayed through a cached CUDA graph (GEOPACT_QD_CUDA_GRAPH=0 to
    disable).
    """
    n = int(kernel.shape[0])
    k = max(1, min(int(keep_k), n))
    device = kernel.device
    _t = time.perf_counter()
    n_b = _qd_bucket(n)
    k_b = min(_qd_bucket(k), n_b)
    ws = _qd_workspace(n_b, k_b, device)
    # Padding rows: zero kernel mass (d2 = 0 -> floor) + sentinel quality, so
    # they lose every argmax until all real rows are selected; their Cholesky
    # columns are exactly zero, leaving real-row picks bit-identical to the
    # exact-shape run. order_buf beyond position k-1 is never read.
    ws["K"].zero_()
    ws["K"][:n, :n].copy_(kernel.detach().to(torch.float32))
    ws["log_q2"].fill_(_QD_PAD_NEG)
    ws["log_q2"][:n].copy_(log_q.detach().to(torch.float32)).mul_(2.0)
    _t = _qd_stage_record("hes_qd_setup_ms", _t)
    _qd_run_greedy(ws, k_b, device)
    _t = _qd_stage_record("hes_qd_loop_ms", _t)

    order = [int(i) for i in ws["order_buf"][:k].tolist()]
    n_exhausted = int((ws["picked_d2"][:k] <= _QD_D2_FLOOR).sum().item())
    _qd_stage_record("hes_qd_read_ms", _t)
    audit: Dict[str, Any] = {
        "qd_picked_greedy": k - n_exhausted,
        "qd_rank_exhausted": int(n_exhausted > 0),
        "qd_quality_backfill": n_exhausted,
    }
    return order, audit


# ---------------------------------------------------------------------------
# BAM: parameter-free Bayes-attention matching pursuit (v3)
# ---------------------------------------------------------------------------


def _bam_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def bam_fast_tail_frac() -> Optional[float]:
    """GEOPACT_BAM_FAST_TAIL: budget-aware greedy horizon (W-L latency).

    Under NAP the entry ordering is consumed at two depths only: positions
    [0, k_hist) survive into history (prefix contraction), positions
    [k_hist, k_cur) only need MEMBERSHIP in S_cur (order irrelevant — all of
    them are prefilled, none outlive the frame). The O(k) sequential greedy
    therefore buys ordering precision the pipeline never reads beyond the
    horizon. Value = fraction h of keep_k to rank greedily (set it to
    r_hist / r_cur, e.g. 0.2 for c50/h10); the tail is filled by quality
    top-k (log a + log m) over the unpicked tokens. Head picks are
    bit-identical to the full greedy's first ceil(h*k) picks (same matrix,
    same argmax sequence).
    """
    raw = os.environ.get("GEOPACT_BAM_FAST_TAIL", "").strip()
    if not raw or raw in ("0", "false", "off", "no"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return min(1.0, v) if v > 0.0 else None


def bam_skip_module_a() -> bool:
    """True when Module A can be skipped entirely for this step.

    Under BAM + GEOPACT_BAM_NO_MASS the spatial prior is never consumed
    (m = 1), so the YOLO detector run + async join (~120ms wall, ~54ms join
    on 8B) buys nothing. W-L: skipping it removes the largest avoidable
    entry-latency block. GEOPACT_BAM_NO_MASS_RUN_A=1 forces A back on for
    diag parity.
    """
    if not _bam_flag("GEOPACT_BAM_NO_MASS"):
        return False
    if _bam_flag("GEOPACT_BAM_NO_MASS_RUN_A"):
        return False
    return qd_selector_kind() == "bam"


def bam_mass_alpha() -> float:
    """GEOPACT_BAM_MASS_ALPHA: prior-strength alpha in m = 1 + alpha*n*p_A.

    The default m = 1 + n*p_A silently pins alpha = 1 (detected mass on the
    same scale as pixel evidence). The review asks for a robustness sweep;
    alpha is an *arm*, not a tuned default (default stays 1.0).
    """
    return max(0.0, _env_f("GEOPACT_BAM_MASS_ALPHA", 1.0))


# last calibrated alpha* (per-frame diagnostic surfaced into the BAM audit)
_BAM_LAST_ALPHA: Dict[str, float] = {}


def bam_alpha_auto_enabled() -> bool:
    """GEOPACT_BAM_ALPHA_AUTO=1: concentration-normalized prior strength.

    The mass boost token j receives is alpha*n*p_j, so its prior-expected
    magnitude is E_p[alpha*n*p] = alpha * G with G = n*sum(p^2) = n/ESS(p)
    (G = 1 for a flat prior, ~n for a one-hot prior). Fixing that expected
    boost to a single constant A0 makes the prior's felt strength invariant
    to how concentrated the detector output happens to be on a given UI:

        alpha_auto = clip(A0 / G, 1, alpha_max)

    Flat/diffuse prior (web pages, G ~ 1) -> full strength A0; already-spiky
    prior (dense mobile layouts, large G) -> weakened toward 1. One shared
    constant A0, no per-dataset freedom. Takes precedence over the ESS
    calibration arm (GEOPACT_BAM_ALPHA_CAL) and never re-enables the
    no-mass ablation (alpha = 0 stays 0).
    """
    return _bam_flag("GEOPACT_BAM_ALPHA_AUTO")


def bam_alpha_auto_a0() -> float:
    """GEOPACT_BAM_ALPHA_AUTO_A0: expected-boost constant A0 (default 4.0)."""
    return max(0.0, _env_f("GEOPACT_BAM_ALPHA_AUTO_A0", 4.0))


def bam_alpha_auto_max() -> float:
    """GEOPACT_BAM_ALPHA_AUTO_MAX: upper clip for alpha_auto (default 6.0)."""
    return max(1.0, _env_f("GEOPACT_BAM_ALPHA_AUTO_MAX", 6.0))


def bam_alpha_cal_enabled() -> bool:
    """GEOPACT_BAM_ALPHA_CAL=1: budget-adaptive effective-support calibration.

    Review §G.3: the fixed-alpha sweep showed a grid-dependent optimum (weak
    prior wins at loose budgets, full-strength at tight budgets). Instead of
    per-grid tuning, pick per frame the largest alpha whose mass measure still
    spreads over at least the selection budget:

        alpha* = max{ alpha : N_eff(m(alpha)) >= zeta * k },
        N_eff  = (sum m)^2 / sum m^2.

    Loose budget (large k) demands large effective support -> small alpha;
    tight budget allows concentration -> large alpha. Scalar bisection, no
    dev-set constant. zeta via GEOPACT_BAM_ALPHA_CAL_ZETA (default 1).
    """
    return _bam_flag("GEOPACT_BAM_ALPHA_CAL")


def bam_alpha_star(
    p_norm: torch.Tensor,
    n: int,
    keep_k: int,
    *,
    zeta: Optional[float] = None,
    alpha_cap: float = 8.0,
) -> float:
    """Largest alpha in [0, alpha_cap] with N_eff(1 + alpha*n*p) >= zeta*k."""
    z = zeta if zeta is not None else max(0.1, _env_f("GEOPACT_BAM_ALPHA_CAL_ZETA", 1.0))
    target = max(1.0, z * float(keep_k))

    def n_eff(alpha: float) -> float:
        m = 1.0 + alpha * float(n) * p_norm
        s = float(m.sum())
        s2 = float((m * m).sum())
        return (s * s) / max(s2, 1e-12)

    # N_eff(0) = n >= k always; monotone non-increasing in alpha with limit
    # 1/sum(p^2) — a flat prior never concentrates below the target.
    if n_eff(alpha_cap) >= target:
        return float(alpha_cap)
    lo, hi = 0.0, float(alpha_cap)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_eff(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


# --- first-layer QK relevance (review §7.3: score text->image relevance in
# the model's own attention geometry instead of raw embedding cosine) --------

_BAM_QK_REG: Dict[str, Any] = {}


def bam_qk_enabled() -> bool:
    return _bam_flag("GEOPACT_BAM_QK_RELEVANCE")


def register_bam_qk_from_model(inner) -> bool:
    """Cache layer-0 input RMSNorm + W_q/W_k references (no copies) once.

    Idempotent and failure-tolerant: returns True when the projections are
    available. Works across the Qwen2/2.5/3-VL layouts by probing the known
    layer paths; per-head q_norm/k_norm (Qwen3) are picked up when present.

    The registry is keyed to the model instance: a different ``inner`` (model
    swap / reload in the same process) re-probes instead of silently reusing
    the previous model's layer-0 weights, and a past failure on another
    instance does not permanently disable QK for the new one.
    """
    mid = id(inner)
    if _BAM_QK_REG.get("model_id") == mid:
        if _BAM_QK_REG.get("ready"):
            return True
        if _BAM_QK_REG.get("failed"):
            return False
    else:
        _BAM_QK_REG.clear()
        _BAM_QK_REG["model_id"] = mid
    layers = None
    for path in (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    ):
        obj = inner
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            layers = obj
            break
    if layers is None:
        _BAM_QK_REG["failed"] = True
        return False
    try:
        layer0 = layers[0]
        attn = layer0.self_attn
        ln = layer0.input_layernorm
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        head_dim = getattr(attn, "head_dim", None)
        cfg = getattr(attn, "config", None)
        n_heads = getattr(attn, "num_heads", None) or getattr(
            cfg, "num_attention_heads", None
        )
        n_kv = getattr(attn, "num_key_value_heads", None) or getattr(
            cfg, "num_key_value_heads", None
        )
        if head_dim is None and n_heads:
            head_dim = q_w.shape[0] // int(n_heads)
        if not (head_dim and n_heads and n_kv):
            _BAM_QK_REG["failed"] = True
            return False
        _BAM_QK_REG.update(
            {
                "ln_w": ln.weight,
                "ln_eps": float(getattr(ln, "variance_epsilon", 1e-6)),
                "q_w": q_w,
                "q_b": getattr(attn.q_proj, "bias", None),
                "k_w": k_w,
                "k_b": getattr(attn.k_proj, "bias", None),
                "q_norm": getattr(attn, "q_norm", None),
                "k_norm": getattr(attn, "k_norm", None),
                "n_heads": int(n_heads),
                "n_kv": int(n_kv),
                "head_dim": int(head_dim),
                "ready": True,
            }
        )
        return True
    except Exception:
        _BAM_QK_REG["failed"] = True
        return False


def _bam_qk_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w.float()


def _bam_qk_project(
    image_embeds: torch.Tensor,
    instr_query: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Project instruction rows through W_q and image rows through W_k of the
    model's layer 0 (own RMSNorm, per-head q/k norms when present, GQA keys
    expanded). Returns (q [T,H,dh], k [n,H,dh], head_dim) or None when the
    projections were never registered / a shape probe fails.
    """
    if not _BAM_QK_REG.get("ready"):
        return None
    try:
        w = _BAM_QK_REG
        dev = w["q_w"].device
        u = instr_query.detach().to(device=dev, dtype=torch.float32)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        e = image_embeds.detach().to(device=dev, dtype=torch.float32)
        u_n = _bam_qk_rmsnorm(u, w["ln_w"], w["ln_eps"])
        e_n = _bam_qk_rmsnorm(e, w["ln_w"], w["ln_eps"])
        q = u_n @ w["q_w"].float().t()
        if w["q_b"] is not None:
            q = q + w["q_b"].float()
        k = e_n @ w["k_w"].float().t()
        if w["k_b"] is not None:
            k = k + w["k_b"].float()
        h, hk, dh = w["n_heads"], w["n_kv"], w["head_dim"]
        q = q.view(-1, h, dh)
        k = k.view(-1, hk, dh)
        if w["q_norm"] is not None:
            q = _bam_qk_rmsnorm(q, w["q_norm"].weight, float(getattr(w["q_norm"], "variance_epsilon", 1e-6)))
        if w["k_norm"] is not None:
            k = _bam_qk_rmsnorm(k, w["k_norm"].weight, float(getattr(w["k_norm"], "variance_epsilon", 1e-6)))
        if hk != h:
            k = k.repeat_interleave(h // hk, dim=1)
        return q, k, int(dh)
    except Exception:
        return None


def bam_qk_scores(
    image_embeds: torch.Tensor,
    instr_query: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Per-token relevance s_j = pool_r mean_h <q_rh, k_jh>/sqrt(d_h).

    Both inputs are pre-layer-0 hidden rows; they go through the model's own
    layer-0 input RMSNorm and W_q/W_k, so the score lives in the geometry the
    first attention layer actually uses. RoPE is skipped deliberately: the
    instruction rows and image rows have no shared positional frame here, and
    the relevance question is content-based. Pooling over query rows matches
    the cosine path (amax; GEOPACT_BAM_QUERY_LSE=1 for log-mean-exp).
    """
    proj = _bam_qk_project(image_embeds, instr_query)
    if proj is None:
        return None
    try:
        q, k, dh = proj
        h = int(q.shape[1])
        s = torch.einsum("thd,nhd->tn", q, k) / (math.sqrt(dh) * h)
        if _bam_flag("GEOPACT_BAM_QUERY_LSE") and int(s.shape[0]) > 1:
            pooled = torch.logsumexp(s, dim=0) - math.log(int(s.shape[0]))
        else:
            pooled = s.amax(dim=0)
        return pooled.to(image_embeds.device)
    except Exception:
        return None


# --- GQC-BAM: grounded query-coverage selection (instruction-conditioning
# design doc §6-§10). The instruction stops being a static per-token scalar:
# layer-0 Q/K heads produce per-word visual grounding distributions, their
# concentration estimates groundability, and a submodular coverage objective
# rewards only tokens that add *uncovered* instruction evidence. ------------


def bam_gqc_enabled() -> bool:
    return _bam_flag("GEOPACT_BAM_GQC")


def bam_gqc_beta() -> float:
    """GEOPACT_BAM_GQC_BETA in [0,1]: weight of the query-coverage term vs the
    visual D-optimal term. The one honest hyperparameter of GQC (design doc
    §10.3): the two objectives are genuinely different and the trade-off
    should be explicit, not hidden in pooling rules."""
    return min(1.0, max(0.0, _env_f("GEOPACT_BAM_GQC_BETA", 0.5)))


def bam_gqc_topr() -> int:
    """Keep at most R high-confidence query maps (design doc §13.2)."""
    try:
        return max(1, int(os.environ.get("GEOPACT_BAM_GQC_TOPR", "8")))
    except ValueError:
        return 8


def bam_gqc_grounding(
    image_embeds: torch.Tensor,
    instr_query: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-word grounding maps and groundability confidences.

    For each (query row r, head h): z-score the layer-0 logits over visual
    positions, softmax to a distribution, then average distributions across
    heads (probability averaging keeps a specialist head's peak from being
    cancelled by other heads' logit offsets — design doc §7). Confidence
    c_r = 1 - H(pi_r)/log n: words whose evidence is uniform over the screen
    (function words, actions with no visual locus) lose weight automatically;
    no stopword list, no POS tagger (§8.1).

    Near-duplicate maps (subword splits of the same word) are merged so one
    word cannot occupy several coverage slots (§13.3); the top-R confident
    maps are kept. Returns (pi [R, n] row-stochastic, omega [R] normalized
    confidence weights) on image_embeds.device, or None when projections are
    unavailable / nothing is groundable (caller falls back to standard BAM).
    """
    proj = _bam_qk_project(image_embeds, instr_query)
    if proj is None:
        return None
    try:
        q, k, dh = proj
        n = int(k.shape[0])
        if n <= 1 or int(q.shape[0]) < 1:
            return None
        # [T, H, n] logits in the model's own attention geometry
        s = torch.einsum("thd,nhd->thn", q, k) / math.sqrt(dh)
        z = (s - s.mean(dim=-1, keepdim=True)) / (s.std(dim=-1, keepdim=True) + 1e-6)
        pi_h = torch.softmax(z, dim=-1)          # [T, H, n]
        pi = pi_h.mean(dim=1)                    # [T, n] cross-head prob mean
        ent = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(dim=-1)
        conf = (1.0 - ent / math.log(n)).clamp(min=0.0)   # [T]
        if float(conf.sum()) < 1e-6:
            return None
        # merge near-duplicate maps (cos >= 0.98), keeping the higher conf row
        pn = F.normalize(pi, dim=-1)
        sim = pn @ pn.t()
        order = torch.argsort(conf, descending=True)
        keep: List[int] = []
        for idx in order.tolist():
            if all(float(sim[idx, j]) < 0.98 for j in keep):
                keep.append(idx)
            if len(keep) >= bam_gqc_topr():
                break
        rows = torch.as_tensor(keep, dtype=torch.long, device=pi.device)
        pi_r = pi.index_select(0, rows)
        conf_r = conf.index_select(0, rows)
        omega = conf_r / conf_r.sum()
        dev = image_embeds.device
        return pi_r.to(dev), omega.to(dev)
    except Exception:
        return None


def bam_gqc_select(
    e_norm: torch.Tensor,
    keep_k: int,
    pi: torch.Tensor,
    omega: torch.Tensor,
    m: torch.Tensor,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy of F(S) = beta*F_Q(S) + (1-beta)*F_V(S), both marginals in [0,1].

      coverage:  Delta_Q(j|S) = sum_r omega_r * max(0, pi_rj - C_r),
                 C_r = max_{i in S} pi_ri              (monotone submodular)
      visual:    Delta_V(j|S) = log(1 + d_j^2) / log(1 + max_j m_j),
                 d_j^2 the mass-weighted orthogonal residual (same incremental
                 Cholesky as BAM; d_j^2 <= K_jj = m_j so the marginal is
                 normalized to [0,1] by construction).

    Both terms are marginals of monotone submodular functions, so the greedy
    keeps the (1 - 1/e) guarantee on the mixture (design doc §10.2). Plain
    device loop (no CUDA graph yet): R <= 8 keeps the coverage update an
    [R, n] broadcast, dominated by the existing residual matvec.
    """
    n = int(e_norm.shape[0])
    device = e_norm.device
    k = max(1, min(int(keep_k), n))
    beta = bam_gqc_beta()

    sqrt_m = m.sqrt()
    K = (e_norm @ e_norm.t()) * (sqrt_m.view(-1, 1) * sqrt_m.view(1, -1))
    d2 = K.diagonal().clone()
    v_norm = math.log(1.0 + max(float(m.max()), 1.0))

    R = int(pi.shape[0])
    cov = torch.zeros(R, dtype=torch.float32, device=device)  # C_r
    chol = torch.zeros(n, k, dtype=torch.float32, device=device)
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    order: List[int] = []
    for t in range(k):
        gain_q = (
            (pi - cov.view(-1, 1)).clamp(min=0.0) * omega.view(-1, 1)
        ).sum(dim=0)                                           # [n]
        gain_v = torch.log1p(d2.clamp(min=0.0)) / v_norm       # [n]
        score = beta * gain_q + (1.0 - beta) * gain_v
        score = score.masked_fill(selected, float("-inf"))
        j = int(torch.argmax(score).item())
        order.append(j)
        selected[j] = True
        cov = torch.maximum(cov, pi[:, j])
        dj2 = float(d2[j])
        if dj2 > _QD_D2_FLOOR:  # exhausted span -> zero column (BAM semantics)
            e = (K[:, j] - chol[:, :t] @ chol[j, :t]) / math.sqrt(dj2)
            chol[:, t] = e
            d2 = (d2 - e * e).clamp(min=0.0)
    audit: Dict[str, Any] = {
        "bam_gqc": True,
        "gqc_R": R,
        "gqc_beta": round(beta, 4),
        "gqc_conf_mean": round(float(omega.mean()), 6),
        "gqc_coverage": round(float((omega * cov).sum()), 6),
    }
    return order, audit


def bam_attention_likelihood(
    e_norm: torch.Tensor,
    instr_query: Optional[torch.Tensor],
    *,
    scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """log a_j = log softmax_j(zscore(s_j)) — scale-free score calibration.

    z-score is the scale-free standardization when the score scale is unknown
    (invariant to affine rescaling of the similarities), and softmax is the
    default map from scores to a distribution with no temperature knob.
    Softmax never returns exact zeros, so there is no multiplicative
    annihilation. No instruction / constant scores -> exact uniform (the
    selection degrades to pure volume maximization).

    ``scores`` (optional, [n]) supplies precomputed raw relevance scores —
    the QK-relevance arm (``bam_qk_scores``) uses this; the default s_j is
    the pooled cosine between e_j and the instruction rows.

    GEOPACT_BAM_REL_TAU (sensitivity probe, default 1.0 = published rule):
    divides the z-scored relevance before the softmax — the temperature the
    paper claims it does not need; the probe arms verify tau=1 sits on a
    plateau rather than a knife edge.
    """
    n = int(e_norm.shape[0])
    device = e_norm.device
    uniform = torch.full((n,), -math.log(max(n, 1)), dtype=torch.float32, device=device)
    if n <= 1:
        return uniform
    if scores is not None and int(scores.numel()) == n:
        s = scores.detach().to(device=device, dtype=torch.float32)
    elif instr_query is None:
        return uniform
    else:
        u = instr_query.detach().to(device=device, dtype=torch.float32)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        u = F.normalize(u, dim=-1)
        sims = e_norm @ u.t()
        if _bam_flag("GEOPACT_BAM_QUERY_LSE") and int(u.shape[0]) > 1:
            # log-mean-exp over query rows: a parameter-free smooth maximum.
            # amax makes the score distribution right-skewed (one lucky row
            # wins), which over-concentrates the softmax at loose budgets;
            # LSE keeps the best-row semantics while every row contributes
            # its evidence.
            s = torch.logsumexp(sims, dim=-1) - math.log(int(u.shape[0]))
        else:
            s = sims.amax(dim=-1)
    std = float(s.std())
    if not (std > 1e-12):
        return uniform
    z = (s - s.mean()) / std
    tau_raw = os.environ.get("GEOPACT_BAM_REL_TAU", "").strip()
    if tau_raw:
        try:
            tau = float(tau_raw)
            if tau > 0.0 and tau != 1.0:
                z = z / tau
        except ValueError:
            pass
    return F.log_softmax(z, dim=0)


def bam_mass(
    spatial_prior: Optional[torch.Tensor],
    n: int,
    device,
    keep_k: Optional[int] = None,
) -> torch.Tensor:
    """m_j = 1 + n * p_A(j): each token counts once as pixel evidence plus its
    share of Module A's detected interaction mass (density field normalized to
    a probability measure; total detected mass = n keeps the two parts on the
    same scale by construction, not by a tuned weight). No detections -> m=1.
    """
    m = torch.ones(n, dtype=torch.float32, device=device)
    if spatial_prior is None:
        return m
    p = spatial_prior.detach().to(device=device, dtype=torch.float32).clamp(min=0.0)
    total = float(p.sum())
    if total <= 0.0 or int(p.shape[0]) != n:
        _BAM_LAST_ALPHA["g"] = 0.0  # no usable prior this frame
        return m
    p_norm = p / total
    # Prior-concentration factor G = n * sum(p^2) = n / ESS(p): 1 for a flat
    # prior, ~n for a one-hot prior. Surfaced per frame into the BAM audit as
    # the driving signal for auto prior strength (alpha_auto = A0 / G).
    g = float(n) * float((p_norm * p_norm).sum())
    _BAM_LAST_ALPHA["g"] = round(g, 4)
    _BAM_LAST_ALPHA["n"] = int(n)
    if keep_k is not None:
        _BAM_LAST_ALPHA["k"] = int(keep_k)
    alpha = bam_mass_alpha()
    if bam_alpha_auto_enabled() and alpha > 0.0:
        # Concentration-normalized strength (see bam_alpha_auto_enabled).
        # G >= 1 by Cauchy-Schwarz; the max() is numeric safety only.
        alpha = min(bam_alpha_auto_max(), max(1.0, bam_alpha_auto_a0() / max(g, 1.0)))
        _BAM_LAST_ALPHA["auto"] = round(float(alpha), 4)
    elif (
        bam_alpha_cal_enabled()
        and alpha > 0.0
        and keep_k is not None
        and int(keep_k) > 0
    ):
        # Budget-adaptive prior strength (review §G.3). Capped at the fixed
        # alpha: calibration can only *weaken* an over-concentrated prior
        # (loose budget -> N_eff must cover k -> smaller alpha); it never
        # re-enables mass on the no-mass arm (alpha = 0 stays 0).
        alpha = min(alpha, bam_alpha_star(p_norm, n, int(keep_k)))
        _BAM_LAST_ALPHA["star"] = round(float(alpha), 4)
    # GEOPACT_BAM_MASS_ALPHA (default 1.0): prior-strength robustness arm,
    # m = 1 + alpha * n * p_A. alpha = 0 degenerates to the no-mass ablation.
    return m + alpha * float(n) * p_norm


def bam_select_tokens(
    image_embeds: torch.Tensor,
    keep_k: int,
    *,
    instr_query: Optional[torch.Tensor],
    spatial_prior: Optional[torch.Tensor],
) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
    """Parameter-free B v3 (BAM): orthogonal matching pursuit of the frame's
    attention readout  r = sum_j a_j psi_j  with
      a_j   = softmax(zscore(cos(e_j, u)))       (attention likelihood)
      psi_j = sqrt(m_j) e_j,  m_j = 1 + n p_A(j) (A's prior as feature mass)
    Per-step score:  log a_j + log d_j^2,  where d_j^2 is the squared
    orthogonal residual of psi_j against the selected span. d^2 starts at m_j
    (log-prior-mass) and collapses under redundancy — the prior fades through
    the diversity mechanism itself, not through a decay knob. Zero
    hyperparameters.

    Ablation switches (arms, not knobs):
      GEOPACT_BAM_NO_MASS=1    forces m = 1 (Module A mass removed);
      GEOPACT_BAM_UNIFORM_A=1  forces the likelihood uniform (pure volume);
      GEOPACT_BAM_NO_DIVERSITY=1  drops the orthogonal-residual term: pure
                               quality top-k by log a_j + log m_j (the greedy
                               step-0 score, i.e. d^2 frozen at its prior m).
                               Isolates the redundancy-collapse mechanism;
      GEOPACT_BAM_SHARP_A=1    restores the DPP-square score
                               (2 log a_j + log d_j^2, MAP of
                               L = diag(a) G diag(a)). Both the 100-step and
                               500-step screens showed the squared likelihood
                               amplifies instruction noise (zscore ~±3 ->
                               e^12 score spread) and loses to the
                               matching-pursuit form on TM/GP.

    Returns (pick order local to the given rows, per-token quality a*m used
    for the B->C rho export, audit).
    """
    n = int(image_embeds.shape[0])
    device = image_embeds.device
    e_norm = F.normalize(image_embeds.detach().to(torch.float32), dim=-1)

    no_mass = _bam_flag("GEOPACT_BAM_NO_MASS")
    uniform_a = _bam_flag("GEOPACT_BAM_UNIFORM_A")

    # GQC-BAM arm: grounded query-coverage + visual D-optimal mixture. Falls
    # through to standard BAM when grounding is unavailable (no instruction,
    # projections unregistered, or nothing groundable on this screen).
    if bam_gqc_enabled() and not uniform_a and instr_query is not None:
        grounding = bam_gqc_grounding(image_embeds, instr_query)
        if grounding is not None:
            pi, omega = grounding
            m_g = (
                torch.ones(n, dtype=torch.float32, device=device)
                if no_mass
                else bam_mass(spatial_prior, n, device, keep_k=keep_k)
            )
            order, audit = bam_gqc_select(e_norm, keep_k, pi, omega, m_g)
            audit["bam_n"] = n
            audit["bam_keep_k"] = int(keep_k)
            audit["bam_has_mass"] = bool(float((m_g - 1.0).abs().max()) > 0.0)
            # quality export for the B->C handoff: singleton coverage * mass
            quality = (pi * omega.view(-1, 1)).sum(dim=0) * m_g
            return order, quality, audit

    _t = time.perf_counter()
    qk_used = False
    if uniform_a:
        log_a = torch.full(
            (n,), -math.log(max(n, 1)), dtype=torch.float32, device=device
        )
    else:
        qk_scores = None
        if bam_qk_enabled() and instr_query is not None:
            # first-layer QK relevance arm (review §7.3): score in the
            # model's own attention geometry; silent fallback to cosine when
            # the projections were never registered.
            qk_scores = bam_qk_scores(image_embeds, instr_query)
        qk_used = qk_scores is not None
        log_a = bam_attention_likelihood(e_norm, instr_query, scores=qk_scores)
    m = (
        torch.ones(n, dtype=torch.float32, device=device)
        if no_mass
        else bam_mass(spatial_prior, n, device, keep_k=keep_k)
    )
    _t = _qd_stage_record("hes_bam_lik_ms", _t)

    no_diversity = _bam_flag("GEOPACT_BAM_NO_DIVERSITY")
    if no_diversity:
        # Quality top-k: the greedy step-0 score log a_j + log m_j with the
        # residual term frozen (no redundancy collapse). Pick order = score
        # order so the NAP prefix semantics stay meaningful.
        score = log_a + m.log()
        k = max(1, min(int(keep_k), n))
        order = [int(i) for i in torch.topk(score, k).indices.tolist()]
        audit: Dict[str, Any] = {"bam_no_diversity": True}
    else:
        sqrt_m = m.sqrt()
        kernel = (e_norm @ e_norm.t()) * (sqrt_m.view(-1, 1) * sqrt_m.view(1, -1))
        _t = _qd_stage_record("hes_bam_kernel_ms", _t)

        # F4 (v3): matching-pursuit scoring is the default; sharp is the
        # ablation. GEOPACT_BAM_SOFT_A=1 accepted for backward compat.
        sharp_a = _bam_flag("GEOPACT_BAM_SHARP_A") and not _bam_flag("GEOPACT_BAM_SOFT_A")
        # greedy doubles log_q internally; halve for the single-likelihood form
        log_q = log_a if sharp_a else 0.5 * log_a
        h_frac = bam_fast_tail_frac()
        k_eff = max(1, min(int(keep_k), n))
        if h_frac is not None and h_frac < 1.0 and k_eff > 1:
            horizon = max(1, min(k_eff, int(math.ceil(h_frac * k_eff))))
        else:
            horizon = k_eff
        if horizon < k_eff:
            order, audit = orthogonal_residual_greedy(kernel, log_q, horizon)
            # tail: quality top-k over the unpicked tokens (order beyond the
            # horizon is never consumed — membership only)
            score = log_a + m.log()
            score = score.clone()
            head_idx = torch.tensor(order, dtype=torch.long, device=device)
            score[head_idx] = float("-inf")
            tail = torch.topk(score, k_eff - horizon).indices.tolist()
            order = order + [int(i) for i in tail]
            audit["bam_fast_tail_horizon"] = horizon
        else:
            order, audit = orthogonal_residual_greedy(kernel, log_q, keep_k)
        _t = _qd_stage_record("hes_bam_greedy_ms", _t)
        audit["bam_soft_a"] = not sharp_a
    audit["bam_n"] = n
    audit["bam_keep_k"] = int(keep_k)
    audit["bam_uniform_a"] = bool(uniform_a or instr_query is None)
    audit["bam_has_mass"] = bool(float((m - 1.0).abs().max()) > 0.0)
    # Relevance-channel concentration G_a = n * sum(a^2) = n / ESS(a): 1 when
    # the instruction likelihood is flat (relevance channel uninformative),
    # large when it names a specific target. Candidate auto-alpha signal:
    # lean on the spatial prior exactly when relevance is ignorant.
    a_prob = torch.softmax(log_a, dim=0)
    audit["bam_rel_g"] = round(float(n) * float((a_prob * a_prob).sum()), 4)
    # Prior-relevance complementarity: how much of the prior's evidence the
    # relevance channel already sees. p_hat recovers the normalized prior
    # from m = 1 + alpha*n*p (alpha scale cancels in the normalization).
    p_hat = m - 1.0
    p_sum = float(p_hat.sum())
    if p_sum > 0.0 and n > 1:
        p_hat = p_hat / p_sum
        pc = p_hat - p_hat.mean()
        ac = a_prob - a_prob.mean()
        denom = float(pc.norm()) * float(ac.norm())
        if denom > 1e-12:
            audit["bam_pa_corr"] = round(float((pc * ac).sum()) / denom, 4)
        k_ov = max(1, min(int(keep_k), n))
        top_p = set(torch.topk(p_hat, k_ov).indices.tolist())
        top_a = set(torch.topk(a_prob, k_ov).indices.tolist())
        audit["bam_pa_overlap"] = round(len(top_p & top_a) / float(k_ov), 4)
    if bam_qk_enabled():
        # Truthful audit: True only when QK scores were actually computed and
        # consumed this call (not merely "projections registered") — a
        # mid-flight scoring failure falls back to cosine and must say so.
        audit["bam_qk_relevance"] = bool(qk_used)
    alpha = bam_mass_alpha()
    if alpha != 1.0:
        audit["bam_mass_alpha"] = alpha
    if bam_alpha_cal_enabled() and "star" in _BAM_LAST_ALPHA:
        audit["bam_alpha_star"] = _BAM_LAST_ALPHA.pop("star")
    if "g" in _BAM_LAST_ALPHA:
        audit["bam_prior_g"] = _BAM_LAST_ALPHA.pop("g")
    if "auto" in _BAM_LAST_ALPHA:
        audit["bam_alpha_auto"] = _BAM_LAST_ALPHA.pop("auto")
    quality = torch.exp(log_a) * m
    return order, quality, audit


# ---------------------------------------------------------------------------
# Top-level selection
# ---------------------------------------------------------------------------


def qd_select_tokens(
    image_embeds: torch.Tensor,
    keep_k: int,
    *,
    instr_query: Optional[torch.Tensor],
    spatial_prior: Optional[torch.Tensor],
    miu_labels: Optional[torch.Tensor],
    grid_h: int,
    grid_w: int,
    flat_indices: Optional[torch.Tensor] = None,
    kappa_tokens: Optional[torch.Tensor] = None,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[List[int], torch.Tensor, Dict[str, Any]]:
    """QD selection over one frame (or a candidate slice of one).

    image_embeds: [n, h] vision rows; instr_query: [h] or [1, h] instruction
    embedding (None -> relevance term uniform 0); spatial_prior: [n] Module A
    density field (None/all-zero -> P term dropped); miu_labels: [n] long
    (-1 = uncovered); flat_indices: [n] original frame-local flat indices when
    the rows are a candidate subset (positions are computed from these, not
    from 0..n-1); kappa_tokens: [n] compensability (None -> no discount).

    Returns (pick order, per-token quality q = exp(log_q), audit). Pick order
    indexes into the given rows (local indices).
    """
    if qd_selector_kind() == "bam":
        # v3 parameter-free path: ignores miu_labels/kappa/params by design
        # (the prior enters as feature mass, redundancy via the Gram kernel).
        order, quality, audit = bam_select_tokens(
            image_embeds,
            keep_k,
            instr_query=instr_query,
            spatial_prior=spatial_prior,
        )
        from lmms_eval.models.geopact_crs import crs_enabled, crs_skeleton_reserve

        if crs_enabled():
            _t = time.perf_counter()
            # Module C v5 single-step operator: re-spend the tail budget on
            # certified regional medoids of the deleted set (real tokens,
            # in-budget). Head picks stay bit-identical.
            order, crs_audit = crs_skeleton_reserve(
                image_embeds, order, keep_k, flat_indices=flat_indices,
                grid_hw=(int(grid_h), int(grid_w)),
            )
            audit.update(crs_audit)
            _qd_stage_record("hes_crs_ms", _t)
        return order, quality, audit
    p = dict(params or qd_params())
    n = int(image_embeds.shape[0])
    device = image_embeds.device
    e_norm = F.normalize(image_embeds.detach().to(torch.float32), dim=-1)

    if instr_query is not None:
        q_vec = instr_query.detach().to(device=device, dtype=torch.float32)
        if q_vec.dim() == 1:
            q_vec = q_vec.unsqueeze(0)
        q_vec = F.normalize(q_vec, dim=-1)
        relevance = (e_norm @ q_vec.t()).amax(dim=-1).clamp(min=0.0)
        if float(relevance.sum()) < 1e-8:
            relevance = torch.zeros(n, dtype=torch.float32, device=device)
    else:
        relevance = torch.zeros(n, dtype=torch.float32, device=device)

    prior = None
    if spatial_prior is not None:
        prior = spatial_prior.detach().to(device=device, dtype=torch.float32)
        if int(prior.shape[0]) != n:
            prior = None

    log_q = fuse_log_quality(
        relevance,
        prior,
        beta_r=float(p["beta_r"]),
        beta_p=float(p["beta_p"]),
        kappa=kappa_tokens,
        gamma_c=float(p.get("gamma_c", 0.0)),
    )

    if flat_indices is not None:
        fi = flat_indices.detach().to(device=device, dtype=torch.long)
        gh = max(1, int(grid_h))
        gw = max(1, int(grid_w))
        r = torch.div(fi, gw, rounding_mode="floor").to(torch.float32)
        c = torch.remainder(fi, gw).to(torch.float32)
        pos = torch.stack([(r + 0.5) / gh, (c + 0.5) / gw], dim=1)
    else:
        pos = grid_positions(n, grid_h, grid_w, device=device)
    labels = None
    if miu_labels is not None:
        labels = miu_labels.detach().to(device=device, dtype=torch.long)
        if int(labels.shape[0]) != n:
            labels = None
    kernel = gui_structured_kernel(
        e_norm,
        pos,
        labels,
        eps=float(p["eps"]),
        sigma=float(p["sigma"]),
        gamma=float(p["gamma"]),
    )

    order, audit = orthogonal_residual_greedy(kernel, log_q, keep_k)
    audit["qd_n"] = n
    audit["qd_keep_k"] = int(keep_k)
    audit["qd_has_prior"] = prior is not None
    audit["qd_has_labels"] = labels is not None
    audit["qd_has_kappa"] = kappa_tokens is not None
    return order, torch.exp(log_q), audit


# ---------------------------------------------------------------------------
# B -> C handoff (master spec §2.5 / §2.6)
# ---------------------------------------------------------------------------


def miu_residual_quality(
    quality: torch.Tensor,
    keep_indices: Sequence[int],
    miu_labels: Optional[torch.Tensor],
) -> Dict[int, float]:
    """rho_m = sum of quality over deleted tokens per MIU (bucket -1 =
    tokens outside every MIU). Consumed by the relay budget allocator.

    Vectorized (scatter_add on device, single host readout) — the naive
    per-token loop cost ~5ms/frame and dominated the small-k perf gate.
    """
    n = int(quality.numel())
    q32 = quality.detach().to(torch.float32)
    device = q32.device
    deleted = torch.ones(n, dtype=torch.bool, device=device)
    if len(keep_indices) > 0:
        keep_t = torch.as_tensor(
            [int(i) for i in keep_indices], dtype=torch.long, device=device
        )
        deleted[keep_t] = False
    out: Dict[int, float] = {}
    if miu_labels is None:
        total = float(q32[deleted].sum())
        if bool(deleted.any()):
            out[-1] = total
        return out
    lab = miu_labels.detach().to(device=device, dtype=torch.long)
    if int(lab.numel()) < n:  # out-of-range tokens fall in the -1 bucket
        lab = torch.cat(
            [lab, torch.full((n - int(lab.numel()),), -1, dtype=torch.long, device=device)]
        )
    idx = lab + 1  # shift so bucket -1 -> slot 0
    n_buckets = int(idx.max().item()) + 1 if n > 0 else 1
    sums = torch.zeros(n_buckets, dtype=torch.float32, device=device)
    counts = torch.zeros(n_buckets, dtype=torch.float32, device=device)
    dmask = deleted.to(torch.float32)
    sums.scatter_add_(0, idx, q32 * dmask)
    counts.scatter_add_(0, idx, dmask)
    sums_h = sums.cpu().tolist()
    counts_h = counts.cpu().tolist()
    for b in range(n_buckets):
        if counts_h[b] > 0.0:
            out[b - 1] = float(sums_h[b])
    return out


def compensability_kappa(
    probe_keys: torch.Tensor,
    miu_labels: torch.Tensor,
    *,
    s_ref: Optional[float] = None,
) -> torch.Tensor:
    """Per-token compensability kappa in [0, 1] from probe-layer keys.

    probe_keys: [H, n, d] post-MRoPE keys of one probe layer (from the frame
    KV snapshot); miu_labels: [n] long (-1 = uncovered -> kappa 0, never
    discounted). kappa_m = 1 / (1 + s_m / s_ref) where s_m is the mean
    per-head key dispersion inside MIU m and s_ref the frame-median
    dispersion (self-calibrating; s_ref may be overridden).
    """
    if probe_keys.dim() != 3:
        raise ValueError(f"probe_keys must be [H, n, d], got {tuple(probe_keys.shape)}")
    n = int(probe_keys.shape[1])
    lab = miu_labels.detach().to(torch.long).cpu()
    if int(lab.numel()) != n:
        raise ValueError(f"miu_labels length {int(lab.numel())} != n {n}")
    k32 = probe_keys.detach().to(torch.float32).cpu()

    disp: Dict[int, float] = {}
    for m in sorted(set(int(x) for x in lab.tolist()) - {-1}):
        idx = (lab == m).nonzero(as_tuple=True)[0]
        if int(idx.numel()) < 2:
            disp[m] = 0.0
            continue
        km = k32.index_select(1, idx)              # [H, m, d]
        centered = km - km.mean(dim=1, keepdim=True)
        disp[m] = float(centered.norm(dim=-1).mean())
    if not disp:
        return torch.zeros(n, dtype=torch.float32)

    ref = float(s_ref) if s_ref is not None else float(
        torch.tensor(sorted(disp.values()))[len(disp) // 2]
    )
    ref = max(ref, 1e-9)
    kappa = torch.zeros(n, dtype=torch.float32)
    for m, s in disp.items():
        kappa[(lab == m)] = 1.0 / (1.0 + s / ref)
    return kappa
