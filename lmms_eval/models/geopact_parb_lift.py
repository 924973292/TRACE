"""PARB-Lift: FA-native logit bias via Q/K rank-r augmentation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from lmms_eval.models.geopact_parb import (
    PARBCtx,
    _beta_intra,
    _beta_inter,
    _eta_tv,
    build_parb_bias,
    parb_density_enabled,
    parb_iou_phi_enabled,
    parb_tv_enabled,
    parb_vv_enabled,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def parb_lift_enabled() -> bool:
    return _env_flag("GEOPACT_PARB_LIFT")


def parb_lift_mode() -> str:
    return (os.environ.get("GEOPACT_PARB_LIFT_MODE") or "decomposed").strip().lower()


def parb_lift_rank() -> int:
    try:
        return max(1, int(os.environ.get("GEOPACT_PARB_LIFT_RANK", "16") or "16"))
    except ValueError:
        return 16


@dataclass
class PARBLiftFeatures:
    u: torch.Tensor  # [N, r] query-side
    v: torch.Tensor  # [N, r] key-side
    rank: int


def _lift_decomposed(ctx: PARBCtx, device: torch.device, rank: int) -> PARBLiftFeatures:
    n = int(ctx.seq_len)
    r = rank
    u = torch.zeros(n, r, device=device, dtype=torch.float32)
    v = torch.zeros(n, r, device=device, dtype=torch.float32)
    is_v = ctx.is_vision.to(device)
    text = ~is_v

    if parb_tv_enabled():
        eta = _eta_tv()
        rp = ctx.region_prior.to(device)
        ps = ctx.psi.to(device)
        u[text, 0] = 1.0
        u[text, 1] = 1.0
        v[is_v, 0] = rp[is_v]
        v[is_v, 1] = eta * ps[is_v]

    if parb_vv_enabled() and bool(is_v.any().item()):
        vis = is_v.nonzero(as_tuple=False).flatten()
        h = ctx.h.to(device)[vis]
        w = ctx.w.to(device)[vis]
        miu = ctx.miu_id.to(device)[vis]
        rho = ctx.rho.to(device)[vis]
        h_span = max(float(h.max() - h.min()), 1.0)
        w_span = max(float(w.max() - w.min()), 1.0)
        hn = (h - h.min()) / h_span
        wn = (w - w.min()) / w_span
        rh = rho + 1e-3 if parb_density_enabled() else torch.ones_like(rho)
        sigma = 0.25
        u[vis, 2] = -hn / (sigma * rh)
        u[vis, 3] = -wn / (sigma * rh)
        v[vis, 2] = hn / sigma
        v[vis, 3] = wn / sigma
        beta = _beta_intra() ** 0.5
        valid = miu >= 0
        u[vis, 4] = valid.float() * beta
        v[vis, 4] = valid.float() * beta
        if parb_iou_phi_enabled() and r > 6:
            inter = (_beta_inter() ** 0.5)
            slots = min(8, r - 6)
            for k in range(int(vis.numel())):
                mj = int(miu[k].item())
                if mj >= 0:
                    slot = mj % slots
                    u[vis[k], 5 + slot] = inter
                    v[vis[k], 5 + slot] = inter

    return PARBLiftFeatures(u=u, v=v, rank=r)


def _lift_svd(ctx: PARBCtx, device: torch.device, rank: int) -> PARBLiftFeatures:
    n = int(ctx.seq_len)
    bias = build_parb_bias(ctx, device=device, dtype=torch.float32)[0, 0]
    if n <= 0:
        return PARBLiftFeatures(
            u=torch.zeros(0, rank, device=device),
            v=torch.zeros(0, rank, device=device),
            rank=rank,
        )
    try:
        u_mat, s, vh = torch.linalg.svd(bias, full_matrices=False)
    except RuntimeError:
        bias_cpu = bias.cpu()
        u_mat, s, vh = torch.linalg.svd(bias_cpu, full_matrices=False)
        u_mat = u_mat.to(device)
        s = s.to(device)
        vh = vh.to(device)
    r = min(rank, int(s.numel()))
    sqrt_s = torch.sqrt(s[:r].clamp_min(0.0))
    u = u_mat[:, :r] * sqrt_s.unsqueeze(0)
    v = vh[:r, :].T * sqrt_s.unsqueeze(0)
    return PARBLiftFeatures(u=u, v=v, rank=r)


def build_parb_lift_features(
    ctx: PARBCtx,
    *,
    device: torch.device,
) -> Optional[PARBLiftFeatures]:
    if not parb_vv_enabled() and not parb_tv_enabled():
        return None
    rank = parb_lift_rank()
    mode = parb_lift_mode()
    if mode == "svd":
        return _lift_svd(ctx, device, rank)
    return _lift_decomposed(ctx, device, rank)


def lift_bias_matrix(feat: PARBLiftFeatures) -> torch.Tensor:
    """Reconstruct bias [N,N] = u @ v^T for verification."""
    return feat.u @ feat.v.T
