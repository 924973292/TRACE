"""PARB — Partition-aware Attention Bias (legacy ablation; use geopact_trp for production)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from lmms_eval.models.geopact_spatial_rpb import (
    build_compact_miu_labels,
    compact_visual_indices,
    extract_hw_from_compact_pos,
    _box_from_miu,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def parb_enabled() -> bool:
    return _env_flag("GEOPACT_PARB")


def parb_mode() -> str:
    return (os.environ.get("GEOPACT_PARB_MODE") or "hybrid").strip().lower()


def parb_vv_enabled() -> bool:
    return _env_flag("GEOPACT_PARB_VV", "1")


def parb_tv_enabled() -> bool:
    return _env_flag("GEOPACT_PARB_TV", "1")


def parb_density_enabled() -> bool:
    return _env_flag("GEOPACT_PARB_DENSITY", "1")


def parb_iou_phi_enabled() -> bool:
    return _env_flag("GEOPACT_PARB_IOU_PHI", "1")


def _beta_intra() -> float:
    try:
        return float(os.environ.get("GEOPACT_PARB_BETA_INTRA", "0.5") or "0.5")
    except ValueError:
        return 0.5


def _beta_inter() -> float:
    try:
        return float(os.environ.get("GEOPACT_PARB_BETA_INTER", "0.3") or "0.3")
    except ValueError:
        return 0.3


def _eta_tv() -> float:
    try:
        return float(os.environ.get("GEOPACT_PARB_ETA_TV", "0.25") or "0.25")
    except ValueError:
        return 0.25


def layer_active(layer_idx: int) -> bool:
    raw = (os.environ.get("GEOPACT_PARB_LAYERS") or "all").strip().lower()
    if raw in ("all", "*", ""):
        return True
    try:
        allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return int(layer_idx) in allowed
    except ValueError:
        return True


@dataclass
class PARBCtx:
    seq_len: int
    is_vision: torch.Tensor  # [N] bool
    h: torch.Tensor  # [N] float32
    w: torch.Tensor  # [N] float32
    miu_id: torch.Tensor  # [N] long, -1 text
    rho: torch.Tensor  # [N] float32, 0 for text
    region_prior: torch.Tensor  # [N] float32 key-side r(m_j)
    psi: torch.Tensor  # [N] float32 key-side geometry
    mius: Tuple[dict, ...]
    audit: dict


def _phi_iou(x: float) -> float:
    import math

    return math.log1p(max(0.0, x))


def _build_region_table(mius: Sequence[dict]) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for mi, m in enumerate(mius):
        dens = float(m.get("info_density", m.get("info_density_proxy", 0.0)) or 0.0)
        table[mi] = 0.1 * dens
    return table


def _psi_key(hv: float, wv: float, h_min: float, h_max: float, w_min: float, w_max: float) -> float:
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    nh = (hv - h_min) / h_span
    nw = (wv - w_min) / w_span
    cx, cy = 0.5, 0.5
    return -float((nh - cx) ** 2 + (nw - cy) ** 2) ** 0.5


def attach_parb_ctx(
    model,
    *,
    compact_ids: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    image_tok_id: int,
    prune_ctx: Optional[dict],
    visual_keep_flat: Optional[torch.Tensor],
) -> Optional[PARBCtx]:
    """Build PARB context and attach to language_model layers."""
    if model is None or compact_pos is None:
        return None
    seq_len = int(compact_ids.shape[1])
    dev = compact_ids.device
    row = compact_ids[0]
    is_vision = row == int(image_tok_id)
    h, w = extract_hw_from_compact_pos(compact_pos)
    miu_id = torch.full((seq_len,), -1, dtype=torch.long, device=dev)
    rho = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    region_prior = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    psi = torch.zeros(seq_len, dtype=torch.float32, device=dev)

    mius: Sequence[dict] = []
    if prune_ctx:
        mius = prune_ctx.get("mius") or []
        miu_token_idxs = prune_ctx.get("miu_token_idxs") or []
        frame_start = int(prune_ctx.get("frame_start") or 0)
        visual_compact_idx = compact_visual_indices(compact_ids, image_tok_id)
        global_to_compact: Dict[int, int] = {}
        if visual_keep_flat is not None:
            vis_idx = compact_visual_indices(compact_ids, image_tok_id)
            for ord_i, flat_i in enumerate(visual_keep_flat.tolist()):
                if ord_i < int(vis_idx.numel()):
                    ci = int(vis_idx[ord_i].item())
                    global_to_compact[int(flat_i)] = ci
                    global_to_compact[int(flat_i) - frame_start] = ci
        miu_id = build_compact_miu_labels(
            seq_len,
            visual_compact_idx,
            miu_token_idxs=miu_token_idxs,
            global_to_compact=global_to_compact,
            device=dev,
        )
        density_flat = prune_ctx.get("cdct_density_flat")
        if density_flat is not None and visual_keep_flat is not None:
            for ord_i, flat_i in enumerate(visual_keep_flat.tolist()):
                ci = global_to_compact.get(int(flat_i))
                if ci is not None and 0 <= ci < seq_len:
                    fi = int(flat_i) - frame_start
                    if not (0 <= fi < int(density_flat.numel())):
                        fi = int(flat_i)
                    if 0 <= fi < int(density_flat.numel()):
                        rho[ci] = float(density_flat[fi].item())

    region_table = _build_region_table(mius)
    h_min = float(h[is_vision].min().item()) if is_vision.any() else 0.0
    h_max = float(h[is_vision].max().item()) if is_vision.any() else 1.0
    w_min = float(w[is_vision].min().item()) if is_vision.any() else 0.0
    w_max = float(w[is_vision].max().item()) if is_vision.any() else 1.0

    for j in range(seq_len):
        if not bool(is_vision[j].item()):
            continue
        mj = int(miu_id[j].item())
        if mj >= 0:
            region_prior[j] = float(region_table.get(mj, 0.0))
        psi[j] = _psi_key(
            float(h[j].item()),
            float(w[j].item()),
            h_min,
            h_max,
            w_min,
            w_max,
        )

    ctx = PARBCtx(
        seq_len=seq_len,
        is_vision=is_vision,
        h=h.to(torch.float32),
        w=w.to(torch.float32),
        miu_id=miu_id,
        rho=rho,
        region_prior=region_prior,
        psi=psi,
        mius=tuple(mius),
        audit={
            "parb": True,
            "parb_mode": parb_mode(),
            "parb_vv": parb_vv_enabled(),
            "parb_tv": parb_tv_enabled(),
            "parb_density": parb_density_enabled(),
            "seq_len": seq_len,
            "n_visual": int(is_vision.sum().item()),
        },
    )

    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_parb_ctx = ctx
        lang._geopact_parb_enabled = True
        layers = getattr(lang, "layers", None)
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn._geopact_language_model_ref = lang
    model._geopact_parb_ctx = ctx
    return ctx


def clear_parb_ctx(model) -> None:
    if model is None:
        return
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_parb_ctx = None
        lang._geopact_parb_enabled = False
    model._geopact_parb_ctx = None


def build_parb_bias(
    ctx: PARBCtx,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return additive bias [1, 1, N, N] (0 where unused). Vectorized for FA-hybrid prefill."""
    n = int(ctx.seq_len)
    bias = torch.zeros(1, 1, n, n, device=device, dtype=torch.float32)
    if n <= 0:
        return bias.to(dtype=dtype)

    is_v = ctx.is_vision.to(device)
    h = ctx.h.to(device)
    w = ctx.w.to(device)
    miu = ctx.miu_id.to(device)
    rho = ctx.rho.to(device)
    eps = 1e-3

    vis_idx = is_v.nonzero(as_tuple=False).flatten()
    if vis_idx.numel() > 0 and parb_vv_enabled():
        hv = h.index_select(0, vis_idx)
        wv = w.index_select(0, vis_idx)
        lv = miu.index_select(0, vis_idx)
        h_span = max(float(hv.max() - hv.min()), 1.0)
        w_span = max(float(wv.max() - wv.min()), 1.0)
        sigma_h = max(h_span * 0.25, 1.0)
        sigma_w = max(w_span * 0.25, 1.0)

        dh = (hv.unsqueeze(1) - hv.unsqueeze(0)).abs() / sigma_h
        dw = (wv.unsqueeze(1) - wv.unsqueeze(0)).abs() / sigma_w
        if parb_density_enabled():
            ri = rho.index_select(0, vis_idx).unsqueeze(1) + eps
            rj = rho.index_select(0, vis_idx).unsqueeze(0) + eps
        else:
            ri = rj = 1.0
        bvv = -(dh / ri + dw / rj)

        same_miu = (lv.unsqueeze(1) == lv.unsqueeze(0)) & (lv.unsqueeze(1) >= 0)
        bvv = bvv + same_miu.float() * _beta_intra()

        if parb_iou_phi_enabled():
            v = int(vis_idx.numel())
            boxes = torch.zeros(v, 4, device=device, dtype=torch.float32)
            h_min, w_min = float(hv.min()), float(wv.min())
            for k in range(v):
                mi = int(lv[k].item())
                if mi >= 0 and mi < len(ctx.mius):
                    bx = _box_from_miu(ctx.mius[mi])
                    boxes[k] = torch.tensor(bx, device=device, dtype=torch.float32)
                else:
                    nh = (float(hv[k]) - h_min) / h_span if h_span else 0.5
                    nw = (float(wv[k]) - w_min) / w_span if w_span else 0.5
                    boxes[k] = torch.tensor(
                        (nw - 1e-4, nh - 1e-4, nw + 1e-4, nh + 1e-4),
                        device=device,
                        dtype=torch.float32,
                    )
            ix1 = torch.maximum(boxes[:, 0].unsqueeze(1), boxes[:, 0].unsqueeze(0))
            iy1 = torch.maximum(boxes[:, 1].unsqueeze(1), boxes[:, 1].unsqueeze(0))
            ix2 = torch.minimum(boxes[:, 2].unsqueeze(1), boxes[:, 2].unsqueeze(0))
            iy2 = torch.minimum(boxes[:, 3].unsqueeze(1), boxes[:, 3].unsqueeze(0))
            inter = (ix2 - ix1).clamp_min(0.0) * (iy2 - iy1).clamp_min(0.0)
            area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
            union = area.unsqueeze(1) + area.unsqueeze(0) - inter
            iou = torch.where(union > 0, inter / union.clamp_min(1e-9), torch.zeros_like(inter))
            diff_miu = (lv.unsqueeze(1) != lv.unsqueeze(0)) & (lv.unsqueeze(1) >= 0) & (lv.unsqueeze(0) >= 0)
            bvv = bvv + diff_miu.float() * (_beta_inter() * torch.log1p(iou.clamp_min(0.0)))

        ii = vis_idx.unsqueeze(1)
        jj = vis_idx.unsqueeze(0)
        bias[0, 0, ii, jj] = bvv

    if parb_tv_enabled():
        eta = _eta_tv()
        text_idx = (~is_v).nonzero(as_tuple=False).flatten()
        if text_idx.numel() > 0 and vis_idx.numel() > 0:
            tv_key = ctx.region_prior.to(device) + eta * ctx.psi.to(device)
            tv_row = tv_key.index_select(0, vis_idx)
            bias[0, 0, text_idx.unsqueeze(1), vis_idx.unsqueeze(0)] = tv_row.unsqueeze(0)

    return bias.to(dtype=dtype)
