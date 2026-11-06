"""TRP — Text-Region Prior: key-side logit bias for pruned GUI VLM (TV-only, FA-native)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from lmms_eval.models.geopact_spatial_rpb import (
    build_compact_miu_labels,
    compact_visual_indices,
    extract_hw_from_compact_pos,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def trp_enabled() -> bool:
    return _env_flag("GEOPACT_TRP")


def trp_mode() -> str:
    return (os.environ.get("GEOPACT_TRP_MODE") or "lift").strip().lower()


def trp_eta() -> float:
    try:
        return float(os.environ.get("GEOPACT_TRP_ETA", "0.25") or "0.25")
    except ValueError:
        return 0.25


def trp_region_enabled() -> bool:
    return _env_flag("GEOPACT_TRP_REGION", "1")


def trp_psi_enabled() -> bool:
    return _env_flag("GEOPACT_TRP_PSI", "1")


def layer_active(layer_idx: int) -> bool:
    raw = (os.environ.get("GEOPACT_TRP_LAYERS") or "all").strip().lower()
    if raw in ("all", "*", ""):
        return True
    try:
        allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return int(layer_idx) in allowed
    except ValueError:
        return True


@dataclass
class TRPCtx:
    seq_len: int
    is_vision: torch.Tensor  # [N] bool
    is_text: torch.Tensor  # [N] bool
    region_prior: torch.Tensor  # [N] r(m_j)
    psi: torch.Tensor  # [N] centrality
    prior: torch.Tensor  # [N] pi_j = region + eta*psi on vision, 0 else
    audit: dict


@dataclass
class TRPLiftFeatures:
    u: torch.Tensor  # [N, 2]
    v: torch.Tensor  # [N, 2]
    rank: int = 2


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
    return -float((nh - 0.5) ** 2 + (nw - 0.5) ** 2) ** 0.5


def build_trp_prior(ctx: TRPCtx, *, device: torch.device) -> torch.Tensor:
    """Key-side prior pi_j [N]: region + eta*psi on vision tokens, 0 elsewhere."""
    return ctx.prior.to(device=device, dtype=torch.float32)


def build_trp_bias(
    ctx: TRPCtx,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive bias [1,1,N,N]: text query x vision key only."""
    n = int(ctx.seq_len)
    bias = torch.zeros(1, 1, n, n, device=device, dtype=torch.float32)
    if n <= 0:
        return bias.to(dtype=dtype)

    is_v = ctx.is_vision.to(device)
    text_idx = (~is_v).nonzero(as_tuple=False).flatten()
    vis_idx = is_v.nonzero(as_tuple=False).flatten()
    if text_idx.numel() == 0 or vis_idx.numel() == 0:
        return bias.to(dtype=dtype)

    pi = build_trp_prior(ctx, device=device)
    tv_row = pi.index_select(0, vis_idx)
    bias[0, 0, text_idx.unsqueeze(1), vis_idx.unsqueeze(0)] = tv_row.unsqueeze(0)
    return bias.to(dtype=dtype)


def build_trp_lift_features(ctx: TRPCtx, *, device: torch.device) -> TRPLiftFeatures:
    """Exact rank-2 lift: u_text=[1,1], v_vis=[r(m), eta*psi]."""
    n = int(ctx.seq_len)
    u = torch.zeros(n, 2, device=device, dtype=torch.float32)
    v = torch.zeros(n, 2, device=device, dtype=torch.float32)
    eta = trp_eta()
    rp = ctx.region_prior.to(device)
    ps = ctx.psi.to(device)
    text = ctx.is_text.to(device)
    vis = ctx.is_vision.to(device)

    if trp_region_enabled():
        u[text, 0] = 1.0
        v[vis, 0] = rp[vis]
    if trp_psi_enabled():
        u[text, 1] = 1.0
        v[vis, 1] = eta * ps[vis]

    return TRPLiftFeatures(u=u, v=v, rank=2)


def attach_trp_ctx(
    model,
    *,
    compact_ids: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    image_tok_id: int,
    prune_ctx: Optional[dict],
    visual_keep_flat: Optional[torch.Tensor],
) -> Optional[TRPCtx]:
    if model is None or compact_pos is None:
        return None
    seq_len = int(compact_ids.shape[1])
    dev = compact_ids.device
    row = compact_ids[0]
    is_vision = row == int(image_tok_id)
    is_text = ~is_vision
    h, w = extract_hw_from_compact_pos(compact_pos)
    miu_id = torch.full((seq_len,), -1, dtype=torch.long, device=dev)
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

    eta = trp_eta()
    prior = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    if trp_region_enabled():
        prior = prior + region_prior
    if trp_psi_enabled():
        prior = prior + eta * psi
    prior = prior * is_vision.float()

    ctx = TRPCtx(
        seq_len=seq_len,
        is_vision=is_vision,
        is_text=is_text,
        region_prior=region_prior,
        psi=psi,
        prior=prior,
        audit={
            "trp": True,
            "trp_mode": trp_mode(),
            "trp_eta": eta,
            "trp_region": trp_region_enabled(),
            "trp_psi": trp_psi_enabled(),
            "seq_len": seq_len,
            "n_visual": int(is_vision.sum().item()),
        },
    )

    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_trp_ctx = ctx
        lang._geopact_trp_enabled = True
        layers = getattr(lang, "layers", None)
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn._geopact_language_model_ref = lang
    model._geopact_trp_ctx = ctx
    return ctx


def clear_trp_ctx(model) -> None:
    if model is None:
        return
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_trp_ctx = None
        lang._geopact_trp_enabled = False
    model._geopact_trp_ctx = None
