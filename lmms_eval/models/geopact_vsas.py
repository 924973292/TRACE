"""VSAS — Virtual Spatial Anchor Splice (sampling-measure correction)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from lmms_eval.models.geopact_sampling_correction import (
    anchor_mrope_hw,
    build_anchor_coords,
    build_anchor_hidden,
    compute_omega,
    normalize_visual_hw,
)
from lmms_eval.models.geopact_spatial_rpb import compact_visual_indices, extract_hw_from_compact_pos

VSASMode = Literal["internal", "seq"]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def vsas_enabled() -> bool:
    return _env_flag("GEOPACT_VSAS")


def vsas_mode() -> VSASMode:
    raw = (os.environ.get("GEOPACT_VSAS_MODE") or "internal").strip().lower()
    return "seq" if raw == "seq" else "internal"


def vsas_sigma() -> float:
    try:
        return float(os.environ.get("GEOPACT_VSAS_SIGMA", "0.4") or "0.4")
    except ValueError:
        return 0.4


def vsas_anchor_mode() -> str:
    return (os.environ.get("GEOPACT_VSAS_ANCHORS") or "nine_grid").strip().lower()


def vsas_vision_only() -> bool:
    return _env_flag("GEOPACT_VSAS_VISION_ONLY", "1")


def vsas_denom_only() -> bool:
    return _env_flag("GEOPACT_VSAS_DENOM_ONLY", "0")


def layer_active(layer_idx: int) -> bool:
    raw = (os.environ.get("GEOPACT_VSAS_LAYERS") or "all").strip().lower()
    if raw in ("all", "*", ""):
        return True
    try:
        allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return int(layer_idx) in allowed
    except ValueError:
        return True


@dataclass
class VSASCtx:
    seq_len: int
    seq_base_len: int
    n_virtual: int
    mode: VSASMode
    is_vision: torch.Tensor
    vis_indices: torch.Tensor
    pos_norm: torch.Tensor
    omega: torch.Tensor
    anchor_coords: torch.Tensor
    anchor_h: torch.Tensor
    anchor_w: torch.Tensor
    denom_only: bool
    audit: dict


def _anchor_count(mode: str) -> int:
    return 4 if mode == "four_corners" else 9


def attach_vsas_ctx(
    model,
    *,
    compact_ids: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    image_tok_id: int,
    prune_ctx: Optional[dict],
) -> Optional[VSASCtx]:
    if model is None or compact_pos is None:
        return None
    seq_len = int(compact_ids.shape[1])
    dev = compact_ids.device
    row = compact_ids[0]
    is_vision = row == int(image_tok_id)
    h, w = extract_hw_from_compact_pos(compact_pos)
    pos_norm = normalize_visual_hw(h, w, is_vision)
    vis_indices = compact_visual_indices(compact_ids, image_tok_id)
    if int(vis_indices.numel()) == 0:
        return None

    anchor_mode = vsas_anchor_mode()
    sigma = vsas_sigma()
    anchors = build_anchor_coords(
        "four_corners" if anchor_mode == "four_corners" else "nine_grid",
        device=dev,
        dtype=torch.float32,
    )
    vis_mask = is_vision if vsas_vision_only() else torch.ones(seq_len, dtype=torch.bool, device=dev)
    omega = compute_omega(anchors, pos_norm, sigma=sigma, vis_mask=vis_mask)
    if vsas_vision_only():
        omega = omega[:, vis_indices]

    h_min = float(h[is_vision].min().item()) if is_vision.any() else 0.0
    h_max = float(h[is_vision].max().item()) if is_vision.any() else 1.0
    w_min = float(w[is_vision].min().item()) if is_vision.any() else 0.0
    w_max = float(w[is_vision].max().item()) if is_vision.any() else 1.0
    ah, aw = anchor_mrope_hw(anchors, h_min, h_max, w_min, w_max, device=dev, dtype=h.dtype)

    mode = vsas_mode()
    n_virt = int(anchors.shape[0])
    ctx = VSASCtx(
        seq_len=seq_len,
        seq_base_len=seq_len,
        n_virtual=n_virt,
        mode=mode,
        is_vision=is_vision,
        vis_indices=vis_indices,
        pos_norm=pos_norm,
        omega=omega,
        anchor_coords=anchors,
        anchor_h=ah,
        anchor_w=aw,
        denom_only=vsas_denom_only(),
        audit={
            "vsas": True,
            "vsas_mode": mode,
            "vsas_sigma": sigma,
            "vsas_anchors": anchor_mode,
            "vsas_k": n_virt,
            "vsas_vision_only": vsas_vision_only(),
            "vsas_denom_only": vsas_denom_only(),
            "seq_len": seq_len,
            "n_visual": int(is_vision.sum().item()),
        },
    )
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_vsas_ctx = ctx
        lang._geopact_vsas_enabled = True
        layers = getattr(lang, "layers", None)
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn._geopact_language_model_ref = lang
    model._geopact_vsas_ctx = ctx
    return ctx


def clear_vsas_ctx(model) -> None:
    if model is None:
        return
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_vsas_ctx = None
        lang._geopact_vsas_enabled = False
    model._geopact_vsas_ctx = None


def splice_vsas_into_compact(
    compact_ids: torch.Tensor,
    compact_embs: torch.Tensor,
    compact_attn: Optional[torch.Tensor],
    compact_pos: torch.Tensor,
    compact_rope: torch.Tensor,
    ctx: VSASCtx,
    image_tok_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, VSASCtx]:
    """Path B: append K virtual vision tokens at sequence end."""
    dev = compact_embs.device
    vis_idx = ctx.vis_indices.to(dev)
    x_vis = compact_embs.index_select(1, vis_idx)
    x_anchor = build_anchor_hidden(compact_embs, ctx.omega.to(dev), vis_indices=vis_idx)
    k = int(x_anchor.shape[1])
    base_len = int(compact_ids.shape[1])

    virt_ids = torch.full((1, k), int(image_tok_id), dtype=compact_ids.dtype, device=dev)
    compact_ids = torch.cat([compact_ids, virt_ids], dim=1)
    compact_embs = torch.cat([compact_embs, x_anchor.to(compact_embs.dtype)], dim=1)
    if compact_attn is not None:
        ones = torch.ones(1, k, dtype=compact_attn.dtype, device=dev)
        compact_attn = torch.cat([compact_attn, ones], dim=1)

    t_row = compact_pos[0, 0, 0:1].expand(1, 1, k)
    h_row = ctx.anchor_h.view(1, 1, k).to(device=compact_pos.device, dtype=compact_pos.dtype)
    w_row = ctx.anchor_w.view(1, 1, k).to(device=compact_pos.device, dtype=compact_pos.dtype)
    extra = torch.cat([t_row, h_row, w_row], dim=0)
    compact_pos = torch.cat([compact_pos, extra], dim=2)

    batch = compact_ids.shape[0]
    seq_row = compact_ids.shape[1]
    deltas = [int(compact_pos[:, b].max().item()) + 1 - seq_row for b in range(batch)]
    compact_rope = torch.tensor(deltas, device=compact_pos.device, dtype=torch.long).unsqueeze(1)

    ctx = VSASCtx(
        seq_len=int(compact_ids.shape[1]),
        seq_base_len=base_len,
        n_virtual=k,
        mode="seq",
        is_vision=ctx.is_vision,
        vis_indices=ctx.vis_indices,
        pos_norm=ctx.pos_norm,
        omega=ctx.omega,
        anchor_coords=ctx.anchor_coords,
        anchor_h=ctx.anchor_h,
        anchor_w=ctx.anchor_w,
        denom_only=ctx.denom_only,
        audit={**ctx.audit, "vsas_seq_spliced": True, "seq_base_len": base_len},
    )
    return compact_ids, compact_embs, compact_attn, compact_pos, compact_rope, ctx
