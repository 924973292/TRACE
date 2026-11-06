"""Training-free spatial geometry for pruned survivors (Flash-Attention safe).

Track-1 implements *embedding-space 2D-RPB* (ES-RPB): a single geometry-weighted
refinement of visual token embeddings. This preserves ``flash_attention_2`` — no
attention-logit hooks and no SDPA/eager fallback.

  e_i <- e_i + lambda * sum_j softmax(B_ij) * (e_j - e_i)

where B_ij uses origpos (h,w) from ``compact_pos``, optional MIU block bonus, and
optional MIU bbox IoU (G²-RPB C2). CDCT residue multiplies visual embeddings by
(1 + alpha_res * rho_i).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def spatial_rpb_enabled() -> bool:
    return _env_flag("GEOPACT_SPATIAL_RPB")


def cdct_residue_enabled() -> bool:
    return _env_flag("GEOPACT_CDCT_RESIDUE")


def rpb_shuffle_positions() -> bool:
    return _env_flag("GEOPACT_RPB_SHUFFLE_POS")


def rpb_shuffle_miu() -> bool:
    return _env_flag("GEOPACT_RPB_SHUFFLE_MIU")


def rpb_iou_enabled() -> bool:
    return _env_flag("GEOPACT_RPB_IOU")


def rpb_manhattan_enabled() -> bool:
    return _env_flag("GEOPACT_RPB_MANHATTAN", "1")


def miu_block_bias_enabled() -> bool:
    return _env_flag("GEOPACT_RPB_MIU_BLOCK", "1")


def _rpb_lambda() -> float:
    try:
        return float(os.environ.get("GEOPACT_SPATIAL_RPB_LAMBDA", "0.15") or "0.15")
    except ValueError:
        return 0.15


def _miu_block_beta() -> float:
    try:
        return float(os.environ.get("GEOPACT_RPB_MIU_BETA", "0.5") or "0.5")
    except ValueError:
        return 0.5


def _rpb_iou_beta() -> float:
    try:
        return float(os.environ.get("GEOPACT_RPB_IOU_BETA", "0.5") or "0.5")
    except ValueError:
        return 0.5


def _cdct_residue_alpha() -> float:
    raw = os.environ.get("GEOPACT_CDCT_RESIDUE_ALPHA", "")
    if raw.strip():
        try:
            return float(raw)
        except ValueError:
            pass
    try:
        return float(os.environ.get("GEOPACT_CDCT_ALPHA", "0.2") or "0.2")
    except ValueError:
        return 0.2


def extract_hw_from_compact_pos(compact_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-seq (h, w) from Qwen3 MRoPE position_ids [3, batch, seq]."""
    if compact_pos.ndim != 3:
        raise ValueError(f"compact_pos must be 3D, got {compact_pos.shape}")
    h = compact_pos[1, 0, :].to(dtype=torch.float32)
    w = compact_pos[2, 0, :].to(dtype=torch.float32)
    return h, w


def compact_visual_indices(compact_ids: torch.Tensor, image_tok_id: int) -> torch.Tensor:
    row = compact_ids[0]
    if image_tok_id < 0:
        return torch.empty(0, dtype=torch.long, device=row.device)
    mask = row == int(image_tok_id)
    return mask.nonzero(as_tuple=False).flatten()


def build_compact_miu_labels(
    seq_len: int,
    visual_compact_idx: torch.Tensor,
    *,
    miu_token_idxs: Sequence[Sequence[int]],
    global_to_compact: Dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Per compact position MIU id (-1 = non-visual / unassigned)."""
    labels = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for mi, idxs in enumerate(miu_token_idxs):
        for g in idxs:
            ci = global_to_compact.get(int(g))
            if ci is not None:
                labels[ci] = int(mi)
    for ci in visual_compact_idx.tolist():
        if int(labels[int(ci)]) < 0:
            labels[int(ci)] = -2
    return labels


def _box_from_miu(m: dict) -> Tuple[float, float, float, float]:
    box = m.get("box")
    if box and len(box) >= 4:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    cx = float(m.get("cx", 0.5))
    cy = float(m.get("cy", 0.5))
    eps = 1e-4
    return (cx - eps, cy - eps, cx + eps, cy + eps)


def _point_box_from_hw(hv: float, wv: float, h_min: float, h_max: float, w_min: float, w_max: float) -> Tuple[float, float, float, float]:
    eps = 1e-4
    if h_max > h_min:
        nh = (hv - h_min) / (h_max - h_min)
    else:
        nh = 0.5
    if w_max > w_min:
        nw = (wv - w_min) / (w_max - w_min)
    else:
        nw = 0.5
    nh = max(0.0, min(1.0, nh))
    nw = max(0.0, min(1.0, nw))
    return (nw - eps, nh - eps, nw + eps, nh + eps)


def _iou_boxes(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _visual_boxes(
    hv: torch.Tensor,
    wv: torch.Tensor,
    lv: torch.Tensor,
    mius: Optional[Sequence[dict]],
) -> List[Tuple[float, float, float, float]]:
    h_min = float(hv.min().item())
    h_max = float(hv.max().item())
    w_min = float(wv.min().item())
    w_max = float(wv.max().item())
    boxes: List[Tuple[float, float, float, float]] = []
    for i in range(int(hv.numel())):
        mi = int(lv[i].item())
        if mius and 0 <= mi < len(mius):
            boxes.append(_box_from_miu(mius[mi]))
        else:
            boxes.append(
                _point_box_from_hw(
                    float(hv[i].item()),
                    float(wv[i].item()),
                    h_min,
                    h_max,
                    w_min,
                    w_max,
                )
            )
    return boxes


def _build_iou_matrix(boxes: List[Tuple[float, float, float, float]], device: torch.device) -> torch.Tensor:
    n = len(boxes)
    if n <= 0:
        return torch.zeros(0, 0, device=device, dtype=torch.float32)
    mat = torch.zeros(n, n, device=device, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = _iou_boxes(boxes[i], boxes[j])
    return mat


def _build_es_rpb_weights(
    h: torch.Tensor,
    w: torch.Tensor,
    visual_idx: torch.Tensor,
    miu_labels: Optional[torch.Tensor],
    *,
    shuffle: bool = False,
    shuffle_miu: bool = False,
    mius: Optional[Sequence[dict]] = None,
) -> torch.Tensor:
    """Return [n_vis, n_vis] softmax weights for ES-RPB mixing."""
    n = int(visual_idx.numel())
    if n <= 1:
        return torch.eye(max(n, 1), device=h.device, dtype=torch.float32)

    hv = h.index_select(0, visual_idx)
    wv = w.index_select(0, visual_idx)
    lv = None
    if miu_labels is not None:
        lv = miu_labels.index_select(0, visual_idx).clone()

    if shuffle:
        # v2: independent h/w permutation breaks grid coherence (content-geometry binding).
        perm_h = torch.randperm(n, device=h.device)
        perm_w = torch.randperm(n, device=h.device)
        hv = hv.index_select(0, perm_h)
        wv = wv.index_select(0, perm_w)

    if shuffle_miu and lv is not None:
        perm_m = torch.randperm(n, device=h.device)
        lv = lv.index_select(0, perm_m)

    h_span = max(float(hv.max() - hv.min()), 1.0)
    w_span = max(float(wv.max() - wv.min()), 1.0)
    sigma_h = max(h_span * 0.25, 1.0)
    sigma_w = max(w_span * 0.25, 1.0)

    bias = torch.zeros(n, n, device=h.device, dtype=torch.float32)
    if rpb_manhattan_enabled():
        dh = (hv.unsqueeze(1) - hv.unsqueeze(0)).abs() / sigma_h
        dw = (wv.unsqueeze(1) - wv.unsqueeze(0)).abs() / sigma_w
        bias = bias - (dh + dw)

    if lv is not None and miu_block_bias_enabled():
        same = (lv.unsqueeze(1) == lv.unsqueeze(0)) & (lv.unsqueeze(1) >= 0)
        bias = bias + same.to(dtype=bias.dtype) * _miu_block_beta()

    if rpb_iou_enabled() and lv is not None:
        boxes = _visual_boxes(hv, wv, lv, mius)
        iou_mat = _build_iou_matrix(boxes, h.device)
        bias = bias + iou_mat.to(dtype=bias.dtype) * _rpb_iou_beta()

    bias = bias - bias.max(dim=1, keepdim=True).values
    weights = torch.exp(bias)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights


def apply_embedding_space_rpb(
    compact_embs: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    visual_compact_idx: torch.Tensor,
    miu_labels: Optional[torch.Tensor] = None,
    mius: Optional[Sequence[dict]] = None,
) -> torch.Tensor:
    """Flash-safe ES-RPB on visual token embeddings only."""
    if compact_pos is None or visual_compact_idx.numel() == 0:
        return compact_embs
    h, w = extract_hw_from_compact_pos(compact_pos)
    weights = _build_es_rpb_weights(
        h,
        w,
        visual_compact_idx,
        miu_labels,
        shuffle=rpb_shuffle_positions(),
        shuffle_miu=rpb_shuffle_miu(),
        mius=mius,
    )
    lam = _rpb_lambda()
    if lam <= 0.0:
        return compact_embs

    out = compact_embs.clone()
    vis = compact_embs[0].index_select(0, visual_compact_idx)
    w_dtype = vis.dtype
    weights = weights.to(device=vis.device, dtype=w_dtype)
    mixed = torch.matmul(weights, vis)
    refined = vis + lam * (mixed - vis)
    out[0, visual_compact_idx] = refined
    return out


def apply_cdct_residue(
    compact_embs: torch.Tensor,
    visual_compact_idx: torch.Tensor,
    flat_visual_idx: torch.Tensor,
    density_flat: Optional[torch.Tensor],
) -> torch.Tensor:
    """Multiply visual embeddings by (1 + alpha * rho) using flat-index density."""
    if density_flat is None or visual_compact_idx.numel() == 0:
        return compact_embs
    alpha = _cdct_residue_alpha()
    if alpha <= 0.0:
        return compact_embs

    out = compact_embs.clone()
    rho = density_flat.index_select(0, flat_visual_idx.to(density_flat.device))
    scale = (1.0 + alpha * rho).to(device=out.device, dtype=out.dtype).unsqueeze(-1)
    out[0, visual_compact_idx] = out[0, visual_compact_idx] * scale
    return out


def apply_spatial_enhancements(
    compact_embs: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    compact_ids: torch.Tensor,
    image_tok_id: int,
    *,
    prune_ctx: Optional[dict],
    visual_keep_flat: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, dict]:
    """Apply ES-RPB and/or CDCT residue; return (embs, audit)."""
    audit: dict = {
        "spatial_rpb": False,
        "cdct_residue": False,
        "es_rpb_lambda": _rpb_lambda(),
        "cdct_residue_alpha": _cdct_residue_alpha(),
        "rpb_shuffle": rpb_shuffle_positions(),
        "rpb_shuffle_miu": rpb_shuffle_miu(),
        "rpb_iou": rpb_iou_enabled(),
        "rpb_manhattan": rpb_manhattan_enabled(),
        "rpb_iou_beta": _rpb_iou_beta(),
    }
    visual_compact_idx = compact_visual_indices(compact_ids, image_tok_id)
    if visual_compact_idx.numel() == 0:
        return compact_embs, audit

    miu_labels = None
    mius: Optional[List[dict]] = None
    density_flat = None
    flat_for_vis = None

    if prune_ctx:
        miu_token_idxs = prune_ctx.get("miu_token_idxs") or []
        mius = list(prune_ctx.get("mius") or [])
        frame_start = int(prune_ctx.get("frame_start", 0) or 0)
        density_flat = prune_ctx.get("cdct_density_flat")
        global_to_compact: Dict[int, int] = {}
        if visual_keep_flat is not None:
            for local_i, flat_i in enumerate(visual_keep_flat.tolist()):
                seq_pos = prune_ctx.get("flat_to_compact", {}).get(int(flat_i))
                if seq_pos is None and visual_compact_idx.numel() > local_i:
                    seq_pos = int(visual_compact_idx[local_i].item())
                if seq_pos is not None:
                    global_to_compact[int(flat_i) + frame_start] = int(seq_pos)
                    global_to_compact[int(flat_i)] = int(seq_pos)
            flat_for_vis = visual_keep_flat
        miu_labels = build_compact_miu_labels(
            int(compact_ids.shape[1]),
            visual_compact_idx,
            miu_token_idxs=miu_token_idxs,
            global_to_compact=global_to_compact,
            device=compact_embs.device,
        )

    if spatial_rpb_enabled():
        compact_embs = apply_embedding_space_rpb(
            compact_embs,
            compact_pos,
            visual_compact_idx,
            miu_labels,
            mius=mius,
        )
        audit["spatial_rpb"] = True

    if cdct_residue_enabled() and density_flat is not None and flat_for_vis is not None:
        compact_embs = apply_cdct_residue(
            compact_embs, visual_compact_idx, flat_for_vis, density_flat
        )
        audit["cdct_residue"] = True

    return compact_embs, audit
