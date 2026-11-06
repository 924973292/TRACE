"""DMTC correction context: deleted-mass + text-region calibration."""
from __future__ import annotations

import math
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


def correction_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION")


def correction_lpr_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION_LPR", "1")


def correction_tv_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION_TV", "1")


def correction_region_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION_REGION", "1")


def correction_psi_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION_PSI", "1")


def correction_eta() -> float:
    try:
        return float(os.environ.get("GEOPACT_CORRECTION_ETA", "0.25") or "0.25")
    except ValueError:
        return 0.25


def correction_tau() -> float:
    try:
        return float(os.environ.get("GEOPACT_CORRECTION_TAU", "0.0") or "0.0")
    except ValueError:
        return 0.0


def correction_schedule_enabled() -> bool:
    return _env_flag("GEOPACT_CORRECTION_SCHEDULE", "0")


def correction_layer_eta(layer_idx: int, num_layers: int = 36) -> float:
    eta = correction_eta()
    if not correction_schedule_enabled():
        return eta
    tau = correction_tau()
    if tau <= 0.0:
        tau = max(float(num_layers) / 3.0, 1.0)
    return float(eta * math.exp(-float(layer_idx) / tau))


def layer_active(layer_idx: int) -> bool:
    raw = (os.environ.get("GEOPACT_CORRECTION_LAYERS") or "all").strip().lower()
    if raw in ("all", "*", ""):
        return True
    try:
        allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return int(layer_idx) in allowed
    except ValueError:
        return True


@dataclass
class CorrectionCtx:
    seq_len: int
    is_vision: torch.Tensor
    is_text: torch.Tensor
    Z_deleted: torch.Tensor
    log_Z_deleted: torch.Tensor
    region_prior: torch.Tensor
    psi: torch.Tensor
    prior: torch.Tensor
    frame_start: int
    n_frame: int
    n_pruned: int
    audit: dict


def _build_region_table(mius: Sequence[dict]) -> Dict[int, float]:
    table: Dict[int, float] = {}
    for mi, m in enumerate(mius):
        dens = float(m.get("info_density", m.get("info_density_proxy", 0.0)) or 0.0)
        table[mi] = 0.1 * dens
    return table


def _psi_center(hv: float, wv: float, h_min: float, h_max: float, w_min: float, w_max: float) -> float:
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    nh = (hv - h_min) / h_span
    nw = (wv - w_min) / w_span
    return -float((nh - 0.5) ** 2 + (nw - 0.5) ** 2) ** 0.5


def _get_grid(image_grid_thw, n_frame: int) -> Tuple[int, int]:
    try:
        flat = image_grid_thw.reshape(-1).detach().cpu().tolist()
        if len(flat) >= 3:
            return max(int(flat[-2]) // 2, 1), max(int(flat[-1]) // 2, 1)
    except Exception:
        pass
    side = max(int(round(max(n_frame, 1) ** 0.5)), 1)
    return side, side


def _flat_to_mrope(m: int, grid_h: int, grid_w: int, h_min: float, h_max: float, w_min: float, w_max: float) -> Tuple[float, float]:
    gh = max(int(grid_h), 1)
    gw = max(int(grid_w), 1)
    r = int(m) // gw
    c = int(m) % gw
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    h = h_min + h_span * (float(r) + 0.5) / float(gh)
    w = w_min + w_span * (float(c) + 0.5) / float(gw)
    return h, w


def _local_keep_list(visual_keep_flat: Optional[torch.Tensor], frame_start: int, n_frame: int) -> List[int]:
    if visual_keep_flat is None:
        return []
    out: List[int] = []
    for x in visual_keep_flat.tolist():
        g = int(x)
        local = g - int(frame_start)
        if 0 <= local < int(n_frame):
            out.append(local)
        elif 0 <= g < int(n_frame):
            out.append(g)
    return out


def _compute_deleted_mass(
    *,
    seq_len: int,
    is_text: torch.Tensor,
    is_vision: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    n_frame: int,
    keep_local: Sequence[int],
    grid_h: int,
    grid_w: int,
    eta: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    Z = torch.zeros(seq_len, dtype=torch.float32, device=device)
    log_Z = torch.full((seq_len,), float("-inf"), dtype=torch.float32, device=device)
    if n_frame <= 0:
        return log_Z, Z, 0

    keep = {int(x) for x in keep_local}
    pruned = [m for m in range(int(n_frame)) if m not in keep]
    if not pruned:
        return log_Z, Z, 0

    if bool(is_vision.any().item()):
        h_min = float(h[is_vision].min().item())
        h_max = float(h[is_vision].max().item())
        w_min = float(w[is_vision].min().item())
        w_max = float(w[is_vision].max().item())
    else:
        h_min = h_max = w_min = w_max = 0.0

    ph: List[float] = []
    pw: List[float] = []
    ppsi: List[float] = []
    for m in pruned:
        hm, wm = _flat_to_mrope(m, grid_h, grid_w, h_min, h_max, w_min, w_max)
        ph.append(hm)
        pw.append(wm)
        ppsi.append(_psi_center(hm, wm, h_min, h_max, w_min, w_max))
    ph_t = torch.tensor(ph, dtype=torch.float32, device=device)
    pw_t = torch.tensor(pw, dtype=torch.float32, device=device)
    ppsi_t = torch.tensor(ppsi, dtype=torch.float32, device=device)

    for i in range(seq_len):
        if bool(is_text[i].item()):
            z = torch.exp(eta * ppsi_t).sum()
        elif bool(is_vision[i].item()):
            dh = (h[i] - ph_t).abs() / max((h_max - h_min) * 0.25, 1.0)
            dw = (w[i] - pw_t).abs() / max((w_max - w_min) * 0.25, 1.0)
            z = torch.exp(-dh - dw).sum()
        else:
            z = torch.tensor(0.0, device=device)
        if float(z.item()) > 0.0:
            Z[i] = z
            log_Z[i] = torch.log(z)
    return log_Z, Z, len(pruned)


def _build_flat_to_compact(compact_ids: torch.Tensor, image_tok_id: int, visual_keep_flat: Optional[torch.Tensor], frame_start: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if visual_keep_flat is None:
        return out
    vis_idx = compact_visual_indices(compact_ids, image_tok_id)
    for ord_i, flat_i in enumerate(visual_keep_flat.tolist()):
        if ord_i >= int(vis_idx.numel()):
            break
        ci = int(vis_idx[ord_i].item())
        g = int(flat_i)
        out[g] = ci
        out[g - int(frame_start)] = ci
    return out


def attach_correction_ctx(
    model,
    *,
    compact_ids: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    image_tok_id: int,
    prune_ctx: Optional[dict],
    visual_keep_flat: Optional[torch.Tensor],
    image_grid_thw=None,
) -> Optional[CorrectionCtx]:
    if model is None or compact_pos is None:
        return None
    seq_len = int(compact_ids.shape[1])
    dev = compact_ids.device
    row = compact_ids[0]
    is_vision = row == int(image_tok_id)
    is_text = ~is_vision
    h, w = extract_hw_from_compact_pos(compact_pos)

    frame_start = int((prune_ctx or {}).get("frame_start") or 0)
    n_frame = int((prune_ctx or {}).get("n_frame") or 0)
    mius: Sequence[dict] = (prune_ctx or {}).get("mius") or []
    miu_token_idxs = (prune_ctx or {}).get("miu_token_idxs") or []

    flat_to_compact = _build_flat_to_compact(compact_ids, image_tok_id, visual_keep_flat, frame_start)
    visual_compact_idx = compact_visual_indices(compact_ids, image_tok_id)
    miu_id = build_compact_miu_labels(
        seq_len,
        visual_compact_idx,
        miu_token_idxs=miu_token_idxs,
        global_to_compact=flat_to_compact,
        device=dev,
    )

    region_prior = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    psi = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    region_table = _build_region_table(mius)
    if bool(is_vision.any().item()):
        h_min = float(h[is_vision].min().item())
        h_max = float(h[is_vision].max().item())
        w_min = float(w[is_vision].min().item())
        w_max = float(w[is_vision].max().item())
    else:
        h_min = h_max = w_min = w_max = 0.0
    for j in range(seq_len):
        if not bool(is_vision[j].item()):
            continue
        mj = int(miu_id[j].item())
        if mj >= 0:
            region_prior[j] = float(region_table.get(mj, 0.0))
        psi[j] = _psi_center(float(h[j].item()), float(w[j].item()), h_min, h_max, w_min, w_max)

    eta = correction_eta()
    prior = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    if correction_tv_enabled():
        if correction_region_enabled():
            prior = prior + region_prior
        if correction_psi_enabled():
            prior = prior + eta * psi
    prior = prior * is_vision.float()

    keep_local = _local_keep_list(visual_keep_flat, frame_start, n_frame)
    grid_h, grid_w = _get_grid(image_grid_thw, n_frame)
    log_Z, Z, n_pruned = _compute_deleted_mass(
        seq_len=seq_len,
        is_text=is_text,
        is_vision=is_vision,
        h=h,
        w=w,
        n_frame=n_frame,
        keep_local=keep_local,
        grid_h=grid_h,
        grid_w=grid_w,
        eta=eta,
        device=dev,
    )

    ctx = CorrectionCtx(
        seq_len=seq_len,
        is_vision=is_vision,
        is_text=is_text,
        Z_deleted=Z,
        log_Z_deleted=log_Z,
        region_prior=region_prior,
        psi=psi,
        prior=prior,
        frame_start=frame_start,
        n_frame=n_frame,
        n_pruned=n_pruned,
        audit={
            "correction": True,
            "correction_lpr": correction_lpr_enabled(),
            "correction_tv": correction_tv_enabled(),
            "correction_region": correction_region_enabled(),
            "correction_psi": correction_psi_enabled(),
            "correction_eta": eta,
            "seq_len": seq_len,
            "n_visual": int(is_vision.sum().item()),
            "frame_start": frame_start,
            "n_frame": n_frame,
            "n_pruned": n_pruned,
            "Z_deleted_max": float(Z.max().item()) if Z.numel() else 0.0,
        },
    )

    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_correction_ctx = ctx
        lang._geopact_correction_enabled = True
        layers = getattr(lang, "layers", None)
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn._geopact_language_model_ref = lang
    model._geopact_correction_ctx = ctx
    return ctx


def clear_correction_ctx(model) -> None:
    if model is None:
        return
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_correction_ctx = None
        lang._geopact_correction_enabled = False
    model._geopact_correction_ctx = None
