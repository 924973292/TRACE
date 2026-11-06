"""GSC — Grounding-oriented Spatial Calibration: VFS (flux sink) + SPF (saliency potential)."""
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
def _spatial_merge_size() -> int:
    try:
        return int(os.environ.get("GEOPACT_SPATIAL_MERGE_SIZE", "2") or "2")
    except ValueError:
        return 2


def _get_grid(image_grid_thw, n_img: int) -> Tuple[int, int, bool]:
    """Return (grid_h, grid_w, grid_matches) in LLM token grid coordinates."""
    if image_grid_thw is None or image_grid_thw.numel() < 3:
        return 1, 1, False
    merge = _spatial_merge_size()
    thw = image_grid_thw[-1].tolist()
    raw_h, raw_w = int(thw[1]), int(thw[2])
    grid_h, grid_w = raw_h // merge, raw_w // merge
    if grid_h * grid_w == n_img:
        return grid_h, grid_w, True
    if raw_h * raw_w == n_img:
        return raw_h, raw_w, True
    return grid_h, grid_w, False


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def gsc_enabled() -> bool:
    return _env_flag("GEOPACT_GSC")


def gsc_mode() -> str:
    return (os.environ.get("GEOPACT_GSC_MODE") or "fa").strip().lower()


def gsc_uses_fa_path() -> bool:
    return gsc_mode() in ("fa", "icc_lpr", "lpr", "")


def gsc_vfs_enabled() -> bool:
    return _env_flag("GEOPACT_GSC_VFS", "1")


def gsc_spf_mode() -> str:
    """off | center | full (center + survival)."""
    return (os.environ.get("GEOPACT_GSC_SPF") or "center").strip().lower()


def gsc_eta0() -> float:
    try:
        return float(os.environ.get("GEOPACT_GSC_ETA0", "0.25") or "0.25")
    except ValueError:
        return 0.25


def gsc_tau() -> float:
    try:
        return float(os.environ.get("GEOPACT_GSC_TAU", "0") or "0")
    except ValueError:
        return 0.0


def gsc_sigma_h() -> float:
    try:
        return float(os.environ.get("GEOPACT_GSC_SIGMA_H", "8.0") or "8.0")
    except ValueError:
        return 8.0


def gsc_sigma_w() -> float:
    try:
        return float(os.environ.get("GEOPACT_GSC_SIGMA_W", "8.0") or "8.0")
    except ValueError:
        return 8.0


def gsc_query_adapt_enabled() -> bool:
    return _env_flag("GEOPACT_GSC_QUERY_ADAPT", "0")


def gsc_schedule_enabled() -> bool:
    return _env_flag("GEOPACT_GSC_SCHEDULE", "0") or (gsc_tau() > 0.0)


def layer_active(layer_idx: int) -> bool:
    raw = (os.environ.get("GEOPACT_GSC_LAYERS") or "all").strip().lower()
    if raw in ("all", "*", ""):
        return True
    try:
        allowed = {int(x.strip()) for x in raw.split(",") if x.strip()}
        return int(layer_idx) in allowed
    except ValueError:
        return True


def layer_eta(layer_idx: int, num_layers: int = 36) -> float:
    eta0 = gsc_eta0()
    if not gsc_schedule_enabled():
        return eta0
    tau = gsc_tau()
    if tau <= 0.0:
        tau = max(float(num_layers) / 3.0, 1.0)
    return float(eta0 * math.exp(-float(layer_idx) / tau))


@dataclass
class GSCCtx:
    seq_len: int
    is_vision: torch.Tensor  # [N] bool
    is_text: torch.Tensor  # [N] bool
    log_Z: torch.Tensor  # [N] ln Z_{i,M}
    Z: torch.Tensor  # [N] flux mass (exp log_Z)
    phi_center: torch.Tensor  # [N] psi_center on vision keys
    phi_surv: torch.Tensor  # [N] survival potential on vision keys
    phi: torch.Tensor  # [N] combined SPF key prior on vision
    n_frame: int
    n_pruned: int
    Z_text_scalar: float
    audit: dict


@dataclass
class GSCLiftFeatures:
    """SPF rank channels (VFS uses log_Z on query aug + sink k_aug ch0)."""

    u: torch.Tensor  # [N, r_spf]
    v: torch.Tensor  # [N, r_spf]
    rank_spf: int


def _psi_center(hv: float, wv: float, h_min: float, h_max: float, w_min: float, w_max: float) -> float:
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    nh = (hv - h_min) / h_span
    nw = (wv - w_min) / w_span
    return -float((nh - 0.5) ** 2 + (nw - 0.5) ** 2) ** 0.5


def _flat_to_mrope(
    flat_idx: int,
    grid_h: int,
    grid_w: int,
    h_min: float,
    h_max: float,
    w_min: float,
    w_max: float,
) -> Tuple[float, float]:
    h_flat = flat_idx // max(grid_w, 1)
    w_flat = flat_idx % max(grid_w, 1)
    denom_h = max(grid_h - 1, 1)
    denom_w = max(grid_w - 1, 1)
    hv = h_min + (float(h_flat) / denom_h) * (h_max - h_min)
    wv = w_min + (float(w_flat) / denom_w) * (w_max - w_min)
    return hv, wv


def _compute_log_Z(
    *,
    seq_len: int,
    is_text: torch.Tensor,
    is_vision: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    n_frame: int,
    keep_flat: Sequence[int],
    grid_h: int,
    grid_w: int,
    eta: float,
    sigma_h: float,
    sigma_w: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_Z = torch.zeros(seq_len, dtype=torch.float32, device=device)
    if n_frame <= 0:
        return log_Z, torch.exp(log_Z)

    keep_set = {int(x) for x in keep_flat}
    pruned = [m for m in range(n_frame) if m not in keep_set]
    if not pruned:
        return log_Z, torch.exp(log_Z)

    vis_mask = is_vision
    if vis_mask.any():
        h_min = float(h[vis_mask].min().item())
        h_max = float(h[vis_mask].max().item())
        w_min = float(w[vis_mask].min().item())
        w_max = float(w[vis_mask].max().item())
    else:
        h_min = h_max = w_min = w_max = 0.0

    pruned_h: List[float] = []
    pruned_w: List[float] = []
    pruned_psi: List[float] = []
    for m in pruned:
        hm, wm = _flat_to_mrope(m, grid_h, grid_w, h_min, h_max, w_min, w_max)
        pruned_h.append(hm)
        pruned_w.append(wm)
        pruned_psi.append(_psi_center(hm, wm, h_min, h_max, w_min, w_max))

    pruned_h_t = torch.tensor(pruned_h, dtype=torch.float32, device=device)
    pruned_w_t = torch.tensor(pruned_w, dtype=torch.float32, device=device)
    pruned_psi_t = torch.tensor(pruned_psi, dtype=torch.float32, device=device)

    for i in range(seq_len):
        if bool(is_text[i].item()):
            z = torch.exp(eta * pruned_psi_t).sum()
        elif bool(is_vision[i].item()):
            dh = (h[i] - pruned_h_t).abs() / max(sigma_h, 1e-6)
            dw = (w[i] - pruned_w_t).abs() / max(sigma_w, 1e-6)
            z = torch.exp(-dh - dw).sum()
        else:
            z = torch.tensor(0.0, device=device)
        log_Z[i] = torch.log(z.clamp(min=1e-12))

    return log_Z, torch.exp(log_Z)


def _build_phi_survival(
    seq_len: int,
    is_vision: torch.Tensor,
    visual_keep_flat: Optional[torch.Tensor],
    cdct_density_flat: Optional[torch.Tensor],
    flat_to_compact: Dict[int, int],
    frame_start: int,
    device: torch.device,
) -> torch.Tensor:
    phi_surv = torch.zeros(seq_len, dtype=torch.float32, device=device)
    if cdct_density_flat is None or visual_keep_flat is None:
        return phi_surv
    dens = cdct_density_flat.detach().float().cpu()
    if dens.numel() == 0:
        return phi_surv
    for ord_i, flat_i in enumerate(visual_keep_flat.tolist()):
        local_i = int(flat_i) - int(frame_start)
        ci = flat_to_compact.get(int(flat_i))
        if ci is None:
            ci = flat_to_compact.get(local_i)
        if ci is None or not bool(is_vision[int(ci)].item()):
            continue
        fi = local_i
        if not (0 <= fi < int(dens.numel())):
            fi = int(flat_i)
        if 0 <= fi < int(dens.numel()):
            p = float(dens[fi].item())
            p = max(p, 1e-8)
            phi_surv[int(ci)] = -math.log(p)
    return phi_surv


def build_gsc_spf_bias(
    ctx: GSCCtx,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_idx: int = 0,
    num_layers: int = 36,
    query_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Additive SPF bias [1,1,N,N]: text query x vision key."""
    n = int(ctx.seq_len)
    bias = torch.zeros(1, 1, n, n, device=device, dtype=torch.float32)
    if n <= 0 or gsc_spf_mode() == "off":
        return bias.to(dtype=dtype)

    is_v = ctx.is_vision.to(device)
    text_idx = (~is_v).nonzero(as_tuple=False).flatten()
    vis_idx = is_v.nonzero(as_tuple=False).flatten()
    if text_idx.numel() == 0 or vis_idx.numel() == 0:
        return bias.to(dtype=dtype)

    eta_l = layer_eta(layer_idx, num_layers)
    phi = ctx.phi.to(device=device, dtype=torch.float32)
    tv = phi.index_select(0, vis_idx)
    if query_gate is not None:
        g = query_gate.to(device=device, dtype=torch.float32)
        if g.shape[0] == n:
            g_row = g.index_select(0, text_idx).unsqueeze(1)
            tv = tv.unsqueeze(0) * g_row
        else:
            tv = tv.unsqueeze(0) * float(g.item())
    bias[0, 0, text_idx.unsqueeze(1), vis_idx.unsqueeze(0)] = (eta_l * tv).unsqueeze(0)
    return bias.to(dtype=dtype)


def build_gsc_lift_features(
    ctx: GSCCtx,
    *,
    device: torch.device,
    layer_idx: int = 0,
    num_layers: int = 36,
    query_gate: Optional[torch.Tensor] = None,
) -> GSCLiftFeatures:
    """Rank-r SPF lift: u on text queries, v on vision keys."""
    n = int(ctx.seq_len)
    mode = gsc_spf_mode()
    r = 0
    if mode in ("center", "full"):
        r += 1
    if mode == "full":
        r += 1
    u = torch.zeros(n, max(r, 1), device=device, dtype=torch.float32)
    v = torch.zeros(n, max(r, 1), device=device, dtype=torch.float32)
    if r <= 0:
        return GSCLiftFeatures(u=u[:, :0], v=v[:, :0], rank_spf=0)

    eta_l = layer_eta(layer_idx, num_layers)
    text = ctx.is_text.to(device)
    vis = ctx.is_vision.to(device)
    gate = torch.ones(n, device=device, dtype=torch.float32)
    if query_gate is not None:
        gate = query_gate.to(device=device, dtype=torch.float32)
        if gate.numel() == 1:
            gate = gate.expand(n)

    ch = 0
    if mode in ("center", "full"):
        u[text, ch] = gate[text]
        v[vis, ch] = eta_l * ctx.phi_center.to(device)[vis]
        ch += 1
    if mode == "full":
        u[text, ch] = gate[text]
        v[vis, ch] = eta_l * ctx.phi_surv.to(device)[vis]

    return GSCLiftFeatures(u=u[:, :r], v=v[:, :r], rank_spf=r)


def build_b_vision_icc(
    ctx: GSCCtx,
    *,
    device: torch.device,
    dtype: torch.dtype,
    layer_idx: int = 0,
    num_layers: int = 36,
    scaling: float,
    query_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ICC key bias on vision rows: eta_l * phi / scaling so (q·k)*scaling adds eta_l*phi on T×S."""
    n = int(ctx.seq_len)
    b = torch.zeros(n, device=device, dtype=dtype)
    if gsc_spf_mode() == "off":
        return b
    eta_l = layer_eta(layer_idx, num_layers)
    vis = ctx.is_vision.to(device)
    phi = ctx.phi.to(device=device, dtype=torch.float32)
    b_vis = eta_l * phi[vis]
    if query_gate is not None and gsc_query_adapt_enabled():
        g = float(query_gate.mean().item()) if query_gate.numel() > 1 else float(query_gate.item())
        b_vis = b_vis * g
    b[vis] = (b_vis / max(float(scaling), 1e-8)).to(dtype)
    return b


def attach_gsc_ctx(
    model,
    *,
    compact_ids: torch.Tensor,
    compact_pos: Optional[torch.Tensor],
    image_tok_id: int,
    prune_ctx: Optional[dict],
    visual_keep_flat: Optional[torch.Tensor],
    image_grid_thw=None,
) -> Optional[GSCCtx]:
    if model is None or compact_pos is None:
        return None
    seq_len = int(compact_ids.shape[1])
    dev = compact_ids.device
    row = compact_ids[0]
    is_vision = row == int(image_tok_id)
    is_text = ~is_vision
    h, w = extract_hw_from_compact_pos(compact_pos)

    n_frame = 0
    keep_list: List[int] = []
    flat_to_compact: Dict[int, int] = {}
    cdct_density_flat = None
    frame_start = 0
    if prune_ctx:
        n_frame = int(prune_ctx.get("n_frame") or 0)
        frame_start = int(prune_ctx.get("frame_start") or 0)
        cdct_density_flat = prune_ctx.get("cdct_density_flat")
        flat_to_compact = dict(prune_ctx.get("flat_to_compact") or {})

    if visual_keep_flat is not None:
        keep_list = []
        for x in visual_keep_flat.tolist():
            g = int(x)
            local = g - frame_start
            if 0 <= local < max(n_frame, 0):
                keep_list.append(local)
            elif 0 <= g < max(n_frame, 0):
                keep_list.append(g)

    grid_h, grid_w = 1, 1
    if image_grid_thw is not None and n_frame > 0:
        grid_h, grid_w, _ = _get_grid(image_grid_thw, n_frame)

    eta = gsc_eta0()
    log_Z, Z = _compute_log_Z(
        seq_len=seq_len,
        is_text=is_text,
        is_vision=is_vision,
        h=h,
        w=w,
        n_frame=n_frame,
        keep_flat=keep_list,
        grid_h=grid_h,
        grid_w=grid_w,
        eta=eta,
        sigma_h=gsc_sigma_h(),
        sigma_w=gsc_sigma_w(),
        device=dev,
    )

    vis_mask = is_vision
    if vis_mask.any():
        h_min = float(h[vis_mask].min().item())
        h_max = float(h[vis_mask].max().item())
        w_min = float(w[vis_mask].min().item())
        w_max = float(w[vis_mask].max().item())
    else:
        h_min = h_max = w_min = w_max = 0.0

    phi_center = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    for j in range(seq_len):
        if bool(is_vision[j].item()):
            phi_center[j] = _psi_center(
                float(h[j].item()),
                float(w[j].item()),
                h_min,
                h_max,
                w_min,
                w_max,
            )

    phi_surv = _build_phi_survival(
        seq_len,
        is_vision,
        visual_keep_flat,
        cdct_density_flat,
        flat_to_compact,
        frame_start,
        dev,
    )

    phi = torch.zeros(seq_len, dtype=torch.float32, device=dev)
    mode = gsc_spf_mode()
    if mode in ("center", "full"):
        phi = phi + phi_center
    if mode == "full":
        phi = phi + phi_surv
    phi = phi * is_vision.float()

    n_pruned = max(0, n_frame - len(keep_list)) if n_frame > 0 else 0
    z_text_scalar = 0.0
    if is_text.any() and Z.numel() > 0:
        text_z = Z[is_text]
        if text_z.numel() > 0:
            z_text_scalar = float(text_z[0].item())

    ctx = GSCCtx(
        seq_len=seq_len,
        is_vision=is_vision,
        is_text=is_text,
        log_Z=log_Z,
        Z=Z,
        phi_center=phi_center,
        phi_surv=phi_surv,
        phi=phi,
        n_frame=n_frame,
        n_pruned=n_pruned,
        Z_text_scalar=z_text_scalar,
        audit={
            "gsc": True,
            "gsc_mode": gsc_mode(),
            "gsc_vfs": gsc_vfs_enabled(),
            "gsc_spf": gsc_spf_mode(),
            "gsc_eta0": eta,
            "gsc_tau": gsc_tau(),
            "gsc_schedule": gsc_schedule_enabled(),
            "gsc_query_adapt": gsc_query_adapt_enabled(),
            "seq_len": seq_len,
            "n_visual": int(is_vision.sum().item()),
            "n_frame": n_frame,
            "n_pruned": n_pruned,
            "Z_text_scalar": z_text_scalar,
            "log_Z_max": float(log_Z.max().item()) if log_Z.numel() else 0.0,
        },
    )

    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_gsc_ctx = ctx
        lang._geopact_gsc_enabled = True
        layers = getattr(lang, "layers", None)
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn._geopact_language_model_ref = lang
    model._geopact_gsc_ctx = ctx
    return ctx


def clear_gsc_ctx(model) -> None:
    if model is None:
        return
    lang = getattr(model, "language_model", None)
    if lang is not None:
        lang._geopact_gsc_ctx = None
        lang._geopact_gsc_enabled = False
    model._geopact_gsc_ctx = None
