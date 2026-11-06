"""VSAS core: Nadaraya-Watson measure reconstruction on spatial anchors."""
from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F

AnchorMode = Literal["four_corners", "nine_grid"]


def build_anchor_coords(mode: AnchorMode, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return normalized anchor coordinates (K, 2) in [0, 1]^2."""
    if mode == "four_corners":
        coords = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    elif mode == "nine_grid":
        coords = [[i, j] for i in (0.1, 0.5, 0.9) for j in (0.1, 0.5, 0.9)]
    else:
        raise ValueError(f"Unknown anchor mode: {mode}")
    return torch.tensor(coords, device=device, dtype=dtype)


def normalize_visual_hw(
    h: torch.Tensor,
    w: torch.Tensor,
    is_vision: torch.Tensor,
) -> torch.Tensor:
    """Map MRoPE (h,w) for vision tokens to normalized (x,y) in [0,1]^2. Shape (N, 2)."""
    n = int(h.shape[0])
    out = torch.zeros(n, 2, device=h.device, dtype=torch.float32)
    vis = is_vision.to(device=h.device)
    if not bool(vis.any().item()):
        return out
    hv = h[vis].to(torch.float32)
    wv = w[vis].to(torch.float32)
    h_min = float(hv.min().item())
    h_max = float(hv.max().item())
    w_min = float(wv.min().item())
    w_max = float(wv.max().item())
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    out[vis, 0] = (h[vis].to(torch.float32) - h_min) / h_span
    out[vis, 1] = (w[vis].to(torch.float32) - w_min) / w_span
    return out


def compute_omega(
    anchor_coords: torch.Tensor,
    pos_surv: torch.Tensor,
    *,
    sigma: float,
    vis_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Nadaraya-Watson weights Omega (K, N) or (B, K, N).

    pos_surv: (N, 2) or (B, N, 2) normalized coordinates.
    vis_mask: if set, non-vision positions get -inf before softmax.
    """
    sigma = max(float(sigma), 1e-6)
    if pos_surv.dim() == 2:
        # (K, 2) vs (N, 2) -> (K, N)
        dist_sq = torch.cdist(anchor_coords.unsqueeze(0), pos_surv.unsqueeze(0)).pow(2).squeeze(0)
        logits = -dist_sq / (2.0 * sigma * sigma)
        if vis_mask is not None:
            logits = logits.masked_fill(~vis_mask.view(1, -1).to(logits.device), float("-inf"))
        return F.softmax(logits, dim=-1)
    # batched (B, N, 2)
    dist_sq = torch.cdist(anchor_coords.unsqueeze(0), pos_surv).pow(2)
    logits = -dist_sq / (2.0 * sigma * sigma)
    if vis_mask is not None:
        logits = logits.masked_fill(~vis_mask.view(1, 1, -1).to(logits.device), float("-inf"))
    return F.softmax(logits, dim=-1)


def build_anchor_hidden(
    x_surv: torch.Tensor,
    omega: torch.Tensor,
    *,
    vis_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    X_anchor = Omega @ X_surv.

    x_surv: (B, N, D) or (N, D)
    omega: (K, N_vis) or (B, K, N_vis) — if vis_indices given, gather columns first.
    """
    if x_surv.dim() == 2:
        x_surv = x_surv.unsqueeze(0)
        batched = False
    else:
        batched = True
    if vis_indices is not None:
        x_use = x_surv.index_select(1, vis_indices.to(x_surv.device))
        if omega.dim() == 2:
            omega = omega.unsqueeze(0).expand(x_surv.shape[0], -1, -1)
    else:
        x_use = x_surv
    if omega.dim() == 2:
        omega = omega.unsqueeze(0).expand(x_use.shape[0], -1, -1)
    x_anchor = torch.bmm(omega, x_use)
    if not batched:
        x_anchor = x_anchor.squeeze(0)
    return x_anchor


def anchor_mrope_hw(
    anchor_coords: torch.Tensor,
    h_min: float,
    h_max: float,
    w_min: float,
    w_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map normalized anchor coords to MRoPE h,w tensors (K,)."""
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)
    ac = anchor_coords.to(device=device, dtype=torch.float32)
    h = h_min + ac[:, 0] * h_span
    w = w_min + ac[:, 1] * w_span
    return h.to(dtype=dtype), w.to(dtype=dtype)
