"""SSPC Module C: zero-token V-path ZAFD on pruned vision embeddings (A+B joint)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, Set

OmegaNorm = Literal["donor", "survivor"]

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_vl_hook import _get_grid as _hook_get_grid


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def sspc_enabled() -> bool:
    return _env_flag("GEOPACT_SSPC")


def sspc_zafd_enabled() -> bool:
    return _env_flag("GEOPACT_SSPC_ZAFD", "1")


def sspc_strict_enabled() -> bool:
    return _env_flag("GEOPACT_SSPC_STRICT", "1")


def sspc_sigma() -> float:
    try:
        return float(os.environ.get("GEOPACT_SSPC_SIGMA", "0.4") or "0.4")
    except ValueError:
        return 0.4


def sspc_zafd_omega_norm() -> OmegaNorm:
    raw = os.environ.get("GEOPACT_ZAFD_OMEGA_NORM", "donor").strip().lower()
    return "survivor" if raw == "survivor" else "donor"


def sspc_zafd_gamma() -> float:
    try:
        return max(0.0, float(os.environ.get("GEOPACT_ZAFD_GAMMA", "0") or "0"))
    except ValueError:
        return 0.0


def sspc_zafd_adaptive_sigma_enabled() -> bool:
    return _env_flag("GEOPACT_ZAFD_ADAPTIVE_SIGMA", "0")


def sspc_zafd_sigma_beta() -> float:
    try:
        return float(os.environ.get("GEOPACT_ZAFD_SIGMA_BETA", "1.0") or "1.0")
    except ValueError:
        return 1.0


def sspc_zafd_delta_clamp() -> float:
    """Receiver-side relative-norm clamp c (GEOPACT_ZAFD_DELTA_CLAMP, default 0=off).

    Delta_i <- Delta_i * min(1, c*||X_i|| / ||Delta_i||): the ZAFD update can
    never move a survivor by more than c times its own norm. Bounds the
    Lipschitz-propagated error of the conservative (donor-norm) transport
    at the cost of breaking exact donor mass
    conservation on clamped rows (clamp rate is audited).
    """
    try:
        return max(0.0, float(os.environ.get("GEOPACT_ZAFD_DELTA_CLAMP", "0") or "0"))
    except ValueError:
        return 0.0


def sspc_zafd_shuffle_mode() -> str:
    """Negative-control shuffles (GEOPACT_ZAFD_SHUFFLE: '', 'omega', 'donor').

    'omega': permute Omega rows (survivor axis) — destroys spatial
             correspondence, preserves column-stochastic mass.
    'donor': permute donor feature rows — features decoupled from positions.
    Deterministic seed derived from (n_surv, n_donor).
    """
    raw = os.environ.get("GEOPACT_ZAFD_SHUFFLE", "").strip().lower()
    return raw if raw in {"omega", "donor"} else ""


def sspc_k_grid() -> int:
    try:
        return max(1, int(os.environ.get("GEOPACT_SSPC_K_GRID", "3") or "3"))
    except ValueError:
        return 3


def sspc_axis_mode() -> str:
    return os.environ.get("GEOPACT_SSPC_AXIS", "decile").strip().lower()


def resolve_frame_grid(grid_h: int, grid_w: int, n_frame: int) -> tuple[int, int, bool]:
    """Return (gh, gw, matched) for skeleton/ZAFD coordinates over n_frame tokens."""
    gh = max(int(grid_h), 1)
    gw = max(int(grid_w), 1)
    n_frame = int(n_frame)
    if n_frame > 0 and gh * gw == n_frame:
        return gh, gw, True
    if n_frame <= 0:
        return 1, 1, False
    gw = max(int(round(n_frame**0.5)), 1)
    gh = max((n_frame + gw - 1) // gw, 1)
    while gh * gw < n_frame:
        gw += 1
        gh = max((n_frame + gw - 1) // gw, 1)
    return gh, gw, False


@dataclass
class SSPCCtx:
    keep_b: Set[int]
    frame_start: int
    n_frame: int
    grid_h: int
    grid_w: int
    sigma: float
    audit: dict


def clear_sspc_ctx(model) -> None:
    """Full reset (tests / fleet teardown)."""
    if model is None:
        return
    model._geopact_sspc_keep_b = set()
    model._geopact_sspc_keep_final = set()
    model._geopact_sspc_ctx = None
    model._geopact_sspc_audit = {}
    model._geopact_sspc_enabled = False


def clear_sspc_attach_ctx(model) -> None:
    """Clear attach-only state; preserve keep_b until ZAFD runs."""
    if model is None:
        return
    model._geopact_sspc_ctx = None
    model._geopact_sspc_enabled = False


def reset_sspc_step_state(model) -> None:
    """Per forward-step reset before prune records new keep sets."""
    if model is None:
        return
    model._geopact_sspc_keep_b = set()
    model._geopact_sspc_keep_final = set()
    model._geopact_sspc_audit = {}
    model._geopact_compact_prune_ctx = None


def record_keep_b(model, keep_b_global: Set[int]) -> None:
    if model is None:
        return
    acc = getattr(model, "_geopact_sspc_keep_b", None) or set()
    model._geopact_sspc_keep_b = set(acc) | {int(x) for x in keep_b_global}


def record_keep_final(model, keep_final_global: Set[int]) -> None:
    if model is None:
        return
    acc = getattr(model, "_geopact_sspc_keep_final", None) or set()
    model._geopact_sspc_keep_final = set(acc) | {int(x) for x in keep_final_global}


_FRAME_POS_CACHE: dict = {}


def _frame_positions_norm(
    n_frame: int,
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Normalized (y, x) in [0,1]^2 for frame-local flat indices.

    Vectorized in fp64 then cast to fp32 so each value matches the original
    per-element python-float computation bit-for-bit; cached per shape (the
    scalar-assignment loop used to cost 2 kernel launches × n_frame per call).
    """
    gh, gw, _ = resolve_frame_grid(grid_h, grid_w, n_frame)
    key = (int(n_frame), int(gh), int(gw), str(device))
    cached = _FRAME_POS_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(int(n_frame), device=device, dtype=torch.long)
    r = torch.div(idx, int(gw), rounding_mode="floor").to(torch.float64)
    c = torch.remainder(idx, int(gw)).to(torch.float64)
    y = (r + 0.5) / float(gh)
    x = (c + 0.5) / float(gw)
    coords = torch.stack([y, x], dim=1).to(torch.float32).clamp(0.0, 1.0)
    while len(_FRAME_POS_CACHE) >= 64:
        _FRAME_POS_CACHE.pop(next(iter(_FRAME_POS_CACHE)))
    _FRAME_POS_CACHE[key] = coords
    return coords


def compute_omega_surv_pruned(
    pos_surv: torch.Tensor,
    pos_pruned: torch.Tensor,
    *,
    sigma: float,
    norm: Optional[OmegaNorm] = None,
    x_surv: Optional[torch.Tensor] = None,
    x_donor: Optional[torch.Tensor] = None,
    gamma: Optional[float] = None,
    adaptive_sigma: Optional[bool] = None,
    sigma_beta: Optional[float] = None,
) -> torch.Tensor:
    """
    Omega (N_surv, N_pruned).

    donor norm (v2): column-stochastic — each pruned donor partitions mass over survivors.
    survivor norm (v0): row-stochastic — each survivor convex-combines donors.
    """
    sigma_min = max(float(sigma), 1e-6)
    omega_norm: OmegaNorm = norm if norm is not None else sspc_zafd_omega_norm()
    use_gamma = sspc_zafd_gamma() if gamma is None else max(0.0, float(gamma))
    use_adaptive = (
        sspc_zafd_adaptive_sigma_enabled() if adaptive_sigma is None else bool(adaptive_sigma)
    )
    use_beta = sspc_zafd_sigma_beta() if sigma_beta is None else float(sigma_beta)

    if pos_surv.numel() == 0 or pos_pruned.numel() == 0:
        return torch.zeros(pos_surv.shape[0], pos_pruned.shape[0], device=pos_surv.device)

    dist = torch.cdist(pos_surv.unsqueeze(0), pos_pruned.unsqueeze(0)).squeeze(0)
    dist_sq = dist.pow(2)
    if use_adaptive:
        min_dist = dist.min(dim=0).values.clamp(min=1e-6)
        sigma_j = torch.maximum(
            torch.full_like(min_dist, sigma_min),
            use_beta * min_dist,
        )
        spatial_logits = -dist_sq / (2.0 * sigma_j.pow(2).unsqueeze(0))
    else:
        spatial_logits = -dist_sq / (2.0 * sigma_min * sigma_min)

    logits = spatial_logits
    if use_gamma > 0.0 and x_surv is not None and x_donor is not None:
        x_s = F.normalize(x_surv.to(torch.float32), dim=-1)
        x_d = F.normalize(x_donor.to(torch.float32), dim=-1)
        cos_sim = x_s @ x_d.T
        logits = spatial_logits - use_gamma * (1.0 - cos_sim)

    softmax_dim = 0 if omega_norm == "donor" else -1
    return F.softmax(logits, dim=softmax_dim)


def apply_embedding_zafd(
    image_embeds: torch.Tensor,
    *,
    keep_b: Set[int],
    frame_start: int,
    n_frame: int,
    grid_h: int,
    grid_w: int,
    sigma: Optional[float] = None,
    keep_final: Optional[Set[int]] = None,
) -> tuple[torch.Tensor, dict]:
    """
    X_surv += Omega @ X_donor on vision encoder embeddings (B receivers only).

    Donor set = frame tokens truly discarded (not in keep_final).
    Skeleton-only tokens are kept via gather but excluded from both donor and receiver.
    """
    sigma = sspc_sigma() if sigma is None else sigma
    out = image_embeds
    if not keep_b or n_frame <= 0:
        return out, {"sspc_zafd": False}

    frame_start = int(frame_start)
    n_frame = int(n_frame)
    frame_vis = set(range(frame_start, frame_start + n_frame))
    keep_b_frame = sorted(int(i) for i in keep_b if int(i) in frame_vis)
    keep_final_frame = (
        sorted(int(i) for i in keep_final if int(i) in frame_vis)
        if keep_final is not None
        else keep_b_frame
    )
    keep_final_set = set(keep_final_frame)
    donor = sorted(frame_vis - keep_final_set)
    if not keep_b_frame or not donor:
        return out, {
            "sspc_zafd": False,
            "sspc_zafd_reason": "empty_partition",
            "sspc_zafd_n_surv": len(keep_b_frame),
            "sspc_zafd_n_donor": len(donor),
        }

    n_rows = int(image_embeds.shape[0])
    used_idx = keep_b_frame + donor
    if any(int(i) < 0 or int(i) >= n_rows for i in used_idx):
        return out, {
            "sspc_zafd": False,
            "sspc_zafd_reason": "index_oob",
            "sspc_zafd_n_surv": len(keep_b_frame),
            "sspc_zafd_n_donor": len(donor),
            "sspc_zafd_n_rows": n_rows,
            "sspc_frame_start": frame_start,
        }

    gh, gw, grid_matched = resolve_frame_grid(grid_h, grid_w, n_frame)
    if not grid_matched:
        return out, {
            "sspc_zafd": False,
            "sspc_zafd_reason": "grid_mismatch",
            "sspc_grid_h": gh,
            "sspc_grid_w": gw,
            "sspc_grid_matched": False,
            "sspc_hook_grid_h": grid_h,
            "sspc_hook_grid_w": grid_w,
        }
    dev = image_embeds.device
    pos = _frame_positions_norm(n_frame, gh, gw, device=dev)
    loc_surv = [i - frame_start for i in keep_b_frame]
    loc_donor = [i - frame_start for i in donor]
    pos_surv = pos[loc_surv]
    pos_donor = pos[loc_donor]
    out = image_embeds.clone()
    x_surv = out[keep_b_frame]
    x_donor = out[donor]
    omega_norm = sspc_zafd_omega_norm()
    zafd_gamma = sspc_zafd_gamma()
    use_adaptive = sspc_zafd_adaptive_sigma_enabled()
    omega = compute_omega_surv_pruned(
        pos_surv,
        pos_donor,
        sigma=sigma,
        norm=omega_norm,
        x_surv=x_surv,
        x_donor=x_donor,
        gamma=zafd_gamma,
        adaptive_sigma=use_adaptive,
    )

    shuffle_mode = sspc_zafd_shuffle_mode()
    if shuffle_mode:
        g = torch.Generator(device="cpu")
        g.manual_seed(1009 * len(keep_b_frame) + len(donor))
        if shuffle_mode == "omega":
            perm = torch.randperm(omega.shape[0], generator=g).to(omega.device)
            omega = omega.index_select(0, perm)
        elif shuffle_mode == "donor":
            perm = torch.randperm(x_donor.shape[0], generator=g).to(x_donor.device)
            x_donor = x_donor.index_select(0, perm)

    x_donor_f = x_donor.to(torch.float32)
    delta = omega.to(torch.float32) @ x_donor_f

    clamp_c = sspc_zafd_delta_clamp()
    clamp_rate = 0.0
    if clamp_c > 0.0:
        x_surv_f = out[keep_b_frame].to(torch.float32)
        d_norm = delta.norm(dim=-1, keepdim=True)
        x_norm = x_surv_f.norm(dim=-1, keepdim=True)
        scale = torch.clamp(clamp_c * x_norm / d_norm.clamp(min=1e-12), max=1.0)
        delta = delta * scale
        if os.environ.get("GEOPACT_ZAFD_AUDIT", "0").strip().lower() in {"1", "true", "yes"}:
            clamp_rate = float((scale < 1.0).float().mean().item())

    out[keep_b_frame] = (out[keep_b_frame].to(torch.float32) + delta).to(image_embeds.dtype)

    skeleton_only = len(keep_final_set - set(keep_b_frame))
    # P14: mass-conservation / delta-magnitude audits force GPU->CPU syncs
    # (allclose + .item()) on every rectify; gate them behind an env flag.
    audit_on = os.environ.get("GEOPACT_ZAFD_AUDIT", "0").strip().lower() in {"1", "true", "yes"}
    donor_mass_ok = True
    delta_abs_max = 0.0
    delta_rel_mean = 0.0
    if audit_on:
        if omega.numel() > 0 and omega_norm == "donor":
            donor_mass_ok = bool(
                torch.allclose(
                    omega.sum(0),
                    torch.ones(omega.shape[1], device=omega.device),
                    atol=1e-4,
                )
            )
        elif omega.numel() > 0 and omega_norm == "survivor":
            donor_mass_ok = bool(
                torch.allclose(
                    omega.sum(-1),
                    torch.ones(omega.shape[0], device=omega.device),
                    atol=1e-4,
                )
            )
        delta_abs_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        if delta.numel():
            x_ref = out[keep_b_frame].to(torch.float32)
            rel = delta.norm(dim=-1) / x_ref.norm(dim=-1).clamp(min=1e-12)
            delta_rel_mean = float(rel.mean().item())
    return out, {
        "sspc_zafd": True,
        "sspc_zafd_n_surv": len(keep_b_frame),
        "sspc_zafd_n_donor": len(donor),
        "sspc_zafd_n_pruned": len(donor),
        "sspc_zafd_n_skeleton_only": skeleton_only,
        "sspc_zafd_omega_norm": omega_norm,
        "sspc_zafd_gamma": zafd_gamma,
        "sspc_zafd_adaptive_sigma": use_adaptive,
        "sspc_zafd_donor_mass_conserved": donor_mass_ok,
        "sspc_zafd_delta_abs_max": delta_abs_max,
        "sspc_zafd_delta_rel_mean": delta_rel_mean,
        "sspc_zafd_delta_clamp": clamp_c,
        "sspc_zafd_clamp_rate": clamp_rate,
        "sspc_zafd_shuffle": shuffle_mode or None,
        "sspc_sigma": sigma,
        "sspc_k_grid": sspc_k_grid(),
        "sspc_grid_h": gh,
        "sspc_grid_w": gw,
        "sspc_grid_matched": grid_matched,
    }


def apply_sspc_embedding_zafd_for_model(
    image_embeds: torch.Tensor,
    model,
    image_grid_thw,
    prune_ctx: Optional[dict],
    *,
    reject_grid_mismatch: bool = False,
) -> torch.Tensor:
    if model is None or not sspc_enabled() or not sspc_zafd_enabled():
        return image_embeds
    keep_b = getattr(model, "_geopact_sspc_keep_b", None) or set()
    if not keep_b:
        return image_embeds
    keep_final = getattr(model, "_geopact_sspc_keep_final", None) or keep_b
    ctx = prune_ctx or getattr(model, "_geopact_compact_prune_ctx", None) or {}
    frame_local = bool(ctx.get("frame_local"))
    frame_start = 0 if frame_local else int(ctx.get("frame_start") or 0)
    n_frame = int(ctx.get("n_frame") or image_embeds.shape[0])
    if frame_local:
        keep_b = {int(i) for i in keep_b if 0 <= int(i) < n_frame}
        keep_final = {int(i) for i in keep_final if 0 <= int(i) < n_frame}
        if not keep_b:
            return image_embeds
    grid_raw = _hook_get_grid(image_grid_thw, n_frame)
    if isinstance(grid_raw, tuple) and len(grid_raw) >= 3:
        grid_h, grid_w, hook_grid_matched = int(grid_raw[0]), int(grid_raw[1]), bool(grid_raw[2])
    elif isinstance(grid_raw, tuple) and len(grid_raw) >= 2:
        grid_h, grid_w, hook_grid_matched = int(grid_raw[0]), int(grid_raw[1]), True
    else:
        grid_h, grid_w, hook_grid_matched = 1, 1, False
    out, audit = apply_embedding_zafd(
        image_embeds,
        keep_b=keep_b,
        keep_final=keep_final,
        frame_start=frame_start,
        n_frame=n_frame,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    audit = {
        **audit,
        "sspc_hook_grid_h": grid_h,
        "sspc_hook_grid_w": grid_w,
        "sspc_hook_grid_matched": hook_grid_matched,
    }
    if reject_grid_mismatch and not bool(audit.get("sspc_grid_matched")):
        raise ValueError(
            "GeoPACT SSPC ZAFD grid mismatch: "
            f"hook_matched={hook_grid_matched} audit={audit}"
        )
    if model is not None:
        prev = getattr(model, "_geopact_sspc_audit", None) or {}
        model._geopact_sspc_audit = {**prev, **audit}
        model._geopact_sspc_ctx = SSPCCtx(
            keep_b=set(keep_b),
            frame_start=frame_start,
            n_frame=n_frame,
            grid_h=grid_h,
            grid_w=grid_w,
            sigma=sspc_sigma(),
            audit={**prev, **audit},
        )
    return out


def attach_sspc_ctx(model, *, audit: Optional[dict] = None) -> Optional[SSPCCtx]:
    if model is None or not sspc_enabled():
        return None
    ctx = getattr(model, "_geopact_sspc_ctx", None)
    if ctx is None:
        keep_b = getattr(model, "_geopact_sspc_keep_b", None) or set()
        prune_ctx = getattr(model, "_geopact_compact_prune_ctx", None) or {}
        ctx = SSPCCtx(
            keep_b=set(keep_b),
            frame_start=int(prune_ctx.get("frame_start") or 0),
            n_frame=int(prune_ctx.get("n_frame") or 0),
            grid_h=0,
            grid_w=0,
            sigma=sspc_sigma(),
            audit=dict(audit or getattr(model, "_geopact_sspc_audit", None) or {}),
        )
        model._geopact_sspc_ctx = ctx
    if audit:
        ctx.audit.update(audit)
    merged = dict(ctx.audit)
    if audit:
        merged.update(audit)
    model._geopact_spatial_audit = merged
    model._geopact_sspc_enabled = True
    return ctx


def sspc_stats_for_compact(model) -> dict:
    """Flatten SSPC audit for geopact_compact_stats logging."""
    audit = getattr(model, "_geopact_sspc_audit", None) or {}
    if not audit and model is not None:
        ctx = getattr(model, "_geopact_sspc_ctx", None)
        if ctx is not None:
            audit = dict(ctx.audit)
    return {
        "sspc_budget_mode": audit.get("sspc_budget_mode"),
        "sspc_keep_k_total": audit.get("sspc_keep_k_total"),
        "sspc_keep_k_b": audit.get("sspc_keep_k_b"),
        "sspc_reserved_n": audit.get("sspc_reserved_n"),
        "sspc_anchor_n": audit.get("sspc_anchor_n"),
        "budget_overage": audit.get("budget_overage"),
        "sspc_extra_mode": audit.get("sspc_extra_mode"),
        "sspc_extra_n": audit.get("sspc_extra_n"),
        "sspc_keep_b_n": audit.get("sspc_keep_b_n"),
        "sspc_keep_final_n": audit.get("sspc_keep_final_n"),
        "sspc_k_grid": audit.get("sspc_k_grid", sspc_k_grid()),
        "sspc_zafd": audit.get("sspc_zafd"),
        "sspc_zafd_n_surv": audit.get("sspc_zafd_n_surv"),
        "sspc_zafd_n_pruned": audit.get("sspc_zafd_n_pruned"),
        "sspc_zafd_n_donor": audit.get("sspc_zafd_n_donor"),
        "sspc_zafd_n_skeleton_only": audit.get("sspc_zafd_n_skeleton_only"),
        "sspc_zafd_omega_norm": audit.get("sspc_zafd_omega_norm"),
        "sspc_zafd_gamma": audit.get("sspc_zafd_gamma"),
        "sspc_zafd_adaptive_sigma": audit.get("sspc_zafd_adaptive_sigma"),
        "sspc_zafd_donor_mass_conserved": audit.get("sspc_zafd_donor_mass_conserved"),
        "sspc_zafd_delta_abs_max": audit.get("sspc_zafd_delta_abs_max"),
        "sspc_zafd_delta_rel_mean": audit.get("sspc_zafd_delta_rel_mean"),
        "sspc_zafd_delta_clamp": audit.get("sspc_zafd_delta_clamp"),
        "sspc_zafd_clamp_rate": audit.get("sspc_zafd_clamp_rate"),
        "sspc_zafd_shuffle": audit.get("sspc_zafd_shuffle"),
        "sspc_sigma": audit.get("sspc_sigma", sspc_sigma()),
        "sspc_grid_matched": audit.get("sspc_grid_matched"),
    }
