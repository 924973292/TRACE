"""DMTC FlashAttention-native correction path."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from lmms_eval.models.geopact_correction import (
    CorrectionCtx,
    correction_layer_eta,
    correction_lpr_enabled,
    correction_region_enabled,
    correction_tv_enabled,
    correction_psi_enabled,
    layer_active,
)


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_rep <= 1:
        return key, value
    b, n_kv, s, d = key.shape
    key = key[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    value = value[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    return key, value


def _pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    pad = int(target_dim) - int(x.shape[-1])
    if pad <= 0:
        return x
    return torch.nn.functional.pad(x, (0, pad))


def _align_dim(dim: int, multiple: int = 8) -> int:
    return ((int(dim) + multiple - 1) // multiple) * multiple


def _build_augmented_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ctx: CorrectionCtx,
    *,
    scaling: float,
    layer_idx: int,
    num_layers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return padded rank-2 augmented q/k/v [B,H,N,Dpad]."""
    b, h, n, d = query.shape
    dtype = query.dtype
    device = query.device
    text = ctx.is_text.to(device=device).view(1, 1, n, 1)
    vis = ctx.is_vision.to(device=device).view(1, 1, n, 1)

    u = torch.zeros(1, 1, n, 2, device=device, dtype=dtype)
    vfeat = torch.zeros(1, 1, n, 2, device=device, dtype=dtype)
    if correction_tv_enabled():
        if correction_region_enabled():
            u[..., 0:1] = text.to(dtype=dtype)
            vfeat[..., 0] = ctx.region_prior.to(device=device, dtype=dtype).view(1, 1, n) * vis.squeeze(-1).to(dtype=dtype)
        if correction_psi_enabled():
            eta_l = correction_layer_eta(layer_idx, num_layers)
            u[..., 1:2] = text.to(dtype=dtype)
            vfeat[..., 1] = (
                eta_l
                * ctx.psi.to(device=device, dtype=dtype).view(1, 1, n)
                * vis.squeeze(-1).to(dtype=dtype)
            )

    zeros = torch.zeros(b, h, n, 2, device=device, dtype=dtype)
    q_aug = torch.cat([query * float(scaling), u.expand(b, h, n, 2)], dim=-1)
    k_aug = torch.cat([key, vfeat.expand(b, h, n, 2)], dim=-1)
    v_aug = torch.cat([value, zeros], dim=-1)
    d_out = int(query.shape[-1])
    d_pad = _align_dim(int(q_aug.shape[-1]), 8)
    q_aug = _pad_last_dim(q_aug, d_pad)
    k_aug = _pad_last_dim(k_aug, d_pad)
    v_aug = _pad_last_dim(v_aug, d_pad)
    return q_aug, k_aug, v_aug, d_out


def _bhnd_to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bshd_to_bhnd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def eager_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, h, n, _ = query.shape
    scores = torch.matmul(query, key.transpose(2, 3))
    if causal:
        mask = torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, n, n), float("-inf"))
    scores_f = scores.float()
    lse = torch.logsumexp(scores_f, dim=-1)
    probs = torch.softmax(scores_f, dim=-1).to(query.dtype)
    out = torch.matmul(probs, value)
    return out, lse


def flash_attention_with_lse_strict(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FA call that returns output and softmax_lse from the same kernel."""
    if query.device.type != "cuda":
        return eager_attention_with_lse(query, key, value, causal=causal)
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError("DMTC requires flash_attn on CUDA; no eager fallback is allowed") from exc

    q = _bhnd_to_bshd(query)
    k = _bhnd_to_bshd(key)
    v = _bhnd_to_bshd(value)
    result = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=1.0,
        causal=causal,
        return_attn_probs=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("flash_attn_func did not return softmax_lse for DMTC")
    out = result[0]
    lse = result[1]
    # flash_attn returns lse as [B,H,S] for recent versions; tolerate [B,S,H].
    if lse.dim() == 3 and lse.shape[1] == q.shape[1] and lse.shape[2] == q.shape[2]:
        lse = lse.transpose(1, 2).contiguous()
    return _bshd_to_bhnd(out), lse.float()


def apply_deleted_mass_rescale(out: torch.Tensor, lse: torch.Tensor, Z_deleted: torch.Tensor) -> torch.Tensor:
    if not correction_lpr_enabled():
        return out
    z = Z_deleted.to(device=out.device, dtype=torch.float32).view(1, 1, -1)
    gamma = 1.0 / (1.0 + z * torch.exp(-lse.float()))
    return out * gamma.unsqueeze(-1).to(dtype=out.dtype)


def try_correction_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    correction_ctx: CorrectionCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
    layer_idx: int = 0,
    num_layers: int = 36,
    **kwargs: Any,
) -> torch.Tensor:
    del kwargs
    key_states, value_states = _repeat_kv(key, value, module.num_key_value_groups)
    q_aug, k_aug, v_aug, head_dim = _build_augmented_qkv(
        query,
        key_states,
        value_states,
        correction_ctx,
        scaling=scaling,
        layer_idx=layer_idx,
        num_layers=num_layers,
    )
    out, lse = flash_attention_with_lse_strict(
        q_aug,
        k_aug,
        v_aug,
        causal=True,
        dropout_p=dropout if module.training else 0.0,
    )
    out = apply_deleted_mass_rescale(out, lse, correction_ctx.Z_deleted)
    return out[..., :head_dim].transpose(1, 2).contiguous()


def should_apply_correction(
    correction_ctx: Optional[CorrectionCtx],
    *,
    past_key_values,
    query_len: int,
    layer_idx: int,
) -> bool:
    if correction_ctx is None:
        return False
    if not layer_active(layer_idx):
        return False
    if query_len <= 1 and past_key_values is not None:
        return False
    if int(correction_ctx.seq_len) != int(query_len):
        return False
    return True
