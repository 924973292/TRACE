"""PARB-Lift attention: concat Q/K/V aug dims and delegate to FlashAttention."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from lmms_eval.models.geopact_parb import PARBCtx
from lmms_eval.models.geopact_parb_lift import PARBLiftFeatures, build_parb_lift_features


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_rep <= 1:
        return key, value
    b, n_kv, s, d = key.shape
    key = key[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    value = value[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    return key, value


def parb_lift_concat_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    feat: PARBLiftFeatures,
    *,
    content_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Augment Q/K/V for lift. query/key/value [B,H,N,d]. Returns head_dim d."""
    head_dim = int(query.shape[-1])
    r = int(feat.rank)
    b, nh, n, _ = query.shape
    dev = query.device
    dtype = query.dtype

    u = feat.u.to(device=dev, dtype=torch.float32)
    v = feat.v.to(device=dev, dtype=torch.float32)
    if u.shape[0] != n or v.shape[0] != n:
        raise ValueError(f"lift feature seq mismatch: n={n} u={u.shape} v={v.shape}")

    u_b = u.view(1, 1, n, r).expand(b, nh, n, r).to(dtype=dtype)
    v_b = v.view(1, 1, n, r).expand(b, nh, n, r).to(dtype=dtype)
    zeros = torch.zeros(b, nh, n, r, device=dev, dtype=dtype)

    q_aug = torch.cat([query * content_scale, u_b], dim=-1)
    k_aug = torch.cat([key, v_b], dim=-1)
    v_aug = torch.cat([value, zeros], dim=-1)
    return q_aug, k_aug, v_aug, head_dim


def try_parb_lift_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    parb_ctx: PARBCtx,
    attention_interface: Callable,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float,
    **kwargs: Any,
) -> Optional[torch.Tensor]:
    """Run FA on augmented Q/K/V; return [B,N,H,d] or None to fall back."""
    feat = build_parb_lift_features(parb_ctx, device=query.device)
    if feat is None:
        return None

    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    q_aug, k_aug, v_aug, head_dim = parb_lift_concat_qkv(
        query,
        key_states,
        value_states,
        feat,
        content_scale=scaling,
    )

    saved_groups = module.num_key_value_groups
    module.num_key_value_groups = 1
    try:
        attn_output, _ = attention_interface(
            module,
            q_aug,
            k_aug,
            v_aug,
            attention_mask,
            dropout=dropout,
            scaling=1.0,
            **kwargs,
        )
    finally:
        module.num_key_value_groups = saved_groups
    attn_output = attn_output[..., :head_dim]
    if attn_output.dim() == 4 and attn_output.shape[1] != query.shape[2]:
        attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
