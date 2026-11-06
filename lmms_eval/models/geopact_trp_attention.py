"""TRP attention: rank-2 FA lift + TV-only hybrid oracle."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_trp import (
    TRPCtx,
    build_trp_bias,
    build_trp_lift_features,
    layer_active,
    trp_mode,
)


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_rep <= 1:
        return key, value
    b, n_kv, s, d = key.shape
    key = key[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    value = value[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    return key, value


def _causal_mask(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((n, n), float("-inf"), device=device, dtype=dtype)
    return m.masked_fill(torch.tril(torch.ones(n, n, device=device, dtype=torch.bool)), 0.0)


def trp_lift_concat_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ctx: TRPCtx,
    *,
    content_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Augment Q/K/V for rank-2 TRP lift. Tensors [B,H,N,d]."""
    head_dim = int(query.shape[-1])
    feat = build_trp_lift_features(ctx, device=device)
    r = int(feat.rank)
    b, nh, n, _ = query.shape
    dtype = query.dtype

    u = feat.u.to(device=device, dtype=dtype)
    v = feat.v.to(device=device, dtype=dtype)
    if u.shape[0] != n or v.shape[0] != n:
        raise ValueError(f"TRP lift seq mismatch: n={n} u={u.shape} v={v.shape}")

    u_b = u.view(1, 1, n, r).expand(b, nh, n, r)
    v_b = v.view(1, 1, n, r).expand(b, nh, n, r)
    zeros = torch.zeros(b, nh, n, r, device=query.device, dtype=dtype)

    q_aug = torch.cat([query * content_scale, u_b], dim=-1)
    k_aug = torch.cat([key, v_b], dim=-1)
    v_aug = torch.cat([value, zeros], dim=-1)
    return q_aug, k_aug, v_aug, head_dim


def trp_hybrid_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    trp_ctx: TRPCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Exact causal attention with additive TRP bias (oracle reference)."""
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    b, h, n, d = query.shape
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    causal = _causal_mask(n, scores.device, scores.dtype)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)

    bias = build_trp_bias(trp_ctx, device=scores.device, dtype=scores.dtype)
    if bias.shape[-1] == n:
        scores = scores + bias

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)
    out = torch.matmul(attn_weights, value_states)
    return out.transpose(1, 2).contiguous()


def try_trp_lift_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    trp_ctx: TRPCtx,
    attention_interface: Callable,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float,
    **kwargs: Any,
) -> torch.Tensor:
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    q_aug, k_aug, v_aug, head_dim = trp_lift_concat_qkv(
        query,
        key_states,
        value_states,
        trp_ctx,
        content_scale=scaling,
        device=query.device,
    )

    # key_states already expanded to num_heads; prevent double repeat in FA wrapper
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


def try_trp_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    trp_ctx: TRPCtx,
    attention_interface: Callable,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Optional[torch.Tensor]:
    mode = trp_mode()
    if mode == "lift":
        return try_trp_lift_attention(
            module,
            query,
            key,
            value,
            trp_ctx,
            attention_interface,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
    return trp_hybrid_forward(
        module,
        query,
        key,
        value,
        trp_ctx,
        scaling=scaling,
        dropout=dropout,
    )


def should_apply_trp(
    trp_ctx: Optional[TRPCtx],
    *,
    past_key_values,
    query_len: int,
    layer_idx: int,
) -> bool:
    if trp_ctx is None:
        return False
    if not layer_active(layer_idx):
        return False
    if query_len <= 1 and past_key_values is not None:
        return False
    if int(trp_ctx.seq_len) != int(query_len):
        return False
    return True
