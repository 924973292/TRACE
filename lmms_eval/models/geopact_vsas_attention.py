"""VSAS path A: internal K/V concat with FlashAttention (KV-safe)."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_sampling_correction import build_anchor_hidden
from lmms_eval.models.geopact_vsas import VSASCtx, layer_active


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, n_rep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_rep <= 1:
        return key, value
    b, n_kv, s, d = key.shape
    key = key[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    value = value[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    return key, value


def _bhnd_to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bshd_to_bhnd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def eager_vsas_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
) -> torch.Tensor:
    b, h, nq, d = query.shape
    nk = key.shape[1]
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    n = int(key.shape[2])
    if causal:
        mask = torch.arange(nq, device=scores.device).view(-1, 1) >= torch.arange(
            n, device=scores.device
        ).view(1, -1)
        scores = scores.masked_fill(~mask.view(1, 1, nq, n), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, value)


def flash_vsas_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if query.device.type != "cuda":
        return eager_vsas_attention(query, key, value, scaling=scaling, causal=causal)
    from flash_attn import flash_attn_func

    q = _bhnd_to_bshd(query)
    k = _bhnd_to_bshd(key)
    v = _bhnd_to_bshd(value)
    out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=scaling,
        causal=causal,
    )
    return _bshd_to_bhnd(out)


def try_vsas_attention_internal(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    vsas_ctx: VSASCtx,
    *,
    scaling: float,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    dropout: float = 0.0,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Concat virtual K/V; attend; return output for first N queries only."""
    del layer_idx
    vis_idx = vsas_ctx.vis_indices.to(hidden_states.device)
    omega = vsas_ctx.omega.to(hidden_states.device, dtype=torch.float32)
    x_anchor = build_anchor_hidden(hidden_states, omega, vis_indices=vis_idx)

    k_anchor = module.k_norm(
        module.k_proj(x_anchor).view(
            x_anchor.shape[0],
            x_anchor.shape[1],
            module.num_key_value_heads,
            module.head_dim,
        )
    ).transpose(1, 2)
    v_anchor = module.v_proj(x_anchor).view(
        x_anchor.shape[0],
        x_anchor.shape[1],
        module.num_key_value_heads,
        module.head_dim,
    ).transpose(1, 2)

    if vsas_ctx.denom_only:
        v_anchor = torch.zeros_like(v_anchor)

    del position_embeddings

    key_full, value_full = _repeat_kv(key, value, module.num_key_value_groups)
    k_cat = torch.cat([key_full, k_anchor], dim=2)
    v_cat = torch.cat([value_full, v_anchor], dim=2)

    out = flash_vsas_attention(
        query,
        k_cat,
        v_cat,
        scaling=scaling,
        causal=True,
        dropout_p=dropout if module.training else 0.0,
    )
    n_q = int(query.shape[2])
    return out[:, :, :n_q, :].transpose(1, 2).contiguous()


def should_apply_vsas(
    vsas_ctx: Optional[VSASCtx],
    *,
    past_key_values,
    query_len: int,
    layer_idx: int,
) -> bool:
    if vsas_ctx is None:
        return False
    if vsas_ctx.mode != "internal":
        return False
    if not layer_active(layer_idx):
        return False
    if query_len <= 1 and past_key_values is not None:
        return False
    base = int(vsas_ctx.seq_base_len)
    if int(query_len) != base:
        return False
    return True
