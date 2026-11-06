"""GSC FlashAttention with softmax_lse for LPR post-rescaling."""
from __future__ import annotations

from typing import Tuple

import torch


def _repeat_kv_bhnd(key: torch.Tensor, value: torch.Tensor, n_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n_kv = int(key.shape[1])
    n_rep = int(n_heads // n_kv)
    if n_rep <= 1:
        return key, value
    b, _, n, d = key.shape
    key = key[:, :, None, :, :].expand(b, n_kv, n_rep, n, d).reshape(b, n_kv * n_rep, n, d)
    value = value[:, :, None, :, :].expand(b, n_kv, n_rep, n, d).reshape(b, n_kv * n_rep, n, d)
    return key, value


def compute_softmax_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
) -> torch.Tensor:
    """Log-sum-exp per query row [B, H, N] from current q/k (post-ICC)."""
    b, h, n, _ = query.shape
    n_rep = int(h // key.shape[1])
    k, _ = _repeat_kv_bhnd(key, key, h)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if causal:
        mask = torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, n, n), float("-inf"))
    return torch.logsumexp(scores.float(), dim=-1)


def _bhnd_to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bshd_to_bhnd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def eager_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference attention [B,H,N,D] -> out [B,H,N,D], lse [B,H,N] float32."""
    b, h, n, d = query.shape
    key_exp, value_exp = _repeat_kv_bhnd(key, value, h)
    scores = torch.matmul(query, key_exp.transpose(2, 3)) * scaling
    if causal:
        mask = torch.triu(torch.ones(n, n, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, n, n), float("-inf"))
    scores_f = scores.float()
    lse = torch.logsumexp(scores_f, dim=-1)
    probs = torch.softmax(scores_f, dim=-1).to(query.dtype)
    out = torch.matmul(probs, value_exp)
    return out, lse


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Native d=128 FA forward; returns [B,H,N,D]."""
    if query.device.type != "cuda":
        out, _ = eager_attention_with_lse(query, key, value, scaling=scaling, causal=causal)
        return out

    q = _bhnd_to_bshd(query)
    k = _bhnd_to_bshd(key)
    v = _bhnd_to_bshd(value)

    try:
        from flash_attn import flash_attn_func

        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=scaling,
            causal=causal,
        )
        return _bshd_to_bhnd(out)
    except ImportError:
        out, _ = eager_attention_with_lse(query, key, value, scaling=scaling, causal=causal)
        return out


def flash_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single FA2 call + LSE from matching logits (post-ICC q/k).
    query/key/value: [B, H, N, D].
    """
    out = flash_attention_forward(
        query, key, value, scaling=scaling, causal=causal, dropout_p=dropout_p
    )
    lse = compute_softmax_lse(query, key, scaling=scaling, causal=causal)
    return out, lse


def apply_lpr_rescale(
    out: torch.Tensor,
    lse: torch.Tensor,
    Z_query: torch.Tensor,
    *,
    vfs_enabled: bool = True,
) -> torch.Tensor:
    """out [B,H,N,D], lse [B,H,N], Z_query [N] -> scaled out."""
    if not vfs_enabled:
        return out
    z = Z_query.to(device=out.device, dtype=torch.float32).view(1, 1, -1)
    lse_f = lse.float()
    gamma = 1.0 / (1.0 + z * torch.exp(-lse_f))
    return out * gamma.unsqueeze(-1).to(dtype=out.dtype)
