"""PARB attention: exact logit bias + online softmax merge (Phase 1a/1b)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_parb import PARBCtx, build_parb_bias, parb_mode, parb_tv_enabled, parb_vv_enabled


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


def _row_logsumexp(scores: torch.Tensor) -> torch.Tensor:
    """Per-row LSE for scores [B,H,N,N] with -inf padding."""
    scores = scores.float()
    valid = torch.isfinite(scores)
    row_has = valid.any(dim=-1)
    safe = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(safe, dim=-1)
    neg_inf = torch.finfo(lse.dtype).min / 2
    return torch.where(row_has, lse, torch.full_like(lse, neg_inf))


def _attn_from_scores(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """scores [B,H,N,N], value [B,H,N,D] -> output, lse (log sum exp), max."""
    scores = scores.float()
    valid = torch.isfinite(scores)
    row_has = valid.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~valid, float("-inf"))
    max_s = scores.max(dim=-1, keepdim=True).values
    max_s = torch.where(row_has, max_s, torch.zeros_like(max_s))
    exp_s = torch.exp(scores - max_s) * valid.float()
    sum_exp = exp_s.sum(dim=-1, keepdim=True)
    sum_exp_safe = sum_exp.clamp_min(1e-9)
    probs = exp_s / sum_exp_safe
    out = torch.matmul(probs, value.float()).to(value.dtype)
    out = torch.where(row_has, out, torch.zeros_like(out))
    lse = max_s + torch.log(sum_exp.clamp_min(1e-9))
    lse = torch.where(row_has, lse, torch.full_like(lse, float("-inf")))
    return out, lse, max_s.squeeze(-1)


def online_softmax_combine(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Merge two disjoint key-group attention outputs (exact)."""
    lse_a = lse_a.float()
    lse_b = lse_b.float()
    neg_inf = torch.finfo(lse_a.dtype).min / 2
    valid_a = torch.isfinite(lse_a) & (lse_a > neg_inf)
    valid_b = torch.isfinite(lse_b) & (lse_b > neg_inf)
    lse_a = torch.where(valid_a, lse_a, torch.full_like(lse_a, neg_inf))
    lse_b = torch.where(valid_b, lse_b, torch.full_like(lse_b, neg_inf))
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m) * valid_a.float()
    wb = torch.exp(lse_b - m) * valid_b.float()
    denom = (wa + wb).clamp_min(1e-9)
    out = (wa.unsqueeze(-1) * out_a.float() + wb.unsqueeze(-1) * out_b.float()) / denom.unsqueeze(-1)
    only_a = valid_a & ~valid_b
    only_b = valid_b & ~valid_a
    out = torch.where(only_a.unsqueeze(-1), out_a.float(), out)
    out = torch.where(only_b.unsqueeze(-1), out_b.float(), out)
    return out.to(out_a.dtype)


def _sdpa_masked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    scaling: float,
    dropout: float,
) -> torch.Tensor:
    """SDPA with additive mask; uses flash/mem_efficient when supported."""
    drop_p = dropout if dropout > 0.0 else 0.0
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
    ):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=drop_p,
            scale=scaling,
        )


def parb_eager_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    parb_ctx: PARBCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Exact causal attention with additive PARB bias [B,H,N,N] (oracle)."""
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    b, h, n, d = query.shape
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    causal = _causal_mask(n, scores.device, scores.dtype)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)

    bias = build_parb_bias(parb_ctx, device=scores.device, dtype=scores.dtype)
    if bias.shape[-1] == n:
        scores = scores + bias

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)
    out = torch.matmul(attn_weights, value_states)
    return out.transpose(1, 2).contiguous()


def parb_hybrid_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    parb_ctx: PARBCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> Optional[torch.Tensor]:
    """FA-compatible hybrid: SDPA per bias partition + online LSE merge (exact)."""
    if not parb_vv_enabled() and not parb_tv_enabled():
        return None

    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    b, nh, n, d = query.shape
    causal = _causal_mask(n, query.device, query.dtype)
    causal4 = causal.unsqueeze(0).unsqueeze(0)

    bias = build_parb_bias(parb_ctx, device=query.device, dtype=query.dtype)
    if bias.shape[-1] != n:
        return parb_eager_forward(
            module, query, key, value, parb_ctx, scaling=scaling, dropout=dropout
        )

    has_bias = bias != 0
    if not bool(has_bias.any().item()):
        return None

    mask_a = causal4.masked_fill(has_bias, float("-inf"))
    mask_b = (causal4 + bias).masked_fill(~has_bias, float("-inf"))

    out_a = _sdpa_masked(
        query, key_states, value_states, mask_a, scaling=scaling, dropout=dropout
    )
    out_b = _sdpa_masked(
        query, key_states, value_states, mask_b, scaling=scaling, dropout=dropout
    )

    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling + causal4
    lse_a = _row_logsumexp(scores.masked_fill(has_bias, float("-inf")))
    lse_b = _row_logsumexp((scores + bias).masked_fill(~has_bias, float("-inf")))

    out = online_softmax_combine(out_a, lse_a, out_b, lse_b)
    return out.transpose(1, 2).contiguous()


def geopact_parb_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    parb_ctx: PARBCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> Optional[torch.Tensor]:
    mode = parb_mode()
    if mode == "hybrid":
        return parb_hybrid_forward(
            module, query, key, value, parb_ctx, scaling=scaling, dropout=dropout
        )
    return parb_eager_forward(
        module, query, key, value, parb_ctx, scaling=scaling, dropout=dropout
    )


def should_apply_parb(
    parb_ctx: Optional[PARBCtx],
    *,
    past_key_values,
    query_len: int,
    layer_idx: int,
) -> bool:
    if parb_ctx is None:
        return False
    from lmms_eval.models.geopact_parb import layer_active

    if not layer_active(layer_idx):
        return False
    if query_len <= 1 and past_key_values is not None:
        return False
    if int(parb_ctx.seq_len) != int(query_len):
        return False
    return True
