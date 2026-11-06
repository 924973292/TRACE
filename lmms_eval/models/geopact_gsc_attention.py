"""GSC attention: single-stream FA + ICC + LPR (production); legacy sink/lift oracle."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_gsc import (
    GSCCtx,
    build_b_vision_icc,
    build_gsc_lift_features,
    build_gsc_spf_bias,
    gsc_mode,
    gsc_query_adapt_enabled,
    gsc_spf_mode,
    gsc_uses_fa_path,
    gsc_vfs_enabled,
    layer_active,
)
from lmms_eval.models.geopact_gsc_fa import apply_lpr_rescale, flash_attention_with_lse


def _expand_attention_mask_for_vfs_sink(
    attention_mask: Optional[torch.Tensor],
    *,
    n_compact: int,
    n_ext: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Prepend virtual sink dim: all queries see sink key; sink query sees only sink."""
    if attention_mask is None or n_ext <= n_compact:
        return attention_mask
    neg = torch.finfo(dtype).min
    if attention_mask.dim() == 4:
        b, nh = int(attention_mask.shape[0]), int(attention_mask.shape[1])
        out = torch.full((b, nh, n_ext, n_ext), neg, device=device, dtype=dtype)
        out[:, :, 0, 0] = 0
        out[:, :, 1:, 0] = 0
        nc = min(n_compact, int(attention_mask.shape[-1]))
        out[:, :, 1 : 1 + nc, 1 : 1 + nc] = attention_mask[:, :, :nc, :nc].to(device=device, dtype=dtype)
        return out
    return attention_mask


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


def _query_gate_from_q(query: torch.Tensor) -> torch.Tensor:
    """Per-position gate g(q_i) = ||q_i|| / sqrt(d) on text rows (filled in caller)."""
    d = int(query.shape[-1])
    return query.norm(dim=-1).mean(dim=1) / max(d**0.5, 1e-6)


def _compute_query_gate(query: torch.Tensor, ctx: GSCCtx) -> Optional[torch.Tensor]:
    if not gsc_query_adapt_enabled():
        return None
    # [B, H, N] -> [N] mean over batch/heads
    g = _query_gate_from_q(query)
    n = int(ctx.seq_len)
    out = torch.ones(n, device=query.device, dtype=torch.float32)
    text = ctx.is_text.to(query.device)
    if text.any():
        g_mean = float(g.mean().item())
        out[text] = g_mean
    return out


def gsc_lift_concat_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ctx: GSCCtx,
    *,
    content_scale: float,
    device: torch.device,
    layer_idx: int = 0,
    num_layers: int = 36,
    query_gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Augment Q/K/V; optionally prepend virtual flux sink at index 0. Tensors [B,H,N,d]."""
    head_dim = int(query.shape[-1])
    b, nh, n, _ = query.shape
    dtype = query.dtype
    use_vfs = gsc_vfs_enabled()
    spf = build_gsc_lift_features(
        ctx,
        device=device,
        layer_idx=layer_idx,
        num_layers=num_layers,
        query_gate=query_gate,
    )
    r_spf = int(spf.rank_spf)
    r_vfs = 1 if use_vfs else 0
    r_total = r_vfs + r_spf

    log_z = ctx.log_Z.to(device=device, dtype=dtype)
    u = spf.u.to(device=device, dtype=dtype)
    v = spf.v.to(device=device, dtype=dtype)

    if use_vfs:
        n_ext = n + 1
        q_content = torch.cat(
            [torch.zeros(b, nh, 1, head_dim, device=query.device, dtype=dtype), query * content_scale],
            dim=2,
        )
        k_content = torch.cat(
            [torch.zeros(b, key.shape[1], 1, head_dim, device=key.device, dtype=dtype), key],
            dim=2,
        )
        v_content = torch.cat(
            [torch.zeros(b, value.shape[1], 1, head_dim, device=value.device, dtype=dtype), value],
            dim=2,
        )

        q_aug = torch.zeros(b, nh, n_ext, r_total, device=query.device, dtype=dtype)
        k_aug = torch.zeros(b, key.shape[1], n_ext, r_total, device=key.device, dtype=dtype)
        v_aug = torch.zeros(b, value.shape[1], n_ext, r_total, device=value.device, dtype=dtype)

        k_aug[:, :, 0, 0] = 1.0
        q_aug[:, :, 1:, 0] = log_z.view(1, 1, n).expand(b, nh, n)
        if r_spf > 0:
            q_aug[:, :, 1:, r_vfs:] = u.view(1, 1, n, r_spf).expand(b, nh, n, r_spf)
            k_aug[:, :, 1:, r_vfs:] = v.view(1, 1, n, r_spf).expand(b, key.shape[1], n, r_spf)

        q_out = torch.cat([q_content, q_aug], dim=-1)
        k_out = torch.cat([k_content, k_aug], dim=-1)
        v_out = torch.cat([v_content, v_aug], dim=-1)
        sink_offset = 1
    else:
        if r_spf <= 0:
            return query * content_scale, key, value, head_dim, 0
        zeros = torch.zeros(b, nh, n, r_spf, device=query.device, dtype=dtype)
        u_b = u.view(1, 1, n, r_spf).expand(b, nh, n, r_spf)
        v_b = v.view(1, 1, n, r_spf).expand(b, key.shape[1], n, r_spf)
        q_out = torch.cat([query * content_scale, u_b], dim=-1)
        k_out = torch.cat([key, v_b], dim=-1)
        v_out = torch.cat([value, zeros], dim=-1)
        sink_offset = 0

    return q_out, k_out, v_out, head_dim, sink_offset


def gsc_hybrid_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gsc_ctx: GSCCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
    layer_idx: int = 0,
    num_layers: int = 36,
) -> torch.Tensor:
    """Exact causal attention with VFS sink + SPF bias (oracle reference)."""
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    b, h, n, d = query.shape
    use_vfs = gsc_vfs_enabled()
    query_gate = _compute_query_gate(query, gsc_ctx) if gsc_query_adapt_enabled() else None

    if use_vfs:
        n_ext = n + 1
        q_z = torch.zeros(b, h, 1, d, device=query.device, dtype=query.dtype)
        k_z = torch.zeros(b, key_states.shape[1], 1, d, device=query.device, dtype=query.dtype)
        v_z = torch.zeros(b, value_states.shape[1], 1, d, device=query.device, dtype=query.dtype)
        q_ext = torch.cat([q_z, query], dim=2)
        k_ext = torch.cat([k_z, key_states], dim=2)
        v_ext = torch.cat([v_z, value_states], dim=2)

        scores = torch.matmul(q_ext, k_ext.transpose(2, 3)) * scaling
        causal = _causal_mask(n_ext, scores.device, scores.dtype)
        scores = scores + causal.unsqueeze(0).unsqueeze(0)

        log_z = gsc_ctx.log_Z.to(device=scores.device, dtype=scores.dtype)
        scores[:, :, 1:, 0] = log_z.view(1, 1, n).expand(b, h, n)

        if gsc_spf_mode() != "off":
            spf_bias = build_gsc_spf_bias(
                gsc_ctx,
                device=scores.device,
                dtype=scores.dtype,
                layer_idx=layer_idx,
                num_layers=num_layers,
                query_gate=query_gate,
            )
            if spf_bias.shape[-1] == n:
                scores[:, :, 1:, 1:] = scores[:, :, 1:, 1:] + spf_bias

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout > 0.0 and module.training:
            attn_weights = F.dropout(attn_weights, p=dropout)
        out = torch.matmul(attn_weights, v_ext)
        out = out[:, :, 1:, :]
    else:
        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        causal = _causal_mask(n, scores.device, scores.dtype)
        scores = scores + causal.unsqueeze(0).unsqueeze(0)
        if gsc_spf_mode() != "off":
            spf_bias = build_gsc_spf_bias(
                gsc_ctx,
                device=scores.device,
                dtype=scores.dtype,
                layer_idx=layer_idx,
                num_layers=num_layers,
                query_gate=query_gate,
            )
            scores = scores + spf_bias
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout > 0.0 and module.training:
            attn_weights = F.dropout(attn_weights, p=dropout)
        out = torch.matmul(attn_weights, value_states)

    return out.transpose(1, 2).contiguous()


def _apply_icc_channels(
    query: torch.Tensor,
    key: torch.Tensor,
    is_vision: torch.Tensor,
    b_vision: torch.Tensor,
) -> None:
    """In-place ICC on last dim: text q=1, vision k=b_j."""
    n = int(query.shape[2])
    vis = is_vision.to(device=query.device).view(1, 1, n)
    b_row = b_vision.to(device=query.device, dtype=key.dtype).view(1, 1, n)
    query[..., -1] = (~vis).to(dtype=query.dtype)
    key[..., -1] = torch.where(vis, b_row, torch.zeros(1, device=key.device, dtype=key.dtype))


def try_gsc_fa_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gsc_ctx: GSCCtx,
    *,
    scaling: float,
    dropout: float = 0.0,
    layer_idx: int = 0,
    num_layers: int = 36,
    **kwargs: Any,
) -> torch.Tensor:
    """Production path: ICC (optional SPF) + single flash_attn + LPR rescale."""
    del kwargs
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)

    q = query.clone()
    k = key_states.clone()

    query_gate = _compute_query_gate(query, gsc_ctx) if gsc_query_adapt_enabled() else None
    if gsc_spf_mode() != "off":
        b_j = build_b_vision_icc(
            gsc_ctx,
            device=query.device,
            dtype=query.dtype,
            layer_idx=layer_idx,
            num_layers=num_layers,
            scaling=scaling,
            query_gate=query_gate,
        )
        _apply_icc_channels(q, k, gsc_ctx.is_vision, b_j)

    out, lse = flash_attention_with_lse(
        q,
        k,
        value_states,
        scaling=scaling,
        causal=True,
        dropout_p=dropout if module.training else 0.0,
    )
    out = apply_lpr_rescale(
        out,
        lse,
        gsc_ctx.Z,
        vfs_enabled=gsc_vfs_enabled(),
    )
    return out.transpose(1, 2).contiguous()


def try_gsc_lift_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gsc_ctx: GSCCtx,
    attention_interface: Callable,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float,
    layer_idx: int = 0,
    num_layers: int = 36,
    **kwargs: Any,
) -> torch.Tensor:
    n_rep = module.num_key_value_groups
    key_states, value_states = _repeat_kv(key, value, n_rep)
    query_gate = _compute_query_gate(query, gsc_ctx) if gsc_query_adapt_enabled() else None

    q_aug, k_aug, v_aug, head_dim, sink_offset = gsc_lift_concat_qkv(
        query,
        key_states,
        value_states,
        gsc_ctx,
        content_scale=scaling,
        device=query.device,
        layer_idx=layer_idx,
        num_layers=num_layers,
        query_gate=query_gate,
    )

    attn_mask = attention_mask
    if sink_offset > 0:
        n_ext = int(q_aug.shape[2])
        attn_mask = _expand_attention_mask_for_vfs_sink(
            attention_mask,
            n_compact=int(gsc_ctx.seq_len),
            n_ext=n_ext,
            device=query.device,
            dtype=query.dtype,
        )

    saved_groups = module.num_key_value_groups
    module.num_key_value_groups = 1
    try:
        attn_output, _ = attention_interface(
            module,
            q_aug,
            k_aug,
            v_aug,
            attn_mask,
            dropout=dropout,
            scaling=1.0,
            **kwargs,
        )
    finally:
        module.num_key_value_groups = saved_groups

    attn_output = attn_output[..., :head_dim]
    if sink_offset > 0:
        if int(attn_output.shape[2]) == int(query.shape[2]) + sink_offset:
            attn_output = attn_output[:, :, sink_offset:, :]
        elif int(attn_output.shape[1]) == int(query.shape[2]) + sink_offset:
            attn_output = attn_output[:, sink_offset:, :, :]
    if attn_output.dim() == 4 and attn_output.shape[1] != query.shape[2]:
        attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def try_gsc_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gsc_ctx: GSCCtx,
    attention_interface: Callable,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float = 0.0,
    layer_idx: int = 0,
    num_layers: int = 36,
    **kwargs: Any,
) -> Optional[torch.Tensor]:
    mode = gsc_mode()
    if gsc_uses_fa_path():
        return try_gsc_fa_attention(
            module,
            query,
            key,
            value,
            gsc_ctx,
            scaling=scaling,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers,
            **kwargs,
        )
    if mode == "lift":
        return try_gsc_lift_attention(
            module,
            query,
            key,
            value,
            gsc_ctx,
            attention_interface,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers,
            **kwargs,
        )
    return gsc_hybrid_forward(
        module,
        query,
        key,
        value,
        gsc_ctx,
        scaling=scaling,
        dropout=dropout,
        layer_idx=layer_idx,
        num_layers=num_layers,
    )


def should_apply_gsc(
    gsc_ctx: Optional[GSCCtx],
    *,
    past_key_values,
    query_len: int,
    layer_idx: int,
) -> bool:
    if gsc_ctx is None:
        return False
    if not layer_active(layer_idx):
        return False
    if query_len <= 1 and past_key_values is not None:
        return False
    if int(gsc_ctx.seq_len) != int(query_len):
        return False
    return True
