"""GeoPACT Module C — QASMR side-car relay attention stream (spec §6).

Two exact softmax streams merged by LSE:

  1. survivor stream: the unmodified KV cache through one FlashAttention2 call
     that also returns the softmax LSE (``softmax_scale=1.0`` with pre-scaled
     queries, causal bottom-right alignment — correct for both square prefill
     and rectangular decode/suffix shapes);
  2. relay stream: R' <= 64 relay objects per layer computed eagerly in fp32
     with logit  lambda_c(q) = (s q)·k̄_c + log m_c + log A_c(q)  and effective
     value  u_c(q) = v̄_c + g_c(q)/A_c(q)  (first order: A=1, g=0);
  3. exact mixture  y = (e^{LSE_S} y_S + e^{LSE_R} y_R) / (e^{LSE_S}+e^{LSE_R}).

The relays never enter the KV cache, so cached_input_ids / virtual MRoPE /
eviction ledgers stay untouched. GQA: relay statistics are stored per KV head
and broadcast to the query heads of the same group.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from lmms_eval.models.geopact_spatial_relay import (
    RelayCtx,
    relay_enabled,
    relay_layer_active,
)

__all__ = [
    "relay_enabled",
    "should_apply_relay",
    "try_relay_attention",
    "relay_stream",
    "merge_attention_streams",
    "eager_attention_with_lse_rect",
]


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------


def _bhnd_to_bshd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _bshd_to_bhnd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def eager_attention_with_lse_rect(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager attention returning (out, lse) with bottom-right causal alignment.

    query [B,H,N,d] (already scaled), key/value [B,H,S,d]; supports N < S
    (decode / suffix over cache) with the same alignment as flash_attn's
    ``causal=True``: query i attends keys j <= S - N + i.
    """
    b, h, n, _ = query.shape
    s = int(key.shape[2])
    dt = torch.promote_types(query.dtype, torch.float32)  # >=fp32, keep fp64
    scores = torch.matmul(query.to(dt), key.to(dt).transpose(2, 3))
    if causal:
        qi = torch.arange(n, device=scores.device).view(n, 1)
        kj = torch.arange(s, device=scores.device).view(1, s)
        mask = kj > (s - n) + qi
        scores = scores.masked_fill(mask.view(1, 1, n, s), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.to(value.dtype), value)
    return out, lse


def flash_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FA2 call returning (out [B,H,N,d], lse [B,H,N]) from one kernel.

    Query must already be scaled (softmax_scale=1.0). Falls back to the eager
    rectangular reference off-CUDA (tests); on CUDA flash_attn is mandatory.
    """
    if query.device.type != "cuda":
        return eager_attention_with_lse_rect(query, key, value, causal=causal)
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GeoPACT relay stream requires flash_attn on CUDA; no eager fallback"
        ) from exc

    q = _bhnd_to_bshd(query)
    k = _bhnd_to_bshd(key)
    v = _bhnd_to_bshd(value)
    result = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0,
        causal=causal,
        return_attn_probs=True,
    )
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("flash_attn_func did not return softmax_lse for relay merge")
    out, lse = result[0], result[1]
    # lse is [B,H,N] on recent flash_attn; tolerate [B,N,H].
    if lse.dim() == 3 and lse.shape[1] == q.shape[1] and lse.shape[2] == q.shape[2]:
        lse = lse.transpose(1, 2).contiguous()
    return _bshd_to_bhnd(out), lse.float()


def relay_stream(
    query_scaled: torch.Tensor,
    layer_relays: Dict[str, Optional[torch.Tensor]],
    *,
    num_key_value_groups: int,
    mode: str = "kv1",
    amin: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the relay softmax stream (y_R fp32 [B,H,N,dv], LSE_R [B,H,N]).

    query_scaled: [B,Hq,N,d] pre-scaled queries (so q·k̄ is already the logit).
    layer_relays: per-layer dict from RelayCtx.layer_tensors with KV-head
    layout kbar [Hkv,R,d], vbar [Hkv,R,dv], logm [R], optional second moments.

    Second order (mode == "kv2"):
        s²_c(q) = Σ_j kk_diag_{c,j} q_j²        (q already carries 1/τ)
        A_c(q)  = clamp(1 + ½ s²_c, min=amin)
        g_c(q)  = vk_diag_c ⊙ q     (diag)  |  U_c W_cᵀ q  (lowrank)
        logit  += log A_c;   value_c = v̄_c + g_c/A_c
    The g/A mixture term factorizes over relays, so no [B,H,N,R,dv] tensor is
    ever materialized:  Σ_c p_c g_c/A_c = ((p/A) @ vk_diag) ⊙ q  etc.
    """
    kbar = layer_relays["kbar"]
    vbar = layer_relays["vbar"]
    logm = layer_relays["logm"]
    if kbar is None or vbar is None or logm is None:
        raise RuntimeError("relay_stream: missing first-order relay tensors")

    dt = torch.promote_types(query_scaled.dtype, torch.float32)  # >=fp32, keep fp64
    q32 = query_scaled.to(dt)
    b, hq, n, d = q32.shape
    h0 = int(kbar.shape[0])
    groups = int(num_key_value_groups)
    if h0 == hq:
        groups = 1  # tensors were pre-broadcast by RelayCtx.layer_tensors
    elif h0 * groups != hq:
        raise RuntimeError(
            f"relay_stream: head mismatch Hq={hq} H0={h0} groups={groups}"
        )

    def rep(t: torch.Tensor) -> torch.Tensor:
        t = t.to(dt)
        return t if groups == 1 else t.repeat_interleave(groups, dim=0)

    kb_t = layer_relays.get("kbar_t")
    if kb_t is not None and groups == 1 and kb_t.dtype == dt:
        lam = torch.matmul(q32, kb_t)                     # [B,Hq,N,R]
    else:
        lam = torch.einsum("bhnd,hrd->bhnr", q32, rep(kbar))
    lam = lam + logm.to(dt).view(1, 1, 1, -1)
    vb = rep(vbar)                                        # [Hq,R,dv]

    second = mode == "kv2"
    a_c = None
    if second:
        kk = layer_relays.get("kk_diag")
        if kk is None:
            second = False
        else:
            s2 = torch.einsum("bhnd,hrd->bhnr", q32.pow(2), rep(kk))
            a_c = (1.0 + 0.5 * s2).clamp(min=float(amin))
            lam = lam + torch.log(a_c)

    lse_r = torch.logsumexp(lam, dim=-1)                  # [B,H,N]
    p = torch.exp(lam - lse_r.unsqueeze(-1))              # softmax reusing lse
    y = torch.matmul(p, vb)                               # [B,H,N,dv]

    if second and a_c is not None:
        c = p / a_c  # [B,H,N,R]
        vk_diag = layer_relays.get("vk_diag")
        vk_u = layer_relays.get("vk_u")
        vk_w = layer_relays.get("vk_w")
        if vk_u is not None and vk_w is not None:
            u_rep = rep(vk_u)                              # [Hq,R,dv,r]
            w_rep = rep(vk_w)                              # [Hq,R,d,r]
            t = torch.einsum("bhnd,hrdk->bhnrk", q32, w_rep)  # [B,H,N,R,r]
            y = y + torch.einsum("bhnrk,hrvk,bhnr->bhnv", t, u_rep, c)
        elif vk_diag is not None:
            y = y + torch.einsum("bhnr,hrd->bhnd", c, rep(vk_diag)) * q32

    return y, lse_r


def merge_attention_streams(
    y_s: torch.Tensor,
    lse_s: torch.Tensor,
    y_r: torch.Tensor,
    lse_r: torch.Tensor,
) -> torch.Tensor:
    """Exact mixture of two normalized softmax streams (>=fp32, shift-stable).

    The two-way softmax weight is exactly a sigmoid of the LSE difference:
      e^{LSE_R} / (e^{LSE_S} + e^{LSE_R}) = sigmoid(LSE_R - LSE_S),
    so the merge is one sigmoid + one lerp — minimal kernel count on the
    per-token decode path, and identical to the max-shifted normalization.
    """
    dt = torch.promote_types(torch.promote_types(y_s.dtype, y_r.dtype), torch.float32)
    w_r = torch.sigmoid(lse_r.to(dt) - lse_s.to(dt)).unsqueeze(-1)
    return torch.lerp(y_s.to(dt), y_r.to(dt), w_r)


# ---------------------------------------------------------------------------
# Model-facing hooks (mirrors the VSAS/DMTC bypass conventions)
# ---------------------------------------------------------------------------


_WARNED_NON_FA2 = False


def should_apply_relay(
    relay_ctx: Optional[RelayCtx],
    *,
    attention_mask: Optional[torch.Tensor],
    layer_idx: int,
    query_len: int,
    attn_implementation: Optional[str] = None,
    device_type: Optional[str] = None,
) -> bool:
    """Side-car applies on every forward (prefill and decode) while relays are
    active, but only under full past visibility (mask None or all-ones 2D).

    FA2 zero-fallback hard constraint (master spec §3.1): on CUDA the relay
    only engages when the model runs flash_attention_2 — if the backend is
    eager/sdpa we disable the relay (accuracy degrades to pure pruning) rather
    than ever changing the main attention path. Off-CUDA (CPU tests) the
    eager reference implementation is allowed.
    """
    if relay_ctx is None or not relay_ctx.frames:
        return False
    if not relay_layer_active(layer_idx):
        return False
    if query_len <= 0:
        return False
    if (
        attn_implementation is not None
        and (device_type or "cuda") == "cuda"
        and str(attn_implementation) != "flash_attention_2"
    ):
        global _WARNED_NON_FA2
        if not _WARNED_NON_FA2:
            _WARNED_NON_FA2 = True
            warnings.warn(
                "GeoPACT relay disabled: attn_implementation="
                f"{attn_implementation!r} != flash_attention_2 on CUDA "
                "(zero-fallback policy: relay off, main path untouched)"
            )
        return False
    if attention_mask is not None:
        if attention_mask.dim() != 2:
            return False
        if not bool((attention_mask != 0).all().item()):
            return False
    return True


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if groups == 1:
        return key, value
    return (
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
    )


def try_relay_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relay_ctx: RelayCtx,
    *,
    scaling: float,
    layer_idx: int,
) -> torch.Tensor:
    """Two-stream QASMR attention; returns [B, N, Hq, d] (pre-o_proj layout).

    query [B,Hq,N,d] post-RoPE; key/value [B,Hkv,S,*] the full cached states.
    """
    groups = int(module.num_key_value_groups)
    layer_relays = relay_ctx.layer_tensors(int(layer_idx), query.device, groups=groups)
    if layer_relays is None:
        raise RuntimeError("try_relay_attention called with empty relay ctx")
    k_rep, v_rep = _repeat_kv(key, value, groups)
    q_scaled = query * float(scaling)

    y_s, lse_s = flash_attention_with_lse(q_scaled, k_rep, v_rep, causal=True)
    y_r, lse_r = relay_stream(
        q_scaled,
        layer_relays,
        num_key_value_groups=groups,
        mode=relay_ctx.mode,
        amin=relay_ctx.amin,
    )
    merged = merge_attention_streams(y_s, lse_s, y_r, lse_r)

    if os.environ.get("GEOPACT_C_RELAY_AUDIT", "0").strip().lower() in ("1", "true", "yes", "on"):
        try:
            frac = (torch.exp(lse_r - torch.maximum(lse_s, lse_r)).mean()).item()
            if layer_idx == 0:
                warnings.warn(
                    f"GeoPACT relay audit layer0: relay mass frac~{frac:.4g} "
                    f"R={layer_relays['logm'].numel()} q_len={int(query.shape[2])}"
                )
        except Exception:
            pass

    return merged.to(query.dtype).transpose(1, 2).contiguous()
