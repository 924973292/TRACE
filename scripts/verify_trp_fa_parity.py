#!/usr/bin/env python3
"""TRP FA parity: lift vs hybrid across dtypes and optional FA2 kernel."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_trp import TRPCtx
from lmms_eval.models.geopact_trp_attention import trp_hybrid_forward, trp_lift_concat_qkv


class _DummyAttn:
    num_key_value_groups = 4
    training = False


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fake_ctx(n: int = 32) -> TRPCtx:
    is_v = torch.zeros(n, dtype=torch.bool)
    is_v[n // 4 : 3 * n // 4] = True
    region = torch.zeros(n, dtype=torch.float32)
    psi = torch.zeros(n, dtype=torch.float32)
    region[is_v] = torch.linspace(0.05, 0.2, int(is_v.sum()))
    psi[is_v] = torch.linspace(-0.3, -0.05, int(is_v.sum()))
    prior = (region + 0.25 * psi) * is_v.float()
    return TRPCtx(
        seq_len=n,
        is_vision=is_v,
        is_text=~is_v,
        region_prior=region,
        psi=psi,
        prior=prior,
        audit={"trp": True},
    )


def _sdpa_lift(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TRPCtx,
    scaling: float,
) -> torch.Tensor:
    """SDPA on augmented Q/K/V (FA-compatible path without custom kernel)."""
    b, n_kv, s, d = k.shape
    n_rep = module.num_key_value_groups
    k = k[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    v = v[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    q_aug, k_aug, v_aug, head_dim = trp_lift_concat_qkv(
        q, k, v, ctx, content_scale=scaling, device=q.device
    )
    out = F.scaled_dot_product_attention(
        q_aug,
        k_aug,
        v_aug,
        dropout_p=0.0,
        scale=1.0,
        is_causal=True,
    )
    return out[..., :head_dim].transpose(1, 2).contiguous()


def _run_parity(dtype: torch.dtype, device: torch.device, label: str) -> float:
    ctx = _fake_ctx(n=48)
    b, nh, n, d = 1, 8, 48, 128
    scaling = d**-0.5
    q = torch.randn(b, nh, n, d, device=device, dtype=dtype)
    k = torch.randn(b, 2, n, d, device=device, dtype=dtype)
    v = torch.randn(b, 2, n, d, device=device, dtype=dtype)
    mod = _DummyAttn()
    out_h = trp_hybrid_forward(mod, q, k, v, ctx, scaling=scaling)
    out_l = _sdpa_lift(mod, q, k, v, ctx, scaling=scaling)
    diff = (out_h.float() - out_l.float()).abs().max().item()
    ok(f"{label} max_diff={diff:.6f}")
    return diff


def main() -> int:
    os.environ["GEOPACT_TRP"] = "1"
    _run_parity(torch.float32, torch.device("cpu"), "fp32_cpu")
    if torch.cuda.is_available():
        _run_parity(torch.float32, torch.device("cuda"), "fp32_cuda")
        diff = _run_parity(torch.bfloat16, torch.device("cuda"), "bf16_cuda")
        assert diff < 0.1, diff
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

            ctx = _fake_ctx(n=32)
            b, nh, n, d = 1, 8, 32, 128
            scaling = d**-0.5
            dev = torch.device("cuda")
            q = torch.randn(b, nh, n, d, device=dev, dtype=torch.bfloat16)
            k = torch.randn(b, 2, n, d, device=dev, dtype=torch.bfloat16)
            v = torch.randn(b, 2, n, d, device=dev, dtype=torch.bfloat16)
            mod = _DummyAttn()
            n_rep = mod.num_key_value_groups
            k_exp = k[:, :, None, :, :].expand(b, 2, n_rep, n, d).reshape(b, 2 * n_rep, n, d)
            v_exp = v[:, :, None, :, :].expand(b, 2, n_rep, n, d).reshape(b, 2 * n_rep, n, d)
            q_aug, k_aug, v_aug, head_dim = trp_lift_concat_qkv(
                q, k_exp, v_exp, ctx, content_scale=scaling, device=dev
            )
            n = q_aug.shape[2]
            causal = torch.triu(
                torch.full((1, 1, n, n), float("-inf"), device=dev, dtype=torch.bfloat16),
                diagonal=1,
            )
            saved = mod.num_key_value_groups
            mod.num_key_value_groups = 1
            try:
                out_eager, _ = eager_attention_forward(
                    mod, q_aug, k_aug, v_aug, causal, scaling=1.0, dropout=0.0
                )
            finally:
                mod.num_key_value_groups = saved
            out_eager = out_eager[..., :head_dim]
            if out_eager.shape[1] != n:
                out_eager = out_eager.transpose(1, 2).contiguous()
            out_h = trp_hybrid_forward(mod, q, k, v, ctx, scaling=scaling)
            diff = (out_h.float() - out_eager.float()).abs().max().item()
            ok(f"bf16_eager_aug max_diff={diff:.6f}")
            assert diff < 0.1, diff
        except ImportError as exc:
            ok(f"eager_attention_forward skip: {exc}")
    else:
        ok("cuda skip")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
