#!/usr/bin/env python3
"""Verify TRP lift vs hybrid oracle (unit gate)."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_trp import TRPCtx, build_trp_bias, build_trp_lift_features
from lmms_eval.models.geopact_trp_attention import (
    trp_hybrid_forward,
    trp_lift_concat_qkv,
)


class _DummyAttn:
    num_key_value_groups = 1
    training = False


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fake_ctx(n: int = 8) -> TRPCtx:
    is_v = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    region = torch.tensor([0, 0, 0.1, 0.1, 0.2, 0.15, 0, 0], dtype=torch.float32)
    psi = torch.tensor([0, 0, -0.1, -0.2, -0.3, -0.1, 0, 0], dtype=torch.float32)
    eta = 0.25
    prior = (region + eta * psi) * is_v.float()
    return TRPCtx(
        seq_len=n,
        is_vision=is_v,
        is_text=~is_v,
        region_prior=region,
        psi=psi,
        prior=prior,
        audit={"trp": True},
    )


def _lift_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: TRPCtx,
    scaling: float,
) -> torch.Tensor:
    n_rep = int(q.shape[1] // k.shape[1])
    if n_rep > 1:
        b, n_kv, s, d = k.shape
        k = k[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
        v = v[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    q_aug, k_aug, v_aug, head_dim = trp_lift_concat_qkv(
        q, k, v, ctx, content_scale=scaling, device=q.device
    )
    scores = torch.matmul(q_aug, k_aug.transpose(2, 3))
    n = scores.shape[-1]
    causal = torch.full((n, n), float("-inf"), device=scores.device, dtype=scores.dtype)
    causal = causal.masked_fill(torch.tril(torch.ones(n, n, device=scores.device, dtype=torch.bool)), 0.0)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_aug)
    return out[..., :head_dim].transpose(1, 2).contiguous()


def test_bias_matrix() -> None:
    ctx = _fake_ctx()
    bias = build_trp_bias(ctx, device=torch.device("cpu"), dtype=torch.float32)[0, 0]
    feat = build_trp_lift_features(ctx, device=torch.device("cpu"))
    approx = feat.u @ feat.v.T
    err = (bias - approx).abs().max().item()
    assert err < 1e-5, err
    ok(f"lift bias reconstruct max_err={err:.6f}")


def test_lift_vs_hybrid() -> None:
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_h = trp_hybrid_forward(mod, q, k, v, ctx, scaling=scaling)
    out_l = _lift_matmul(q, k, v, ctx, scaling)
    diff = (out_h - out_l).abs().max().item()
    assert diff < 1e-2, diff
    ok(f"lift vs hybrid max_diff={diff:.6f}")


def test_lift_vs_hybrid_bf16() -> None:
    if not torch.cuda.is_available():
        ok("bf16 skip (no cuda)")
        return
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    dev = torch.device("cuda")
    q = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, 1, n, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, 1, n, d, device=dev, dtype=torch.bfloat16)
    mod = _DummyAttn()
    out_h = trp_hybrid_forward(mod, q, k, v, ctx, scaling=scaling)
    out_l = _lift_matmul(q, k, v, ctx, scaling)
    diff = (out_h.float() - out_l.float()).abs().max().item()
    assert diff < 0.05, diff
    ok(f"lift vs hybrid bf16 max_diff={diff:.6f}")


def main() -> int:
    os.environ["GEOPACT_TRP"] = "1"
    test_bias_matrix()
    test_lift_vs_hybrid()
    test_lift_vs_hybrid_bf16()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
