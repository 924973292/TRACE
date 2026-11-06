#!/usr/bin/env python3
"""Verify PARB-Lift vs oracle eager/hybrid attention."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_parb import PARBCtx, build_parb_bias
from lmms_eval.models.geopact_parb_attention import parb_eager_forward, parb_hybrid_forward
from lmms_eval.models.geopact_parb_lift import (
    build_parb_lift_features,
    lift_bias_matrix,
)
from lmms_eval.models.geopact_parb_lift_attention import parb_lift_concat_qkv


class _DummyAttn:
    num_key_value_groups = 1
    training = False


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fake_ctx(n: int = 8) -> PARBCtx:
    is_v = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    h = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0], dtype=torch.float32)
    w = torch.tensor([0, 0, 0, 1, 0, 1, 0, 0], dtype=torch.float32)
    miu = torch.tensor([-1, -1, 0, 0, 1, 1, -1, -1], dtype=torch.long)
    rho = torch.tensor([0, 0, 0.5, 0.5, 0.8, 0.8, 0, 0], dtype=torch.float32)
    region = torch.tensor([0, 0, 0.1, 0.1, 0.2, 0.15, 0, 0], dtype=torch.float32)
    psi = torch.tensor([0, 0, -0.1, -0.2, -0.3, -0.1, 0, 0], dtype=torch.float32)
    mius = (
        {"box": (0.1, 0.1, 0.3, 0.3), "info_density": 1.0},
        {"box": (0.5, 0.5, 0.9, 0.9), "info_density": 0.8},
    )
    return PARBCtx(
        seq_len=n,
        is_vision=is_v,
        h=h,
        w=w,
        miu_id=miu,
        rho=rho,
        region_prior=region,
        psi=psi,
        mius=mius,
        audit={"parb": True},
    )


def _lift_matmul_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: PARBCtx,
    scaling: float,
) -> torch.Tensor:
    feat = build_parb_lift_features(ctx, device=q.device)
    assert feat is not None
    n_rep = int(q.shape[1] // k.shape[1])
    if n_rep > 1:
        b, n_kv, s, d = k.shape
        k = k[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
        v = v[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)
    q_aug, k_aug, v_aug, head_dim = parb_lift_concat_qkv(q, k, v, feat, content_scale=scaling)
    scores = torch.matmul(q_aug, k_aug.transpose(2, 3))
    n = scores.shape[-1]
    causal = torch.full((n, n), float("-inf"), device=scores.device, dtype=scores.dtype)
    causal = causal.masked_fill(torch.tril(torch.ones(n, n, device=scores.device, dtype=torch.bool)), 0.0)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(probs, v_aug)
    return out[..., :head_dim].transpose(1, 2).contiguous()


def test_svd_bias_fit() -> None:
    os.environ["GEOPACT_PARB"] = "1"
    os.environ["GEOPACT_PARB_LIFT_MODE"] = "svd"
    os.environ["GEOPACT_PARB_LIFT_RANK"] = "8"
    ctx = _fake_ctx()
    oracle = build_parb_bias(ctx, device=torch.device("cpu"), dtype=torch.float32)[0, 0]
    feat = build_parb_lift_features(ctx, device=torch.device("cpu"))
    assert feat is not None
    approx = lift_bias_matrix(feat)
    err = (oracle - approx).abs().max().item()
    assert err < 0.5, err
    ok(f"svd bias fit max_err={err:.4f}")


def test_lift_tv_exact() -> None:
    os.environ["GEOPACT_PARB"] = "1"
    os.environ["GEOPACT_PARB_VV"] = "0"
    os.environ["GEOPACT_PARB_TV"] = "1"
    os.environ["GEOPACT_PARB_LIFT_MODE"] = "decomposed"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_e = parb_eager_forward(mod, q, k, v, ctx, scaling=scaling)
    out_l = _lift_matmul_forward(q, k, v, ctx, scaling)
    diff = (out_e - out_l).abs().max().item()
    assert diff < 1e-2, diff
    ok(f"lift_decomposed_tv vs eager max_diff={diff:.6f}")


def test_lift_svd_vs_eager() -> None:
    os.environ["GEOPACT_PARB"] = "1"
    os.environ["GEOPACT_PARB_VV"] = "1"
    os.environ["GEOPACT_PARB_TV"] = "1"
    os.environ["GEOPACT_PARB_LIFT_MODE"] = "svd"
    os.environ["GEOPACT_PARB_LIFT_RANK"] = "16"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_e = parb_eager_forward(mod, q, k, v, ctx, scaling=scaling)
    out_l = _lift_matmul_forward(q, k, v, ctx, scaling)
    diff = (out_e - out_l).abs().max().item()
    assert diff < 1e-2, diff
    ok(f"lift_svd vs eager max_diff={diff:.6f}")


def test_lift_decomposed_bias_approx() -> None:
    """Decomposed VV is a bounded bilinear approx — log error only (not a gate)."""
    os.environ["GEOPACT_PARB"] = "1"
    os.environ["GEOPACT_PARB_VV"] = "1"
    os.environ["GEOPACT_PARB_TV"] = "1"
    os.environ["GEOPACT_PARB_LIFT_MODE"] = "decomposed"
    ctx = _fake_ctx()
    oracle = build_parb_bias(ctx, device=torch.device("cpu"), dtype=torch.float32)[0, 0]
    feat = build_parb_lift_features(ctx, device=torch.device("cpu"))
    assert feat is not None
    approx = lift_bias_matrix(feat)
    err = (oracle - approx).abs().max().item()
    ok(f"decomposed bias approx max_err={err:.4f} (informational)")


def test_hybrid_unchanged() -> None:
    os.environ["GEOPACT_PARB_MODE"] = "hybrid"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_e = parb_eager_forward(mod, q, k, v, ctx, scaling=d**-0.5)
    out_h = parb_hybrid_forward(mod, q, k, v, ctx, scaling=d**-0.5)
    diff = (out_e - out_h).abs().max().item()
    assert diff < 1e-3, diff
    ok(f"hybrid oracle max_diff={diff:.6f}")


def main() -> int:
    test_svd_bias_fit()
    test_hybrid_unchanged()
    test_lift_decomposed_bias_approx()
    test_lift_tv_exact()
    test_lift_svd_vs_eager()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
