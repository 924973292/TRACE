#!/usr/bin/env python3
"""Unit checks: PARB eager vs hybrid numerical match; bias shapes."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_parb import PARBCtx, build_parb_bias, parb_enabled
from lmms_eval.models.geopact_parb_attention import (
    online_softmax_combine,
    parb_eager_forward,
    parb_hybrid_forward,
)


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
    region = torch.zeros(n, dtype=torch.float32)
    psi = torch.zeros(n, dtype=torch.float32)
    mius = ({"box": (0.1, 0.1, 0.3, 0.3), "info_density": 1.0}, {"box": (0.5, 0.5, 0.9, 0.9), "info_density": 0.8})
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


def test_bias_shape() -> None:
    os.environ["GEOPACT_PARB"] = "1"
    os.environ["GEOPACT_PARB_VV"] = "1"
    os.environ["GEOPACT_PARB_TV"] = "1"
    assert parb_enabled()
    ctx = _fake_ctx()
    b = build_parb_bias(ctx, device=torch.device("cpu"), dtype=torch.float32)
    assert b.shape == (1, 1, 8, 8), b.shape
    assert float(b[0, 0, 2, 3].item()) != 0.0
    ok("build_parb_bias shape and VV nonzero")


def test_eager_vs_hybrid() -> None:
    os.environ["GEOPACT_PARB_MODE"] = "eager"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_e = parb_eager_forward(mod, q, k, v, ctx, scaling=d**-0.5)
    os.environ["GEOPACT_PARB_MODE"] = "hybrid"
    out_h = parb_hybrid_forward(mod, q, k, v, ctx, scaling=d**-0.5)
    diff = (out_e - out_h).abs().max().item()
    assert diff < 1e-3, diff
    ok(f"eager vs hybrid max_diff={diff:.6f}")


def test_online_combine() -> None:
    out_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out_b = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    lse_a = torch.tensor([[0.0, 0.0]])
    lse_b = torch.tensor([[0.0, 0.0]])
    out = online_softmax_combine(out_a, lse_a, out_b, lse_b)
    assert out.shape == out_a.shape
    ok("online_softmax_combine shape")


def main() -> int:
    test_bias_shape()
    test_online_combine()
    test_eager_vs_hybrid()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
