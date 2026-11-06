#!/usr/bin/env python3
"""Verify GSC v2: ICC + single FA + LPR (production path)."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_gsc import GSCCtx, build_gsc_spf_bias, gsc_spf_mode, gsc_vfs_enabled
from lmms_eval.models.geopact_gsc_attention import try_gsc_fa_attention
from lmms_eval.models.geopact_gsc_fa import apply_lpr_rescale, eager_attention_with_lse


class _DummyAttn:
    num_key_value_groups = 1
    training = False


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fake_ctx(n: int = 8, *, vfs_z: bool = True) -> GSCCtx:
    is_v = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    log_z = torch.tensor([2.0, 1.5, 0.8, 1.2, 0.5, 0.9, 1.0, 0.3], dtype=torch.float32)
    if not vfs_z:
        log_z = torch.zeros_like(log_z)
    phi_c = torch.tensor([0, 0, -0.1, -0.2, -0.3, -0.1, 0, 0], dtype=torch.float32)
    phi_s = torch.zeros(n, dtype=torch.float32)
    phi = phi_c * is_v.float()
    z = torch.exp(log_z)
    return GSCCtx(
        seq_len=n,
        is_vision=is_v,
        is_text=~is_v,
        log_Z=log_z,
        Z=z,
        phi_center=phi_c,
        phi_surv=phi_s,
        phi=phi,
        n_frame=16,
        n_pruned=8,
        Z_text_scalar=float(z[(~is_v)][0].item()) if (~is_v).any() else 0.0,
        audit={"gsc": True},
    )


def _icc_null_feature_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: GSCCtx,
    scaling: float,
    *,
    layer_idx: int = 0,
    num_layers: int = 36,
) -> torch.Tensor:
    """Reference: ICC cannibalization + pruned-flux denominator (no additive SPF)."""
    from lmms_eval.models.geopact_gsc import build_b_vision_icc
    from lmms_eval.models.geopact_gsc_attention import _apply_icc_channels

    b, h, n, d = q.shape
    n_rep = int(h // k.shape[1])
    if n_rep > 1:
        k = k[:, :, None, :, :].expand(b, k.shape[1], n_rep, n, d).reshape(b, h, n, d)
        v = v[:, :, None, :, :].expand(b, v.shape[1], n_rep, n, d).reshape(b, h, n, d)
    qc, kc = q.clone(), k.clone()
    if gsc_spf_mode() != "off":
        b_j = build_b_vision_icc(
            ctx,
            device=q.device,
            dtype=q.dtype,
            layer_idx=layer_idx,
            num_layers=num_layers,
            scaling=scaling,
        )
        _apply_icc_channels(qc, kc, ctx.is_vision, b_j)
    scores = torch.matmul(qc, kc.transpose(2, 3)) * scaling
    causal = torch.full((n, n), float("-inf"), device=scores.device, dtype=scores.dtype)
    causal = causal.masked_fill(torch.tril(torch.ones(n, n, device=scores.device, dtype=torch.bool)), 0.0)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    log_z = ctx.log_Z.to(scores.device).view(1, 1, n, 1)
    max_l = torch.maximum(scores.max(dim=-1, keepdim=True).values, log_z)
    exp_s = torch.exp(scores.float() - max_l.float())
    exp_z = torch.exp(log_z.float() - max_l.float())
    numer = torch.matmul(exp_s, v.float())
    denom = exp_s.sum(dim=-1, keepdim=True) + exp_z
    out = (numer / denom.clamp(min=1e-12)).to(dtype=v.dtype)
    return out.transpose(1, 2).contiguous()


def _null_feature_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: GSCCtx,
    scaling: float,
) -> torch.Tensor:
    """Reference: pruned-flux denominator + optional SPF additive bias."""
    b, h, n, d = q.shape
    n_rep = int(h // k.shape[1])
    if n_rep > 1:
        k = k[:, :, None, :, :].expand(b, k.shape[1], n_rep, n, d).reshape(b, h, n, d)
        v = v[:, :, None, :, :].expand(b, v.shape[1], n_rep, n, d).reshape(b, h, n, d)
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    causal = torch.full((n, n), float("-inf"), device=scores.device, dtype=scores.dtype)
    causal = causal.masked_fill(torch.tril(torch.ones(n, n, device=scores.device, dtype=torch.bool)), 0.0)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    if gsc_spf_mode() != "off":
        scores = scores + build_gsc_spf_bias(ctx, device=scores.device, dtype=scores.dtype)
    log_z = ctx.log_Z.to(scores.device).view(1, 1, n, 1)
    max_l = torch.maximum(scores.max(dim=-1, keepdim=True).values, log_z)
    exp_s = torch.exp(scores - max_l)
    exp_z = torch.exp(log_z - max_l)
    numer = torch.matmul(exp_s, v)
    denom = exp_s.sum(dim=-1, keepdim=True) + exp_z
    out = numer / denom.clamp(min=1e-12)
    return out.transpose(1, 2).contiguous()


def test_lpr_vs_null_feature() -> None:
    os.environ["GEOPACT_GSC_MODE"] = "fa"
    os.environ["GEOPACT_GSC_VFS"] = "1"
    os.environ["GEOPACT_GSC_SPF"] = "off"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_fa = try_gsc_fa_attention(mod, q, k, v, ctx, scaling=scaling)
    out_ref = _null_feature_forward(q, k, v, ctx, scaling)
    diff = (out_fa - out_ref).abs().max().item()
    assert diff < 1e-2, diff
    ok(f"LPR+FA vs null-feature max_diff={diff:.6f}")


def test_lpr_identity() -> None:
    os.environ["GEOPACT_GSC_VFS"] = "1"
    b, h, n, d = 1, 2, 4, 8
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    out, lse = eager_attention_with_lse(q, k, v, scaling=scaling)
    z = torch.tensor([0.5, 1.0, 0.2, 0.8], dtype=torch.float32)
    out_lpr = apply_lpr_rescale(out, lse, z, vfs_enabled=True)
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.view(1, 1, n, n), float("-inf"))
    max_l = torch.maximum(scores.max(dim=-1, keepdim=True).values, torch.log(z).view(1, 1, n, 1))
    exp_s = torch.exp(scores - max_l)
    exp_z = torch.exp(torch.log(z).view(1, 1, n, 1) - max_l)
    ref = torch.matmul(exp_s, v) / (exp_s.sum(dim=-1, keepdim=True) + exp_z)
    diff = (out_lpr - ref).abs().max().item()
    assert diff < 1e-5, diff
    ok(f"LPR identity max_diff={diff:.6f}")


def test_fa_full_vs_null() -> None:
    os.environ["GEOPACT_GSC_MODE"] = "fa"
    os.environ["GEOPACT_GSC_VFS"] = "1"
    os.environ["GEOPACT_GSC_SPF"] = "center"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, 1, n, d)
    v = torch.randn(b, 1, n, d)
    mod = _DummyAttn()
    out_fa = try_gsc_fa_attention(mod, q, k, v, ctx, scaling=scaling)
    out_ref = _icc_null_feature_forward(q, k, v, ctx, scaling)
    diff = (out_fa - out_ref).abs().max().item()
    assert diff < 1e-2, diff
    ok(f"ICC+LPR+FA vs ICC-null max_diff={diff:.6f}")


def test_fa_bf16() -> None:
    if not torch.cuda.is_available():
        ok("bf16 skip (no cuda)")
        return
    os.environ["GEOPACT_GSC_MODE"] = "fa"
    os.environ["GEOPACT_GSC_VFS"] = "1"
    os.environ["GEOPACT_GSC_SPF"] = "center"
    ctx = _fake_ctx()
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    dev = torch.device("cuda")
    q = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, 1, n, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, 1, n, d, device=dev, dtype=torch.bfloat16)
    mod = _DummyAttn()
    out_fa = try_gsc_fa_attention(mod, q, k, v, ctx, scaling=scaling)
    out_ref = _icc_null_feature_forward(q, k, v, ctx, scaling)
    diff = (out_fa.float() - out_ref.float()).abs().max().item()
    assert diff < 0.08, diff
    ok(f"ICC+LPR+FA bf16 max_diff={diff:.6f}")


def main() -> int:
    os.environ["GEOPACT_GSC"] = "1"
    test_lpr_identity()
    test_lpr_vs_null_feature()
    test_fa_full_vs_null()
    test_fa_bf16()
    print("verify_gsc: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
