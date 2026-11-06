#!/usr/bin/env python3
"""Verify DMTC correction: low-rank TV prior + deleted-mass LPR."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_correction import CorrectionCtx, attach_correction_ctx
from lmms_eval.models.geopact_correction_attention import try_correction_attention


class _DummyAttn:
    num_key_value_groups = 1
    training = False


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _ctx(n: int = 8, *, z: torch.Tensor | None = None, device: torch.device | None = None) -> CorrectionCtx:
    dev = device or torch.device("cpu")
    is_v = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool, device=dev)
    region = torch.zeros(n, dtype=torch.float32, device=dev)
    psi = torch.zeros(n, dtype=torch.float32, device=dev)
    region[is_v] = torch.tensor([0.10, 0.20, -0.05, 0.15], dtype=torch.float32, device=dev)
    psi[is_v] = torch.tensor([-0.1, -0.2, -0.3, -0.4], dtype=torch.float32, device=dev)
    prior = (region + 0.25 * psi) * is_v.float()
    if z is None:
        z = torch.zeros(n, dtype=torch.float32, device=dev)
    return CorrectionCtx(
        seq_len=n,
        is_vision=is_v,
        is_text=~is_v,
        Z_deleted=z.to(dev),
        log_Z_deleted=torch.where(z.to(dev) > 0, torch.log(z.to(dev)), torch.full_like(z.to(dev), float("-inf"))),
        region_prior=region,
        psi=psi,
        prior=prior,
        frame_start=0,
        n_frame=16,
        n_pruned=8,
        audit={"correction": True},
    )


def _oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx: CorrectionCtx, scaling: float) -> torch.Tensor:
    b, h, n, d = q.shape
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.view(1, 1, n, n), float("-inf"))
    text = ctx.is_text.to(q.device)
    vis = ctx.is_vision.to(q.device)
    bias = torch.zeros(n, n, device=q.device, dtype=torch.float32)
    bias[text.unsqueeze(1) & vis.unsqueeze(0)] = ctx.prior.to(q.device)[vis].repeat(int(text.sum()), 1).reshape(-1)
    scores = scores + bias.view(1, 1, n, n).to(scores.dtype)
    scores_f = scores.float()
    lse = torch.logsumexp(scores_f, dim=-1)
    probs = torch.softmax(scores_f, dim=-1).to(q.dtype)
    out = torch.matmul(probs, v)
    z = ctx.Z_deleted.to(q.device, dtype=torch.float32).view(1, 1, n)
    gamma = 1.0 / (1.0 + z * torch.exp(-lse))
    return (out * gamma.unsqueeze(-1).to(q.dtype)).transpose(1, 2).contiguous()


def test_low_rank_tv_parity() -> None:
    os.environ["GEOPACT_CORRECTION"] = "1"
    os.environ["GEOPACT_CORRECTION_LPR"] = "1"
    torch.manual_seed(1)
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    ctx = _ctx(n)
    got = try_correction_attention(_DummyAttn(), q, k, v, ctx, scaling=scaling)
    ref = _oracle(q, k, v, ctx, scaling)
    diff = (got - ref).abs().max().item()
    assert diff < 1e-5, diff
    ok(f"low-rank TV + LPR parity max_diff={diff:.6f}")


def test_deleted_mass_identity() -> None:
    os.environ["GEOPACT_CORRECTION"] = "1"
    os.environ["GEOPACT_CORRECTION_LPR"] = "1"
    torch.manual_seed(2)
    b, h, n, d = 1, 2, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    z = torch.tensor([0.2, 0.5, 0.1, 1.0, 0.3, 0.2, 0.4, 0.7], dtype=torch.float32)
    ctx = _ctx(n, z=z)
    got = try_correction_attention(_DummyAttn(), q, k, v, ctx, scaling=scaling)
    ref = _oracle(q, k, v, ctx, scaling)
    diff = (got - ref).abs().max().item()
    assert diff < 1e-5, diff
    ok(f"deleted-mass null-feature identity max_diff={diff:.6f}")


def test_frame_local_attach() -> None:
    os.environ["GEOPACT_CORRECTION"] = "1"

    class M:
        pass

    model = M()
    compact_ids = torch.tensor([[1, 999, 999, 2]], dtype=torch.long)
    compact_pos = torch.tensor([[[0, 0, 0, 0]], [[0, 2, 5, 0]], [[0, 1, 3, 0]]], dtype=torch.long)
    prune_ctx = {
        "frame_start": 100,
        "n_frame": 8,
        "mius": [{"info_density": 1.0}],
        "miu_token_idxs": [[102, 105]],
        "cdct_density_flat": torch.ones(8),
    }
    ctx = attach_correction_ctx(
        model,
        compact_ids=compact_ids,
        compact_pos=compact_pos,
        image_tok_id=999,
        prune_ctx=prune_ctx,
        visual_keep_flat=torch.tensor([102, 105]),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )
    assert ctx is not None
    assert ctx.n_pruned == 6, ctx.n_pruned
    assert float(ctx.region_prior[1].item()) > 0.0
    assert float(ctx.region_prior[2].item()) > 0.0
    ok("frame-local keep indices and MIU aliases")


def test_cuda_bf16_if_available() -> None:
    if not torch.cuda.is_available():
        ok("bf16 skip (no cuda)")
        return
    os.environ["GEOPACT_CORRECTION"] = "1"
    os.environ["GEOPACT_CORRECTION_LPR"] = "1"
    torch.manual_seed(3)
    dev = torch.device("cuda")
    b, h, n, d = 1, 4, 8, 16
    scaling = d**-0.5
    q = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
    z = torch.full((n,), 0.5, device=dev, dtype=torch.float32)
    ctx = _ctx(n, z=z, device=dev)
    got = try_correction_attention(_DummyAttn(), q, k, v, ctx, scaling=scaling)
    ref = _oracle(q.float(), k.float(), v.float(), ctx, scaling).to(torch.bfloat16)
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff < 0.08, diff
    ok(f"cuda bf16 FA/LSE parity max_diff={diff:.6f}")


def main() -> int:
    test_low_rank_tv_parity()
    test_deleted_mass_identity()
    test_frame_local_attach()
    test_cuda_bf16_if_available()
    print("verify_correction: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
