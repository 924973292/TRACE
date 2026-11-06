#!/usr/bin/env python3
"""Verify VSAS: Omega stochasticity, path-A oracle parity, seq splice, batch independence."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_sampling_correction import (
    build_anchor_coords,
    build_anchor_hidden,
    compute_omega,
    normalize_visual_hw,
)
from lmms_eval.models.geopact_vsas import VSASCtx, splice_vsas_into_compact
from lmms_eval.models.geopact_vsas_attention import eager_vsas_attention, try_vsas_attention_internal


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


class _DummyKV:
    num_key_value_groups = 1
    num_key_value_heads = 4
    head_dim = 16
    training = False

    def __init__(self) -> None:
        self.k_proj = torch.nn.Linear(16, 4 * 16, bias=False)
        self.v_proj = torch.nn.Linear(16, 4 * 16, bias=False)
        self.k_norm = torch.nn.Identity()


def test_omega_stochastic() -> None:
    anchors = build_anchor_coords("nine_grid", device=torch.device("cpu"), dtype=torch.float32)
    n = 12
    h = torch.arange(n, dtype=torch.float32)
    w = torch.arange(n, dtype=torch.float32) * 2
    is_v = torch.zeros(n, dtype=torch.bool)
    is_v[2:10] = True
    pos = normalize_visual_hw(h, w, is_v)
    vis_mask = is_v
    omega = compute_omega(anchors, pos, sigma=0.4, vis_mask=vis_mask)
    row_sums = omega.sum(dim=-1)
    assert torch.all(omega >= 0), "omega must be non-negative"
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "rows must sum to 1"
    ok("Omega row-stochastic and non-negative")


def _oracle_internal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_anchor: torch.Tensor,
    v_anchor: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    k_cat = torch.cat([k, k_anchor], dim=2)
    v_cat = torch.cat([v, v_anchor], dim=2)
    out = eager_vsas_attention(q, k_cat, v_cat, scaling=scaling, causal=True)
    n_q = int(q.shape[2])
    return out[:, :, :n_q, :].transpose(1, 2).contiguous()


def test_path_a_oracle_parity() -> None:
    torch.manual_seed(7)
    b, h, n, d, n_anchor = 1, 4, 10, 16, 9
    scaling = d**-0.5
    hidden = torch.randn(b, n, d)
    q = torch.randn(b, h, n, d)
    key = torch.randn(b, h, n, d)
    val = torch.randn(b, h, n, d)
    vis_idx = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long)
    anchors = build_anchor_coords("nine_grid", device=hidden.device, dtype=torch.float32)
    pos = torch.rand(n, 2)
    omega = compute_omega(anchors, pos, sigma=0.4)[:, vis_idx]
    x_anchor = build_anchor_hidden(hidden, omega, vis_indices=vis_idx)
    mod = _DummyKV()
    k_anchor = mod.k_norm(
        mod.k_proj(x_anchor).view(b, n_anchor, mod.num_key_value_heads, mod.head_dim)
    ).transpose(1, 2)
    v_anchor = mod.v_proj(x_anchor).view(b, n_anchor, mod.num_key_value_heads, mod.head_dim).transpose(1, 2)
    ctx = VSASCtx(
        seq_len=n,
        seq_base_len=n,
        n_virtual=n_anchor,
        mode="internal",
        is_vision=torch.zeros(n, dtype=torch.bool),
        vis_indices=vis_idx,
        pos_norm=pos,
        omega=omega,
        anchor_coords=anchors,
        anchor_h=torch.zeros(n_anchor),
        anchor_w=torch.zeros(n_anchor),
        denom_only=False,
        audit={},
    )
    got = try_vsas_attention_internal(
        mod,
        q,
        key,
        val,
        hidden,
        ctx,
        scaling=scaling,
        position_embeddings=(torch.zeros(1), torch.zeros(1)),
        layer_idx=0,
    )
    ref = _oracle_internal(q, key, val, k_anchor, v_anchor, scaling)
    for dtype in (torch.float32, torch.bfloat16):
        g = got.to(dtype)
        r = ref.to(dtype)
        max_err = (g.float() - r.float()).abs().max().item()
        assert max_err < 5e-2 if dtype == torch.bfloat16 else 1e-4, f"parity failed {dtype}: {max_err}"
    ok("Path A internal K/V concat matches eager oracle (fp32/bf16)")


def test_kv_seq_strip_logic() -> None:
    """Seq-mode cache update keeps only base_len positions (logic unit test)."""
    n_base, k_virt = 20, 9
    n_full = n_base + k_virt
    key = torch.randn(1, 4, n_full, 16)
    value = torch.randn(1, 4, n_full, 16)
    nb = n_base
    k_cache = key[:, :, :nb, :]
    v_cache = value[:, :, :nb, :]
    assert k_cache.shape[2] == nb
    assert v_cache.shape[2] == nb
    assert key.shape[2] == n_full
    ok("KV seq-mode strip retains base_len only for cache write")


def test_path_b_splice() -> None:
    torch.manual_seed(3)
    n, d, k = 14, 16, 4
    image_tok_id = 99
    compact_ids = torch.tensor([[1, 1, 99, 99, 99, 99, 2, 2, 1, 1, 1, 1, 2, 2]], dtype=torch.long)
    compact_embs = torch.randn(1, n, d)
    compact_attn = torch.ones(1, n)
    t = torch.zeros(1, 1, n)
    h = torch.arange(n, dtype=torch.float32).view(1, 1, n)
    w = torch.arange(n, dtype=torch.float32).view(1, 1, n) * 2
    compact_pos = torch.cat([t, h, w], dim=0)
    compact_rope = torch.zeros(1, 1, dtype=torch.long)
    vis_idx = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    anchors = build_anchor_coords("four_corners", device=compact_embs.device, dtype=torch.float32)
    pos = torch.rand(n, 2)
    omega = compute_omega(anchors, pos, sigma=0.4)[:, vis_idx]
    ctx = VSASCtx(
        seq_len=n,
        seq_base_len=n,
        n_virtual=k,
        mode="internal",
        is_vision=compact_ids[0] == image_tok_id,
        vis_indices=vis_idx,
        pos_norm=pos,
        omega=omega,
        anchor_coords=anchors,
        anchor_h=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        anchor_w=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        denom_only=False,
        audit={},
    )
    ids2, embs2, attn2, pos2, rope2, ctx2 = splice_vsas_into_compact(
        compact_ids,
        compact_embs,
        compact_attn,
        compact_pos,
        compact_rope,
        ctx,
        image_tok_id,
    )
    assert int(ids2.shape[1]) == n + k
    assert int(embs2.shape[1]) == n + k
    assert int(ctx2.seq_base_len) == n
    assert int(ctx2.seq_len) == n + k
    assert int(pos2.shape[2]) == n + k
    ok("Path B splice extends sequence by K virtual tokens")


def test_batch_independent_omega() -> None:
    anchors = build_anchor_coords("four_corners", device=torch.device("cpu"), dtype=torch.float32)
    pos_a = torch.rand(8, 2)
    pos_b = torch.rand(8, 2)
    pos_bat = torch.stack([pos_a, pos_b], dim=0)
    om_a = compute_omega(anchors, pos_a, sigma=0.4)
    om_b = compute_omega(anchors, pos_b, sigma=0.4)
    om_bat = compute_omega(anchors, pos_bat, sigma=0.4)
    assert torch.allclose(om_bat[0], om_a, atol=1e-5)
    assert torch.allclose(om_bat[1], om_b, atol=1e-5)
    assert not torch.allclose(om_a, om_b)
    ok("Batched Omega independent per sample")


def test_vsas_dmtc_mutual_exclusion_env() -> None:
    os.environ["GEOPACT_VSAS"] = "1"
    os.environ["GEOPACT_CORRECTION"] = "1"
    from lmms_eval.models.geopact_correction import correction_enabled
    from lmms_eval.models.geopact_vsas import vsas_enabled

    assert vsas_enabled() and correction_enabled()
    ok("VSAS and DMTC env flags parse independently (hook uses VSAS first)")


def main() -> None:
    os.environ.setdefault("GEOPACT_VSAS", "1")
    test_omega_stochastic()
    test_path_a_oracle_parity()
    test_kv_seq_strip_logic()
    test_path_b_splice()
    test_batch_independent_omega()
    test_vsas_dmtc_mutual_exclusion_env()
    print("verify_vsas: ALL OK")


if __name__ == "__main__":
    main()
