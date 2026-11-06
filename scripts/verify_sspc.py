#!/usr/bin/env python3
"""Verify SSPC: skeleton additive keep, Omega stochasticity, embedding ZAFD, hook order."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_sspc import (
    apply_embedding_zafd,
    apply_sspc_embedding_zafd_for_model,
    clear_sspc_attach_ctx,
    compute_omega_surv_pruned,
    record_keep_b,
    record_keep_final,
    reset_sspc_step_state,
    resolve_frame_grid,
    sspc_enabled,
    _frame_positions_norm,
)
from lmms_eval.models.geopact_sspc_prune import (
    apply_sspc_extra_keep,
    plan_sspc_keep_budget,
    skeleton_anchor_count,
    skeleton_flat_local_indices,
    sspc_in_budget,
)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def test_skeleton_additive() -> None:
    keep_b = {1, 5, 10, 20}
    final, audit = apply_sspc_extra_keep(
        keep_b,
        grid_h=24,
        grid_w=24,
        n_frame=576,
    )
    assert final >= keep_b
    assert audit["sspc_keep_b_n"] == 4
    assert audit["sspc_extra_n"] >= 0
    ok("Skeleton union is additive to Module B keep")


def test_skeleton_k25_k49_count() -> None:
    os.environ["GEOPACT_SSPC_K_GRID"] = "5"
    os.environ["GEOPACT_SSPC_AXIS"] = "linspace"
    sk25 = skeleton_flat_local_indices(12, 12, 576, k_grid=5, axis_mode="linspace")
    assert len(sk25) == 25
    os.environ["GEOPACT_SSPC_K_GRID"] = "7"
    sk49 = skeleton_flat_local_indices(12, 12, 576, k_grid=7, axis_mode="linspace")
    assert len(sk49) == 49
    ok("Dense skeleton k25/k49 anchor counts")


def test_full_frame_grid_576() -> None:
    gh, gw, matched = resolve_frame_grid(12, 12, 576)
    assert gh * gw >= 576
    sk = skeleton_flat_local_indices(12, 12, 576, k_grid=5, axis_mode="linspace")
    assert min(sk) >= 0
    assert max(sk) < 576
    assert max(sk) - min(sk) > 400, "skeleton should span most of the frame"
    ok("Full-frame 576 skeleton spread covers frame extent")


def test_positions_unit_square() -> None:
    pos = _frame_positions_norm(576, 12, 12, device=torch.device("cpu"))
    assert pos.min().item() >= 0.0
    assert pos.max().item() <= 1.0
    ok("Normalized positions in [0,1]")


def test_omega_survivor_norm() -> None:
    os.environ["GEOPACT_ZAFD_OMEGA_NORM"] = "survivor"
    pos_s = torch.rand(8, 2)
    pos_p = torch.rand(12, 2)
    omega = compute_omega_surv_pruned(pos_s, pos_p, sigma=0.4, norm="survivor")
    assert torch.all(omega >= 0)
    assert torch.allclose(omega.sum(-1), torch.ones(8), atol=1e-4)
    ok("Omega survivor-norm row-stochastic (v0)")


def test_omega_donor_norm() -> None:
    os.environ["GEOPACT_ZAFD_OMEGA_NORM"] = "donor"
    pos_s = torch.rand(8, 2)
    pos_p = torch.rand(12, 2)
    omega = compute_omega_surv_pruned(pos_s, pos_p, sigma=0.4, norm="donor")
    assert torch.all(omega >= 0)
    assert torch.allclose(omega.sum(0), torch.ones(12), atol=1e-4)
    ok("Omega donor-norm column-stochastic (v2 IPCC)")


def test_donor_mass_conservation() -> None:
    pos_s = torch.rand(6, 2)
    pos_p = torch.rand(10, 2)
    omega = compute_omega_surv_pruned(pos_s, pos_p, sigma=0.4, norm="donor")
    assert torch.allclose(omega.sum(0), torch.ones(10), atol=1e-4)
    ok("Donor mass conserved: sum_i Omega[i,j] == 1")


def test_total_delta_equals_sum_donors() -> None:
    n_frame = 16
    gh, gw = 4, 4
    d = 8
    emb = torch.randn(n_frame, d)
    keep_b = {0, 1, 2, 3, 8}
    os.environ["GEOPACT_ZAFD_OMEGA_NORM"] = "donor"
    os.environ["GEOPACT_ZAFD_GAMMA"] = "0"
    out, audit = apply_embedding_zafd(
        emb,
        keep_b=keep_b,
        frame_start=0,
        n_frame=n_frame,
        grid_h=gh,
        grid_w=gw,
        sigma=0.4,
    )
    assert audit.get("sspc_zafd_omega_norm") == "donor"
    donors = sorted(set(range(n_frame)) - keep_b)
    delta_sum = out[list(keep_b)].to(torch.float32).sum(0) - emb[list(keep_b)].to(torch.float32).sum(0)
    donor_sum = emb[donors].to(torch.float32).sum(0)
    assert torch.allclose(delta_sum, donor_sum, atol=1e-3)
    ok("Donor-norm: sum of receiver deltas equals sum of donor embeddings")


def test_hybrid_gamma_zero_matches_spatial() -> None:
    pos_s = torch.rand(5, 2)
    pos_p = torch.rand(7, 2)
    x_s = torch.randn(5, 16)
    x_d = torch.randn(7, 16)
    omega_spatial = compute_omega_surv_pruned(
        pos_s, pos_p, sigma=0.4, norm="donor", gamma=0.0,
    )
    omega_hybrid = compute_omega_surv_pruned(
        pos_s, pos_p, sigma=0.4, norm="donor", gamma=0.0,
        x_surv=x_s, x_donor=x_d,
    )
    assert torch.allclose(omega_spatial, omega_hybrid, atol=1e-5)
    ok("gamma=0 hybrid kernel matches pure spatial donor-norm")


def test_donor_mass_audit_cuda() -> None:
    if not torch.cuda.is_available():
        ok("donor_mass audit on CUDA (skipped: no GPU)")
        return
    os.environ["GEOPACT_ZAFD_OMEGA_NORM"] = "donor"
    os.environ["GEOPACT_ZAFD_AUDIT"] = "1"
    emb = torch.randn(16, 32, device="cuda")
    keep_b = {0, 1, 2, 3, 8}
    _, audit = apply_embedding_zafd(
        emb,
        keep_b=keep_b,
        frame_start=0,
        n_frame=16,
        grid_h=4,
        grid_w=4,
        sigma=0.4,
    )
    assert audit.get("sspc_zafd_donor_mass_conserved") is True
    ok("Donor mass audit passes on CUDA (no cpu/cuda allclose crash)")


def test_omega_row_stochastic() -> None:
    """Legacy alias: survivor-norm row stochasticity."""
    test_omega_survivor_norm()


def test_zafd_shape_and_delta() -> None:
    n_frame = 16
    gh, gw = 4, 4
    d = 32
    emb = torch.randn(n_frame, d)
    keep_b = {0, 1, 2, 3, 8}
    os.environ["GEOPACT_ZAFD_OMEGA_NORM"] = "survivor"
    out, audit = apply_embedding_zafd(
        emb,
        keep_b=keep_b,
        frame_start=0,
        n_frame=n_frame,
        grid_h=gh,
        grid_w=gw,
        sigma=0.4,
    )
    assert out.shape == emb.shape
    assert audit.get("sspc_zafd") is True
    changed = (out[list(keep_b)] - emb[list(keep_b)]).abs().max().item()
    assert changed > 0
    pruned = set(range(n_frame)) - keep_b
    unchanged = (out[sorted(pruned)] - emb[sorted(pruned)]).abs().max().item()
    assert unchanged == 0
    ok("ZAFD modifies B-survivors only; donor rows unchanged pre-gather")


def test_zafd_donor_excludes_skeleton() -> None:
    n_frame = 25
    gh, gw = 5, 5
    emb = torch.randn(n_frame, 32)
    keep_b = {0, 1, 2, 3, 4}
    skeleton_only = {20, 21, 22}
    keep_final = keep_b | skeleton_only
    out, audit = apply_embedding_zafd(
        emb,
        keep_b=keep_b,
        keep_final=keep_final,
        frame_start=0,
        n_frame=n_frame,
        grid_h=gh,
        grid_w=gw,
    )
    assert audit["sspc_zafd"] is True
    assert audit["sspc_zafd_n_skeleton_only"] == len(skeleton_only)
    donor_n = audit["sspc_zafd_n_donor"]
    assert donor_n == n_frame - len(keep_final)
    skel_unchanged = (out[sorted(skeleton_only)] - emb[sorted(skeleton_only)]).abs().max().item()
    assert skel_unchanged == 0
    ok("ZAFD donor excludes skeleton-only tokens")


def test_hook_order_keep_b_survives_clear() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_SSPC_ZAFD"] = "1"
    model = SimpleNamespace()
    reset_sspc_step_state(model)
    record_keep_b(model, {0, 1, 2, 3, 8})
    record_keep_final(model, {0, 1, 2, 3, 8, 20})
    clear_sspc_attach_ctx(model)
    assert getattr(model, "_geopact_sspc_keep_b", set()), "attach clear must not wipe keep_b"
    emb = torch.randn(25, 32)
    prune_ctx = {"frame_start": 0, "n_frame": 25}
    grid = torch.tensor([[1, 5, 5]])
    out = apply_sspc_embedding_zafd_for_model(emb, model, grid, prune_ctx)
    audit = getattr(model, "_geopact_sspc_audit", {})
    assert audit.get("sspc_zafd") is True
    assert (out[[0, 1]] - emb[[0, 1]]).abs().max().item() > 0
    ok("Hook order: keep_b survives attach clear; ZAFD applies delta")


def test_env() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    assert sspc_enabled()
    ok("GEOPACT_SSPC env parses")


def test_in_budget_keep_count() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_SSPC_SKELETON"] = "1"
    os.environ["GEOPACT_SSPC_ZAFD"] = "1"
    os.environ["GEOPACT_SSPC_BUDGET"] = "in_budget"
    os.environ["GEOPACT_SSPC_K_GRID"] = "5"
    keep_k_total = 100
    reserved, keep_k_b, audit = plan_sspc_keep_budget(
        keep_k_total, grid_h=24, grid_w=24, n_frame=576,
    )
    assert audit["sspc_budget_mode"] == "in_budget"
    assert len(reserved) == 25
    assert keep_k_b == keep_k_total - 25
    cands = [i for i in range(576) if i not in reserved]
    keep_b = set(cands[:keep_k_b])
    final, fin_audit = apply_sspc_extra_keep(
        keep_b | reserved,
        grid_h=24,
        grid_w=24,
        n_frame=576,
        keep_k_total=keep_k_total,
    )
    assert len(final) == keep_k_total
    assert fin_audit.get("budget_overage") == 0
    ok("In-budget: keep_final == keep_k_total, budget_overage=0")


def test_anchor_count_g1_g9() -> None:
    for g in range(1, 10):
        assert skeleton_anchor_count(g) == g * g
    ok("Anchor counts g^2 for g=1..9")


def test_keep_k_b_zero_budget() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_SSPC_SKELETON"] = "1"
    os.environ["GEOPACT_SSPC_BUDGET"] = "in_budget"
    os.environ["GEOPACT_SSPC_K_GRID"] = "3"
    keep_k_total = 9
    reserved, keep_k_b, audit = plan_sspc_keep_budget(
        keep_k_total, grid_h=12, grid_w=12, n_frame=144,
    )
    assert keep_k_b == 0
    assert len(reserved) == keep_k_total
    assert audit.get("sspc_reserved_truncated") is True
    ok("keep_k_b=0 when skeleton fills budget; reserved_truncated audit set")


def test_in_budget_partition_disjoint() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_SSPC_SKELETON"] = "1"
    os.environ["GEOPACT_SSPC_BUDGET"] = "in_budget"
    os.environ["GEOPACT_SSPC_K_GRID"] = "3"
    keep_k_total = 50
    reserved, keep_k_b, _ = plan_sspc_keep_budget(
        keep_k_total, grid_h=12, grid_w=12, n_frame=144,
    )
    cands = [i for i in range(144) if i not in reserved]
    keep_b = set(cands[:keep_k_b])
    keep_final = keep_b | reserved
    assert keep_b.isdisjoint(reserved)
    assert len(keep_final) == keep_k_total
    ok("keep_b ∩ reserved = ∅ and |keep_final| == keep_k")


def test_prior_subset_single_index() -> None:
    from lmms_eval.models.geopact_vl_hook import _hes_token_kernel_herding_subset

    torch.manual_seed(0)
    n_img = 64
    embeds = torch.randn(n_img, 16)
    prior = torch.rand(n_img)
    cand = [2, 5, 9, 11, 20, 33, 40, 55]
    inputs = torch.zeros(1, 4, 16)
    mask = torch.zeros(4, dtype=torch.bool)
    ref = prior[torch.tensor(cand, dtype=torch.long)]
    picked = _hes_token_kernel_herding_subset(
        embeds,
        cand,
        inputs,
        mask,
        keep_k_tokens=3,
        policy="token_ickh_relevance",
        spatial_prior=prior,
        cdct_alpha=0.2,
    )
    assert len(picked) == 3
    assert all(p in cand for p in picked)
    assert ref.shape[0] == len(cand)
    ok("Subset herding prior index_select: picks ⊆ cand, no OOB")


def test_worker_yolo_default_same_gpu() -> None:
    from lmms_eval.models.geopact_module_a_worker import worker_yolo_device

    old = os.environ.pop("GEOPACT_MODULE_A_WORKER_YOLO_DEVICE", None)
    try:
        assert worker_yolo_device() == "0"
    finally:
        if old is not None:
            os.environ["GEOPACT_MODULE_A_WORKER_YOLO_DEVICE"] = old
    ok("worker_yolo_device defaults to same visible GPU (0)")


def test_flash_attn_import() -> None:
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise AssertionError(
            "flash_attn required for production eval (attn_implementation=flash_attention_2)"
        ) from exc
    ok("flash_attn importable")


def main() -> None:
    os.environ.setdefault("GEOPACT_SSPC", "1")
    os.environ.setdefault("GEOPACT_SSPC_SKELETON", "1")
    test_skeleton_additive()
    test_skeleton_k25_k49_count()
    test_full_frame_grid_576()
    test_positions_unit_square()
    test_omega_survivor_norm()
    test_omega_donor_norm()
    test_donor_mass_conservation()
    test_total_delta_equals_sum_donors()
    test_hybrid_gamma_zero_matches_spatial()
    test_donor_mass_audit_cuda()
    test_zafd_shape_and_delta()
    test_zafd_donor_excludes_skeleton()
    test_hook_order_keep_b_survives_clear()
    test_in_budget_keep_count()
    test_anchor_count_g1_g9()
    test_keep_k_b_zero_budget()
    test_in_budget_partition_disjoint()
    test_prior_subset_single_index()
    test_worker_yolo_default_same_gpu()
    test_flash_attn_import()
    test_env()
    print("verify_sspc: ALL OK")


if __name__ == "__main__":
    main()
