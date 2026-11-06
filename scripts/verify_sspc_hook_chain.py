#!/usr/bin/env python3
"""SSPC hook-chain smoke: compact stats fields + ZAFD partition invariants."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_sspc import (
    apply_sspc_embedding_zafd_for_model,
    clear_sspc_attach_ctx,
    record_keep_b,
    record_keep_final,
    reset_sspc_step_state,
    sspc_stats_for_compact,
)
from lmms_eval.models.geopact_sspc_prune import apply_sspc_extra_keep, plan_sspc_keep_budget


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def test_prune_to_zafd_chain() -> None:
    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_SSPC_SKELETON"] = "1"
    os.environ["GEOPACT_SSPC_ZAFD"] = "1"
    os.environ["GEOPACT_SSPC_K_GRID"] = "5"
    os.environ["GEOPACT_SSPC_AXIS"] = "linspace"
    os.environ["GEOPACT_SSPC_BUDGET"] = "in_budget"

    model = SimpleNamespace()
    reset_sspc_step_state(model)

    keep_k_total = 100
    reserved, keep_k_b, pre_audit = plan_sspc_keep_budget(
        keep_k_total, grid_h=24, grid_w=24, n_frame=576,
    )
    cands = [i for i in range(576) if i not in reserved]
    keep_b_local = set(cands[:keep_k_b])
    keep_local = keep_b_local | reserved
    _, audit = apply_sspc_extra_keep(
        keep_local,
        grid_h=24,
        grid_w=24,
        n_frame=576,
        keep_k_total=keep_k_total,
    )
    frame_start = 0
    record_keep_b(model, {frame_start + i for i in keep_b_local})
    record_keep_final(model, {frame_start + i for i in keep_local})
    model._geopact_sspc_audit = {**pre_audit, **audit}

    assert pre_audit["sspc_extra_mode"] == "skeleton"
    assert pre_audit["sspc_reserved_n"] == 25
    assert len(keep_local) == keep_k_total
    assert audit.get("budget_overage") == 0

    clear_sspc_attach_ctx(model)
    emb = torch.randn(576, 64)
    grid = torch.tensor([[1, 24, 24]])
    prune_ctx = {"frame_start": frame_start, "n_frame": 576}
    out = apply_sspc_embedding_zafd_for_model(emb, model, grid, prune_ctx)
    zafd_audit = getattr(model, "_geopact_sspc_audit", {})
    assert zafd_audit.get("sspc_zafd") is True
    assert zafd_audit.get("sspc_zafd_n_pruned", 0) > 0

    stats = sspc_stats_for_compact(model)
    assert stats.get("sspc_zafd") is True
    assert stats.get("sspc_reserved_n") == 25
    assert (out[min(keep_b_local)] - emb[min(keep_b_local)]).abs().max().item() > 0
    ok("In-budget prune -> attach clear -> ZAFD chain with compact stats")


def test_origpos_grid_resolution() -> None:
    from lmms_eval.models.geopact_sspc import resolve_frame_grid

    gh, gw, matched = resolve_frame_grid(12, 12, 576)
    assert not matched or gh * gw == 576
    assert gh * gw >= 576
    ok("resolve_frame_grid covers 576 tokens")


def test_anchor_excluded_from_hes_select() -> None:
    from lmms_eval.models.geopact_vl_hook import _hes_select

    os.environ["GEOPACT_SSPC"] = "1"
    os.environ["GEOPACT_CDCT_ALPHA"] = "0.2"
    torch.manual_seed(1)
    n_img = 32
    embeds = torch.randn(n_img, 16)
    inputs = torch.zeros(1, 8, 16)
    mask = torch.zeros(8, dtype=torch.bool)
    grid = torch.tensor([[1, 8, 8]])
    mius = [
        {
            "box": (0.1, 0.1, 0.4, 0.4),
            "info_density": 1.0,
            "interaction_energy": 1.0,
            "est_token_count": 4,
        }
    ]
    reserved = {0}
    candidate_local = set(range(n_img)) - reserved
    result = _hes_select(
        mius=mius,
        image_embeds=embeds,
        inputs_embeds=inputs,
        image_mask=mask,
        image_grid_thw=grid,
        keep_k_tokens=5,
        policy="oas_ickh_cdct_no_coord_yolo_v2_drop_pent_scale",
        candidate_local=candidate_local,
    )
    picked = set(int(i) for i in result)
    assert 0 not in picked
    assert picked <= candidate_local
    ok("CDCT _hes_select with candidate_local excludes reserved anchor index 0")


def test_module_a_env_contract() -> None:
    from lmms_eval.models.geopact_module_a_worker import worker_yolo_device

    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "cpu"
    os.environ["GEOPACT_MODULE_A_SUBPROCESS"] = "1"
    os.environ["GEOPACT_MODULE_A_WORKER_YOLO_DEVICE"] = "0"
    os.environ["MIU_YOLO_DEVICE"] = "cpu"
    assert worker_yolo_device() == "0"
    assert os.environ.get("GEOPACT_MODULE_A_SUBPROCESS") == "1"
    assert os.environ.get("MIU_YOLO_DEVICE") == "cpu"
    ok("Module A env contract: subprocess YOLO on visible GPU 0, main on cpu")


def test_zafd_skeleton_excluded() -> None:
    from lmms_eval.models.geopact_sspc import apply_embedding_zafd

    emb = torch.randn(16, 8)
    keep_b = {2, 3, 4}
    keep_final = keep_b | {0}
    out, audit = apply_embedding_zafd(
        emb,
        keep_b=keep_b,
        keep_final=keep_final,
        frame_start=0,
        n_frame=16,
        grid_h=4,
        grid_w=4,
    )
    assert audit.get("sspc_zafd_n_skeleton_only") == 1
    assert torch.allclose(out[0], emb[0])
    ok("ZAFD: skeleton-only token 0 unchanged (excluded from donor/receiver)")


def main() -> None:
    test_origpos_grid_resolution()
    test_prune_to_zafd_chain()
    test_anchor_excluded_from_hes_select()
    test_module_a_env_contract()
    test_zafd_skeleton_excluded()
    print("verify_sspc_hook_chain: ALL OK")


if __name__ == "__main__":
    main()
