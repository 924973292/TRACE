#!/usr/bin/env python3
"""Offline checks: incremental KV flow vs ABC prune-rewrite decoupling."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _ok(name: str) -> None:
    print(f"[OK] {name}")


def _fail(name: str, detail: str = "") -> None:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""), file=sys.stderr)


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            _ok(name)
        else:
            _fail(name, detail)
            failures.append(name)

    from lmms_eval.models.geopact_decode_contract import (
        promote_pinned_rope_fields,
        should_use_pinned_generate,
    )
    from lmms_eval.models.geopact_session_kv import (
        build_dialogue_generate_payload,
        kv_flow_mode_label,
        kv_history_prune_rewrite_enabled,
        kv_incremental_flow_enabled,
        _should_ensure_virtual_mrope_cache,
        _visual_prune_active,
    )

    # Arm2-style: dialogue KV on, no prune ratio
    arm2_inner = SimpleNamespace(
        _geopact_prune_ratio=None,
        _geopact_prune_policy="",
    )
    arm2_model = SimpleNamespace(model=arm2_inner)

    import os

    os.environ["GEOPACT_SESSION_KV"] = "1"
    os.environ["GEOPACT_SESSION_KV_MODE"] = "dialogue"
    os.environ.pop("GEOPACT_QWEN_DIALOGUE_CACHE", None)
    os.environ["GEOPACT_KV_HISTORY_REWRITE"] = "1"
    os.environ["GEOPACT_KV_HISTORY_COMPRESS_LEGACY"] = "0"
    os.environ["GEOPACT_PRUNE_TARGET"] = "history"

    check("arm2 incremental flow enabled", kv_incremental_flow_enabled())
    check("arm2 prune rewrite disabled", not kv_history_prune_rewrite_enabled(arm2_model))
    check("arm2 flow mode incremental", kv_flow_mode_label(arm2_model) == "incremental")
    check(
        "arm2 skips virtual mrope cache",
        not _should_ensure_virtual_mrope_cache(arm2_model, {}),
    )

    # ABC-style: prune ratio + policy
    abc_inner = SimpleNamespace(
        _geopact_prune_ratio=0.4,
        _geopact_prune_policy="oas_ickh_cdct_no_coord_yolo_v2_drop_pent_scale",
    )
    abc_model = SimpleNamespace(model=abc_inner)
    check("abc visual prune active", _visual_prune_active(abc_model))
    check("abc prune rewrite enabled", kv_history_prune_rewrite_enabled(abc_model))
    check("abc flow mode prune_rewrite", kv_flow_mode_label(abc_model) == "prune_rewrite")
    check(
        "abc needs virtual mrope when rewrite on",
        _should_ensure_virtual_mrope_cache(abc_model, {}),
    )
    check(
        "pending sliding rewrite always needs virtual mrope",
        _should_ensure_virtual_mrope_cache(arm2_model, {"_sliding_rewrite_pending": {}}),
    )

    import torch

    arm2_inner.rope_deltas = torch.zeros(1, 1, dtype=torch.long)
    arm2_suffix_payload = {
        "position_ids": torch.zeros(3, 1, 1),
        "_geopact_pinned_cache_start": 2213,
        "past_key_values": type("PKV", (), {"get_seq_length": lambda self: 2212})(),
        "_geopact_decode_only_generate": True,
    }
    check(
        "arm2 suffix not pinned before rope promote",
        not should_use_pinned_generate(arm2_suffix_payload),
    )
    built = build_dialogue_generate_payload(
        arm2_model,
        arm2_suffix_payload,
        logical_prompt_len=2213,
        dialogue_suffix_only=True,
    )
    check(
        "build uses pinned decode for incremental suffix",
        built.get("position_ids") is not None
        and built.get("_geopact_pinned_cache_start") is not None
        and built.get("_geopact_rope_deltas") is not None,
    )
    check(
        "build strips hf-forbidden bookkeeping",
        "rope_deltas" not in built and "_geopact_decode_only_generate" not in built,
    )

    # Simulate inner.rope_deltas set during suffix prefill (not in raw payload).
    class _RopeInner(SimpleNamespace):
        pass

    rope_inner = _RopeInner(
        _geopact_prune_ratio=None,
        _geopact_prune_policy="",
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
    )
    rope_model = SimpleNamespace(model=rope_inner)
    built_rope = build_dialogue_generate_payload(
        rope_model,
        arm2_suffix_payload,
        logical_prompt_len=2213,
        dialogue_suffix_only=True,
    )
    check(
        "build promotes inner.rope_deltas for incremental pinned decode",
        built_rope.get("_geopact_rope_deltas") is not None,
    )

    print()
    if failures:
        print(f"FAILED {len(failures)}: {failures}", file=sys.stderr)
        return 1
    print("ALL KV INCREMENTAL FLOW CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
