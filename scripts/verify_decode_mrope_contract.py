#!/usr/bin/env python3
"""Offline pinned MRoPE decode contract checks (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.geopact_decode_contract import (  # noqa: E402
    attach_pinned_decode_contract,
    compute_pinned_cache_start,
    contract_complete,
    promote_pinned_rope_fields,
    should_use_pinned_generate,
    validate_pinned_decode_contract,
)


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

    check(
        "compute_pinned_cache_start step0",
        compute_pinned_cache_start(past_kv_len=0, prompt_token_count=500) == 499,
    )
    check(
        "compute_pinned_cache_start suffix",
        compute_pinned_cache_start(past_kv_len=1800, prompt_token_count=25) == 1824,
    )

    incomplete = {"position_ids": torch.zeros(3, 1, 8)}
    check("incomplete contract rejected", not contract_complete(incomplete))

    step0 = attach_pinned_decode_contract(
        {
            "position_ids": torch.arange(0, 24, dtype=torch.long).view(3, 1, 8),
            "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        },
        logical_prompt_len=8,
    )
    check(
        "step0 attach completes contract",
        contract_complete(step0) and int(step0["_geopact_pinned_cache_start"]) == 7,
    )
    check("should_use_pinned_generate", should_use_pinned_generate(step0))

    arm2_suffix = {
        "position_ids": torch.zeros(3, 1, 1),
        "_geopact_pinned_cache_start": 2213,
        "past_key_values": object(),
    }
    check(
        "incremental suffix without rope not pinned",
        not should_use_pinned_generate(arm2_suffix),
    )
    arm2_with_rope = promote_pinned_rope_fields(
        arm2_suffix,
        model_rope=torch.zeros(1, 1, dtype=torch.long),
        flow_mode="incremental",
    )
    check(
        "incremental suffix pinned after model rope promote",
        should_use_pinned_generate(arm2_with_rope),
    )
    none_flow = promote_pinned_rope_fields(
        {"position_ids": torch.zeros(3, 1, 8)},
        model_rope=torch.zeros(1, 1, dtype=torch.long),
        flow_mode="none",
        past_kv_len=1,
    )
    check(
        "none flow with past_kv skips stale inner rope",
        not should_use_pinned_generate(none_flow),
    )
    none_step0 = promote_pinned_rope_fields(
        {"position_ids": torch.zeros(3, 1, 8)},
        model_rope=torch.zeros(1, 1, dtype=torch.long),
        flow_mode="none",
        past_kv_len=0,
    )
    check(
        "none flow step0 may use fresh inner rope",
        should_use_pinned_generate(none_step0),
    )
    pruned_suffix = dict(arm2_suffix)
    pruned_suffix["rope_deltas"] = torch.zeros(1, 1, dtype=torch.long)
    check("pruned suffix with rope is pinned", should_use_pinned_generate(pruned_suffix))

    incomplete_pinned = {
        "position_ids": torch.zeros(3, 1, 8),
        "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
    }
    try:
        validate_pinned_decode_contract(incomplete_pinned)
        check("validate raises on incomplete pinned", False)
    except RuntimeError:
        check("validate raises on incomplete pinned", True)

    anchor = step0["position_ids"][:, :, -1:]
    for offset in (1, 2, 3):
        pos = anchor + offset
        check(
            f"anchor+offset={offset}",
            torch.equal(pos[:, :, 0], step0["position_ids"][:, :, -1] + offset),
        )

    try:
        validate_pinned_decode_contract(incomplete)
        check("incomplete without rope skipped by validate", True)
    except RuntimeError:
        check("incomplete without rope skipped by validate", False)

    print()
    if failures:
        print(f"FAILED {len(failures)}: {failures}", file=sys.stderr)
        return 1
    print("ALL DECODE MROPE CONTRACT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
