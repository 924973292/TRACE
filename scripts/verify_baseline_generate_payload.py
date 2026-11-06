#!/usr/bin/env python3
"""Offline checks: baseline native HF path keeps pixel_values; embed paths strip them."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.geopact_decode_contract import should_use_pinned_generate
from lmms_eval.models.geopact_session_kv import (
    _uses_embed_prefill,
    build_dialogue_generate_payload,
    kv_flow_mode_label,
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

    check("embed prefill detector", _uses_embed_prefill({"inputs_embeds": torch.zeros(1, 8, 4)}))
    check(
        "native prefill not embed",
        not _uses_embed_prefill({"input_ids": torch.zeros(1, 8), "pixel_values": torch.zeros(1, 3, 4, 4)}),
    )

    baseline_model = SimpleNamespace(
        model=SimpleNamespace(
            _geopact_prune_ratio=None,
            _geopact_prune_policy="",
        )
    )
    import os

    os.environ.pop("GEOPACT_SESSION_KV", None)
    check("baseline flow mode none", kv_flow_mode_label(baseline_model) == "none")

    native_payload = {
        "input_ids": torch.zeros(1, 16, dtype=torch.long),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "pixel_values": torch.zeros(1, 588, 1176),
        "image_grid_thw": torch.tensor([[1, 14, 14]]),
    }
    built_native = build_dialogue_generate_payload(
        baseline_model,
        dict(native_payload),
        logical_prompt_len=16,
    )
    check(
        "native baseline keeps pixel_values",
        built_native.get("pixel_values") is not None,
    )
    check(
        "native baseline keeps input_ids",
        built_native.get("input_ids") is not None,
    )
    check(
        "native baseline not pinned",
        not should_use_pinned_generate(built_native),
    )

    os.environ["GEOPACT_SESSION_KV"] = "1"
    os.environ["GEOPACT_SESSION_KV_MODE"] = "dialogue"
    kv_model = SimpleNamespace(
        model=SimpleNamespace(
            _geopact_prune_ratio=None,
            _geopact_prune_policy="",
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        )
    )
    embed_payload = {
        "inputs_embeds": torch.zeros(1, 16, 2560),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 14, 14]]),
        "pixel_values": torch.zeros(1, 588, 1176),
        "position_ids": torch.zeros(3, 1, 16, dtype=torch.long),
        "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        "past_key_values": type("PKV", (), {"get_seq_length": lambda self: 10})(),
        "_geopact_pinned_cache_start": 15,
        "_geopact_decode_only_generate": True,
    }
    built_kv = build_dialogue_generate_payload(
        kv_model,
        embed_payload,
        logical_prompt_len=16,
        dialogue_suffix_only=True,
    )
    check("incremental embed strips pixel_values", "pixel_values" not in built_kv)
    check(
        "incremental embed pinned decode",
        built_kv.get("position_ids") is not None
        and built_kv.get("_geopact_pinned_cache_start") is not None
        and built_kv.get("_geopact_rope_deltas") is not None,
    )

    compact_payload = {
        "inputs_embeds": torch.zeros(1, 20, 2560),
        "attention_mask": torch.ones(1, 20, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 14, 14]]),
        "pixel_values": torch.zeros(1, 588, 1176),
        "position_ids": torch.zeros(3, 1, 20, dtype=torch.long),
        "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
    }
    built_compact = build_dialogue_generate_payload(
        baseline_model,
        compact_payload,
        logical_prompt_len=20,
    )
    check("origpos compact strips pixel_values", "pixel_values" not in built_compact)
    check("origpos compact pinned", should_use_pinned_generate(built_compact))

    print()
    if failures:
        print(f"FAILED {len(failures)}: {failures}", file=sys.stderr)
        return 1
    print("ALL BASELINE GENERATE PAYLOAD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
