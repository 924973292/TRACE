#!/usr/bin/env python3
"""Offline check: dialogue KV full-prefill routes native (GUI) vs embed (suffix)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    os.environ.setdefault("GEOPACT_SESSION_KV", "1")
    os.environ.setdefault("GEOPACT_SESSION_KV_MODE", "dialogue")
    os.environ["GEOPACT_KV_STEP0_NATIVE"] = "1"

    from lmms_eval.models.geopact_vl_prepare import (
        _dialogue_kv_use_native_full_prefill,
        kv_step0_native_enabled,
        prepare_active_prefill_inputs,
    )

    class _Inner:
        _geopact_dialogue_suffix_only_step = False

        def get_image_features(self):
            return None

    class _Model:
        model = _Inner()

    if not kv_step0_native_enabled():
        print("FAIL: GEOPACT_KV_STEP0_NATIVE should default on")
        return 1
    if not _dialogue_kv_use_native_full_prefill(_Model()):
        print("FAIL: expected native full prefill flag for dialogue KV step0")
        return 1

    _Inner._geopact_dialogue_suffix_only_step = True
    if _dialogue_kv_use_native_full_prefill(_Model()):
        print("FAIL: suffix-only step must not use native full prefill")
        return 1

    os.environ["GEOPACT_KV_STEP0_NATIVE"] = "0"
    _Inner._geopact_dialogue_suffix_only_step = False
    if _dialogue_kv_use_native_full_prefill(_Model()):
        print("FAIL: native disabled should return false")
        return 1

    # Native step0 must not route to dialogue_embed in prepare (falls through).
    _Inner._geopact_dialogue_suffix_only_step = False
    os.environ["GEOPACT_KV_STEP0_NATIVE"] = "1"
    active = {"input_ids": None, "pixel_values": "pv"}
    result, _raw = prepare_active_prefill_inputs(_Model(), active, use_pruning=False)
    if result is not active:
        print(f"FAIL: native step0 prepare should return active inputs, got keys={list(result)}")
        return 1

    print("OK kv_step0_native routing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
