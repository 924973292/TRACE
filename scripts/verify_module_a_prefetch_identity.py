#!/usr/bin/env python3
"""Regression check: Module A prefetch results must never cross frames.

Root-cause context: try_consume_module_a_prefetch joined whatever future was
pending without checking which image it was detected on. The last step of an
episode stages a prefetch (stage_pending_last_full_frame) that is never
consumed, so the NEXT episode's first rectify would splice the previous
episode's last-frame MIUs onto its own frame 0 (wrong info_density / CDCT
prior, silent keep-set corruption). This verifier fails if identity checking
or episode-boundary cleanup regresses.

Runs on CPU in seconds; no model weights needed.
"""
from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _handle(image_id: str) -> dict:
    fut: Future = Future()
    fut.set_result(([{"miu": image_id}], {"timings_ms": {}}))
    return {
        "future": fut,
        "started_at": 0.0,
        "vit_done_at": None,
        "detector": "omniparser",
        "policy": "oas_ickh_cdct_no_coord_yolo",
        "image_id": image_id,
        "grid_hw": (10, 10),
        "source": "history_capture",
    }


def main() -> int:
    from lmms_eval.models.geopact_module_a_prefetch import (
        discard_module_a_prefetch,
        try_consume_module_a_prefetch,
    )

    failures: list[str] = []

    # 1. Mismatched image id -> refuse to consume, handle cleared.
    model = SimpleNamespace()
    model._geopact_module_a_async = _handle("ep1/frame_last.png")
    out = try_consume_module_a_prefetch(model, expected_image_id="ep2/frame_0.png")
    if out is not None:
        failures.append("stale prefetch consumed across frames (expected None)")
    if getattr(model, "_geopact_module_a_async", "sentinel") is not None:
        failures.append("stale handle not cleared after identity mismatch")

    # 2. Matching image id -> consumed normally.
    model = SimpleNamespace()
    model._geopact_module_a_async = _handle("ep2/frame_0.png")
    out = try_consume_module_a_prefetch(model, expected_image_id="ep2/frame_0.png")
    if out is None:
        failures.append("matching prefetch was not consumed")
    else:
        mius, meta = out
        if not (mius and mius[0].get("miu") == "ep2/frame_0.png"):
            failures.append(f"wrong mius consumed: {mius}")
        if not meta.get("module_a_prefetch_async"):
            failures.append("consume meta missing module_a_prefetch_async")

    # 3. No expected id (legacy callers) -> consume unconditionally.
    model = SimpleNamespace()
    model._geopact_module_a_async = _handle("whatever.png")
    if try_consume_module_a_prefetch(model) is None:
        failures.append("legacy consume without expected id broke")

    # 4. Episode boundary discard drops the handle without joining.
    model = SimpleNamespace()
    model._geopact_module_a_async = _handle("ep1/frame_last.png")
    discard_module_a_prefetch(model)
    if model._geopact_module_a_async is not None:
        failures.append("discard_module_a_prefetch left the handle in place")

    # 5. clear_session_kv discards any pending prefetch.
    import lmms_eval.models.geopact_session_kv as skv

    model = SimpleNamespace()
    model._geopact_module_a_async = _handle("ep1/frame_last.png")
    skv.clear_session_kv(model)
    if getattr(model, "_geopact_module_a_async", "sentinel") is not None:
        failures.append("clear_session_kv did not discard pending prefetch")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS verify_module_a_prefetch_identity: 5/5 checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
