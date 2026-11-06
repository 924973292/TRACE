#!/usr/bin/env python3
"""PNG vs SHM worker IPC parity: same image must yield identical mius payloads."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image


def _mius_equal(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if dict(left) != dict(right):
            return False
    return True


def main() -> int:
    os.environ["GEOPACT_MODULE_A_SUBPROCESS"] = "1"
    os.environ["GEOPACT_MODULE_A_SUBPROCESS_POOL"] = "1"
    os.environ.setdefault("MIU_YOLO_DEVICE", "cpu")
    os.environ.setdefault("MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK", "1")

    from lmms_eval.models.geopact_module_a_worker import (
        get_module_a_worker,
        shutdown_module_a_worker,
    )

    pil = Image.new("RGB", (128, 96), (240, 240, 240))
    detector = "omniparser"
    policy = "oas_ickh_cdct_no_coord_yolo"

    try:
        os.environ["GEOPACT_MODULE_A_WORKER_SHM"] = "0"
        shutdown_module_a_worker()
        mius_png, _ = get_module_a_worker().detect(pil, detector, policy)

        os.environ["GEOPACT_MODULE_A_WORKER_SHM"] = "1"
        shutdown_module_a_worker()
        mius_shm, _ = get_module_a_worker().detect(pil, detector, policy)
    finally:
        shutdown_module_a_worker()

    if not _mius_equal(mius_png, mius_shm):
        print(
            f"SHM parity failed: png={len(mius_png)} shm={len(mius_shm)}",
            file=sys.stderr,
        )
        return 1
    print(f"SHM parity OK ({len(mius_png)} mius)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
