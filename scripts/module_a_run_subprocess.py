#!/usr/bin/env python3
"""Isolated Module A worker for prefetch (no Qwen LLM in this process)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PIL import Image


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _apply_worker_yolo_device() -> None:
    explicit = os.environ.get("GEOPACT_MODULE_A_WORKER_YOLO_DEVICE", "").strip()
    if explicit:
        os.environ["MIU_YOLO_DEVICE"] = explicit
    else:
        os.environ.setdefault("MIU_YOLO_DEVICE", "cpu")
    os.environ.setdefault("MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK", "1")
    try:
        import cv2  # type: ignore

        cv2.setNumThreads(1)
    except Exception:
        pass


def _load_image_from_request(msg: dict[str, Any]) -> Image.Image:
    shm_info = msg.get("image_shm")
    if isinstance(shm_info, dict) and shm_info.get("name"):
        width = int(shm_info.get("width") or 0)
        height = int(shm_info.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image_shm dimensions: {width}x{height}")
        shm = shared_memory.SharedMemory(name=str(shm_info["name"]))
        try:
            arr = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
            return Image.fromarray(arr.copy(), mode="RGB")
        finally:
            shm.close()
    image_path = str(msg.get("image_path") or "")
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path!r}")
    return Image.open(image_path).convert("RGB")


def _run_once_pil(
    pil: Image.Image,
    detector: str,
    policy: str,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> dict[str, Any]:
    _apply_worker_yolo_device()

    from lmms_eval.models.geopact_vl_hook import (
        _apply_attribute_drops,
        _detect_raw_boxes,
        _geopact_fail,
        _geopact_module_a_warn_skip,
        _policy_is_token_baseline,
    )
    from lmms_eval.models.module_a_pipeline import run_module_a

    mius, meta = run_module_a(
        pil,
        detector,
        policy,
        grid_hw=grid_hw,
        detect_raw_fn=lambda: _detect_raw_boxes(pil, detector, policy),
        apply_attribute_drops_fn=_apply_attribute_drops,
        policy_is_token_baseline_fn=_policy_is_token_baseline,
        warn_skip_abc_fn=lambda msg: _geopact_module_a_warn_skip(msg, policy),
        fail_fn=_geopact_fail,
    )
    return {"mius": mius, "meta": meta}


def _run_once(
    image_path: str,
    detector: str,
    policy: str,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> dict[str, Any]:
    pil = Image.open(image_path).convert("RGB")
    return _run_once_pil(pil, detector, policy, grid_hw=grid_hw)


def _warm_worker() -> None:
    try:
        from lmms_eval.models.gui_miu_detectors import warm_omniparser_icon_yolo

        warm_omniparser_icon_yolo()
    except Exception:
        pass


def _serve_loop() -> int:
    _warm_worker()
    sys.stdout.write(json.dumps({"ready": True, "mode": "serve"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps({"ok": False, "error": f"bad json: {exc}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        if msg.get("cmd") == "shutdown":
            sys.stdout.write(json.dumps({"ok": True, "shutdown": True}) + "\n")
            sys.stdout.flush()
            return 0

        req_id = msg.get("id")
        detector = str(msg.get("detector") or "omniparser")
        policy = str(msg.get("policy") or "oas_ickh_cdct_no_coord_yolo")
        grid_raw = msg.get("grid_hw")
        grid_hw = None
        if isinstance(grid_raw, (list, tuple)) and len(grid_raw) == 2:
            grid_hw = (int(grid_raw[0]), int(grid_raw[1]))

        try:
            pil = _load_image_from_request(msg)
            payload = _run_once_pil(pil, detector, policy, grid_hw=grid_hw)
            sys.stdout.write(
                json.dumps(
                    {
                        "id": req_id,
                        "ok": True,
                        "mius": payload["mius"],
                        "meta": payload["meta"],
                    },
                    default=_json_default,
                )
                + "\n"
            )
        except Exception as exc:
            sys.stdout.write(
                json.dumps({"id": req_id, "ok": False, "error": str(exc)}) + "\n"
            )
        sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Module A pipeline in an isolated process.")
    parser.add_argument("image_path", type=str, nargs="?", default=None, help="PNG/JPEG path")
    parser.add_argument("--detector", type=str, default="omniparser")
    parser.add_argument("--policy", type=str, default="oas_ickh_cdct_no_coord_yolo")
    parser.add_argument("--grid-hw", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Persistent worker: read JSON requests from stdin, one response per line on stdout",
    )
    args = parser.parse_args()

    if args.serve:
        return _serve_loop()

    if not args.image_path:
        parser.error("image_path required unless --serve")
    payload = _run_once(
        args.image_path,
        args.detector,
        args.policy,
        grid_hw=tuple(args.grid_hw) if args.grid_hw is not None else None,
    )
    sys.stdout.write(json.dumps(payload, default=_json_default))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
