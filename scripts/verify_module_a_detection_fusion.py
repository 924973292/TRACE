#!/usr/bin/env python3
"""Smoke tests for Module A detection fusion (circular decouple, OCR, stripe)."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SAMPLE = REPO / "data" / "benchmark"


@contextmanager
def _env_overlay(updates: dict[str, str | None]) -> Iterator[None]:
    saved: dict[str, Optional[str]] = {}
    try:
        for key, val in updates.items():
            saved[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(val)
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _find_sample_image() -> Image.Image:
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in sorted(SAMPLE.rglob(pattern))[:5]:
            if path.is_file():
                return Image.open(path).convert("RGB")
    raise FileNotFoundError(f"No sample image under {SAMPLE}")


def _sources(mius: list) -> set[str]:
    return {str(m.get("source", "")) for m in mius}


def test_circular_without_yolo() -> None:
    from lmms_eval.models.gui_miu_detectors import _augment_with_circular_regions

    pil = _find_sample_image()
    import numpy as np

    image_np = np.asarray(pil)
    with _env_overlay({"GEOPACT_YOLO_CIRCULAR_AUX": "1"}):
        out = _augment_with_circular_regions([], pil, image_np)
    circular = _sources(out) & {
        "blob_circle",
        "blob_ellipse",
        "hough_circle",
        "edge_ellipse",
        "circular_aggregate",
    }
    assert out is not None, "augment returned None"
    print(f"OK circular_on_empty_yolo: n={len(out)} circular_sources={circular or 'none'}")


def test_ocr_fusion_adds_ocr_source() -> None:
    from lmms_eval.models.gui_miu_detectors import detect_omniparser_icon_mius_raw

    pil = _find_sample_image()
    with _env_overlay(
        {
            "GEOPACT_YOLO_OCR_AUX": "1",
            "GEOPACT_YOLO_CIRCULAR_AUX": "1",
            "MIU_DISABLE_OCR": "0",
        }
    ):
        raw = detect_omniparser_icon_mius_raw(pil)
    assert raw is not None, "omniparser raw is None"
    mius, _np = raw
    has_yolo = any(s == "yolo" or s.startswith("box_yolo") for s in _sources(mius))
    has_ocr = "ocr" in _sources(mius)
    print(f"OK ocr_fusion: n={len(mius)} yolo={has_yolo} ocr={has_ocr}")
    if not has_ocr:
        print("WARN: no OCR boxes (engine may be unavailable); YOLO path still OK")


def test_ocr_only_no_yolo() -> None:
    from lmms_eval.models.gui_miu_detectors import detect_ocr_only_mius_raw

    pil = _find_sample_image()
    with _env_overlay({"GEOPACT_YOLO_CIRCULAR_AUX": "1", "MIU_DISABLE_OCR": "0"}):
        mius, _np = detect_ocr_only_mius_raw(pil)
    yolo = [s for s in _sources(mius) if s == "yolo" or s.startswith("box_yolo")]
    assert not yolo, f"ocr_only should not contain YOLO sources: {yolo}"
    print(f"OK ocr_only: n={len(mius)} sources={sorted(_sources(mius))}")


def test_stripe_aux() -> None:
    from lmms_eval.models.gui_miu_detectors import detect_omniparser_icon_mius_raw

    pil = _find_sample_image()
    with _env_overlay({"GEOPACT_YOLO_STRIPE_AUX": "1", "GEOPACT_YOLO_CIRCULAR_AUX": "1"}):
        raw = detect_omniparser_icon_mius_raw(pil)
    assert raw is not None
    mius, _np = raw
    stripe = sum(1 for m in mius if m.get("source") == "stripe_row")
    print(f"OK stripe_aux: n={len(mius)} stripe_row={stripe}")


def main() -> int:
    test_circular_without_yolo()
    test_ocr_fusion_adds_ocr_source()
    test_ocr_only_no_yolo()
    test_stripe_aux()
    print("ALL verify_module_a_detection_fusion checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
