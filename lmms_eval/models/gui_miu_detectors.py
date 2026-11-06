"""Pluggable GUI component detectors for MIU construction (heuristic vs learned parsers)."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

_YOLO_STRIDE = 32


def _snap_yolo_stride(v: int, stride: int = _YOLO_STRIDE) -> int:
    """Round vision dimension up to YOLO max stride (avoids Ultralytics resize warning)."""
    return max(stride, int(math.ceil(float(v) / float(stride)) * stride))

def _allow_fallback_yolo() -> bool:
    return os.environ.get("GEOPACT_ALLOW_FALLBACK_YOLO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_omniparser_icon_weights() -> str:
    """Paper LIP uses OmniParser v2 icon weights. Do not silently fall back."""
    for key in ("OMNIPARSER_ICON_WEIGHTS", "MIU_YOLO_WEIGHTS"):
        val = os.environ.get(key, "").strip()
        if not val:
            continue
        if val == "yolov8n.pt" and not _allow_fallback_yolo():
            raise RuntimeError(
                "OMNIPARSER_ICON_WEIGHTS=yolov8n.pt is not the paper detector. "
                "Point it at OmniParser v2 icon_detect/model.pt, or set "
                "GEOPACT_ALLOW_FALLBACK_YOLO=1 for non-paper debugging."
            )
        if val != "yolov8n.pt" and not Path(val).is_file():
            raise FileNotFoundError(f"{key}={val} is not a file")
        return val
    repo_default = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "omniparser"
        / "icon_detect"
        / "model.pt"
    )
    if repo_default.is_file():
        return str(repo_default)
    if _allow_fallback_yolo():
        return "yolov8n.pt"
    raise RuntimeError(
        "OmniParser v2 icon weights not found. Set OMNIPARSER_ICON_WEIGHTS to "
        "icon_detect/model.pt (required for paper LIP). Set "
        "GEOPACT_ALLOW_FALLBACK_YOLO=1 only for non-paper debugging."
    )

MiuDict = dict
DetectFn = Callable[[Image.Image], tuple[list[MiuDict], dict[str, float]]]

_YOLO_MODEL = None
_YOLO_WEIGHTS = None
_YOLO_LOAD_ERROR: Optional[str] = None
_EASYOCR_READER = None
_EASYOCR_FAILED = False
_RAPIDOCR_ENGINE = None
_RAPIDOCR_FAILED = False
_PYTESS_FAILED = False
_OCR_SUBPROC_FALLBACK_WARNED = False
_OCR_INPROC_IMPORT_OK: Optional[bool] = None

# Florence-2 icon-caption model (lazy, loaded once per worker, GPU if available).
# Mirrors load_caption_model() in scripts/profile_omniparser_v2_full.py.
_FLORENCE_MODEL = None
_FLORENCE_FAILED = False

# v5: detection cache + import-state sentinel.
#
# ``_YOLO_IMPORT_OK`` is tri-state:
#   None   -> haven't tried yet, will attempt on first call
#   True   -> in-process ultralytics+cv2 works on this venv; never subprocess
#   False  -> in-process load raised ImportError (cv2 / ultralytics missing);
#             fall back to subprocess once and stick with it
# v4 used ``if in_proc:`` (falsy on empty list) which caused a 25 s subprocess
# fallback on every icon-free screen (~1.4 % of samples in the v4 21-run
# matrix).  The sentinel separates "loader is broken" from "this screen
# happens to have zero icons", so the latter no longer wastes 25 s.
_YOLO_IMPORT_OK: Optional[bool] = None

# Perceptual-hash LRU for ``build_raw_mius``.  GUI agent trajectories
# routinely keep the same screenshot across consecutive steps (e.g. typing
# into an input field or scrolling small deltas); caching by an 8x8 average
# hash gives 25-40 % hit rate in our v4 logs with zero false positives at
# bit-distance 0.  Cache is disabled when ``MIU_DETECT_CACHE=0``.
_DETECT_CACHE: "OrderedDict[tuple, tuple[list[MiuDict], dict[str, float]]]" = OrderedDict()
_DETECT_CACHE_MAX = int(os.environ.get("MIU_DETECT_CACHE_MAX", "64"))
_DETECT_CACHE_HITS = 0
_DETECT_CACHE_MISSES = 0


def _env_true(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_yolo_gpu() -> bool:
    return _env_true("MIU_REQUIRE_YOLO_GPU", False)


def _disable_omniparser_subproc_fallback() -> bool:
    return _env_true("MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK", False)


def _phash_8x8(image: Image.Image) -> bytes:
    """Stable 8-bit perceptual hash (avg-hash on 8x8 grayscale)."""
    small = image.convert("L").resize((8, 8), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int32)
    mean = float(arr.mean())
    bits = (arr > mean).astype(np.uint8).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out.to_bytes(8, "big")


def detector_cache_stats() -> dict[str, float]:
    """Read-only stats; exposed so the qwen2_5_omni log can carry hit rate."""
    total = _DETECT_CACHE_HITS + _DETECT_CACHE_MISSES
    return {
        "miu_cache_size": float(len(_DETECT_CACHE)),
        "miu_cache_hits": float(_DETECT_CACHE_HITS),
        "miu_cache_misses": float(_DETECT_CACHE_MISSES),
        "miu_cache_hit_rate": float(_DETECT_CACHE_HITS / total) if total else 0.0,
    }


def reset_detector_cache() -> None:
    global _DETECT_CACHE_HITS, _DETECT_CACHE_MISSES
    _DETECT_CACHE.clear()
    _DETECT_CACHE_HITS = 0
    _DETECT_CACHE_MISSES = 0


def _normalize_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return (
        float(max(0.0, min(1.0, x1))),
        float(max(0.0, min(1.0, y1))),
        float(max(0.0, min(1.0, x2))),
        float(max(0.0, min(1.0, y2))),
    )


def detect_heuristic_visual(image: Image.Image, max_edge: int = 420) -> list[MiuDict]:
    width, height = image.size
    scale = min(max_edge / max(width, height), 1.0)
    small = image.convert("RGB")
    if scale < 1.0:
        small = small.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    arr = np.asarray(small).astype(np.float32)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = gx + gy
    color_var = np.max(arr, axis=2) - np.min(arr, axis=2)
    threshold = max(float(np.percentile(grad, 82)), 10.0)
    mask = (grad > threshold) | (color_var > 38)
    for _ in range(2):
        padded = np.pad(mask, 1, mode="constant")
        mask = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    mius: list[MiuDict] = []
    min_area = max(8, int(h * w * 0.00015))
    max_area = int(h * w * 0.35)
    for yy in range(h):
        xs = np.flatnonzero(mask[yy] & ~visited[yy])
        for start_x in xs:
            if visited[yy, start_x] or not mask[yy, start_x]:
                continue
            stack = [(yy, int(start_x))]
            visited[yy, start_x] = True
            min_x = max_x = int(start_x)
            min_y = max_y = yy
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            if area < min_area or area > max_area or bw < 3 or bh < 3:
                continue
            nx1, ny1 = min_x / w, min_y / h
            nx2, ny2 = (max_x + 1) / w, (max_y + 1) / h
            aspect = bw / max(bh, 1)
            comp_type = "text_like" if aspect > 3.0 else "icon_button"
            if bw * bh > h * w * 0.04:
                comp_type = "panel"
            mius.append(
                {
                    "box": _normalize_box(nx1, ny1, nx2, ny2),
                    "type": comp_type,
                    "text": "",
                    "source": "vision",
                    "det_score": min(1.0, area / max(bw * bh, 1)),
                }
            )
    return mius


def _detect_ocr_via_pytesseract(image: Image.Image) -> Optional[list[MiuDict]]:
    """Return None if pytesseract is unavailable; [] is a valid empty result."""
    global _PYTESS_FAILED
    try:
        import pytesseract  # type: ignore
    except Exception as exc:
        if not _PYTESS_FAILED:
            _PYTESS_FAILED = True
            _log.warning("pytesseract unavailable; falling back to easyocr if installed (%r)", exc)
        return None
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        _log.debug("pytesseract.image_to_data failed: %r", exc)
        return []

    width, height = image.size
    mius: list[MiuDict] = []
    for idx, text in enumerate(data.get("text", [])):
        text = str(text or "").strip()
        try:
            conf = float(data.get("conf", [0])[idx])
        except Exception:
            conf = 0.0
        if not text or conf < 25:
            continue
        x = float(data["left"][idx]) / max(width, 1)
        y = float(data["top"][idx]) / max(height, 1)
        w = float(data["width"][idx]) / max(width, 1)
        h = float(data["height"][idx]) / max(height, 1)
        if w * h < 0.00002:
            continue
        mius.append(
            {
                "box": _normalize_box(x, y, x + w, y + h),
                "type": "text",
                "text": text[:48],
                "source": "ocr",
                "det_score": min(max(conf / 100.0, 0.0), 1.0),
            }
        )
    return mius


def preload_ocr_if_requested() -> bool:
    """Warm EasyOCR once per worker process (``MIU_PRELOAD_EASYOCR=1``)."""
    if os.environ.get("MIU_PRELOAD_EASYOCR", "0").strip().lower() not in {"1", "true", "yes"}:
        return False
    reader = _get_easyocr_reader()
    return reader is not None


def _get_easyocr_reader():
    """Lazy-init EasyOCR reader; install via scripts/setup_omini_eval_omniparser.sh."""
    global _EASYOCR_READER, _EASYOCR_FAILED
    if _EASYOCR_FAILED:
        return None
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr  # type: ignore

        # EasyOCR is the detector's dominant cost.  On CPU, readtext over a
        # full-res mobile screenshot (en+ch_sim) is ~7-16 s; on GPU it is a
        # few hundred ms.  The en+ch_sim recognizer/detector weights are tiny
        # (tens of MB) so co-loading next to the 7B LLM is safe on an 80 GB
        # card.  Default to GPU when CUDA is available; MIU_OCR_GPU overrides
        # ("0" forces CPU, "1" forces GPU).
        gpu_env = os.environ.get("MIU_OCR_GPU", "").strip().lower()
        if gpu_env in {"0", "false", "no"}:
            use_gpu = False
        elif gpu_env in {"1", "true", "yes"}:
            use_gpu = True
        else:
            try:
                import torch  # type: ignore

                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
        _EASYOCR_READER = easyocr.Reader(["en", "ch_sim"], gpu=use_gpu, verbose=False)
        return _EASYOCR_READER
    except Exception as exc:
        _EASYOCR_FAILED = True
        _log.warning("easyocr unavailable; OCR-derived MIUs will be empty (%r)", exc)
        return None


def _easyocr_readtext_kwargs() -> dict:
    """Tunable EasyOCR readtext knobs.

    EasyOCR cost is dominated by the CRAFT text *detector* running on the
    full-resolution image, NOT the per-line recognizer (a profiling sweep on
    mobile GUIs found recognizer batch_size tuning ~flat).  ``mag_ratio`` scales
    the CRAFT input: ``0.75`` shaved ~8% wall time while slightly *raising*
    recall (no loss) in the sweep, so it is the default.  Override via
    ``OMNIPARSER_OCR_MAG`` (set 1.0 for stock EasyOCR) / ``OMNIPARSER_OCR_CANVAS``.
    """
    kw: dict = {}
    try:
        mag = float(os.environ.get("OMNIPARSER_OCR_MAG", "0.75"))
        if mag > 0:
            kw["mag_ratio"] = mag
    except Exception:
        pass
    canvas = os.environ.get("OMNIPARSER_OCR_CANVAS", "").strip()
    if canvas:
        try:
            kw["canvas_size"] = int(canvas)
        except Exception:
            pass
    return kw


def _ocr_engine_name() -> str:
    """Configured OCR backend: ``easyocr`` (default) or ``rapidocr``.

    ``rapidocr`` runs PP-OCRv5 *mobile* det+rec via ONNXRuntime (det DBNet-mobile
    + CRNN-mobile, ~10-16 MB total, CPU- or GPU-capable).  It replaces EasyOCR's
    heavy CRAFT full-image text detector (the dominant ~660 ms GPU cost) with a
    far lighter mobile detector, and drops the hard PyTorch/GPU dependency.
    Select via ``MIU_OCR_ENGINE=rapidocr``.
    """
    name = os.environ.get("MIU_OCR_ENGINE", "easyocr").strip().lower()
    return name if name in {"easyocr", "rapidocr"} else "easyocr"


def _ocr_use_gpu() -> bool:
    """Shared GPU policy for OCR backends (``MIU_OCR_GPU`` overrides auto)."""
    gpu_env = os.environ.get("MIU_OCR_GPU", "").strip().lower()
    if gpu_env in {"0", "false", "no"}:
        return False
    if gpu_env in {"1", "true", "yes"}:
        return True
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_rapidocr_engine():
    """Lazy-init RapidOCR (PP-OCRv5/v4 mobile det+rec, ONNXRuntime).

    Install with ``pip install rapidocr onnxruntime`` (3.x, PP-OCRv5 mobile) or
    ``pip install rapidocr_onnxruntime onnxruntime`` (1.4.x legacy, PP-OCRv4
    mobile).  Both ship lightweight mobile det+rec; we only toggle the CUDA
    provider when a GPU is requested.
    """
    global _RAPIDOCR_ENGINE, _RAPIDOCR_FAILED
    if _RAPIDOCR_FAILED:
        return None
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    use_gpu = _ocr_use_gpu()
    # ---- Try modern rapidocr 3.x first ----
    try:
        from rapidocr import RapidOCR  # type: ignore

        engine = None
        if use_gpu:
            for params in (
                {"Det.engine_cfg.use_cuda": True, "Rec.engine_cfg.use_cuda": True},
                {"EngineConfig.onnxruntime.use_cuda": True},
            ):
                try:
                    engine = RapidOCR(params=params)
                    break
                except Exception:
                    engine = None
        if engine is None:
            engine = RapidOCR()
        _RAPIDOCR_ENGINE = engine
        return _RAPIDOCR_ENGINE
    except Exception:
        pass
    # ---- Fall back to legacy rapidocr_onnxruntime 1.4.x ----
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = None
        if use_gpu:
            try:
                engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
            except Exception:
                engine = None
        if engine is None:
            engine = RapidOCR()
        _RAPIDOCR_ENGINE = engine
        return _RAPIDOCR_ENGINE
    except Exception as exc:
        _RAPIDOCR_FAILED = True
        _log.warning("rapidocr unavailable; falling back/OCR empty (%r)", exc)
        return None


def _rapidocr_readtext(image_np: "np.ndarray") -> list:
    """Run RapidOCR and normalize to EasyOCR's ``[(quad, text, conf), ...]``.

    Handles both the RapidOCR 3.x result-object API (``.boxes/.txts/.scores``)
    and the legacy ``(list, elapse)`` tuple API.
    """
    engine = _get_rapidocr_engine()
    if engine is None:
        return []
    try:
        out = engine(image_np)
    except Exception as exc:
        _log.debug("rapidocr inference failed: %r", exc)
        return []
    if out is None:
        return []
    results: list = []
    # New object API.
    boxes = getattr(out, "boxes", None)
    txts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if boxes is not None and txts is not None:
        try:
            for box, txt, sc in zip(boxes, txts, scores if scores is not None else [1.0] * len(txts)):
                quad = [[float(p[0]), float(p[1])] for p in box]
                results.append((quad, str(txt or ""), float(sc) if sc is not None else 1.0))
            return results
        except Exception:
            results = []
    # Legacy tuple API: (list_of_[box,text,score], elapse) or bare list.
    seq = out[0] if isinstance(out, tuple) else out
    if not seq:
        return []
    for entry in seq:
        try:
            quad = [[float(p[0]), float(p[1])] for p in entry[0]]
            results.append((quad, str(entry[1] or ""), float(entry[2])))
        except Exception:
            continue
    return results


def _run_ocr_readtext(image_np: "np.ndarray") -> list:
    """Unified OCR entry: returns ``[(quad, text, conf), ...]`` for the
    configured backend so all downstream box-merge logic is engine-agnostic.
    """
    if _ocr_engine_name() == "rapidocr":
        res = _rapidocr_readtext(image_np)
        if res:
            return res
        # If RapidOCR produced nothing only because it is unavailable, fall
        # back to EasyOCR so the pipeline degrades gracefully.
        if _RAPIDOCR_FAILED:
            reader = _get_easyocr_reader()
            if reader is not None:
                try:
                    return reader.readtext(image_np, detail=1, paragraph=False, **_easyocr_readtext_kwargs())
                except Exception:
                    return []
        return res
    reader = _get_easyocr_reader()
    if reader is None:
        return []
    try:
        return reader.readtext(image_np, detail=1, paragraph=False, **_easyocr_readtext_kwargs())
    except Exception as exc:
        _log.debug("easyocr.readtext failed: %r", exc)
        return []


def _icon_recocr_enabled() -> bool:
    """Rec-only labeling of YOLO icon crops (A_text without full-image OCR det).

    When ``MIU_ICON_RECOCR=1`` the lean icon front-end reuses the YOLO boxes as
    the detection stage and runs RapidOCR's *recognizer only* (no DBNet text
    detector) on each box crop in ONE batched GPU call.  This is the
    speed-first path: it gives a TRUE ``has_text`` / ``A_text`` bit (rec read
    real characters in the box) at recognizer-only cost, and lets
    ``distinctiveness`` dedup on the recognized text.
    """
    return os.environ.get("MIU_ICON_RECOCR", "0").strip().lower() in {"1", "true", "yes"}


def _rapidocr_rec_crops(image_np: "np.ndarray", boxes_norm: list) -> list:
    """Batched RapidOCR rec-only over normalized boxes; returns [(text, score), ...].

    Reuses the YOLO boxes (no text detector) and the already-loaded RapidOCR
    engine's ``text_rec`` component, which accepts a list of crops and runs a
    single batched recognizer pass on GPU.  Missing/degenerate crops map to
    ``("", 0.0)`` so the output aligns 1:1 with ``boxes_norm``.
    """
    n = len(boxes_norm)
    if n == 0:
        return []
    engine = _get_rapidocr_engine()
    rec = getattr(engine, "text_rec", None) if engine is not None else None
    if rec is None:
        return [("", 0.0)] * n
    H, W = image_np.shape[0], image_np.shape[1]
    crops: list = []
    crop_index: list[int] = []  # map crop position -> original box index
    out: list = [("", 0.0)] * n
    for i, b in enumerate(boxes_norm):
        x1 = max(0, min(W - 1, int(round(float(b[0]) * W))))
        y1 = max(0, min(H - 1, int(round(float(b[1]) * H))))
        x2 = max(0, min(W, int(round(float(b[2]) * W))))
        y2 = max(0, min(H, int(round(float(b[3]) * H))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        crops.append(image_np[y1:y2, x1:x2, :])
        crop_index.append(i)
    if not crops:
        return out
    try:
        res = rec(crops)
        # TextRecognizer returns (List[(text, score)], elapse).
        pairs = res[0] if isinstance(res, tuple) else res
        for pos, pair in enumerate(pairs):
            if pos >= len(crop_index):
                break
            try:
                txt = str(pair[0] or "")
                sc = float(pair[1]) if len(pair) > 1 and pair[1] is not None else 0.0
            except Exception:
                txt, sc = "", 0.0
            out[crop_index[pos]] = (txt, sc)
    except Exception as exc:
        _log.debug("rapidocr rec-only batch failed: %r", exc)
    return out


def _detect_ocr_via_easyocr(image: Image.Image) -> list[MiuDict]:
    arr = np.asarray(image.convert("RGB"))
    try:
        results = _run_ocr_readtext(arr)
    except Exception as exc:
        _log.debug("ocr readtext failed: %r", exc)
        return []
    if not results:
        return []
    width, height = image.size
    mius: list[MiuDict] = []
    for entry in results:
        # easyocr returns [(quad, text, conf), ...] where quad is 4 corners.
        try:
            quad, text, conf = entry[0], entry[1], float(entry[2])
        except Exception:
            continue
        text = str(text or "").strip()
        if not text or conf < 0.25:
            continue
        xs = [float(pt[0]) for pt in quad]
        ys = [float(pt[1]) for pt in quad]
        x1, x2 = min(xs) / max(width, 1), max(xs) / max(width, 1)
        y1, y2 = min(ys) / max(height, 1), max(ys) / max(height, 1)
        if (x2 - x1) * (y2 - y1) < 0.00002:
            continue
        mius.append(
            {
                "box": _normalize_box(x1, y1, x2, y2),
                "type": "text",
                "text": text[:48],
                "source": "ocr",
                "det_score": float(conf),
            }
        )
    return mius


def detect_ocr_mius(image: Image.Image) -> list[MiuDict]:
    """OCR-derived text MIUs.

    Backend priority:
    1. ``pytesseract`` (legacy, lightweight) when its native binary is on PATH;
    2. ``easyocr`` (deep-learning OCR, available in the ``OminiDetect`` env)
       which we install once per remote machine so the subprocess detector
       does not silently return [] when tesseract is missing.
    """
    pytess_result = _detect_ocr_via_pytesseract(image)
    if pytess_result is not None:
        return pytess_result
    return _detect_ocr_via_easyocr(image)


def _ocr_inproc_import_available() -> bool:
    """True when pytesseract or easyocr can be imported in the current process."""
    global _OCR_INPROC_IMPORT_OK
    if _OCR_INPROC_IMPORT_OK is not None:
        return _OCR_INPROC_IMPORT_OK
    ok = False
    try:
        import pytesseract  # type: ignore  # noqa: F401

        ok = True
    except Exception:
        pass
    if not ok:
        try:
            import easyocr  # type: ignore  # noqa: F401

            ok = True
        except Exception:
            pass
    _OCR_INPROC_IMPORT_OK = ok
    return ok


def _detect_ocr_via_subprocess(image: Image.Image) -> Optional[list[MiuDict]]:
    """Run ``scripts/ocr_detect_mius.py`` under ``OMINI_DETECT_PYTHON`` (OminiDetect pkgs)."""
    detect_py = os.environ.get("OMINI_DETECT_PYTHON", "").strip()
    if not detect_py or not Path(detect_py).exists():
        return None
    script = Path(__file__).resolve().parents[2] / "scripts" / "ocr_detect_mius.py"
    if not script.exists():
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = handle.name
            image.convert("RGB").save(tmp_path)
        cmd = [detect_py, str(script), tmp_path]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        return json.loads(proc.stdout.strip() or "[]")
    except subprocess.CalledProcessError as exc:
        _log.warning(
            "OCR subprocess failed (exit=%s): %s",
            exc.returncode,
            (exc.stderr or "").strip().splitlines()[-1] if exc.stderr else "<no stderr>",
        )
        return None
    except Exception as exc:
        _log.warning("OCR subprocess invocation error: %r", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def resolve_ocr_mius(image: Image.Image) -> tuple[list[MiuDict], str]:
    """In-process OCR with automatic OminiDetect subprocess fallback.

    OminiEval often has YOLO/cv2 but not easyocr; icon boxes then carry
    ``text=""`` and A_ref collapses without this fallback.

    Set ``MIU_DISABLE_OCR=1`` (or ``MIU_OCR_SUBPROC_FALLBACK=0``) when the
    per-step easyocr subprocess dominates wall time (~20--30s/image).

    Returns ``(mius, backend_tag)`` where ``backend_tag`` is one of
    ``disabled``, ``inproc``, ``subproc``, or ``empty``.
    """
    global _OCR_SUBPROC_FALLBACK_WARNED
    if os.environ.get("MIU_DISABLE_OCR", "0").strip().lower() in {"1", "true", "yes"}:
        return [], "disabled"
    mius = detect_ocr_mius(image)
    if mius:
        return mius, "inproc"
    if os.environ.get("MIU_OCR_SUBPROC_FALLBACK", "1").strip() in {"0", "false", "no"}:
        return [], "empty"
    remote = _detect_ocr_via_subprocess(image)
    if remote:
        if not _OCR_SUBPROC_FALLBACK_WARNED:
            _OCR_SUBPROC_FALLBACK_WARNED = True
            _log.info(
                "OCR in-process returned 0 MIUs; using OMINI_DETECT_PYTHON subprocess "
                "(set MIU_OCR_SUBPROC_FALLBACK=0 to disable)."
            )
        return remote, "subproc"
    if not _OCR_SUBPROC_FALLBACK_WARNED:
        _OCR_SUBPROC_FALLBACK_WARNED = True
        if not _ocr_inproc_import_available():
            _log.warning(
                "OCR unavailable in-process (no pytesseract/easyocr) and subprocess OCR "
                "failed or OMINI_DETECT_PYTHON unset; text MIUs will stay empty. "
                "Run scripts/setup_omniparser_detect_env.sh and export OMINI_DETECT_PYTHON."
            )
        elif os.environ.get("OMINI_DETECT_PYTHON", "").strip():
            _log.warning(
                "OCR subprocess returned 0 MIUs; check OminiDetect easyocr install "
                "(scripts/setup_omniparser_detect_env.sh)."
            )
    return [], "empty"


def _get_yolo_model(weights: str):
    global _YOLO_MODEL, _YOLO_WEIGHTS, _YOLO_IMPORT_OK
    if _YOLO_MODEL is not None and _YOLO_WEIGHTS == weights:
        return _YOLO_MODEL
    from ultralytics import YOLO  # type: ignore

    _YOLO_MODEL = YOLO(weights)
    _YOLO_WEIGHTS = weights
    _YOLO_IMPORT_OK = True
    device = _yolo_predict_device()
    if device is not None and str(device).lower() != "cpu":
        torch_device = f"cuda:{device}" if str(device).isdigit() else str(device)
        if torch_device.startswith("cuda"):
            try:
                _YOLO_MODEL.to(torch_device)
            except Exception as exc:
                if _require_yolo_gpu():
                    raise RuntimeError(f"YOLO model could not move to {torch_device}: {exc!r}") from exc
                _log.warning("YOLO model could not move to %s; predict() will choose device (%r)", torch_device, exc)
    return _YOLO_MODEL


def _yolo_empty_cache_enabled() -> bool:
    return os.environ.get("GEOPACT_YOLO_EMPTY_CACHE", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _yolo_runs_on_gpu() -> bool:
    """True when Module A YOLO will execute on CUDA (explicit or default device)."""
    yolo_dev = os.environ.get("MIU_YOLO_DEVICE", "").strip().lower()
    if yolo_dev == "cpu":
        return False
    if yolo_dev:
        return True
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def maybe_release_yolo_gpu(*, allow_empty_cache: Optional[bool] = None) -> None:
    """Move YOLO weights off GPU after Module A so LLM decode is not contended."""
    if os.environ.get("GEOPACT_YOLO_RELEASE_GPU", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return
    yolo_dev = os.environ.get("MIU_YOLO_DEVICE", "").strip().lower()
    if yolo_dev == "cpu":
        return
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        return
    try:
        import torch  # type: ignore

        _YOLO_MODEL.to("cpu")
        use_empty_cache = (
            _yolo_empty_cache_enabled()
            if allow_empty_cache is None
            else bool(allow_empty_cache)
        )
        if use_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        _log.debug("YOLO GPU release skipped: %r", exc)


def _yolo_use_half() -> bool:
    """FP16 YOLO inference (A2); disable with GEOPACT_YOLO_HALF=0 or MIU_YOLO_DEVICE=cpu."""
    if os.environ.get("GEOPACT_YOLO_HALF", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    if os.environ.get("MIU_YOLO_DEVICE", "").strip().lower() == "cpu":
        return False
    return True


def _yolo_predict_device() -> Optional[str]:
    device = os.environ.get("MIU_YOLO_DEVICE", "").strip()
    if device:
        if device.lower() == "cpu" and _require_yolo_gpu():
            raise RuntimeError("MIU_REQUIRE_YOLO_GPU=1 but MIU_YOLO_DEVICE=cpu")
        if _require_yolo_gpu() and device.lower() != "cpu":
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda.is_available() is false")
            except Exception as exc:
                raise RuntimeError(f"MIU_REQUIRE_YOLO_GPU=1 but CUDA is unavailable: {exc!r}") from exc
        return device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    if _require_yolo_gpu():
        raise RuntimeError("MIU_REQUIRE_YOLO_GPU=1 but no CUDA YOLO device was resolved")
    return None


def warm_omniparser_icon_yolo() -> dict[str, float]:
    """Load and warm the in-process OmniParser icon detector once per worker."""
    from PIL import ImageDraw

    image = Image.new("RGB", (640, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((64, 72, 160, 168), radius=12, outline="black", width=4, fill=(230, 240, 255))
    draw.rectangle((250, 96, 430, 140), outline="black", width=3, fill=(220, 255, 220))
    draw.line((94, 120, 130, 120), fill="black", width=4)
    draw.line((112, 102, 112, 138), fill="black", width=4)
    t0 = time.perf_counter()
    mius = detect_omniparser_full_inproc(image, mode="icon")
    first_ms = (time.perf_counter() - t0) * 1000.0
    if mius is None:
        raise RuntimeError("in-process OmniParser icon warmup returned None")
    t1 = time.perf_counter()
    warm_mius = detect_omniparser_full_inproc(image, mode="icon")
    warm_ms = (time.perf_counter() - t1) * 1000.0
    return {
        "miu_yolo_warmup_first_ms": float(first_ms),
        "miu_yolo_warmup_second_ms": float(warm_ms),
        "miu_yolo_warmup_mius": float(len(mius)),
        "miu_yolo_warmup_second_mius": float(len(warm_mius or [])),
    }


def _op_box_is_solid(bbox, image_np, std_thresh: float = 6.0) -> bool:
    """True if a box interior is (near) a single flat color.

    Icon detectors routinely fire on blank backgrounds, solid white cards and
    uniform color fills; these carry no interactive affordance and are pure
    false positives.  We measure the per-channel pixel std-dev over the (inset)
    interior -- a flat region has near-zero variance regardless of which color
    it is (white background, black bar, solid fill all qualify).
    """
    try:
        H, W = image_np.shape[0], image_np.shape[1]
        x1, y1, x2, y2 = bbox
        xa, xb = int(x1 * W), int(x2 * W)
        ya, yb = int(y1 * H), int(y2 * H)
        if xb <= xa or yb <= ya:
            return True  # degenerate -> drop
        # inset 10% to ignore anti-aliased borders / drop shadows / 1px frames
        iw, ih = xb - xa, yb - ya
        ix, iy = int(iw * 0.1), int(ih * 0.1)
        xa, xb, ya, yb = xa + ix, xb - ix, ya + iy, yb - iy
        if xb <= xa or yb <= ya:
            return False
        patch = image_np[ya:yb, xa:xb, :3]
        if patch.size == 0:
            return False
        std = float(patch.reshape(-1, patch.shape[-1]).std(axis=0).max())
        return std < std_thresh
    except Exception:
        return False


def detect_yolo_mius_raw(
    image: Image.Image,
    weights: Optional[str] = None,
    conf: float = 0.15,
    imgsz: int = 1280,
) -> Optional[tuple[list[MiuDict], np.ndarray]]:
    """YOLO boxes only (no pentagonal attach). See ``detect_yolo_mius`` for return semantics."""
    global _YOLO_LOAD_ERROR, _YOLO_IMPORT_OK
    weights = weights or _default_omniparser_icon_weights()
    try:
        model = _get_yolo_model(weights)
    except Exception as exc:
        _YOLO_IMPORT_OK = False
        if _YOLO_LOAD_ERROR != repr(exc):
            _YOLO_LOAD_ERROR = repr(exc)
            _log.warning(
                "In-process YOLO load failed (weights=%s, %r); "
                "set OMINI_DETECT_PYTHON to a venv with ultralytics+cv2 to use the subprocess path.",
                weights, exc,
            )
        return None

    width, height = image.size
    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb)
    solid_std = float(os.environ.get("OMNIPARSER_SOLID_STD", "6.0"))

    try:
        max_edge = int(os.environ.get("OMNIPARSER_ICON_MAX_EDGE", os.environ.get("MIU_YOLO_MAX_EDGE", "768")))
    except ValueError:
        max_edge = 768
    detect_image = image_rgb
    detect_width, detect_height = width, height
    if max_edge > 0 and max(width, height) > max_edge:
        scale = float(max_edge) / float(max(width, height))
        detect_width = max(1, int(round(width * scale)))
        detect_height = max(1, int(round(height * scale)))
        detect_width = _snap_yolo_stride(detect_width)
        detect_height = _snap_yolo_stride(detect_height)
        detect_image = image_rgb.resize((detect_width, detect_height), Image.BILINEAR)

    if max_edge > 0 and (detect_width != width or detect_height != height):
        effective_imgsz: int | tuple[int, int] = (detect_height, detect_width)
    else:
        env_imgsz = os.environ.get("OMNIPARSER_ICON_IMGSZ", "").strip()
        if env_imgsz:
            try:
                effective_imgsz = int(env_imgsz)
            except ValueError:
                effective_imgsz = (
                    _snap_yolo_stride(detect_height),
                    _snap_yolo_stride(detect_width),
                )
        elif max_edge <= 0:
            # Full-res: native HxW snapped to YOLO stride (avoids Ultralytics stdout warnings).
            effective_imgsz = (
                _snap_yolo_stride(detect_height),
                _snap_yolo_stride(detect_width),
            )
        else:
            effective_imgsz = imgsz

    iou_box = float(os.environ.get("OMNIPARSER_ICON_IOU", "0.1"))
    predict_kwargs = {"conf": conf, "iou": iou_box, "imgsz": effective_imgsz, "verbose": False}
    device = _yolo_predict_device()
    if device is not None:
        predict_kwargs["device"] = device
    if _yolo_use_half():
        predict_kwargs["half"] = True
    import torch  # type: ignore
    with torch.inference_mode():
        results = model.predict(detect_image, **predict_kwargs)
    mius: list[MiuDict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, classes):
            nx1, ny1, nx2, ny2 = (
                float(x1) / max(detect_width, 1),
                float(y1) / max(detect_height, 1),
                float(x2) / max(detect_width, 1),
                float(y2) / max(detect_height, 1),
            )
            bw = max(0.0, nx2 - nx1)
            bh = max(0.0, ny2 - ny1)
            if bw * bh < 0.00002:
                continue
            # drop pure solid-color false positives (blank bg, white cards) — cheap per-box
            # std check; official OmniParser uses the same filter (OMNIPARSER_SOLID_STD=0 to skip).
            if solid_std > 0 and _op_box_is_solid((nx1, ny1, nx2, ny2), image_np, solid_std):
                continue
            aspect = bw / max(bh, 1e-6)
            # Labeling hints only (not upstream OmniParser types); prefilter must not drop by type.
            comp_type = "text_like" if aspect > 3.5 else "icon_button"
            if bw * bh > 0.04:
                comp_type = "panel"
            mius.append(
                {
                    "box": _normalize_box(nx1, ny1, nx2, ny2),
                    "type": comp_type,
                    "text": "",
                    "source": "yolo",
                    "det_score": float(score),
                    "class_id": float(cls_id),
                }
            )
    return mius, image_np


def detect_yolo_mius(
    image: Image.Image,
    weights: Optional[str] = None,
    conf: float = 0.15,
    imgsz: int = 1280,
) -> Optional[list[MiuDict]]:
    """Run in-process YOLO detection + pentagonal attach (legacy / viz scripts).

    Return semantics (v5):
      * ``[...]`` (possibly empty) when the model loaded and produced a result
        (empty == "this screen genuinely has no icons");
      * ``None`` ONLY when in-process import/load failed and the caller should
        try the subprocess path.
    """
    raw = detect_yolo_mius_raw(image, weights=weights, conf=conf, imgsz=imgsz)
    if raw is None:
        return None
    mius, image_np = raw
    return _attach_pentagonal_interaction_energy(mius, image_np)


def _fast_downscale_for_detect(
    image: Image.Image,
    *,
    max_edge_env: str = "FAST_MIU_MAX_EDGE",
    fallback_env: str = "CANNY_MIU_MAX_EDGE",
) -> tuple[Image.Image, np.ndarray, int, int]:
    """Downscale RGB PIL + numpy for fast classical detectors (boxes stay normalized)."""
    width, height = image.size
    image_rgb = image.convert("RGB")
    try:
        max_edge = int(os.environ.get(max_edge_env, os.environ.get(fallback_env, "384")))
    except ValueError:
        max_edge = 384
    if max_edge > 0 and max(width, height) > max_edge:
        scale = float(max_edge) / float(max(width, height))
        proc_w = max(1, int(round(width * scale)))
        proc_h = max(1, int(round(height * scale)))
        image_rgb = image_rgb.resize((proc_w, proc_h), Image.BILINEAR)
    image_np = np.asarray(image_rgb)
    proc_h, proc_w = int(image_np.shape[0]), int(image_np.shape[1])
    return image_rgb, image_np, proc_w, proc_h


_LAST_DETECT_TIMINGS_MS: dict[str, float] = {}


def pop_last_detect_timings() -> dict[str, float]:
    """Sub-stage detect timings from the most recent omniparser detect call."""
    global _LAST_DETECT_TIMINGS_MS
    out = dict(_LAST_DETECT_TIMINGS_MS)
    _LAST_DETECT_TIMINGS_MS = {}
    return out


@dataclass(frozen=True)
class _AuxDetectContext:
    aux_np: np.ndarray
    proc_w: int
    proc_h: int
    gray: np.ndarray


def _blob_aux_use_full_res() -> bool:
    return os.environ.get("BLOB_AUX_USE_FULL_RES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_aux_detect_context(image: Image.Image, image_np: np.ndarray) -> _AuxDetectContext:
    """Single downscale + shared grayscale for all classical-CV aux collectors."""
    import cv2  # type: ignore

    if _blob_aux_use_full_res():
        aux_np = image_np
        proc_h, proc_w = int(aux_np.shape[0]), int(aux_np.shape[1])
    else:
        _rgb, aux_np, proc_w, proc_h = _fast_downscale_for_detect(
            image, max_edge_env="BLOB_AUX_MAX_EDGE", fallback_env="FAST_MIU_MAX_EDGE"
        )
    gray = cv2.cvtColor(aux_np, cv2.COLOR_RGB2GRAY)
    return _AuxDetectContext(aux_np=aux_np, proc_w=int(proc_w), proc_h=int(proc_h), gray=gray)


def _module_a_detect_parallel_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_DETECT_PARALLEL", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _module_a_cv_parallel_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_CV_PARALLEL", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


@lru_cache(maxsize=32)
def _morph_kernel_rect(kx: int, ky: int) -> np.ndarray:
    import cv2  # type: ignore

    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kx)), max(1, int(ky))))


@lru_cache(maxsize=32)
def _morph_kernel_ellipse(k: int) -> np.ndarray:
    import cv2  # type: ignore

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, int(k)), max(1, int(k))))


def _gui_miu_filter_thresholds() -> dict:
    return {
        "solid_std": float(os.environ.get("OMNIPARSER_SOLID_STD", "6.0")),
        "min_area": float(os.environ.get("CANNY_MIU_MIN_AREA", "0.0004")),
        "max_area": float(os.environ.get("CANNY_MIU_MAX_AREA", "0.40")),
        "max_aspect": float(os.environ.get("CANNY_MIU_MAX_ASPECT", "20.0")),
        "max_boxes": int(os.environ.get("CANNY_MIU_MAX_BOXES", "160")),
    }


def _norm_rect_from_pixels(x: int, y: int, w: int, h: int, proc_w: int, proc_h: int) -> tuple[float, float, float, float]:
    nx1 = float(x) / max(proc_w, 1)
    ny1 = float(y) / max(proc_h, 1)
    nx2 = float(x + w) / max(proc_w, 1)
    ny2 = float(y + h) / max(proc_h, 1)
    return nx1, ny1, nx2, ny2


def _passes_gui_miu_geom(
    nx1: float,
    ny1: float,
    nx2: float,
    ny2: float,
    *,
    image_np,
    solid_std: float,
    min_area: float,
    max_area: float,
    max_aspect: float,
) -> bool:
    bw = max(0.0, nx2 - nx1)
    bh = max(0.0, ny2 - ny1)
    area = bw * bh
    if area < min_area or area > max_area:
        return False
    aspect = bw / max(bh, 1e-6)
    if aspect > max_aspect or aspect < (1.0 / max_aspect):
        return False
    if solid_std > 0 and _op_box_is_solid((nx1, ny1, nx2, ny2), image_np, solid_std):
        return False
    return True


def _miu_dict_from_norm_box(
    nx1: float,
    ny1: float,
    nx2: float,
    ny2: float,
    *,
    source: str,
    image_np,
    solid_std: float,
    min_area: float,
    max_area: float,
    max_aspect: float,
) -> Optional[MiuDict]:
    if not _passes_gui_miu_geom(
        nx1, ny1, nx2, ny2,
        image_np=image_np,
        solid_std=solid_std,
        min_area=min_area,
        max_area=max_area,
        max_aspect=max_aspect,
    ):
        return None
    bw = max(0.0, nx2 - nx1)
    bh = max(0.0, ny2 - ny1)
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    comp_type = "text_like" if aspect > 3.5 else "icon_button"
    if area > 0.04:
        comp_type = "panel"
    det_score = float(min(1.0, 0.30 + 0.70 * min(1.0, area / 0.02)))
    return {
        "box": _normalize_box(nx1, ny1, nx2, ny2),
        "type": comp_type,
        "text": "",
        "source": source,
        "det_score": det_score,
        "class_id": 0.0,
    }


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)


def _nms_merge_mius(mius: list[MiuDict], iou_threshold: float = 0.5) -> list[MiuDict]:
    """Greedy NMS on normalized boxes; prefer higher det_score / larger area."""
    if not mius:
        return []

    def _rank(m: MiuDict) -> tuple[float, float]:
        box = m.get("box") or (0, 0, 0, 0)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        return float(m.get("det_score", 0.0) or 0.0), area

    ordered = sorted(mius, key=_rank, reverse=True)
    kept: list[MiuDict] = []
    for cand in ordered:
        box = cand.get("box")
        if not box or len(box) < 4:
            continue
        cb = tuple(float(v) for v in box[:4])
        if any(_box_iou(cb, tuple(float(v) for v in k["box"][:4])) >= iou_threshold for k in kept if k.get("box")):
            continue
        kept.append(cand)
    return kept


def _collect_canny_boxes(image_np, proc_w: int, proc_h: int) -> list[MiuDict]:
    """Improved Canny+morph: RECT+ELLIPSE close, RETR_TREE contours."""
    import cv2  # type: ignore

    filt = _gui_miu_filter_thresholds()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_k = int(os.environ.get("CANNY_MIU_BLUR_KSIZE", "5"))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    lo = int(os.environ.get("CANNY_MIU_THRESH_LO", "50"))
    hi = int(os.environ.get("CANNY_MIU_THRESH_HI", "150"))
    edges = cv2.Canny(blurred, lo, hi)
    kx = int(os.environ.get("CANNY_MIU_MORPH_KSIZE", "9"))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, kx))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, kx))
    closed_rect = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_rect)
    closed_ellipse = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_ellipse)
    merged_edges = cv2.bitwise_or(closed_rect, closed_ellipse)
    contours, _ = cv2.findContours(merged_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mius: list[MiuDict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 1 or h <= 1:
            continue
        key = (x, y, w, h)
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        nx1, ny1, nx2, ny2 = _norm_rect_from_pixels(x, y, w, h, proc_w, proc_h)
        miu = _miu_dict_from_norm_box(
            nx1, ny1, nx2, ny2,
            source="canny",
            image_np=image_np,
            **{k: filt[k] for k in ("solid_std", "min_area", "max_area", "max_aspect")},
        )
        if miu is not None:
            mius.append(miu)
        if len(mius) >= filt["max_boxes"]:
            break
    return mius


def _collect_blob_boxes(
    image_np,
    proc_w: int,
    proc_h: int,
    *,
    gray: np.ndarray | None = None,
) -> list[MiuDict]:
    """Adaptive-threshold CC + MSER + circular/ellipse contours."""
    import cv2  # type: ignore

    filt = _gui_miu_filter_thresholds()
    if gray is None:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_k = int(os.environ.get("BLOB_MIU_BLUR_KSIZE", "5"))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    mius: list[MiuDict] = []
    seen: set[tuple[int, int, int, int]] = set()

    def _add_rect(x: int, y: int, w: int, h: int, source: str = "blob") -> None:
        if w <= 1 or h <= 1:
            return
        key = (x, y, w, h)
        if key in seen:
            return
        seen.add(key)
        nx1, ny1, nx2, ny2 = _norm_rect_from_pixels(x, y, w, h, proc_w, proc_h)
        miu = _miu_dict_from_norm_box(
            nx1, ny1, nx2, ny2,
            source=source,
            image_np=image_np,
            **{k: filt[k] for k in ("solid_std", "min_area", "max_area", "max_aspect")},
        )
        if miu is not None:
            mius.append(miu)

    block = int(os.environ.get("BLOB_MIU_ADAPTIVE_BLOCK", "11"))
    if block % 2 == 0:
        block += 1
    c_val = int(os.environ.get("BLOB_MIU_ADAPTIVE_C", "2"))
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c_val,
    )
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(adaptive, connectivity=8)
    for label in range(1, num_labels):
        x, y, w, h, area_px = stats[label]
        if area_px <= 1:
            continue
        _add_rect(int(x), int(y), int(w), int(h), "blob_cc")

    try:
        min_mser = int(os.environ.get("BLOB_MIU_MSER_MIN", "60"))
        max_mser = int(os.environ.get("BLOB_MIU_MSER_MAX", "8000"))
        mser = cv2.MSER_create(_delta=5, _min_area=min_mser, _max_area=max_mser)
        regions, _ = mser.detectRegions(blurred)
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            _add_rect(x, y, w, h, "blob_mser")
    except Exception:
        pass

    circ_min = float(os.environ.get("BLOB_MIU_MIN_CIRCULARITY", "0.55"))
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        if area_px < 16.0:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0.0:
            continue
        circularity = 4.0 * np.pi * area_px / (peri * peri)
        if circularity >= circ_min:
            x, y, w, h = cv2.boundingRect(cnt)
            _add_rect(x, y, w, h, "blob_circle")
        elif len(cnt) >= 5:
            try:
                _ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (ma, mi), _angle = _ellipse
                if mi > 0 and ma / mi < filt["max_aspect"]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    _add_rect(x, y, w, h, "blob_ellipse")
            except cv2.error:
                pass

    max_boxes = int(os.environ.get("BLOB_MIU_MAX_BOXES", filt["max_boxes"]))
    return mius[:max_boxes]


def _collect_hough_circle_mius(
    image_np,
    proc_w: int,
    proc_h: int,
    *,
    gray: np.ndarray | None = None,
) -> list[MiuDict]:
    """HoughCircles for round avatars / radio / FAB when YOLO misses."""
    import cv2  # type: ignore

    filt = _gui_miu_filter_thresholds()
    if gray is None:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_k = int(os.environ.get("HOUGH_CIRCLE_BLUR", "9"))
    if blur_k % 2 == 0:
        blur_k += 1
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 2)
    min_dim = min(proc_w, proc_h)
    min_r = max(4, int(min_dim * float(os.environ.get("HOUGH_CIRCLE_MIN_R_RATIO", "0.012"))))
    max_r = max(min_r + 2, int(min_dim * float(os.environ.get("HOUGH_CIRCLE_MAX_R_RATIO", "0.075"))))
    param2 = int(os.environ.get("HOUGH_CIRCLE_PARAM2", "26"))
    dp = float(os.environ.get("HOUGH_CIRCLE_DP", "1.2"))
    min_dist = max(min_r * 2, int(os.environ.get("HOUGH_CIRCLE_MIN_DIST", str(min_r * 2))))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=100,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    mius: list[MiuDict] = []
    if circles is None:
        return mius
    for cx, cy, r in np.round(circles[0]).astype(int):
        x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
        w, h = min(proc_w, int(cx + r)) - x1, min(proc_h, int(cy + r)) - y1
        if w <= 1 or h <= 1:
            continue
        nx1, ny1, nx2, ny2 = _norm_rect_from_pixels(x1, y1, w, h, proc_w, proc_h)
        miu = _miu_dict_from_norm_box(
            nx1, ny1, nx2, ny2,
            source="hough_circle",
            image_np=image_np,
            **{k: filt[k] for k in ("solid_std", "min_area", "max_area", "max_aspect")},
        )
        if miu is not None:
            miu["shape"] = "circle"
            miu["ellipse"] = {
                "cx": float(cx) / max(proc_w, 1),
                "cy": float(cy) / max(proc_h, 1),
                "axis_major": float(r * 2) / max(proc_w, 1),
                "axis_minor": float(r * 2) / max(proc_h, 1),
                "angle_deg": 0.0,
            }
            mius.append(miu)
    cap = int(os.environ.get("HOUGH_CIRCLE_MAX", "32"))
    return mius[:cap]


def _collect_edge_ellipse_mius(
    image_np,
    proc_w: int,
    proc_h: int,
    *,
    gray: np.ndarray | None = None,
) -> list[MiuDict]:
    """Canny edge contours → fitEllipse for oval/circular widget boundaries."""
    import cv2  # type: ignore

    filt = _gui_miu_filter_thresholds()
    if gray is None:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_k = int(os.environ.get("EDGE_ELLIPSE_BLUR", "5"))
    if blur_k % 2 == 0:
        blur_k += 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    lo = int(os.environ.get("EDGE_ELLIPSE_CANNY_LO", "40"))
    hi = int(os.environ.get("EDGE_ELLIPSE_CANNY_HI", "120"))
    edges = cv2.Canny(blurred, lo, hi)
    kx = int(os.environ.get("EDGE_ELLIPSE_MORPH_K", "5"))
    kernel = _morph_kernel_ellipse(kx)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circ_min = float(os.environ.get("EDGE_ELLIPSE_MIN_CIRCULARITY", "0.40"))
    ecc_max = float(os.environ.get("EDGE_ELLIPSE_MAX_ECCENTRICITY", "0.93"))
    mius: list[MiuDict] = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area_px = float(cv2.contourArea(cnt))
        if area_px < 32.0:
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0.0:
            continue
        circularity = 4.0 * np.pi * area_px / (peri * peri)
        if circularity < circ_min:
            continue
        try:
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if mi < 4.0 or ma < 4.0:
            continue
        if ma / max(mi, 1e-6) > filt["max_aspect"]:
            continue
        ecc = math.sqrt(max(0.0, 1.0 - (mi / ma) ** 2))
        if ecc > ecc_max:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 1 or h <= 1:
            continue
        nx1, ny1, nx2, ny2 = _norm_rect_from_pixels(x, y, w, h, proc_w, proc_h)
        miu = _miu_dict_from_norm_box(
            nx1,
            ny1,
            nx2,
            ny2,
            source="edge_ellipse",
            image_np=image_np,
            **{k: filt[k] for k in ("solid_std", "min_area", "max_area", "max_aspect")},
        )
        if miu is None:
            continue
        miu["shape"] = "ellipse"
        miu["ellipse"] = {
            "cx": float(cx) / max(proc_w, 1),
            "cy": float(cy) / max(proc_h, 1),
            "axis_major": float(ma) / max(proc_w, 1),
            "axis_minor": float(mi) / max(proc_h, 1),
            "angle_deg": float(angle),
        }
        mius.append(miu)
    cap = int(os.environ.get("EDGE_ELLIPSE_MAX", "24"))
    return mius[:cap]


def _yolo_circular_aux_enabled() -> bool:
    return os.environ.get("GEOPACT_YOLO_CIRCULAR_AUX", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _aggregate_yolo_on_circular_boundary(
    yolo_mius: list[MiuDict],
    circular_aux: list[MiuDict],
) -> list[MiuDict]:
    """Merge multiple YOLO boxes that lie on a detected circle/ellipse into one proposal."""
    min_cluster = int(os.environ.get("CIRCULAR_YOLO_MIN_CLUSTER", "2"))
    ring_tol_ratio = float(os.environ.get("CIRCULAR_YOLO_RING_TOL", "0.40"))
    proposals: list[MiuDict] = []
    circular_sources = {"hough_circle", "blob_circle", "blob_ellipse", "edge_ellipse"}
    for circ in circular_aux:
        if str(circ.get("source", "")) not in circular_sources:
            continue
        cbox = circ.get("box")
        if not cbox or len(cbox) < 4:
            continue
        cx = (float(cbox[0]) + float(cbox[2])) * 0.5
        cy = (float(cbox[1]) + float(cbox[3])) * 0.5
        rw = max(0.0, float(cbox[2]) - float(cbox[0])) * 0.5
        rh = max(0.0, float(cbox[3]) - float(cbox[1])) * 0.5
        r = max(rw, rh, 1e-6)
        ring_tol = ring_tol_ratio * r
        hits: list[MiuDict] = []
        for ym in yolo_mius:
            ybox = ym.get("box")
            if not ybox or len(ybox) < 4:
                continue
            ycx = (float(ybox[0]) + float(ybox[2])) * 0.5
            ycy = (float(ybox[1]) + float(ybox[3])) * 0.5
            dist = math.hypot(ycx - cx, ycy - cy)
            if dist <= r + ring_tol and dist >= max(0.0, r - ring_tol * 2.0):
                hits.append(ym)
            elif dist <= r * 0.85:
                hits.append(ym)
        if len(hits) < min_cluster:
            continue
        best_score = max(float(h.get("det_score", 0.0) or 0.0) for h in hits)
        merged = dict(circ)
        merged["source"] = "circular_aggregate"
        merged["det_score"] = float(best_score)
        merged["shape"] = circ.get("shape") or "circle"
        merged["n_yolo_cluster"] = len(hits)
        proposals.append(merged)
    return proposals


def _is_yolo_origin_miu(miu: MiuDict) -> bool:
    src = str(miu.get("source", ""))
    return src == "yolo" or src.startswith("box_yolo")


def _yolo_ocr_aux_enabled() -> bool:
    return os.environ.get("GEOPACT_YOLO_OCR_AUX", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _yolo_stripe_aux_enabled() -> bool:
    return os.environ.get("GEOPACT_YOLO_STRIPE_AUX", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _collect_circular_aux_from_ctx(ctx: _AuxDetectContext) -> list[MiuDict]:
    """Circle/ellipse/Hough aux from a shared aux context."""
    blob_all = _collect_blob_boxes(ctx.aux_np, ctx.proc_w, ctx.proc_h, gray=ctx.gray)
    circular = [dict(m) for m in blob_all if str(m.get("source", "")) in {"blob_circle", "blob_ellipse"}]
    hough = [dict(m) for m in _collect_hough_circle_mius(ctx.aux_np, ctx.proc_w, ctx.proc_h, gray=ctx.gray)]
    merged = _nms_merge_mius(
        circular + hough,
        iou_threshold=float(os.environ.get("CIRCULAR_AUX_INTERNAL_IOU", "0.5")),
    )
    aux_score = float(os.environ.get("BLOB_AUX_DET_SCORE", "0.35"))
    for m in merged:
        m["det_score"] = aux_score
        m.setdefault("type", "icon")
        m["wrapper"] = "circular_aux"
    max_aux = int(os.environ.get("YOLO_BLOB_AUX_MAX", "48"))
    return merged[:max_aux]


def _collect_cv_aux_bundle(
    ctx: _AuxDetectContext,
    *,
    timings_out: dict[str, float] | None = None,
) -> dict[str, list[MiuDict]]:
    """Collect circular / edge-ellipse / stripe aux parts (optionally in parallel)."""
    circular_on = _yolo_circular_aux_enabled()
    stripe_on = _yolo_stripe_aux_enabled()

    def _circular_part() -> list[MiuDict]:
        return _collect_circular_aux_from_ctx(ctx) if circular_on else []

    def _edge_part() -> list[MiuDict]:
        if not circular_on:
            return []
        return _collect_edge_ellipse_mius(ctx.aux_np, ctx.proc_w, ctx.proc_h, gray=ctx.gray)

    def _stripe_part() -> list[MiuDict]:
        if not stripe_on:
            return []
        t0 = time.perf_counter()
        out = _collect_stripe_row_mius(ctx.aux_np, ctx.proc_w, ctx.proc_h, gray=ctx.gray)
        if timings_out is not None:
            timings_out["stripe_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return out

    t_cv = time.perf_counter()
    if _module_a_cv_parallel_enabled() and (circular_on or stripe_on):
        circular_merged: list[MiuDict] = []
        edge_ell: list[MiuDict] = []
        stripe_mius: list[MiuDict] = []
        tasks: list[tuple[str, Callable[[], list[MiuDict]]]] = []
        if circular_on:
            tasks.append(("circular", _circular_part))
            tasks.append(("edge", _edge_part))
        if stripe_on:
            tasks.append(("stripe", _stripe_part))
        with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as pool:
            futs = {name: pool.submit(fn) for name, fn in tasks}
            if "circular" in futs:
                circular_merged = futs["circular"].result()
            if "edge" in futs:
                edge_ell = futs["edge"].result()
            if "stripe" in futs:
                stripe_mius = futs["stripe"].result()
    else:
        circular_merged = _circular_part()
        edge_ell = _edge_part()
        stripe_mius = _stripe_part()

    if timings_out is not None:
        timings_out.setdefault("stripe_ms", 0.0)
        timings_out["cv_aux_ms"] = round((time.perf_counter() - t_cv) * 1000.0, 3)

    return {
        "circular_merged": circular_merged,
        "edge_ell": edge_ell,
        "stripe_mius": stripe_mius,
    }


def _merge_yolo_with_cv_aux(
    yolo_mius: list[MiuDict],
    cv_bundle: dict[str, list[MiuDict]],
) -> list[MiuDict]:
    """NMS-fuse YOLO boxes with CV aux (circular, edge, stripe)."""
    base = list(yolo_mius)
    circular_merged = list(cv_bundle.get("circular_merged") or [])
    edge_ell = list(cv_bundle.get("edge_ell") or [])
    stripe_mius = list(cv_bundle.get("stripe_mius") or [])

    if _yolo_circular_aux_enabled():
        yolo_only = [m for m in base if _is_yolo_origin_miu(m)]
        aux_internal_iou = float(os.environ.get("CIRCULAR_AUX_INTERNAL_IOU", "0.5"))
        aux_merged = _nms_merge_mius(circular_merged + edge_ell, iou_threshold=aux_internal_iou)
        aggregated = _aggregate_yolo_on_circular_boundary(yolo_only, aux_merged)
        fusion_iou = float(os.environ.get("YOLO_CIRCULAR_FUSION_IOU", "0.45"))
        out = _nms_merge_mius(base + aux_merged + aggregated, iou_threshold=fusion_iou)
    else:
        out = base

    if stripe_mius:
        fusion_iou = float(os.environ.get("YOLO_STRIPE_FUSION_IOU", "0.4"))
        out = _nms_merge_mius(out + stripe_mius, iou_threshold=fusion_iou)
    return out


def _augment_with_circular_regions(
    mius: list[MiuDict],
    image: Image.Image,
    image_np: np.ndarray,
    *,
    ctx: _AuxDetectContext | None = None,
) -> list[MiuDict]:
    """Run blob/Hough/edge-ellipse aux; NMS-fuse with existing MIUs (YOLO may be empty)."""
    if not _yolo_circular_aux_enabled():
        return list(mius)
    if ctx is None:
        ctx = _build_aux_detect_context(image, image_np)
    cv_bundle = _collect_cv_aux_bundle(ctx)
    return _merge_yolo_with_cv_aux(mius, cv_bundle)


def _augment_yolo_with_circular_regions(
    yolo_mius: list[MiuDict],
    image: Image.Image,
    image_np: np.ndarray,
) -> list[MiuDict]:
    """Legacy alias for circular aux after YOLO."""
    return _augment_with_circular_regions(yolo_mius, image, image_np)


def _augment_with_ocr_regions(mius: list[MiuDict], image: Image.Image) -> list[MiuDict]:
    """Merge OCR text boxes after YOLO+circular; YOLO wins on IoU overlap."""
    if not _yolo_ocr_aux_enabled():
        return list(mius)
    ocr_mius, _backend = resolve_ocr_mius(image)
    if not ocr_mius:
        return list(mius)
    aux_score = float(os.environ.get("OCR_AUX_DET_SCORE", "0.25"))
    for m in ocr_mius:
        m["det_score"] = aux_score
        m["source"] = "ocr"
        m.setdefault("type", "text")
    fusion_iou = float(os.environ.get("YOLO_OCR_FUSION_IOU", "0.35"))
    return _nms_merge_mius(list(mius) + ocr_mius, iou_threshold=fusion_iou)


def _collect_stripe_row_mius(
    image_np,
    proc_w: int,
    proc_h: int,
    *,
    gray: np.ndarray | None = None,
) -> list[MiuDict]:
    """Horizontal list-row / search-bar / slider-track proposals via morph open."""
    import cv2  # type: ignore

    if gray is None:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    block = max(3, int(min(proc_w, proc_h) * 0.02) | 1)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        5,
    )
    kx = max(15, int(proc_w * 0.08))
    kernel = _morph_kernel_rect(kx, 3)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w_ratio = float(os.environ.get("STRIPE_ROW_MIN_W_RATIO", "0.15"))
    max_h_ratio = float(os.environ.get("STRIPE_ROW_MAX_H_RATIO", "0.08"))
    min_aspect = float(os.environ.get("STRIPE_ROW_MIN_ASPECT", "4.0"))
    aux_score = float(os.environ.get("STRIPE_ROW_DET_SCORE", "0.30"))
    mius: list[MiuDict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        nw = w / max(proc_w, 1)
        nh = h / max(proc_h, 1)
        if nw < min_w_ratio or nh > max_h_ratio:
            continue
        aspect = nw / max(nh, 1e-6)
        if aspect < min_aspect:
            continue
        mius.append(
            {
                "box": (
                    x / proc_w,
                    y / proc_h,
                    (x + w) / proc_w,
                    (y + h) / proc_h,
                ),
                "source": "stripe_row",
                "type": "list_row",
                "det_score": aux_score,
            }
        )
    cap = int(os.environ.get("STRIPE_ROW_MAX", "32"))
    return mius[:cap]


def _augment_with_stripe_regions(
    mius: list[MiuDict],
    image: Image.Image,
    image_np: np.ndarray,
    *,
    ctx: _AuxDetectContext | None = None,
) -> list[MiuDict]:
    if not _yolo_stripe_aux_enabled():
        return list(mius)
    if ctx is None:
        ctx = _build_aux_detect_context(image, image_np)
    stripe_mius = _collect_stripe_row_mius(ctx.aux_np, ctx.proc_w, ctx.proc_h, gray=ctx.gray)
    if not stripe_mius:
        return list(mius)
    fusion_iou = float(os.environ.get("YOLO_STRIPE_FUSION_IOU", "0.4"))
    return _nms_merge_mius(list(mius) + stripe_mius, iou_threshold=fusion_iou)


def _apply_module_a_detection_augmentations(
    mius: list[MiuDict],
    image: Image.Image,
    image_np: np.ndarray,
) -> list[MiuDict]:
    """YOLO (or OCR-only) base → circular → OCR fusion → stripe aux."""
    ctx = _build_aux_detect_context(image, image_np)
    out = mius
    if _yolo_circular_aux_enabled() or _yolo_stripe_aux_enabled():
        cv_bundle = _collect_cv_aux_bundle(ctx)
        out = _merge_yolo_with_cv_aux(out, cv_bundle)
    out = _augment_with_ocr_regions(out, image)
    return out


def detect_canny_morph_mius_raw(image: Image.Image) -> tuple[list[MiuDict], np.ndarray]:
    """Canny boxes without pentagonal attach (for fusion)."""
    _rgb, image_np, proc_w, proc_h = _fast_downscale_for_detect(image)
    return _collect_canny_boxes(image_np, proc_w, proc_h), image_np


def detect_blob_mius_raw(image: Image.Image) -> tuple[list[MiuDict], np.ndarray]:
    """Blob boxes without pentagonal attach (for fusion)."""
    _rgb, image_np, proc_w, proc_h = _fast_downscale_for_detect(image)
    return _collect_blob_boxes(image_np, proc_w, proc_h), image_np


def detect_blob_mius(image: Image.Image) -> list[MiuDict]:
    """Blob-only fast GUI detector (adaptive CC + MSER + circular blobs)."""
    mius, image_np = detect_blob_mius_raw(image)
    return _attach_pentagonal_interaction_energy(mius, image_np)


def detect_fusion_fast_mius_raw(image: Image.Image) -> tuple[list[MiuDict], np.ndarray]:
    """Blob + Canny → NMS (no pentagonal attach)."""
    _rgb, image_np, proc_w, proc_h = _fast_downscale_for_detect(image)
    canny_mius = _collect_canny_boxes(image_np, proc_w, proc_h)
    blob_mius = _collect_blob_boxes(image_np, proc_w, proc_h)
    iou = float(os.environ.get("FUSION_MIU_NMS_IOU", "0.5"))
    merged = _nms_merge_mius(canny_mius + blob_mius, iou_threshold=iou)
    cap = int(os.environ.get("FUSION_MIU_MAX_BOXES", "200"))
    return merged[:cap], image_np


def detect_fusion_fast_mius(image: Image.Image) -> list[MiuDict]:
    """Blob + improved Canny → NMS → single pentagonal attach."""
    mius, image_np = detect_fusion_fast_mius_raw(image)
    return _attach_pentagonal_interaction_energy(mius, image_np)


def detect_canny_morph_mius(image: Image.Image) -> list[MiuDict]:
    """Ultra-light traditional-CV GUI box extractor (Canny + morphological close).

    Improved v2: MORPH_RECT + MORPH_ELLIPSE dual close, RETR_TREE contours.
    Boxes feed the shared pentagonal attribute calculator unchanged.
    """
    mius, image_np = detect_canny_morph_mius_raw(image)
    return _attach_pentagonal_interaction_energy(mius, image_np)


def _tag_omniparser_wrapper_mius(mius: list[MiuDict]) -> list[MiuDict]:
    """Preserve detector ``source``; record wrapper separately (A3)."""
    for miu in mius:
        miu.setdefault("detector_source", str(miu.get("source") or "yolo"))
        miu["wrapper"] = "omniparser_icon"
    return mius


def detect_omniparser_icon_mius_raw(
    image: Image.Image,
    conf: float = 0.05,
    imgsz: int = 1280,
) -> Optional[tuple[list[MiuDict], np.ndarray]]:
    """OmniParser-v2 YOLO boxes only (no pentagonal attach)."""
    global _LAST_DETECT_TIMINGS_MS
    _LAST_DETECT_TIMINGS_MS = {}

    conf = float(os.environ.get("OMNIPARSER_ICON_CONF", conf))
    try:
        imgsz = int(os.environ.get("OMNIPARSER_ICON_IMGSZ", str(imgsz)))
    except ValueError:
        pass
    weights = os.environ.get(
        "OMNIPARSER_ICON_WEIGHTS",
        _default_omniparser_icon_weights(),
    )
    raw = detect_yolo_mius_raw(image, weights=weights, conf=conf, imgsz=imgsz)
    if raw is None:
        return None
    mius, image_np = raw
    mius = _apply_module_a_detection_augmentations(mius, image, image_np)
    return _tag_omniparser_wrapper_mius(mius), image_np


def detect_ocr_only_mius_raw(image: Image.Image) -> tuple[list[MiuDict], np.ndarray]:
    """OCR-only detection stack (Stack C): OCR → circular aux; no YOLO."""
    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb)
    ocr_mius, _backend = resolve_ocr_mius(image)
    aux_score = float(os.environ.get("OCR_AUX_DET_SCORE", "0.25"))
    for m in ocr_mius:
        m["det_score"] = aux_score
        m["source"] = "ocr"
        m.setdefault("type", "text")
    mius = _apply_module_a_detection_augmentations(ocr_mius, image, image_np)
    return _tag_omniparser_wrapper_mius(mius), image_np


def _circular_aux_mius_from_image(image: Image.Image, *, image_np=None) -> list[MiuDict]:
    """Circle/ellipse/Hough aux — complements YOLO on round avatars, radio, FAB."""
    if image_np is None:
        image_rgb = image.convert("RGB")
        image_np = np.asarray(image_rgb)
    ctx = _build_aux_detect_context(image, image_np)
    return _collect_circular_aux_from_ctx(ctx)


def _circular_blob_mius_from_image(image: Image.Image, *, image_np=None) -> list[MiuDict]:
    """Alias for circular aux (blob + Hough)."""
    return _circular_aux_mius_from_image(image, image_np=image_np)


def detect_omniparser_yolo_blob_fusion_mius_raw(
    image: Image.Image,
    conf: float = 0.05,
    imgsz: int = 1280,
) -> Optional[tuple[list[MiuDict], np.ndarray]]:
    """Legacy alias: circular aux is now wired into ``detect_omniparser_icon_mius_raw``."""
    return detect_omniparser_icon_mius_raw(image, conf=conf, imgsz=imgsz)


def detect_omniparser_icon_mius(
    image: Image.Image,
    conf: float = 0.05,
    imgsz: int = 1280,
) -> Optional[list[MiuDict]]:
    """OmniParser-v2 icon_detect YOLO weights (microsoft/OmniParser-v2.0/icon_detect/model.pt).

    Returns ``None`` only when the in-process detector cannot load
    (cv2 / ultralytics missing); otherwise returns a (possibly empty) list.
    The empty case is a genuine "no icons" detection, NOT an error -- so
    the OmniParser path no longer triggers a 25 s subprocess fallback on
    icon-free screens (v4 bug, see communication.md §12.1 #6).
    """
    raw = detect_omniparser_icon_mius_raw(image, conf=conf, imgsz=imgsz)
    if raw is None:
        return None
    mius, image_np = raw
    return _attach_pentagonal_interaction_energy(mius, image_np)


# --------------------------------------------------------------------------- #
# In-process OmniParser-v2 full pipeline (YOLO icon_detect + EasyOCR + Florence
# caption) with the 3 validated optimizations and a 3-way ablation switch.
#
# Validated on remote A800 (12 mobile imgs, see /memories/repo profiling notes):
#   * caption token 20->10            -- main speed win, denoises hallucination tail
#   * container-gate                  -- skip captioning boxes that are layout
#                                        containers (area>0.10 AND enclose>=3 boxes)
#   * crop-dedup (exact + 8x8 avg-hash) -- caption each unique glyph once
#   * KEEP full-res OCR (half/third recall collapses on 288px mobile UIs)
#   * KEEP Florence (64% of interactive icons are text-less; only label is caption)
#
# Ablation modes (env MIU_OMNIPARSER_MODE or detector string):
#   "ocr"      -- YOLO icons + OCR text, merge OCR into icons; NO captioning.
#                 Text-less icons keep content=None.
#   "florence" -- YOLO icons captioned by Florence; NO separate OCR text boxes.
#   "both"     -- full pipeline (OCR text boxes + merge + caption text-less icons).
#
# No Set-of-Mark overlay, no parsed-image, no data-fusion serialization: we
# only extract per-element key info and store it for downstream MIU scoring.
# --------------------------------------------------------------------------- #


def _op_box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _op_inter(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _op_iou(a, b) -> float:
    inter = _op_inter(a, b)
    aa, ab = _op_box_area(a), _op_box_area(b)
    union = aa + ab - inter + 1e-6
    r1 = inter / aa if aa > 0 else 0.0
    r2 = inter / ab if ab > 0 else 0.0
    return max(inter / union, r1, r2)


def _op_is_inside(a, b) -> float:
    """True when >80% of box ``a`` is covered by box ``b`` (official is_inside)."""
    return _op_inter(a, b) / max(_op_box_area(a), 1e-6) > 0.80


def _op_remove_overlap_new(icon_boxes, iou_threshold, ocr_boxes=None):
    """Faithful port of OmniParser util.remove_overlap_new (normalized coords).

    ``icon_boxes``/``ocr_boxes`` are lists of element dicts with a "bbox" key in
    normalized xyxy.  Returns merged elements carrying interactivity / content /
    source, assigning OCR text to enclosing icons (box_yolo_content_ocr) and
    leaving text-less icons as box_yolo_content_yolo for later captioning.
    """
    filtered = []
    if ocr_boxes:
        filtered.extend(ocr_boxes)
    for i, e1 in enumerate(icon_boxes):
        b1 = e1["bbox"]
        valid = True
        for j, e2 in enumerate(icon_boxes):
            if i != j and _op_iou(b1, e2["bbox"]) > iou_threshold and _op_box_area(b1) > _op_box_area(e2["bbox"]):
                valid = False
                break
        if not valid:
            continue
        if ocr_boxes:
            box_added = False
            ocr_labels = ""
            for e3 in list(ocr_boxes):
                if box_added:
                    break
                b3 = e3["bbox"]
                if _op_is_inside(b3, b1):
                    try:
                        ocr_labels += (e3.get("content") or "") + " "
                        filtered.remove(e3)
                    except ValueError:
                        continue
                elif _op_is_inside(b1, b3):
                    box_added = True
                    break
            if not box_added:
                filtered.append(
                    {
                        "type": "icon",
                        "bbox": b1,
                        "interactivity": True,
                        "content": ocr_labels.strip() or None,
                        "source": "box_yolo_content_ocr" if ocr_labels.strip() else "box_yolo_content_yolo",
                        "det_score": float(e1.get("det_score", 0.0) or 0.0),
                    }
                )
        else:
            filtered.append(
                {
                    "type": "icon",
                    "bbox": b1,
                    "interactivity": True,
                    "content": None,
                    "source": "box_yolo_content_yolo",
                    "det_score": float(e1.get("det_score", 0.0) or 0.0),
                }
            )
    return filtered


def _op_container_mask(boxes, area_thresh: float, min_children: int) -> list[bool]:
    """Flag layout-container boxes: large (area>thresh) AND enclosing >= N centers.

    Validated to lose 0 informative captions vs a pure-area cutoff (which lost
    5.9%); used to skip captioning containers, not to drop them.
    """
    n = len(boxes)
    mask = [False] * n
    for i, b in enumerate(boxes):
        if _op_box_area(b) < area_thresh:
            continue
        cnt = 0
        ai = _op_box_area(b)
        for j, o in enumerate(boxes):
            if i == j:
                continue
            if _op_box_area(o) < ai:
                cx = 0.5 * (o[0] + o[2])
                cy = 0.5 * (o[1] + o[3])
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    cnt += 1
                    if cnt >= min_children:
                        break
        if cnt >= min_children:
            mask[i] = True
    return mask


def _op_rank_ascending(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _op_pixel_box(box, image_np) -> tuple[int, int, int, int]:
    height, width = image_np.shape[0], image_np.shape[1]
    left = max(0, min(width, int(np.floor(float(box[0]) * width))))
    top = max(0, min(height, int(np.floor(float(box[1]) * height))))
    right = max(0, min(width, int(np.ceil(float(box[2]) * width))))
    bottom = max(0, min(height, int(np.ceil(float(box[3]) * height))))
    return left, top, right, bottom


def _op_gradient_entropy_score(box, image_np) -> float:
    scores = _batch_gradient_entropy_scores([box], image_np)
    return scores[0] if scores else 0.0


def _entropy_from_gray(gray: np.ndarray | None) -> float:
    if gray is None:
        return 0.0
    try:
        import cv2  # type: ignore

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        bins = np.clip(np.rint(grad), 0, 255).astype(np.int32).reshape(-1)
        hist = np.bincount(bins, minlength=256).astype(np.float64)
        total = float(hist.sum())
        if total <= 0.0:
            return 0.0
        probs = hist[hist > 0.0] / total
        entropy = float(-(probs * np.log2(probs)).sum())
        return float(max(0.0, min(1.0, entropy / 8.0)))
    except Exception:
        return 0.0


def _batch_gradient_entropy_scores(boxes, image_np) -> list[float]:
    """Batch gradient-entropy for pentagonal H term (shared cv2 path)."""
    if not boxes:
        return []
    try:
        import cv2  # type: ignore

        grays: list[np.ndarray | None] = []
        for box in boxes:
            left, top, right, bottom = _op_pixel_box(box, image_np)
            if right <= left or bottom <= top:
                grays.append(None)
                continue
            crop = image_np[top:bottom, left:right, :3]
            if crop.size == 0:
                grays.append(None)
                continue
            small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32))

        if len(grays) > 16:
            with ThreadPoolExecutor(max_workers=4) as pool:
                scores = list(pool.map(_entropy_from_gray, grays))
        else:
            scores = [_entropy_from_gray(gray) for gray in grays]
        return scores
    except Exception:
        return [_op_gradient_entropy_score_legacy(box, image_np) for box in boxes]


def _op_gradient_entropy_score_legacy(box, image_np) -> float:
    left, top, right, bottom = _op_pixel_box(box, image_np)
    if right <= left or bottom <= top:
        return 0.0
    crop = image_np[top:bottom, left:right, :3]
    if crop.size == 0:
        return 0.0
    try:
        import cv2  # type: ignore

        small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    except Exception:
        small_img = Image.fromarray(crop.astype(np.uint8)).convert("L").resize((32, 32), Image.BILINEAR)
        gray = np.asarray(small_img, dtype=np.float32)
        grad_y, grad_x = np.gradient(gray)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    bins = np.clip(np.rint(grad), 0, 255).astype(np.int32).reshape(-1)
    hist = np.bincount(bins, minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    probs = hist[hist > 0.0] / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return float(max(0.0, min(1.0, entropy / 8.0)))


def _op_boundary_contrast_raw(box, image_np, ring_px: int = 3, lab_img=None) -> float:
    left, top, right, bottom = _op_pixel_box(box, image_np)
    if right <= left or bottom <= top:
        return 0.0
    height, width = image_np.shape[0], image_np.shape[1]
    outer_left = max(0, left - ring_px)
    outer_top = max(0, top - ring_px)
    outer_right = min(width, right + ring_px)
    outer_bottom = min(height, bottom + ring_px)
    if lab_img is not None:
        color_img = lab_img
    else:
        try:
            import cv2  # type: ignore

            color_img = cv2.cvtColor(image_np[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        except Exception:
            color_img = image_np[:, :, :3].astype(np.float32)

    inner = color_img[top:bottom, left:right]
    outer = color_img[outer_top:outer_bottom, outer_left:outer_right]
    if inner.size == 0 or outer.size == 0:
        return 0.0

    inner_height, inner_width = inner.shape[0], inner.shape[1]
    yy, xx = np.ogrid[:inner_height, :inner_width]
    inner_mask = (
        (xx < ring_px)
        | (xx >= max(0, inner_width - ring_px))
        | (yy < ring_px)
        | (yy >= max(0, inner_height - ring_px))
    )
    outer_mask = np.ones((outer.shape[0], outer.shape[1]), dtype=bool)
    inner_left = left - outer_left
    inner_top = top - outer_top
    outer_mask[inner_top : inner_top + inner_height, inner_left : inner_left + inner_width] = False
    if not np.any(inner_mask) or not np.any(outer_mask):
        return 0.0
    inner_mean = inner[inner_mask].reshape(-1, inner.shape[-1]).mean(axis=0)
    outer_mean = outer[outer_mask].reshape(-1, outer.shape[-1]).mean(axis=0)
    return float(np.linalg.norm(inner_mean - outer_mean))


def _batch_alignment_resonance_scores(
    boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    """Row/column alignment resonance — tab bars, bottom nav, vertical tool rails.

    For each box center, count peers whose centers fall within a tolerance band
    derived from the median box size (parameter-free).  A horizontal row of
    ≥3 spread-out boxes (e.g. 首页/我的) scores higher than accidental single
    overlaps.  Symmetric treatment for vertical columns (sidebar icons).
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    centers = np.array(
        [[(float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5] for b in boxes],
        dtype=np.float64,
    )
    heights = np.array([max(float(b[3]) - float(b[1]), 1e-6) for b in boxes], dtype=np.float64)
    widths = np.array([max(float(b[2]) - float(b[0]), 1e-6) for b in boxes], dtype=np.float64)
    med_h = float(np.median(heights))
    med_w = float(np.median(widths))
    tol_y = max(med_h * 0.5, 0.01)
    tol_x = max(med_w * 0.5, 0.01)

    raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        cy_i = centers[i, 1]
        cx_i = centers[i, 0]
        row_peers: list[int] = []
        col_peers: list[int] = []
        for j in range(n):
            if i == j:
                continue
            if abs(centers[j, 1] - cy_i) <= tol_y:
                row_peers.append(j)
            if abs(centers[j, 0] - cx_i) <= tol_x:
                col_peers.append(j)

        row_score = 0.0
        col_score = 0.0
        if len(row_peers) >= 2:
            xs = [centers[j, 0] for j in row_peers] + [cx_i]
            x_span = max(xs) - min(xs)
            row_score = float(len(row_peers)) + min(1.0, x_span / max(med_w * 3.0, 0.05))
        elif len(row_peers) == 1:
            row_score = 0.5

        if len(col_peers) >= 2:
            ys = [centers[j, 1] for j in col_peers] + [cy_i]
            y_span = max(ys) - min(ys)
            col_score = float(len(col_peers)) + min(1.0, y_span / max(med_h * 3.0, 0.05))
        elif len(col_peers) == 1:
            col_score = 0.5

        raw[i] = max(row_score, col_score)

    rmin = float(raw.min())
    rmax = float(raw.max())
    if rmax > rmin:
        return [float((raw[i] - rmin) / (rmax - rmin + 1e-6)) for i in range(n)]
    return [0.0] * n


def _batch_grid_resonance_scores_legacy(
    boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    """Legacy 1D grid-histogram resonance (top-left corner bins); diagnostic only."""
    grid_x = [max(0, min(100, int(round(box[0] * 100.0)))) for box in boxes]
    grid_y = [max(0, min(100, int(round(box[1] * 100.0)))) for box in boxes]
    hist_x = np.bincount(np.asarray(grid_x, dtype=np.int32), minlength=101)
    hist_y = np.bincount(np.asarray(grid_y, dtype=np.int32), minlength=101)
    resonance_raw = [float(hist_x[x_coord] + hist_y[y_coord]) for x_coord, y_coord in zip(grid_x, grid_y)]
    resonance_min = min(resonance_raw)
    resonance_max = max(resonance_raw)
    if resonance_max > resonance_min:
        return [
            float((value - resonance_min) / (resonance_max - resonance_min + 1e-6))
            for value in resonance_raw
        ]
    return [0.0 for _ in resonance_raw]


def _pentagonal_eligible_source(source: str) -> bool:
    src = str(source or "")
    if src in {
        "yolo",
        "ocr",
        "canny",
        "blob_circle",
        "blob_ellipse",
        "hough_circle",
        "edge_ellipse",
        "circular_aggregate",
        "stripe_row",
    }:
        return True
    return src.startswith("box_yolo") or src.startswith("box_canny")


def _batch_containment_scores(
    boxes: list[tuple[float, float, float, float]],
    areas: list[float],
) -> list[float]:
    """Vectorized G term: count strictly smaller fully-contained children per box."""
    count_total = len(boxes)
    if count_total == 0:
        return []
    if count_total == 1:
        return [1.0]
    b = np.asarray(boxes, dtype=np.float64)
    a = np.asarray(areas, dtype=np.float64)
    ii = np.arange(count_total, dtype=np.int32)[:, None]
    jj = np.arange(count_total, dtype=np.int32)[None, :]
    mask = (ii != jj) & (a[jj] < a[ii])
    contained = (
        (b[ii, 0] <= b[jj, 0])
        & (b[ii, 1] <= b[jj, 1])
        & (b[ii, 2] >= b[jj, 2])
        & (b[ii, 3] >= b[jj, 3])
    )
    child_counts = (mask & contained).sum(axis=1)
    return [float(1.0 / (1.0 + int(c))) for c in child_counts]


def _attach_pentagonal_interaction_energy(mius: list[MiuDict], image_np) -> list[MiuDict]:
    """Attach zero-learned-parameter YOLO interaction-energy attributes.

    Five terms, all screen-local and bounded in [0, 1]:
      S  inverse scale rank
      H  internal gradient entropy
      C  boundary contrast rank
      G  containment freedom
      R  row/column alignment resonance (tab bars, nav rails)
    """
    valid_indices: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    for index, miu in enumerate(mius):
        source = str(miu.get("source", ""))
        if not _pentagonal_eligible_source(source):
            continue
        box = miu.get("box")
        if not box or len(box) < 4:
            continue
        norm_box = _normalize_box(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if _op_box_area(norm_box) <= 0.0:
            continue
        valid_indices.append(index)
        boxes.append(norm_box)
    count_total = len(boxes)
    if count_total == 0:
        return mius

    areas = [_op_box_area(box) for box in boxes]
    area_ranks = _op_rank_ascending(areas)
    if count_total <= 1:
        scale_scores = [1.0]
    else:
        scale_scores = [float((count_total - rank) / max(count_total - 1, 1)) for rank in area_ranks]

    entropy_scores = _batch_gradient_entropy_scores(boxes, image_np)
    # Compute the LAB color image ONCE (previously every box re-ran a full-frame
    # RGB->LAB cvtColor inside _op_boundary_contrast_raw -- O(n) full-image
    # conversions; now O(1)).
    lab_img = None
    try:
        import cv2  # type: ignore

        lab_img = cv2.cvtColor(image_np[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    except Exception:
        lab_img = image_np[:, :, :3].astype(np.float32)
    contrast_raw = [_op_boundary_contrast_raw(box, image_np, lab_img=lab_img) for box in boxes]
    contrast_ranks = _op_rank_ascending(contrast_raw)
    contrast_scores = [float(rank / max(count_total, 1)) for rank in contrast_ranks]

    containment_scores = _batch_containment_scores(boxes, areas)

    resonance_scores = _batch_alignment_resonance_scores(boxes)
    resonance_grid_legacy = _batch_grid_resonance_scores_legacy(boxes)

    for local_index, miu_index in enumerate(valid_indices):
        miu = mius[miu_index]
        pentagonal_sum = (
            scale_scores[local_index]
            + entropy_scores[local_index]
            + contrast_scores[local_index]
            + containment_scores[local_index]
            + resonance_scores[local_index]
        )
        det_score = float(miu.get("det_score", 0.0) or 0.0)
        det_clamped = max(0.0, min(1.0, det_score))
        confidence_attr = float(math.log1p(4.0 * det_clamped) / math.log1p(4.0))
        miu["pentagonal_scale"] = float(scale_scores[local_index])
        miu["pentagonal_entropy"] = float(entropy_scores[local_index])
        miu["pentagonal_contrast"] = float(contrast_scores[local_index])
        miu["pentagonal_containment"] = float(containment_scores[local_index])
        miu["pentagonal_resonance"] = float(resonance_scores[local_index])
        miu["pentagonal_resonance_grid_legacy"] = float(resonance_grid_legacy[local_index])
        miu["pentagonal_confidence_attr"] = confidence_attr
        miu["pentagonal_energy_sum"] = float(pentagonal_sum)
        miu["pentagonal_energy_sum_no_resonance"] = float(pentagonal_sum - resonance_scores[local_index])
        miu["pentagonal_energy"] = float(det_clamped * pentagonal_sum)
        miu["pentagonal_energy_additive"] = float(pentagonal_sum + confidence_attr)
        miu["pentagonal_energy_no_resonance"] = float(det_clamped * (pentagonal_sum - resonance_scores[local_index]))
        miu["module_a_source"] = "pentagonal_yolo"
    return mius


def _op_crop_keys(boxes, image_np):
    """Return (unique_boxes, inverse_map) deduping identical glyph crops.

    Key = 8x8 average-hash bits + first 4 bytes of the 64x64 crop md5 (exact +
    near-exact). ``unique_boxes`` are the deduped boxes to caption; ``inverse[i]``
    gives the index into ``unique_boxes`` for input box ``i`` (degenerate -> -1).
    """
    import hashlib

    import cv2  # type: ignore

    H, W = image_np.shape[0], image_np.shape[1]
    seen: dict[bytes, int] = {}
    unique_boxes: list = []
    inverse: list[int] = []
    for c in boxes:
        xmin, xmax = int(c[0] * W), int(c[2] * W)
        ymin, ymax = int(c[1] * H), int(c[3] * H)
        if xmax <= xmin or ymax <= ymin:
            inverse.append(-1)
            continue
        crop = cv2.resize(image_np[ymin:ymax, xmin:xmax, :], (64, 64))
        gray = cv2.cvtColor(cv2.resize(crop, (8, 8)), cv2.COLOR_RGB2GRAY)
        key = (gray > gray.mean()).tobytes() + hashlib.md5(crop.tobytes()).digest()[:4]
        pos = seen.get(key)
        if pos is None:
            pos = len(unique_boxes)
            seen[key] = pos
            unique_boxes.append(c)
        inverse.append(pos)
    return unique_boxes, inverse


def _op_visual_hash_key(box, image_np) -> Optional[bytes]:
    """8x8 average-hash key for one normalized box (None if degenerate).

    Same hash family as ``_op_crop_keys`` -- used at element level so we can
    count how many other elements share the same glyph (visual redundancy).
    """
    import cv2  # type: ignore

    H, W = image_np.shape[0], image_np.shape[1]
    xmin, xmax = int(box[0] * W), int(box[2] * W)
    ymin, ymax = int(box[1] * H), int(box[3] * H)
    if xmax <= xmin or ymax <= ymin:
        return None
    crop = cv2.resize(image_np[ymin:ymax, xmin:xmax, :], (8, 8))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return (gray > gray.mean()).tobytes()


def _op_distinctiveness(elems, image_np) -> list[float]:
    """Continuous local-distinctiveness (non-redundancy) per element, in [0,1].

    GUI screens are dominated by REPEATED units (grid cells, list rows, a row of
    identical toolbar icons, bullet glyphs); ShowUI/DART treat such redundancy as
    low-information.  We fold three cheap, orthogonal redundancy signals:

      * visual duplication  n_vis  -- #other elements sharing the 8x8 avg-hash bucket
      * content/role dup    n_txt  -- #other elements with identical text (or, when
                                      text-less, identical role/type)
      * container demotion          -- layout containers get a soft 0.5 multiplier

        A_dist = 1/(1+n_vis) * 1/(1+n_txt) * (0.5 if container else 1.0)

    Unique, atomic widgets score ~1.0; one of twenty identical list thumbnails
    scores ~1/20.  No learned weights; reuses the crop-dedup hash already computed.
    """
    n = len(elems)
    vis_keys: list[Optional[bytes]] = []
    txt_keys: list[str] = []
    for e in elems:
        vis_keys.append(_op_visual_hash_key(e["bbox"], image_np))
        content = (e.get("content") or "").strip().lower()
        if content:
            txt_keys.append("t:" + content)
        else:
            txt_keys.append("r:" + str(e.get("type", "")))
    vis_count: dict[bytes, int] = {}
    for k in vis_keys:
        if k is not None:
            vis_count[k] = vis_count.get(k, 0) + 1
    txt_count: dict[str, int] = {}
    for k in txt_keys:
        txt_count[k] = txt_count.get(k, 0) + 1

    out: list[float] = []
    for i in range(n):
        n_vis = (vis_count.get(vis_keys[i], 1) - 1) if vis_keys[i] is not None else 0
        n_txt = txt_count.get(txt_keys[i], 1) - 1
        score = 1.0 / (1.0 + n_vis) * 1.0 / (1.0 + n_txt)
        if elems[i].get("_is_container"):
            score *= 0.5
        out.append(float(score))
    return out


def _get_florence_model():
    """Lazy-load Florence-2 icon_caption model once per worker (GPU if available)."""
    global _FLORENCE_MODEL, _FLORENCE_FAILED
    if _FLORENCE_FAILED:
        return None
    if _FLORENCE_MODEL is not None:
        return _FLORENCE_MODEL
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        cap_dir = os.environ.get("OMNIPARSER_CAPTION_WEIGHTS", "").strip()
        if not cap_dir:
            icon_w = os.environ.get("OMNIPARSER_ICON_WEIGHTS", "").strip()
            if icon_w:
                # .../v2.0/icon_detect/model.pt -> .../v2.0/icon_caption_florence
                cap_dir = str(Path(icon_w).resolve().parents[1] / "icon_caption_florence")
        src = cap_dir if cap_dir and Path(cap_dir).exists() else "microsoft/Florence-2-base"
        use_gpu = torch.cuda.is_available() and os.environ.get("MIU_CAPTION_GPU", "1") != "0"
        device = "cuda" if use_gpu else "cpu"
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, trust_remote_code=True).to(device)
        _FLORENCE_MODEL = {"model": model, "processor": processor}
        return _FLORENCE_MODEL
    except Exception as exc:
        _FLORENCE_FAILED = True
        _log.warning("Florence-2 caption model unavailable; icon captions will stay empty (%r)", exc)
        return None


def _op_caption_boxes(model_proc, boxes, image_np, max_new_tokens: int, batch_size: int = 128) -> list[str]:
    """Caption normalized boxes via 64x64 crops, <CAPTION>, greedy (token<=max)."""
    import cv2  # type: ignore
    import torch  # type: ignore
    from torchvision.transforms import ToPILImage  # type: ignore

    to_pil = ToPILImage()
    H, W = image_np.shape[0], image_np.shape[1]
    crops = []
    for c in boxes:
        xmin, xmax = int(c[0] * W), int(c[2] * W)
        ymin, ymax = int(c[1] * H), int(c[3] * H)
        if xmax <= xmin or ymax <= ymin:
            continue
        crops.append(to_pil(cv2.resize(image_np[ymin:ymax, xmin:xmax, :], (64, 64))))
    if not crops:
        return []
    model, processor = model_proc["model"], model_proc["processor"]
    device = model.device
    out: list[str] = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        if device.type == "cuda":
            inputs = processor(
                images=batch, text=["<CAPTION>"] * len(batch), return_tensors="pt", do_resize=False
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=["<CAPTION>"] * len(batch), return_tensors="pt").to(device)
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        out.extend(t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True))
    return out


def _op_element_to_miu(
    elem: dict,
    has_text: bool,
    is_container: bool,
    content_source: str,
    distinctiveness: float = 1.0,
) -> MiuDict:
    """Map an internal OmniParser element dict to the downstream MIU schema."""
    bbox = elem["bbox"]
    box = _normalize_box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    content = (elem.get("content") or "").strip()
    is_text = elem.get("type") == "text"
    interactive = bool(elem.get("interactivity", True))
    return {
        "box": box,
        "type": "text" if is_text else "icon_button",
        "text": content[:64],
        "source": elem.get("source", "box_yolo_content_yolo"),
        "det_score": float(elem.get("det_score", 0.0) or 0.0),
        # --- Module-1 attribute inputs (stored, scored downstream) ---
        "interactivity": interactive,        # A_act source (official field)
        "interactable": 1.0 if interactive else 0.0,
        "has_text": 1.0 if has_text else 0.0,  # A_text bit
        "is_container": 1.0 if is_container else 0.0,  # kept as a raw flag (debug)
        "distinctiveness": float(distinctiveness),  # A_dist continuous [0,1]
        "content_source": content_source,    # "ocr" | "florence" | ""
    }


def detect_omniparser_full_inproc(
    image: Image.Image,
    mode: str = "both",
) -> Optional[list[MiuDict]]:
    """In-process OmniParser-v2 pipeline with a 4-way ablation. Returns MIU dicts.

    ``mode`` in {"icon", "ocr", "florence", "both"}:
      * "icon"     -- YOLO icon boxes ONLY (+ solid-color filter, container gate,
                      visual-hash distinctiveness).  NO EasyOCR, NO Florence.
                      This is the LEAN/FAST front-end (~YOLO cost, tens of ms):
                      downstream MIU selection (info-density + ICKH over projector
                      embeddings) needs the interaction-unit boxes + cheap geometric
                      attributes, NOT detector-side OCR/caption text (the model
                      already reads on-screen text via the kept visual tokens).
      * "ocr"      -- + EasyOCR text boxes merged into icons; NO captioning.
      * "florence" -- + Florence caption for text-less icons; NO separate OCR.
      * "both"     -- full pipeline (OCR + merge + caption).

    Returns ``None`` only when the YOLO icon detector cannot load (caller may
    fall back to subprocess).
    """
    mode = (mode or "both").lower().strip()
    if mode not in {"icon", "ocr", "florence", "both"}:
        mode = "both"

    weights = os.environ.get(
        "OMNIPARSER_ICON_WEIGHTS",
        _default_omniparser_icon_weights(),
    )
    conf = float(os.environ.get("OMNIPARSER_ICON_CONF", "0.05"))
    iou_box = float(os.environ.get("OMNIPARSER_ICON_IOU", "0.1"))
    iou_merge = float(os.environ.get("OMNIPARSER_MERGE_IOU", "0.7"))
    max_tokens = int(os.environ.get("MIU_CAPTION_MAX_TOKENS", "10"))
    cont_area = float(os.environ.get("MIU_CONTAINER_AREA", "0.10"))
    cont_children = int(os.environ.get("MIU_CONTAINER_CHILDREN", "3"))

    # ---- YOLO icon boxes (normalized xyxy) ----
    try:
        model = _get_yolo_model(weights)
    except Exception as exc:
        global _YOLO_IMPORT_OK, _YOLO_LOAD_ERROR
        _YOLO_IMPORT_OK = False
        if _YOLO_LOAD_ERROR != repr(exc):
            _YOLO_LOAD_ERROR = repr(exc)
            _log.warning("In-process YOLO load failed for OmniParser full (%r)", exc)
        return None

    width, height = image.size
    image_rgb = image.convert("RGB")
    image_np = np.asarray(image_rgb)
    max_edge_default = "384" if mode == "icon" else "0"
    try:
        max_edge = int(os.environ.get("OMNIPARSER_ICON_MAX_EDGE", max_edge_default))
    except ValueError:
        max_edge = int(max_edge_default)
    detect_image = image_rgb
    detect_width, detect_height = width, height
    if max_edge > 0 and max(width, height) > max_edge:
        scale = float(max_edge) / float(max(width, height))
        detect_width = max(1, int(round(width * scale)))
        detect_height = max(1, int(round(height * scale)))
        detect_width = _snap_yolo_stride(detect_width)
        detect_height = _snap_yolo_stride(detect_height)
        detect_image = image_rgb.resize((detect_width, detect_height), Image.BILINEAR)
    imgsz_h = _snap_yolo_stride(detect_height)
    imgsz_w = _snap_yolo_stride(detect_width)
    predict_kwargs = {"conf": conf, "iou": iou_box, "imgsz": (imgsz_h, imgsz_w), "verbose": False}
    device = _yolo_predict_device()
    if device is not None:
        predict_kwargs["device"] = device
    if _yolo_use_half():
        predict_kwargs["half"] = True
    import torch  # type: ignore
    with torch.inference_mode():
        results = model.predict(detect_image, **predict_kwargs)
    icon_boxes: list[dict] = []
    solid_std = float(os.environ.get("OMNIPARSER_SOLID_STD", "6.0"))
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, scores):
            nb = (
                float(x1) / max(detect_width, 1),
                float(y1) / max(detect_height, 1),
                float(x2) / max(detect_width, 1),
                float(y2) / max(detect_height, 1),
            )
            if _op_box_area(nb) < 0.00002:
                continue
            # drop pure solid-color false positives (blank bg, white cards)
            if solid_std > 0 and _op_box_is_solid(nb, image_np, solid_std):
                continue
            icon_boxes.append({"bbox": nb, "det_score": float(score)})

    # ---- OCR text boxes (mode in {ocr, both}; KEEP full-res) ----
    ocr_elems: Optional[list[dict]] = None
    if mode in {"ocr", "both"}:
        ocr_elems = []
        try:
            ocr_res = _run_ocr_readtext(image_np)
        except Exception as exc:
            _log.debug("OmniParser OCR readtext failed: %r", exc)
            ocr_res = []
        for entry in ocr_res:
            try:
                quad, text, conf_ocr = entry[0], str(entry[1] or "").strip(), float(entry[2])
            except Exception:
                continue
            if not text or conf_ocr < 0.25:
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            nb = (
                min(xs) / max(width, 1),
                min(ys) / max(height, 1),
                max(xs) / max(width, 1),
                max(ys) / max(height, 1),
            )
            if _op_box_area(nb) < 0.00002:
                continue
            ocr_elems.append(
                {
                    "type": "text",
                    "bbox": nb,
                    "interactivity": False,
                    "content": text,
                    "source": "box_ocr_content_ocr",
                    "det_score": float(conf_ocr),
                }
            )

    # ---- Merge OCR into icons (assigns interactivity/source/content) ----
    merged = _op_remove_overlap_new(icon_boxes, iou_merge, ocr_boxes=ocr_elems)

    # Split merged elements into OCR-text vs icon for downstream handling.
    icon_elems = [e for e in merged if e.get("type") == "icon"]
    text_elems = [e for e in merged if e.get("type") == "text"]

    # ---- Caption text-less icons (mode in {florence, both}) ----
    # container-gate + crop-dedup so we never waste a forward pass on a layout
    # container or a repeated glyph.
    icon_boxes_norm = [e["bbox"] for e in icon_elems]
    container_mask = _op_container_mask(icon_boxes_norm, cont_area, cont_children)
    for e, is_cont in zip(icon_elems, container_mask):
        e["_is_container"] = bool(is_cont)

    if mode in {"florence", "both"}:
        # Caption every text-less icon that is not a container.
        to_caption_idx = [
            i
            for i, e in enumerate(icon_elems)
            if not (e.get("content") or "").strip() and not e["_is_container"]
        ]
        if to_caption_idx:
            cap_model = _get_florence_model()
            if cap_model is not None:
                cap_boxes = [icon_elems[i]["bbox"] for i in to_caption_idx]
                try:
                    uniq_boxes, inverse = _op_crop_keys(cap_boxes, image_np)
                    uniq_caps = _op_caption_boxes(cap_model, uniq_boxes, image_np, max_tokens)
                    for k, i in enumerate(to_caption_idx):
                        inv = inverse[k]
                        cap = uniq_caps[inv] if 0 <= inv < len(uniq_caps) else ""
                        if cap:
                            icon_elems[i]["content"] = cap
                            icon_elems[i]["source"] = "box_yolo_content_yolo"
                except Exception as exc:
                    _log.warning("Florence captioning failed; icons stay unlabelled (%r)", exc)

    # ---- Rec-only A_text labeling (reuse YOLO boxes, NO text detector) ----
    # Speed-first lean path: run RapidOCR's recognizer on each YOLO icon crop
    # in one batched GPU pass so ``has_text`` (A_text) is a TRUE read of the
    # box's characters.  Only labels icons still lacking content (so it never
    # overrides OCR-merged or Florence text), and only when MIU_ICON_RECOCR=1.
    if _icon_recocr_enabled():
        rec_conf = float(os.environ.get("MIU_ICON_RECOCR_CONF", "0.5"))
        to_rec_idx = [
            i for i, e in enumerate(icon_elems)
            if not (e.get("content") or "").strip() and not e.get("_is_container")
        ]
        if to_rec_idx:
            rec_pairs = _rapidocr_rec_crops(image_np, [icon_elems[i]["bbox"] for i in to_rec_idx])
            for pos, i in enumerate(to_rec_idx):
                if pos >= len(rec_pairs):
                    break
                txt, sc = rec_pairs[pos]
                txt = (txt or "").strip()
                if txt and sc >= rec_conf:
                    icon_elems[i]["content"] = txt
                    icon_elems[i]["source"] = "box_yolo_content_recocr"

    # ---- Emit MIU dicts ----
    # Distinctiveness is computed over ALL final elements (text + icons) so that
    # a repeated list row of text behaves like a repeated icon grid.
    all_elems = text_elems + icon_elems
    dist_scores = _op_distinctiveness(all_elems, image_np)
    mius: list[MiuDict] = []
    n_text = len(text_elems)
    for j, e in enumerate(text_elems):
        mius.append(
            _op_element_to_miu(
                e,
                has_text=True,
                is_container=False,
                content_source="ocr",
                distinctiveness=dist_scores[j],
            )
        )
    for k, e in enumerate(icon_elems):
        content = (e.get("content") or "").strip()
        src = e.get("source", "")
        if src == "box_yolo_content_ocr":
            csrc = "ocr"
        elif src == "box_yolo_content_recocr":
            csrc = "recocr"
        elif content:
            csrc = "florence"
        else:
            csrc = ""
        mius.append(
            _op_element_to_miu(
                e,
                has_text=bool(content),
                is_container=bool(e.get("_is_container")),
                content_source=csrc,
                distinctiveness=dist_scores[n_text + k],
            )
        )
    return _attach_pentagonal_interaction_energy(mius, image_np)


def _detect_omniparser_full(image: Image.Image) -> Optional[list[MiuDict]]:
    """v9-full path: in-process OmniParser-v2 pipeline (mode from env).

    Replaces the v8 per-image subprocess.  ``MIU_OMNIPARSER_MODE`` selects the
    ablation arm ("ocr" | "florence" | "both", default "ocr" -- Florence
    captioning is OFF by default: it is the detector's autoregressive latency
    hotspot and is redundant with the visual tokens we already keep).
    """
    mode = os.environ.get("MIU_OMNIPARSER_MODE", "ocr")
    return detect_omniparser_full_inproc(image, mode=mode)


def _detect_omniparser_via_subprocess(image: Image.Image, with_ocr: bool) -> Optional[list[MiuDict]]:
    detect_py = os.environ.get("OMINI_DETECT_PYTHON", "").strip()
    if not detect_py or not Path(detect_py).exists():
        return None
    script = Path(__file__).resolve().parents[2] / "scripts" / "omniparser_detect_mius.py"
    if not script.exists():
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = handle.name
            image.convert("RGB").save(tmp_path)
        cmd = [detect_py, str(script), tmp_path]
        if with_ocr:
            cmd.append("--with-ocr")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        return json.loads(proc.stdout.strip() or "[]")
    except subprocess.CalledProcessError as exc:
        # Capture stderr once so we can see why OmniParser is failing instead
        # of silently turning off the detector and degrading the experiment.
        _log.warning(
            "OmniParser subprocess failed (exit=%s): %s", exc.returncode,
            (exc.stderr or "").strip().splitlines()[-1] if exc.stderr else "<no stderr>",
        )
        return None
    except Exception as exc:
        _log.warning("OmniParser subprocess invocation error: %r", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def layout_skeleton_mius() -> list[MiuDict]:
    boxes = [
        ((0.0, 0.0, 1.0, 0.12), "top_bar"),
        ((0.0, 0.88, 1.0, 1.0), "bottom_bar"),
        ((0.0, 0.12, 0.18, 0.88), "left_rail"),
        ((0.82, 0.12, 1.0, 0.88), "right_rail"),
        ((0.12, 0.18, 0.88, 0.82), "content_panel"),
    ]
    return [
        {"box": box, "type": role, "text": "", "source": "skeleton", "det_score": 0.35}
        for box, role in boxes
    ]


def build_raw_mius(
    image: Image.Image,
    detector: str = "heuristic",
    include_skeleton: bool = True,
) -> tuple[list[MiuDict], dict[str, float]]:
    detector = (detector or "heuristic").lower().strip()

    # v5: perceptual-hash detection cache.  Many GUI agent trajectories
    # keep the same screenshot across consecutive steps (typing into an
    # input, scrolling small deltas, modal popups); caching by 8x8 avg
    # hash gives 25-40 % hit rate on AgentCPM-GUI traces.  We cache the
    # raw detection output (pre-skeleton) so toggling ``include_skeleton``
    # still works correctly across calls.
    global _DETECT_CACHE_HITS, _DETECT_CACHE_MISSES
    cache_enabled = os.environ.get("MIU_DETECT_CACHE", "1") != "0" and image is not None
    cache_key: Optional[tuple] = None
    if cache_enabled:
        try:
            cache_key = (detector, _phash_8x8(image))
        except Exception:
            cache_key = None
    if cache_key is not None and cache_key in _DETECT_CACHE:
        cached_mius, cached_meta = _DETECT_CACHE[cache_key]
        _DETECT_CACHE.move_to_end(cache_key)
        _DETECT_CACHE_HITS += 1
        out_mius = [dict(m) for m in cached_mius]
        if include_skeleton:
            out_mius.extend(layout_skeleton_mius())
        meta = dict(cached_meta)
        meta["miu_detect_ms"] = 0.0
        meta["miu_detect_cache_hit"] = 1.0
        meta["miu_raw_count"] = float(len(out_mius))
        meta["miu_skeleton_count"] = float(
            sum(1 for m in out_mius if m.get("source") == "skeleton")
        )
        return out_mius, meta

    t0 = time.perf_counter()
    mius: list[MiuDict] = []
    backend = detector

    ocr_backend = "none"
    if detector in {"heuristic", "default", "v1"}:
        mius.extend(detect_heuristic_visual(image))
        ocr_mius, ocr_backend = resolve_ocr_mius(image)
        mius.extend(ocr_mius)
    elif detector in {"yolo", "yolo_icon"}:
        mius.extend(detect_yolo_mius(image))
    elif detector in {
        "canny",
        "canny_morph",
        "cannymorph",
        "omniparser_lean",
        "omniparser_icon_attr",
    }:
        # Lightweight traditional-CV metadata extractor (Canny + morphological
        # close).  Replaces the ~208 ms YOLO front-end with a <5 ms classical
        # pipeline.  omniparser_lean / omniparser_icon_attr now route here so the
        # A-module is a pure, no-injection metadata extractor: boxes are fed
        # unchanged into the shared pentagonal attribute calculator.
        backend = "canny_morph"
        ocr_backend = "none"
        mius.extend(detect_canny_morph_mius(image))
    elif detector in {
        "omniparser_full",
        "omniparser_v2_full",
        "omniparser_both",
        "omniparser_ocr",
        "omniparser_florence",
    }:
        # v9: in-process OmniParser-v2 pipeline with 4-way ablation.
        #   omniparser_ocr      -> OCR text + icon merge, NO captioning
        #   omniparser_florence -> caption all icons, NO separate OCR text boxes
        #   omniparser_both     -> full pipeline (OCR + merge + caption)
        #   omniparser_full     -> env MIU_OMNIPARSER_MODE (default "ocr", no Florence)
        if detector == "omniparser_ocr":
            mode = "ocr"
        elif detector == "omniparser_florence":
            mode = "florence"
        elif detector == "omniparser_both":
            mode = "both"
        else:
            mode = os.environ.get("MIU_OMNIPARSER_MODE", "ocr")
        backend = f"omniparser_{mode}"
        ocr_backend = "inproc_full" if mode in {"ocr", "both"} else "none"
        full = detect_omniparser_full_inproc(image, mode=mode)
        if full is not None:
            mius.extend(full)
        elif _disable_omniparser_subproc_fallback():
            _log.error(
                "OmniParser in-process YOLO unavailable and subprocess fallback is disabled "
                "(MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK=1)."
            )
        else:
            # YOLO loader broken -> try the legacy subprocess detector once.
            remote = _detect_omniparser_via_subprocess(image, with_ocr=(mode in {"ocr", "both"}))
            if remote is not None:
                mius.extend(remote)
                ocr_backend = "subproc_full" if mode in {"ocr", "both"} else "none"
            else:
                _log.warning(
                    "OmniParser full (in-process) unavailable and subprocess fallback failed; "
                    "check OMNIPARSER_ICON_WEIGHTS / OMINI_DETECT_PYTHON."
                )
    elif detector in {"omniparser_icon", "omniparser_v2", "omniparser"}:
        backend = "omniparser_v2"
        # v8-min: icon_detect YOLO + OCR.  v8-full uses detector=omniparser_full.
        # v5 fix: distinguish "loader broken" (in_proc is None) from "no icons
        # on this screen" (in_proc == []).  In v4 the latter wrongly fell
        # back to a 25 s subprocess re-detection on every icon-free screen.
        in_proc = detect_omniparser_icon_mius(image)
        if in_proc is None:
            remote = _detect_omniparser_via_subprocess(image, with_ocr=True)
            if remote is not None:
                mius.extend(remote)
                ocr_backend = "subproc_full"
            else:
                _log.warning(
                    "OmniParser detector unavailable from both in-process AND subprocess paths; "
                    "check OMNIPARSER_ICON_WEIGHTS / OMINI_DETECT_PYTHON."
                )
        else:
            mius.extend(in_proc)
            ocr_mius, ocr_backend = resolve_ocr_mius(image)
            mius.extend(ocr_mius)
    elif detector in {"yolo_ocr"}:
        mius.extend(detect_yolo_mius(image))
        ocr_mius, ocr_backend = resolve_ocr_mius(image)
        mius.extend(ocr_mius)
        backend = "yolo_ocr"
    elif detector in {"ocr"}:
        ocr_mius, ocr_backend = resolve_ocr_mius(image)
        mius.extend(ocr_mius)
    else:
        mius.extend(detect_heuristic_visual(image))
        ocr_mius, ocr_backend = resolve_ocr_mius(image)
        mius.extend(ocr_mius)
        backend = "heuristic"

    # Capture pre-skeleton detection result for the cache so toggling
    # ``include_skeleton`` later does not poison the cached value.
    pre_skel_mius = [dict(m) for m in mius]
    if include_skeleton:
        mius.extend(layout_skeleton_mius())

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    meta = {
        "miu_detect_ms": float(elapsed_ms),
        "miu_detect_backend": backend,
        "miu_raw_count": float(len(mius)),
        "miu_vision_count": float(
            sum(
                1
                for m in mius
                if m.get("source")
                in {"vision", "yolo", "omniparser_icon", "box_yolo_content_yolo", "box_yolo_content_ocr"}
            )
        ),
        "miu_ocr_count": float(
            sum(1 for m in mius if m.get("source") in {"ocr", "box_ocr_content_ocr"})
        ),
        "miu_ocr_backend": ocr_backend,
        "miu_skeleton_count": float(sum(1 for m in mius if m.get("source") == "skeleton")),
        "miu_detect_cache_hit": 0.0,
    }
    if cache_key is not None:
        _DETECT_CACHE_MISSES += 1
        cached_meta_for_store = {k: v for k, v in meta.items() if k not in {"miu_detect_ms", "miu_detect_cache_hit", "miu_raw_count", "miu_skeleton_count"}}
        _DETECT_CACHE[cache_key] = (pre_skel_mius, cached_meta_for_store)
        if len(_DETECT_CACHE) > _DETECT_CACHE_MAX:
            _DETECT_CACHE.popitem(last=False)
    return mius, meta


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_inter_over_child(parent: tuple[float, float, float, float], child: tuple[float, float, float, float]) -> float:
    """Fraction of ``child`` covered by ``parent`` (NOT symmetric IoU).

    This is the right metric for parent/child detection: a small button
    fully contained in a large panel has ``inter/child_area == 1.0`` and
    ``IoU`` close to the panel size ratio, which IoU would penalize.
    """
    ix1, iy1 = max(parent[0], child[0]), max(parent[1], child[1])
    ix2, iy2 = min(parent[2], child[2]), min(parent[3], child[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(_box_area(child), 1e-9)


_INTERACTIVE_PARENT_ROLES = frozenset(
    {
        "icon_button",
        "button",
        "icon",
        "input",
        "input_field",
        "text_field",
        "search_button",
        "tab",
        "toolbar",
        "link",
        "checkbox",
        "radio",
        "switch",
        "menu",
        "list_item",
    }
)


def filter_redundant_image_mius(
    mius: list[MiuDict],
    *,
    max_area_ratio: float = 0.10,
    large_text_max_area_ratio: float = 0.16,
    cover_w_ratio: float = 0.70,
    cover_h_ratio: float = 0.40,
    parent_min_children: int = 3,
    parent_child_cover: float = 0.85,
    min_area_ratio: float = 1e-4,
    max_aspect_ratio: float = 50.0,
    keep_sources: tuple[str, ...] = (),
    drop_non_atomic_types: tuple[str, ...] = ("panel", "container", "background", "boundary", "skeleton"),
) -> tuple[list[MiuDict], dict[str, float]]:
    """Drop container & noise boxes via geometry-only UI priors (no type-based drops).

    Rules: area / coverage / aspect ratio / parent-of-many.  Type labels
    (panel, text_like, icon_button) are kept for downstream scoring only.
    Large blank regions are down-ranked by pentagonal ``E = conf × (S+H+C+G)``
    rather than removed solely by type.

    Set ``GEOPACT_PREFILTER_DROP_TYPE=1`` to restore legacy type-based drops.
    """
    if not mius:
        return [], {
            "miu_prefilter_in": 0.0,
            "miu_prefilter_kept": 0.0,
            "miu_prefilter_dropped": 0.0,
            "miu_prefilter_drop_area": 0.0,
            "miu_prefilter_drop_cover": 0.0,
            "miu_prefilter_drop_tiny": 0.0,
            "miu_prefilter_drop_aspect": 0.0,
            "miu_prefilter_drop_parent": 0.0,
            "miu_prefilter_drop_non_atomic": 0.0,
        }

    n_in = len(mius)
    drop_area = drop_cover = drop_tiny = drop_aspect = drop_parent = drop_non_atomic = 0
    drop_by_type = os.environ.get("GEOPACT_PREFILTER_DROP_TYPE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # ---- Pass 1: per-MIU geometric rules ----
    kept: list[MiuDict] = []
    for miu in mius:
        source = str(miu.get("source", "")).lower()
        role = str(miu.get("type", miu.get("role", ""))).lower()
        if source in keep_sources:
            kept.append(miu)
            continue
        if drop_by_type and (role in drop_non_atomic_types or source in drop_non_atomic_types):
            drop_non_atomic += 1
            continue
        box = miu.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        if area < min_area_ratio:
            drop_tiny += 1
            continue
        if area > max_area_ratio:
            drop_area += 1
            continue
        if w >= cover_w_ratio and h >= cover_h_ratio and source != "stripe_row":
            drop_cover += 1
            continue
        aspect = max(w, h) / max(min(w, h), 1e-6)
        if aspect > max_aspect_ratio:
            drop_aspect += 1
            continue
        kept.append(miu)

    # ---- Pass 2: parent-of-many container removal ----
    # A "parent" is a MIU whose box covers >= ``parent_child_cover`` of at
    # least ``parent_min_children`` other (smaller) MIU boxes.  We compute
    # this on the post-pass-1 ``kept`` list so we don't waste work on
    # already-dropped containers.
    if len(kept) >= parent_min_children + 1:
        # Sort descending by area; parents are necessarily larger than their
        # children so this enables an O(N log N + N*K) scan instead of N^2.
        areas = [(_box_area(m["box"]), i) for i, m in enumerate(kept) if m.get("box") and str(m.get("source", "")) not in keep_sources]
        areas.sort(reverse=True)
        # Build a list of (i, box, area) in descending area order.
        sorted_entries = [(i, kept[i]["box"], a) for a, i in areas]
        drop_mask: set[int] = set()
        for rank, (i, pbox, parea) in enumerate(sorted_entries):
            child_hits = 0
            for j, cbox, carea in sorted_entries[rank + 1 :]:
                if j in drop_mask:
                    continue
                if carea >= parea * 0.9:
                    # Not strictly smaller -> not a "child"; skip.
                    continue
                if _box_inter_over_child(pbox, cbox) >= parent_child_cover:
                    child_hits += 1
                    if child_hits >= parent_min_children:
                        break
            if child_hits >= parent_min_children:
                drop_mask.add(i)
                drop_parent += 1
        if drop_mask:
            kept = [m for i, m in enumerate(kept) if i not in drop_mask]

    meta = {
        "miu_prefilter_in": float(n_in),
        "miu_prefilter_kept": float(len(kept)),
        "miu_prefilter_dropped": float(n_in - len(kept)),
        "miu_prefilter_drop_area": float(drop_area),
        "miu_prefilter_drop_cover": float(drop_cover),
        "miu_prefilter_drop_tiny": float(drop_tiny),
        "miu_prefilter_drop_aspect": float(drop_aspect),
        "miu_prefilter_drop_parent": float(drop_parent),
        "miu_prefilter_drop_non_atomic": float(drop_non_atomic),
    }
    return kept, meta


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def compare_detector_overlap(a: list[MiuDict], b: list[MiuDict], iou_thresh: float = 0.3) -> dict[str, float]:
    a_boxes = [m["box"] for m in a if m.get("source") not in {"skeleton"}]
    b_boxes = [m["box"] for m in b if m.get("source") not in {"skeleton"}]
    if not a_boxes or not b_boxes:
        return {"recall_a_in_b": 0.0, "recall_b_in_a": 0.0, "mean_best_iou": 0.0}

    def recall(src, dst):
        hits = 0
        for box in src:
            if max(box_iou(box, other) for other in dst) >= iou_thresh:
                hits += 1
        return hits / max(len(src), 1)

    mean_best = float(
        np.mean([max(box_iou(box, other) for other in b_boxes) for box in a_boxes])
    )
    return {
        "recall_a_in_b": float(recall(a_boxes, b_boxes)),
        "recall_b_in_a": float(recall(b_boxes, a_boxes)),
        "mean_best_iou": mean_best,
    }
