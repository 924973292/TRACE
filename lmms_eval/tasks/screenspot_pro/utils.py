"""ScreenSpot-Pro grounding eval for GUI-Owl (official point-in-bbox protocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.gui_owl_cross.utils import (
    gui_owl_grounding_doc_to_messages,
    parse_click_point_official,
    point_in_bbox_1000,
)


def screenspot_pro_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/screenspot_pro/images")
    img_path = doc.get("image_path") or doc.get("img_filename") or doc.get("file_name")
    if not img_path:
        return []
    p = Path(str(img_path))
    if not p.is_file():
        p = Path(media_dir) / p.name
    if not p.is_file():
        for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
            cand = base / str(img_path)
            if cand.is_file():
                p = cand
                break
            for hit in (base / media_dir).rglob(Path(str(img_path)).name):
                if hit.is_file():
                    p = hit
                    break
    if not p.is_file():
        eval_logger.warning(f"screenspot_pro missing image: {img_path}")
        return []
    doc = dict(doc)
    doc["image_path"] = str(p)
    return [Image.open(p).convert("RGB")]


def screenspot_pro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def screenspot_pro_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return gui_owl_grounding_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def screenspot_pro_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc.get("instruction", "")


def _pixel_bbox(doc: Dict[str, Any]) -> List[float]:
    bbox = doc.get("bbox") or doc.get("gt_bbox")
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    # Support x1,y1,w,h
    if x2 <= x1 or y2 <= y1:
        x2, y2 = x1 + float(bbox[2]), y1 + float(bbox[3])
    return [x1, y1, x2, y2]


def screenspot_pro_process_result(doc, result, lmms_eval_specific_kwargs=None):
    pred_text = result[0] if result else ""
    img_path = doc.get("image_path") or doc.get("img_filename")
    width = int(doc.get("img_size", [0, 0])[0] or doc.get("width", 0) or 1920)
    height = int(doc.get("img_size", [0, 0])[1] or doc.get("height", 0) or 1080)
    if img_path and (width <= 0 or height <= 0):
        try:
            from PIL import Image

            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            width, height = 1920, 1080

    point = parse_click_point_official(pred_text)
    bbox = _pixel_bbox(doc)
    hit = point_in_bbox_1000(point, bbox, width, height)
    payload = {
        "hit": hit,
        "platform": doc.get("platform", doc.get("data_source", "unknown")),
        "ui_type": doc.get("ui_type", doc.get("data_type", "unknown")),
        "instruction_style": doc.get("instruction_style", "instruction"),
        "language": doc.get("language", "en"),
    }
    return {
        "screenspot_pro_center_acc": payload,
        "screenspot_pro_action_acc": payload,
    }


def _aggregate_acc(results, key: str = "hit") -> float:
    if not results:
        return 0.0
    hits = [1.0 if r.get(key) else 0.0 for r in results]
    return sum(hits) / len(hits)


def screenspot_pro_aggregate_center_acc(results):
    return _aggregate_acc(results)


def screenspot_pro_aggregate_action_acc(results):
    return _aggregate_acc(results)
