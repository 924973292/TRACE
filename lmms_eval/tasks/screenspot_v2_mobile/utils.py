"""ScreenSpot-V2 Mobile domain grounding eval for GUI-Owl (phone screenshots only).

Protocol: center_acc — predicted click center inside GT bbox (pixel coords normalized per image).
Official split: OS-Copilot/ScreenSpot-v2 screenspot_mobile_v2.json (~502 samples).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.gui_owl_cross.utils import (
    gui_owl_grounding_doc_to_messages,
    parse_click_point_normalized,
    point_in_bbox_norm,
)


def screenspot_v2_mobile_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/screenspot_v2_mobile/images")
    img_path = doc.get("image_path") or doc.get("img_filename") or doc.get("file_name")
    if not img_path:
        return []
    p = Path(str(img_path))
    if not p.is_file():
        p = Path(media_dir) / p.name
    if not p.is_file():
        eval_logger.warning(f"screenspot_v2_mobile missing image: {img_path}")
        return []
    doc = dict(doc)
    doc["image_path"] = str(p)
    return [Image.open(p).convert("RGB")]


def screenspot_v2_mobile_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def screenspot_v2_mobile_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return gui_owl_grounding_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def screenspot_v2_mobile_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc.get("instruction", "")


def _norm_bbox(doc: Dict[str, Any], width: int, height: int) -> List[float]:
    bbox = doc.get("bbox") or doc.get("gt_bbox")
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if x2 <= x1 or y2 <= y1:
        x2, y2 = x1 + float(bbox[2]), y1 + float(bbox[3])
    w = max(width, 1)
    h = max(height, 1)
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def screenspot_v2_mobile_process_result(doc, result):
    pred_text = result[0] if result else ""
    img_path = doc.get("image_path") or doc.get("img_filename")
    width = int(doc.get("img_size", [0, 0])[0] or doc.get("width", 0) or 0)
    height = int(doc.get("img_size", [0, 0])[1] or doc.get("height", 0) or 0)
    if img_path and (width <= 0 or height <= 0):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            width, height = 1080, 2400

    point = parse_click_point_normalized(pred_text, scale=1000.0)
    bbox = _norm_bbox(doc, width, height)
    hit = bool(point and point_in_bbox_norm(point, bbox))
    payload = {
        "hit": hit,
        "platform_domain": "mobile",
        "data_type": doc.get("data_type", "unknown"),
        "data_source": doc.get("data_source", "unknown"),
    }
    return {
        "screenspot_v2_mobile_center_acc": payload,
    }


def screenspot_v2_mobile_aggregate_center_acc(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("hit") else 0.0 for r in results) / len(results)
