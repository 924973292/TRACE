"""ScreenSpot-V2 full (Mobile+Web+Desktop) grounding eval for GUI-Owl."""

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


def screenspot_v2_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/screenspot_v2/images")
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
        eval_logger.warning(f"screenspot_v2 missing image: {img_path}")
        return []
    doc = dict(doc)
    doc["image_path"] = str(p)
    return [Image.open(p).convert("RGB")]


def screenspot_v2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def screenspot_v2_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return gui_owl_grounding_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def screenspot_v2_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc.get("instruction", "")


def _pixel_bbox(doc: Dict[str, Any], width: int, height: int) -> List[float]:
    """Return GT bbox as pixel-space [x1,y1,x2,y2].

    ScreenSpot-v2 annotations are ALWAYS [x, y, w, h] (OS-Copilot format; the
    official eval does bbox[2]+=bbox[0], bbox[3]+=bbox[1]). Never apply an
    xyxy-vs-xywh heuristic here: elements with w>x / h>y would be silently
    misread as corners, truncating the GT region.
    """
    bbox = doc.get("bbox") or doc.get("gt_bbox")
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x, y, w, h = (float(v) for v in bbox[:4])
    if max(x, y, w, h) <= 1.5:  # normalized x,y,w,h
        return [x * width, y * height, (x + w) * width, (y + h) * height]
    return [x, y, x + w, y + h]


def screenspot_v2_process_result(doc, result, lmms_eval_specific_kwargs=None):
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

    point = parse_click_point_official(pred_text)
    bbox = _pixel_bbox(doc, width, height)
    hit = point_in_bbox_1000(point, bbox, width, height)
    payload = {
        "hit": hit,
        "domain": doc.get("domain", doc.get("platform_domain", "unknown")),
        "data_type": doc.get("data_type", ""),
        "data_source": doc.get("data_source", ""),
    }
    return {"screenspot_v2_center_acc": payload}


def screenspot_v2_aggregate_center_acc(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("hit") else 0.0 for r in results) / len(results)


def screenspot_v2_aggregate_center_acc_by_domain(results):
    """Macro average across mobile/web/desktop."""
    if not results:
        return 0.0
    by_domain: dict[str, list] = {}
    for r in results:
        d = str(r.get("domain", "unknown"))
        by_domain.setdefault(d, []).append(r)
    if not by_domain:
        return 0.0
    per_domain = [sum(1.0 if x.get("hit") else 0.0 for x in rows) / len(rows) for rows in by_domain.values()]
    return sum(per_domain) / len(per_domain)
