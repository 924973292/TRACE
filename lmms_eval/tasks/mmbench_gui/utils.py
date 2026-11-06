"""MMBench-GUI offline eval for GUI-Owl (L1 QA + L2 grounding)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.gui_owl_cross.utils import (
    parse_click_point_official,
    point_in_bbox_1000,
)

# Plain VQA protocol for L1 content-understanding MCQ (no agent tool schema,
# which previously made the model emit `answer` actions on 55.8% of items).
MMBENCH_L1_QA_SYSTEM_PROMPT = "You are a helpful assistant."


def _lmms_text(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _lmms_image(path: str) -> Dict[str, str]:
    return {"type": "image", "url": path}


def _resolve_image(doc: Dict[str, Any], media_dir: str) -> str:
    img_path = doc.get("image_path") or ""
    if img_path and Path(str(img_path)).is_file():
        return str(img_path)
    name = Path(str(img_path)).name if img_path else ""
    if name:
        for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
            for base in (Path(media_dir), root / media_dir):
                p = base / name
                if p.is_file():
                    return str(p)
                for hit in base.rglob(name):
                    if hit.is_file():
                        return str(hit)
            if img_path:
                cand = root / str(img_path)
                if cand.is_file():
                    return str(cand)
    return ""


def mmbench_gui_l1_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    kw = lmms_eval_specific_kwargs or {}
    media_dir = kw.get("media_dir", "data/mmbench_gui/offline_images")
    img = _resolve_image(doc, media_dir)
    if not img:
        eval_logger.warning(f"mmbench_gui_l1 missing image: {doc.get('image_path')}")
        return []
    return [Image.open(img).convert("RGB")]


def mmbench_gui_l1_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def mmbench_gui_l1_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    kw = lmms_eval_specific_kwargs or {}
    media_dir = kw.get("media_dir", "data/mmbench_gui/offline_images")
    img = _resolve_image(doc, media_dir)
    opts = doc.get("options") or {}
    lines = [f"Question: {doc.get('question', '')}", "Options:"]
    for key in sorted(opts.keys()):
        lines.append(f"{key}. {opts[key]}")
    lines.append("Answer with the option letter only (A/B/C/D/E).")
    user_text = "\n".join(lines)
    content: List[Dict[str, Any]] = [_lmms_text(user_text)]
    if img:
        content.insert(0, _lmms_image(img))
    return [
        {"role": "system", "content": [_lmms_text(MMBENCH_L1_QA_SYSTEM_PROMPT)]},
        {"role": "user", "content": content},
    ]


def mmbench_gui_l1_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return str(doc.get("answer", "")).strip().upper()


def _parse_option_letter(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    m2 = re.search(r"answer[\"']?\s*[:=]\s*[\"']?([A-E])", text, re.I)
    if m2:
        return m2.group(1).upper()
    return ""


def mmbench_gui_l1_process_result(doc, result, lmms_eval_specific_kwargs=None):
    pred = _parse_option_letter(result[0] if result else "")
    gt = str(doc.get("answer", "")).strip().upper()
    correct = bool(pred and gt and pred == gt)
    n_opts = len(doc.get("options") or {})
    return {
        "mmbench_gui_l1_acc": {
            "correct": correct,
            "platform": doc.get("platform", ""),
            "difficulty": doc.get("difficulty", ""),
            "n_options": n_opts,
        }
    }


def mmbench_gui_l1_aggregate_acc(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("correct") else 0.0 for r in results) / len(results)


def mmbench_gui_l1_aggregate_acc_weighted(results):
    """MMBench-GUI L1 overall acc with per-question alpha = 1/num_options (paper Eq.4)."""
    if not results:
        return 0.0
    total_w = 0.0
    total_s = 0.0
    for r in results:
        n_opts = max(int(r.get("n_options", 4)), 1)
        w = 1.0 / n_opts
        total_w += w
        total_s += w * (1.0 if r.get("correct") else 0.0)
    return total_s / total_w if total_w else 0.0


def mmbench_gui_l1_aggregate_acc_by_platform(results):
    """Macro average of per-platform weighted accuracies."""
    if not results:
        return 0.0
    by_platform: dict[str, list] = {}
    for r in results:
        plat = str(r.get("platform", "unknown"))
        by_platform.setdefault(plat, []).append(r)
    per_plat = [mmbench_gui_l1_aggregate_acc_weighted(rows) for rows in by_platform.values()]
    return sum(per_plat) / len(per_plat) if per_plat else 0.0


def mmbench_gui_l2_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    kw = lmms_eval_specific_kwargs or {}
    media_dir = kw.get("media_dir", "data/mmbench_gui/offline_images")
    img = _resolve_image(doc, media_dir)
    if not img:
        eval_logger.warning(f"mmbench_gui_l2 missing image: {doc.get('image_path')}")
        return []
    return [Image.open(img).convert("RGB")]


def mmbench_gui_l2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def mmbench_gui_l2_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.gui_owl_cross.utils import gui_owl_grounding_doc_to_messages

    entry = dict(doc)
    kw = lmms_eval_specific_kwargs or {}
    media_dir = kw.get("media_dir", "data/mmbench_gui/offline_images")
    img = _resolve_image(doc, media_dir)
    if img:
        entry["image_path"] = img
    return gui_owl_grounding_doc_to_messages(entry, kw)


def mmbench_gui_l2_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return doc.get("instruction", "")


def mmbench_gui_l2_process_result(doc, result, lmms_eval_specific_kwargs=None):
    pred_text = result[0] if result else ""
    img_path = doc.get("image_path")
    size = doc.get("image_size") or [1920, 1080]
    width, height = int(size[0] or 1920), int(size[1] or 1080)
    if img_path and (width <= 0 or height <= 0):
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            pass

    point = parse_click_point_official(pred_text)
    bbox = doc.get("bbox") or [0, 0, 0, 0]
    if len(bbox) >= 4:
        x1, y1, x2, y2 = map(float, bbox[:4])
        if max(x1, y1, x2, y2) <= 1.5:
            # L2 annotations are normalized xyxy; convert to pixels first
            # (official converts via image_size before the 1000-space scaling).
            pixel_bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
        else:
            pixel_bbox = [x1, y1, x2, y2]
    else:
        pixel_bbox = [0.0, 0.0, 0.0, 0.0]

    hit = point_in_bbox_1000(point, pixel_bbox, width, height)
    return {
        "mmbench_gui_l2_acc": {
            "hit": hit,
            "platform": doc.get("platform", ""),
            "grounding_type": doc.get("grounding_type", ""),
        }
    }


def mmbench_gui_l2_aggregate_acc(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("hit") else 0.0 for r in results) / len(results)


def mmbench_gui_l2_aggregate_acc_by_type(results):
    if not results:
        return 0.0
    by: dict[str, list] = {}
    for r in results:
        t = str(r.get("grounding_type", "unknown"))
        by.setdefault(t, []).append(r)
    per = [sum(1.0 if x.get("hit") else 0.0 for x in rows) / len(rows) for rows in by.values()]
    return sum(per) / len(per) if per else 0.0
