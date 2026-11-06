#!/usr/bin/env python3
"""Verify GUI-Owl / UI-TARS eval path: ConfigurableMessagesTask + official prompt.

Run from repo root (where data/benchmark exists):
  GUIOWL_DEBUG_MESSAGES=1 python scripts/verify_guiowl_eval_path.py
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _load_utils():
    import importlib.util
    path = os.path.join(REPO, "lmms_eval", "tasks", "agentcpm_gui", "utils.py")
    spec = importlib.util.spec_from_file_location("agentcpm_gui_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main() -> None:
    utils = _load_utils()
    from lmms_eval.models.simple.qwen_vl_omnigui_common import convert_messages_to_qwen

    # GUIOwl class check without importing torch stack
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen3_vl_gui",
        os.path.join(REPO, "lmms_eval", "models", "simple", "qwen3_vl_gui.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
    except ModuleNotFoundError as exc:
        print(f"SKIP model import ({exc}); checking utils path only")
        mod = None
    if mod is not None:
        assert mod.GUIOwl.is_simple is False, "GUIOwl.is_simple must be False"
        print("OK: GUIOwl.is_simple=False")

    media_dir = os.path.join(REPO, "data", "benchmark")
    doc_path = os.path.join(REPO, "data", "merged_sorted.json")
    if not os.path.isfile(doc_path):
        print(f"SKIP: missing {doc_path}")
        return

    docs = json.load(open(doc_path, encoding="utf-8"))
    doc = docs[0]
    kwargs = {"media_dir": "data/benchmark", "coordinate_type": "normalized"}

    raw = utils.ablation_v2_no_av_doc_to_messages(doc, kwargs)
    assert raw[0]["role"] == "system", "first message must be system"
    sys_text = raw[0]["content"]
    if isinstance(sys_text, list):
        sys_text = sys_text[0].get("text", "")
    assert "action_type" in sys_text and "TASK_COMPLETE" in sys_text, "UNIFIED system prompt expected"
    assert "PRESS_HOME" not in sys_text, "wrapper prompt must not appear"

    from lmms_eval.models.simple.qwen_vl_omnigui_common import resolve_media_path

    qwen = convert_messages_to_qwen(raw)
    paths = [
        p.get("image")
        for m in qwen
        for p in (m.get("content") or [])
        if isinstance(p, dict) and p.get("type") == "image"
    ]
    assert paths, "no images after conversion"
    for p in paths:
        assert os.path.isabs(p), f"path must be absolute: {p}"
        assert os.path.isfile(p), f"image missing: {p}"

    print(f"OK: {len(paths)} image(s), first={paths[0]}")
    print("OK: ablation_v2_no_av uses official UNIFIED_SYSTEM_PROMPT (8=BACK, 10=TASK_COMPLETE)")

    # Coordinate sanity for T0001 step 0 (飙升榜 bottom tab)
    pred = [568, 727]
    gt_info = utils.parse_ground_truth_coordinate(doc)
    w, h = doc.get("image_width", 540), doc.get("image_height", 1200)
    norm_rect = [
        utils.auto_detect_and_normalize_coordinate(gt_info["rect"][0], w, h, "pixel"),
        utils.auto_detect_and_normalize_coordinate(gt_info["rect"][1], w, h, "pixel"),
    ]
    norm_pred = utils.auto_detect_and_normalize_coordinate(pred, w, h, "normalized")
    inside = utils.is_point_in_rect(norm_pred, norm_rect)
    print(
        f"COORD_CHECK episode={doc.get('episode_id')} gt_rect_norm={norm_rect} "
        f"pred_norm={norm_pred} inside={inside} (expected False for [568,727] sample)"
    )


if __name__ == "__main__":
    main()
