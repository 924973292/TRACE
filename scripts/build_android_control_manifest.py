#!/usr/bin/env python3
"""Build AndroidControl manifests for dialogue-KV / OmniGUI-style flat loading."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(os.environ.get("OMINI_REPO", Path(__file__).resolve().parents[1]))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HF_ROOT = REPO / "data" / "android_control_hf"
OUT = REPO / "data" / "android_control"
IMAGES_DIR = OUT / "images"


def _rel_image_path(img_name: str) -> str:
    return f"images/{Path(img_name).name}".replace("\\", "/")


def _normalize_image_path(row: dict) -> str:
    raw = row.get("image_path") or row.get("img_url") or ""
    name = Path(str(raw)).name
    if name:
        return _rel_image_path(name)
    return ""


def _to_merged_row(
    row: dict,
    *,
    step_idx: int,
    episode_length: int,
    action_history: list[dict],
) -> dict:
    img_size = row.get("img_size") or [1080, 2400]
    w, h = int(img_size[0] or 1080), int(img_size[1] or 2400)
    goal = str(row.get("goal") or row.get("instruction") or "")
    subset = str(row.get("subset") or "android_control_high")
    episode_id = str(row.get("episode_id", "unknown"))
    image_path = _normalize_image_path(row)
    action = dict(row.get("action") or row.get("gt_action") or {})
    merged = {
        "episode_id": episode_id,
        "step_index": step_idx,
        "step_id": step_idx,
        "episode_length": episode_length,
        "subset": subset,
        "goal": goal,
        "instruction": goal,
        "step_instruction": str(row.get("instruction") or ""),
        "image_path": image_path,
        "img_url": Path(image_path).name if image_path else "",
        "image_width": w,
        "image_height": h,
        "img_size": [w, h],
        "action": action,
        "gt_action": action,
        "action_history": [dict(h) for h in action_history],
        "app": "android_control",
        "task_category": subset,
        "task_category_full": subset,
    }
    return merged


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = HF_ROOT / "test-AndroidControl.json"
    if not src.is_file():
        for p in OUT.rglob("test-AndroidControl.json"):
            src = p
            break
        if not src.is_file():
            cand = OUT / "test-AndroidControl.json"
            if cand.is_file():
                src = cand
    if not src.is_file():
        print("WARN: test-AndroidControl.json not found; download endlesswinds/AndroidControl-JSON-Splits first")
        return 1

    dst = OUT / "test-AndroidControl.json"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    data = json.loads(dst.read_text(encoding="utf-8"))
    n_ep = len(data) if isinstance(data, list) else 0

    from lmms_eval.tasks.gui_owl_cross.android_control_eval import ac_gt_to_teacher_assistant_text

    steps: list[dict] = []
    merged_rows: list[dict] = []
    if isinstance(data, list):
        for ep_idx, episode in enumerate(data):
            if not isinstance(episode, list):
                continue
            ep_history: list[dict] = []
            episode_length = len(episode)
            for step_idx, step in enumerate(episode):
                if not isinstance(step, dict):
                    continue
                row = dict(step)
                row["episode_id"] = row.get("episode_id", f"ac_test_{ep_idx}")
                row["step_index"] = step_idx
                row["step_id"] = step_idx
                row.setdefault("subset", "android_control_high")
                img_name = Path(str(row.get("img_url", ""))).name
                if img_name:
                    row["image_path"] = _rel_image_path(img_name)
                row["action_history"] = [dict(h) for h in ep_history]
                steps.append(row)
                merged_rows.append(
                    _to_merged_row(
                        row,
                        step_idx=step_idx,
                        episode_length=episode_length,
                        action_history=ep_history,
                    )
                )

                gt_action = row.get("action") or row.get("gt_action") or {}
                if isinstance(gt_action, dict) and gt_action:
                    size = row.get("img_size") or [1080, 2400]
                    screen = (int(size[0] or 1080), int(size[1] or 2400))
                    hist_path = row.get("image_path")
                    ep_history.append(
                        {
                            "step_index": step_idx,
                            "image_path": hist_path,
                            "action": dict(gt_action),
                            "action_text": ac_gt_to_teacher_assistant_text(
                                gt_action, screen_size=screen
                            ),
                        }
                    )

    steps_path = OUT / "eval_steps.json"
    merged_path = OUT / "merged_sorted.json"
    steps_path.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"android_control: {n_ep} test episodes -> {dst}")
    print(f"android_control: {len(steps)} steps (with action_history) -> {steps_path}")
    print(f"android_control: {len(merged_rows)} OmniGUI-style rows -> {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
