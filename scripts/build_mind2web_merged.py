#!/usr/bin/env python3
"""Convert SeeClick-format Mind2Web annotations into TRACE merged_sorted JSON.

The original research-repo converter was not committed. This reconstruction
follows the episode-level filter documented in ``scripts/_m2w_filter_stats.py``
and the fields consumed by ``lmms_eval/tasks/gui_owl_cross/mind2web_eval.py``.

Layout (under ``data/mind2web/``):

  mind2web_annots/mind2web_data_test_{task,website,domain}.json
  ming2web_images/{annotation_id}-{action_uid}.jpg   # historical SeeClick dump spelling
  mind2web_images/...                                # accepted alias

An episode is dropped if *any* step is missing its screenshot, has a
degenerate bbox, or uses an op other than CLICK / TYPE / SELECT. Expected
kept sizes on the dump used in the paper (approximate):

  task     242 eps / 2008 steps
  website  173 eps / 1343 steps
  domain   833 eps / 5429 steps

Usage::

  python scripts/build_mind2web_merged.py
  python scripts/build_mind2web_merged.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(os.environ.get("OMINI_REPO", Path(__file__).resolve().parents[1]))
ALLOWED_OPS = {"CLICK", "TYPE", "SELECT"}


def _image_root(data_dir: Path) -> Path:
    for name in ("ming2web_images", "mind2web_images"):
        cand = data_dir / name
        if cand.is_dir():
            return cand
    return data_dir / "ming2web_images"


def _rel_image(img_root: Path, data_dir: Path, name: str) -> str:
    return str((img_root / name).relative_to(data_dir)).replace("\\", "/")


def _bbox_xyxy(bbox: dict) -> Optional[list[float]]:
    if not bbox:
        return None
    try:
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("width", 0))
        h = float(bbox.get("height", 0))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _episode_drop_reason(ep: dict, img_root: Path) -> Optional[str]:
    ann_id = str(ep.get("annotation_id") or "")
    actions = ep.get("actions") or []
    if not ann_id or not actions:
        return "bad_op"
    for act in actions:
        uid = str(act.get("action_uid") or "")
        if not uid or not (img_root / f"{ann_id}-{uid}.jpg").is_file():
            return "missing_image"
        if _bbox_xyxy(act.get("bbox") or {}) is None:
            return "no_bbox"
        op = str((act.get("operation") or {}).get("op") or "").upper()
        if op not in ALLOWED_OPS:
            return "bad_op"
    return None


def _img_size(path: Path) -> list[int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return [int(im.size[0]), int(im.size[1])]
    except Exception:
        return [1280, 720]


def _center_mille(bbox_px: list[float], size: list[int]) -> list[int]:
    x1, y1, x2, y2 = bbox_px
    w, h = max(int(size[0] or 1), 1), max(int(size[1] or 1), 1)
    cx = int(round(((x1 + x2) / 2.0) * 1000.0 / w))
    cy = int(round(((y1 + y2) / 2.0) * 1000.0 / h))
    return [max(0, min(1000, cx)), max(0, min(1000, cy))]


def _gt_action(act: dict, size: list[int]) -> dict[str, Any]:
    op = str((act.get("operation") or {}).get("op") or "CLICK").lower()
    value = str((act.get("operation") or {}).get("value") or "")
    bbox_px = _bbox_xyxy(act.get("bbox") or {}) or [0, 0, 0, 0]
    coord = _center_mille(bbox_px, size)
    out: dict[str, Any] = {
        "action_type": op,
        "coordinate": coord,
        "bbox_px": bbox_px,
    }
    if op in {"type", "select"} and value:
        out["text"] = value
    return out


def _teacher_text(action: dict[str, Any]) -> str:
    at = str(action.get("action_type") or "click")
    coord = action.get("coordinate") or [0, 0]
    args: dict[str, Any] = {
        "action": at,
        "coordinate": [int(coord[0]), int(coord[1])],
    }
    if at in {"type", "select"} and action.get("text"):
        args["text"] = str(action["text"])
    tc = {"name": "mobile_use", "arguments": args}
    summary = f"{at.capitalize()} at ({args['coordinate'][0]}, {args['coordinate'][1]})"
    return (
        f"Action: {summary}\n"
        f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
    )


def convert_split(
    split: str,
    data_dir: Path,
    *,
    strict_images: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    annot_path = data_dir / "mind2web_annots" / f"mind2web_data_test_{split}.json"
    img_root = _image_root(data_dir)
    stats = {
        "raw_eps": 0,
        "raw_steps": 0,
        "kept_eps": 0,
        "kept_steps": 0,
        "missing_image": 0,
        "no_bbox": 0,
        "bad_op": 0,
    }
    if not annot_path.is_file():
        raise FileNotFoundError(
            f"Missing {annot_path}. Place SeeClick Mind2Web test annotations "
            f"under data/mind2web/mind2web_annots/."
        )
    episodes = json.loads(annot_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    if not isinstance(episodes, list):
        return rows, stats

    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        actions = list(ep.get("actions") or [])
        stats["raw_eps"] += 1
        stats["raw_steps"] += len(actions)
        reason = _episode_drop_reason(ep, img_root) if strict_images else None
        if reason:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        ann_id = str(ep["annotation_id"])
        goal = str(ep.get("confirmed_task") or ep.get("task") or "")
        website = str(ep.get("website") or "")
        stats["kept_eps"] += 1
        stats["kept_steps"] += len(actions)
        history: list[dict] = []
        n_steps = len(actions)
        for step_idx, act in enumerate(actions):
            uid = str(act["action_uid"])
            img_name = f"{ann_id}-{uid}.jpg"
            img_path = _rel_image(img_root, data_dir, img_name)
            size = _img_size(img_root / img_name)
            gt = _gt_action(act, size)
            row = {
                "episode_id": ann_id,
                "annotation_id": ann_id,
                "step_index": step_idx,
                "step_id": step_idx,
                "episode_length": n_steps,
                "subset": f"mind2web_{split}",
                "goal": goal,
                "instruction": goal,
                "step_instruction": goal,
                "website": website,
                "image_path": img_path,
                "img_url": img_name,
                "image_width": size[0],
                "image_height": size[1],
                "img_size": size,
                "action": dict(gt),
                "gt_action": dict(gt),
                "action_history": [dict(h) for h in history],
                "app": "mind2web",
                "task_category": split,
                "task_category_full": split,
            }
            rows.append(row)
            history.append(
                {
                    "step_index": step_idx,
                    "image_path": img_path,
                    "action": dict(gt),
                    "action_text": _teacher_text(gt),
                }
            )
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO))
    parser.add_argument(
        "--splits",
        default="task,website,domain",
        help="comma-separated SeeClick test splits",
    )
    parser.add_argument(
        "--no-strict-images",
        action="store_true",
        help="keep episodes even if a screenshot is missing (not the paper protocol)",
    )
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    data_dir = repo / "data" / "mind2web"
    data_dir.mkdir(parents=True, exist_ok=True)

    any_ok = False
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        try:
            rows, stats = convert_split(
                split, data_dir, strict_images=not args.no_strict_images
            )
        except FileNotFoundError as exc:
            print(f"WARN: {exc}", file=sys.stderr)
            continue
        out = data_dir / f"merged_sorted_{split}.json"
        out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        any_ok = True
        print(
            f"[{split}] raw {stats['raw_eps']} eps/{stats['raw_steps']} steps "
            f"-> kept {stats['kept_eps']} eps/{stats['kept_steps']} steps "
            f"| dropped eps: missing_image={stats['missing_image']} "
            f"no_bbox={stats['no_bbox']} bad_op={stats['bad_op']} -> {out}"
        )
    if not any_ok:
        print("ERROR: no Mind2Web split converted. See docs/datasets.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
