#!/usr/bin/env python3
"""Pre-flight gate before launching GUI-Owl OmniGUI eval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "PLACEHOLDER"


def _default_omniparser_weights() -> str:
    env_w = os.environ.get("OMNIPARSER_ICON_WEIGHTS") or os.environ.get("MIU_YOLO_WEIGHTS")
    if env_w:
        return env_w
    return str(REPO / "models/omniparser/icon_detect/model.pt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pre-flight gate before GUI-Owl eval.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip Module A (YOLO/cv2) checks — baseline prompt eval only.",
    )
    args = parser.parse_args()

    merged = REPO / "data" / "merged_sorted.json"
    benchmark = REPO / "data" / "benchmark"
    yaml_a = REPO / "lmms_eval/tasks/agentcpm_gui/task_v2_guiowl_imageonly_a.yaml"
    yaml_k5 = REPO / "lmms_eval/tasks/agentcpm_gui/task_v2_guiowl_imageonly_a_hist_k5.yaml"

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    check("merged_sorted.json exists", merged.is_file(), str(merged))
    digest = _sha256(merged) if merged.is_file() else ""
    check("benchmark/ exists", benchmark.is_dir(), str(benchmark))

    docs = json.loads(merged.read_text(encoding="utf-8")) if merged.is_file() else []
    check("doc count == 2572", len(docs) == 2572, str(len(docs)))

    sys.path.insert(0, str(REPO / "scripts"))
    from audit_merged_sorted import audit  # noqa: E402

    report = audit(docs, benchmark)
    check("action_history bad paths == 0", report["history_bad_count"] == 0, str(report["history_bad_count"]))
    check("missing primary media == 0", report["missing_primary_count"] == 0, str(report["missing_primary_count"]))

    for ypath in (yaml_a, yaml_k5):
        text = ypath.read_text(encoding="utf-8") if ypath.is_file() else ""
        check(
            f"{ypath.name} uses merged_sorted.json",
            'data_files: "data/merged_sorted.json"' in text,
            ypath.name,
        )

    spec = importlib.util.spec_from_file_location("utils", REPO / "lmms_eval/tasks/agentcpm_gui/utils.py")
    utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(utils)
    kwargs = {
        "media_dir": "data/benchmark",
        "coordinate_type": "normalized",
        "history_image_mode": "fifo",
        "max_history_images": 5,
    }
    crash_doc = None
    for i, doc in enumerate(docs):
        try:
            utils.ablation_v2_no_av_hist_k5_doc_to_messages(doc, kwargs)
        except Exception as exc:
            crash_doc = (i, doc.get("episode_id"), exc)
            break
    check(
        "hist_k5 builds all docs",
        crash_doc is None,
        "none" if crash_doc is None else f"doc_id={crash_doc[0]} ep={crash_doc[1]} err={crash_doc[2]}",
    )

    # Spot-check previously failing T0271 step 1 (doc ~869)
    t0271 = [i for i, d in enumerate(docs) if d.get("episode_id") == "T0271" and d.get("step_index") == 1]
    if t0271:
        doc = docs[t0271[0]]
        hist = (doc.get("action_history") or [{}])[0].get("image_path", "")
        check(
            "T0271 history path fixed",
            hist == "douyin/media/T0144/1.png",
            hist,
        )

    print(f"merged_sorted sha256: {digest}")

    if args.baseline_only:
        failed = [name for name, ok, _ in checks if not ok]
        if failed:
            print("PREFLIGHT FAILED:", ", ".join(failed))
            raise SystemExit(1)
        print("PREFLIGHT PASSED (baseline-only) — safe to launch baseline eval.")
        return

    # Module A (YOLO) requires headless OpenCV + OmniParser weights.
    try:
        import cv2  # noqa: F401

        check("opencv (cv2) import", True, getattr(cv2, "__version__", "?"))
    except Exception as exc:
        check(
            "opencv (cv2) import",
            False,
            f"{exc} — run: pip uninstall opencv-python; pip install opencv-python-headless==4.8.1.78",
        )

    yolo_weights = Path(
        os.environ.get("OMNIPARSER_ICON_WEIGHTS", _default_omniparser_weights())
    )
    check("YOLO weights (OMNIPARSER_ICON_WEIGHTS)", yolo_weights.is_file(), str(yolo_weights))

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print("PREFLIGHT FAILED:", ", ".join(failed))
        raise SystemExit(1)
    print("PREFLIGHT PASSED — safe to launch eval.")


if __name__ == "__main__":
    main()
