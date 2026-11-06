#!/usr/bin/env python3
"""Verify local cross-benchmark dataset layout (offline eval splits only)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_DATASETS = {
    "omnigui": "XIAOCHENLIN00zz/OmniGUI",
    "screenspot_v2": "OS-Copilot/ScreenSpot-v2",
    "screenspot_pro": "likaixin/ScreenSpot-Pro",
    "mmbench_gui": "OpenGVLab/MMBench-GUI",
    "android_control_json": "endlesswinds/AndroidControl-JSON-Splits",
}


def _count_json_array(path: Path) -> int:
    if not path.is_file():
        return -1
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "episodes" in data:
        return len(data["episodes"])
    return len(data)


def verify_omnigui(repo: Path = REPO_ROOT) -> tuple[str, bool, str]:
    merged = repo / "data" / "merged_sorted.json"
    benchmark = repo / "data" / "benchmark"
    ok = merged.is_file() and benchmark.is_dir()
    n = _count_json_array(merged) if merged.is_file() else 0
    detail = f"docs={n}, benchmark_dir={benchmark.is_dir()}"
    return "omnigui", ok and n == 2572, detail


def verify_screenspot_v2(repo: Path = REPO_ROOT) -> tuple[str, bool, str]:
    root = repo / "data" / "screenspot_v2"
    manifest = root / "manifest.json"
    eval_index = root / "eval_index.json"
    images = root / "images"
    ok = manifest.is_file() and eval_index.is_file() and images.is_dir()
    n = _count_json_array(eval_index) if eval_index.is_file() else 0
    return "screenspot_v2", ok and n >= 1200, f"samples={n}"


def verify_screenspot_pro(repo: Path = REPO_ROOT) -> tuple[str, bool, str]:
    root = repo / "data" / "screenspot_pro"
    manifest = root / "manifest.json"
    eval_index = root / "eval_index.json"
    images = root / "images"
    ok = manifest.is_file() and eval_index.is_file() and images.is_dir()
    n = 0
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        n = int(meta.get("num_samples", 0))
    return "screenspot_pro", ok and n > 0, f"samples={n}"


def verify_mmbench_gui(repo: Path = REPO_ROOT) -> tuple[str, bool, str]:
    root = repo / "data" / "mmbench_gui"
    l1 = root / "l1_eval_index.json"
    l2 = root / "l2_eval_index.json"
    images = root / "offline_images"
    n1 = _count_json_array(l1) if l1.is_file() else 0
    n2 = _count_json_array(l2) if l2.is_file() else 0
    ok = l1.is_file() and l2.is_file() and images.is_dir() and n1 > 1000 and n2 > 3000
    return "mmbench_gui", ok, f"l1={n1} l2={n2}"


def verify_android_control(repo: Path = REPO_ROOT) -> tuple[str, bool, str]:
    test_json = repo / "data" / "android_control" / "test-AndroidControl.json"
    eval_steps = repo / "data" / "android_control" / "eval_steps.json"
    images = repo / "data" / "android_control" / "images"
    ok_json = test_json.is_file() and eval_steps.is_file()
    n = _count_json_array(test_json) if test_json.is_file() else 0
    n_steps = _count_json_array(eval_steps) if eval_steps.is_file() else 0
    n_img = len(list(images.glob("*.png"))) if images.is_dir() else 0
    ok = ok_json and n > 0 and n_steps > 0
    detail = f"episodes={n} steps={n_steps} images={n_img}"
    return "android_control_test", ok, detail


def verify_all(*, repo: Path = REPO_ROOT, strict: bool = True) -> list[tuple[str, bool, str]]:
    checks = [
        verify_omnigui(repo),
        verify_screenspot_v2(repo),
        verify_screenspot_pro(repo),
        verify_mmbench_gui(repo),
        verify_android_control(repo),
    ]
    if strict:
        return checks
    optional = {
        "screenspot_v2",
        "screenspot_pro",
        "mmbench_gui",
        "android_control_test",
    }
    out: list[tuple[str, bool, str]] = []
    for name, ok, detail in checks:
        if name == "omnigui":
            out.append((name, ok, detail))
        elif name in optional and not ok:
            out.append((name, True, f"optional missing ({detail})"))
        else:
            out.append((name, ok, detail))
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Require all offline datasets")
    parser.add_argument("--repo-root", default="", help="override repo root (e.g. $REPO)")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT

    failed = []
    for name, ok, detail in verify_all(repo=repo, strict=args.strict):
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nverify_cross_benchmark_data FAILED: {failed}", file=__import__("sys").stderr)
        return 1
    print("\nverify_cross_benchmark_data OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
