#!/usr/bin/env python3
"""Pre-flight for cross-benchmark GUI-Owl eval (OmniGUI 8B + offline datasets)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    return name, ok, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true", help="OmniGUI + 8B model only")
    parser.add_argument("--require-8b", action="store_true", help="Require GUI-Owl-1.5-8B-Instruct weights")
    parser.add_argument("--offline-only", action="store_true", help="Also verify offline cross-benchmark data (loose)")
    parser.add_argument(
        "--offline-matrix",
        action="store_true",
        help="Strict verify all offline datasets + 8B + restore_q3vl patch prerequisites",
    )
    args = parser.parse_args()

    if args.offline_matrix:
        args.require_8b = True
        args.offline_only = True

    checks: list[tuple[str, bool, str]] = []

    merged = REPO / "data" / "merged_sorted.json"
    checks.append(_check("merged_sorted.json", merged.is_file(), str(merged)))
    if merged.is_file():
        docs = json.loads(merged.read_text(encoding="utf-8"))
        checks.append(_check("doc count == 2572", len(docs) == 2572, str(len(docs))))

    yaml_8b = REPO / "lmms_eval/tasks/gui_owl_cross/task_guiowl_omnigui_native_mt4_8b.yaml"
    checks.append(_check("8b task yaml", yaml_8b.is_file(), yaml_8b.name))

    model_8b = REPO / "models" / "GUI-Owl-1.5-8B-Instruct"
    config_json = model_8b / "config.json"
    if args.require_8b or not args.baseline_only:
        checks.append(
            _check(
                "GUI-Owl-1.5-8B-Instruct",
                config_json.is_file(),
                str(model_8b) if config_json.is_file() else "run: python scripts/download_omini_models.py --models-dir models GUI-Owl-1.5-8B-Instruct",
            )
        )

    if args.offline_only or not args.baseline_only:
        from scripts.verify_cross_benchmark_data import verify_all  # noqa: WPS433

        strict_data = bool(args.offline_matrix)
        offline = verify_all(repo=REPO, strict=strict_data)
        for name, ok, detail in offline:
            checks.append(_check(name, ok, detail))

    if args.offline_matrix:
        patch_src = REPO / "patches" / "qwen3_vl" / "modeling_qwen3_vl.py"
        restore_py = REPO / "scripts" / "restore_q3vl_geopact_patches.py"
        checks.append(
            _check(
                "restore_q3vl_geopact_patches",
                restore_py.is_file() and patch_src.is_file(),
                str(patch_src) if patch_src.is_file() else "missing patches/qwen3_vl/modeling_qwen3_vl.py",
            )
        )

    if args.baseline_only:
        merged_ok = merged.is_file() and len(json.loads(merged.read_text(encoding="utf-8"))) == 2572
        bench_ok = (REPO / "data" / "benchmark").is_dir()
        checks.append(_check("omnigui media gate", merged_ok and bench_ok, "merged_sorted + benchmark/"))

    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"\nPREFLIGHT FAILED ({len(failed)} checks)", file=sys.stderr)
        return 1
    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
