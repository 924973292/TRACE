#!/usr/bin/env python3
"""Collect cross-benchmark JSON/log outputs into docs/cross_benchmark_baseline_8b.md."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "cross_benchmark_baseline_8b.md"

OFFICIAL = {
    "omnigui": {"metric": "GP/TM/EM", "official": "OmniGUI leaderboard"},
    "screenspot_pro": {"metric": "center_acc", "official": "GUI-Owl ScreenSpot-Pro report"},
    "screenspot_v2": {"metric": "center_acc", "official": "GUI-Owl ScreenSpot-v2 report"},
    "mmbench_l1": {"metric": "l1_acc", "official": "MMBench-GUI L1"},
    "mmbench_l2": {"metric": "l2_acc", "official": "MMBench-GUI L2"},
    "android_control": {"metric": "exact_match", "official": "AndroidControl test"},
    "androidworld": {"metric": "task_sr", "official": "69.0%"},
}


def _grep_overall(log_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not log_path.is_file():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("OVERALL ALL"):
            for m in re.finditer(r"(\w+)=([\d.]+)%?", line):
                out[m.group(1).lower()] = m.group(2)
    return out


def _read_lmms_result(out_dir: Path, metric_key: str) -> str:
    for p in sorted(out_dir.glob("**/results.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metric_key in data:
            val = data[metric_key]
            if isinstance(val, (int, float)):
                return f"{100.0 * val:.2f}%"
            return str(val)
    return "—"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnigui-out", default="outputs/guiowl_native_mt4_8b_full_baseline")
    parser.add_argument("--screenspot-pro-out", default="outputs/screenspot_pro_guiowl_8b")
    parser.add_argument("--screenspot-v2-out", default="outputs/screenspot_v2_guiowl_8b")
    parser.add_argument("--mmbench-l1-out", default="outputs/mmbench_gui_l1_guiowl_8b")
    parser.add_argument("--mmbench-l2-out", default="outputs/mmbench_gui_l2_guiowl_8b")
    parser.add_argument("--android-control-out", default="outputs/android_control_guiowl_8b")
    parser.add_argument("--androidworld-log", default="")
    args = parser.parse_args()

    rows = []
    omni_dir = REPO / args.omnigui_out
    omni = _grep_overall(omni_dir / "run.log")
    rows.append(
        (
            "OmniGUI (image)",
            "GP / TM / EM",
            f"{omni.get('gp', '—')} / {omni.get('tm', '—')} / {omni.get('em', '—')}",
            OFFICIAL["omnigui"]["official"],
            str(omni_dir),
        )
    )

    rows.append(
        (
            "ScreenSpot-Pro",
            "center_acc",
            _read_lmms_result(REPO / args.screenspot_pro_out, "screenspot_pro_center_acc"),
            OFFICIAL["screenspot_pro"]["official"],
            args.screenspot_pro_out,
        )
    )
    rows.append(
        (
            "ScreenSpot-v2",
            "center_acc",
            _read_lmms_result(REPO / args.screenspot_v2_out, "screenspot_v2_center_acc"),
            OFFICIAL["screenspot_v2"]["official"],
            args.screenspot_v2_out,
        )
    )
    rows.append(
        (
            "MMBench-GUI L1",
            "qa_acc",
            _read_lmms_result(REPO / args.mmbench_l1_out, "mmbench_gui_l1_acc"),
            OFFICIAL["mmbench_l1"]["official"],
            args.mmbench_l1_out,
        )
    )
    rows.append(
        (
            "MMBench-GUI L2",
            "grounding_acc",
            _read_lmms_result(REPO / args.mmbench_l2_out, "mmbench_gui_l2_acc"),
            OFFICIAL["mmbench_l2"]["official"],
            args.mmbench_l2_out,
        )
    )
    rows.append(
        (
            "AndroidControl test",
            "exact_match",
            _read_lmms_result(REPO / args.android_control_out, "android_control_exact_match"),
            OFFICIAL["android_control"]["official"],
            args.android_control_out,
        )
    )

    aw_log = args.androidworld_log or "third_party/MobileAgent/Mobile-Agent-v3.5/android_world/results_guiowl15.log"
    rows.append(("AndroidWorld", "task_sr", "—", OFFICIAL["androidworld"]["official"], aw_log))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Cross-benchmark baseline — GUI-Owl-1.5-8B-Instruct",
        "",
        f"_Updated: {ts}_",
        "",
        "| Benchmark | Native metric | Measured (8B) | Official ref | Output |",
        "|-----------|---------------|---------------|--------------|--------|",
    ]
    for bench, metric, measured, official, out in rows:
        lines.append(f"| {bench} | {metric} | {measured} | {official} | `{out}` |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Matrix: OmniGUI image, ScreenSpot-Pro, ScreenSpot-v2, AndroidControl, MMBench-GUI L1/L2, AndroidWorld.",
            "- Offline: `lmms_eval` + `gui_owl`; online AndroidWorld: vLLM + MobileAgent.",
            "- Re-run: `python scripts/cross_benchmark/collect_cross_benchmark_results.py`",
            "",
        ]
    )

    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
