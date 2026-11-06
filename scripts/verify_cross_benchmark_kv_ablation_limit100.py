#!/usr/bin/env python3
"""Compare Track A (SESSION_KV=0) vs Track B (dialogue KV) limit100 ablation outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]

_METRIC_RE = re.compile(
    r"(?P<metric>exact_match|goal_progress|type_match|success_rate|android_control_exact_match)"
    r"\s*,\s*(?P<value>[\d.]+)"
)


def _parse_overall(log_path: Path) -> Dict[str, float]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("OVERALL ALL"):
            continue
        for m in _METRIC_RE.finditer(line):
            out[m.group("metric")] = float(m.group("value"))
    return out


def _latest_results_json(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("**/results*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _metrics_from_results(path: Path) -> Dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    results = payload.get("results") or {}
    out: Dict[str, float] = {}
    for task_metrics in results.values():
        if not isinstance(task_metrics, dict):
            continue
        for key, val in task_metrics.items():
            if isinstance(val, (int, float)):
                out[key.replace(",", "")] = float(val)
    return out


def _load_track_metrics(track_dir: Path) -> Dict[str, float]:
    log_metrics = _parse_overall(track_dir / "run.log")
    if log_metrics:
        return log_metrics
    res = _latest_results_json(track_dir)
    if res:
        return _metrics_from_results(res)
    return {}


def _diff(
    track_a: Dict[str, float],
    track_b: Dict[str, float],
    *,
    tol_pp: float,
) -> Tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    keys = sorted(set(track_a) | set(track_b))
    for key in keys:
        a = track_a.get(key)
        b = track_b.get(key)
        if a is None or b is None:
            lines.append(f"  {key}: A={a} B={b} (missing)")
            continue
        delta = (b - a) * 100.0
        status = "OK" if abs(delta) <= tol_pp else "FAIL"
        if status == "FAIL":
            ok = False
        lines.append(f"  {key}: A={a:.4f} B={b:.4f} delta={delta:+.2f}pp [{status}]")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "outputs" / "cross_benchmark_kv_ablation_limit100",
    )
    parser.add_argument("--tol-pp", type=float, default=1.0, help="max allowed metric drift in pp")
    args = parser.parse_args()
    root = args.root

    suites = [
        ("omnigui", "exact_match"),
        ("android_control", "android_control_exact_match"),
    ]
    all_ok = True
    for name, _primary in suites:
        track_a_dir = root / f"{name}_track_a"
        track_b_dir = root / f"{name}_track_b"
        if not track_a_dir.is_dir() and not track_b_dir.is_dir():
            print(f"SKIP {name}: no tracks under {root}")
            continue
        a = _load_track_metrics(track_a_dir)
        b = _load_track_metrics(track_b_dir)
        print(f"=== {name} Track A vs B (tol={args.tol_pp}pp) ===")
        ok, lines = _diff(a, b, tol_pp=args.tol_pp)
        for line in lines:
            print(line)
        if not ok:
            all_ok = False
        print()

    if not all_ok:
        print("FAIL: metric drift exceeds tolerance")
        return 1
    print("PASS: KV ablation within tolerance (or no tracks to compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
