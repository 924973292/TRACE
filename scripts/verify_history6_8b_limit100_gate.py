#!/usr/bin/env python3
"""Gate for history6_8b LIMIT=100 phase-1 campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse smoke gate checks (same tracks / launcher rules).
from verify_history6_8b_smoke_gate import (  # noqa: E402
    DEFAULT_BENCHES,
    EXPECTED_TRACKS,
    SINGLE_STEP_BENCHES,
    _check_abc_launcher,
    _check_bench_launcher,
    _check_omnigui_launcher,
    _check_run_log,
    _parse_metric,
    _run_sspc_unit_tests,
    EM_RE,
    EXACT_RE,
    GP_RE,
    ROLLING_RE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tol-pp", type=float, default=3.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--benches", nargs="*", default=list(DEFAULT_BENCHES))
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"FAIL: missing root {root}")
        return 1

    errors: list[str] = []
    benches = tuple(args.benches)

    if args.strict:
        for bench in benches:
            for track in EXPECTED_TRACKS:
                run_log = root / bench / track / "run.log"
                if not run_log.is_file():
                    errors.append(f"missing job: {bench}/{track}/run.log")
                if track.startswith("abc_"):
                    errors.extend(_check_abc_launcher(root / bench / track / "launcher.log"))
                if bench == "omnigui":
                    errors.extend(
                        _check_omnigui_launcher(root / bench / track / "launcher.log")
                    )
                elif bench in SINGLE_STEP_BENCHES and track == "baseline_full":
                    errors.extend(
                        _check_bench_launcher(bench, root / bench / track / "launcher.log")
                    )
        errors.extend(_run_sspc_unit_tests())

    logs = sorted(root.rglob("run.log"))
    if not logs:
        errors.append(f"no run.log under {root}")
    else:
        for log in logs:
            errors.extend(_check_run_log(log))

    omnigui_kv = root / "omnigui" / "baseline_kv" / "run.log"
    omnigui_full = root / "omnigui" / "baseline_full" / "run.log"
    if omnigui_kv.is_file() and omnigui_full.is_file():
        kv_text = omnigui_kv.read_text(encoding="utf-8", errors="replace")
        full_text = omnigui_full.read_text(encoding="utf-8", errors="replace")
        kv_gp = _parse_metric(kv_text, GP_RE)
        full_gp = _parse_metric(full_text, GP_RE)
        if kv_gp is not None and full_gp is not None:
            drift = abs(kv_gp - full_gp)
            if drift > args.tol_pp:
                errors.append(
                    f"omnigui GP drift baseline_full={full_gp}% vs baseline_kv={kv_gp}% "
                    f"({drift:.2f}pp > {args.tol_pp}pp)"
                )
        if args.strict and not ROLLING_RE.search(kv_text):
            errors.append(
                "omnigui/baseline_kv: expected rolling window reset log at LIMIT=100"
            )

    for bench in benches:
        full_log = root / bench / "baseline_full" / "run.log"
        kv_log = root / bench / "baseline_kv" / "run.log"
        if not full_log.is_file() or not kv_log.is_file():
            continue
        full_t = full_log.read_text(encoding="utf-8", errors="replace")
        kv_t = kv_log.read_text(encoding="utf-8", errors="replace")
        for metric_re, name in ((GP_RE, "GP"), (EM_RE, "EM")):
            a = _parse_metric(full_t, metric_re)
            b = _parse_metric(kv_t, metric_re)
            if a is not None and b is not None:
                drift = abs(a - b)
                if drift > args.tol_pp:
                    errors.append(
                        f"{bench} {name} drift {drift:.2f}pp "
                        f"(full={a}% kv={b}%) > {args.tol_pp}pp"
                    )
        if args.strict and bench in SINGLE_STEP_BENCHES:
            acc = _parse_metric(full_t, EXACT_RE)
            if acc is not None and acc <= 0.0:
                errors.append(f"{bench}/baseline_full: exact_match is zero")

    expected = len(benches) * len(EXPECTED_TRACKS) if args.strict else len(logs)
    print(f"checked {len(logs)} run.log files under {root} (strict expected={expected})")
    if errors:
        print("FAIL limit100 gate:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PASS history6_8b limit100 gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
