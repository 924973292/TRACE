#!/usr/bin/env python3
"""Smoke gate for history6_8b eval campaign (LIMIT=20)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OVERALL_RE = re.compile(r"^OVERALL ALL", re.M)
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)", re.M)
NAN_RE = re.compile(r"\b(nan|NaN)\b")
ROLLING_RE = re.compile(
    r"sliding window roll|rolling window reset|dialogue_kv_reason.*rolling_window_reset",
    re.I,
)
GP_RE = re.compile(r"GP=([0-9.]+)%")
EM_RE = re.compile(r"EM=([0-9.]+)%")
EXACT_RE = re.compile(r"exact_match,none=([0-9.]+)")
ABC_ARM_RE = re.compile(r"18_prod_drop_scale_a02_sspc_zafd")
OMNIGUI_MAINLINE_TASK = "task_v2_qwen3_omni_novideo"
OMNIGUI_COMPARE_TASK = "task_guiowl_omnigui_native_mt4_8b"
OFFICIAL_BENCH_TASKS = {
    "omnigui": OMNIGUI_MAINLINE_TASK,
    "screenspot_v2": "task_screenspot_v2_guiowl",
    "screenspot_pro": "task_screenspot_pro_guiowl",
    "mmbench_l1": "task_mmbench_gui_l1_guiowl",
    "mmbench_l2": "task_mmbench_gui_l2_guiowl",
}

DEFAULT_BENCHES = (
    "omnigui",
    "screenspot_v2",
    "screenspot_pro",
    "mmbench_l1",
    "mmbench_l2",
)
EXPECTED_TRACKS = (
    "baseline_full",
    "baseline_kv",
    "abc_r80",
    "abc_r40",
    "abc_r20",
    "abc_r10",
    "abc_r05",
)
SINGLE_STEP_BENCHES = {"screenspot_v2", "screenspot_pro", "mmbench_l1", "mmbench_l2"}


def _parse_metric(log_text: str, pattern: re.Pattern[str]) -> float | None:
    for line in log_text.splitlines():
        if "OVERALL ALL" not in line:
            continue
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def _check_run_log(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    if TRACEBACK_RE.search(text):
        errors.append(f"{path}: traceback found")
    if not OVERALL_RE.search(text):
        errors.append(f"{path}: missing OVERALL ALL")
    if NAN_RE.search(text):
        for line in text.splitlines():
            if "OVERALL ALL" in line and NAN_RE.search(line):
                errors.append(f"{path}: NaN in OVERALL line")
                break
    return errors


def _check_abc_launcher(launcher: Path) -> list[str]:
    errors: list[str] = []
    if not launcher.is_file():
        errors.append(f"{launcher}: missing launcher.log for ABC track")
        return errors
    text = launcher.read_text(encoding="utf-8", errors="replace")
    upper = text.upper()
    if "SSPC" not in upper and "ZAFD" not in upper:
        errors.append(f"{launcher}: ABC arm must be SSPC-ZAFD (18_prod_drop_scale_a02_sspc_zafd)")
    if "RPB" in upper or "09_PROD" in upper:
        errors.append(f"{launcher}: invalid legacy RPB arm detected")
    return errors


def _check_omnigui_launcher(launcher: Path) -> list[str]:
    errors: list[str] = []
    if not launcher.is_file():
        return errors
    text = launcher.read_text(encoding="utf-8", errors="replace")
    if f"TASK={OMNIGUI_MAINLINE_TASK}" not in text:
        errors.append(
            f"{launcher}: OmniGUI must use mainline {OMNIGUI_MAINLINE_TASK} "
            "(qwen history6 KV-compatible prompt)"
        )
    if f"TASK={OMNIGUI_COMPARE_TASK}" in text:
        errors.append(
            f"{launcher}: native_mt4 compare task must not appear in main campaign"
        )
    return errors


def _check_bench_launcher(bench: str, launcher: Path) -> list[str]:
    errors: list[str] = []
    expected = OFFICIAL_BENCH_TASKS.get(bench)
    if not expected or not launcher.is_file():
        return errors
    text = launcher.read_text(encoding="utf-8", errors="replace")
    if f"TASK={expected}" not in text:
        errors.append(f"{launcher}: expected TASK={expected} for bench={bench}")
    return errors


def _run_sspc_unit_tests() -> list[str]:
    errors: list[str] = []
    py = sys.executable
    for script in ("scripts/verify_sspc.py", "scripts/verify_sspc_hook_chain.py"):
        proc = subprocess.run(
            [py, str(REPO / script)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(f"{script} failed (exit {proc.returncode})")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Campaign OUT_ROOT")
    ap.add_argument("--tol-pp", type=float, default=3.0, help="Max GP/EM drift baseline vs KV")
    ap.add_argument("--strict", action="store_true", help="Require full 5x7 job grid + SSPC checks")
    ap.add_argument(
        "--benches",
        nargs="*",
        default=list(DEFAULT_BENCHES),
        help="Expected benchmarks (default excludes AC)",
    )
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
                "omnigui/baseline_kv: expected rolling window reset log at LIMIT>=20"
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
        print("FAIL smoke gate:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("PASS history6_8b smoke gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
