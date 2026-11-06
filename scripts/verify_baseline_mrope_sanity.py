#!/usr/bin/env python3
"""Verify MRoPE-fixed hist_k5 baseline sanity run against known references."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# limit=100 no-cache reference (baseline_vision_cache_compare_limit100)
REF_LIMIT100 = {"tm": 0.54, "em": 0.38}
# full 2572-step reference (geopact_hist_k5_results_analysis.md)
REF_FULL = {"tm": 0.4125, "em": 0.2659}
DEFAULT_TOL = 0.06


def parse_overall(log_path: Path) -> tuple[float | None, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"^OVERALL ALL TM=([\d.]+)% EM=([\d.]+)%",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        return None, None
    tm_s, em_s = matches[-1]
    return float(tm_s) / 100.0, float(em_s) / 100.0


def within(actual: float | None, ref: float, tol: float) -> bool:
    if actual is None:
        return False
    return abs(actual - ref) <= tol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = parser.parse_args()

    log_path = args.run_dir / "run.log"
    if not log_path.is_file():
        print(f"FAIL: missing {log_path}", file=sys.stderr)
        return 1

    tm, em = parse_overall(log_path)
    if tm is None or em is None:
        print("FAIL: OVERALL ALL line not found", file=sys.stderr)
        return 1

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    vision_hits = text.count("GeoPACT vision-cache:")
    session_hits = text.count("GeoPACT session-kv:")

    print(f"RESULT TM={tm:.2%} EM={em:.2%}")
    print(f"REF limit={args.limit}: TM={REF_LIMIT100['tm']:.2%} EM={REF_LIMIT100['em']:.2%} (tol={args.tol:.0%})")
    print(f"REF full 2572: TM={REF_FULL['tm']:.2%} EM={REF_FULL['em']:.2%}")
    print(f"vision-cache log lines: {vision_hits}  session-kv log lines: {session_hits}")

    ok_limit = within(tm, REF_LIMIT100["tm"], args.tol) and within(em, REF_LIMIT100["em"], args.tol)
    if ok_limit:
        print("PASS: within limit=100 reference tolerance")
        return 0

    print(
        f"FAIL: TM delta={abs(tm - REF_LIMIT100['tm']):.2%} "
        f"EM delta={abs(em - REF_LIMIT100['em']):.2%} (tol={args.tol:.0%})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
