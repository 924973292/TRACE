#!/usr/bin/env python3
"""Verify ABC r=0.05 sanity run does not collapse (MRoPE + metadata-only Module C)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BROKEN_EM = 0.006
MIN_EM_ABS = 0.03


def parse_em(log_path: Path) -> float | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"^OVERALL ALL TM=[\d.]+% EM=([\d.]+)%", text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1]) / 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    log_path = args.run_dir / "run.log"
    if not log_path.is_file():
        print(f"FAIL: missing {log_path}", file=sys.stderr)
        return 1

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    em = parse_em(log_path)
    if em is None:
        print("FAIL: OVERALL ALL not found", file=sys.stderr)
        return 1

    errors: list[str] = []
    if "text-only RoPE" in text or "compact MRoPE must come from the hook" in text:
        errors.append("text-only RoPE / origpos failure in log")
    if "GeoPACT origpos" in text and "RuntimeError" in text:
        errors.append("origpos RuntimeError in log")
    if em <= BROKEN_EM * 2:
        errors.append(f"EM={em:.2%} near broken-chain floor (~0.6%)")
    if em < MIN_EM_ABS:
        errors.append(f"EM={em:.2%} < {MIN_EM_ABS:.0%} minimum")

    print(f"ABC r05 RESULT EM={em:.2%}")
    if errors:
        print("FAIL:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"PASS: ABC r05 sanity (limit={args.limit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
