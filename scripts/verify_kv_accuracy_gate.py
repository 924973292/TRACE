#!/usr/bin/env python3
"""Compare baseline_full vs baseline_kv run.log: accuracy match + KV faster."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GP_RE = re.compile(r"GP=([0-9.]+)%")
EM_RE = re.compile(r"EM=([0-9.]+)%")
E2E_RE = re.compile(r"E2E=([0-9.]+)ms")
MISC_RE = re.compile(r"Misc=([0-9.]+)ms")
KVFIN_RE = re.compile(r"KVFinalize=([0-9.]+)ms")
ROLL_RE = re.compile(
    r"sliding window roll|rolling window reset|dialogue_kv_reason.*rolling",
    re.I,
)


def _metric(text: str, pattern: re.Pattern[str]) -> float | None:
    for line in text.splitlines():
        if "OVERALL ALL" not in line and "LATENCY_STEP" not in line:
            continue
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--kv", type=Path, required=True)
    ap.add_argument("--tol-pp", type=float, default=0.5)
    ap.add_argument("--require-kv-faster", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=1.05)
    args = ap.parse_args()

    errors: list[str] = []
    base_t = args.baseline.read_text(encoding="utf-8", errors="replace")
    kv_t = args.kv.read_text(encoding="utf-8", errors="replace")

    for name, pattern in (("GP", GP_RE), ("EM", EM_RE)):
        a = _metric(base_t, pattern)
        b = _metric(kv_t, pattern)
        if a is None or b is None:
            errors.append(f"missing {name} in logs")
            continue
        drift = abs(a - b)
        if drift > args.tol_pp:
            errors.append(f"{name} drift {drift:.2f}pp (full={a}% kv={b}%) > {args.tol_pp}pp")
        else:
            print(f"OK {name}: full={a}% kv={b}% drift={drift:.2f}pp")

    base_e2e = _metric(base_t, E2E_RE)
    kv_e2e = _metric(kv_t, E2E_RE)
    if base_e2e is not None and kv_e2e is not None:
        speedup = base_e2e / kv_e2e if kv_e2e > 0 else 0.0
        print(f"latency E2E: full={base_e2e:.1f}ms kv={kv_e2e:.1f}ms speedup={speedup:.2f}x")
        if args.require_kv_faster and speedup < args.min_speedup:
            errors.append(
                f"KV not faster enough: speedup={speedup:.2f}x < {args.min_speedup}x"
            )
    else:
        print("WARN: could not parse E2E from logs")

    for label, pattern in (
        ("misc_ms", MISC_RE),
        ("kv_finalize_ms", KVFIN_RE),
    ):
        base_v = _metric(base_t, pattern)
        kv_v = _metric(kv_t, pattern)
        if base_v is not None and kv_v is not None:
            print(f"latency {label}: full={base_v:.1f}ms kv={kv_v:.1f}ms")
        else:
            print(f"NOTE: could not parse {label} from LATENCY_STEP")

    if ROLL_RE.search(kv_t):
        print("OK: rolling/slide path exercised in KV log")
    else:
        print("NOTE: no rolling/slide log line (LIMIT may not reach step>=N+1)")

    if errors:
        print("FAIL kv accuracy gate:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS kv accuracy gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
