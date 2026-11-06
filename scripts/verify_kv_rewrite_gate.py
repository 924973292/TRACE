#!/usr/bin/env python3
"""2-arm gate: baseline_full vs baseline_kv_rewrite (no ABC)."""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

GP_RE = re.compile(r"GP=([0-9.]+)%")
EM_RE = re.compile(r"EM=([0-9.]+)%")
E2E_RE = re.compile(r"E2E=([0-9.]+)ms")
SLIDE_RE = re.compile(r"sliding window roll", re.I)
RECTIFY_RE = re.compile(r"sliding_rewrite_rectify|rectify_selector", re.I)
MODULE_A_RE = re.compile(r"module_a_ms['\"]?\s*[:=]\s*([1-9][0-9.]+)", re.I)


def _metric(text: str, pattern: re.Pattern[str]) -> float | None:
    for line in text.splitlines():
        if "OVERALL ALL" not in line and "LATENCY_STEP" not in line:
            continue
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def _load_profiles(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _profile_p50(rows: list[dict], key: str) -> float | None:
    vals = []
    for row in rows:
        v = row.get(key)
        if v is None:
            timing = row.get("timing") or {}
            v = timing.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return float(st.median(vals))


def _step_e2e_median(rows: list[dict], *, min_step: int = 0) -> float | None:
    vals = []
    for row in rows:
        try:
            step = int(row.get("step_id", 0))
        except (TypeError, ValueError):
            step = 0
        if step <= min_step:
            continue
        try:
            vals.append(float(row.get("e2e_latency_ms", 0)))
        except (TypeError, ValueError):
            pass
    if not vals:
        return None
    return float(st.median(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--kv", type=Path, required=True)
    ap.add_argument("--tol-pp", type=float, default=0.5)
    ap.add_argument(
        "--em-tol-pp",
        type=float,
        default=None,
        help="Separate EM drift tolerance (default: --tol-pp). Forced-eviction "
        "runs carry inherent EM jitter from the StreamingLLM-style eviction "
        "approximation; widen EM only, keep GP tight.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kv-profile", type=Path, default=None)
    ap.add_argument("--baseline-profile", type=Path, default=None)
    ap.add_argument("--require-kv-faster", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=1.05)
    ap.add_argument("--max-decode-forward-ratio", type=float, default=1.25)
    ap.add_argument("--max-rectify-ms", type=float, default=0.5)
    ap.add_argument("--max-module-a-ms", type=float, default=0.5)
    args = ap.parse_args()

    errors: list[str] = []
    base_t = args.baseline.read_text(encoding="utf-8", errors="replace")
    kv_t = args.kv.read_text(encoding="utf-8", errors="replace")

    if "OVERALL ALL" not in kv_t:
        errors.append("KV rewrite log missing OVERALL ALL")

    if RECTIFY_RE.search(kv_t):
        errors.append("baseline_kv_rewrite log contains sliding_rewrite_rectify markers (ABC leak)")

    em_tol = args.em_tol_pp if args.em_tol_pp is not None else args.tol_pp
    for name, pattern, tol in (("GP", GP_RE, args.tol_pp), ("EM", EM_RE, em_tol)):
        ref = _metric(base_t, pattern)
        kv_v = _metric(kv_t, pattern)
        if ref is None or kv_v is None:
            errors.append(f"missing {name} in baseline or kv log")
            continue
        drift = abs(ref - kv_v)
        if drift > tol:
            errors.append(f"{name} drift {drift:.2f}pp > {tol}pp")
        else:
            print(f"OK {name}: full={ref}% kv={kv_v}% drift={drift:.2f}pp")

    base_e2e = _metric(base_t, E2E_RE)
    kv_e2e = _metric(kv_t, E2E_RE)
    if base_e2e is not None and kv_e2e is not None:
        speedup_all = base_e2e / kv_e2e if kv_e2e > 0 else 0.0
        print(f"latency KV speedup vs full (aggregate): {speedup_all:.2f}x")

    base_rows = _load_profiles(args.baseline_profile)
    kv_rows = _load_profiles(args.kv_profile)
    speedup_checked = False
    if base_rows and kv_rows:
        base_step_e2e = _step_e2e_median(base_rows, min_step=0)
        kv_step_e2e = _step_e2e_median(kv_rows, min_step=0)
        if base_step_e2e is not None and kv_step_e2e is not None:
            speedup_step = base_step_e2e / kv_step_e2e if kv_step_e2e > 0 else 0.0
            print(
                f"latency KV speedup step>0: {speedup_step:.2f}x "
                f"(full_med={base_step_e2e:.1f}ms kv_med={kv_step_e2e:.1f}ms)"
            )
            speedup_checked = True
            if args.require_kv_faster and speedup_step < args.min_speedup:
                errors.append(
                    f"KV step>0 not faster enough: speedup={speedup_step:.2f}x < {args.min_speedup}x"
                )
        elif args.require_kv_faster:
            print("NOTE: no step>0 profile rows; skipping step>0 speedup gate")

    if (
        args.require_kv_faster
        and not speedup_checked
        and base_e2e is not None
        and kv_e2e is not None
        and args.limit >= 32
    ):
        speedup_all = base_e2e / kv_e2e if kv_e2e > 0 else 0.0
        if speedup_all < args.min_speedup:
            errors.append(f"KV not faster enough: speedup={speedup_all:.2f}x < {args.min_speedup}x")

    profiles = kv_rows
    if profiles:
        rectify_p50 = _profile_p50(profiles, "sliding_rewrite_rectify_ms")
        module_a_p50 = _profile_p50(profiles, "module_a_ms")
        dec_pt = _profile_p50(profiles, "decode_per_token_ms")
        fwd_ratio = _profile_p50(profiles, "decode_forward_ratio")
        flow_modes = {str(r.get("kv_flow_mode", "unknown")) for r in profiles}
        hit = sum(1 for r in profiles if str(r.get("dialogue_kv_reason", "")) == "hit")
        parts = [f"kv_flow_mode={','.join(sorted(flow_modes))}"]
        if dec_pt is not None:
            parts.append(f"decode_per_token={dec_pt:.2f}ms")
        if fwd_ratio is not None:
            parts.append(f"decode_fwd_ratio={fwd_ratio:.2f}")
        if rectify_p50 is not None:
            parts.append(f"rectify_p50={rectify_p50:.2f}ms")
        print("profile kv: " + " ".join(parts))
        if rectify_p50 is not None and rectify_p50 > args.max_rectify_ms:
            errors.append(f"sliding_rewrite_rectify_ms p50={rectify_p50:.2f} > {args.max_rectify_ms}")
        if module_a_p50 is not None and module_a_p50 > args.max_module_a_ms:
            errors.append(f"module_a_ms p50={module_a_p50:.2f} > {args.max_module_a_ms}")
        if fwd_ratio is not None and fwd_ratio > args.max_decode_forward_ratio:
            errors.append(
                f"decode_forward_ratio p50={fwd_ratio:.2f} > {args.max_decode_forward_ratio}"
            )
        if "prune_rewrite" in flow_modes:
            errors.append("kv_flow_mode=prune_rewrite in Arm2 profiles (ABC leak)")
        if hit > 0:
            print(f"OK dialogue_kv hit rate: {hit}/{len(profiles)}")

    slide_hits = len(SLIDE_RE.findall(kv_t))
    if slide_hits > 0:
        print(f"OK: sliding window roll {slide_hits} time(s)")
    elif args.limit and args.limit < 200:
        print("NOTE: no sliding window roll (LIMIT may not reach hist9 roll)")

    if errors:
        print("FAIL kv rewrite gate:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS kv rewrite gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
