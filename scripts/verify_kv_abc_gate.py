#!/usr/bin/env python3
"""ABC + dialogue KV gate: accuracy vs baseline_full, error scan on ABC arm."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

GP_RE = re.compile(r"GP=([0-9.]+)%")
EM_RE = re.compile(r"EM=([0-9.]+)%")
E2E_RE = re.compile(r"E2E=([0-9.]+)ms")
PRUNE_RE = re.compile(r"Pruning=([0-9.]+)ms")
MISC_RE = re.compile(r"Misc=([0-9.]+)ms")
SLIDE_RE = re.compile(r"sliding window roll", re.I)
NO_CAPTURE_RE = re.compile(r"no_captured_frame", re.I)
ROLL_RE = re.compile(
    r"sliding window roll|rolling window reset|dialogue_kv_reason.*rolling",
    re.I,
)
SLIDING_OK_RE = re.compile(r"'sliding_rewrite_ok':\s*True", re.I)
LEGACY_COMPRESS_RE = re.compile(r"kv_history_compress_ok.*legacy|legacy_compress_enabled", re.I)
MROPE_MS_RE = re.compile(r"dialogue_kv_mrope_ms['\"]?\s*[:=]\s*([0-9.]+)", re.I)
ABC_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)", re.I),
    re.compile(r"text-only RoPE", re.I),
    re.compile(r"compact MRoPE must come from the hook", re.I),
    re.compile(r"GeoPACT origpos.*RuntimeError", re.I),
    re.compile(r"SSPC.*RuntimeError", re.I),
)


def _metric(text: str, pattern: re.Pattern[str]) -> float | None:
    for line in text.splitlines():
        if "OVERALL ALL" not in line and "LATENCY_STEP" not in line:
            continue
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def _drift_check(
    errors: list[str],
    *,
    name: str,
    ref: float | None,
    other: float | None,
    tol_pp: float,
    ref_label: str,
    other_label: str,
    latency_focus: bool = False,
) -> None:
    if ref is None or other is None:
        # Missing metrics always indicate a broken run, even in latency mode.
        errors.append(f"missing {name} in {ref_label} or {other_label} log")
        return
    drift = abs(ref - other)
    if drift > tol_pp:
        msg = (
            f"{name} drift {drift:.2f}pp ({ref_label}={ref}% {other_label}={other}%) "
            f"> {tol_pp}pp"
        )
        if latency_focus:
            print(f"WARN (non-fatal): {msg}")
        else:
            errors.append(msg)
    else:
        print(f"OK {name}: {ref_label}={ref}% {other_label}={other}% drift={drift:.2f}pp")


def _scan_abc_errors(text: str) -> list[str]:
    found: list[str] = []
    for pat in ABC_ERROR_PATTERNS:
        if pat.search(text):
            found.append(pat.pattern)
    return found


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


def _profile_step_p50(rows: list[dict], key: str, *, step_id: int | None = None) -> float | None:
    if step_id is None:
        return _profile_p50(rows, key)
    subset = [
        row
        for row in rows
        if int(row.get("step_id", row.get("step_index", row.get("step_idx", 0))) or 0) == step_id
    ]
    return _profile_p50(subset, key)


def _profile_step_gt0_p50(rows: list[dict], key: str) -> float | None:
    subset = [
        row
        for row in rows
        if int(row.get("step_id", row.get("step_index", row.get("step_idx", 0))) or 0) > 0
    ]
    return _profile_p50(subset, key)


def _sliding_rewrite_ok_rate(abc_log: str, profiles: list[dict]) -> tuple[int, int]:
    ok_hits = len(SLIDING_OK_RE.findall(abc_log))
    step_ge1 = sum(1 for row in profiles if int(row.get("step_index", row.get("step_idx", 0)) or 0) > 0)
    if step_ge1 <= 0:
        step_ge1 = max(ok_hits, 1)
    return ok_hits, step_ge1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--kv", type=Path, required=True)
    ap.add_argument("--abc", type=Path, required=True)
    ap.add_argument("--tol-pp", type=float, default=0.5)
    ap.add_argument(
        "--em-tol-pp",
        type=float,
        default=None,
        help="Separate EM drift tolerance (default: --tol-pp). Forced-eviction "
        "runs (hist2) carry inherent EM jitter from the StreamingLLM-style KV "
        "approximation at eviction steps even when GP is flat and the bilateral "
        "trace proves position bookkeeping is exact; widen EM only, keep GP tight.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Eval LIMIT (for slide gate leniency)")
    ap.add_argument("--require-kv-faster", action="store_true")
    ap.add_argument("--min-speedup", type=float, default=1.05)
    ap.add_argument(
        "--abc-drift-warn-only",
        action="store_true",
        default=os.environ.get("GEOPACT_GATE_ABC_DRIFT_WARN_ONLY", "").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Report kv-vs-abc GP/EM drift as WARN instead of FAIL. Pruning "
        "legitimately trades accuracy for tokens; the gate then only enforces "
        "chain correctness (full-vs-kv drift, proven slides, rectify health).",
    )
    ap.add_argument(
        "--latency-focus",
        action="store_true",
        default=os.environ.get("GEOPACT_GATE_LATENCY_FOCUS", "").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Report GP/EM drift as non-fatal warnings; keep fatal errors and "
        "missing-metric checks as failures. Useful when accuracy is expected to "
        "improve and only latency is under evaluation.",
    )
    ap.add_argument(
        "--sliding-gate",
        action="store_true",
        help="Enable sliding-rewrite L100 gate checks (rewrite ok rate, mrope/e2e regression).",
    )
    ap.add_argument(
        "--abc-profile",
        type=Path,
        default=None,
        help="ABC step_profiles.jsonl for sliding gate latency checks.",
    )
    ap.add_argument(
        "--abc-profile-before",
        type=Path,
        default=None,
        help="Pre-optimization ABC profiles for -30%% mrope / -20%% e2e regression gate.",
    )
    ap.add_argument("--min-sliding-ok-rate", type=float, default=0.95)
    ap.add_argument("--max-mrope-regression", type=float, default=0.70, help="p50 ratio cap (0.7 = -30%%)")
    ap.add_argument("--max-e2e-regression", type=float, default=0.80, help="p50 ratio cap (0.8 = -20%%)")
    ap.add_argument("--max-decode-forward-ratio", type=float, default=1.25)
    ap.add_argument(
        "--max-abc-decode-per-token-ratio",
        type=float,
        default=1.30,
        help="Max ABC/KV p50 decode_per_token_ms ratio on step>0 (default 1.30).",
    )
    ap.add_argument("--kv-profile", type=Path, default=None)
    args = ap.parse_args()

    if args.latency_focus:
        print("MODE: latency-focus (GP/EM drift is WARN-only, not a gate failure)")

    errors: list[str] = []
    base_t = args.baseline.read_text(encoding="utf-8", errors="replace")
    kv_t = args.kv.read_text(encoding="utf-8", errors="replace")
    abc_t = args.abc.read_text(encoding="utf-8", errors="replace")

    if "OVERALL ALL" not in abc_t:
        errors.append("ABC log missing OVERALL ALL")

    for pat in _scan_abc_errors(abc_t):
        errors.append(f"ABC log error pattern: {pat}")

    em_tol = args.em_tol_pp if args.em_tol_pp is not None else args.tol_pp
    for name, pattern, tol in (("GP", GP_RE, args.tol_pp), ("EM", EM_RE, em_tol)):
        ref = _metric(base_t, pattern)
        kv_v = _metric(kv_t, pattern)
        abc_v = _metric(abc_t, pattern)
        _drift_check(
            errors,
            name=name,
            ref=ref,
            other=kv_v,
            tol_pp=tol,
            ref_label="full",
            other_label="kv",
            latency_focus=args.latency_focus,
        )
        _drift_check(
            errors,
            name=name,
            ref=kv_v,
            other=abc_v,
            tol_pp=tol,
            ref_label="kv",
            other_label="abc",
            latency_focus=args.latency_focus or args.abc_drift_warn_only,
        )

    for label, text in (("full", base_t), ("kv", kv_t), ("abc", abc_t)):
        e2e = _metric(text, E2E_RE)
        prune = _metric(text, PRUNE_RE)
        misc = _metric(text, MISC_RE)
        parts = [f"E2E={e2e:.1f}ms" if e2e is not None else "E2E=?"]
        if prune is not None:
            parts.append(f"Pruning={prune:.1f}ms")
        if misc is not None:
            parts.append(f"Misc={misc:.1f}ms")
        print(f"latency {label}: " + " ".join(parts))

    base_e2e = _metric(base_t, E2E_RE)
    kv_e2e = _metric(kv_t, E2E_RE)
    if base_e2e is not None and kv_e2e is not None:
        speedup = base_e2e / kv_e2e if kv_e2e > 0 else 0.0
        print(f"latency KV speedup vs full: {speedup:.2f}x")
        if args.require_kv_faster and speedup < args.min_speedup:
            errors.append(
                f"KV not faster enough: speedup={speedup:.2f}x < {args.min_speedup}x"
            )

    kv_profiles = _load_profiles(args.kv_profile)
    abc_profiles_gate = _load_profiles(args.abc_profile)
    for label, profiles in (("kv", kv_profiles), ("abc", abc_profiles_gate)):
        if not profiles:
            continue
        dec_pt = _profile_p50(profiles, "decode_per_token_ms")
        fwd_ratio = _profile_p50(profiles, "decode_forward_ratio")
        dec0 = _profile_step_p50(profiles, "decode_latency_ms", step_id=0)
        dec_gt0 = _profile_step_gt0_p50(profiles, "decode_latency_ms")
        parts = []
        if dec_pt is not None:
            parts.append(f"decode_per_token={dec_pt:.2f}ms")
        if fwd_ratio is not None:
            parts.append(f"decode_fwd_ratio={fwd_ratio:.2f}")
        if dec0 is not None:
            parts.append(f"step0_decode={dec0:.0f}ms")
        if dec_gt0 is not None:
            parts.append(f"step>0_decode={dec_gt0:.0f}ms")
        if parts:
            print(f"profile {label}: " + " ".join(parts))
        if fwd_ratio is not None and fwd_ratio > args.max_decode_forward_ratio:
            errors.append(
                f"{label} decode_forward_ratio p50={fwd_ratio:.2f} > "
                f"{args.max_decode_forward_ratio}"
            )

    kv_profiles_ratio = _load_profiles(args.kv_profile)
    abc_profiles_ratio = _load_profiles(args.abc_profile)
    if kv_profiles_ratio and abc_profiles_ratio:
        kv_gt0 = [
            r
            for r in kv_profiles_ratio
            if int(r.get("step_id", r.get("step_index", 0)) or 0) > 0
        ]
        abc_gt0 = [
            r
            for r in abc_profiles_ratio
            if int(r.get("step_id", r.get("step_index", 0)) or 0) > 0
        ]
        kv_pt = _profile_p50(kv_gt0, "decode_per_token_ms")
        abc_pt = _profile_p50(abc_gt0, "decode_per_token_ms")
        if kv_pt is not None and abc_pt is not None and kv_pt > 0:
            ratio = abc_pt / kv_pt
            print(
                f"decode_per_token step>0 p50: kv={kv_pt:.2f}ms abc={abc_pt:.2f}ms "
                f"ratio={ratio:.2f}x (max {args.max_abc_decode_per_token_ratio}x)"
            )
            if ratio > args.max_abc_decode_per_token_ratio:
                errors.append(
                    f"ABC decode_per_token ratio {ratio:.2f}x > "
                    f"{args.max_abc_decode_per_token_ratio}x"
                )

    slide_hits = len(SLIDE_RE.findall(abc_t))
    if slide_hits > 0:
        print(f"OK: sliding window roll succeeded {slide_hits} time(s) in ABC log")
    elif args.limit and args.limit < 200:
        print("NOTE: no sliding window roll in ABC (LIMIT may not reach hist9 roll)")
    else:
        errors.append("ABC log missing successful sliding window roll")

    no_capture_hits = len(NO_CAPTURE_RE.findall(abc_t))
    if no_capture_hits > 0:
        errors.append(f"ABC log has no_captured_frame x{no_capture_hits}")
    else:
        print("OK: no no_captured_frame in ABC log")

    if ROLL_RE.search(abc_t):
        print("OK: rolling/slide path exercised in ABC log")
    else:
        print("NOTE: no rolling/slide log line in ABC (LIMIT may not reach step>=7)")

    if args.sliding_gate:
        if LEGACY_COMPRESS_RE.search(abc_t):
            errors.append("ABC log indicates legacy compress path (forbidden when sliding)")
        else:
            print("OK: no legacy compress markers in ABC log")

        abc_profiles = _load_profiles(args.abc_profile)
        ok_hits, denom = _sliding_rewrite_ok_rate(abc_t, abc_profiles)
        rate = ok_hits / denom if denom > 0 else 0.0
        print(f"sliding_rewrite_ok: {ok_hits}/{denom} = {rate:.1%}")
        if rate < args.min_sliding_ok_rate:
            errors.append(
                f"sliding_rewrite_ok rate {rate:.1%} < {args.min_sliding_ok_rate:.0%}"
            )

        mrope_p50 = _profile_p50(abc_profiles, "dialogue_kv_mrope_ms")
        if mrope_p50 is not None:
            print(f"ABC dialogue_kv_mrope_ms p50={mrope_p50:.1f}ms")
        before_profiles = _load_profiles(args.abc_profile_before)
        if before_profiles and abc_profiles:
            before_mrope = _profile_p50(before_profiles, "dialogue_kv_mrope_ms")
            before_e2e = _profile_p50(before_profiles, "e2e_latency_ms")
            after_e2e = _profile_p50(abc_profiles, "e2e_latency_ms")
            if before_mrope and mrope_p50:
                ratio = mrope_p50 / before_mrope
                print(f"mrope p50 ratio after/before={ratio:.2f} (target <= {args.max_mrope_regression})")
                if ratio > args.max_mrope_regression:
                    errors.append(
                        f"dialogue_kv_mrope_ms p50 not improved enough: ratio={ratio:.2f}"
                    )
            if before_e2e and after_e2e:
                e2e_ratio = after_e2e / before_e2e
                print(f"e2e p50 ratio after/before={e2e_ratio:.2f} (target <= {args.max_e2e_regression})")
                if e2e_ratio > args.max_e2e_regression:
                    errors.append(f"e2e_latency_ms p50 not improved enough: ratio={e2e_ratio:.2f}")
        elif args.abc_profile_before and str(args.abc_profile_before).strip():
            print("WARN: --abc-profile-before set but profiles missing; skip regression gate")

        abc_e2e = _metric(abc_t, E2E_RE)
        kv_gp = _metric(kv_t, GP_RE)
        abc_gp = _metric(abc_t, GP_RE)
        if kv_gp is not None and abc_gp is not None:
            gp_drift = abs(kv_gp - abc_gp)
            print(f"GP kv vs abc drift={gp_drift:.2f}pp (sliding gate <= {args.tol_pp}pp)")
            if gp_drift > args.tol_pp:
                msg = f"GP drift vs baseline_kv {gp_drift:.2f}pp > {args.tol_pp}pp"
                if args.abc_drift_warn_only or args.latency_focus:
                    print(f"WARN (non-fatal): {msg}")
                else:
                    errors.append(msg)

    # Chain-correctness checks on the ABC arm (always fatal): every roll must
    # be a proven slide (no fallback resets once rolling engages), every
    # rectify must succeed on the intended frame with its prefetch identity.
    roll_slides = len(re.findall(r"sliding window roll|anchored eviction roll", abc_t))
    roll_resets = len(re.findall(r"rolling window reset", abc_t))
    if roll_slides > 0 and roll_resets > 0:
        errors.append(
            f"ABC arm fell back to rolling window reset x{roll_resets} "
            f"(slides={roll_slides}) — slide boundary not proven"
        )
    else:
        print(f"OK rolls: slides={roll_slides} fallback_resets={roll_resets}")
    rectify_reasons = re.findall(r"rectify_rectify_reason'?\s*[:=]\s*'([a-z_:]+)'", abc_t)
    bad_rectify = [r for r in rectify_reasons if r != "ok"]
    if bad_rectify:
        errors.append(
            f"ABC rectify non-ok reasons x{len(bad_rectify)}: {sorted(set(bad_rectify))[:5]}"
        )
    elif rectify_reasons:
        print(f"OK rectify: {len(rectify_reasons)} frames all reason=ok")
    prefetch_mismatch = len(re.findall(r"rectify_prefetch_match'?\s*[:=]\s*False", abc_t))
    if prefetch_mismatch:
        errors.append(f"ABC rectify prefetch identity mismatch x{prefetch_mismatch}")
    instr_sources = re.findall(r"rectify_instr_source'?\s*[:=]\s*'([a-z_]+)'", abc_t)
    if instr_sources:
        from collections import Counter

        print(f"rectify instr_source: {dict(Counter(instr_sources))}")
        if any(s == "none" for s in instr_sources):
            errors.append("ABC rectify ran with no relevance query (instr_source=none)")

    if errors:
        print("FAIL kv abc gate:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS kv abc gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
