#!/usr/bin/env python3
"""Verify hist2 rolling-window + sliding-rewrite smoke results.

The smoke is intentionally behavioral: it must prove the combined path actually
ran, not merely that a short ABC eval completed.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


ROLL_RE = re.compile(r"sliding window roll|anchored eviction roll", re.I)
ROLL_RESET_RE = re.compile(r"rolling window reset", re.I)
SEGMENT_RESET_RE = re.compile(r"segment reset", re.I)
SLIDING_OK_RE = re.compile(r"'sliding_rewrite_ok':\s*True", re.I)
TRACEBACK_RE = re.compile(r"Traceback|RuntimeError|AssertionError", re.I)
INVARIANT_RE = re.compile(
    r"past_len_mismatch|prompt_token_len_mismatch|cropped KV length mismatch|"
    r"cached_input_ids vs KV length mismatch|mrope_len_mismatch|length invariant",
    re.I,
)


def _find_profile(out_dir: Path) -> Path | None:
    matches = glob.glob(str(out_dir / "**" / "step_profiles.jsonl"), recursive=True)
    if not matches:
        return None
    return max((Path(p) for p in matches), key=lambda p: p.stat().st_mtime)


def _num(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for subkey in ("ms", "value", "total_ms"):
            subvalue = value.get(subkey)
            if isinstance(subvalue, (int, float)):
                return float(subvalue)
    return 0.0


def _profile_counts(
    profile_path: Path | None,
    *,
    max_output_tokens: int,
) -> tuple[int, int, list[str]]:
    if profile_path is None:
        return 0, 0, []
    rewrite_rows = 0
    slide_rows = 0
    degenerate: list[str] = []
    with profile_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                _num(row, "sliding_rewrite_lm_ms") > 0.0
                or _num(row, "sliding_rewrite_rectify_ms") > 0.0
                or _num(row, "sliding_rewrite_mrope_ms") > 0.0
            ):
                rewrite_rows += 1
            if _num(row, "kv_slide_ms") > 0.0:
                slide_rows += 1
            audit = row.get("timing_audit") or {}
            out_len = audit.get("output_token_len")
            if isinstance(out_len, (int, float)) and out_len >= max_output_tokens:
                degenerate.append(
                    f"profile row {line_no}: output_token_len={int(out_len)} "
                    f">= {max_output_tokens}"
                )
    return rewrite_rows, slide_rows, degenerate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-rolls", type=int, default=1)
    ap.add_argument("--min-slides", type=int, default=1)
    ap.add_argument("--min-rewrites", type=int, default=1)
    ap.add_argument("--max-output-tokens", type=int, default=512)
    args = ap.parse_args()

    run_log = args.out / "run.log"
    if not run_log.is_file():
        print(f"FAIL: missing run.log under {args.out}")
        return 1

    text = run_log.read_text(encoding="utf-8", errors="replace")
    profile_path = _find_profile(args.out)
    profile_rewrites, profile_slides, profile_degenerate = _profile_counts(
        profile_path,
        max_output_tokens=args.max_output_tokens,
    )

    roll_hits = len(ROLL_RE.findall(text))
    rewrite_hits = len(SLIDING_OK_RE.findall(text))
    errors: list[str] = []

    if TRACEBACK_RE.search(text):
        errors.append("run.log contains exception marker")
    if INVARIANT_RE.search(text):
        errors.append("run.log contains KV/MRoPE length invariant error")
    # Fallback reset at roll is the safe eviction form when the slide boundary
    # cannot be proven (system-prompt + task-header layout); it counts as a
    # handled roll, not a failure.
    reset_hits = len(ROLL_RESET_RE.findall(text))
    if SEGMENT_RESET_RE.search(text):
        errors.append("segment reset occurred during hist2 smoke")
    if roll_hits + profile_slides + reset_hits < args.min_rolls:
        errors.append(
            f"roll slides {roll_hits} + profile slide rows {profile_slides} "
            f"+ fallback resets {reset_hits} < {args.min_rolls}"
        )
    # Anchored eviction must actually engage under the position-independent
    # protocol: proven slides, not just fallback resets.
    if max(roll_hits, profile_slides) < args.min_slides:
        errors.append(
            f"proven KV slides {max(roll_hits, profile_slides)} < {args.min_slides} "
            f"(fallback resets {reset_hits} do not count)"
        )
    if rewrite_hits + profile_rewrites < args.min_rewrites:
        errors.append(
            f"rewrite hits {rewrite_hits} + profile rewrite rows {profile_rewrites} "
            f"< {args.min_rewrites}"
        )
    for item in profile_degenerate:
        errors.append(f"degenerate decode length: {item}")
    if re.search(r"DEGENERATE step=", text):
        errors.append("run.log contains DEGENERATE output marker")

    print(
        "hist2_smoke "
        f"roll_log={roll_hits} slide_profile={profile_slides} "
        f"fallback_resets={reset_hits} "
        f"rewrite_log={rewrite_hits} rewrite_profile={profile_rewrites} "
        f"profile={profile_path}"
    )
    if errors:
        print("FAIL hist2 roll+rewrite smoke:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("PASS hist2 roll+rewrite smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
