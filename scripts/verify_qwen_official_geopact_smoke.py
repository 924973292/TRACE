#!/usr/bin/env python3
"""Verify official-Qwen GeoPACT smoke logs before full sweep dispatch."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LEGACY_REQUIRED = {
    "smoke_baseline_vcache",
    "smoke_random_r20",
    "smoke_a_random_r20",
    "smoke_ab_r20",
    "smoke_abc_r20",
}

FOURCLASS_SMOKE_REQUIRED = {
    "smoke_c1_baseline",
    "smoke_c1_random_r20",
    "smoke_c2_random_r20",
    "smoke_c3_random_r20",
    "smoke_c4_ab_r20",
}


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _ints(pattern: str, text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(pattern, text)]


def _last_int(pattern: str, text: str, default: int | None = None) -> int | None:
    vals = [int(m.group(1)) for m in re.finditer(pattern, text)]
    return vals[-1] if vals else default


def _check_prune_run(run: str, text: str) -> None:
    if "GeoPACT source-compact:" not in text:
        _fail(f"{run}: missing source-compact logs")
    kept = _last_int(r"'n_visual_kept':\s*(\d+)", text, 0)
    if kept is None or kept <= 0:
        _fail(f"{run}: expected n_visual_kept > 0")
    deepstack_rows = _ints(r"'compact_deepstack_rows':\s*(\d+)", text)
    if not deepstack_rows or max(deepstack_rows) <= 0:
        _fail(
            f"{run}: expected compact_deepstack_rows > 0, got {deepstack_rows[-5:] if deepstack_rows else deepstack_rows}"
        )


def _check_video_prune(run: str, text: str) -> None:
    total = _last_int(r"'video_tokens_total':\s*(\d+)", text)
    kept = _last_int(r"'video_tokens_kept':\s*(\d+)", text)
    if total is None or total <= 0:
        _fail(f"{run}: expected video_tokens_total > 0 in compact stats")
    if kept is None or kept <= 0:
        _fail(f"{run}: expected video_tokens_kept > 0")
    if kept is not None and total is not None and kept > total:
        _fail(f"{run}: video_tokens_kept {kept} > total {total}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("logs"))
    ap.add_argument("--family", default="qwen_geopact_fourclass_smoke")
    ap.add_argument("--legacy", action="store_true", help="Use legacy smoke run names")
    args = ap.parse_args()

    required = LEGACY_REQUIRED if args.legacy else FOURCLASS_SMOKE_REQUIRED
    seen = set()
    for run in sorted(required):
        matches = sorted(args.root.glob(f"{args.family}_*__{run}"))
        if not matches:
            _fail(f"missing log dir for {run}")
        log_dir = matches[-1]
        run_log = log_dir / "run.log"
        if not run_log.exists():
            _fail(f"{run}: missing run.log")
        text = run_log.read_text(encoding="utf-8", errors="replace")
        if "OVERALL ALL" not in text:
            _fail(f"{run}: missing OVERALL ALL")

        is_baseline = run.endswith("_baseline") or run == "smoke_baseline_vcache"
        if not is_baseline:
            _check_prune_run(run, text)
            if run.startswith("smoke_c1_") or run.startswith("smoke_c2_"):
                _check_video_prune(run, text)
            if run != "smoke_baseline_vcache" and not args.legacy:
                hits = _ints(r"GeoPACT vision-cache: hits=(\d+)", text)
                if run.startswith("smoke_c2_"):
                    pass  # cache off by design
                elif not hits or max(hits) <= 0:
                    _fail(f"{run}: expected vision-cache hit > 0, got {hits[-5:] if hits else hits}")

        if run == "smoke_abc_r20":
            coord_lines = [int(m.group(1)) for m in re.finditer(r"'coord_metadata_lines':\s*(\d+)", text)]
            if not coord_lines:
                _fail(f"{run}: missing coord_metadata_lines")
            if max(coord_lines) > 5:
                _fail(f"{run}: C injected more than top-5 coord lines: max={max(coord_lines)}")

        seen.add(run)
        print(f"OK {run}: {log_dir}")

    missing = required - seen
    if missing:
        _fail(f"missing runs: {sorted(missing)}")
    print(f"OK qwen official GeoPACT smoke ({args.family})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
