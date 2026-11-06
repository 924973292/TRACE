#!/usr/bin/env python3
"""Audit baseline_kv run.log: step0 must use native_first_step, not dialogue_embed."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NATIVE_RE = re.compile(r"dialogue_kv_reason['\"]?\s*[:=]\s*['\"]?native_first_step", re.I)
FIRST_STEP_RE = re.compile(r"dialogue_kv_reason['\"]?\s*[:=]\s*['\"]?first_step", re.I)
DIALOGUE_EMBED_FULL_RE = re.compile(r"GeoPACT dialogue-embed:\s*\{", re.I)
DIALOGUE_EMBED_SUFFIX_RE = re.compile(r"GeoPACT dialogue-embed \(suffix\)", re.I)
SOURCE_COMPACT_RE = re.compile(r"GeoPACT source-compact", re.I)
HIT_RE = re.compile(r"dialogue_kv_reason['\"]?\s*[:=]\s*['\"]?hit", re.I)
TRACEBACK_RE = re.compile(r"^Traceback ", re.M)


def audit_log(text: str) -> dict:
    return {
        "native_first_step": len(NATIVE_RE.findall(text)),
        "first_step_legacy": len(FIRST_STEP_RE.findall(text)),
        "dialogue_embed_full": len(DIALOGUE_EMBED_FULL_RE.findall(text)),
        "dialogue_embed_suffix": len(DIALOGUE_EMBED_SUFFIX_RE.findall(text)),
        "source_compact": len(SOURCE_COMPACT_RE.findall(text)),
        "dialogue_kv_hit": len(HIT_RE.findall(text)),
        "traceback": bool(TRACEBACK_RE.search(text)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log", type=Path, help="baseline_kv_rewrite/run.log")
    ap.add_argument("--require-native", action="store_true", default=True)
    args = ap.parse_args()

    if not args.run_log.is_file():
        print(f"FAIL missing log: {args.run_log}", file=sys.stderr)
        return 1

    text = args.run_log.read_text(encoding="utf-8", errors="replace")
    stats = audit_log(text)
    print(f"native_first_step={stats['native_first_step']}")
    print(f"first_step_legacy={stats['first_step_legacy']}")
    print(f"dialogue_embed_full={stats['dialogue_embed_full']}")
    print(f"dialogue_embed_suffix={stats['dialogue_embed_suffix']}")
    print(f"source_compact={stats['source_compact']}")
    print(f"dialogue_kv_hit={stats['dialogue_kv_hit']}")
    print(f"traceback={stats['traceback']}")

    errors: list[str] = []
    if stats["traceback"]:
        errors.append("run.log contains Traceback")
    if args.require_native and stats["dialogue_embed_full"] > 0:
        errors.append(
            f"step0 dialogue_embed detected ({stats['dialogue_embed_full']}x); "
            "expected native_first_step path"
        )
    if args.require_native and stats["native_first_step"] == 0 and stats["first_step_legacy"] > 0:
        errors.append("legacy first_step without native_first_step")

    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1

    print("OK kv_step0_path_audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
