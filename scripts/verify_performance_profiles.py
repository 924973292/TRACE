#!/usr/bin/env python3
"""Verify step/sequence performance profile JSONL matches the profiling protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lmms_eval.tasks.agentcpm_gui.performance_profile import (  # noqa: E402
    EXCLUSIVE_BUCKET_KEYS,
    SEQUENCE_PROFILE_KEYS,
    STEP_PROFILE_KEYS,
    aggregate_sequence_profile,
    export_profiles_from_run_dir,
    normalize_step_profile,
)

# Fields that must be present (value may be 0 for pruning/token reduction on baselines).
REQUIRED_STEP_KEYS = tuple(STEP_PROFILE_KEYS)

# Latency/token fields that must be > 0 on a successful GPU inference step.
REQUIRED_NONZERO_STEP_KEYS = (
    "e2e_latency_ms",
    "ttft_ms",
    "vision_prefill_ms",
    "llm_prefill_ms",
    "decode_latency_ms",
    "peak_gpu_memory_mb",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def verify_step_profiles(rows: List[Dict[str, Any]], *, min_rows: int = 1) -> List[str]:
    issues: List[str] = []
    if len(rows) < min_rows:
        issues.append(f"step_profiles.jsonl has {len(rows)} rows, need >= {min_rows}")
        return issues

    for idx, row in enumerate(rows):
        norm = normalize_step_profile(row)
        missing = [k for k in REQUIRED_STEP_KEYS if k not in norm]
        if missing:
            issues.append(f"step[{idx}] missing keys: {missing}")
            continue
        for key in REQUIRED_NONZERO_STEP_KEYS:
            val = norm.get(key)
            if val is None or (isinstance(val, (int, float)) and float(val) <= 0):
                issues.append(f"step[{idx}] {key} invalid: {val!r}")
        if norm.get("pruning_ms") is None:
            issues.append(f"step[{idx}] pruning_ms missing")
        if norm.get("token_reduction_ratio") is None:
            issues.append(f"step[{idx}] token_reduction_ratio missing")
        _verify_exclusive_ledger(norm, idx, issues)
    return issues


def _verify_exclusive_ledger(norm: Dict[str, Any], idx: int, issues: List[str]) -> None:
    """Check mutually exclusive timing buckets close against ledger e2e (when present)."""
    e2e = float(norm.get("timing_e2e_ms") or norm.get("e2e_latency_ms") or 0.0)
    if e2e <= 0:
        return
    has_exclusive = any(norm.get(k) is not None for k in EXCLUSIVE_BUCKET_KEYS)
    if not has_exclusive:
        return
    bucket_sum = sum(float(norm.get(k) or 0.0) for k in EXCLUSIVE_BUCKET_KEYS)
    misc = float(norm.get("misc_ms") or 0.0)
    total = bucket_sum + misc
    gap = float(norm.get("timing_gap_ms") if norm.get("timing_gap_ms") is not None else (e2e - total))
    rel_err = abs(gap) / e2e
    if rel_err > 0.02 and abs(gap) > 25.0:
        issues.append(
            f"step[{idx}] exclusive ledger closure failed: "
            f"sum={total:.1f}ms e2e={e2e:.1f}ms gap={gap:.1f}ms ({rel_err * 100:.1f}%)"
        )


def verify_sequence_rollup(
    step_rows: List[Dict[str, Any]],
    seq_rows: List[Dict[str, Any]],
    *,
    tolerance: float = 0.01,
) -> List[str]:
    issues: List[str] = []
    if not seq_rows:
        issues.append("sequence_profiles.jsonl is empty")
        return issues

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in step_rows:
        norm = normalize_step_profile(row)
        grouped.setdefault(str(norm["sequence_id"]), []).append(norm)

    derived = {str(r["sequence_id"]): r for r in (aggregate_sequence_profile(g) for g in grouped.values())}
    for seq_id, expected in derived.items():
        logged = next((r for r in seq_rows if str(r.get("sequence_id")) == seq_id), None)
        if logged is None:
            issues.append(f"sequence {seq_id} missing from sequence_profiles.jsonl")
            continue
        exp_e2e = float(expected.get("total_e2e_latency_ms", 0))
        log_e2e = float(logged.get("total_e2e_latency_ms", 0))
        if exp_e2e > 0 and abs(log_e2e - exp_e2e) / exp_e2e > tolerance:
            issues.append(
                f"sequence {seq_id} total_e2e_latency_ms mismatch: logged={log_e2e} expected={exp_e2e}"
            )
    return issues


def _expected_step_count(run_dir: Path) -> int | None:
    for path in sorted(run_dir.glob("**/*_results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ns = data.get("n-samples") or {}
        if isinstance(ns, dict):
            for rec in ns.values():
                if isinstance(rec, dict):
                    eff = rec.get("effective", rec.get("original"))
                    if isinstance(eff, (int, float)) and int(eff) > 0:
                        return int(eff)
    return None


def verify_run_dir(run_dir: Path, *, min_steps: int = 1, require_profiles: bool = True) -> int:
    prof_dir = run_dir / "performance_profiles"
    step_path = prof_dir / "step_profiles.jsonl"
    seq_path = prof_dir / "sequence_profiles.jsonl"

    if not step_path.is_file() and require_profiles:
        export_profiles_from_run_dir(run_dir)

    step_rows = _load_jsonl(step_path)
    seq_rows = _load_jsonl(seq_path)

    expected = _expected_step_count(run_dir)
    required_min = max(min_steps, expected or 0) if expected else min_steps

    issues: List[str] = []
    issues.extend(verify_step_profiles(step_rows, min_rows=required_min))
    if step_rows:
        issues.extend(verify_sequence_rollup(step_rows, seq_rows))

    log_path = run_dir / "run.log"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "OVERALL ALL" in text:
            for required in ("LATENCY_STEP", "LATENCY_BREAKDOWN", "LATENCY_EPISODE"):
                if required not in text:
                    issues.append(f"run.log missing {required}")

    print(f"=== verify performance profiles: {run_dir} ===")
    print(f"  step_profiles: {len(step_rows)} rows")
    print(f"  sequence_profiles: {len(seq_rows)} rows")
    if step_rows:
        sample = normalize_step_profile(step_rows[0])
        print(
            "  sample step: "
            f"e2e={sample.get('e2e_latency_ms')}ms "
            f"vision={sample.get('vision_prefill_ms')}ms "
            f"prune={sample.get('pruning_ms')}ms "
            f"llm={sample.get('llm_prefill_ms')}ms "
            f"decode={sample.get('decode_latency_ms')}ms "
            f"peak_gpu={sample.get('peak_gpu_memory_mb')}MB "
            f"tokens={sample.get('input_tokens')}/{sample.get('output_tokens')}"
        )
    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("OK: performance profiles complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--no-require", action="store_true")
    args = parser.parse_args()
    return verify_run_dir(
        args.run_dir,
        min_steps=args.min_steps,
        require_profiles=not args.no_require,
    )


if __name__ == "__main__":
    raise SystemExit(main())
