#!/usr/bin/env python3
"""Verify OmniGUI eval output has complete metrics (step + episode latency + 6 dims)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (REPO, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from omnigui_data_utils import preferred_omnigui_data_file
from omnigui_metrics_common import DIM_ORDER, category_from_doc, compute_episode_latency_from_samples, normalize_category

REQUIRED_LOG_LINES = ("OVERALL ALL", "SUBTASK", "GROUP", "DIM_HEADER", "DIM_VALUES", "LATENCY_STEP")
OPTIONAL_LOG_LINES = ("LATENCY_EPISODE", "LATENCY_BREAKDOWN")
PROFILING_RESULT_KEYS = (
    "e2e_latency_ms,none",
    "ttft_ms,none",
    "vision_prefill_ms,none",
    "pruning_ms,none",
    "llm_prefill_ms,none",
    "decode_latency_ms,none",
    "peak_gpu_memory_mb,none",
    "input_tokens,none",
    "output_tokens,none",
    "vision_tokens,none",
    "history_tokens,none",
    "tokens_before_pruning,none",
    "tokens_after_pruning,none",
    "token_reduction_ratio,none",
    "episode_sum_e2e_latency_ms,none",
    "episode_sum_vision_prefill_ms,none",
)


def latest(path: Path, pattern: str) -> Path | None:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse_log_metrics(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    out: dict[str, str] = {}
    for line in text.splitlines():
        for prefix in REQUIRED_LOG_LINES + OPTIONAL_LOG_LINES:
            if line.startswith(prefix):
                out[prefix] = line
    return out


def dimension_from_samples(samples_path: Path, docs: list[dict]) -> dict[str, dict]:
    from collections import defaultdict

    records: list[dict] = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = int(sample.get("doc_id", 0))
            doc = docs[doc_id] if 0 <= doc_id < len(docs) else {}
            step_eval = sample.get("step_eval") if isinstance(sample.get("step_eval"), dict) else {}
            records.append(
                {
                    "category": category_from_doc(doc, step_eval),
                    "type_match": bool(step_eval.get("type_match", sample.get("type_match", False))),
                    "exact_match": bool(step_eval.get("exact_match", sample.get("exact_match", False))),
                    "subset": doc.get("subset", "unknown"),
                    "episode_id": doc.get("episode_id", "unknown"),
                    "step_id": doc.get("step_index", doc.get("step_id", doc_id)),
                }
            )

    def agg(recs: list[dict]) -> dict:
        if not recs:
            return {"tm": None, "em": None, "count": 0}
        return {
            "count": len(recs),
            "tm": sum(1.0 if r["type_match"] else 0.0 for r in recs) / len(recs),
            "em": sum(1.0 if r["exact_match"] else 0.0 for r in recs) / len(recs),
        }

    out = {"Overall": agg(records)}
    for short, cat in DIM_ORDER[1:]:
        out[short] = agg([r for r in records if r["category"] == cat])
    return out


def verify_run_dir(run_dir: Path, *, append_summary: bool = False, require_profiling: bool = False) -> int:
    log_path = run_dir / "run.log"
    if not log_path.is_file():
        # launch script may write directly to run.log at root
        alt = list(run_dir.glob("**/run.log"))
        log_path = alt[0] if alt else log_path

    issues: list[str] = []
    log_metrics = parse_log_metrics(log_path)
    for prefix in REQUIRED_LOG_LINES:
        if prefix not in log_metrics:
            issues.append(f"missing log line: {prefix}")

    for prefix in OPTIONAL_LOG_LINES:
        if prefix not in log_metrics:
            issues.append(f"missing log line: {prefix}")

    dim_line = log_metrics.get("DIM_VALUES", "")
    if dim_line.count("n/a") > 18:
        issues.append(f"too many n/a in DIM_VALUES ({dim_line.count('n/a')})")

    results_path = latest(run_dir, "**/*_results.json")
    samples_path = latest(run_dir, "**/*_samples_*.jsonl")
    if results_path:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        task_name = next(iter(data.get("results", {})), "")
        rec = data.get("results", {}).get(task_name, {})
        for key in (
            "episode_sum_e2e_latency_ms,none",
            "episode_mean_step_e2e_latency_ms,none",
            "type_match,none",
            "e2e_latency_ms,none",
            *PROFILING_RESULT_KEYS,
        ):
            if key not in rec and require_profiling:
                issues.append(f"results.json missing {key}")
            elif key not in rec and key in PROFILING_RESULT_KEYS[:6]:
                issues.append(f"results.json missing profiling key {key}")

    docs_path = preferred_omnigui_data_file(REPO)
    docs = json.loads(docs_path.read_text(encoding="utf-8")) if docs_path else []
    if samples_path and docs:
        dims = dimension_from_samples(samples_path, docs)
        sample_count = int(dims.get("Overall", {}).get("count", 0) or 0)
        for short, _ in DIM_ORDER[1:]:
            count = dims.get(short, {}).get("count", 0)
            if count == 0 and sample_count >= 200:
                issues.append(f"dimension {short} has zero samples (category alias bug?)")
        if samples_path:
            samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            derived = compute_episode_latency_from_samples(samples)
            if derived.get("episode_sum_e2e_latency_ms") is None:
                issues.append("cannot derive episode_sum_e2e_latency_ms from samples")

    if require_profiling or (run_dir / "performance_profiles").is_dir():
        try:
            from verify_performance_profiles import verify_run_dir as verify_profiles

            if verify_profiles(run_dir, min_steps=1) != 0:
                issues.append("performance profile verification failed")
        except Exception as exc:
            issues.append(f"performance profile check error: {exc}")

    print(f"=== verify {run_dir} ===")
    for prefix in REQUIRED_LOG_LINES + OPTIONAL_LOG_LINES:
        print(log_metrics.get(prefix, f"[MISSING] {prefix}"))
    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("OK: metrics complete")
    if append_summary:
        pass  # optional: python scripts/append_omnigui_log_summary.py --task-dir ...
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--append-summary", action="store_true")
    parser.add_argument("--require-profiling", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        verify_run_dir(
            Path(args.run_dir),
            append_summary=args.append_summary,
            require_profiling=args.require_profiling,
        )
    )


if __name__ == "__main__":
    main()
