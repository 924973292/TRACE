"""Unified Step/Sequence performance profiling for GUI Agent experiments."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lmms_eval.models.geopact_step_timing import EXCLUSIVE_BUCKET_KEYS

EXCLUSIVE_STEP_KEYS: tuple[str, ...] = EXCLUSIVE_BUCKET_KEYS + (
    "timing_exclusive_sum_ms",
    "timing_e2e_ms",
    "timing_gap_ms",
    "misc_ms",
    "tokens_pruned",
    "llm_tokens",
    "keep_ratio",
    "mrope_path",
)

STEP_PROFILE_KEYS: tuple[str, ...] = (
    "sample_id",
    "sequence_id",
    "step_id",
    "e2e_latency_ms",
    "ttft_ms",
    "vision_prefill_ms",
    "pruning_ms",
    "sticky_keep_ms",
    "llm_prefill_ms",
    "decode_latency_ms",
    "peak_gpu_memory_mb",
    "average_gpu_memory_mb",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "vision_tokens",
    "history_tokens",
    "tokens_before_pruning",
    "tokens_after_pruning",
    "tokens_pruned",
    "llm_tokens",
    "keep_ratio",
    "token_reduction_ratio",
    "gpu_model",
    "precision",
    "batch_size",
    "kv_cache_enabled",
) + EXCLUSIVE_STEP_KEYS

SEQUENCE_PROFILE_KEYS: tuple[str, ...] = (
    "sequence_id",
    "number_of_steps",
    "total_e2e_latency_ms",
    "first_step_ttft_ms",
    "mean_ttft_ms",
    "total_ttft_ms",
    "total_vision_prefill_ms",
    "total_pruning_ms",
    "total_llm_prefill_ms",
    "total_decode_latency_ms",
    "sequence_peak_gpu_memory_mb",
    "sequence_average_gpu_memory_mb",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "total_vision_tokens",
    "total_history_tokens",
    "total_tokens_before_pruning",
    "total_tokens_after_pruning",
    "overall_token_reduction_ratio",
    "average_tokens_per_step",
)

_LATENCY_ALIASES: Dict[str, str] = {
    "vlm_prefill_ms": "vision_prefill_ms",
    "llm_prefill_forward_ms": "llm_prefill_ms",
    "decode_total_ms": "decode_latency_ms",
}

_MEMORY_ALIASES: Dict[str, str] = {
    "gpu_memory_gb": "peak_gpu_memory_mb",
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _finite_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def _usage_from_eval_result(raw: Mapping[str, Any]) -> Dict[str, Any]:
    usage = raw.get("generation_usage")
    if isinstance(usage, dict):
        return dict(usage)
    step_eval = raw.get("step_eval")
    if isinstance(step_eval, dict):
        nested = step_eval.get("generation_usage")
        if isinstance(nested, dict):
            return dict(nested)
    return {}


def _merge_profile_sources(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        for key, value in src.items():
            if value is not None:
                out[key] = value
    return out


def normalize_step_profile(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Map legacy usage keys to the canonical profiling protocol schema."""
    usage = _usage_from_eval_result(raw)
    merged = _merge_profile_sources(raw, usage)

    for src_key, dst_key in _LATENCY_ALIASES.items():
        if dst_key not in merged and src_key in merged:
            merged[dst_key] = merged[src_key]

    if "peak_gpu_memory_mb" not in merged:
        gb = merged.get("gpu_memory_gb")
        if isinstance(gb, (int, float)) and float(gb) > 0:
            merged["peak_gpu_memory_mb"] = float(gb) * 1024.0
        elif "peak_gpu_memory_mb" in usage:
            merged["peak_gpu_memory_mb"] = usage["peak_gpu_memory_mb"]

    if "vision_prefill_ms" not in merged or _finite_float(merged.get("vision_prefill_ms")) == 0.0:
        exclusive = merged.get("timings_exclusive_ms")
        if isinstance(exclusive, dict) and exclusive:
            from lmms_eval.models.geopact_step_timing import vision_prefill_from_exclusive

            merged["vision_prefill_ms"] = vision_prefill_from_exclusive(exclusive)
        else:
            preprocess = _finite_float(merged.get("preprocess_ms"))
            geopact_prepare = _finite_float(merged.get("geopact_prepare_ms"))
            pruning = _finite_float(merged.get("pruning_ms"))
            vlm = _finite_float(merged.get("vlm_prefill_ms"))
            merged["vision_prefill_ms"] = round(
                preprocess + max(geopact_prepare - pruning, 0.0) + vlm, 2
            )

    exclusive = merged.get("timings_exclusive_ms")
    if isinstance(exclusive, dict) and exclusive:
        from lmms_eval.models.geopact_step_timing import llm_prefill_from_exclusive

        merged["llm_prefill_ms"] = llm_prefill_from_exclusive(exclusive)
        for key, value in exclusive.items():
            if key not in merged:
                merged[key] = value

    batch_prefill = _finite_float(merged.get("dialogue_kv_batch_prefill_ms"))
    if batch_prefill > 0:
        merged["llm_prefill_ms"] = round(
            _finite_float(merged.get("llm_prefill_ms")) + batch_prefill,
            2,
        )

    input_tokens = _finite_int(merged.get("input_tokens"))
    output_tokens = _finite_int(merged.get("output_tokens"))
    decode_fwd_calls = merged.get("decode_forward_calls")
    if decode_fwd_calls is None:
        usage = merged.get("generation_usage") or {}
        decode_fwd_calls = usage.get("decode_forward_calls")
    decode_ms_val = _finite_float(merged.get("decode_latency_ms", merged.get("decode_total_ms")))
    if "total_tokens" not in merged:
        merged["total_tokens"] = input_tokens + output_tokens

    before = _finite_int(merged.get("tokens_before_pruning"))
    after = _finite_int(merged.get("tokens_after_pruning"))
    if before <= 0 and after <= 0:
        orig_len = _finite_int(merged.get("orig_seq_len"))
        compact_len = _finite_int(merged.get("compact_seq_len"))
        processor_len = _finite_int(merged.get("processor_prompt_tokens"))
        if orig_len > 0:
            before = orig_len
            after = compact_len if compact_len > 0 else input_tokens
        elif processor_len > 0:
            before = processor_len
            after = input_tokens
        else:
            before = input_tokens
            after = input_tokens
    if before <= 0:
        before = max(input_tokens, after)
    if after <= 0:
        after = input_tokens if input_tokens > 0 else before

    merged["tokens_before_pruning"] = before
    merged["tokens_after_pruning"] = after
    if "token_reduction_ratio" not in merged:
        merged["token_reduction_ratio"] = (
            round((before - after) / before, 6) if before > 0 else 0.0
        )
    merged["tokens_pruned"] = _finite_int(merged.get("tokens_pruned", max(before - after, 0)))
    merged["keep_ratio"] = round(
        _finite_float(merged.get("keep_ratio", after / before if before > 0 else 1.0)),
        6,
    )
    if "llm_tokens" not in merged or _finite_int(merged.get("llm_tokens")) <= 0:
        vis = _finite_int(merged.get("vision_tokens"))
        merged["llm_tokens"] = max(_finite_int(merged.get("input_tokens")) - vis, 0)

    doc = merged.get("doc") if isinstance(merged.get("doc"), dict) else {}
    sample_id = merged.get("sample_id", merged.get("doc_id", doc.get("id")))
    if sample_id is None:
        sample_id = merged.get("step_id", 0)
    sequence_id = merged.get(
        "sequence_id",
        merged.get("episode_id", doc.get("episode_id", "unknown")),
    )
    step_id = merged.get(
        "step_id",
        merged.get("step_index", doc.get("step_index", doc.get("step_id", 0))),
    )

    profile: Dict[str, Any] = {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "step_id": step_id,
        "e2e_latency_ms": round(
            _finite_float(merged.get("timing_e2e_ms", merged.get("e2e_latency_ms"))),
            2,
        ),
        "ttft_ms": round(_finite_float(merged.get("ttft_ms")), 2),
        "vision_prefill_ms": round(_finite_float(merged.get("vision_prefill_ms")), 2),
        "pruning_ms": round(_finite_float(merged.get("pruning_ms")), 2),
        "sticky_keep_ms": round(
            _finite_float(merged.get("sticky_keep_ms", merged.get("pruning_ms"))),
            2,
        ),
        "llm_prefill_ms": round(
            _finite_float(merged.get("llm_prefill_ms", merged.get("llm_prefill_forward_ms"))),
            2,
        ),
        "decode_latency_ms": round(decode_ms_val, 2),
        "decode_per_token_ms": round(
            decode_ms_val / max(output_tokens, 1),
            4,
        ),
        "decode_forward_ratio": round(
            _finite_float(decode_fwd_calls, 0.0) / max(output_tokens, 1),
            4,
        ),
        "decode_forward_calls": round(_finite_float(decode_fwd_calls, 0.0), 2),
        "peak_gpu_memory_mb": round(_finite_float(merged.get("peak_gpu_memory_mb")), 2),
        "average_gpu_memory_mb": round(
            _finite_float(
                merged.get(
                    "average_gpu_memory_mb",
                    merged.get("peak_gpu_memory_mb", 0.0),
                )
            ),
            2,
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _finite_int(merged.get("total_tokens", input_tokens + output_tokens)),
        "vision_tokens": _finite_int(merged.get("vision_tokens")),
        "history_tokens": _finite_int(merged.get("history_tokens")),
        "tokens_before_pruning": before,
        "tokens_after_pruning": after,
        "token_reduction_ratio": round(_finite_float(merged.get("token_reduction_ratio")), 6),
        "tokens_pruned": _finite_int(merged.get("tokens_pruned")),
        "llm_tokens": _finite_int(merged.get("llm_tokens")),
        "keep_ratio": round(_finite_float(merged.get("keep_ratio")), 6),
        "gpu_model": str(merged.get("gpu_model") or os.environ.get("CUDA_VISIBLE_DEVICES", "cuda:0")),
        "precision": str(merged.get("precision") or os.environ.get("GEOPACT_PRECISION", "bf16")),
        "batch_size": _finite_int(merged.get("batch_size"), 1),
        "kv_cache_enabled": bool(
            merged.get(
                "kv_cache_enabled",
                os.environ.get("GEOPACT_VISION_EMBED_CACHE", "1").strip().lower()
                not in {"0", "false", "off", "no"},
            )
        ),
    }
    audit = merged.get("timing_audit")
    mrope_path = None
    if isinstance(audit, dict) and audit.get("mrope_path") is not None:
        mrope_path = str(audit.get("mrope_path"))
    elif merged.get("mrope_path") is not None:
        mrope_path = str(merged.get("mrope_path"))
    profile["mrope_path"] = mrope_path or "unknown"
    # Pass through rectify sub-stage diagnostics (audit channel scalars) so ABC
    # latency breakdowns are analyzable offline from step_profiles alone.
    for src in (audit if isinstance(audit, dict) else {}, merged):
        for key, value in src.items():
            if (
                key.startswith(("rectify_", "slide_"))
                and key not in profile
                and isinstance(value, (int, float, bool, str))
            ):
                profile[key] = value
    # Gap-audit diagnostics (GEOPACT_TIMING_GAP_AUDIT=1): unbucketed wall per
    # bucket-transition edge, for misc_ms attribution.
    for src in (audit if isinstance(audit, dict) else {}, merged):
        gaps = src.get("timing_gaps")
        if isinstance(gaps, dict) and gaps:
            profile["timing_gaps"] = {str(k): _finite_float(v) for k, v in gaps.items()}
            break
    for key in EXCLUSIVE_BUCKET_KEYS:
        if key in merged:
            profile[key] = round(_finite_float(merged.get(key)), 2)
    if "timing_exclusive_sum_ms" in merged:
        profile["timing_exclusive_sum_ms"] = round(
            _finite_float(merged.get("timing_exclusive_sum_ms")), 2
        )
    if "misc_ms" in merged:
        profile["misc_ms"] = round(_finite_float(merged.get("misc_ms")), 2)
    kv_finalize = merged.get("kv_finalize_ms", merged.get("dialogue_kv_finalize_ms"))
    if kv_finalize is not None:
        profile["kv_finalize_ms"] = round(_finite_float(kv_finalize), 2)
    if "timing_e2e_ms" in merged:
        profile["timing_e2e_ms"] = round(_finite_float(merged.get("timing_e2e_ms")), 2)
    if "timing_gap_ms" in merged:
        profile["timing_gap_ms"] = round(_finite_float(merged.get("timing_gap_ms")), 2)
    kv_flow = merged.get("kv_flow_mode")
    if kv_flow is not None:
        profile["kv_flow_mode"] = str(kv_flow)
    batch_pf = merged.get("dialogue_kv_batch_prefill_ms")
    if batch_pf is not None:
        profile["dialogue_kv_batch_prefill_ms"] = round(_finite_float(batch_pf), 2)
    return profile


def aggregate_sequence_profile(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll up normalized step profiles into one sequence record."""
    if not steps:
        return {key: 0 for key in SEQUENCE_PROFILE_KEYS}

    normalized = [normalize_step_profile(step) for step in steps]
    sequence_id = normalized[0].get("sequence_id", "unknown")
    n_steps = len(normalized)

    def _sum(key: str) -> float:
        return sum(_finite_float(row.get(key)) for row in normalized)

    ttfts = [_finite_float(row.get("ttft_ms")) for row in normalized]
    peaks = [_finite_float(row.get("peak_gpu_memory_mb")) for row in normalized]
    avgs = [_finite_float(row.get("average_gpu_memory_mb")) for row in normalized]

    total_before = sum(_finite_int(row.get("tokens_before_pruning")) for row in normalized)
    total_after = sum(_finite_int(row.get("tokens_after_pruning")) for row in normalized)

    return {
        "sequence_id": sequence_id,
        "number_of_steps": n_steps,
        "total_e2e_latency_ms": round(_sum("e2e_latency_ms"), 2),
        "first_step_ttft_ms": round(ttfts[0], 2) if ttfts else 0.0,
        "mean_ttft_ms": round(sum(ttfts) / n_steps, 2) if n_steps else 0.0,
        "total_ttft_ms": round(sum(ttfts), 2),
        "total_vision_prefill_ms": round(_sum("vision_prefill_ms"), 2),
        "total_pruning_ms": round(_sum("pruning_ms"), 2),
        "total_llm_prefill_ms": round(_sum("llm_prefill_ms"), 2),
        "total_decode_latency_ms": round(_sum("decode_latency_ms"), 2),
        "sequence_peak_gpu_memory_mb": round(max(peaks) if peaks else 0.0, 2),
        "sequence_average_gpu_memory_mb": round(
            sum(avgs) / len(avgs) if avgs else 0.0,
            2,
        ),
        "total_input_tokens": sum(_finite_int(row.get("input_tokens")) for row in normalized),
        "total_output_tokens": sum(_finite_int(row.get("output_tokens")) for row in normalized),
        "total_tokens": sum(_finite_int(row.get("total_tokens")) for row in normalized),
        "total_vision_tokens": sum(_finite_int(row.get("vision_tokens")) for row in normalized),
        "total_history_tokens": sum(_finite_int(row.get("history_tokens")) for row in normalized),
        "total_tokens_before_pruning": total_before,
        "total_tokens_after_pruning": total_after,
        "overall_token_reduction_ratio": round(
            (total_before - total_after) / total_before if total_before > 0 else 0.0,
            6,
        ),
        "average_tokens_per_step": round(
            sum(_finite_int(row.get("input_tokens")) for row in normalized) / n_steps,
            2,
        )
        if n_steps
        else 0.0,
    }


def write_profile_jsonl(path: os.PathLike | str, records: Iterable[Mapping[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def _collect_metric_scalars_from_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Pull scalar profiling values from lmms_eval per-metric dict records."""
    out: Dict[str, Any] = {}
    profile_keys = set(STEP_PROFILE_KEYS) | set(_LATENCY_ALIASES.keys()) | set(_MEMORY_ALIASES.keys())
    for key, value in sample.items():
        if key in profile_keys and isinstance(value, (int, float, bool, str)):
            out[key] = value
            continue
        if isinstance(value, dict):
            scalar = value.get(key)
            if scalar is None and key in profile_keys:
                for alias in (*_LATENCY_ALIASES.keys(), *_MEMORY_ALIASES.keys()):
                    if alias in value:
                        scalar = value.get(alias)
                        break
            if isinstance(scalar, (int, float, bool, str)):
                out[key] = scalar
    return out


def extract_step_profile_from_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a step profile dict from one lmms_eval logged sample."""
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
    step_eval = sample.get("step_eval") if isinstance(sample.get("step_eval"), dict) else {}

    doc_id = sample.get("doc_id", doc.get("id"))
    raw: Dict[str, Any] = {
        "sample_id": doc_id,
        "sequence_id": doc.get("episode_id", step_eval.get("episode_id", "unknown")),
        "step_id": doc.get("step_index", step_eval.get("step_id", doc_id if doc_id is not None else 0)),
        "doc": doc,
        "step_eval": step_eval,
    }
    usage = _merge_profile_sources(
        _usage_from_eval_result(step_eval),
        _usage_from_eval_result(sample),
    )
    raw.update(_collect_metric_scalars_from_sample(sample))
    if isinstance(step_eval, dict):
        raw.update(_collect_metric_scalars_from_sample(step_eval))
    raw.update(usage)
    return normalize_step_profile(raw)


def aggregate_sequence_profiles(steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for step in steps:
        profile = extract_step_profile_from_sample(step) if "e2e_latency_ms" not in step else normalize_step_profile(step)
        seq_key = str(profile.get("sequence_id", "unknown"))
        grouped[seq_key].append(profile)
    return [aggregate_sequence_profile(group) for group in grouped.values()]


def export_performance_profiles(output_dir: os.PathLike | str, samples: Mapping[str, Any] | Sequence[Any]) -> Dict[str, int]:
    """Write step_profiles.jsonl and sequence_profiles.jsonl under output_dir/performance_profiles/."""
    out_root = Path(output_dir)
    prof_dir = out_root / "performance_profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    flat_samples: List[Dict[str, Any]] = []
    if isinstance(samples, dict):
        for task_samples in samples.values():
            if isinstance(task_samples, list):
                flat_samples.extend(s for s in task_samples if isinstance(s, dict))
    elif isinstance(samples, list):
        flat_samples = [s for s in samples if isinstance(s, dict)]

    step_profiles = [extract_step_profile_from_sample(sample) for sample in flat_samples]
    sequence_profiles = aggregate_sequence_profiles(flat_samples)

    step_path = prof_dir / "step_profiles.jsonl"
    seq_path = prof_dir / "sequence_profiles.jsonl"
    n_steps = write_profile_jsonl(step_path, step_profiles)
    n_seq = write_profile_jsonl(seq_path, sequence_profiles)
    return {"step_profiles": n_steps, "sequence_profiles": n_seq}


def export_profiles_from_run_dir(run_dir: os.PathLike | str) -> Dict[str, int]:
    """Load saved sample JSONL from a run directory and export performance profiles."""
    run_path = Path(run_dir)
    sample_files = sorted(run_path.glob("*samples*.jsonl"))
    if not sample_files:
        sample_files = sorted(run_path.rglob("*samples*.jsonl"))
    flat: List[Dict[str, Any]] = []
    for path in sample_files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    flat.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not flat:
        return {"step_profiles": 0, "sequence_profiles": 0}
    return export_performance_profiles(run_path, flat)


def latency_breakdown_percentages(step: Mapping[str, Any]) -> Dict[str, float]:
    """Return VisionPrefill/Pruning/LLMPrefill/Decode share of e2e latency."""
    profile = normalize_step_profile(step)
    e2e = _finite_float(profile.get("e2e_latency_ms"))
    if e2e <= 0:
        return {"vision_prefill_pct": 0.0, "pruning_pct": 0.0, "llm_prefill_pct": 0.0, "decode_pct": 0.0}
    parts = {
        "vision_prefill_pct": _finite_float(profile.get("vision_prefill_ms")) / e2e * 100.0,
        "pruning_pct": _finite_float(profile.get("pruning_ms")) / e2e * 100.0,
        "llm_prefill_pct": _finite_float(profile.get("llm_prefill_ms")) / e2e * 100.0,
        "decode_pct": _finite_float(profile.get("decode_latency_ms")) / e2e * 100.0,
    }
    return {k: round(v, 2) for k, v in parts.items()}
