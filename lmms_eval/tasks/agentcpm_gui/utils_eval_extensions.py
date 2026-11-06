"""Eval extensions layered on upstream OmniGUI utils.py (unchanged).

- Fixed omni_process_results (upstream returns eval dict as metric values — broken).
- GUI-Owl native_mt4 doc_to_messages wrappers (live in gui_owl_native_messages.py).
- merged_sorted → official qwen_doc_to_messages adapter (data shaping only).
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as eval_logger

from lmms_eval.tasks.agentcpm_gui import utils as omni_utils
from lmms_eval.tasks.agentcpm_gui.gui_owl_native_messages import (
    gui_owl_native_current_doc_to_messages as _gui_owl_native_current_impl,
    gui_owl_native_multiturn_doc_to_messages as _gui_owl_native_multiturn_impl,
    parse_action_history_text,
)
from lmms_eval.tasks.agentcpm_gui.performance_profile import normalize_step_profile


# ---------------------------------------------------------------------------
# Generation usage / metric map (from pre-official overlay; not in upstream)
# ---------------------------------------------------------------------------

def _split_generation_output(output: Any) -> Tuple[str, Dict[str, Any]]:
    usage_marker_start = "<MINICPM_O_USAGE_JSON>"
    usage_marker_end = "</MINICPM_O_USAGE_JSON>"
    if isinstance(output, dict):
        text = output.get("text", "")
        if not text and "choices" in output:
            choices = output.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content", "")
        usage = output.get("usage", {}) or {}
        return str(text), usage if isinstance(usage, dict) else {}
    text = str(output or "")
    if usage_marker_start in text and usage_marker_end in text:
        prefix, rest = text.split(usage_marker_start, 1)
        usage_blob, suffix = rest.split(usage_marker_end, 1)
        try:
            usage = json.loads(usage_blob)
        except json.JSONDecodeError:
            usage = {}
        return (prefix + suffix).strip(), usage if isinstance(usage, dict) else {}
    return text, {}


def _attach_generation_usage(eval_result: Dict[str, Any], usage: Dict[str, Any]) -> Dict[str, Any]:
    if not usage:
        return eval_result
    out = dict(eval_result)
    out["generation_usage"] = usage
    return out


_OMNI_EPISODE_EVAL_KEYS = ("subset", "episode_id", "step_id", "exact_match")

_OMNI_PROFILE_SCALAR_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "preprocess_ms",
    "generation_ms",
    "e2e_latency_ms",
    "ttft_ms",
    "prefill_ms",
    "vlm_prefill_ms",
    "vision_prefill_ms",
    "pruning_ms",
    "llm_prefill_forward_ms",
    "llm_prefill_ms",
    "decode_total_ms",
    "decode_latency_ms",
    "decode_per_token_ms",
    "throughput_tokens_per_s",
    "forward_calls",
    "decode_forward_calls",
    "gpu_memory_gb",
    "peak_gpu_memory_mb",
    "average_gpu_memory_mb",
    "vision_tokens",
    "history_tokens",
    "tokens_before_pruning",
    "tokens_after_pruning",
    "token_reduction_ratio",
    "misc_ms",
    "kv_finalize_ms",
    "dialogue_kv_finalize_ms",
    "dialogue_suffix_mrope_ms",
)

_EPISODE_SUM_KEYS = (
    "e2e_latency_ms",
    "generation_ms",
    "input_tokens",
    "vision_prefill_ms",
    "pruning_ms",
    "llm_prefill_ms",
    "decode_latency_ms",
    "output_tokens",
    "total_tokens",
    "vision_tokens",
    "history_tokens",
    "tokens_before_pruning",
    "tokens_after_pruning",
)

_EPISODE_MEAN_KEYS = ("e2e_latency_ms", "ttft_ms", "token_reduction_ratio")

_EPISODE_MAX_KEYS = ("gpu_memory_gb", "peak_gpu_memory_mb")


def _profile_scalar(eval_result: Dict[str, Any], key: str) -> float:
    usage = eval_result.get("generation_usage") or {}
    value = usage.get(key)
    if value is None:
        value = eval_result.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _profile_metric_record(eval_result: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {
        "subset": eval_result.get("subset", "unknown"),
        "episode_id": eval_result.get("episode_id", "unknown"),
        "step_id": eval_result.get("step_id", eval_result.get("step_index", 0)),
        key: _profile_scalar(eval_result, key),
    }


def _omni_metric_result_map(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    episode_eval = {key: eval_result.get(key) for key in _OMNI_EPISODE_EVAL_KEYS}
    out: Dict[str, Any] = {
        "step_eval": eval_result,
        "type_match": bool(eval_result.get("type_match", False)),
        "exact_match": bool(eval_result.get("exact_match", False)),
        "success_rate": episode_eval,
        "goal_progress": episode_eval,
    }
    for key in _OMNI_PROFILE_SCALAR_KEYS:
        out[key] = _profile_metric_record(eval_result, key)
    out["episode_sum_e2e_latency_ms"] = out["e2e_latency_ms"]
    out["episode_mean_step_e2e_latency_ms"] = out["e2e_latency_ms"]
    out["episode_sum_generation_ms"] = out["generation_ms"]
    out["episode_sum_input_tokens"] = out["input_tokens"]
    out["episode_max_gpu_memory_gb"] = out["gpu_memory_gb"]
    for key in _EPISODE_SUM_KEYS:
        metric_key = f"episode_sum_{key}"
        if metric_key not in out:
            out[metric_key] = out.get(key, _profile_metric_record(eval_result, key))
    for key in _EPISODE_MEAN_KEYS:
        metric_key = f"episode_mean_{key}"
        if metric_key not in out:
            out[metric_key] = out.get(key, _profile_metric_record(eval_result, key))
    for key in _EPISODE_MAX_KEYS:
        metric_key = f"episode_max_{key}"
        if metric_key not in out:
            out[metric_key] = out.get(key, _profile_metric_record(eval_result, key))
    return out


def omni_process_results(
    doc: Dict,
    results: List[str],
    lmms_eval_specific_kwargs: Optional[Dict] = None,
) -> Dict:
    """Score with upstream evaluate_omni_step; fix upstream omni_process_results return shape."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pred_output, generation_usage = _split_generation_output(results[0] if results else "")
    coordinate_type = lmms_eval_specific_kwargs.get("coordinate_type", "auto")
    screen_width = lmms_eval_specific_kwargs.get("screen_width", 1080)
    screen_height = lmms_eval_specific_kwargs.get("screen_height", 2400)

    eval_result = omni_utils.evaluate_omni_step(
        doc,
        pred_output,
        screen_width=screen_width,
        screen_height=screen_height,
        coordinate_type=coordinate_type,
    )
    eval_result = _attach_generation_usage(eval_result, generation_usage)
    if generation_usage:
        normalized = normalize_step_profile({**eval_result, **generation_usage})
        eval_result = {**eval_result, **normalized}
        eval_result["generation_usage"] = {**generation_usage, **normalized}
    return _omni_metric_result_map(eval_result)


# ---------------------------------------------------------------------------
# Timing aggregates (task yaml references)
# ---------------------------------------------------------------------------

def _numeric_profile_values(results: List[Dict], key: str) -> List[float]:
    values: List[float] = []
    for result in results:
        if isinstance(result, (int, float)):
            if math.isfinite(float(result)):
                values.append(float(result))
            continue
        value = result.get(key)
        if value is None:
            value = (result.get("generation_usage") or {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _aggregate_profile_mean(results: List[Dict], key: str, label: str) -> float:
    values = _numeric_profile_values(results, key)
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    eval_logger.info(f"{label} [step-level]: mean={mean_value:.4f} from {len(values)} steps")
    return mean_value


def _episode_group_key(record: Dict[str, Any]) -> str:
    return f"{record.get('subset', 'unknown')}-{record.get('episode_id', 'unknown')}"


def _group_results_by_episode(results: List[Dict]) -> Dict[str, List[Dict]]:
    episode_results: Dict[str, List[Dict]] = defaultdict(list)
    for result in results:
        if isinstance(result, dict):
            episode_results[_episode_group_key(result)].append(result)
    return episode_results


def _aggregate_episode_profile(results: List[Dict], key: str, label: str, *, step_reducer: str) -> float:
    episode_results = _group_results_by_episode(results)
    episode_values: List[float] = []
    for ep_results in episode_results.values():
        step_values = _numeric_profile_values(ep_results, key)
        if not step_values:
            continue
        if step_reducer == "sum":
            episode_values.append(sum(step_values))
        elif step_reducer == "mean":
            episode_values.append(sum(step_values) / len(step_values))
        elif step_reducer == "max":
            episode_values.append(max(step_values))
    if not episode_values:
        return 0.0
    return sum(episode_values) / len(episode_values)


def _metric_bool(result: Any, key: str) -> bool:
    """lmms_eval appends process_results[metric] per step — may be bare bool or dict."""
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        return bool(result.get(key, False))
    return False


def omni_aggregate_type_match(results: List[Any]) -> float:
    if not results:
        return 0.0
    total = len(results)
    correct = sum(1 for r in results if _metric_bool(r, "type_match"))
    accuracy = correct / total if total > 0 else 0.0
    eval_logger.info(f"Type Match: {correct}/{total} = {accuracy:.4f}")
    return accuracy


def omni_aggregate_exact_match(results: List[Any]) -> float:
    if not results:
        return 0.0
    total = len(results)
    correct = sum(1 for r in results if _metric_bool(r, "exact_match"))
    accuracy = correct / total if total > 0 else 0.0
    eval_logger.info(f"Exact Match: {correct}/{total} = {accuracy:.4f}")
    return accuracy


omni_aggregate_success_rate = omni_utils.agentcpm_aggregate_success_rate
omni_aggregate_goal_progress = omni_utils.agentcpm_aggregate_goal_progress


def omni_aggregate_input_tokens(results: List[Dict]) -> float:
    return _aggregate_profile_mean(results, "input_tokens", "Input Tokens")


def omni_aggregate_output_tokens(results: List[Dict]) -> float:
    return _aggregate_profile_mean(results, "output_tokens", "Output Tokens")
omni_aggregate_e2e_latency_ms = lambda r: _aggregate_profile_mean(r, "e2e_latency_ms", "E2E Latency (ms)")
omni_aggregate_ttft_ms = lambda r: _aggregate_profile_mean(r, "ttft_ms", "TTFT (ms)")
omni_aggregate_preprocess_ms = lambda r: _aggregate_profile_mean(r, "preprocess_ms", "Preprocess Latency (ms)")
omni_aggregate_generation_ms = lambda r: _aggregate_profile_mean(r, "generation_ms", "Generation Latency (ms)")
omni_aggregate_prefill_ms = lambda r: _aggregate_profile_mean(r, "prefill_ms", "Prefill (ms)")
omni_aggregate_vlm_prefill_ms = lambda r: _aggregate_profile_mean(r, "vlm_prefill_ms", "VLM Prefill (ms)")
omni_aggregate_llm_prefill_forward_ms = lambda r: _aggregate_profile_mean(r, "llm_prefill_forward_ms", "LLM Prefill Forward (ms)")
omni_aggregate_decode_total_ms = lambda r: _aggregate_profile_mean(r, "decode_total_ms", "Decode Total Latency (ms)")
omni_aggregate_decode_per_token_ms = lambda r: _aggregate_profile_mean(r, "decode_per_token_ms", "Decode / Token (ms)")
omni_aggregate_throughput_tokens_per_s = lambda r: _aggregate_profile_mean(r, "throughput_tokens_per_s", "Throughput (tokens/s)")
omni_aggregate_episode_sum_e2e_latency_ms = lambda r: _aggregate_episode_profile(r, "e2e_latency_ms", "Episode Sum E2E", step_reducer="sum")
omni_aggregate_episode_mean_step_e2e_latency_ms = lambda r: _aggregate_episode_profile(r, "e2e_latency_ms", "Episode Mean Step E2E", step_reducer="mean")
omni_aggregate_episode_sum_generation_ms = lambda r: _aggregate_episode_profile(r, "generation_ms", "Episode Sum Generation", step_reducer="sum")
omni_aggregate_episode_sum_input_tokens = lambda r: _aggregate_episode_profile(r, "input_tokens", "Episode Sum Input Tokens", step_reducer="sum")
omni_aggregate_episode_max_gpu_memory_gb = lambda r: _aggregate_episode_profile(r, "gpu_memory_gb", "Episode Peak GPU", step_reducer="max")
omni_aggregate_peak_gpu_memory_mb = lambda r: _aggregate_profile_mean(r, "peak_gpu_memory_mb", "Peak GPU Memory (MB)")
omni_aggregate_average_gpu_memory_mb = lambda r: _aggregate_profile_mean(r, "average_gpu_memory_mb", "Average GPU Memory (MB)")
omni_aggregate_vision_prefill_ms = lambda r: _aggregate_profile_mean(r, "vision_prefill_ms", "Vision Prefill (ms)")
omni_aggregate_pruning_ms = lambda r: _aggregate_profile_mean(r, "pruning_ms", "Pruning (ms)")
omni_aggregate_llm_prefill_ms = lambda r: _aggregate_profile_mean(r, "llm_prefill_ms", "LLM Prefill (ms)")
omni_aggregate_decode_latency_ms = lambda r: _aggregate_profile_mean(r, "decode_latency_ms", "Decode Latency (ms)")
omni_aggregate_misc_ms = lambda r: _aggregate_profile_mean(r, "misc_ms", "Misc (ms)")
omni_aggregate_kv_finalize_ms = lambda r: _aggregate_profile_mean(
    r, "kv_finalize_ms", "KV Finalize (ms)"
)
omni_aggregate_dialogue_suffix_mrope_ms = lambda r: _aggregate_profile_mean(
    r, "dialogue_suffix_mrope_ms", "Dialogue Suffix MRoPE (ms)"
)
omni_aggregate_total_tokens = lambda r: _aggregate_profile_mean(r, "total_tokens", "Total Tokens")
omni_aggregate_vision_tokens = lambda r: _aggregate_profile_mean(r, "vision_tokens", "Vision Tokens")
omni_aggregate_history_tokens = lambda r: _aggregate_profile_mean(r, "history_tokens", "History Tokens")
omni_aggregate_tokens_before_pruning = lambda r: _aggregate_profile_mean(r, "tokens_before_pruning", "Tokens Before Pruning")
omni_aggregate_tokens_after_pruning = lambda r: _aggregate_profile_mean(r, "tokens_after_pruning", "Tokens After Pruning")
omni_aggregate_token_reduction_ratio = lambda r: _aggregate_profile_mean(r, "token_reduction_ratio", "Token Reduction Ratio")
omni_aggregate_episode_sum_vision_prefill_ms = lambda r: _aggregate_episode_profile(r, "vision_prefill_ms", "Episode Sum Vision Prefill", step_reducer="sum")
omni_aggregate_episode_sum_pruning_ms = lambda r: _aggregate_episode_profile(r, "pruning_ms", "Episode Sum Pruning", step_reducer="sum")
omni_aggregate_episode_sum_llm_prefill_ms = lambda r: _aggregate_episode_profile(r, "llm_prefill_ms", "Episode Sum LLM Prefill", step_reducer="sum")
omni_aggregate_episode_sum_decode_latency_ms = lambda r: _aggregate_episode_profile(r, "decode_latency_ms", "Episode Sum Decode", step_reducer="sum")
omni_aggregate_episode_max_peak_gpu_memory_mb = lambda r: _aggregate_episode_profile(r, "peak_gpu_memory_mb", "Episode Peak GPU MB", step_reducer="max")


def omni_aggregate_sequence_profiles(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from lmms_eval.tasks.agentcpm_gui.performance_profile import aggregate_sequence_profiles

    return aggregate_sequence_profiles(samples)


# ---------------------------------------------------------------------------
# GUI-Owl native wrappers (prompt lives in gui_owl_native_messages.py)
# ---------------------------------------------------------------------------

def gui_owl_native_current_doc_to_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    return _gui_owl_native_current_impl(doc, lmms_eval_specific_kwargs)


def gui_owl_native_multiturn_doc_to_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    return _gui_owl_native_multiturn_impl(doc, lmms_eval_specific_kwargs)


# ---------------------------------------------------------------------------
# Official Qwen prompt on merged_sorted (adapter only — utils.qwen_doc_to_messages untouched)
# ---------------------------------------------------------------------------

def _action_history_to_official_history(doc: Dict) -> List[Dict[str, Any]]:
    """Map merged_sorted action_history → official qwen_doc_to_messages history entries."""
    img_w = int(doc.get("image_width") or 540)
    img_h = int(doc.get("image_height") or 1200)
    history: List[Dict[str, Any]] = []
    for ah in sorted(doc.get("action_history") or [], key=lambda x: x.get("step_index", 0)):
        step_idx = int(ah.get("step_index", 0))
        action = parse_action_history_text(ah.get("action_text", ""), img_w=img_w, img_h=img_h)
        if action is None:
            action = {"action_type": -1}
        entry: Dict[str, Any] = {
            "step_index": step_idx,
            "action_json": json.dumps(action, ensure_ascii=False),
        }
        image_path = ah.get("image_path")
        if image_path:
            entry["image_path"] = image_path
        history.append(entry)
    return history


def _image_only_doc(doc: Dict) -> Dict:
    """Drop video/audio fields so both prompt arms are screenshot-only."""
    out = dict(doc)
    for key in (
        "video_path",
        "audio_path",
        "current_video",
        "current_audio",
        "video",
        "audio",
    ):
        out.pop(key, None)
    history = []
    for hist in out.get("history") or []:
        h = dict(hist)
        h.pop("video_path", None)
        h.pop("audio_path", None)
        history.append(h)
    if history:
        out["history"] = history
    return out


def qwen_official_merged_doc_to_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """Official OmniGUI Qwen prompt (utils.qwen_doc_to_messages) on merged_sorted rows.

    Only shapes doc fields; upstream qwen_doc_to_messages / QWEN_SYSTEM_PROMPT are unmodified.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    adapted = dict(doc)
    adapted["history"] = _action_history_to_official_history(doc)
    if lmms_eval_specific_kwargs.get("image_only", True):
        adapted = _image_only_doc(adapted)

    return omni_utils.qwen_doc_to_messages(adapted, lmms_eval_specific_kwargs)


def gui_owl_dialogue_suffix_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """Incremental dialogue KV suffix for GUI-Owl native_mt4 (same doc_to_messages source)."""
    from lmms_eval.tasks.eval_message_registry import dialogue_incremental_suffix_messages

    return dialogue_incremental_suffix_messages(
        doc,
        lmms_eval_specific_kwargs,
        _gui_owl_native_multiturn_impl,
    )


def gui_owl_dialogue_suffix_allowed(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> bool:
    """Whether step>0 may use incremental dialogue KV with GUI-Owl native_mt4 messages."""
    from lmms_eval.tasks.eval_message_registry import dialogue_incremental_suffix_allowed

    return dialogue_incremental_suffix_allowed(
        doc,
        lmms_eval_specific_kwargs,
        _gui_owl_native_multiturn_impl,
    )
