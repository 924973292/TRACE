import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

import hashlib
from pathlib import Path
from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        logging.warning(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _is_light_log_enabled() -> bool:
    # Default to compact logging for all experiments; allow opt-out only if explicitly disabled.
    return str(os.environ.get("LMMS_LIGHT_LOG", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _get_metric_value(record: dict, aliases: list[str]):
    if not isinstance(record, dict):
        return None
    # Respect alias priority (e.g. prefill_ms before ttft_ms for "Prefill").
    for alias in aliases:
        alias_l = alias.lower()
        for key, value in record.items():
            k = str(key).lower()
            if "stderr" in k:
                continue
            if k == alias_l or k.startswith(alias_l + ","):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _fmt_metric(value):
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _fmt_num(value, unit: str = ""):
    if value is None:
        return "n/a"
    return f"{value:.2f}{unit}"


def _weighted_mean(values_with_weights):
    valid = [(v, w) for v, w in values_with_weights if v is not None and w is not None and w > 0]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    if total_w <= 0:
        return None
    return sum(v * w for v, w in valid) / total_w


def _aggregate_dim_records(records):
    if not records:
        return {"tm": None, "em": None, "sr": None, "gp": None}

    episodes = defaultdict(list)
    for record in records:
        ep = (record.get("subset", "unknown"), record.get("episode_id", "unknown"))
        episodes[ep].append(record)

    tm = sum(1.0 if r.get("type_match") else 0.0 for r in records) / len(records)
    em = sum(1.0 if r.get("exact_match") else 0.0 for r in records) / len(records)

    successes = []
    progresses = []
    for ep_records in episodes.values():
        ep_records.sort(key=lambda x: x.get("step_id", 0))
        successes.append(all(bool(x.get("exact_match")) for x in ep_records))
        prefix = 0
        for x in ep_records:
            if bool(x.get("exact_match")):
                prefix += 1
            else:
                break
        progresses.append(prefix / len(ep_records) if ep_records else 0.0)

    sr = (sum(1.0 if s else 0.0 for s in successes) / len(successes)) if successes else None
    gp = (sum(progresses) / len(progresses)) if progresses else None
    return {"tm": tm, "em": em, "sr": sr, "gp": gp}


_OMNIGUI_CATEGORY_ALIASES = {
    "localization": "localization",
    "loc": "localization",
    "loc.": "localization",
    "semantic": "semantic_understanding",
    "sem": "semantic_understanding",
    "sem.": "semantic_understanding",
    "semantic_understanding": "semantic_understanding",
    "cross_modal": "cross_modal_discrimination",
    "cross_modal_discrimination": "cross_modal_discrimination",
    "crm": "cross_modal_discrimination",
    "crm.": "cross_modal_discrimination",
    "cro": "cross_modal_discrimination",
    "temporal": "temporal_reasoning",
    "temporal_reasoning": "temporal_reasoning",
    "tmp": "temporal_reasoning",
    "tmp.": "temporal_reasoning",
    "tem": "temporal_reasoning",
    "instant": "instant_response",
    "instant_response": "instant_response",
    "inst": "instant_response",
    "inst.": "instant_response",
    "ins": "instant_response",
}


def _normalize_omnigui_category(value):
    raw = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return _OMNIGUI_CATEGORY_ALIASES.get(raw, raw)


def _extract_sample_eval_fields(sample: dict, doc: dict | None = None) -> dict:
    """Read per-step eval fields from logged samples (step_eval dict or legacy layouts)."""
    doc = doc if isinstance(doc, dict) else {}
    step_eval = sample.get("step_eval")
    if isinstance(step_eval, dict):
        metric = step_eval
    else:
        type_match = sample.get("type_match")
        exact_match = sample.get("exact_match")
        if isinstance(type_match, dict):
            metric = type_match
        elif isinstance(exact_match, dict):
            metric = exact_match
        else:
            metric = {
                "type_match": bool(type_match) if isinstance(type_match, bool) else False,
                "exact_match": bool(exact_match) if isinstance(exact_match, bool) else False,
                "subset": doc.get("subset", "unknown"),
                "episode_id": doc.get("episode_id", "unknown"),
                "step_id": doc.get("step_index", doc.get("step_id", sample.get("doc_id", 0))),
            }

    return {
        "subset": metric.get("subset", doc.get("subset", "unknown")),
        "episode_id": metric.get("episode_id", doc.get("episode_id", "unknown")),
        "step_id": metric.get("step_id", doc.get("step_index", doc.get("step_id", sample.get("doc_id", 0)))),
        "type_match": bool(metric.get("type_match", False)),
        "exact_match": bool(metric.get("exact_match", False)),
    }


def _load_omnigui_docs():
    repo = Path(__file__).resolve().parents[1]
    for rel in ("data/merged_sorted_media_valid.json", "data/merged_sorted.json"):
        path = repo / rel
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []
    return []


def _build_omnigui_dimension_matrix(samples) -> dict:
    # Canonical order and labels required by the user.
    dim_order = [
        ("Overall", None),
        ("Loc.", "localization"),
        ("Sem.", "semantic_understanding"),
        ("CrM.", "cross_modal_discrimination"),
        ("Tmp.", "temporal_reasoning"),
        ("Inst.", "instant_response"),
    ]

    docs = _load_omnigui_docs()
    all_records = []
    if isinstance(samples, dict):
        iter_lists = samples.values()
    elif isinstance(samples, list):
        iter_lists = [samples]
    else:
        iter_lists = []

    for task_samples in iter_lists:
        if not isinstance(task_samples, list):
            continue
        for sample in task_samples:
            if not isinstance(sample, dict):
                continue
            doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else {}
            if not doc:
                try:
                    doc_id = int(sample.get("doc_id", -1))
                except (TypeError, ValueError):
                    doc_id = -1
                if 0 <= doc_id < len(docs) and isinstance(docs[doc_id], dict):
                    doc = docs[doc_id]

            fields = _extract_sample_eval_fields(sample, doc)
            step_eval = sample.get("step_eval") if isinstance(sample.get("step_eval"), dict) else {}
            category = _normalize_omnigui_category(
                doc.get("task_category_full")
                or doc.get("task_category")
                or step_eval.get("task_category_full")
                or step_eval.get("task_category")
                or "unknown"
            )

            all_records.append({"category": category, **fields})

    by_dim = {}
    by_dim["Overall"] = _aggregate_dim_records(all_records)
    for short_name, key in dim_order[1:]:
        recs = [r for r in all_records if r.get("category") == key]
        by_dim[short_name] = _aggregate_dim_records(recs)

    return {"order": [x[0] for x in dim_order], "values": by_dim}


def _print_light_summary(results: dict, samples=None) -> None:
    if not isinstance(results, dict):
        return

    result_entries = results.get("results", {}) if isinstance(results.get("results"), dict) else {}
    if not result_entries:
        return

    metric_aliases = {
        "TM": ["type_match", "tm"],
        "EM": ["exact_match", "em"],
        "SR": ["success_rate", "sr"],
        "GP": ["goal_progress", "gp"],
    }

    step_latency_aliases = {
        "E2E": ["e2e_latency_ms", "e2e_step_ms", "e2e_latency", "total_step_ms"],
        "TTFT": ["ttft_ms", "ttft"],
        "VisionPrefill": ["vision_prefill_ms", "vlm_prefill_ms"],
        "Pruning": ["pruning_ms", "compact_ms"],
        "LLMPrefill": ["llm_prefill_ms", "llm_prefill_forward_ms", "prefill_ms"],
        "Decode": ["decode_latency_ms", "decode_total_ms", "generation_ms", "generate_ms"],
        "KVFinalize": ["kv_finalize_ms", "dialogue_kv_finalize_ms"],
        "Misc": ["misc_ms"],
        "InputTokens": ["input_tokens", "prompt_tokens"],
        "OutputTokens": ["output_tokens", "generated_tokens"],
        "PeakGPU_MB": ["peak_gpu_memory_mb"],
    }
    episode_latency_aliases = {
        "SumE2E": ["episode_sum_e2e_latency_ms"],
        "MeanStepE2E": ["episode_mean_step_e2e_latency_ms"],
        "SumVisionPrefill": ["episode_sum_vision_prefill_ms"],
        "SumPruning": ["episode_sum_pruning_ms"],
        "SumLLMPrefill": ["episode_sum_llm_prefill_ms"],
        "SumDecode": ["episode_sum_decode_latency_ms"],
        "SumGenerate": ["episode_sum_generation_ms"],
        "SumInputTok": ["episode_sum_input_tokens"],
        "MaxGPU": ["episode_max_gpu_memory_gb"],
        "MaxGPU_MB": ["episode_max_peak_gpu_memory_mb"],
    }

    nsamples = results.get("n-samples", {}) if isinstance(results.get("n-samples"), dict) else {}

    def _effective_n(task_name: str):
        rec = nsamples.get(task_name)
        if isinstance(rec, dict):
            try:
                return float(rec.get("effective", rec.get("original", 0)) or 0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    # OVERALL: weighted by effective sample count across all result entries.
    overall_values = {}
    for metric_name, aliases in metric_aliases.items():
        vals = []
        for task_name, task_record in result_entries.items():
            vals.append((_get_metric_value(task_record, aliases), _effective_n(task_name)))
        overall_values[metric_name] = _weighted_mean(vals)

    overall_acc = " ".join(
        f"{name}={_fmt_metric(overall_values.get(name))}"
        for name in ["TM", "EM", "SR", "GP"]
    )
    print(f"OVERALL ALL {overall_acc}")

    # Use the most representative task (largest effective sample count) for latency summary.
    rep_name = max(result_entries.keys(), key=_effective_n)
    rep_record = result_entries[rep_name]

    def _format_latency_line(prefix: str, aliases_map: dict, *, ms_fields: set[str], gb_fields: set[str] | None = None, mb_fields: set[str] | None = None) -> str | None:
        gb_fields = gb_fields or set()
        mb_fields = mb_fields or set()
        parts = []
        has_value = False
        for name, aliases in aliases_map.items():
            value = _get_metric_value(rep_record, aliases)
            if value is None:
                parts.append(f"{name}=n/a")
                continue
            has_value = True
            if name in gb_fields:
                parts.append(f"{name}={_fmt_num(value, 'GB')}")
            elif name in mb_fields:
                parts.append(f"{name}={_fmt_num(value, 'MB')}")
            elif name in ms_fields:
                parts.append(f"{name}={_fmt_num(value, 'ms')}")
            else:
                parts.append(f"{name}={_fmt_num(value)}")
        return (prefix + " " + " ".join(parts)) if has_value else None

    step_line = _format_latency_line(
        "LATENCY_STEP",
        step_latency_aliases,
        ms_fields={"E2E", "TTFT", "VisionPrefill", "Pruning", "LLMPrefill", "Decode", "KVFinalize", "Misc"},
        mb_fields={"PeakGPU_MB"},
    )
    episode_line = _format_latency_line(
        "LATENCY_EPISODE",
        episode_latency_aliases,
        ms_fields={"SumE2E", "MeanStepE2E", "SumVisionPrefill", "SumPruning", "SumLLMPrefill", "SumDecode", "SumGenerate"},
        gb_fields={"MaxGPU"},
        mb_fields={"MaxGPU_MB"},
    )
    if not episode_line and samples:
        # Fallback: derive episode latency from logged per-step profile records.
        flat_samples: list[dict] = []
        if isinstance(samples, dict):
            for task_samples in samples.values():
                if isinstance(task_samples, list):
                    flat_samples.extend(task_samples)
        elif isinstance(samples, list):
            flat_samples = samples
        if flat_samples:
            try:
                import sys
                from pathlib import Path as _Path

                scripts_dir = _Path(__file__).resolve().parents[1] / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                from omnigui_metrics_common import compute_episode_latency_from_samples

                derived = compute_episode_latency_from_samples(flat_samples)
                derived_parts = []
                for name, aliases in episode_latency_aliases.items():
                    key = aliases[0]
                    value = derived.get(key)
                    if value is None:
                        derived_parts.append(f"{name}=n/a")
                    elif name == "MaxGPU":
                        derived_parts.append(f"{name}={_fmt_num(value, 'GB')}")
                    else:
                        derived_parts.append(f"{name}={_fmt_num(value, 'ms')}")
                if any("n/a" not in p.split("=", 1)[1] for p in derived_parts):
                    episode_line = "LATENCY_EPISODE " + " ".join(derived_parts)
            except Exception:
                pass
    if step_line:
        print(step_line)
        print("LATENCY " + step_line.split(" ", 1)[1])
        try:
            from lmms_eval.tasks.agentcpm_gui.performance_profile import latency_breakdown_percentages

            e2e_val = _get_metric_value(rep_record, step_latency_aliases["E2E"])
            if e2e_val and e2e_val > 0:
                parts = {}
                for label, aliases in (
                    ("VisionPrefill", step_latency_aliases["VisionPrefill"]),
                    ("Pruning", step_latency_aliases["Pruning"]),
                    ("LLMPrefill", step_latency_aliases["LLMPrefill"]),
                    ("Decode", step_latency_aliases["Decode"]),
                ):
                    val = _get_metric_value(rep_record, aliases) or 0.0
                    parts[label] = round(float(val) / float(e2e_val) * 100.0, 1)
                print(
                    "LATENCY_BREAKDOWN "
                    + " ".join(f"{k}={v}%" for k, v in parts.items())
                )
        except Exception:
            pass
    if episode_line:
        print(episode_line)
    elif step_line:
        print("LATENCY_EPISODE SumE2E=n/a MeanStepE2E=n/a SumGenerate=n/a SumInputTok=n/a MaxGPU=n/a")

    # SUBTASK: always compute and print all result entries.
    for name, rec in sorted(result_entries.items()):
        per_line = " ".join(
            f"{m}={_fmt_metric(_get_metric_value(rec, aliases))}"
            for m, aliases in metric_aliases.items()
        )
        print(f"SUBTASK {name} {per_line}")

    # GROUP: must compute and print. Prefer provided groups; otherwise compute from group_subtasks.
    group_entries = results.get("groups", {}) if isinstance(results.get("groups"), dict) else {}
    group_subtasks = results.get("group_subtasks", {}) if isinstance(results.get("group_subtasks"), dict) else {}

    printed_group = False
    for group_name in sorted(group_subtasks.keys()):
        subtasks = group_subtasks.get(group_name) or []
        if not isinstance(subtasks, list) or not subtasks:
            continue

        if group_name in group_entries and isinstance(group_entries[group_name], dict):
            group_record = group_entries[group_name]
            metrics = {
                m: _get_metric_value(group_record, aliases)
                for m, aliases in metric_aliases.items()
            }
        else:
            metrics = {}
            for m, aliases in metric_aliases.items():
                vals = []
                for st in subtasks:
                    if st not in result_entries:
                        continue
                    vals.append((_get_metric_value(result_entries[st], aliases), _effective_n(st)))
                metrics[m] = _weighted_mean(vals)

        per_line = " ".join(f"{m}={_fmt_metric(metrics.get(m))}" for m in ["TM", "EM", "SR", "GP"])
        print(f"GROUP {group_name} {per_line}")
        printed_group = True

    # Ensure group section is always present even when no group configs exist.
    if not printed_group:
        fallback = " ".join(
            f"{m}={_fmt_metric(overall_values.get(m))}" for m in ["TM", "EM", "SR", "GP"]
        )
        print(f"GROUP ALL {fallback}")

    # Mandatory OmniGUI 6-dimension matrix with fixed headers.
    dim_matrix = _build_omnigui_dimension_matrix(samples)
    dims = dim_matrix["order"]
    values = dim_matrix["values"]

    print("DIM_HEADER " + " ".join(dims))
    print("DIM_METRIC_HEADER " + " ".join(["TM EM SR GP"] * len(dims)))

    row_vals = []
    for d in dims:
        v = values.get(d, {})
        row_vals.extend(
            [
                _fmt_metric(v.get("tm")),
                _fmt_metric(v.get("em")),
                _fmt_metric(v.get("sr")),
                _fmt_metric(v.get("gp")),
            ]
        )
    print("DIM_VALUES " + " ".join(row_vals))


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--launcher_args",
        default=None,
        help="String arguments for launcher for local llm as judge, e.g. `tp=8`, if None then no launcher will be used.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument("--process_with_media", action="store_true", help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.")
    parser.add_argument("--force_simple", action="store_true", help="Force the evaluation to use the simple mode of the models")
    args = parser.parse_args()
    return args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    default_args = parse_eval_args()

    if args is None and len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ Use `lmms-eval --help` for more information.                                  │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    # If args were provided, override the defaults
    if args:
        for key, value in vars(args).items():
            setattr(default_args, key, value)

    args = default_args

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    # Configure logger with detailed format including file path, function name, and line number
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity, format=log_format)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    args_list = []
    results_list = []
    samples_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator only if not already in a distributed context
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        if accelerator.is_main_process:
            is_main_process = True
        else:
            is_main_process = False

    for args in args_list:
        try:
            # if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
            #     wandb_logger = WandbLogger()

            results, samples = cli_evaluate_single(args)
            results_list.append(results)
            samples_list.append(samples)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
                # wandb_logger.finish()

        except Exception as e:
            if args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)
                samples_list.append(None)

    for args, results, samples in zip(args_list, results_list, samples_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            _print_light_summary(results, samples)

    if args.wandb_args:
        wandb_logger.run.finish()


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lmms-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lmms-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend="torchrun" if (torch.distributed.is_available() and torch.distributed.is_initialized()) else "accelerate",
        force_simple=args.force_simple,
        launcher_args=args.launcher_args,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

            try:
                from lmms_eval.tasks.agentcpm_gui.performance_profile import export_performance_profiles

                export_path = args.output_path
                if export_path and not os.path.isdir(export_path):
                    export_path = os.path.dirname(export_path) or "."
                counts = export_performance_profiles(export_path, samples)
                eval_logger.info(f"Exported performance profiles: {counts}")
            except Exception as exc:
                eval_logger.warning(f"Performance profile export failed: {exc}")

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
