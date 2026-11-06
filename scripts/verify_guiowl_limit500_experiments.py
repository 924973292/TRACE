#!/usr/bin/env python3
"""Verify GUI-Owl limit-500 runs: imageonly_a (no history) and hist_k5 (FIFO K=5).

Checks:
- samples JSONL exists and step count matches --limit
- overall TM/EM/SR/GP from JSONL agree with *_results.json (within tolerance)
- per-dimension breakdown is non-degenerate (not all zeros when steps > 0)
- optional: regenerate compact summary block

Usage (from repo root on remote):
  python scripts/verify_guiowl_limit500_experiments.py
  python scripts/verify_guiowl_limit500_experiments.py --append-summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (REPO, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from omnigui_data_utils import preferred_omnigui_data_file

DIM_ORDER = [
    ("Overall", None),
    ("Loc.", "localization"),
    ("Sem.", "semantic_understanding"),
    ("CrM.", "cross_modal_discrimination"),
    ("Tmp.", "temporal_reasoning"),
    ("Inst.", "instant_response"),
]
CATEGORY_ALIASES = {
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

EXPERIMENTS = [
    {
        "name": "imageonly_a",
        "task": "task_v2_guiowl_imageonly_a",
        "output_root": "outputs/guiowl_imageonly_a_limit500",
        "expect_history_images": 0,
    },
    {
        "name": "hist_k5",
        "task": "task_v2_guiowl_imageonly_a_hist_k5",
        "output_root": "outputs/guiowl_imageonly_a_hist_k5_limit500",
        "expect_history_images_min": 1,
    },
    {
        "name": "native_mt4",
        "task": "task_v2_guiowl_native_mt4",
        "output_root": "outputs/guiowl_native_mt4_limit500_baseline",
        "expect_history_images_min": 1,
    },
]


def normalize_category(value: Any) -> str:
    raw = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(raw, raw)


def latest_file(root: Path, pattern: str) -> Path | None:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def find_run_dir(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("models__*/**/"), key=lambda p: p.stat().st_mtime)
    # Prefer leaf dirs that contain samples or results.
    leafs = [p for p in candidates if list(p.glob("*_samples_*.jsonl")) or list(p.glob("*_results.json"))]
    if leafs:
        return leafs[-1]
    return candidates[-1] if candidates else None


def extract_eval_fields(sample: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
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
            doc_id = int(sample.get("doc_id", 0))
            metric = {
                "type_match": bool(type_match) if isinstance(type_match, bool) else False,
                "exact_match": bool(exact_match) if isinstance(exact_match, bool) else False,
                "subset": doc.get("subset", "unknown"),
                "episode_id": doc.get("episode_id", "unknown"),
                "step_id": doc.get("step_index", doc.get("step_id", doc_id)),
            }
    return {
        "subset": metric.get("subset", doc.get("subset", "unknown")),
        "episode_id": metric.get("episode_id", doc.get("episode_id", "unknown")),
        "step_id": metric.get("step_id", doc.get("step_index", doc.get("step_id", sample.get("doc_id", 0)))),
        "type_match": bool(metric.get("type_match", False)),
        "exact_match": bool(metric.get("exact_match", False)),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"steps": 0, "episodes": 0, "tm": None, "em": None, "sr": None, "gp": None}
    episodes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episodes[(str(record["subset"]), str(record["episode_id"]))].append(record)
    successes = []
    progresses = []
    for ep_records in episodes.values():
        ep_records.sort(key=lambda x: int(x.get("step_id", 0) or 0))
        successes.append(all(bool(x.get("exact_match")) for x in ep_records))
        prefix = 0
        for item in ep_records:
            if bool(item.get("exact_match")):
                prefix += 1
            else:
                break
        progresses.append(prefix / len(ep_records) if ep_records else 0.0)
    return {
        "steps": len(records),
        "episodes": len(episodes),
        "tm": sum(1.0 if r.get("type_match") else 0.0 for r in records) / len(records),
        "em": sum(1.0 if r.get("exact_match") else 0.0 for r in records) / len(records),
        "sr": sum(1.0 if s else 0.0 for s in successes) / len(successes) if successes else None,
        "gp": sum(progresses) / len(progresses) if progresses else None,
    }


def load_samples_records(samples_path: Path, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = int(sample.get("doc_id", -1))
            doc = docs[doc_id] if 0 <= doc_id < len(docs) else {}
            fields = extract_eval_fields(sample, doc)
            category = normalize_category(doc.get("task_category_full") or doc.get("task_category"))
            records.append({"category": category, **fields})
    return records


def detect_task_name(run_dir: Path, fallback: str) -> str:
    sample_path = latest_file(run_dir, "*_samples_*.jsonl")
    if sample_path:
        m = re.search(r"_samples_(.+)\.jsonl$", sample_path.name)
        if m:
            return m.group(1)
    result_path = latest_file(run_dir, "*_results.json")
    if result_path:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            keys = list((payload.get("results") or {}).keys())
            if keys:
                return str(keys[0])
        except Exception:
            pass
    return fallback


def load_results_metrics(run_dir: Path, task_name: str) -> dict[str, float | None]:
    result_path = latest_file(run_dir, "*_results.json")
    out = {"TM": None, "EM": None, "SR": None, "GP": None}
    if not result_path:
        return out
    rec = json.loads(result_path.read_text(encoding="utf-8")).get("results", {}).get(task_name, {})
    aliases = {"TM": "type_match", "EM": "exact_match", "SR": "success_rate", "GP": "goal_progress"}
    for metric, prefix in aliases.items():
        for key, value in rec.items():
            if "stderr" in str(key).lower():
                continue
            if str(key).lower().startswith(prefix):
                try:
                    out[metric] = float(value)
                except Exception:
                    pass
                break
    return out


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.2f}%"


def close(a: float | None, b: float | None, tol: float = 0.02) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def verify_experiment(
    repo: Path,
    spec: dict[str, Any],
    docs: list[dict[str, Any]],
    *,
    limit: int,
    tol: float,
) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    name = spec["name"]
    output_root = repo / spec["output_root"]
    run_dir = find_run_dir(output_root)
    lines.append(f"=== {name} ({spec['task']}) ===")
    if run_dir is None:
        lines.append(f"FAIL: no run dir under {output_root}")
        return False, lines

    samples_path = latest_file(run_dir, "*_samples_*.jsonl")
    if samples_path is None:
        lines.append(f"FAIL: missing samples jsonl in {run_dir}")
        return False, lines

    task_name = detect_task_name(run_dir, spec["task"])
    records = load_samples_records(samples_path, docs)
    overall = aggregate(records)
    results = load_results_metrics(run_dir, task_name)

    lines.append(f"run_dir={run_dir.relative_to(repo)}")
    lines.append(f"samples={samples_path.name} steps={overall['steps']} episodes={overall['episodes']}")
    lines.append(
        f"JSONL  TM={pct(overall['tm'])} EM={pct(overall['em'])} SR={pct(overall['sr'])} GP={pct(overall['gp'])}"
    )
    lines.append(
        f"RESULT TM={pct(results['TM'])} EM={pct(results['EM'])} SR={pct(results['SR'])} GP={pct(results['GP'])}"
    )

    if overall["steps"] != limit:
        lines.append(f"WARN: expected {limit} steps, got {overall['steps']}")
    if not close(overall["tm"], results["TM"], tol):
        lines.append(f"FAIL: TM mismatch (tol={tol})")
        ok = False
    if not close(overall["em"], results["EM"], tol):
        lines.append(f"FAIL: EM mismatch (tol={tol})")
        ok = False
    if not close(overall["sr"], results["SR"], tol):
        lines.append(f"FAIL: SR mismatch (tol={tol})")
        ok = False
    if not close(overall["gp"], results["GP"], tol):
        lines.append(f"FAIL: GP mismatch (tol={tol})")
        ok = False

    lines.append("DIMENSION\tsteps\tTM%\tEM%\tSR%\tGP%")
    dim_groups: dict[str, dict[str, Any]] = {"Overall": overall}
    for short, category in DIM_ORDER[1:]:
        sub = [r for r in records if r.get("category") == category]
        dim_groups[short] = aggregate(sub)

    for short, _ in DIM_ORDER:
        d = dim_groups[short]
        lines.append(
            f"{short}\t{d['steps']}\t{pct(d['tm'])}\t{pct(d['em'])}\t{pct(d['sr'])}\t{pct(d['gp'])}"
        )
        if short != "Overall" and d["steps"] > 0 and (d["tm"] or 0) == 0 and (d["em"] or 0) == 0:
            lines.append(f"FAIL: {short} has {d['steps']} steps but TM/EM are 0 (dimension parser bug?)")
            ok = False

    if overall["steps"] > 0 and (overall["tm"] or 0) == 0 and (overall["em"] or 0) == 0:
        lines.append("FAIL: overall TM/EM are zero with non-empty samples")
        ok = False

    lines.append("PASS" if ok else "FAIL")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify GUI-Owl imageonly_a and hist_k5 limit-500 runs.")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--tol", type=float, default=0.02, help="Max abs diff for TM/EM/SR/GP JSONL vs results")
    ap.add_argument("--append-summary", action="store_true", help="Append fixed compact summary to each run.log")
    args = ap.parse_args()

    repo = args.repo.resolve()
    data_path = preferred_omnigui_data_file(repo)
    if data_path is None:
        print(f"FAIL: OmniGUI data file not found under {repo / 'data'}")
        return 1
    docs = json.loads(data_path.read_text(encoding="utf-8"))
    print(f"data={data_path.relative_to(repo)} docs={len(docs)}")

    all_ok = True
    for spec in EXPERIMENTS:
        ok, lines = verify_experiment(repo, spec, docs, limit=args.limit, tol=args.tol)
        all_ok = all_ok and ok
        print("\n".join(lines))
        print()

        if args.append_summary and ok:
            run_dir = find_run_dir(repo / spec["output_root"])
            if run_dir is not None:
                import append_omnigui_log_summary as append_summary

                summary = append_summary.build_summary(run_dir)
                log_path = run_dir / "run.log"
                text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
                text = re.sub(
                    r"\n?OMNIGUI_COMPACT_SUMMARY_V2\n.*?END_OMNIGUI_COMPACT_SUMMARY_V2\n?",
                    "\n",
                    text,
                    flags=re.S,
                )
                log_path.write_text(text.rstrip() + "\n" + summary, encoding="utf-8")
                print(f"appended summary -> {log_path.relative_to(repo)}")

    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
