#!/usr/bin/env python3
"""Static checks for the public TRACE extract (no GPU)."""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "lmms_eval" / "tasks"
FAIL = 0


def fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print("FAIL:", msg)


def ok(msg: str) -> None:
    print("OK  :", msg)


def job_tasks() -> set[str]:
    tasks: set[str] = set()
    jobs = ROOT / "configs" / "jobs"
    for path in list(jobs.glob("*.tsv")) + list((jobs / "raw").glob("*.tsv")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                tasks.add(parts[1].strip())
    return tasks


def yaml_tasks() -> set[str]:
    found: set[str] = set()
    for path in TASK_DIR.rglob("*.yaml"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines()[:8]:
            if line.startswith("task:"):
                found.add(line.split(":", 1)[1].strip())
                break
    return found


def main() -> int:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    required = [
        ROOT / "lmms_eval" / "tasks" / "gui_owl_cross" / "task_mind2web_qwen3_omni_task.yaml",
        ROOT / "scripts" / "build_mind2web_merged.py",
        ROOT / "scripts" / "demo.sh",
        ROOT / "scripts" / "setup_eval_env.sh",
        ROOT / "scripts" / "harvest_paper_metrics.py",
        ROOT / "requirements.txt",
        ROOT / "environment.yml",
        ROOT / "configs" / "experiments" / "adopted_8b.json",
        ROOT / "artifacts" / "metrics" / "w45_uitars_results.json",
        ROOT / "artifacts" / "metrics" / "w47_uitars_cal_ledger.json",
        ROOT / "artifacts" / "metrics" / "xds2b_tabledata.txt",
        ROOT / "docs" / "reproduction.md",
    ]
    for p in required:
        (ok if p.is_file() else fail)(f"exists {p.relative_to(ROOT)}")

    yaml = yaml_tasks()
    missing = sorted(t for t in job_tasks() if t and t not in yaml)
    paper = {
        "task_screenspot_v2_guiowl",
        "task_screenspot_pro_guiowl",
        "task_mmbench_gui_l2_guiowl",
        "task_mmbench_gui_l1_guiowl",
        "task_v2_qwen3_omni_novideo",
        "task_mind2web_qwen3_omni_task",
        "task_mind2web_qwen3_omni_website",
        "task_mind2web_qwen3_omni_domain",
        "task_android_control_qwen3_omni",
    }
    paper_missing = sorted(t for t in paper if t not in yaml)
    if paper_missing:
        fail(f"paper task YAML missing: {paper_missing}")
    else:
        ok(f"paper task YAMLs present ({len(paper)})")
    extra_missing = [t for t in missing if t not in paper]
    if extra_missing:
        print("WARN: job TSV tasks with no YAML (non-paper or historical):")
        for t in extra_missing[:30]:
            print("     ", t)
        if len(extra_missing) > 30:
            print(f"      ... {len(extra_missing) - 30} more")
    else:
        ok("all job TSV tasks have YAML")

    ledger = json.loads((ROOT / "configs" / "experiments" / "adopted_8b.json").read_text(encoding="utf-8"))
    ok(f"adopted_8b cells={len(ledger.get('cells') or [])}")

    try:
        import lmms_eval  # noqa: F401
        from lmms_eval.models.geopact_nap import nap_anticipate_enabled, nap_contract_mode, ratio_current
        from lmms_eval.models.geopact_crs import crs_rep_mode

        ok(
            "import geopact defaults "
            f"contract={nap_contract_mode()} anticipate={nap_anticipate_enabled()} "
            f"ratio={ratio_current()} crs_rep={crs_rep_mode()}"
        )
    except Exception as exc:
        fail(f"import TRACE modules: {exc}")

    for rel in (
        "scripts/build_mind2web_merged.py",
        "scripts/harvest_paper_metrics.py",
        "scripts/gen_tables.py",
        "scripts/_w33_gen_2b_tables.py",
        "scripts/_w53_harvest.py",
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        try:
            ast.parse(src)
            ok(f"syntax {rel}")
        except SyntaxError as exc:
            fail(f"syntax {rel}: {exc}")

    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
