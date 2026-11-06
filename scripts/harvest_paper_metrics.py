#!/usr/bin/env python3
"""Harvest adopted TRACE paper cells from local eval dumps.

This is the dump -> metric path. It does **not** rewrite
``scripts/gen_tables.py`` literals. After a GPU rerun::

    python scripts/harvest_paper_metrics.py
    python scripts/harvest_paper_metrics.py --logs logs

Writes ``artifacts/metrics/harvested_main.json``. Compare harvested vs
``published`` there; then decide whether to update table literals.

AndroidControl Step SR uses ``scripts/_ac_hist_metrics.py`` when available.
Mind2Web pooled Step SR weights: task 2008 / domain 5429 / website 1343.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs" / "experiments" / "adopted_8b.json"
OUT_JSON = ROOT / "artifacts" / "metrics" / "harvested_main.json"
M2W_WEIGHTS = {"task": 2008.0, "domain": 5429.0, "website": 1343.0}


def _load_results_json(out_dir: Path) -> Optional[dict]:
    hits = sorted(out_dir.glob("results/**/*_results.json"))
    if not hits:
        hits = sorted(Path(p) for p in glob.glob(str(out_dir / "results" / "*" / "*_results.json")))
    if not hits:
        return None
    try:
        return json.loads(hits[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _flatten_metrics(blob: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for _task, m in (blob.get("results") or {}).items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[k] = float(v) * 100.0 if 0.0 <= float(v) <= 1.0 else float(v)
    return out


def _pick(metrics: dict[str, float], *needles: str) -> Optional[float]:
    lower = {k.lower(): v for k, v in metrics.items()}
    for n in needles:
        nl = n.lower()
        for k, v in lower.items():
            if nl in k and "stderr" not in k:
                return v
    return None


def _ac_hist(out_dir: Path) -> Optional[float]:
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import _ac_hist_metrics as ac  # type: ignore
    except Exception:
        return None
    parent = out_dir.parent
    try:
        if hasattr(ac, "main"):
            # Prefer a library-style helper if present.
            pass
    except Exception:
        return None
    # Fall back: parse printed protocol from a sidecar if the module exposes score().
    score_fn = getattr(ac, "score_run", None) or getattr(ac, "metrics_for_run", None)
    if callable(score_fn):
        try:
            rec = score_fn(str(out_dir))
            if isinstance(rec, dict):
                for key in ("step_sr", "StepSR", "exact_match"):
                    if key in rec:
                        v = float(rec[key])
                        return v * 100.0 if v <= 1.0 else v
        except Exception:
            return None
    _ = parent
    return None


def harvest_cell(cell: dict[str, Any], logs_root: Path) -> dict[str, Any]:
    rec = dict(cell)
    rel = str(cell.get("out") or "").replace("\\", "/")
    out_dir = logs_root / rel[5:] if rel.startswith("logs/") else ROOT / rel
    if rel.startswith("logs/"):
        out_dir = logs_root / rel[len("logs/") :]
    rec["resolved_out"] = str(out_dir)
    if not out_dir.is_dir():
        rec["status"] = "missing_dir"
        rec["harvested"] = None
        return rec
    blob = _load_results_json(out_dir)
    if blob is None:
        rec["status"] = "missing_results"
        rec["harvested"] = None
        return rec
    metrics = _flatten_metrics(blob)
    rec["raw_keys"] = sorted(metrics)
    kind = cell.get("metric") or "auto"
    got: Optional[float] = None
    if kind == "center_acc":
        got = _pick(metrics, "center_acc", "acc")
    elif kind == "exact_match":
        got = _pick(metrics, "exact_match")
    elif kind == "m2w_step_sr":
        got = _pick(metrics, "m2w_step_sr", "exact_match")
    elif kind == "ac_step_sr":
        hist = _ac_hist(out_dir)
        got = hist if hist is not None else _pick(metrics, "exact_match")
        rec["ac_source"] = "histprune" if hist is not None else "results_json"
    else:
        got = _pick(metrics, "center_acc", "m2w_step_sr", "exact_match", "acc")
    rec["harvested"] = None if got is None else round(float(got), 2)
    pub = cell.get("published")
    if rec["harvested"] is None:
        rec["status"] = "no_metric"
    elif pub is None:
        rec["status"] = "ok"
    elif abs(float(rec["harvested"]) - float(pub)) <= 0.6:
        rec["status"] = "match"
    else:
        rec["status"] = "drift"
    return rec


def pool_m2w(cells: list[dict]) -> Optional[dict]:
    parts = {c["id"]: c for c in cells if c.get("id", "").startswith("m2w_")}
    needed = {
        "m2w_tight_task": "task",
        "m2w_tight_website": "website",
        "m2w_tight_domain": "domain",
    }
    vals = []
    wsum = 0.0
    acc = 0.0
    for cid, split in needed.items():
        rec = parts.get(cid)
        if not rec or rec.get("harvested") is None:
            return None
        w = M2W_WEIGHTS[split]
        acc += float(rec["harvested"]) * w
        wsum += w
        vals.append(rec["harvested"])
    return {
        "id": "m2w_tight_pooled",
        "paper": "tab:main-multi Mind2Web tight pooled Step SR",
        "published": None,
        "harvested": round(acc / wsum, 2) if wsum else None,
        "parts": vals,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--logs", default=None, help="logs/ root (default: <repo>/logs)")
    parser.add_argument("--ledger", default=str(LEDGER))
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    logs_root = Path(args.logs).resolve() if args.logs else repo / "logs"
    ledger_path = Path(args.ledger)
    if not ledger_path.is_file():
        print(f"ERROR: missing ledger {ledger_path}", file=sys.stderr)
        return 1
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    harvested = [harvest_cell(c, logs_root) for c in ledger.get("cells") or []]
    pooled = pool_m2w(harvested)
    blob = {
        "note": (
            "Harvested metrics from local dumps. GUI-Owl table generator "
            "still uses frozen literals until you edit them."
        ),
        "logs_root": str(logs_root),
        "cells": harvested,
        "m2w_tight_pooled": pooled,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    for cell in harvested:
        print(
            f"  {cell.get('id'):24s} harvested={cell.get('harvested')} "
            f"published={cell.get('published')} [{cell.get('status')}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
