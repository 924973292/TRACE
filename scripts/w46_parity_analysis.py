#!/usr/bin/env python3
"""Detector-parity attribution report.

Question. TRACE consumes a task-agnostic UI detector that no published baseline
uses, and removing that prior costs more (23.74 on ScreenSpot-v2) than TRACE's
margin over the strongest baseline (14.94). The open question is whether
the margin measures the detector rather than the selection algorithm. Cost
accounting does not answer this; only giving the SAME detector to the baselines
does.

Design. Three readings are combined:
  (a) baseline vs baseline+prior at the same budget -- how much of the margin
      does the prior transfer?
  (b) an alpha sweep over the prior strength, because the parity arms build
      m = 1 + alpha*P with P max-normalised, while TRACE's own mass is
      m = 1 + alpha*n*p with p a probability measure: at equal alpha the
      baselines receive a ~n times weaker prior. We report each parity arm at
      its OWN best alpha, i.e. the reading least favourable to us.
  (c) the prior-only arm (no diversity, no instruction term), which bounds what
      the detector can do with no selection algorithm at all.

Usage (remote): python3 scripts/w46_parity_analysis.py
Writes docs/w46_parity_report.json and prints a summary table.
"""
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

SSV2 = "screenspot_v2_center_acc"

# Anchors already harvested and reported in the paper / docs.
ANCHORS = {
    "trace_r10": 73.90,
    "trace_r05": 55.19,
    "divprune_r10": 51.97,
    "divprune_r05": 25.71,
    "vispruner_r10": 51.73,
    "vispruner_r05": 20.68,
    "prunesid_r10": 58.96,             # strongest published baseline
    "prunesid_r05": 34.20,
    "cdpruner_r10": 33.18,             # tables/main_single.tex L22
    "cdpruner_r05": 15.88,             # tables/main_single.tex L35
    "strongest_baseline_r10": 58.96,   # PruneSID
    "strongest_baseline_r05": 34.20,   # PruneSID
    "prior_only_r10": 32.63,           # aonly: prior top-k, no diversity
    "prior_only_r05": 17.06,
    # alpha=2 cells already harvested under logs/xds8b_abl2 (DivPrune/VisPruner
    # only); CDPruner/PruneSID alpha=2 is part of w46_parity_strong.tsv and is
    # therefore read from the sweep dirs like every other cell.
    "hybrid_alpha2_divprune_r10": 54.95,
    "hybrid_alpha2_divprune_r05": 29.72,
    "hybrid_alpha2_vispruner_r10": 49.29,
    "hybrid_alpha2_vispruner_r05": 23.27,
}

# All four selectors that define a hybrid rule (see HYBRID_CAPABLE_SELECTORS in
# lmms_eval/models/geopact_pub_baselines.py). Ordered strongest-first, because
# the decisive parity reading is the one against the arm that
# comes closest to us.
SELECTORS = ("prunesid", "cdpruner", "divprune", "vispruner")

SWEEP_DIRS = "logs/w46_parity/screenspot_v2"


def read_metric(run_dir, metric=SSV2):
    cands = sorted(glob.glob(os.path.join(run_dir, "results", "**", "*results.json"),
                             recursive=True))
    if not cands:
        return None
    try:
        blob = json.load(open(cands[-1], encoding="utf-8"))
    except Exception:
        return None
    for task, vals in (blob.get("results") or {}).items():
        if not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            if k.split(",")[0] == metric and isinstance(v, (int, float)):
                return round(float(v) * (100.0 if v <= 1.0 else 1.0), 2)
    return None


def sweep_cells():
    """Run directories only — the queue drops sibling *.wrapper.log files next to
    them and counting those as pending cells makes the report lie."""
    return [d for d in sorted(glob.glob(os.path.join(SWEEP_DIRS, "hyb_*_a*")))
            if os.path.isdir(d)]


def collect_sweep():
    """arm -> {alpha: score} for every completed alpha-sweep cell."""
    out = {}
    for d in sweep_cells():
        m = re.match(r"hyb_(\w+?)_(r\d+)_a(\d+)$", os.path.basename(d))
        if not m:
            continue
        sel, budget, alpha = m.group(1), m.group(2), int(m.group(3))
        score = read_metric(d)
        if score is None:
            continue
        out.setdefault(f"{sel}_{budget}", {})[alpha] = score
    return out


def main():
    sweep = collect_sweep()
    rows = []
    for sel in SELECTORS:
        for budget in ("r10", "r05"):
            base = ANCHORS[f"{sel}_{budget}"]
            trace = ANCHORS[f"trace_{budget}"]
            curve = dict(sweep.get(f"{sel}_{budget}", {}))
            legacy = ANCHORS.get(f"hybrid_alpha2_{sel}_{budget}")
            if legacy is not None:
                curve.setdefault(2, legacy)
            if not curve:
                continue                      # cell still running
            best_alpha = max(curve, key=lambda a: curve[a])
            best = curve[best_alpha]
            gap_before = round(trace - base, 2)
            gap_after = round(trace - best, 2)
            rows.append({
                "selector": sel, "budget": budget,
                "baseline": base, "alpha_curve": {str(k): v for k, v in sorted(curve.items())},
                "best_alpha": best_alpha, "baseline_plus_prior": best,
                "prior_transfer": round(best - base, 2),
                "trace": trace,
                "gap_to_trace_before": gap_before,
                "gap_to_trace_after": gap_after,
                "share_of_margin_explained_by_prior": (
                    round(100.0 * (best - base) / gap_before, 1) if gap_before > 0 else None
                ),
            })

    report = {
        "question": ("Is TRACE's single-step margin attributable to its external "
                     "UI detector rather than to its selection algorithm?"),
        "protocol": ("Baselines receive the identical OmniParser density field as a "
                     "multiplicative mass (GEOPACT_PUB_HYBRID_MASS=1), grafted onto "
                     "each selector's OWN score channel and nowhere else. VisPruner: "
                     "mass x attention saliency. DivPrune: mass x the greedy max-min "
                     "utility at every iteration. CDPruner: mass x the DPP relevance, "
                     "renormalised to max 1 so the relevance/diversity balance keeps "
                     "its published scale. PruneSID: mass x the PCA loading "
                     "magnitudes, leaving grouping, adaptive NMS threshold and quotas "
                     "untouched. Empty detections reduce to mass == 1, i.e. "
                     "bit-identical to the published rule (unit-tested in "
                     "scripts/_w46_test_hybrid_ext.py). Each parity arm is reported "
                     "at its best alpha over {2,8,16}."),
        "coverage_note": (
            "Parity now reaches the strongest ScreenSpot-v2 baseline (PruneSID, "
            "58.96) and both instruction-conditioned selectors. VisionTrim, the "
            "strongest baseline by retained average at rho=10%, is not covered: its "
            "DVTS score fuses real ViT attention with an LTAM locality term, so a "
            "per-token mass has no single well-defined insertion point and any choice "
            "would be our construction rather than theirs."
        ),
        "prior_only_floor": {
            "r10": ANCHORS["prior_only_r10"], "r05": ANCHORS["prior_only_r05"],
            "strongest_baseline_r10": ANCHORS["strongest_baseline_r10"],
            "strongest_baseline_r05": ANCHORS["strongest_baseline_r05"],
            "reading": ("The prior alone, selected top-k with no diversity term, "
                        "scores far below the strongest saliency baseline, so the "
                        "detector is not sufficient for the observed accuracy."),
        },
        "alpha_asymmetry_note": (
            "Parity arms use m = 1 + alpha*P with P max-normalised to [0,1] "
            "(geopact_vl_hook.py:2383); TRACE uses m = 1 + alpha*n*p with p a "
            "probability measure (geopact_qd_selector.py:1006). The sweep exists "
            "to remove this confound rather than to tune the baselines down."
        ),
        "rows": rows,
        "cells_pending": [os.path.basename(d) for d in sweep_cells()
                          if read_metric(d) is None],
    }
    os.makedirs("docs", exist_ok=True)
    json.dump(report, open("docs/w46_parity_report.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"{'selector':<11}{'bud':<5}{'base':>7}{'+prior':>8}{'a*':>4}"
          f"{'transf':>8}{'TRACE':>8}{'gap':>8}{'expl%':>7}")
    for r in rows:
        pct = r["share_of_margin_explained_by_prior"]
        print(f"{r['selector']:<11}{r['budget']:<5}{r['baseline']:>7}"
              f"{r['baseline_plus_prior']:>8}{r['best_alpha']:>4}"
              f"{r['prior_transfer']:>+8}{r['trace']:>8}"
              f"{r['gap_to_trace_after']:>8}{'' if pct is None else pct:>7}")
    print(f"\nprior-only floor: r10={ANCHORS['prior_only_r10']} "
          f"r05={ANCHORS['prior_only_r05']}  vs strongest baseline "
          f"{ANCHORS['strongest_baseline_r10']}/{ANCHORS['strongest_baseline_r05']}")
    if report["cells_pending"]:
        print("pending cells:", ", ".join(report["cells_pending"]))
    print("wrote docs/w46_parity_report.json")


if __name__ == "__main__":
    main()
    sys.exit(0)
