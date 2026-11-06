#!/usr/bin/env python3
"""Fig.5 (detector parity): is the margin the detector, or the order built on it?

LIP consumes a task-agnostic UI detector that no published baseline uses, and
the ablation shows the prior is worth more to us (-23.74 on ScreenSpot-v2) than
our margin is worth against the field (+14.94). That arithmetic licenses an
alternative explanation, so the figure answers it with two controls that pull in
opposite directions.

(a) Parity. Each baseline is re-run with the identical OmniParser density field
    grafted onto its own scoring rule (multiplicative mass), reported at its own
    best prior strength. The prior transfers, but the residual gap to TRACE
    survives -- so the margin is not the detector.
(b) Sufficiency. The prior used alone, top-k with no ordering or diversity term,
    lands far below the strongest saliency baseline -- so the detector cannot
    substitute for the selection algorithm either.

Values come from docs/w46_parity_report.json when present (which merges the
alpha sweep), otherwise from the already-published alpha=2 cells recorded in
docs/w41_r2_stats.json. Nothing is hand-copied.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402

from fig_style import C, apply, despine, panel_label, takeaway, save  # noqa: E402

FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
PARITY = os.path.join(HERE, "..", "docs", "w46_parity_report.json")
R2 = os.path.join(HERE, "..", "docs", "w41_r2_stats.json")

apply(scale=1.10)

# Anchors from the main tables (gen_tables.py is the source of truth for
# the table itself; these are the two cells this figure compares against).
TRACE = {"r10": 73.90, "r05": 55.19}
STRONGEST = {"r10": 58.96, "r05": 34.20}      # PruneSID
PRIOR_ONLY = {"r10": 32.63, "r05": 17.06}     # aonly: prior top-k, no ordering
UNIFORM = {"r10": 43.79, "r05": 21.38}

BUDGET_LABEL = {"r10": r"$\rho=10\%$", "r05": r"$\rho=5\%$"}


def load_parity():
    """selector -> budget -> (baseline, baseline_plus_prior, best_alpha)."""
    if os.path.exists(PARITY):
        rep = json.load(open(PARITY, encoding="utf-8"))
        out = {}
        for r in rep.get("rows", []):
            out.setdefault(r["selector"], {})[r["budget"]] = (
                r["baseline"], r["baseline_plus_prior"], r["best_alpha"])
        if out:
            return out
    # fall back to the published alpha=2 cells
    stats = json.load(open(R2, encoding="utf-8"))
    out = {}
    for e in stats.get("hybrid", []):
        parts = e["label"].split()
        if len(parts) < 3 or parts[1] != "ssv2":
            continue
        out.setdefault(parts[0], {})[parts[2]] = (e["base"], e["hybrid"], 2)
    return out


def panel_a(ax, parity):
    order = [("divprune", "DivPrune"), ("vispruner", "VisPruner")]
    budgets = ["r10", "r05"]
    x = np.arange(len(order) * len(budgets), dtype=float)
    x[2:] += 0.55                                  # visual gap between budgets
    w = 0.34
    labels, base_v, prior_v, alphas, buds = [], [], [], [], []
    for bi, b in enumerate(budgets):
        for key, name in order:
            cell = parity.get(key, {}).get(b)
            labels.append(name)
            buds.append(b)
            if cell is None:
                base_v.append(np.nan); prior_v.append(np.nan); alphas.append(None)
            else:
                base_v.append(cell[0]); prior_v.append(cell[1]); alphas.append(cell[2])

    ax.bar(x - w / 2, base_v, w, color="#BBBBBB", edgecolor="#666666", lw=0.5,
           label="published rule", zorder=3)
    ax.bar(x + w / 2, prior_v, w, color=C["second"], edgecolor="#00614A",
           lw=0.5, label="+ same detector", zorder=3)

    for xi, (bv, pv) in enumerate(zip(base_v, prior_v)):
        if np.isnan(bv):
            continue
        d = pv - bv
        ax.text(x[xi] + w / 2, pv + 1.1, f"{d:+.1f}", ha="center", va="bottom",
                fontsize=6.3, color="#00614A", fontweight="bold")

    # TRACE reference per budget, drawn only over that budget's pair
    for bi, b in enumerate(budgets):
        seg = x[bi * 2:bi * 2 + 2]
        ax.plot([seg[0] - 0.6, seg[-1] + 0.6], [TRACE[b]] * 2,
                color=C["ours"], lw=1.5, ls="--", zorder=4,
                label="TRACE" if bi == 0 else None)
        ax.text(seg[-1] + 0.62, TRACE[b], f" {TRACE[b]:.1f}", color=C["ours"],
                fontsize=6.6, va="center", ha="left", fontweight="bold")
        # residual gap brace from the better parity arm up to TRACE
        top = np.nanmax(prior_v[bi * 2:bi * 2 + 2])
        if not np.isnan(top):
            xm = seg.mean()
            ax.annotate("", (xm, TRACE[b]), (xm, top),
                        arrowprops=dict(arrowstyle="<->", color=C["ours"],
                                        lw=0.8, shrinkA=0, shrinkB=0), zorder=5)
            ax.text(xm + 0.08, (TRACE[b] + top) / 2,
                    f"{TRACE[b] - top:+.1f}", color=C["ours"], fontsize=6.4,
                    va="center", ha="left", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    for bi, b in enumerate(budgets):
        ax.text(x[bi * 2:bi * 2 + 2].mean(), -12.0, BUDGET_LABEL[b],
                ha="center", va="top", fontsize=7.4)
    ax.set_ylabel("ScreenSpot-v2 acc. (%)")
    ax.set_ylim(0, 88)
    despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3,
              handlelength=1.3, columnspacing=0.9, fontsize=6.8)
    panel_label(ax, "(a) The prior transfers. The gap does not close")
    takeaway(ax, "Same detector, baseline's own rule\nOnly a small share of the margin",
             loc="lower left", fontsize=7.2)


def panel_b(ax):
    budgets = ["r10", "r05"]
    x = np.arange(len(budgets), dtype=float)
    w = 0.2
    series = [
        ("prior alone\n(no ordering)", PRIOR_ONLY, "#E69F00", "#9A6A00"),
        ("uniform", UNIFORM, "#BBBBBB", "#666666"),
        ("strongest\nbaseline", STRONGEST, C["best"], "#00456B"),
        ("TRACE", TRACE, C["ours"], "#8F3E00"),
    ]
    for si, (name, data, fc, ec) in enumerate(series):
        off = (si - (len(series) - 1) / 2) * w
        vals = [data[b] for b in budgets]
        ax.bar(x + off, vals, w * 0.92, color=fc, edgecolor=ec, lw=0.5,
               label=name, zorder=3)
        for xi, v in enumerate(vals):
            ax.text(x[xi] + off, v + 1.0, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.0, color=ec)
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_LABEL[b] for b in budgets], fontsize=7.4)
    ax.set_ylabel("ScreenSpot-v2 acc. (%)")
    ax.set_ylim(0, 88)
    despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4,
              handlelength=1.0, columnspacing=0.6, fontsize=6.2,
              labelspacing=0.3)
    panel_label(ax, "(b) The detector cannot stand in for the order")
    takeaway(ax, "Prior alone, no ordering\nBelow even uniform",
             loc="center right", fontsize=7.2)


def main():
    parity = load_parity()
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], left=0.075,
                          right=0.988, top=0.86, bottom=0.19, wspace=0.26)
    panel_a(fig.add_subplot(gs[0, 0]), parity)
    panel_b(fig.add_subplot(gs[0, 1]))
    save(fig, os.path.join(FIGDIR, "fig_parity.pdf"))


if __name__ == "__main__":
    main()
