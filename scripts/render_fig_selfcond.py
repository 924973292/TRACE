#!/usr/bin/env python3
"""Render figA11_selfcond.pdf: where the self-conditioned run departs, and what
happens after a contaminated history step.

Three panels, each answering one question the tables cannot:
  (a) WHEN does the loop leave the reference action sequence? A distribution over
      the first divergent step, so a reader can see whether divergence is an
      early-episode event or spread out.
  (b) Does the gap ACCUMULATE? Step SR against step index under both protocols.
      A flat offset means self-conditioning costs a fixed amount per step; a
      widening gap is genuine error accumulation. The tables report one pooled
      number and cannot distinguish the two.
  (c) Does pruning cost recoverability? Recovery rate after a contaminated step
      with episode-clustered intervals, dense against TRACE.

Reads only docs/w60_selfcond.json plus the raw sample dumps for panel (b), and
refuses to invent numbers: with no artifact present it writes a placeholder
figure that is visibly unfilled rather than a plausible-looking plot.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402

from fig_style import C, apply, despine, panel_label, save, takeaway  # noqa: E402

REPO = os.path.dirname(HERE)
FIGDIR = os.path.join(REPO, "artifacts", "figures")
SRC = os.path.join(REPO, "docs", "w60_selfcond.json")
OUT = os.path.join(FIGDIR, "figA11_selfcond.pdf")

SC = os.path.join(REPO, "logs", "w60_selfcond", "omnigui")
TF_DENSE = os.path.join(REPO, "logs", "w57b_unified", "omnigui", "baseline_full")
TF_TRACE = os.path.join(REPO, "logs", "w57b_acc", "omnigui", "final_c50_h10")

ARM_LABEL = {"dense": "Dense", "trace_c50_h10": "TRACE",
             "trace_c25_h05": "TRACE tight"}
ARM_COLOR = {"dense": C["dense"], "trace_c50_h10": C["ours"],
             "trace_c25_h05": C["witness"]}


def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_steps(arm_dir):
    """(episode, step) -> exact match, read from a finished run's sample dump."""
    f = latest(os.path.join(arm_dir, "results", "*", "*samples*.jsonl"))
    if not f:
        return None
    out = {}
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        se = row.get("step_eval")
        if isinstance(se, str):
            try:
                se = json.loads(se)
            except json.JSONDecodeError:
                continue
        if not isinstance(se, dict):
            continue
        try:
            st = int(se.get("step_id", 0))
        except (TypeError, ValueError):
            continue
        out[(str(se.get("episode_id")), st)] = 1.0 if se.get("exact_match") else 0.0
    return out or None


def sr_by_step(steps, max_step=6):
    """Step SR as a function of step index, with the tail pooled.

    The tail is pooled because episodes average about 3.5 steps, so per-index
    estimates beyond the pooling point rest on too few episodes to plot honestly.
    """
    buckets = defaultdict(list)
    for (_, st), em in steps.items():
        buckets[min(st, max_step)].append(em)
    xs = sorted(buckets)
    ys = [100.0 * float(np.mean(buckets[x])) for x in xs]
    ns = [len(buckets[x]) for x in xs]
    return xs, ys, ns


def panel_a(ax, rep):
    """First divergent step, per arm."""
    div = ((rep.get("tier_ii") or {}).get("divergence") or {})
    panel_label(ax, "(a) First step that departs from the reference")
    if not div:
        placeholder(ax)
        return
    arms = [a for a in ("dense", "trace_c50_h10", "trace_c25_h05") if a in div]
    cats = ["0", "1", "2", "3", "4", "5", "none"]
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        h = div[arm]["onset_hist"]
        tot = sum(h.values()) or 1
        vals = [100.0 * h.get(c, 0) / tot for c in cats]
        ax.bar(np.arange(len(cats)) + i * w - 0.4 + w / 2, vals, width=w * 0.92,
               color=ARM_COLOR[arm], label=ARM_LABEL[arm],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["0", "1", "2", "3", "4", r"$\geq$5", "Never"])
    ax.set_xlabel("First divergent step index")
    ax.set_ylabel("Episodes (%)")
    ax.legend(loc="upper center", ncol=len(arms))
    despine(ax)
    never = {a: div[a]["never_diverges_pct"] for a in arms}
    lo, hi = min(never.values()), max(never.values())
    takeaway(ax, f"Never departs: {lo:.0f}-{hi:.0f}% of episodes",
             loc="upper right")


def panel_b(ax, curves):
    """Does the gap accumulate over the episode, or is it a flat offset?"""
    panel_label(ax, "(b) Step SR by step index, both protocols")
    if not curves:
        placeholder(ax)
        return
    for (arm, proto), (xs, ys, _ns) in sorted(curves.items()):
        ax.plot(xs, ys,
                color=ARM_COLOR.get(arm, C["uniform"]),
                ls="-" if proto == "self" else ":",
                marker="o" if proto == "self" else "s", ms=3.2,
                label=f"{ARM_LABEL.get(arm, arm)}, "
                      f"{'self-conditioned' if proto == 'self' else 'teacher-forced'}")
    ax.set_xlabel("Step index within episode")
    ax.set_ylabel("Step SR (%)")
    ax.legend(loc="lower left", ncol=1)
    despine(ax)


def panel_c(ax, rep):
    """Recovery after a contaminated history step, with intervals."""
    panel_label(ax, "(c) Recovery after a contaminated history step")
    t3 = rep.get("tier_iii") or {}
    rec, base = t3.get("recovery") or {}, t3.get("clean_baseline") or {}
    if not rec:
        placeholder(ax)
        return
    arms = [a for a in ("dense", "trace_c50_h10") if a in rec]
    x = np.arange(len(arms))
    for i, arm in enumerate(arms):
        r = rec[arm]
        lo, hi = r["ci95_pct"]
        ax.bar(i, r["rate_pct"], width=0.5, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=0.5)
        ax.errorbar(i, r["rate_pct"],
                    yerr=[[r["rate_pct"] - lo], [hi - r["rate_pct"]]],
                    fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2.5)
        b = base.get(arm)
        if b:
            ax.hlines(b["rate_pct"], i - 0.3, i + 0.3, color="#333333",
                      ls="--", lw=0.9,
                      label="Clean history" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax.set_ylabel("Next step correct (%)")
    ax.legend(loc="upper right")
    despine(ax)
    dR = t3.get("delta_R_dense_minus_trace_c50_h10")
    if dR:
        lo, hi = dR["ci95_pp"]
        verdict = ("Interval spans zero" if lo <= 0 <= hi
                   else "Interval excludes zero")
        takeaway(ax, f"{verdict}: $\\Delta R$ = {dR['delta_pp']:+.1f} pp",
                 loc="lower left")


def placeholder(ax):
    ax.text(0.5, 0.5, "awaiting W60 runs", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#BB0000",
            style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#BBBBBB")


def build_curves():
    """Panel (b) series: self-conditioned and teacher-forced, per arm."""
    curves = {}
    pairs = [("dense", os.path.join(SC, "sc_dense"), TF_DENSE),
             ("trace_c50_h10", os.path.join(SC, "sc_trace_c50_h10"), TF_TRACE)]
    for arm, sc_dir, tf_dir in pairs:
        sc_steps, tf_steps = load_steps(sc_dir), load_steps(tf_dir)
        if sc_steps is None or tf_steps is None:
            continue
        # Common step set only: an arm missing steps would otherwise shift the
        # per-index means and fake an accumulation effect.
        common = set(sc_steps) & set(tf_steps)
        curves[(arm, "self")] = sr_by_step({k: sc_steps[k] for k in common})
        curves[(arm, "tf")] = sr_by_step({k: tf_steps[k] for k in common})
    return curves


def main():
    apply(scale=1.0)
    rep = {}
    if os.path.exists(SRC):
        with open(SRC, encoding="utf-8") as fh:
            rep = json.load(fh)
    else:
        print(f"note: {SRC} absent -- rendering placeholder panels")
    curves = build_curves()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))
    panel_a(axes[0], rep)
    panel_b(axes[1], curves)
    panel_c(axes[2], rep)
    fig.tight_layout(w_pad=1.6)
    os.makedirs(FIGDIR, exist_ok=True)
    save(fig, OUT)


if __name__ == "__main__":
    main()
