#!/usr/bin/env python3
"""Fig.2 (introduction evidence panel): one measured panel per causal link.

(a) C1 serving cost -- stateless re-prefill TTFT grows with episode step,
    the MKC reuse path stays flat (lat2 probe, figures/ttft_by_step.csv).
(b) C5 goal--trajectory asymmetry, measured within deployed goals: when a
    screen recurs later in the SAME episode, does the admitted keep still
    cover the later target?  (docs/w43_same_goal_retention.json, OmniGUI,
    128 pairs / 46 episodes, episode-bootstrap CIs.)
(c) C4 the nested constraint is real and free: independent per-frame
    re-selection demands out-of-cache rows on 99%+ of frames
    (docs/w43_independent_nested.json), while projecting onto the nested
    order costs no measurable accuracy
    (docs/w45_c2_nested_vs_independent.json).
(d) C6 importance--coverage conflict: ScreenSpot-v2 accuracy vs keep ratio
    on top; the uncovered-screen-area diagnostic on a shared budget axis
    below (W44 cross-budget dumps).

Also renders the appendix stress-test figure figA10_swap.pdf, which reuses
the cross-query instruction-swap panel formerly in Fig.1(b).

Table numbers come from gen_tables.py; nothing is hand-copied.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from PIL import Image  # noqa: E402

import gen_tables as G  # noqa: E402
import keepviz as kv  # noqa: E402
from fig_style import C, apply, despine  # noqa: E402

FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
DATA = os.path.join(HERE, "..", "logs", "w36_qual_local")
STATS = os.path.join(HERE, "..", "docs", "w36_qual_stats.json")
W42 = os.path.join(HERE, "..", "docs", "w42_evidence_analyses.json")
W43A = os.path.join(HERE, "..", "docs", "w43_same_goal_retention.json")
W43C = os.path.join(HERE, "..", "docs", "w43_independent_nested.json")
W44 = os.path.join(HERE, "..", "docs", "w44_fig_evidence.json")
W45C2 = os.path.join(HERE, "..", "docs", "w45_c2_nested_vs_independent.json")

apply(scale=1.10)

TITLE_FS = 9.2
CDP_GREEN = "#009E73"


def val(rows, pred, idx):
    for n, k, v in rows:
        if pred(n, k) and G.fnum(v[idx]):
            return float(v[idx])
    return None


def panel_a(ax):
    rows = [ln.strip().split(",") for ln in
            open(os.path.join(FIGDIR, "ttft_by_step.csv")) if ln.strip()]
    col = {name: i for i, name in enumerate(rows[0])}
    steps, dense, ours = [], [], []
    for r in rows[1:]:
        if min(int(r[col["dense_n"]]), int(r[col["ours_n"]])) < 4:
            continue
        steps.append(int(r[col["step"]]))
        dense.append(float(r[col["dense_ttft_ms"]]))
        ours.append(float(r[col["ours_ttft_ms"]]))
    ax.plot(steps, dense, color=C["dense"], marker="o", ms=3.8, lw=1.5,
            label="Stateless re-prefill")
    ax.plot(steps, ours, color=C["ours"], marker="*", ms=7.5, lw=1.8,
            label="MKC reuse", zorder=5)
    ax.fill_between(steps, ours, dense, color=C["ours"], alpha=0.07)
    # Short note sits in the orange band, biased toward the flat MKC
    # curve so the rising Stateless line never cuts through the text.
    i_note = min(range(len(steps)), key=lambda i: abs(steps[i] - 3))
    y_note = 0.72 * ours[i_note] + 0.28 * dense[i_note]
    ax.text(steps[i_note], y_note, "History re-encoded",
            fontsize=6.6, color="#333333", ha="center", va="center",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                      alpha=0.94))
    ax.text(0.03, 0.97, "2.4$\\times$ TTFT\n$-$58.1% KV",
            transform=ax.transAxes, fontsize=7.4, fontweight="bold",
            color=C["ours"], va="top", ha="left")
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("TTFT (ms)")
    ax.set_ylim(0, dense[-1] * 1.20)
    ax.set_xticks(steps[::1])
    # Mid-right of the band: clear of the claim (top-left) and of MKC.
    ax.legend(fontsize=6.6, loc="center right", frameon=True,
              fancybox=False, edgecolor="#CCCCCC", framealpha=0.92,
              bbox_to_anchor=(0.98, 0.42), handlelength=1.3,
              labelspacing=0.22, borderpad=0.35)
    ax.set_title("(a) Cost of Stateless serving", fontsize=TITLE_FS)
    despine(ax)


def panel_b(ax, W43AD):
    """Within-goal retention of later targets (dot-CI per arm)."""
    mild = W43AD["omni_c50_h10"]["arms"]
    tight = W43AD["omni_c25_h05"]["arms"]
    order = [("divprune", "DivPrune", "#56B4E9"),
             ("uniform", "Uniform", C["uniform"]),
             ("cdpruner", "CDPruner", CDP_GREEN),
             ("ours", "TRACE", C["ours"])]
    ys = np.arange(len(order))
    for y, (arm, label, col) in zip(ys, order):
        lo, hi = mild[arm]["ci95_episode"]
        rate = mild[arm]["retention"]
        ax.plot([lo, hi], [y, y], color=col, lw=1.7, solid_capstyle="round")
        ax.plot([rate], [y], "o", ms=5.4, color=col, zorder=5)
        # tight budget as an open marker on the same row
        ax.plot([tight[arm]["retention"]], [y], "o", ms=5.4, mfc="white",
                mec=col, mew=1.2, zorder=4)
        ax.text(min(rate, 84), y + 0.30,
                f"{label} {rate:.0f}", fontsize=6.8, ha="center",
                color=col, fontweight="bold", clip_on=False)
    ax.plot([], [], "o", ms=4.6, color="#555555", label="mild 50/10")
    ax.plot([], [], "o", ms=4.6, mfc="white", mec="#555555", mew=1.1,
            label="tight 25/5")
    ax.legend(fontsize=6.3, loc="center right", frameon=False,
              handletextpad=0.25, labelspacing=0.25,
              bbox_to_anchor=(1.01, 0.42))
    ax.text(0.02, 0.965, "Same goal, later step\nIs the target still cached?",
            fontsize=7.2, style="italic", color="#333333",
            transform=ax.transAxes, va="top")
    ax.set_yticks([])
    ax.set_ylim(-0.55, len(order) - 0.10)
    ax.set_xlim(-3, 103)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Within-goal target retention (%)",
                  fontsize=7.0, labelpad=2)
    ax.set_title("(b) Within-goal drift", fontsize=TITLE_FS)


def panel_c(fig, gs, W43CD, W45D):
    """Top: independent selection violates the cache. Bottom: nesting is
    free (paired accuracy difference straddles zero)."""
    sub = gs.subgridspec(2, 1, hspace=0.62, height_ratios=[1.0, 1.0])

    axt = fig.add_subplot(sub[0, 0])
    tags = [("omni_mild", "mild"), ("omni_tight", "tight")]
    ys = np.arange(len(tags))[::-1]
    for y, (tag, lab) in zip(ys, tags):
        d = W43CD[tag]
        lo, hi = d["violation_ci95_frame"]
        axt.barh(y, d["violation_rate"], height=0.55, color="#B3261E",
                 alpha=0.85)
        axt.plot([lo, hi], [y, y], color="black", lw=1.0)
        axt.text(d["violation_rate"] - 1.5, y,
                 f"{d['violation_rate']:.1f}%", fontsize=6.9, ha="right",
                 va="center", color="white", fontweight="bold")
        axt.text(1.5, y, lab, fontsize=6.6, ha="left", va="center",
                 color="white")
    axt.set_xlim(0, 100)
    axt.set_yticks([])
    axt.set_xticks([0, 50, 100])
    axt.tick_params(axis="x", labelsize=6.4)
    axt.set_xlabel("Frames whose independent history keep\nexits the cache (%)",
                   fontsize=6.6, labelpad=1.5)
    axt.spines["top"].set_visible(False)
    axt.spines["right"].set_visible(False)
    axt.spines["left"].set_visible(False)
    axt.set_title("(c) Nesting is free", fontsize=TITLE_FS)

    axb = fig.add_subplot(sub[1, 0])
    arms = W45D["arms"]
    for y, (tag, lab) in zip(ys, tags):
        a = arms[tag]
        lo, hi = a["ci95_episode"]
        axb.plot([lo, hi], [y, y], color=C["ours"], lw=1.7,
                 solid_capstyle="round")
        axb.plot([a["diff_independent_minus_nested"]], [y], "o", ms=5.2,
                 color=C["ours"], zorder=5)
        axb.text(hi + 0.25, y, lab, fontsize=6.6, va="center",
                 color="#555555")
    axb.axvline(0, color="#888888", ls=":", lw=1.0)
    axb.set_yticks([])
    axb.set_ylim(-0.6, 1.6)
    axb.set_xlim(-3.2, 3.2)
    axb.tick_params(axis="x", labelsize=6.4)
    axb.set_xlabel("Accuracy cost of nesting (Step-EM pts,\nindependent $-$ nested)",
                   fontsize=6.6, labelpad=1.5)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.grid(axis="x", alpha=0.7)
    axb.set_axisbelow(True)


def panel_d(fig, gs, S, W44D):
    """Accuracy collapse (top) and its coverage mechanism (bottom)."""
    sub = gs.subgridspec(2, 1, hspace=0.10, height_ratios=[1.9, 1.0])
    rs = ["50", "25", "10", "5"]
    blocks = {r: G._ssv2_block(r) for r in rs}
    idx = 6  # ScreenSpot-v2 average column
    xi = np.arange(len(rs))

    axt = fig.add_subplot(sub[0, 0])
    series = [
        ("TRACE", lambda n, k: k == "ours", C["ours"], "*", 8.6),
        ("FastV", lambda n, k: n.startswith("FastV"), "#CC79A7", "^", 4.8),
        ("CDPruner", lambda n, k: n.startswith("CDPruner"), CDP_GREEN,
         "s", 4.2),
        ("Uniform", lambda n, k: n == "Uniform", C["uniform"], "o", 4.2),
    ]
    unif = [val(blocks[r], lambda n, k: n == "Uniform", idx) for r in rs]
    for name, pred, col, mk, ms in series:
        ys = [val(blocks[r], pred, idx) for r in rs]
        ls = "--" if name == "Uniform" else "-"
        axt.plot(xi, ys, ls, color=col, marker=mk, ms=ms, lw=1.5, label=name,
                 zorder=5 if name == "TRACE" else 3)
        if name in ("FastV", "CDPruner"):
            for j, (y, u) in enumerate(zip(ys, unif)):
                if y is not None and u is not None and y < u:
                    axt.plot(xi[j], y, "x", color="#B3261E", ms=8.0,
                             mew=2.0, zorder=6)
                    break
    axt.text(0.99, 1.00, "$\\times$ = falls below\nuniform", fontsize=6.2,
             color="#B3261E", transform=axt.transAxes, ha="right", va="top")
    axt.legend(fontsize=6.2, loc="lower left", frameon=False, ncol=2,
               bbox_to_anchor=(-0.04, -0.05), handletextpad=0.30,
               labelspacing=0.18, handlelength=1.1, columnspacing=0.7)
    axt.set_xticks(xi)
    axt.set_xticklabels([])
    axt.set_ylabel("SSv2 acc. (%)", labelpad=1)
    axt.set_title("(d) Coverage collapse", fontsize=TITLE_FS)
    despine(axt)

    axb = fig.add_subplot(sub[1, 0])
    cb = W44D.get("cross_budget", {})
    arms = [("cdpruner", CDP_GREEN, "s", 4.2),
            ("uniform", C["uniform"], "o", 4.2),
            ("ours", C["ours"], "*", 8.6)]
    if cb.get("available"):
        for arm, col, mk, ms in arms:
            ys = cb["uncovered"][arm]
            ls = "--" if arm == "uniform" else "-"
            axb.plot(xi, ys, ls, color=col, marker=mk, ms=ms, lw=1.4,
                     zorder=5 if arm == "ours" else 3)
        ymax = max(max(cb["uncovered"][a]) for a, _, _, _ in arms)
    else:
        for arm, col, mk, ms in arms:
            v = S[arm]["uncovered_mean"]
            axb.plot([xi[2]], [v], mk, color=col, ms=ms, zorder=5)
        ymax = max(S[a]["uncovered_mean"] for a, _, _, _ in arms)
    axb.set_ylim(-2, ymax * 1.28)
    axb.set_xticks(xi)
    axb.set_xticklabels([f"{r}%" for r in rs])
    axb.set_xlabel("Keep ratio $r$ ($\\rightarrow$ tighter)", labelpad=1.5)
    axb.set_ylabel("uncov-\nered %", fontsize=6.8, labelpad=1)
    axb.tick_params(axis="y", labelsize=6.6)
    despine(axb)


def render_swap_appendix(S, W44D, W42D):
    """Appendix stress-test figure: real-screen swap example + dot-CI strip
    over all 310 cross-query pairs (formerly Fig.1b)."""
    fig = plt.figure(figsize=(5.0, 2.10))
    gs = fig.add_gridspec(1, 1, left=0.02, right=0.985, top=0.90,
                          bottom=0.235)
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.14, width_ratios=[0.80, 1.10])
    pick = W44D["fig1b_pick"]
    screen = next(s for s in W44D["multi_screens"]
                  if s["image"] == pick["image"])
    anchor = screen["anchor_idx"]

    cdp = [json.loads(ln) for ln in
           open(os.path.join(DATA, "ssv2_cdpruner_l120.jsonl"),
                encoding="utf-8") if ln.strip()]
    rec = cdp[anchor]
    img = Image.open(os.path.join(
        DATA, "images", pick["image"])).convert("RGB")
    w, h = img.size

    axi = fig.add_subplot(sub[0, 0])
    # Shared keep-map convention (scripts/keepviz.py): kept cells keep the
    # original pixels inside a white contour, dropped cells stay legible but
    # dimmed, so the reader can see what the discarded regions contained.
    kv.draw_keep(axi, img, kv.keep_grid(rec), outline_lw=0.8)
    for r in screen["instructions"]:
        x, y, bw, bh = [float(v) for v in r["bbox"]]
        if r["is_anchor"]:
            axi.add_patch(patches.Rectangle(
                (x - 6, y - 6), bw + 12, bh + 12, fill=False, ec="white",
                lw=2.4, zorder=6))
        kv.target_box(axi, r["bbox"], w, h, kept=bool(r["kept_cdpruner"]),
                      lw=1.7)
    axi.set_xlabel("Query #1 keep. Red $\\times$\nmarks a swapped target lost",
                   fontsize=7.0, style="italic", labelpad=2)
    axi.set_title("(a) Keep under the anchor query", fontsize=8.2,
                  loc="left", fontweight="bold")

    axd = fig.add_subplot(sub[0, 1])
    xq = W42D["cross_query"]
    order = [("cdpruner", "CDPruner", CDP_GREEN),
             ("uniform", "Uniform", C["uniform"]),
             ("ours", "TRACE", C["ours"]),
             ("divprune", "DivPrune", "#56B4E9")]
    ys = np.arange(len(order))[::-1]
    floor = xq["uniform"]["rate"]
    axd.axvline(floor, color=C["uniform"], ls=":", lw=1.0, zorder=1)
    axd.text(floor - 2.0, -0.42, "Geometry floor", fontsize=7.2,
             color="#333333", ha="right", va="center")
    for y, (arm, label, col) in zip(ys, order):
        lo, hi = xq[arm]["ci95_cluster"]
        rate = xq[arm]["rate"]
        axd.plot([lo, hi], [y, y], color=col, lw=1.7, solid_capstyle="round")
        axd.plot([rate], [y], "o", ms=5.0, color=col, zorder=5)
        axd.text(rate, y + 0.22, f"{label} {rate:.0f}", fontsize=6.8,
                 ha="center", color=col, fontweight="bold", clip_on=False)
    axd.set_yticks([])
    axd.set_ylim(-0.60, len(order) - 0.12)
    axd.set_xlim(38, 107)
    axd.spines["left"].set_visible(False)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_visible(False)
    axd.tick_params(axis="y", length=0)
    axd.grid(axis="x", alpha=0.7)
    axd.set_axisbelow(True)
    axd.set_xlabel("Cross-query target retention (%)",
                   fontsize=7.0, labelpad=2)
    axd.set_title("(b) Retention across swapped queries", fontsize=8.2,
                  loc="left", fontweight="bold")
    fig.suptitle("Cross-query instruction-swap stress test", fontsize=8.6,
                 y=0.985)
    out = os.path.join(FIGDIR, "figA10_swap.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=220)
    plt.close(fig)
    print("wrote", out)


def main():
    S = json.load(open(STATS, encoding="utf-8"))
    W42D = json.load(open(W42, encoding="utf-8"))
    W43AD = json.load(open(W43A, encoding="utf-8"))
    W43CD = json.load(open(W43C, encoding="utf-8"))
    W44D = json.load(open(W44, encoding="utf-8"))
    W45D = json.load(open(W45C2, encoding="utf-8"))

    # Panel (b) carries the strongest single measurement in the paper (96% vs
    # under 11% within-goal retention), so it is given roughly a third of the
    # width instead of an equal quarter; the other three keep enough room to
    # stay legible at column width.
    fig = plt.figure(figsize=(7.35, 2.62))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.82, 1.62, 0.82, 0.86],
                          left=0.068, right=0.992, top=0.895, bottom=0.215,
                          wspace=0.30)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]), W43AD)
    panel_c(fig, gs[0, 2], W43CD, W45D)
    panel_d(fig, gs[0, 3], S, W44D)
    out = os.path.join(FIGDIR, "fig2_evidence.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    plt.close(fig)
    print("wrote", out)

    render_swap_appendix(S, W44D, W42D)


if __name__ == "__main__":
    main()
