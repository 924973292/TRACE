#!/usr/bin/env python3
"""Fig.1 (introduction teaser): the causal chain, one measured panel per link.

(a) Serving cost -- dense re-prefill TTFT grows with episode step, the
    write-once cache stays flat (lat2 probe, figures/ttft_by_step.csv).
(b) Goal--trajectory asymmetry, shown on a real screen: the keep CDPruner
    commits for one instruction is overlaid on a screenshot that carries
    five ScreenSpot-v2 instructions; the targets that keep loses are marked.
    A dot-CI strip lifts the example to all 310 ordered same-screen pairs
    (cluster-bootstrap intervals from docs/w42_evidence_analyses.json).
    This is an upper-bound cross-query stress test, not a within-goal
    trajectory estimate.
(c) Importance--coverage conflict: ScreenSpot-v2 accuracy vs keep ratio on
    top; the unwitnessed-screen-area diagnostic on a shared budget axis
    below (full sweep from the W44 dumps when available, r=10% otherwise).
(d) Irreversible commitment: paired dumbbells of the acted-on-patch loss
    rate after frame retirement (OmniGUI, docs/w41_commit_conditional.json).
    A mechanical cache audit, not a future-demand oracle.

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
COMMIT = os.path.join(HERE, "..", "docs", "w41_commit_conditional.json")
W42 = os.path.join(HERE, "..", "docs", "w42_evidence_analyses.json")
W44 = os.path.join(HERE, "..", "docs", "w44_fig_evidence.json")

# design width 7.35in prints at 5.5in (x0.75); explicit sizes below carry
# the small text, so the rc bump stays mild.
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
            label="dense re-prefill")
    ax.plot(steps, ours, color=C["ours"], marker="*", ms=7.5, lw=1.8,
            label="write-once (ours)", zorder=5)
    ax.fill_between(steps, ours, dense, color=C["ours"], alpha=0.07)
    # the saved area is the re-encoding cost; label it inside the band
    ax.text(0.70, 0.28, "history\nre-encoded\nevery step", fontsize=6.8,
            color="#444444", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.035, 0.965, "2.1$\\times$ TTFT\n$-$58.1% KV",
            transform=ax.transAxes, fontsize=7.6, fontweight="bold",
            color=C["ours"], va="top", ha="left")
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("TTFT (ms)")
    ax.set_ylim(0, dense[-1] * 1.20)
    ax.set_xticks(steps[::1])
    ax.legend(fontsize=6.9, loc="upper left", frameon=False,
              bbox_to_anchor=(0.015, 0.80), handlelength=1.4,
              labelspacing=0.25)
    ax.set_title("(a) Serving Cost", fontsize=TITLE_FS)
    despine(ax)


def panel_b(fig, gs, S, W44D, W42D):
    """Real-screen evidence + population dot-CI strip."""
    sub = gs.subgridspec(1, 2, wspace=0.14, width_ratios=[0.88, 1.10])
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
    # Shared keep-map convention (scripts/keepviz.py).
    kv.draw_keep(axi, img, kv.keep_grid(rec), outline_lw=0.8)
    anchor_instr = ""
    for r in screen["instructions"]:
        x, y, bw, bh = [float(v) for v in r["bbox"]]
        if r["is_anchor"]:
            anchor_instr = r["instruction"]
            # double stroke marks the query the keep was committed for
            axi.add_patch(patches.Rectangle(
                (x - 6, y - 6), bw + 12, bh + 12, fill=False, ec="white",
                lw=2.4, zorder=6))
        kv.target_box(axi, r["bbox"], w, h, kept=bool(r["kept_cdpruner"]),
                      lw=1.7)
    axi.set_xlabel("query #1 keep; red $\\times$:\nlater target lost",
                   fontsize=6.2, style="italic", labelpad=2)
    del anchor_instr  # instruction text lives in the caption

    # population strip: retention over all 310 ordered pairs, cluster CIs
    axd = fig.add_subplot(sub[0, 1])
    xq = W42D["cross_query"]
    order = [("cdpruner", "CDPruner", CDP_GREEN),
             ("uniform", "Uniform", C["uniform"]),
             ("ours", "Ours", C["ours"]),
             ("divprune", "DivPrune", "#56B4E9")]
    ys = np.arange(len(order))[::-1]
    floor = xq["uniform"]["rate"]
    axd.axvline(floor, color=C["uniform"], ls=":", lw=1.0, zorder=1)
    axd.text(floor - 2.0, -0.42, "geometry floor", fontsize=6.0,
             color="#666666", ha="right", va="center")
    for y, (arm, label, col) in zip(ys, order):
        lo, hi = xq[arm]["ci95_cluster"]
        rate = xq[arm]["rate"]
        axd.plot([lo, hi], [y, y], color=col, lw=1.7, solid_capstyle="round")
        axd.plot([rate], [y], "o", ms=5.0, color=col, zorder=5)
        axd.text(rate, y + 0.22, f"{label} {rate:.0f}", fontsize=6.8,
                 ha="center", color=col, fontweight="bold",
                 clip_on=False)
    axd.set_yticks([])
    axd.set_ylim(-0.60, len(order) - 0.12)
    axd.set_xlim(38, 107)
    axd.spines["left"].set_visible(False)
    axd.tick_params(axis="y", length=0)
    axd.set_xlabel("cross-query target retention (%)\n310 pairs, 95% CI",
                   fontsize=6.8, labelpad=2)
    axd.spines["top"].set_visible(False)
    axd.spines["right"].set_visible(False)
    axd.grid(axis="x", alpha=0.7)
    axd.set_axisbelow(True)
    axd.set_title("(b) Cross-Query Commitment Stress", fontsize=TITLE_FS,
                  x=-0.02)


def panel_c(fig, gs, S, W44D):
    """Accuracy collapse (top) and its coverage mechanism (bottom), one axis."""
    sub = gs.subgridspec(2, 1, hspace=0.10, height_ratios=[1.9, 1.0])
    rs = ["50", "25", "10", "5"]
    blocks = {r: G._ssv2_block(r) for r in rs}
    idx = 6  # ScreenSpot-v2 average column
    xi = np.arange(len(rs))

    axt = fig.add_subplot(sub[0, 0])
    series = [
        ("WITNESS", lambda n, k: k == "ours", C["ours"], "*", 8.6),
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
                 zorder=5 if name == "WITNESS" else 3)
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
    axt.set_title("(c) Importance\u2013Coverage Conflict", fontsize=TITLE_FS)
    despine(axt)

    # mechanism on the shared budget axis: screen area left unwitnessed
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
    else:  # r=10% dumps only (before the W44 sweep lands)
        for arm, col, mk, ms in arms:
            v = S[arm]["uncovered_mean"]
            axb.plot([xi[2]], [v], mk, color=col, ms=ms, zorder=5)
        ymax = max(S[a]["uncovered_mean"] for a, _, _, _ in arms)
    axb.set_ylim(-2, ymax * 1.28)
    axb.set_xticks(xi)
    axb.set_xticklabels([f"{r}%" for r in rs])
    axb.set_xlabel("Keep ratio $r$ ($\\rightarrow$ tighter)", labelpad=1.5)
    axb.set_ylabel("unwit-\nnessed %", fontsize=6.8, labelpad=1)
    axb.tick_params(axis="y", labelsize=6.6)
    despine(axb)


def panel_d(ax, D):
    """Paired dumbbells: acted-on patch lost after retirement."""
    rows = [("omni_c50_h10", "50/10\nmild"), ("omni_c25_h05", "25/5\ntight")]
    ys = [1.0, 0.0]
    for y, (tag, lab) in zip(ys, rows):
        p = D[tag]["paired_rate_diff_ours_minus_uniform"]
        ax.plot([p["rate_a"], p["rate_b"]], [y, y], color="#BBBBBB", lw=1.6,
                zorder=2)
        ax.plot([p["rate_b"]], [y], "o", ms=6.5, color=C["uniform"], zorder=4)
        ax.plot([p["rate_a"]], [y], "o", ms=6.5, color=C["ours"], zorder=4)
        ax.text(p["rate_b"], y + 0.13, f"{p['rate_b']:.0f}", ha="center",
                fontsize=7.2, fontweight="bold", color="#777777")
        ax.text(p["rate_a"], y + 0.13, f"{p['rate_a']:.0f}", ha="center",
                fontsize=7.2, fontweight="bold", color=C["ours"])
        ax.text((p["rate_a"] + p["rate_b"]) / 2, y - 0.155,
                f"$\\Delta${p['diff']:.1f}  "
                f"[{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}]",
                ha="center", fontsize=6.3, color="#333333")
    # the gap between the two rows carries the ratio readout
    ra = [D[t]["paired_rate_diff_ours_minus_uniform"] for t, _ in rows]
    ax.text(46, 0.5, "uniform loses it "
            f"{ra[0]['rate_b'] / ra[0]['rate_a']:.0f}$\\times$ / "
            f"{ra[1]['rate_b'] / ra[1]['rate_a']:.1f}$\\times$ more often",
            ha="center", va="center", fontsize=6.2, style="italic",
            color="#666666")
    ax.plot([], [], "o", ms=5, color=C["ours"], label="WITNESS")
    ax.plot([], [], "o", ms=5, color=C["uniform"], label="uniform")
    ax.legend(fontsize=6.8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=1, handletextpad=0.3,
              labelspacing=0.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=7.2)
    ax.set_ylim(-0.55, 1.85)
    ax.set_xlim(0, 100)
    ax.set_xlabel("acted-on target lost (%)")
    ax.text(0.02, 0.02, "retirement audit,\nnot a future oracle",
            transform=ax.transAxes, fontsize=6.2, style="italic",
            color="#666666", va="bottom")
    ax.set_title("(d) Irreversible Commitment", fontsize=8.7, x=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def main():
    S = json.load(open(STATS, encoding="utf-8"))
    D = json.load(open(COMMIT, encoding="utf-8"))
    W42D = json.load(open(W42, encoding="utf-8"))
    W44D = json.load(open(W44, encoding="utf-8"))
    fig = plt.figure(figsize=(7.35, 2.62))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.92, 1.26, 0.98, 0.90],
                          left=0.072, right=0.992, top=0.895, bottom=0.215,
                          wspace=0.44)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig, gs[0, 1], S, W44D, W42D)
    panel_c(fig, gs[0, 2], S, W44D)
    panel_d(fig.add_subplot(gs[0, 3]), D)
    out = os.path.join(FIGDIR, "fig1_motivation.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
