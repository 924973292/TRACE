#!/usr/bin/env python3
"""Fig.6: one row per module -- mechanism exemplar pair + full-population
paired statistic, all from the evaluated 120-frame ScreenSpot-v2 dumps
(logs/w36_qual_local/) and the derived JSONs. Nothing is re-simulated.

Row A (LIP): the prior field of one screen that carries five ScreenSpot-v2
    instructions, next to the prior-only keep; the field is conditioned on
    no instruction yet covers every target. Population: detected-element
    coverage recall per selector (docs/w42_evidence_analyses.json).
Row B (NEO): diversity-off vs full ordering on the same screen at the same
    budget. Population: per-frame keep-dispersion pairs; 108 of 120 frames
    sit above the diagonal (paired diff CI excludes zero).
Row C (NCR): greedy prefix vs repaired keep on the same screen. Population:
    per-frame uncovered-area pairs colored by target rescue outcome
    (44 rescued vs 5 lost).

Exemplar indices come from docs/w44_fig_evidence.json (selection rules are
recorded there).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import keepviz as kv  # noqa: E402
from fig_style import C, apply  # noqa: E402

DATA = os.path.join(HERE, "..", "logs", "w36_qual_local")
FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
W42 = os.path.join(HERE, "..", "docs", "w42_evidence_analyses.json")
W44 = os.path.join(HERE, "..", "docs", "w44_fig_evidence.json")
STATS = os.path.join(HERE, "..", "docs", "w36_qual_stats.json")

ABL = "#56B4E9"          # ablated-twin overlay (matches Fig.9 convention)
RESCUE = "#007D45"
LOST = "#B3261E"

# 7.05in design prints at 5.4in -> mild upscale keeps print sizes honest
apply(scale=1.28)
ROW_TITLE_FS = 11.6
XLAB_FS = 8.6


def load_jsonl(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8")
            if ln.strip()]


# Drawing primitives are shared with every other selection figure so the
# kept/dropped convention cannot drift between figures.
keep_grid = kv.keep_grid
draw_screenshot = kv.show_image


def overlay_mask(ax, img, mask, color, special=None,
                 special_color=None, tint_alpha=0.20):
    """Keep map in the shared convention: kept cells keep the original pixels
    (optionally tinted faintly to identify the arm), dropped cells are dimmed
    but stay legible. `special` cells get a stronger wash."""
    w, h = img.size
    kv.draw_keep(ax, img, mask, tint=color, tint_alpha=tint_alpha,
                 outline_lw=0.8)
    if special is not None and special.any():
        rgba = np.zeros(mask.shape + (4,))
        from matplotlib.colors import to_rgba as _rgba
        rgba[special] = _rgba(special_color, 0.62)
        ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest",
                  zorder=3.5)
        kv.keep_outline(ax, special, w, h, color=special_color, lw=0.9)


def gt_box(ax, bbox, img, lw=2.0, kept=True):
    kv.target_box(ax, bbox, img.size[0], img.size[1], kept=kept, lw=lw,
                  cross_on_miss=not kept)


def badge(ax, ok, fontsize=8.6):
    ax.text(0.04, 0.03, "HIT" if ok else "MISS", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color="white",
            va="bottom", ha="left", zorder=9,
            bbox=dict(boxstyle="round,pad=0.26",
                      fc=("#007D45" if ok else "#B3261E"), ec="none"))


def xlab(ax, text):
    ax.set_xlabel(text, fontsize=XLAB_FS, style="italic", labelpad=2.5)


def row_title(fig, y, text):
    fig.text(0.008, y, text, fontsize=ROW_TITLE_FS, fontweight="bold",
             va="bottom")


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.55)
    ax.set_axisbelow(True)


def row_a(fig, gs, dumps, W44D, W42D):
    pick = W44D["fig6_picks"]["lmp"]
    screen = next(s for s in W44D["multi_screens"]
                  if s["image"] == pick["image"])
    i = screen["anchor_idx"]
    img = Image.open(os.path.join(DATA, "images",
                                  screen["image"])).convert("RGB")
    w, h = img.size
    mass = json.load(open(os.path.join(DATA, "harvest_l120",
                                       f"mass_{i:03d}.json"),
                          encoding="utf-8"))
    P = np.asarray(mass["P"], float).reshape(mass["grid_h"], mass["grid_w"])

    ax = fig.add_subplot(gs[0, 0])
    kv.score_heat(ax, img, P, vmax=max(P.max(), 1e-6))
    kv.element_boxes(ax, mass["boxes"], w, h, ec=C["best"], lw=0.7, alpha=0.8)
    for r in screen["instructions"]:
        gt_box(ax, r["bbox"], img, lw=1.8)
    xlab(ax, "Mass field over the input. Boxes\nmark all five queries' targets")

    ax = fig.add_subplot(gs[0, 1])
    overlay_mask(ax, img, keep_grid(dumps["aonly"][i]), C["ours"])
    for r in screen["instructions"]:
        gt_box(ax, r["bbox"], img, lw=1.8)
    xlab(ax, "prior-only keep @ $r{=}10\\%$\ncovers every target")

    # population: detected-element coverage recall with frame-bootstrap CIs
    ax = fig.add_subplot(gs[0, 2:])
    er = W42D["element_recall"]
    rows = [("cdpruner", "CDPruner", "#009E73"),
            ("uniform", "Uniform", C["uniform"]),
            ("divprune", "DivPrune", ABL),
            ("ours", "TRACE (LIP prior)", C["ours"])]
    ys = np.arange(len(rows))
    for y, (arm, label, col) in zip(ys, rows):
        lo, hi = er[arm]["ci95_frame"]
        v = er[arm]["recall_mean"]
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot([v], [y], "o", ms=6.5, color=col, zorder=5)
        ax.text(hi + 1.2, y, f"{v:.1f}", fontsize=8.8, va="center",
                color=col, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows], fontsize=8.8)
    ax.set_xlim(38, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Detected-element coverage recall (%)",
                  fontsize=8.6, labelpad=2)
    ax.text(0.985, 0.06, "Score concentration leaves half of all\n"
            "elements uncovered (CDPruner)", transform=ax.transAxes,
            fontsize=9.0, style="italic", color="#333333", ha="right",
            va="bottom")
    despine(ax)
    ax.grid(axis="y", alpha=0.0)


def row_b(fig, gs, dumps, gt, hits, W44D):
    i = W44D["fig6_picks"]["nwe"]["idx"]
    img = Image.open(os.path.join(
        DATA, "images",
        os.path.basename(dumps["ours"][i]["image_path"]))).convert("RGB")
    w, h = img.size

    ax = fig.add_subplot(gs[1, 0])
    overlay_mask(ax, img, keep_grid(dumps["nodiv"][i]), ABL)
    gt_box(ax, gt[i]["bbox"], img, kept=hits["nodiv"][i] == 1)
    badge(ax, hits["nodiv"][i] == 1)
    xlab(ax, "w/o diversity: budget\nstacks on look-alikes")

    ax = fig.add_subplot(gs[1, 1])
    overlay_mask(ax, img, keep_grid(dumps["ours"][i]), C["ours"])
    gt_box(ax, gt[i]["bbox"], img, kept=hits["ours"][i] == 1)
    badge(ax, hits["ours"][i] == 1)
    xlab(ax, "full ordering: same $k$,\nspread across evidence")

    P = W44D["paired"]
    dn = np.asarray(P["disp_nodiv"], float)
    do = np.asarray(P["disp_ours"], float)
    ok = np.isfinite(dn) & np.isfinite(do)
    above = int((do[ok] > dn[ok]).sum())

    ax = fig.add_subplot(gs[1, 2])
    lo = min(dn[ok].min(), do[ok].min()) - 0.03
    hi = max(dn[ok].max(), do[ok].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, ls=":")
    ax.scatter(dn[ok], do[ok], s=11, color="#9DB8CE", alpha=0.75, lw=0)
    ax.scatter([dn[i]], [do[i]], marker="*", s=110, color=C["ours"],
               zorder=6, label="exemplar")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_xlabel("w/o diversity dispersion", fontsize=8.6)
    ax.set_ylabel("TRACE dispersion", fontsize=8.6, labelpad=1)
    ax.text(0.04, 0.965, f"{above}/{int(ok.sum())} frames above\n"
            "the diagonal", transform=ax.transAxes, fontsize=8.2,
            va="top", color="#333333")
    ax.text(0.97, 0.05, f"{int(ok.sum()) - above}/{int(ok.sum())} below",
            transform=ax.transAxes, fontsize=7.6, ha="right", va="bottom",
            color="#888888")
    despine(ax)

    ax = fig.add_subplot(gs[1, 3])
    d = (do - dn)[ok]
    ax.hist(d, bins=24, color="#9DB8CE", edgecolor="white", lw=0.4)
    ax.axvline(0, color="#555555", ls=":", lw=1.0)
    ax.axvline(float(d.mean()), color=C["ours"], lw=1.6)
    ax.text(float(d.mean()) + 0.006, ax.get_ylim()[1] * 0.97,
            "$\\Delta$ +0.055\n[0.048, 0.063]", fontsize=8.0,
            color=C["ours"], va="top")
    ax.set_xlabel("dispersion gain per frame", fontsize=8.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    despine(ax)
    ax.grid(alpha=0.0)


def row_c(fig, gs, dumps, gt, hits, W44D, W42D):
    i = W44D["fig6_picks"]["awc"]["idx"]
    img = Image.open(os.path.join(
        DATA, "images",
        os.path.basename(dumps["ours"][i]["image_path"]))).convert("RGB")
    w, h = img.size
    kg_n = keep_grid(dumps["nocrs"][i])
    kg_o = keep_grid(dumps["ours"][i])
    witness = kg_o & ~kg_n

    ax = fig.add_subplot(gs[2, 0])
    overlay_mask(ax, img, kg_n, ABL)
    gt_box(ax, gt[i]["bbox"], img, kept=hits["nocrs"][i] == 1)
    badge(ax, hits["nocrs"][i] == 1)
    xlab(ax, "greedy prefix only:\ncoverage holes remain")

    ax = fig.add_subplot(gs[2, 1])
    overlay_mask(ax, img, kg_o, C["ours"], special=witness,
                 special_color=C["witness"])
    gt_box(ax, gt[i]["bbox"], img, kept=hits["ours"][i] == 1)
    badge(ax, hits["ours"][i] == 1)
    xlab(ax, "+NCR repair (amber):\nsame budget, holes filled")

    P = W44D["paired"]
    un = np.asarray(P["unc_nocrs"], float)
    uo = np.asarray(P["unc_ours"], float)
    lab = P["rescue_label"]

    ax = fig.add_subplot(gs[2, 2])
    m = max(un.max(), uo.max()) * 1.06
    ax.plot([0, m], [0, m], color="#888888", lw=0.9, ls=":")
    for grp, col, size, z in (("same", "#B9B9B9", 11, 3),
                              ("rescued", RESCUE, 17, 5),
                              ("lost", LOST, 17, 5)):
        xs = [u for u, l_ in zip(un, lab) if l_ == grp]
        ys = [u for u, l_ in zip(uo, lab) if l_ == grp]
        ax.scatter(xs, ys, s=size, color=col, alpha=0.85, lw=0, zorder=z,
                   label=f"{grp} ({len(xs)})")
    ax.scatter([un[i]], [uo[i]], marker="*", s=110, color=C["witness"],
               zorder=7)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_box_aspect(1)
    ax.set_xlabel("uncovered % w/o repair", fontsize=8.6)
    ax.set_ylabel("uncovered % with NCR", fontsize=8.6, labelpad=1)
    ax.legend(fontsize=7.4, loc="upper left", frameon=False,
              handletextpad=0.2, labelspacing=0.22, borderaxespad=0.15)
    ax.text(0.97, 0.60, "Below diagonal =\ncoverage repaired",
            transform=ax.transAxes, fontsize=9.0, style="italic",
            ha="right", va="bottom", color="#333333")
    despine(ax)

    ax = fig.add_subplot(gs[2, 3])
    d = un - uo
    groups = [("same", "#B9B9B9"), ("rescued", RESCUE), ("lost", LOST)]
    stacks = [[v for v, l_ in zip(d, lab) if l_ == g] for g, _ in groups]
    ax.hist(stacks, bins=18, stacked=True,
            color=[c for _, c in groups], edgecolor="white", lw=0.3)
    aw = W42D["awc_coupling"]
    ax.text(0.97, 0.97,
            f"{aw['n_rescued']} rescued /\n{aw['n_lost']} lost targets",
            transform=ax.transAxes, fontsize=8.2, ha="right", va="top",
            color="#333333")
    ax.set_xlabel("coverage repair per frame (pts)", fontsize=8.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    despine(ax)
    ax.grid(alpha=0.0)


def main():
    kv.enable_cjk()
    W42D = json.load(open(W42, encoding="utf-8"))
    W44D = json.load(open(W44, encoding="utf-8"))
    S = json.load(open(STATS, encoding="utf-8"))
    gt = {d["idx"]: d for d in
          json.load(open(os.path.join(DATA, "gt120.json"),
                         encoding="utf-8"))}
    dumps = {a: load_jsonl(os.path.join(DATA, f"ssv2_{a}_l120.jsonl"))
             for a in ("ours", "nocrs", "nodiv", "aonly")}
    hits = {a: S[a]["hits"] for a in ("ours", "nocrs", "nodiv")}

    fig = plt.figure(figsize=(7.05, 7.15))
    gs = fig.add_gridspec(3, 4, width_ratios=[0.94, 0.94, 1.55, 1.15],
                          left=0.055, right=0.985, top=0.962, bottom=0.052,
                          wspace=0.28, hspace=0.48)
    row_title(fig, 0.970, "(a) LIP: structural affordance $\\to$ "
              "query-independent interaction mass")
    row_a(fig, gs, dumps, W44D, W42D)
    row_title(fig, 0.645, "(b) NEO: instruction relevance without "
              "redundancy, one nested order")
    row_b(fig, gs, dumps, gt, hits, W44D)
    row_title(fig, 0.322, "(c) NCR: native-token repair fixes the ordering "
              "tail within budget")
    row_c(fig, gs, dumps, gt, hits, W44D, W42D)

    out = os.path.join(FIGDIR, "fig4_modules.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
