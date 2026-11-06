#!/usr/bin/env python3
"""Fig.1: the admission-and-contraction lifecycle of one frame (schematic).

Left block: a screenshot enters at step t, is encoded once, and a single
admission-time ORDER over its native tokens is committed. The first
|S_cur| entries serve the frame while it is current; when it retires, the
cache CONTRACTS by deleting rows so that exactly the nested prefix
|S_hist| survives. Historical frames are never re-encoded or re-scored,
and future steps (unknown at admission) read from the surviving rows.

Right block: why the order must be nested -- an independently re-selected
history keep demands rows that are no longer in the cache (unservable on
99%+ of frames), while a nested prefix is realizable by deletion alone.

Pure schematic: the only numbers shown are cited from measured results
(docs/w43_independent_nested.json, docs/w45_c2_nested_vs_independent.json).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from fig_style import C, apply  # noqa: E402

FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")

apply(scale=1.06)

ORANGE = C["ours"]          # history prefix (survives contraction)
AMBER = "#F0A860"           # current-only tokens (deleted at retirement)
GREY = "#D8D8D8"            # never admitted
RED = "#B3261E"
GREEN = "#2E7D32"
INK = "#333333"


def token_strip(ax, x, y, w, h, n, n_hist, n_cur, order_numbers=False):
    """Draw one ordered token strip with nested prefixes."""
    cw = w / n
    for i in range(n):
        if i < n_hist:
            fc = ORANGE
        elif i < n_cur:
            fc = AMBER
        else:
            fc = GREY
        ax.add_patch(patches.FancyBboxPatch(
            (x + i * cw + 0.06 * cw, y), cw * 0.88, h,
            boxstyle="round,pad=0.008", fc=fc, ec="white", lw=0.4))
    return cw


def brace(ax, x0, x1, y, label, color, above=True, fs=6.4, dy=0.030):
    ym = y + dy if above else y - dy
    ax.plot([x0, x0, x1, x1],
            [y, ym, ym, y], color=color, lw=0.9, clip_on=False)
    ax.text((x0 + x1) / 2, ym + (0.012 if above else -0.012), label,
            ha="center", va="bottom" if above else "top", fontsize=fs,
            color=color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, label=None, fs=6.4, color=INK, lw=1.1,
          style="-|>", ldy=0.03, ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=9, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + ldy, label, fontsize=fs,
                ha=ha, va="bottom", color=color)


def screenshot_icon(ax, x, y, w, h):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006", fc="#F5F7FA",
        ec="#8899AA", lw=0.9))
    # abstract UI: header bar, two buttons, text lines
    ax.add_patch(patches.Rectangle((x + 0.008, y + h - 0.055), w - 0.016,
                                   0.040, fc="#B7C6D9", ec="none"))
    ax.add_patch(patches.Rectangle((x + 0.012, y + 0.032), 0.052, 0.036,
                                   fc="#7FA8D9", ec="none"))
    ax.add_patch(patches.Rectangle((x + 0.076, y + 0.032), 0.052, 0.036,
                                   fc="#7FA8D9", ec="none"))
    for k in range(3):
        ax.add_patch(patches.Rectangle(
            (x + 0.012, y + 0.095 + 0.034 * k), w - 0.045, 0.012,
            fc="#C9D4E0", ec="none"))


def main():
    fig = plt.figure(figsize=(7.35, 2.00))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- left block: the lifecycle ----------------
    # 1. screenshot arrives
    screenshot_icon(ax, 0.015, 0.42, 0.145, 0.42)
    ax.text(0.0875, 0.355, "step $t$: new screen\n(goal known,\ntrajectory unknown)",
            ha="center", va="top", fontsize=6.4, color=INK)

    arrow(ax, 0.168, 0.63, 0.225, 0.63, "encode\nonce", fs=6.2, ldy=0.045)

    # 2. admission strip (current keep)
    N, NH, NC = 22, 5, 12
    sx, sy, sw, sh = 0.235, 0.575, 0.30, 0.105
    token_strip(ax, sx, sy, sw, sh, N, NH, NC)
    cw = sw / N
    brace(ax, sx, sx + NC * cw - 0.12 * cw, sy + sh,
          "current keep $S^{\\mathrm{cur}}$ (admitted)", ORANGE, above=True)
    brace(ax, sx, sx + NH * cw - 0.12 * cw, sy,
          "history prefix $S^{\\mathrm{hist}}$", "#A04000", above=False)
    ax.text(sx + (NC + (N - NC) / 2) * cw, sy + sh / 2, "never admitted",
            fontsize=5.6, color="#777777", ha="center", va="center",
            rotation=0)
    ax.text(sx + sw / 2, 0.86,
            "one admission-time order per frame",
            fontsize=6.8, ha="center", color=INK, style="italic")

    # 3. retirement arrow: delete rows only
    arrow(ax, sx + sw + 0.008, 0.63, sx + sw + 0.070, 0.63,
          "frame retires:\ndelete rows", fs=6.2, ldy=0.045)

    # 4. session cache after contraction (three frames)
    kx, ky = 0.625, 0.30
    ax.add_patch(patches.FancyBboxPatch(
        (kx - 0.012, ky - 0.045), 0.235, 0.62,
        boxstyle="round,pad=0.010", fc="#FBF4EC", ec="#C8A87E", lw=0.9))
    ax.text(kx + 0.105, ky + 0.525, "session KV cache", fontsize=6.9,
            ha="center", color="#7A5C33", fontweight="bold")
    rows = [("frame $t$ (current)", NC), ("frame $t-1$", NH),
            ("frame $t-2$", NH)]
    for j, (lab, keep) in enumerate(rows):
        yy = ky + 0.335 - j * 0.155
        token_strip(ax, kx, yy, 0.21 * keep / NC, 0.075, keep,
                    NH if j == 0 else keep, keep)
        ax.text(kx, yy - 0.012, lab, fontsize=5.8, color=INK, va="top",
                ha="left")
    ax.text(kx + 0.105, ky - 0.085,
            "history never re-encoded, never re-scored",
            fontsize=6.0, ha="center", color="#7A5C33", style="italic")

    # 5. unknown future demand
    ax.text(0.735, 0.995, "future steps $t{+}1..T$: grounding demand\nunknown at admission time",
            fontsize=6.3, ha="center", va="top", color=RED, style="italic")
    arrow(ax, 0.700, 0.90, 0.685, 0.76, color=RED, lw=0.9)

    # ---------------- right block: why nested ----------------
    rx = 0.882
    ax.add_patch(patches.FancyBboxPatch(
        (rx - 0.010, 0.06), 0.125, 0.87, boxstyle="round,pad=0.010",
        fc="#F4F6F8", ec="#AABBCC", lw=0.8))
    ax.text(rx + 0.052, 0.885, "why one nested order?", fontsize=6.6,
            ha="center", color=INK, fontweight="bold")

    # independent re-selection: some wanted rows are outside the cache
    yA = 0.70
    token_strip(ax, rx, yA, 0.104, 0.070, 8, 0, 5)
    # mark two demanded tokens that fall outside surviving rows
    for i in (5, 7):
        ax.add_patch(patches.Rectangle(
            (rx + i * 0.013 + 0.0008, yA), 0.0114, 0.070, fill=False,
            ec=RED, lw=1.1))
        ax.plot([rx + i * 0.013 + 0.001, rx + (i + 1) * 0.013 - 0.001],
                [yA, yA + 0.070], color=RED, lw=1.0)
    ax.text(rx + 0.052, yA - 0.035, "independent re-selection:\nneeds deleted rows",
            fontsize=5.8, ha="center", va="top", color=RED)
    ax.text(rx + 0.052, yA - 0.135, "unservable on 99%+ frames",
            fontsize=5.8, ha="center", va="top", color=RED,
            fontweight="bold")

    yB = 0.30
    token_strip(ax, rx, yB, 0.104, 0.070, 8, 3, 5)
    ax.text(rx + 0.052, yB - 0.035, "nested prefix:\ndeletion-only, realizable",
            fontsize=5.8, ha="center", va="top", color=GREEN)
    ax.text(rx + 0.052, yB - 0.135, "accuracy cost $\\approx$ 0",
            fontsize=5.8, ha="center", va="top", color=GREEN,
            fontweight="bold")

    out = os.path.join(FIGDIR, "fig1_lifecycle.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + "_preview.png", dpi=220)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
