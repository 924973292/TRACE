#!/usr/bin/env python
"""Figure 17 (fig:commit-timeline): the multi-step token lifecycle.

What changed relative to the previous version of this figure. The old panel (a)
approximated retirement by running the same episode twice at two entry budgets
and overlaying the two keep sets, because the diagnostic dump it read carried
only one keep set per step. The W45 independent-probe dump carries the real
thing per step:

  keep_cur              tokens admitted when the frame is the current screen
  nap_order             the committed admission order (rank) over those tokens
  keep_hist_nested      the tokens that survive the one-shot retirement
                        contraction, exactly nap_order[:k_h]
  keep_hist_independent what an unconstrained re-election would have picked

so every cell of this figure is measured, and per-step "admitted / survived /
retired" is exact rather than inferred. Verified on all 1000 records of the
dump: keep_hist_nested is a subset of keep_cur with zero violations and equals
the rank prefix, and an independent re-election would move 23.5% of the
retirement set.

Writes artifacts/figures/figA6_commit_timeline.pdf (filename preserved so
the .tex include is unchanged).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import keepviz as kv  # noqa: E402
from fig_style import apply, save  # noqa: E402

VIZ = os.path.join(HERE, "..", "logs", "w59_viz_local")
FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
os.makedirs(FIGDIR, exist_ok=True)

EPISODE = os.environ.get("W59_EPISODE", "T0094")
N_STEPS = int(os.environ.get("W59_NSTEPS", "5"))
HIST_N = 5

# Lifecycle state ids used by the cell maps
S_DROP, S_RETIRE, S_SURVIVE, S_ADMIT = 0, 1, 2, 3
STATE_COLORS = {S_RETIRE: kv.LIFECYCLE["retire"],
                S_SURVIVE: kv.LIFECYCLE["survive"],
                S_ADMIT: kv.LIFECYCLE["admit"]}


def load_probe():
    by_ep = {}
    for line in open(os.path.join(VIZ, "probe_subset.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            by_ep.setdefault(
                str(r["episode_key"]).split("-")[-1], []).append(r)
    return by_ep


def img_of(ep, step):
    return Image.open(os.path.join(
        VIZ, "images", f"{ep}_{step:02d}.png")).convert("RGB")


def frame_states(rec, at_step, frame_step):
    """Lifecycle state grid for frame `frame_step` as seen at step `at_step`.

    While a frame is the current screen its whole admitted set is live, so the
    cells split into the ones that will survive retirement and the ones that
    will be dropped by it. Once the frame is history, only the survivors remain.
    """
    gh, gw = rec["grid_h"], rec["grid_w"]
    cur = kv.idx_grid(rec["keep_cur"], gh, gw)
    hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)
    st = np.zeros((gh, gw), dtype=int)
    if at_step == frame_step:
        st[cur & ~hist] = S_ADMIT      # admitted now, will not survive
        st[hist] = S_SURVIVE           # admitted now, will survive retirement
    else:
        st[hist] = S_SURVIVE           # history footprint after contraction
    return st, cur, hist


def retirement_states(rec):
    """The contraction itself: survivors vs evicted, on one frame."""
    gh, gw = rec["grid_h"], rec["grid_w"]
    cur = kv.idx_grid(rec["keep_cur"], gh, gw)
    hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)
    st = np.zeros((gh, gw), dtype=int)
    st[cur & ~hist] = S_RETIRE
    st[hist] = S_SURVIVE
    return st


def panel_steps(fig, cell, recs, ep, steps, led):
    """(a) One column per step: what the current frame admits, split by whether
    each admitted token will survive that frame's later retirement.

    This is the per-step "what was detected, what is new, what will be dropped"
    read. The full frame-by-step cache matrix needs a page of its own and lives
    in the selection gallery.
    """
    gs = cell.subgridspec(1, steps, wspace=0.045)
    for t in range(steps):
        rec = recs[t]
        img = img_of(ep, t)
        st, cur, hist = frame_states(rec, t, t)
        ax = fig.add_subplot(gs[0, t])
        kv.draw_states(ax, img, st, STATE_COLORS, alpha=0.50,
                       ghost=kv.GHOST_SOFT, outline_states=(S_SURVIVE,),
                       outline_lw=0.55)
        tgt = led["steps"][t].get("target_xywh")
        if tgt:
            kv.target_box(ax, tgt, img.size[0], img.size[1],
                          kept=target_covered(rec, tgt, img), lw=1.5)
        ax.set_title(f"Step {t + 1}", fontsize=8.0, pad=2.5)
        n_s, n_r = int((st == S_SURVIVE).sum()), int((st == S_ADMIT).sum())
        ax.set_xlabel(f"admits {n_s + n_r}\n{n_s} survive, {n_r} retire",
                      fontsize=6.8, labelpad=1.8, color="#333333")


def target_covered(rec, bbox, img):
    """Does any token that SURVIVES retirement cover the action target?"""
    W, H = img.size
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    gh, gw = rec["grid_h"], rec["grid_w"]
    cw, ch = W / gw, H / gh
    for j in rec["keep_hist_nested"]:
        cx, cy = (j % gw + 0.5) * cw, (j // gw + 0.5) * ch
        if x <= cx <= x + bw and y <= cy <= y + bh:
            return True
    return False


def panel_matrix(fig, cell, recs, ep, steps, compact=True):
    """The cache x step lifecycle matrix.

    Row f is the frame admitted at step f, column t is the state of the cache at
    step t. On the diagonal the frame is the current screen at the entry budget,
    below the diagonal it has retired to its nested subset, and above the
    diagonal it does not exist yet.
    """
    gs = cell.subgridspec(steps, steps, wspace=0.05, hspace=0.16)
    for f in range(steps):
        for t in range(steps):
            if t < f:
                continue
            ax = fig.add_subplot(gs[f, t])
            rec = recs[f]
            img = img_of(ep, f)
            st, cur, hist = frame_states(rec, t, f)
            in_window = (t - f) <= HIST_N
            if not in_window:
                kv.show_image(ax, img)
                ax.imshow(np.full((2, 2, 4), (0.30, 0.30, 0.30, 0.80)),
                          extent=(0, img.size[0], img.size[1], 0), zorder=3)
                kv.panel_tag(ax, "left cache", fontsize=5.4)
            else:
                kv.draw_states(ax, img, st, STATE_COLORS, alpha=0.52,
                               ghost=kv.GHOST_SOFT,
                               outline_states=(S_SURVIVE,), outline_lw=0.4)
            if t == f:
                for s in ax.spines.values():
                    s.set_linewidth(1.6)
                    s.set_edgecolor(kv.LIFECYCLE["admit"])
                ax.set_xlabel(f"admit {int(cur.sum())}", fontsize=6.2,
                              labelpad=1.2, color="#8A5A00")
            elif in_window:
                ax.set_xlabel(f"hold {int(hist.sum())}", fontsize=6.2,
                              labelpad=1.2, color="#9A4A00")
            if f == 0:
                ax.set_title(f"step {t + 1}", fontsize=7.6, pad=2.5)
            if t == f:
                ax.text(-0.14, 0.5, f"frame {f + 1}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=7.0,
                        fontweight="bold")


def panel_anatomy(fig, cell, rec, ep, step):
    """(b) One retirement, dissected: input, admitted, survived, evicted."""
    gs = cell.subgridspec(1, 4, wspace=0.05)
    img = img_of(ep, step)
    w, h = img.size
    gh, gw = rec["grid_h"], rec["grid_w"]
    cur = kv.idx_grid(rec["keep_cur"], gh, gw)
    hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)

    ax = fig.add_subplot(gs[0, 0])
    kv.show_image(ax, img)
    ax.set_title("input frame", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"{gh}$\\times${gw} $=$ {gh * gw} tokens", fontsize=6.6,
                  labelpad=1.5)

    ax = fig.add_subplot(gs[0, 1])
    kv.draw_keep(ax, img, cur, tint=kv.LIFECYCLE["admit"], tint_alpha=0.26,
                 ghost=kv.GHOST_SOFT, outline_lw=0.6)
    ax.set_title("admitted at entry", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"$k_c{{=}}{int(cur.sum())}$ "
                  f"({100.0 * cur.sum() / cur.size:.0f}\\%)", fontsize=6.6,
                  labelpad=1.5)

    ax = fig.add_subplot(gs[0, 2])
    kv.draw_keep(ax, img, hist, tint=kv.LIFECYCLE["survive"], tint_alpha=0.30,
                 ghost=kv.GHOST_SOFT, outline_lw=0.6)
    ax.set_title("survives retirement", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"$k_h{{=}}{int(hist.sum())}$ "
                  f"({100.0 * hist.sum() / hist.size:.0f}\\%)", fontsize=6.6,
                  labelpad=1.5)

    ax = fig.add_subplot(gs[0, 3])
    kv.draw_keep(ax, img, cur & ~hist, tint=kv.LIFECYCLE["retire"],
                 tint_alpha=0.30, ghost=kv.GHOST_SOFT, outline_lw=0.6)
    ax.set_title("evicted by retirement", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"$k_c-k_h{{=}}{int((cur & ~hist).sum())}$ tokens freed",
                  fontsize=6.6, labelpad=1.5)


def panel_rank(fig, cell, stats):
    """(c) Survival as a function of position in the committed admission order.

    The contract predicts a single step at k_h/k_c: everything above that rank
    survives retirement, everything below is freed, and nothing is re-scored.
    Plotting the measured curve makes the contract visible instead of asking the
    reader to trust the claim.
    """
    ax = fig.add_subplot(cell)
    curve = stats["rank_survival_curve"]
    nb = len(curve)
    xs = (np.arange(nb) + 0.5) * 100.0 / nb
    ys = [100.0 * v for v in curve]
    ax.fill_between(xs, 0, ys, step="mid", color=kv.LIFECYCLE["survive"],
                    alpha=0.30, lw=0)
    ax.step(xs, ys, where="mid", color=kv.LIFECYCLE["survive"], lw=1.7)
    thresh = 100.0 * float(stats["mean_kh_over_kc"])
    ax.axvline(thresh, color="#333333", ls="--", lw=0.9)
    ax.text(thresh + 2.5, 52, f"$k_h/k_c={thresh:.0f}\\%$", fontsize=6.8,
            color="#333333")
    ax.annotate("survive", (thresh * 0.45, 104), fontsize=6.8, ha="center",
                va="top", color=kv.LIFECYCLE["survive"], fontweight="bold")
    ax.annotate("freed at retirement", ((100 + thresh) / 2, 12), fontsize=6.8,
                ha="center", color=kv.LIFECYCLE["retire"])
    ax.set_xlabel("Position in the committed admission order (\\%)",
                  fontsize=7.2, labelpad=1.5)
    ax.set_ylabel("Survives retirement (\\%)", fontsize=7.2, labelpad=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=6.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.30, "Retirement is exactly the rank\nprefix, so no token "
            "is re-scored\nor re-admitted.", transform=ax.transAxes,
            fontsize=6.8, style="italic", ha="right", va="bottom",
            color="#333333")


def panel_budget(fig, cell, recs, steps, stats):
    """(d) Per-step cache accounting: what the matrix above adds up to."""
    ax = fig.add_subplot(cell)
    xs = np.arange(1, steps + 1)
    admitted, held, freed = [], [], []
    for t in range(steps):
        rec = recs[t]
        gh, gw = rec["grid_h"], rec["grid_w"]
        n = gh * gw
        k_c, k_h = len(rec["keep_cur"]), len(rec["keep_hist_nested"])
        admitted.append(100.0 * k_c / n)
        # history frames still in the window, each contracted to k_h
        n_hist = min(t, HIST_N)
        held.append(100.0 * n_hist * k_h / n)
        freed.append(100.0 * (k_c - k_h) / n)
    ax.bar(xs, admitted, 0.62, color=kv.LIFECYCLE["admit"], alpha=0.92,
           label="Current frame admitted")
    ax.bar(xs, held, 0.62, bottom=admitted, color=kv.LIFECYCLE["survive"],
           alpha=0.92, label="History survivors held")
    for x, a, hh in zip(xs, admitted, held):
        ax.text(x, a + hh + 2.0, f"{a + hh:.0f}", ha="center", fontsize=6.4,
                color="#333333")
    ax.set_xticks(list(xs))
    ax.set_xlabel("Step", fontsize=7.2, labelpad=1.5)
    ax.set_ylabel("Cache size (\\% of one frame)", fontsize=7.2, labelpad=2)
    ax.tick_params(labelsize=6.6)
    ax.legend(fontsize=6.4, loc="upper left", frameon=False, handlelength=1.1,
              borderaxespad=0.1)
    ax.set_ylim(0, max(a + h for a, h in zip(admitted, held)) * 1.42)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.97, 0.05,
            f"{stats['nesting_violations']} nesting violations and "
            f"{stats['readmissions']} re-admissions\nover "
            f"{stats['n_records']} steps of {stats['n_episodes']} episodes.",
            transform=ax.transAxes, fontsize=6.8, style="italic", ha="right",
            va="bottom", color="#2A6041")


def main():
    apply(scale=8.6 / 5.5)
    kv.enable_cjk()
    by_ep = load_probe()
    recs = by_ep[EPISODE]
    steps = min(N_STEPS, len(recs))
    stats = json.load(open(os.path.join(VIZ, "lifecycle_stats.json"),
                           encoding="utf-8"))
    led = json.load(open(os.path.join(VIZ, "episodes.json"),
                         encoding="utf-8"))[EPISODE]

    im0 = img_of(EPISODE, 0)
    ar = im0.size[1] / im0.size[0]
    # Designed at 8.6in and printed at 5.5in, so panels land larger on the page
    # than a 1:1 design would allow while text stays at its intended size.
    W_FIG = 8.6
    PAD_T, PAD_B = 0.42, 0.24
    row_h = (W_FIG - 0.34) / steps * ar + 0.62       # (a) step row
    ana_h = (W_FIG - 0.34) / 4.0 * ar + 0.70         # (b) anatomy row
    bot_h = 2.05                                     # (c)(d) charts
    tot = PAD_T + row_h + ana_h + bot_h + PAD_B

    fig = plt.figure(figsize=(W_FIG, tot))
    gs = fig.add_gridspec(3, 1, height_ratios=[row_h, ana_h, bot_h],
                          hspace=0.34, top=1 - PAD_T / tot,
                          bottom=PAD_B / tot, left=0.072, right=0.986)

    y_a = 1 - 0.05 / tot
    fig.text(0.008, y_a, "(a) Measured per-step admission on one episode: "
             "what each step selects, and how much of it will survive",
             fontsize=9.0, fontweight="bold", va="top")
    fig.text(0.008, y_a - 0.150 / tot,
             "Bright cells survive this frame's retirement, mid cells are "
             "admitted now but freed at retirement, dimmed cells are never "
             "selected.", fontsize=6.9, va="top", color="#333333")
    panel_steps(fig, gs[0], recs, EPISODE, steps, led)

    y_b = 1 - (PAD_T + row_h) / tot
    fig.text(0.008, y_b, "(b) One retirement dissected: the contraction keeps "
             "the rank prefix and frees the rest", fontsize=9.0,
             fontweight="bold", va="top")
    panel_anatomy(fig, gs[1], recs[0], EPISODE, 0)

    bot = gs[2].subgridspec(1, 2, wspace=0.30)
    y_c = 1 - (PAD_T + row_h + ana_h) / tot
    fig.text(0.008, y_c, "(c) Rank prefix over all episodes", fontsize=9.0,
             fontweight="bold", va="top")
    fig.text(0.545, y_c, "(d) Per-step cache accounting", fontsize=9.0,
             fontweight="bold", va="top")
    panel_rank(fig, bot[0, 0], stats)
    panel_budget(fig, bot[0, 1], recs, steps, stats)

    fig.text(0.008, 0.002,
             f"OmniGUI episode {EPISODE} ({led['app']}), "
             f"$r_c{{=}}50\\%$ entry and $r_h{{=}}10\\%$ retirement, "
             f"history window $H{{=}}{HIST_N}$. Panels (c) and (d) aggregate "
             f"{stats['n_records']} steps over {stats['n_episodes']} episodes.",
             fontsize=6.6, style="italic", color="#555555", va="bottom")
    save(fig, os.path.join(FIGDIR, "figA6_commit_timeline.pdf"))


if __name__ == "__main__":
    main()
