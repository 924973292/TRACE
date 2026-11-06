#!/usr/bin/env python3
"""Render the script-drawn figures (Fig.3/4 + appendix A1/A2/A3/A4/A5).

All table-backed numbers are imported from gen_tables.py -- no manual
transcription. Hand-anchored series (alpha ladder, dose curves, prior-value
lineage) carry provenance comments pointing at their table/worklog source.
Every figure embeds its claim as an in-panel takeaway; deltas/gaps are
annotated directly on the data. Output: vector PDFs in artifacts/figures/.
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import gen_tables as G  # noqa: E402
from fig_style import (C, MARKERS, METHOD_STYLE, apply, despine,  # noqa: E402
                       gap_label, panel_label, save, takeaway)

FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def clean(name):
    """LaTeX method name -> plain legend label."""
    name = re.sub(r"\\pub\{[^}]*\}", "", name)
    name = re.sub(r"\\inllm\{([^}]*)\}", r"\1", name)
    name = name.replace(r"\method{}", "TRACE")
    name = re.sub(r"\$[^$]*\$", "", name)
    name = name.replace("{", "").replace("}", "")
    return name.strip()


def val(rows, name_pred, idx):
    for n, k, v in rows:
        if name_pred(n, k):
            x = v[idx]
            return float(x) if G.fnum(x) else None
    return None


def best_published(rows, idx):
    """Max over published baselines (excl. Random/Uniform/ours/gray)."""
    best, who = None, None
    for n, k, v in rows:
        if k != "n" or n in ("Random", "Uniform"):
            continue
        if not G.fnum(v[idx]):
            continue
        x = float(v[idx])
        if best is None or x > best:
            best, who = x, clean(n)
    return best, who


# ---------------------------------------------------------------- Fig. 3
def fig3():
    """Budget-accuracy curves. Advantage band = ours minus the pointwise
    best published baseline; every gap at the two tightest budgets is
    annotated (grey when we honestly trail)."""
    fig, axes = plt.subplots(
        2, 3, figsize=(7.05, 3.95),
        gridspec_kw={"height_ratios": [1.0, 0.74]})

    def curve(ax, xs, ours, best, unif, dense, title, lab, tick_suffix="%"):
        xi = list(range(len(xs)))
        ax.axhline(dense, color=C["dense"], ls=":", lw=1.0, label="Dense")
        ax.text(len(xs) - 0.55, dense, f"{dense:.1f}", fontsize=6,
                color="#555555", va="bottom", ha="right")
        ax.plot(xi, unif, color=C["uniform"], ls="--",
                marker=MARKERS["uniform"], ms=3.5, label="Uniform")
        ax.plot(xi, best, color=C["best"], marker=MARKERS["best"], ms=3.5,
                label="Best published (pointwise)")
        ax.plot(xi, ours, color=C["ours"], marker=MARKERS["ours"], ms=7,
                label="TRACE", zorder=5)
        # advantage band: shade only where we lead
        o, b = np.array(ours, float), np.array(best, float)
        ax.fill_between(xi, b, o, where=o >= b, color=C["band"], alpha=0.10,
                        interpolate=True, zorder=1)
        allv = [v for v in ours + best + unif + [dense] if v is not None]
        span = max(allv) - min(allv)
        # annotate tightest budgets + concessions (skip the loosest budget,
        # where sub-point differences just clutter the dense-line corner)
        js = sorted(set([len(xs) - 2, len(xs) - 1]
                        + [j for j in range(1, len(xs))
                           if ours[j] < best[j]]))
        for j in js:
            g = ours[j] - best[j]
            col = C["ours"] if g >= 0 else C["concede"]
            lo, hi = min(ours[j], best[j]), max(ours[j], best[j])
            if hi - lo < 0.16 * span:
                # near-overlapping markers: keep the delta clear of both
                # curves -- right of the final point, above interior points
                if j == len(xs) - 1:
                    ax.text(xi[j] + 0.09, (lo + hi) / 2, f"{g:+.1f}",
                            fontsize=6.4, color=col, fontweight="bold",
                            ha="left", va="center")
                else:
                    ax.text(xi[j] + 0.06, hi + 0.05 * span, f"{g:+.1f}",
                            fontsize=6.4, color=col, fontweight="bold",
                            ha="left", va="bottom")
            else:
                gap_label(ax, xi[j], lo, hi, col,
                          dx=0.07 if j < len(xs) - 1 else -0.42,
                          text=f"{g:+.1f}")
        ax.margins(y=0.09)
        ax.set_xticks(xi)
        ax.set_xticklabels([f"{x}{tick_suffix}" for x in xs])
        ax.set_xlim(-0.35, len(xs) - 0.6)
        panel_label(ax, f"{lab} {title}", fontsize=8.5)
        despine(ax)
        return ours, best

    # ---- top row: single-step, x = keep ratio ----
    rs = ["50", "25", "10", "5"]
    ss_rows = {r: G._ssv2_block(r) for r in rs}
    sp_rows = {r: G._sspro_block(r) for r in rs}
    is_ours = lambda n, k: k == "ours"
    is_unif = lambda n, k: n == "Uniform"

    specs = [
        (axes[0][0], "ScreenSpot-v2", ss_rows, 6, float(G.SS_MAIN_UB[0])),
        (axes[0][1], "ScreenSpot-Pro", sp_rows, 12, float(G.SS_MAIN_UB[1])),
        (axes[0][2], "MMBench-GUI L2", ss_rows, 7, float(G.SS_MAIN_UB[2])),
    ]
    for i, (ax, title, rows_by_r, idx, dense) in enumerate(specs):
        ours = [val(rows_by_r[r], is_ours, idx) for r in rs]
        unif = [val(rows_by_r[r], is_unif, idx) for r in rs]
        bests = [best_published(rows_by_r[r], idx) for r in rs]
        best = [b for b, _ in bests]
        curve(ax, rs, ours, best, unif, dense, title, f"({chr(97 + i)})")
        # collapse marker: first budget where even the best published
        # baseline falls below uniform stripes (coverage collapse, cf. Fig.1c)
        for j in range(len(rs)):
            if best[j] is not None and unif[j] is not None and best[j] < unif[j]:
                ax.plot(j, best[j], "x", color="#B3261E", ms=7, mew=1.6,
                        zorder=6)
                ax.annotate("best published\ndrops below uniform",
                            (j, best[j]),
                            xytext=(j - 1.85, best[j] - 3.0), fontsize=5.4,
                            color="#B3261E",
                            arrowprops=dict(arrowstyle="->", color="#B3261E",
                                            lw=0.6))
                break
        g25, g5 = ours[1] - best[1], ours[3] - best[3]
        if g25 >= 0:
            takeaway(ax, f"lead grows {g25:+.1f}$\\to${g5:+.1f} pts"
                     "\nas the budget tightens")
        else:
            takeaway(ax, f"trails {bests[1][1]} by {-g25:.1f} at 25%,"
                     f"\nreclaims {g5:+.1f} at 5%")
        ax.set_ylabel("Accuracy (%)" if i == 0 else "")
        ax.set_xlabel(r"Keep ratio $r$  ($\rightarrow$ tighter)")

    # ---- bottom row: multi-step, x = budget tier ----
    tiers = ["I", "II"]
    m_specs = [
        (axes[1][0], "OmniGUI", lambda t: G.OMNI[t], 3),
        (axes[1][1], "Mind2Web (pooled)", None, 2),
        (axes[1][2], "AndroidControl", lambda t: G.AC[t], 2),
    ]
    for i, (ax, title, getrows, idx) in enumerate(m_specs):
        if getrows is None:  # pooled M2W via the table's own pooling
            names = [n for n, k, _ in G.M2W["task"]["I"]]
            kinds = {n: k for n, k, _ in G.M2W["task"]["I"]}
            ours, unif, best = [], [], []
            for t in tiers:
                per = {n: G._m2w_pooled(t, n) for n in names}
                per = {n: float(v[idx]) for n, v in per.items() if v}
                ours.append(per.get(r"\method{}"))
                unif.append(per.get("Uniform"))
                pubs = [v for n, v in per.items()
                        if kinds.get(n) == "n" and n not in ("Random", "Uniform")]
                best.append(max(pubs))
            dref = G._m2w_pooled("I", r"No-prune KV ref")
            dense = float(dref[idx]) if dref else None
        else:
            ours = [val(getrows(t), is_ours, idx) for t in tiers]
            unif = [val(getrows(t), is_unif, idx) for t in tiers]
            best = [best_published(getrows(t), idx)[0] for t in tiers]
            dense = val(getrows("I"),
                        lambda n, k: n == r"No-prune KV ref", idx)
        curve(ax, ["50/10", "25/5"], ours, best, unif, dense, title,
              f"({chr(100 + i)})", tick_suffix="")
        # claim line in the otherwise-empty upper-right corner
        ax.text(0.97, 0.78, "margin widens\n"
                f"{ours[0] - best[0]:+.1f} $\\to$ {ours[1] - best[1]:+.1f}",
                transform=ax.transAxes, fontsize=6.4, style="italic",
                color="#333333", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#CCCCCC", lw=0.5, alpha=0.85))
        ax.set_xlabel("Budget (current/history %)")
        ax.set_ylabel("Step SR (%)" if i == 0 else "")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=1.2)
    save(fig, os.path.join(FIGDIR, "fig3_budget_curves.pdf"))


# ---------------------------------------------------------------- Fig. 4
# Anatomy-probe waterfall (appendix tables/efficiency.tex, OmniGUI mild
# block): W57b round-2 pipelined-path probe (logs/w57b_anatomy): dense TTFT
# 1116.8 -> session-KV reuse 487.4 -> + full selection stack 452.7 (the
# selector nets -34.7 ms at mild; -89.6 ms at tight, cited in prose).
# Reproduces the unified probe's dense anchor (1102.6) within 1.3%.
ANATOMY = [("Stateless\nre-prefill", 1116.8, None),
           ("Session-KV\nreuse", 487.4, -629.4),
           ("+ selection\nstack", 452.7, -34.7)]


def fig4():
    """(a) Efficiency Pareto with a broken TTFT axis (no dead band between
    the fast cluster and the dense/PruneSID outliers); (b) TTFT waterfall
    from the anatomy probe + native-selector reference strip. Data:
    G.EFF_ROWS (unified probe) and tables/efficiency.tex (anatomy).

    Panel (a) stays clean: name only TRACE / Dense / PruneSID, one orange
    claim in empty sky, and an in-axes marker legend (incl. hollow=ported).
    Path/order prose lives in the LaTeX caption, not on the plot.
    """
    # Compensate \\includegraphics shrink: 7.05in design → ~5.5in column.
    S = 1.15
    apply(scale=S)
    # Extra bottom room so panel-(a) legend sits under the axes, not on data.
    fig = plt.figure(figsize=(7.05, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.55, 2.60],
                          left=0.065, right=0.985, top=0.925, bottom=0.24,
                          wspace=0.22)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.045,
                                width_ratios=[2.55, 0.72])
    axl = fig.add_subplot(gs_a[0, 0])
    axr = fig.add_subplot(gs_a[0, 1], sharey=axl)
    gs_b = gs[0, 1].subgridspec(1, 2, wspace=0.56,
                                width_ratios=[1.35, 0.95])
    axw = fig.add_subplot(gs_b[0, 0])
    axs = fig.add_subplot(gs_b[0, 1])

    pts = []
    for name, kind, tok, gf, sel, ttft, e2e, kv, sr in G.EFF_ROWS:
        pts.append((clean(name) if kind != "gray" else "Dense", kind,
                    float(ttft), float(sr), float(kv), float(sel),
                    float(e2e)))

    # W42 / R3-N1: transplanted committable selectors riding our MKC
    # path (hollow markers). Step SR = full 708-episode xds8b_abl pubsel
    # arms; TTFT/KV = W57 anchor-queue per-step means (logs/w57_lat2f).
    PORTED = [
        ("VisPruner", 427.2, 46.89, 292.0),
        ("DivPrune", 429.4, 46.27, 291.9),
        ("ZOO-Prune", 517.0, 47.51, 291.9),
    ]

    # Fallback identities for methods not yet in METHOD_STYLE.
    STYLE_EXTRA = {
        "PruMerge+": {"color": "#0072B2", "marker": "o"},
        "TRIM": {"color": "#332288", "marker": "o"},
        "VisionTrim": {"color": "#88CCEE", "marker": "o"},
        "ZOO-Prune": {"color": "#44AA99", "marker": "o"},
    }

    def mstyle(name):
        return METHOD_STYLE.get(name) or STYLE_EXTRA.get(
            name, {"color": C["best"], "marker": "o"})

    # ---------- (a) Pareto bubble on a broken x axis ----------
    # TRACE/Dense: bubble ∝ visual-KV. Natives share ~292 MB → equal markers.
    BUB = 0.28
    XL = (395, 1005)   # fast cluster
    XR = (1035, 1585)  # dense + PruneSID outliers
    dense_x, dense_y, dense_kv = next(
        (x, y, kv) for n, k, x, y, kv, *_ in pts if k == "gray")
    ours_x, ours_y = next((x, y) for n, k, x, y, *_ in pts if k == "ours")

    def seg_ax(x):
        return axl if x <= XL[1] else axr

    legend_names = []
    for name, kind, x, y, kv, sel, e2e in pts:
        ax = seg_ax(x)
        st = mstyle(name)
        if kind == "ours":
            ax.scatter(x, y, s=kv * BUB, marker="*", color=C["ours"],
                       zorder=6, edgecolor="white", linewidth=0.45)
        elif kind == "gray":
            ax.scatter(x, y, s=kv * BUB, color="#BBBBBB", zorder=3,
                       edgecolor="white", linewidth=0.3)
        else:
            ax.scatter(x, y, s=48, color=st["color"], alpha=0.92,
                       marker=st["marker"], zorder=3, edgecolor="white",
                       linewidth=0.4)
            if name == "PruneSID":
                ax.annotate("PruneSID", (x, y),
                            xytext=(-8, -12), textcoords="offset points",
                            fontsize=7.0 * S, color=C["concede"], ha="right",
                            arrowprops=dict(arrowstyle="-",
                                            color=C["concede"], lw=0.55,
                                            shrinkA=0, shrinkB=2))
            else:
                legend_names.append(name)

    for name, x, y, kv in PORTED:
        st = mstyle(name)
        axl.scatter(x, y, s=44, marker=st["marker"], facecolor="none",
                    edgecolor=st["color"], linewidth=1.1, zorder=4)

    # Pareto frontier behind markers (no callout boxes over it).
    cand = [(p[2], p[3]) for p in pts] + [(x, y) for _, x, y, _ in PORTED]
    front = []
    for x, y in sorted(cand):
        if not front or y > front[-1][1]:
            front.append((x, y))
    fx, fy = zip(*front)
    for ax in (axl, axr):
        ax.plot(fx, fy, ls="--", lw=0.95, color=C["ours"], alpha=0.40,
                zorder=1)

    # Point names only — offset into empty sky / away from the frontier.
    axl.annotate("TRACE", (ours_x, ours_y),
                 xytext=(8, 10), textcoords="offset points",
                 fontsize=8.0 * S, fontweight="bold", color=C["ours"],
                 ha="left", va="bottom")
    axr.annotate(f"Dense\n{dense_kv:.0f} MB KV", (dense_x, dense_y),
                 xytext=(6, -16), textcoords="offset points",
                 fontsize=6.8 * S, color="#555555", ha="left", va="top")

    # One claim in empty upper-left sky (above every marker). No box.
    axl.text(0.02, 0.98,
             "$-$59% TTFT, $-$58% KV\nonly $-$3.5 Step SR vs dense",
             transform=axl.transAxes, fontsize=7.0 * S, fontweight="bold",
             color=C["ours"], ha="left", va="top", zorder=7)

    # Legend = markers only, parked under panel (a) so nothing covers data.
    seen = []
    handles = []
    for name in sorted(legend_names, key=str.lower):
        if name in seen:
            continue
        seen.append(name)
        st = mstyle(name)
        handles.append(Line2D([0], [0], marker=st["marker"], color="none",
                              markerfacecolor=st["color"],
                              markeredgecolor="white", markeredgewidth=0.35,
                              markersize=6.2, label=name))
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor="none", markeredgecolor="#555555",
                          markeredgewidth=1.1, markersize=6.2,
                          label="Hollow = ported"))
    leg = axl.legend(handles=handles, loc="upper center", fontsize=7.4 * S,
                     ncol=3, frameon=False, handletextpad=0.32,
                     columnspacing=1.05, labelspacing=0.24, borderpad=0.20,
                     handlelength=1.25, bbox_to_anchor=(0.62, -0.20))
    leg.set_zorder(8)

    # axis cosmetics + break marks
    axl.set_xlim(*XL)
    axr.set_xlim(*XR)
    axl.set_ylim(40.0, 54.2)
    axl.spines["top"].set_visible(False)
    axl.spines["right"].set_visible(False)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.tick_params(left=False, labelleft=False)
    axr.set_xticks([1200, 1500])
    for ax in (axl, axr):
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
    axl.plot([1 - 0.004, 1 + 0.004], [-0.012, 0.012],
             transform=axl.transAxes, color="k", clip_on=False, lw=0.8)
    axr.plot([-0.013, 0.013], [-0.012, 0.012],
             transform=axr.transAxes, color="k", clip_on=False, lw=0.8)
    axl.set_xlabel("TTFT per step (ms)  ($\\leftarrow$ faster)"
                   "   ·   bubble $\\propto$ visual-KV", x=0.62)
    axl.set_ylabel("OmniGUI Step SR (%) @ mild budget")
    panel_label(axl, "(a) Accuracy-TTFT trade-off", fontsize=9)

    # ---------- (b) waterfall: where the speed comes from ----------
    prev = None
    for i, (lab, level, delta) in enumerate(ANATOMY):
        if delta is None:
            axw.bar(i, level, 0.58, color="#BBBBBB", zorder=3)
            axw.text(i, level + 22, f"{level:.0f}", ha="center",
                     fontsize=6.8, fontweight="bold", color="#555555")
        else:
            lo, hi = min(prev, level), max(prev, level)
            colr = C["best"] if delta < 0 else C["witness"]
            axw.bar(i, hi - lo, 0.58, bottom=lo, color=colr, zorder=3)
            axw.plot([i - 1 + 0.29, i - 0.29], [prev, prev], color="#888888",
                     lw=0.8, ls=":", zorder=2)
            axw.text(i, hi + 22, f"{delta:+.0f}", ha="center", fontsize=6.8,
                     fontweight="bold",
                     color=C["best"] if delta < 0 else "#8A6D00")
        prev = level
    axw.text(len(ANATOMY) - 1, ANATOMY[-1][1] - 60,
             f"{ANATOMY[-1][1]:.0f}", ha="center", va="top", fontsize=6.8,
             fontweight="bold", color="#333333")
    # net-change bracket in the empty area above the final bar
    axw.set_xlim(-0.55, 2.60)
    axw.annotate("", (2.46, 452.7), (2.46, 1116.8),
                 arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.7))
    axw.text(1.95, 940, "Net $-$664 ms\n(2.5$\\times$ faster)",
             fontsize=7.0 * S, color="#333333", ha="center", va="center",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                       alpha=0.9))
    # Stack breakdown lives in the free zone below the blue reuse bar,
    # clear of the -35 delta and the 453 value label.
    axw.text(1.60, 330, "Selector on path $+$67.4\n"
             "Prefill $-$188.0\nThe stack pays for itself",
             fontsize=6.8 * S, color="#333333", ha="center", va="top")
    axw.set_xticks(range(len(ANATOMY)))
    axw.set_xticklabels([a[0] for a in ANATOMY], fontsize=6.4)
    axw.set_ylabel("TTFT (ms), anatomy probe", fontsize=7.4)
    axw.set_ylim(0, 1210)
    panel_label(axw, "(b) Latency attribution", fontsize=9)
    despine(axw)

    # native selectors cannot reach the reuse regime: reference strip
    natives = sorted([p for p in pts if p[1] == "n"], key=lambda p: p[2])
    ys = np.arange(len(natives))
    for yy, (name, kind, ttft, sr, kv, sel, e2e) in zip(ys, natives):
        st = METHOD_STYLE.get(name, {"color": C["best"], "marker": "o"})
        axs.scatter(ttft, yy, s=16, color=st["color"],
                    marker=st["marker"], zorder=4)
    axs.axvline(453.9, color=C["ours"], lw=1.3, zorder=3)
    axs.text(453.9 + 24, 8.45, "TRACE\n453.9", fontsize=6.2,
             color=C["ours"], ha="left", va="top", fontweight="bold")
    axs.axvline(dense_x, color="#999999", ls=":", lw=0.8)
    axs.text(dense_x - 20, 0.0, "dense", fontsize=5.8, color="#777777",
             rotation=90, ha="right", va="bottom")
    axs.set_yticks(ys)
    axs.set_yticklabels([p[0] for p in natives], fontsize=6.2)
    axs.set_xlim(380, 1590)
    axs.set_xticks([500, 1000, 1500])
    axs.set_xlabel("native TTFT (ms)", fontsize=7.0)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.grid(axis="x", alpha=0.7)
    axs.set_axisbelow(True)
    axs.tick_params(axis="y", length=0)

    save(fig, os.path.join(FIGDIR, "fig7_pareto.pdf"))


# ---------------------------------------------------------------- Fig. 7
def fig7():
    """E2 certificate-dose response on SSv2 (full 1,272 docs per point).
    Source: logs/xds8b_tune/screenspot_v2/covdose{20,30,50,70,100}_r{05,10}
    + tab:ablation-main A+B rows as the zero-dose anchor; the rho_cov=1
    endpoint is definitionally the uniform stride grid. Variant markers:
    logs/xds8b_abl2/screenspot_v2/{stridefull,hilbert}*."""
    xs = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    r10 = [56.37, 69.34, 72.33, 73.90, 74.06, 43.79]
    r05 = [36.95, 51.57, 52.44, 51.97, 50.63, 21.38]
    ad10, ad05 = 3, 2          # adopted-dose index per budget
    unif10, unif05 = 43.79, 21.38
    var10 = [(72.72, "full-frame stride"), (75.39, "Hilbert")]
    var05 = [(53.93, "full-frame stride"), (53.30, "Hilbert")]

    fig, ax = plt.subplots(figsize=(4.15, 2.75))
    for y_u, x_lab, lab in (
        (unif10, 0.33, r"uniform, $\kappa{=}10\%$"),
        (unif05, 0.60, r"uniform, $\kappa{=}5\%$"),
    ):
        ax.axhline(y_u, color=C["uniform"], ls=":", lw=0.9)
        ax.text(x_lab, y_u + 0.7, lab, fontsize=5.8, color="#777777")
    for ys, adopt, col, lab in (
        (r10, ad10, C["ours"], r"$\kappa{=}10\%$"),
        (r05, ad05, C["best"], r"$\kappa{=}5\%$"),
    ):
        ax.plot(xs, ys, color=col, marker="o", ms=3.6, lw=1.3, label=lab,
                zorder=4)
        ax.plot(xs[adopt], ys[adopt], marker="*", ms=13, color=col,
                mec="white", mew=0.4, zorder=6)
    # spacing-parameterization variants at the adopted dose
    for (adopt, vs, col) in ((ad10, var10, C["ours"]), (ad05, var05, C["best"])):
        for v, _name in vs:
            ax.plot(xs[adopt], v, marker="D", ms=3.2, mfc="none", mec=col,
                    mew=0.8, ls="none", zorder=5)
    ax.plot([], [], marker="D", ms=3.2, mfc="none", mec="#555555", ls="none",
            label="stride-full / Hilbert variant")
    ax.plot([], [], marker="*", ms=9, color="#555555", ls="none",
            label="adopted dose")
    # annotate the two mechanisms at the ends
    ax.annotate("greedy alone:\nblind spots", (0.0, r10[0]),
                (0.045, 62.2), fontsize=6.2, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    ax.annotate("full dose $\\equiv$ uniform:\nquality signal gone",
                (1.0, r10[-1]), (0.60, 33.5), fontsize=6.2, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    takeaway(ax, "certificates repair the selector's blind spots,\n"
             "then a plateau ($\\pm$0.9 over 0.3--0.7) before collapse",
             loc="upper right", fontsize=6.0)
    ax.set_xlabel(r"coverage dose $\rho_{\mathrm{cov}}$ "
                  "(fraction of budget re-spent on repair)")
    ax.set_ylabel("ScreenSpot-v2 center acc (%)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(15, 84)
    ax.legend(fontsize=6.0, loc="lower left", ncol=2, columnspacing=0.9,
              handletextpad=0.5)
    despine(ax)
    save(fig, os.path.join(FIGDIR, "fig7_dose.pdf"))


# ---------------------------------------------------------------- Fig. A9
def figA9():
    """Robustness knobs: (a) E5a detector-degradation curve (random MIU
    dropout, SSv2 r=10, logs/xds8b_abl2/screenspot_v2/dropout*_r10);
    (b) Minor-10 instruction-likelihood temperature sweep (tau*_r10)."""
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.4, 2.45))

    # (a) detection dropout
    xs = [0, 25, 50, 75, 100]
    ys = [73.90, 68.55, 64.15, 55.82, 53.85]
    ax.axhline(43.79, color=C["uniform"], ls=":", lw=0.9)
    ax.text(2, 44.6, "uniform (no prior, no certificate)", fontsize=5.8,
            color="#777777")
    ax.plot(xs, ys, color=C["ours"], marker="o", ms=4, lw=1.3, zorder=4)
    ax.plot(xs[0], ys[0], marker="*", ms=12, color=C["ours"], mec="white",
            mew=0.4, zorder=6)
    ax.annotate(f"zero detections:\nstill +{ys[-1] - 43.79:.1f} over uniform",
                (100, ys[-1]), (48, 48.6), fontsize=6.2, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    takeaway(ax, "graceful: certificates + selector floor the\n"
             "damage even when LIP is fully lost",
             loc="upper right", fontsize=5.9)
    ax.set_xlabel("detected elements dropped (%)")
    ax.set_ylabel("SSv2 center acc (%), $\\kappa{=}10\\%$")
    ax.set_ylim(40, 80)
    panel_label(ax, "(a) Detector degradation")
    despine(ax)

    # (b) temperature sweep
    ts = [0.5, 1.0, 2.0, 4.0]
    ys_t = [63.36, 73.90, 72.96, 66.82]
    axb.plot(range(len(ts)), ys_t, color=C["ours"], marker="o", ms=4, lw=1.3,
             zorder=4)
    axb.plot(1, ys_t[1], marker="*", ms=12, color=C["ours"], mec="white",
             mew=0.4, zorder=6)
    axb.annotate("Adopted $\\tau{=}1$\nPlateau with $\\tau{=}2$ ($-0.9$)",
                 (1, ys_t[1]), (1.45, 76.4), fontsize=7.2, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    axb.set_xticks(range(len(ts)))
    axb.set_xticklabels([f"{t:g}" for t in ts])
    axb.set_xlabel(r"relevance temperature $\tau$")
    axb.set_ylabel("SSv2 center acc (%), $\\kappa{=}10\\%$")
    axb.set_ylim(58, 80)
    panel_label(axb, "(b) Relevance temperature")
    despine(axb)

    fig.tight_layout(w_pad=1.8)
    save(fig, os.path.join(FIGDIR, "figA9_robustness.pdf"))


# ---------------------------------------------------------------- Fig. A1
def figA1():
    """Alpha-ladder response; series transcribed from tables/alpha_sens.tex
    (hand-maintained table; adopted rung = filled star)."""
    ALPHAS = [1, 2, 4, 6]
    # benchmark -> budget -> (values aligned with ALPHAS (None = not run),
    #                         adopted index)
    DATA = {
        "Mind2Web Task (Step SR)": {
            "c=50%": ([43.68, 45.22, 46.22, 46.86], 3),
            "c=25%": ([31.18, 32.22, 33.81, 33.96], 3),
        },
        "MMBench-GUI L2 (acc)": {
            "r=50%": ([80.02, 80.66, 81.39, 79.97], 2),
            "r=25%": ([71.06, 72.15, 72.34, 72.45], 3),
        },
        "ScreenSpot-v2 (acc)": {
            "r=25%": ([89.15, 90.33, 89.86, None], 1),
            "r=10%": ([73.90, 73.35, 72.80, None], 0),
            "r=5%": ([52.52, 55.19, 54.17, None], 1),
        },
        "ScreenSpot-Pro (acc)": {
            "r=50%": ([66.79, 65.40, 67.30, None], 2),
            "r=25%": ([57.18, 57.50, 57.50, None], 2),
            "r=5%": ([20.30, 19.29, 17.39, None], 0),
        },
    }
    NOTES = {  # per-panel takeaway (regime read straight off the curves)
        "Mind2Web Task (Step SR)": ("monotone $\\uparrow$: history-heavy,\n"
                                    "strong prior pays", "lower right"),
        "MMBench-GUI L2 (acc)": ("capped: $\\alpha{=}6$ falls back\n"
                                 "at $r{=}50\\%$", "center right"),
        "ScreenSpot-v2 (acc)": ("plateau by $\\alpha{=}2$:\n"
                                "saliency already suffices", "center"),
        "ScreenSpot-Pro (acc)": ("tight budgets prefer\nlow $\\alpha$",
                                 "center right"),
    }
    LEG = {  # per-panel legend anchor (kept off the curves)
        "Mind2Web Task (Step SR)": {"loc": "center left"},
        "MMBench-GUI L2 (acc)": {"loc": "center left"},
        "ScreenSpot-v2 (acc)": {"loc": "center left",
                                "bbox_to_anchor": (0.0, 0.20)},
        "ScreenSpot-Pro (acc)": {"loc": "center left"},
    }
    shades = ["#D55E00", "#0072B2", "#009E73"]
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.05))
    for ax, (bench, series), lab in zip(axes, DATA.items(), "abcd"):
        xmax = 1
        for (budget, (ys, adopt)), col in zip(series.items(), shades):
            xs = [a for a, y in zip(ALPHAS, ys) if y is not None]
            vv = [y for y in ys if y is not None]
            xmax = max(xmax, max(xs))
            ax.plot(xs, vv, color=col, marker="o", ms=3,
                    mfc="white", label=budget, lw=1.1)
            ax.plot([ALPHAS[adopt]], [ys[adopt]], marker="*", ms=8.5,
                    color=col, zorder=5, mec="white", mew=0.4)
        ticks = [a for a in ALPHAS if a <= xmax]
        ax.set_xticks(ticks)
        ax.set_xlim(0.6, xmax + 0.4)
        ax.set_xlabel(r"prior strength $\alpha$ (public ladder)")
        ax.set_title(f"({lab}) {bench}", fontsize=7.5)
        note, loc = NOTES[bench]
        takeaway(ax, note, loc=loc, fontsize=5.8)
        ax.legend(fontsize=6, handlelength=1.2, borderaxespad=0.1,
                  labelspacing=0.2, **LEG[bench])
        despine(ax)
    axes[0].set_ylabel("Score (%)")
    fig.text(0.995, 0.98, r"$\bigstar$ = adopted rung (one per grid point)",
             fontsize=6.2, color="#555555", va="top", ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, os.path.join(FIGDIR, "figA1_alpha_ladder.pdf"))


# ---------------------------------------------------------------- Fig. A2
def figA2():
    """Certificate-dose monotonicity on AndroidControl.
    Source: AndroidControl dose sweep
    (rho_cov in {0, 0.3, 0.5}); nested rung 57.06 = adopted c25 arm."""
    doses = [0.0, 0.3, 0.5]
    c50 = [59.84, 59.91, 60.20]
    c25 = [52.53, 55.45, 56.58]
    unif = {"mild": 59.08, "tight": 56.13}  # AC uniform rows (Table I-6)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(doses, c50, color=C["ours"], marker="o", ms=4)
    ax.plot(doses, c25, color=C["best"], marker="s", ms=4)
    # direct curve labels (no legend: keeps the middle band free)
    ax.text(0.06, 60.35, r"mild ($c{=}50\%$)", fontsize=6.8,
            color=C["ours"], fontweight="bold")
    ax.text(0.015, 55.05, r"tight ($c{=}25\%$)", fontsize=6.8,
            color=C["best"], fontweight="bold")
    # value labels: mild curve above its points, tight curve below
    for x, y in zip(doses, c50):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(x, y + 0.22),
                    fontsize=5.8, color=C["ours"], ha="center",
                    annotation_clip=False)
    ax.annotate(f"{c25[0]:.2f}", (0.0, c25[0]), xytext=(-0.017, c25[0]),
                fontsize=5.8, color=C["best"], ha="right", va="center",
                annotation_clip=False)
    ax.annotate(f"{c25[1]:.2f}", (0.3, c25[1]), xytext=(0.3, c25[1] - 0.55),
                fontsize=5.8, color=C["best"], ha="center",
                annotation_clip=False)
    ax.annotate(f"{c25[-1]:.2f}", (0.5, c25[-1]),
                xytext=(0.455, c25[-1] - 0.68), fontsize=5.8,
                color=C["best"], ha="center")
    # total gain brackets at rho=0.5
    for series, col in ((c50, C["ours"]), (c25, C["best"])):
        gap_label(ax, 0.525, series[0], series[-1], col, fontsize=6.2,
                  dx=0.012)
    ax.scatter([0.5], [57.06], marker="*", s=80, color=C["best"], zorder=5,
               edgecolor="white", linewidth=0.4)
    ax.annotate("nested rung $\\rho{=}0.1{+}\\rho_{cov}{=}0.5$ (adopted)",
                (0.5, 57.06), fontsize=5.8, xytext=(0.03, 57.85),
                color=C["best"],
                arrowprops=dict(arrowstyle="->", color=C["best"], lw=0.6))
    for k, y in unif.items():
        ax.axhline(y, color=C["uniform"], ls="--", lw=0.8)
        ax.annotate(f"uniform ({k})", (-0.055, y + 0.10), fontsize=5.8,
                    color=C["uniform"], annotation_clip=False)
    ax.text(0.60, 0.05, "Monotone in dose. The tight budget\n"
            f"gains {c25[-1] - c25[0]:.1f} pts (vs {c50[-1] - c50[0]:.1f}"
            " at the mild budget)", transform=ax.transAxes, fontsize=7.2,
            style="italic", color="#333333", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC",
                      lw=0.5, alpha=0.85), zorder=6)
    ax.set_xlabel(r"Coverage-certificate dose $\rho_{\mathrm{cov}}$"
                  " (public ladder)")
    ax.set_ylabel("AndroidControl Step SR (%)")
    ax.set_xticks(doses)
    ax.set_xlim(-0.07, 0.63)
    ax.set_ylim(51.8, 61.0)
    despine(ax)
    save(fig, os.path.join(FIGDIR, "figA2_dose.pdf"))


# ---------------------------------------------------------------- Fig. A4
def figA4():
    """Prior value lineage: (A+B) - uniform at the domain's representative
    grid (SSv2 r10
    56.37-43.79, SSPro r25 57.05-29.85, M2W task c50 42.93-40.2x,
    AC c50 58.57-59.08). W58 Step-SR unification: the OmniGUI bar reports
    step exact-match (abl_mass_c50_h10 EM 47.43 - uniform_c50_h10 EM 46.11),
    not goal progress."""
    domains = ["AndroidControl\n(c=50%)", "OmniGUI\n(c=50%, Step SR)",
               "Mind2Web task\n(c=50%)", "ScreenSpot-v2\n(r=10%)",
               "ScreenSpot-Pro\n(r=25%)"]
    deltas = [-0.51, 1.32, 2.70, 12.58, 27.20]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    cols = [C["uniform"] if d < 0 else C["ours"] for d in deltas]
    ax.barh(range(len(domains)), deltas, color=cols, height=0.62, zorder=3)
    for i, d in enumerate(deltas):
        ax.text(d + (0.4 if d >= 0 else -0.4), i, f"{d:+.1f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7.5,
                fontweight="bold" if abs(d) > 10 else "normal")
    ax.axvline(0, color="black", lw=0.7)
    ax.text(0.5, 4.52, "uniform parity", fontsize=5.8, color="#555555")
    ax.set_ylim(-0.55, 4.85)
    # density/resolution gradient annotation along the y ordering
    ax.annotate("", (0.965, 0.88), (0.965, 0.12), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8))
    ax.text(0.985, 0.5, "denser layouts,\nhigher resolution",
            transform=ax.transAxes, fontsize=6, color="#555555",
            rotation=90, ha="left", va="center")
    takeaway(ax, "Mobile UIs: saliency suffices.\n"
             "Dense desktop/web: geometry is\nworth up to +27 pts",
             loc="lower right", fontsize=7.2)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=7)
    ax.set_xlabel(r"(LIP+NEO) $-$ uniform (points)")
    ax.set_xlim(-4.5, 33)
    ax.set_title("Layout density & resolution predict prior value",
                 fontsize=8)
    despine(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.7)
    save(fig, os.path.join(FIGDIR, "figA4_prior_value.pdf"))


# ---------------------------------------------------------------- Fig. A5
def figA5():
    """Damage localization: every Mind2Web arm (3 splits x 2 tiers), Op.F1
    vs Ele.Acc from the M2W source dicts."""
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    xs_pub, ys_pub, xs_ours, ys_ours = [], [], [], []
    ylo, yhi = 999.0, 0.0
    for split in ("task", "domain", "website"):
        for tier in ("I", "II"):
            for n, k, v in G.M2W[split][tier]:
                if not (G.fnum(v[0]) and G.fnum(v[1])):
                    continue
                x, y = float(v[0]), float(v[1])
                if k == "gray":
                    continue
                ylo, yhi = min(ylo, y), max(yhi, y)
                (xs_ours if k == "ours" else xs_pub).append(x)
                (ys_ours if k == "ours" else ys_pub).append(y)
    ax.axhspan(ylo, yhi, color=C["best"], alpha=0.07)
    ax.scatter(xs_pub, ys_pub, s=12, color="#888888", alpha=0.7,
               label=f"baselines ({len(xs_pub)} arms)")
    ax.scatter(xs_ours, ys_ours, s=60, marker="*", color=C["ours"],
               zorder=5, label="TRACE (6 grids)", edgecolor="white",
               linewidth=0.3)
    all_x = xs_pub + xs_ours
    xlo, xhi = min(all_x), max(all_x)
    # spread comparison: x-range arrow at the bottom vs the narrow y band
    y_arrow = ylo - 1.7
    ax.annotate("", (xhi, y_arrow), (xlo + 7, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=C["ours"], lw=0.8))
    ax.text((xlo + xhi) / 2, y_arrow + 0.3,
            f"grounding spread: {xhi - xlo:.0f} pts"
            f" ($ {(xhi - xlo) / (yhi - ylo):.0f}\\times$ the decision band)",
            fontsize=6.2, color=C["ours"], ha="center", va="bottom")
    ax.annotate(f"decision band: {yhi - ylo:.1f} pts "
                f"({ylo:.0f}--{yhi:.0f})",
                (0.03, 0.92), xycoords="axes fraction", fontsize=6.5,
                color=C["best"])
    ax.set_xlabel("Ele.Acc (grounding, %)")
    ax.set_ylabel("Op.F1 (decision, %)")
    ax.set_ylim(ylo - 4.6, yhi + 2.5)
    ax.set_title("Pruning damages grounding, not decisions", fontsize=8)
    ax.legend(fontsize=6.6, loc="lower left")
    despine(ax)
    save(fig, os.path.join(FIGDIR, "figA5_damage.pdf"))


# ---------------------------------------------------------------- Fig. A3
def figA3():
    """TTFT vs within-episode step: dense grows with history, TRACE flat.
    Data: figures/ttft_by_step.csv (extracted by _w33_ttft_by_step.py from the
    lat2 OmniGUI probe; steps kept where n >= MIN_N for a stable mean)."""
    MIN_N = 4
    path = os.path.join(FIGDIR, "ttft_by_step.csv")
    rows = [ln.strip().split(",") for ln in open(path) if ln.strip()]
    hdr, data = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(hdr)}
    steps, dense, ours = [], [], []
    for r in data:
        n = min(int(r[col["dense_n"]]), int(r[col["ours_n"]]))
        if n < MIN_N:
            continue
        steps.append(int(r[col["step"]]))
        dense.append(float(r[col["dense_ttft_ms"]]))
        ours.append(float(r[col["ours_ttft_ms"]]))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(steps, dense, color=C["dense"], marker="o", ms=4,
            label="Dense re-prefill")
    ax.plot(steps, ours, color=C["ours"], marker="*", ms=7,
            label="TRACE (KV reuse)")
    ax.fill_between(steps, ours, dense, color=C["ours"], alpha=0.08)
    # Least-squares slopes summarize relative history-dependent growth.
    sl = np.polyfit(steps, dense, 1)[0]
    ax.annotate(f"grows $\\approx${sl:+.0f} ms/step\n(history re-prefilled"
                " every step)", (steps[-3], dense[-3]),
                xytext=(steps[0] + 0.15, 2650), fontsize=6.2,
                color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6))
    sl_ours = np.polyfit(steps, ours, 1)[0]
    ax.annotate(f"slower growth: $\\approx${sl_ours:+.0f} ms/step\n"
                "(retired frames are not re-encoded)", (steps[-3], ours[-3]),
                xytext=(steps[1] + 0.4, 155), fontsize=6.2,
                color=C["ours"],
                arrowprops=dict(arrowstyle="->", color=C["ours"], lw=0.6))
    j = len(steps) - 1
    gap_label(ax, steps[j], ours[j], dense[j], C["ours"], fontsize=6.2,
              text=" ")
    ax.text(steps[j] - 0.12, 1500, f"step {steps[j]}: dense pays\n"
            f"{dense[j] / ours[j]:.1f}$\\times$ the TTFT", fontsize=6.2,
            color=C["ours"], ha="right", va="center", fontweight="bold")
    ax.set_xlabel("Step index within episode")
    ax.set_ylabel("TTFT per step (ms)")
    ax.set_ylim(0, dense[-1] * 1.07)
    ax.set_title("KV reuse slows history-dependent TTFT growth", fontsize=8)
    ax.legend(fontsize=6.8, loc="upper left")
    despine(ax)
    save(fig, os.path.join(FIGDIR, "figA3_ttft_by_step.pdf"))


# ---------------------------------------------------------------- Fig. A7
def figA7():
    """Cross-scale rank (in)stability: method ordering at 8B vs the 2B
    carrier. The measured story is asymmetric and worth stating exactly:
    the 2B carrier scrambles the ordering of published baselines (rho as low
    as 0.02 on MMBench), while TRACE holds rank 1 on four of the six
    plotted settings (12 of all 22 grids; top two on 15) -- i.e. the paper's
    conclusion transfers even though the baseline league table does not.
    Random/Uniform excluded (trivially last)."""
    import _w33_gen_2b_tables as R

    def ranks(rows, idx):
        vals = {}
        for n, k, v in rows:
            if k not in ("n", "ours") or n in ("Random", "Uniform"):
                continue
            if G.fnum(v[idx]):
                vals[clean(n)] = float(v[idx])
        order = sorted(vals, key=lambda n: -vals[n])
        return {n: r + 1 for r, n in enumerate(order)}

    specs = [
        ("ScreenSpot-v2", G._single_main_rows("10"), R.single_rows("10"), 2),
        ("ScreenSpot-Pro", G._single_main_rows("10"), R.single_rows("10"), 5),
        ("MMBench-GUI L2", G._single_main_rows("10"), R.single_rows("10"), 8),
        ("OmniGUI", G._multi_main_rows("I"), R.multi_rows("I"), 1),
        ("Mind2Web (pooled)", G._multi_main_rows("I"), R.multi_rows("I"), 4),
        ("AndroidControl", G._multi_main_rows("I"), R.multi_rows("I"), 7),
    ]
    apply(scale=1.3)  # 7.05in design prints at 5.5in
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35))
    rhos = []
    n_rank1 = 0
    for k, (title, rows8, rows2, idx) in enumerate(specs):
        ax = axes[k // 3][k % 3]
        r8, r2 = ranks(rows8, idx), ranks(rows2, idx)
        if r2.get("TRACE") == 1:
            n_rank1 += 1
        common = sorted(set(r8) & set(r2))
        n = len(common)
        d2 = sum((r8[m] - r2[m]) ** 2 for m in common)
        rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
        rhos.append(rho)
        ax.plot([0.5, n + 0.5], [0.5, n + 0.5], ls=":", color="#999999",
                lw=1.0)
        for m in common:
            is_ours = m == "TRACE"
            ax.scatter(r8[m], r2[m], s=64 if is_ours else 30,
                       marker="*" if is_ours else "o",
                       color=C["ours"] if is_ours else C["best"], zorder=5)
            # label big movers, but keep the rho text corner clear
            in_text_zone = r8[m] >= 5 and r2[m] <= 3
            if is_ours or (abs(r8[m] - r2[m]) >= 3 and not in_text_zone):
                # near the top edge, drop the label below the dot so it
                # cannot run into the bold rho/rank note
                dy = 0.85 if (not is_ours and r2[m] <= 3) else -0.42
                ax.annotate(m, (r8[m], r2[m]),
                            xytext=(r8[m] + 0.25, r2[m] + dy), fontsize=7.0,
                            color=C["ours"] if is_ours else "#333333")
        ax.set_title(title, fontsize=10.4)
        ours_lab = (f"TRACE: {r8.get('TRACE','?')}$\\to$"
                    f"{r2.get('TRACE','?')}")
        ax.text(0.04, 0.955, f"$\\rho_s{{=}}{rho:.2f}$ ($n{{=}}{n}$)\n"
                + ours_lab,
                transform=ax.transAxes, fontsize=8.1, va="top",
                fontweight="bold", color="#333333", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.8))
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ticks = sorted(set([1] + list(range(4, n + 1, 3)) + [n]))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=7.3)
        ax.invert_xaxis()
        ax.invert_yaxis()
        if k // 3 == 1:
            ax.set_xlabel("rank on 8B carrier (1 = best)", fontsize=8.6)
        if k % 3 == 0:
            ax.set_ylabel("rank on 2B carrier", fontsize=8.6)
        despine(ax)
    fig.suptitle("Baselines scramble across scale. TRACE holds rank 1 on "
                 f"{n_rank1}/6 benchmarks (mean $\\rho_s{{=}}"
                 f"{np.mean(rhos):.2f}$)", fontsize=10.6, y=0.995)
    # one shared key instead of repeating it in every panel
    fig.text(0.5, 0.945, "$\\star$ TRACE   $\\bullet$ published "
             "baselines   $\\rho_s$ = Spearman rank correlation, 8B vs 2B",
             ha="center", fontsize=8.4, color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.930))
    save(fig, os.path.join(FIGDIR, "figA7_rank_stability.pdf"))


# ---------------------------------------------------------------- Fig. A8
def figA8():
    """Ablation forest: every paired-bootstrap anchor pair of Table 5 with
    its measured 95% CI (docs/w36_bootstrap_ci.json, 10k resamples). Positive
    delta = removing the component costs accuracy. W58: the json is now
    produced by scripts/_w58_em_boot.py in the unified Step-SR convention --
    OmniGUI rows pair per-episode exact-match and Mind2Web rows pool the
    three official splits with episode clustering, so every row is
    episode-safe and starrable."""
    import json

    F = json.load(open(os.path.join(HERE, "..", "docs",
                                    "w36_bootstrap_ci.json"),
                       encoding="utf-8"))
    BENCH = {"ssv2": "SSv2", "sspro": "SSPro", "m2w": "M2W", "omni": "OmniGUI"}
    TWIN = [("noA", "$-$ prior (LMP)", "#0072B2"),
            ("nodiv", "$-$ diversity", "#009E73"),
            ("unifa", "$-$ instruction", "#CC79A7"),
            ("noC", "$-$ certificate (AWC)", "#E69F00")]

    rows = []
    for sec, (pref, lab, col) in enumerate(TWIN):
        for f in F:
            if not f["twin"].startswith(pref):
                continue
            bench = BENCH[f["anchor"].split()[0]]
            budget = f["anchor"].split()[1]
            tag = "tight" if "tight" in f["anchor"] else "mild"
            rows.append((sec, f"{bench} {budget} ({tag})", lab, col, f))

    apply(scale=1.15)  # 4.6in design prints at ~4.0in
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    ypos = np.arange(len(rows))[::-1]
    # group bands: prior necessity / factor knockouts / isolated C
    GROUP = {0: "Prior Necessity", 1: "Factor Knockouts",
             2: "Factor Knockouts", 3: "Certificate (AWC)"}
    xmax = max(f["hi"] for *_, f in rows) + 5.6
    xmin = min(f["lo"] for *_, f in rows) - 1.2
    sec_prev, grp_prev = None, None
    for y, (sec, bench_lab, lab, col, f) in zip(ypos, rows):
        if sec_prev is not None and sec != sec_prev:
            ax.axhline(y + 0.5, color="#DDDDDD", lw=0.7)
        if GROUP[sec] != grp_prev:
            # sits just above the group's first row, clear of the delta text
            ax.text(xmax - 0.4, y + 0.42, GROUP[sec], fontsize=6.8,
                    style="italic", color="#777777", ha="right", va="bottom")
        sec_prev, grp_prev = sec, GROUP[sec]
        ax.plot([f["lo"], f["hi"]], [y, y], color=col, lw=1.5,
                solid_capstyle="butt")
        ax.plot([f["lo"], f["lo"]], [y - 0.16, y + 0.16], color=col, lw=1.2)
        ax.plot([f["hi"], f["hi"]], [y - 0.16, y + 0.16], color=col, lw=1.2)
        ax.plot([f["delta"]], [y], "s", ms=4.6, color=col, zorder=5)
        star = "$^{\\star}$" if f["p"] < 0.05 else "  (ns)"
        ax.text(f["hi"] + 0.45, y, f"{f['delta']:+.2f}{star}", fontsize=6.8,
                va="center", color=col, fontweight="bold")
    ax.axvline(0, color="#555555", ls=":", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b} $\\cdot$ {l}" for _, b, l, _, _ in rows],
                       fontsize=6.8)
    ax.set_xlabel("$\\Delta$ (pts) when the component is removed "
                  "(paired 95% interval)", fontsize=7.8)
    ax.set_xlim(xmin, xmax)
    ax.set_title("Every Tabulated Ablation, with Its Measured Interval",
                 fontsize=9.2)
    despine(ax)
    fig.tight_layout()
    save(fig, os.path.join(FIGDIR, "figA8_ablation_forest.pdf"))


# ------------------------------------------------------- fig_sens (main text)
def fig_sens():
    """Half-column sensitivity figure for the main-text wrapfigure. Distinct
    encodings from the appendix versions (figA1 = absolute ladders on four
    suites; fig7 = full dose-response with variants):
    (a) alpha response as deltas vs the alpha=1 rung on the two suites with
        opposite regimes (source series: figA1 / tables/alpha_sens.tex);
    (b) coverage-dose inverted U at r=10% only (source series: fig7).
    Designed at print size (0.5\\linewidth = 2.7in) with explicit fonts."""
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(2.72, 3.30),
                                  gridspec_kw={"hspace": 0.62},
                                  constrained_layout=False)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.935, bottom=0.115)

    # ---- (a) two opposite alpha regimes, delta vs alpha=1 ----
    alphas = [1, 2, 4, 6]
    m2w = [31.18, 32.22, 33.81, 33.96]      # Mind2Web Task, c=25% (adopted 6)
    sspro = [20.30, 19.29, 17.39, None]     # ScreenSpot-Pro, r=5% (adopted 1)
    for ys, adopt, col, lab in ((m2w, 3, C["ours"],
                                 "Mind2Web c=25% (history-heavy)"),
                                (sspro, 0, C["best"],
                                 "SSPro r=5% (tight budget)")):
        xs = [a for a, y in zip(alphas, ys) if y is not None]
        dv = [y - ys[0] for y in ys if y is not None]
        ax.plot(xs, dv, color=col, marker="o", ms=2.8, lw=1.1, label=lab)
        ax.plot([alphas[adopt]], [dv[adopt]], marker="*", ms=9, color=col,
                zorder=5, mec="white", mew=0.4)
    ax.axhline(0, color="#999999", ls=":", lw=0.8)
    ax.set_xticks(alphas)
    ax.tick_params(labelsize=6.2)
    ax.set_xlabel(r"prior strength $\alpha$", fontsize=6.6, labelpad=1.5)
    ax.set_ylabel(r"$\Delta$ vs $\alpha{=}1$ (pts)", fontsize=6.6,
                  labelpad=1.5)
    # direct curve labels (color-matched) instead of a legend box: the
    # upper-left hosts the rising curve's name, the empty lower-right the
    # falling curve's name, next to their line ends
    ax.text(1.0, 2.55, "Mind2Web c=25%\n(history-heavy)", fontsize=5.6,
            color=C["ours"], ha="left", va="top")
    ax.text(4.25, -2.45, "SSPro r=5%\n(tight budget)", fontsize=5.6,
            color=C["best"], ha="left", va="top")
    ax.set_title("(a) Opposite prior regimes ($\\bigstar$ adopted)",
                 fontsize=7.0)
    despine(ax)

    # ---- (b) coverage dose inverted U at r=10% ----
    xs = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    r10 = [56.37, 69.34, 72.33, 73.90, 74.06, 43.79]
    axb.axhline(43.79, color=C["uniform"], ls=":", lw=0.9)
    axb.text(0.03, 45.0, "uniform grid", fontsize=5.6, color="#777777")
    axb.plot(xs, r10, color=C["ours"], marker="o", ms=2.8, lw=1.2, zorder=4)
    axb.plot(xs[3], r10[3], marker="*", ms=10, color=C["ours"], mec="white",
             mew=0.4, zorder=6)
    axb.annotate("greedy alone", (0.0, r10[0]), (0.10, 59.5), fontsize=5.6,
                 color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6))
    axb.annotate("full dose\n$\\equiv$ uniform", (1.0, r10[-1]),
                 (0.70, 52.0), fontsize=5.6, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=0.6))
    axb.text(0.50, 77.6, "plateau $\\pm$0.9 over 0.3\u20130.7",
             fontsize=5.6, color="#555555", ha="center")
    axb.tick_params(labelsize=6.2)
    axb.set_xlabel(r"coverage dose $\rho_{\mathrm{cov}}$", fontsize=6.6,
                   labelpad=1.5)
    axb.set_ylabel("SSv2 acc (%), $r{=}10\\%$", fontsize=6.6, labelpad=1.5)
    axb.set_ylim(40, 82)
    axb.set_title("(b) Coverage dose: repair, plateau, collapse",
                  fontsize=7.0)
    despine(axb)

    save(fig, os.path.join(FIGDIR, "fig6_sens.pdf"))


if __name__ == "__main__":
    apply()
    fig3()
    fig4()
    fig7()
    fig_sens()
    figA9()
    figA1()
    figA2()
    figA3()
    figA4()
    figA5()
    figA7()
    figA8()
    print("all figures rendered ->", os.path.abspath(FIGDIR))
