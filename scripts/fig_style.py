"""Unified figure style (Okabe-Ito colorblind-safe).

Method colors/markers are constant across every figure in the paper.
All text renders in Times New Roman (STIX math, visually matched).

Copy rules for every matplotlib panel (skip the draw.io method overview):
  1. Panel titles are always "(a) Sentence-style caption", never a bare "(a)".
     Hyphen compounds lowercase the second stem: Within-goal, not Within-Goal.
  2. Legend first words are capitalized (Stateless re-prefill). Never append
     "(ours)"; method curves are TRACE or MKC reuse.
  3. Annotation sentences start with a capital (History re-encoded...).
  4. No semicolons anywhere in figure text (use a period, dash, or newline).
  5. Keep statistical meta (N pairs, 95% CI) out of axes; put them in the
     LaTeX caption.
  6. Light-gray callouts use color "#333333" and fontsize >= 7.2 on main-text
     figures (>= 7.4 on the Pareto figure).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito anchored palette (docs/figure_material.md section 3)
C = {
    "ours": "#D55E00",       # orange-red, WITNESS everywhere
    "best": "#0072B2",       # deep blue, strongest published baseline
    "second": "#009E73",     # green, secondary baseline
    "uniform": "#999999",    # grey (dashed), uniform/random
    "dense": "#000000",      # black (dotted), dense upper bound
    "witness": "#E69F00",    # amber, certificate/witness emphasis
    "removed": "#CC3311",    # removal overlay (35% alpha)
    "band": "#D55E00",       # advantage band fill (low alpha)
    "concede": "#7A7A7A",    # honest-concession annotations
}

# Per-method constant identity (used by Fig.4 and any per-method plot)
METHOD_STYLE = {
    "WITNESS":  {"color": "#D55E00", "marker": "*"},
    "DivPrune": {"color": "#56B4E9", "marker": "o"},
    "CDPruner": {"color": "#009E73", "marker": "s"},
    "VisPruner": {"color": "#0072B2", "marker": "D"},
    "FastV":    {"color": "#CC79A7", "marker": "^"},
    "PruneSID": {"color": "#E69F00", "marker": "v"},
    "Dense":    {"color": "#555555", "marker": "P"},
}

MARKERS = {"ours": "*", "best": "o", "second": "s", "uniform": "^"}


def apply(scale=1.0):
    """`scale` compensates \\includegraphics shrinkage: figures designed wider
    than the 5.5in text block should pass scale = design_w / print_w so
    fonts land at their intended physical size on paper."""
    # Times New Roman first; fall back to metric-compatible serifs when the
    # renderer host lacks it (remote Linux pods).
    names = {f.name for f in fm.fontManager.ttflist}
    serif = [f for f in ("Times New Roman", "Nimbus Roman", "Liberation Serif",
                         "STIXGeneral", "DejaVu Serif") if f in names] or \
        ["DejaVu Serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",
        "font.size": 8.5 * scale,
        "axes.labelsize": 8.5 * scale,
        "axes.titlesize": 9 * scale,
        "legend.fontsize": 7.5 * scale,
        "xtick.labelsize": 8 * scale,
        "ytick.labelsize": 8 * scale,
        "axes.linewidth": 0.6 * scale,
        "lines.linewidth": 1.4 * scale,
        "grid.linewidth": 0.5 * scale,
        "grid.color": "#DDDDDD",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })
    if serif[0] != "Times New Roman":
        print("[fig_style] Times New Roman unavailable; using", serif[0])


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)


def panel_label(ax, s, fontsize=9):
    """Left-aligned bold panel title. Callers must pass a full caption such as
    '(a) Accuracy-TTFT trade-off', never a bare '(a)'."""
    ax.set_title(s, loc="left", fontweight="bold", fontsize=fontsize)


def takeaway(ax, text, loc="lower left", color="#333333", fontsize=7.2):
    """One-line italic claim embedded in the panel, boxed for legibility."""
    if ";" in text:
        raise ValueError(f"figure text forbids semicolons: {text!r}")
    xy = {"lower left": (0.03, 0.04, "left", "bottom"),
          "lower right": (0.97, 0.04, "right", "bottom"),
          "upper left": (0.03, 0.96, "left", "top"),
          "upper right": (0.97, 0.96, "right", "top"),
          "center": (0.55, 0.40, "center", "center"),
          "center right": (0.97, 0.50, "right", "center")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=fontsize,
            style="italic", color=color, ha=xy[2], va=xy[3], zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC",
                      lw=0.5, alpha=0.85))


def gap_label(ax, x, y_lo, y_hi, color, fontsize=6.4, dx=0.06, text=None):
    """Vertical brace-style delta annotation between two curve values."""
    d = y_hi - y_lo
    ax.annotate("", (x, y_hi), (x, y_lo),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(x + dx, (y_lo + y_hi) / 2, text or f"{d:+.1f}",
            fontsize=fontsize, color=color, va="center", ha="left",
            fontweight="bold")


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    if path.endswith(".pdf"):  # raster preview for quick visual checks
        fig.savefig(path[:-4] + ".png", bbox_inches="tight",
                    pad_inches=0.02, dpi=220)
    plt.close(fig)
    print("wrote", path)
