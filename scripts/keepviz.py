"""Shared primitives for every token-selection visualization in the paper.

Design rule (the whole point of this module): a keep map must be readable as an
image, and the reader must simultaneously see (i) what was selected and (ii)
what the untouched input actually contained. Two failure modes to avoid:

  * tinting the KEPT cells (the old `overlay_mask`: orange at alpha 0.42 over
    kept, black at only alpha 0.66 over dropped) veils the evidence and leaves
    the dropped cells半 visible -- the contrast reads backwards and looks muddy.
  * blacking the DROPPED cells out completely destroys all context, so the
    reader cannot tell whether a missing region held a button or blank padding.

The convention used here instead is a spotlight:

    KEPT    -> the ORIGINAL screenshot pixels at full fidelity
    DROPPED -> a "ghost": desaturated and flattened toward neutral gray, so the
               layout stays legible but is unmistakably de-emphasized

Compositing happens at pixel level (not as an alpha raster), so cell edges are
exact and the kept regions carry no overlay haze at all.

Layer order, from bottom: composited screenshot -> score field -> element boxes
-> target box -> annotations.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.patches as patches

# Lifecycle palette. Deliberately distinct hues so a three-state cell map reads
# without a legend lookup: admitted-then-retired is cool, survivor is the method
# orange, and freshly admitted is amber.
LIFECYCLE = {
    "survive": "#D55E00",   # retained through the retirement contraction
    "retire": "#0072B2",    # admitted at entry, dropped at retirement
    "admit": "#E69F00",     # newly admitted at this step
    "absent": "#BBBBBB",    # not in cache
}

TARGET_EC = "#00E676"      # ground-truth action target
MISS_EC = "#FF1744"
BOX_EC = "#FFFFFF"         # detector element outline

# transparent -> deep orange, for score fields laid over a screenshot
MASS_CMAP = LinearSegmentedColormap.from_list(
    "mass", [(1, 1, 1, 0), (0.84, 0.37, 0.0, 0.62)])
# FocusUI-style perceptual score ramp for standalone heat panels
HEAT_CMAP = "turbo"


def keep_grid(rec, key="local_keep"):
    """Flat row-major token indices -> boolean (grid_h, grid_w) mask."""
    gw = rec["grid_w"]
    g = np.zeros((rec["grid_h"], gw), dtype=bool)
    for j in rec.get(key) or []:
        g[j // gw, j % gw] = True
    return g


def idx_grid(idxs, gh, gw):
    g = np.zeros((gh, gw), dtype=bool)
    for j in idxs:
        g[j // gw, j % gw] = True
    return g


def show_image(ax, img, frame_lw=0.6, frame_ec="#444444"):
    """Bare screenshot with ticks stripped and a hairline frame."""
    ax.imshow(img, interpolation="lanczos", zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(frame_lw)
        s.set_edgecolor(frame_ec)
    return ax


# Ghost treatment for dropped cells: how far to desaturate, how far to pull
# toward the flat base tone, and which tone. Tuned on real screenshots (see
# scripts/_w59_ghost_probe.py) so that dropped widgets remain identifiable as
# widgets while the kept region reads as the only thing in focus. A dark base
# separates far better than a light one, because UI screenshots are mostly
# light: fading toward white erases structure, fading toward near-black keeps
# edges as visible outlines.
GHOST = {"desat": 0.90, "fade": 0.72, "gray": 0.18}
# Even gentler variant for panels where the surrounding context carries the
# argument (e.g. lifecycle maps with three simultaneous states).
GHOST_SOFT = {"desat": 0.82, "fade": 0.58, "gray": 0.22}


def _as_float_rgb(img):
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    arr = arr[:, :, :3]
    return arr / 255.0 if arr.max() > 1.001 else arr


def _cell_mask_to_pixels(mask, H, W):
    """Nearest-neighbour upsample of a (gh, gw) cell mask to (H, W) pixels."""
    gh, gw = mask.shape
    ri = (np.arange(H) * gh // H).clip(0, gh - 1)
    ci = (np.arange(W) * gw // W).clip(0, gw - 1)
    return mask[ri][:, ci]


def ghost_composite(img, mask, ghost=None, kept_boost=0.0):
    """Spotlight composite: kept pixels untouched, dropped pixels ghosted.

    Returns an RGB float array. Dropped regions are desaturated toward
    luminance and then blended toward a neutral gray, which keeps the input
    layout legible (unlike a solid black-out) while making the selected region
    pop (unlike a uniform translucent veil).
    """
    g = dict(GHOST, **(ghost or {}))
    rgb = _as_float_rgb(img)
    H, W = rgb.shape[:2]
    pm = _cell_mask_to_pixels(np.asarray(mask, dtype=bool), H, W)

    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
           + 0.0722 * rgb[..., 2])[..., None]
    ghosted = rgb + (lum - rgb) * float(g["desat"])
    ghosted = ghosted + (float(g["gray"]) - ghosted) * float(g["fade"])

    out = np.where(pm[..., None], rgb, ghosted)
    if kept_boost:
        # gentle local contrast lift on the kept region only
        out = np.where(pm[..., None],
                       np.clip((out - 0.5) * (1.0 + kept_boost) + 0.5, 0, 1),
                       out)
    return np.clip(out, 0.0, 1.0)


def draw_keep(ax, img, mask, tint=None, tint_alpha=0.22, grid_lw=0.0,
              grid_color="#FFFFFF", ghost=None, outline=True,
              outline_color="#FFFFFF", outline_lw=0.9, kept_boost=0.0):
    """The canonical keep map: kept cells keep原 pixels, dropped cells ghost.

    `tint` optionally washes the KEPT cells with a faint method color for
    figures that must distinguish two keep sets side by side -- keep tint_alpha
    low so pixels stay readable. `outline` traces the boundary of the kept
    region, which is what makes the selection shape readable at page scale.
    """
    w, h = img.size if hasattr(img, "size") else (img.shape[1], img.shape[0])
    comp = ghost_composite(img, mask, ghost=ghost, kept_boost=kept_boost)
    ax.imshow(comp, interpolation="lanczos", extent=(0, w, h, 0), zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor("#444444")

    gh, gw = mask.shape
    if tint is not None:
        rgba = np.zeros((gh, gw, 4))
        rgba[mask] = to_rgba(tint, tint_alpha)
        ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest", zorder=3)
    if outline:
        keep_outline(ax, mask, w, h, color=outline_color, lw=outline_lw)
    if grid_lw > 0:
        for r in range(gh + 1):
            ax.axhline(r * h / gh, color=grid_color, lw=grid_lw, alpha=0.30,
                       zorder=4)
        for c in range(gw + 1):
            ax.axvline(c * w / gw, color=grid_color, lw=grid_lw, alpha=0.30,
                       zorder=4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return ax


def keep_outline(ax, mask, w, h, color="#FFFFFF", lw=0.9, alpha=0.95,
                 shadow=True):
    """Trace only the boundary edges of the kept region.

    Drawing the silhouette rather than per-cell boxes keeps the vector output
    small and lets the eye read the selected shape instantly. A thin dark
    shadow underneath keeps the white contour visible on pale screenshots.
    """
    m = np.asarray(mask, dtype=bool)
    gh, gw = m.shape
    cw, ch = w / gw, h / gh
    segs = []
    pad = np.zeros((gh + 2, gw + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    for r in range(gh):
        for c in range(gw):
            if not m[r, c]:
                continue
            x0, y0 = c * cw, r * ch
            if not pad[r, c + 1]:                 # top neighbour empty
                segs.append(((x0, y0), (x0 + cw, y0)))
            if not pad[r + 2, c + 1]:             # bottom
                segs.append(((x0, y0 + ch), (x0 + cw, y0 + ch)))
            if not pad[r + 1, c]:                 # left
                segs.append(((x0, y0), (x0, y0 + ch)))
            if not pad[r + 1, c + 2]:             # right
                segs.append(((x0 + cw, y0), (x0 + cw, y0 + ch)))
    if not segs:
        return ax
    from matplotlib.collections import LineCollection

    if shadow:
        ax.add_collection(LineCollection(segs, colors="#000000",
                                         linewidths=lw * 2.1, alpha=0.34,
                                         zorder=4.8))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=5))
    return ax


def draw_states(ax, img, states, colors, alpha=0.52, ghost=None,
                outline_states=(), outline_color="#FFFFFF", outline_lw=0.9):
    """Categorical cell map: `states` is an int grid, 0 = dropped/background.

    `colors` maps state id -> color. State 0 keeps the ghost treatment (so the
    input stays legible), and every other state is a translucent wash over the
    ORIGINAL pixels, so a lifecycle map still shows the underlying widgets.
    """
    w, h = img.size if hasattr(img, "size") else (img.shape[1], img.shape[0])
    st = np.asarray(states)
    comp = ghost_composite(img, st != 0, ghost=ghost)
    ax.imshow(comp, interpolation="lanczos", extent=(0, w, h, 0), zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor("#444444")
    gh, gw = st.shape
    rgba = np.zeros((gh, gw, 4))
    for sid, col in colors.items():
        if sid == 0:
            continue
        rgba[st == sid] = to_rgba(col, alpha)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest", zorder=3)
    for sid in outline_states:
        keep_outline(ax, st == sid, w, h, color=outline_color, lw=outline_lw)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return ax


def score_heat(ax, img, field, cmap=None, alpha=0.72, over_image=True,
               vmax=None):
    """Score/prior field over the screenshot (the FocusUI left-hand panel).

    `field` is a 2-D array on the token grid. When `over_image` the screenshot
    stays visible underneath and the ramp starts transparent, so structure and
    score are readable at once.
    """
    w, h = img.size if hasattr(img, "size") else (img.shape[1], img.shape[0])
    if over_image:
        show_image(ax, img)
        ax.imshow(field, extent=(0, w, h, 0), origin="upper",
                  cmap=cmap or MASS_CMAP, interpolation="bilinear",
                  vmin=0.0, vmax=vmax, zorder=3)
    else:
        ax.imshow(field, cmap=cmap or HEAT_CMAP, interpolation="bilinear",
                  vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_edgecolor("#444444")
    return ax


def element_boxes(ax, boxes, w, h, ec=BOX_EC, lw=0.55, alpha=0.85,
                  max_boxes=None, top_by_density=False):
    """Detector element outlines. `boxes` rows are normalized
    [x0, y0, x1, y1, info_density] as written by the offline Module-A harvest."""
    rows = list(boxes or [])
    if top_by_density and rows and len(rows[0]) >= 5:
        rows.sort(key=lambda b: -float(b[4]))
    if max_boxes:
        rows = rows[:max_boxes]
    for b in rows:
        if len(b) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        if max(x0, y0, x1, y1) <= 1.5:      # normalized -> pixels
            x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, ec=ec, lw=lw,
            alpha=alpha, zorder=6))
    return ax


def target_box(ax, bbox, w, h, kept=True, lw=1.7, cross_on_miss=True,
               fmt="xywh"):
    """Ground-truth target. Green when its region survived, red with an X when
    it was pruned away -- the single most important read in every keep map."""
    if not bbox or len(bbox) < 4:
        return ax
    a, b, c, d = [float(v) for v in bbox[:4]]
    if fmt == "xyxy":
        x, y, bw, bh = a, b, c - a, d - b
    else:
        x, y, bw, bh = a, b, c, d
    if max(x, y, bw, bh) <= 1.5:
        x, y, bw, bh = x * w, y * h, bw * w, bh * h
    ec = TARGET_EC if kept else MISS_EC
    ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, ec=ec, lw=lw,
                                   zorder=8))
    if not kept and cross_on_miss:
        ax.plot([x, x + bw], [y, y + bh], color=ec, lw=lw, zorder=8)
        ax.plot([x, x + bw], [y + bh, y], color=ec, lw=lw, zorder=8)
    return ax


def panel_tag(ax, text, fontsize=7.4, color="#111111"):
    """Small tag inside the top-left of an image panel (kept off the axes so
    the screenshot is never cropped)."""
    ax.text(0.025, 0.977, text, transform=ax.transAxes, fontsize=fontsize,
            va="top", ha="left", color=color, zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BBBBBB",
                      lw=0.4, alpha=0.92))
    return ax


def keep_badge(ax, kept_frac, hit=None, fontsize=7.0):
    """Bottom-right chip reporting the retention and, when known, the outcome."""
    txt = f"keep {100.0 * kept_frac:.0f}%"
    if hit is not None:
        txt += "  |  " + ("HIT" if hit else "MISS")
    fc = "white" if hit is None else ("#E8F8EE" if hit else "#FDE8EC")
    ec = "#BBBBBB" if hit is None else (TARGET_EC if hit else MISS_EC)
    ax.text(0.977, 0.023, txt, transform=ax.transAxes, fontsize=fontsize,
            va="bottom", ha="right", zorder=10,
            bbox=dict(boxstyle="round,pad=0.20", fc=fc, ec=ec, lw=0.6,
                      alpha=0.94))
    return ax


def column_title(ax, text, fontsize=8.0, pad=4.0, weight="normal",
                 color="#111111"):
    ax.set_title(text, fontsize=fontsize, pad=pad, fontweight=weight,
                 color=color)
    return ax


def query_caption(fig, axes, text, fontsize=7.6, dy=0.012, color="#222222",
                  max_chars=112):
    """Italic instruction line centered under a group of image panels, in the
    style of the reference figures (`Instruction query: "..."`)."""
    if not text:
        return
    t = str(text).replace("\n", " ").strip()
    if len(t) > max_chars:
        t = t[:max_chars - 1].rstrip() + "\u2026"
    boxes = [a.get_position() for a in axes]
    x = 0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes))
    y = min(b.y0 for b in boxes) - dy
    fig.text(x, y, f'Instruction query: "{t}"', fontsize=fontsize,
             style="italic", ha="center", va="top", color=color)


def row_label(fig, axes, text, fontsize=8.4, dx=0.014, weight="bold"):
    """Rotated label to the left of an image row."""
    boxes = [a.get_position() for a in axes]
    y = 0.5 * (min(b.y0 for b in boxes) + max(b.y1 for b in boxes))
    x = min(b.x0 for b in boxes) - dx
    fig.text(x, y, text, fontsize=fontsize, rotation=90, ha="right",
             va="center", fontweight=weight)


def legend_chips(ax, entries, fontsize=6.9, y=1.035, x0=0.0, gap=0.012,
                 chip_w=0.030, chip_h=0.030):
    """Inline color-chip legend above a panel: entries = [(label, color), ...].
    Used instead of matplotlib legends so image panels stay uncropped."""
    x = x0
    for label, col in entries:
        ax.add_patch(patches.Rectangle(
            (x, y), chip_w, chip_h, transform=ax.transAxes, clip_on=False,
            fc=col, ec="#555555", lw=0.35, zorder=11))
        ax.text(x + chip_w + 0.006, y + chip_h / 2, label,
                transform=ax.transAxes, fontsize=fontsize, va="center",
                ha="left", zorder=11)
        x += chip_w + 0.006 + 0.0092 * len(label) + gap
    return ax


def enable_cjk():
    """OmniGUI instructions are Chinese. Keep Times for Latin and fall back
    to a CJK face for missing glyphs (matplotlib 3.6+ family-list fallback)."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    have = {f.name for f in fm.fontManager.ttflist}
    latin = [n for n in ("Times New Roman", "Nimbus Roman", "Liberation Serif",
                         "STIXGeneral", "DejaVu Serif") if n in have]
    cjk = [n for n in ("SimSun", "Microsoft YaHei", "Noto Serif CJK SC",
                       "Source Han Serif SC", "SimHei") if n in have]
    stack = latin + cjk
    if stack:
        plt.rcParams["font.serif"] = stack
        plt.rcParams["font.family"] = stack
    plt.rcParams["axes.unicode_minus"] = False
