#!/usr/bin/env python
"""Figure 9 (fig:keepmaps): selection outcomes across methods and budgets.

Replaces the earlier `render_fig5_qual.py` keep-map row. The presentation
convention is the one that makes a keep map legible (see `keepviz`): kept cells
show the ORIGINAL screenshot pixels and dropped cells are solid black, so the
reader can literally read which widgets survived. Each keep panel also carries a
zoom of the target neighbourhood, because at r=10% the decisive evidence is a
few cells wide and invisible at page scale.

Writes artifacts/figures/fig5_keepmaps.pdf (filename kept so the .tex
include is unchanged).
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
from fig_style import C, apply, save  # noqa: E402

DATA = os.path.join(HERE, "..", "logs", "w36_qual_local")
SWEEP = os.path.join(HERE, "..", "logs", "w44_sweep_local")
W44 = os.path.join(HERE, "..", "docs", "w44_fig_evidence.json")
FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
os.makedirs(FIGDIR, exist_ok=True)

FIG_IDX = int(os.environ.get("W59_FIG9_IDX", "62"))
ARMS = ["ours", "nocrs", "nomass", "uniform"]


def load_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_ssv2():
    dumps = {a: load_jsonl(os.path.join(DATA, f"ssv2_{a}_l120.jsonl"))
             for a in ARMS}
    rows, hdr = {}, None
    for ln in open(os.path.join(DATA, "harvest_l120", "chooser.tsv"),
                   encoding="utf-8"):
        parts = ln.rstrip("\n").split("\t")
        if hdr is None:
            hdr = parts
            continue
        rows[int(parts[0])] = dict(zip(hdr, parts))
    for d in json.load(open(os.path.join(DATA, "gt120.json"),
                            encoding="utf-8")):
        if d["idx"] in rows:
            rows[d["idx"]]["bbox_xywh"] = d.get("bbox")
            rows[d["idx"]]["instruction"] = d.get("instruction") or \
                rows[d["idx"]].get("instruction", "")
    return dumps, rows


def load_sweep(arm, key):
    if key == "r10":
        return load_jsonl(os.path.join(DATA, f"ssv2_{arm}_l120.jsonl"))
    return load_jsonl(os.path.join(SWEEP, f"ssv2_{arm}_{key}_l120.jsonl"))


def local_img(remote_path):
    return os.path.join(DATA, "images", os.path.basename(remote_path))


def box_witnessed(rec, bbox, W, H):
    """Does any kept cell centre fall inside the target box?"""
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    gh, gw = rec["grid_h"], rec["grid_w"]
    cw, ch = W / gw, H / gh
    for j in rec["local_keep"]:
        cx, cy = (j % gw + 0.5) * cw, (j // gw + 0.5) * ch
        if x <= cx <= x + bw and y <= cy <= y + bh:
            return True
    return False


def zoom_extent(bbox, W, H, pad_frac=1.25, min_frac=0.16, aspect=None):
    """Crop around the target, padded so context stays visible. `aspect` is the
    desired width/height of the crop so the panel fills without side margins."""
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    cx, cy = x + bw / 2.0, y + bh / 2.0
    half = max(bw, bh) * (0.5 + pad_frac)
    half = max(half, min_frac * min(W, H) / 2.0)
    hx = hy = half
    if aspect:
        if aspect >= 1.0:
            hx = half * aspect
        else:
            hy = half / aspect
    # keep the crop inside the image by shifting the centre, not shrinking it
    hx, hy = min(hx, W / 2.0), min(hy, H / 2.0)
    cx = min(max(cx, hx), W - hx)
    cy = min(max(cy, hy), H - hy)
    return cx - hx, cx + hx, cy - hy, cy + hy


def build():
    apply(scale=2.0)
    kv.enable_cjk()
    dumps, rows = load_ssv2()
    i = FIG_IDX
    meta = rows[i]
    rec = {a: dumps[a][i] for a in ARMS}
    img = Image.open(local_img(rec["ours"]["image_path"])).convert("RGB")
    w, h = img.size
    mass = json.load(open(
        os.path.join(DATA, "harvest_l120", f"mass_{i:03d}.json"),
        encoding="utf-8"))
    gh, gw = rec["ours"]["grid_h"], rec["ours"]["grid_w"]
    P = np.asarray(mass["P"], dtype=float).reshape(gh, gw)
    keeps = {a: kv.keep_grid(rec[a]) for a in ARMS}
    witness = keeps["ours"] & ~keeps["nocrs"]
    bbox = meta.get("bbox_xywh")

    strip = build_strip()
    return fig9(img, w, h, P, mass, keeps, witness, meta, bbox, rec, strip)


def build_strip():
    """Budget filmstrip: one multi-instruction screen at four budgets."""
    W44D = json.load(open(W44, encoding="utf-8"))
    cb = W44D.get("cross_budget", {})
    if not cb.get("available"):
        return None
    pick = cb["filmstrip_pick"]
    fi = pick["anchor_idx"]
    scr = next(s for s in W44D["multi_screens"] if s["image"] == pick["image"])
    im = Image.open(os.path.join(DATA, "images", pick["image"])).convert("RGB")
    budgets = [("r50", "50"), ("r25", "25"), ("r10", "10"), ("r05", "5")]
    return {"idx": fi, "img": im, "screen": scr, "budgets": budgets,
            "recs": {(a, k): load_sweep(a, k)[fi]
                     for a in ("cdpruner", "ours") for k, _ in budgets}}


PANELS = [
    ("ours", "(c) TRACE keep", C["ours"]),
    ("nomass", "(d) w/o prior mass", "#56B4E9"),
    ("uniform", "(e) Uniform", "#999999"),
]


def fig9(img, w, h, P, mass, keeps, witness, meta, bbox, rec, strip):
    ar = h / w
    pw = 2.15                      # per-panel width, 11in design at x0.50
    zoom_h = pw * 0.62
    row1_h = pw * ar + zoom_h + 1.05
    fw_, fh_ = strip["img"].size
    strip_w = (5 * pw + 0.3 - 0.75) / 4.0
    row2_h = 2 * strip_w * (fh_ / fw_) + 1.00

    fig = plt.figure(figsize=(5 * pw + 0.3, row1_h + row2_h))
    gs0 = fig.add_gridspec(2, 1, height_ratios=[row1_h, row2_h], hspace=0.20,
                           top=0.985, bottom=0.012, left=0.008, right=0.992)
    # each column is [screenshot panel; target zoom]
    top = gs0[0].subgridspec(2, 5, wspace=0.055, hspace=0.10,
                             height_ratios=[pw * ar, zoom_h])
    axes_main = [fig.add_subplot(top[0, c]) for c in range(5)]
    axes_zoom = [fig.add_subplot(top[1, c]) for c in range(5)]

    def plabel(ax, s):
        ax.set_title(s, loc="left", fontweight="bold", fontsize=13.2, pad=3.5)

    zoom_ar = pw / zoom_h          # panel width/height, so the crop fills it

    def zoom_panel(axz, mask=None, kept=None):
        """Same content as the column above, cropped to the target region and
        drawn with cell gridlines so individual token decisions are countable."""
        if mask is None:
            kv.show_image(axz, img)
        else:
            kv.draw_keep(axz, img, mask, grid_lw=0.5)
        zoom_panel_axes(axz, bbox, w, h,
                        kept=(True if kept is None else kept), aspect=zoom_ar)

    # (a) the clean input -- the panel the previous version never showed
    ax = axes_main[0]
    kv.show_image(ax, img)
    kv.target_box(ax, bbox, w, h, kept=True, lw=3.0, cross_on_miss=False)
    plabel(ax, "(a) Input screenshot")
    ax.set_xlabel("Unmodified pixels.\nGreen box $=$ action target",
                  fontsize=11.4, labelpad=3)
    zoom_panel(axes_zoom[0])

    # (b) the query-independent prior field
    ax = axes_main[1]
    kv.score_heat(ax, img, P, vmax=max(P.max(), 1e-6))
    kv.element_boxes(ax, mass["boxes"], w, h, ec=C["best"], lw=1.5, alpha=0.85)
    plabel(ax, "(b) LIP prior mass")
    lift = json.load(open(os.path.join(HERE, "..", "docs",
                                       "w36_qual_stats.json"),
                          encoding="utf-8"))["mass_lift"]
    ax.set_xlabel(f"{len(mass['boxes'])} detected regions.\nGT-box lift "
                  f"{lift['per_frame'][FIG_IDX]:.1f}$\\times$ "
                  f"(median {lift['median']:.1f}$\\times$)",
                  fontsize=11.4, labelpad=3)
    axz = axes_zoom[1]
    kv.score_heat(axz, img, P, vmax=max(P.max(), 1e-6))
    kv.element_boxes(axz, mass["boxes"], w, h, ec=C["best"], lw=1.8)
    zoom_panel_axes(axz, bbox, w, h, aspect=zoom_ar)

    # (c)-(e) the three keep maps, all in the kept-pixels convention
    for col, (arm, lab, tint) in enumerate(PANELS, start=2):
        ax = axes_main[col]
        mask = keeps[arm]
        kv.draw_keep(ax, img, mask)
        hit = meta[arm] == "1"
        kv.target_box(ax, bbox, w, h, kept=hit, lw=3.0)
        plabel(ax, lab)
        kv.keep_badge(ax, mask.sum() / mask.size, hit=hit, fontsize=11.0)
        if arm == "ours":
            n_w = int(witness.sum())
            ax.set_xlabel(f"$k{{=}}{int(mask.sum())}$ tokens kept.\n"
                          f"{n_w} added by NCR repair", fontsize=11.4,
                          labelpad=3)
        elif arm == "nomass":
            ax.set_xlabel("Keep collapses onto\nhigh-texture background",
                          fontsize=11.4, labelpad=3)
        else:
            ax.set_xlabel("Target-blind stripes\nignore the layout",
                          fontsize=11.4, labelpad=3)
        zoom_panel(axes_zoom[col], mask=mask, kept=hit)

    # Encoding legend, placed under the zoom strip where there is free space
    # rather than above the panels where it would collide with panel titles.
    kv.query_caption(fig, axes_zoom[:5], meta.get("instruction", ""),
                     fontsize=12.2, dy=0.007)
    zb = [a.get_position() for a in axes_zoom]
    fig.text(0.5 * (min(b.x0 for b in zb) + max(b.x1 for b in zb)),
             min(b.y0 for b in zb) - 0.030,
             "Bottom strip zooms the target neighbourhood.  Kept cells keep the "
             "original pixels inside the white contour\nwhile dropped cells are "
             "dimmed but still legible.  Green box $=$ target retained, red "
             "crossed box $=$ target lost.",
             fontsize=11.0, ha="center", va="top", color="#333333",
             linespacing=1.45)

    draw_filmstrip(fig, gs0[1], strip, row1_h, row2_h)
    save(fig, os.path.join(FIGDIR, "fig5_keepmaps.pdf"))


def zoom_panel_axes(axz, bbox, w, h, kept=True, aspect=None):
    if bbox:
        kv.target_box(axz, bbox, w, h, kept=kept, lw=2.8,
                      cross_on_miss=False)
        x0, x1, y0, y1 = zoom_extent(bbox, w, h, aspect=aspect)
        axz.set_xlim(x0, x1)
        axz.set_ylim(y1, y0)
    axz.set_xticks([])
    axz.set_yticks([])
    for s in axz.spines.values():
        s.set_linewidth(1.2)
        s.set_edgecolor("#333333")


def draw_filmstrip(fig, cell, strip, row1_h, row2_h):
    """(f) one screen carrying four instructions, budget tightening 50 -> 5%."""
    fig.text(0.008, 1.0 - row1_h / (row1_h + row2_h) + 0.010,
             "(f) One screen with four instructions, budget tightening. "
             "Kept regions stay readable, dropped regions go black",
             fontsize=13.2, fontweight="bold", va="bottom")
    gs = cell.subgridspec(2, 5, width_ratios=[0.10, 1, 1, 1, 1],
                          wspace=0.035, hspace=0.14)
    im, scr = strip["img"], strip["screen"]
    fw, fh = im.size
    for r_i, (arm, lab, col) in enumerate(
            (("cdpruner", "CDPruner", "#009E73"),
             ("ours", "TRACE", C["ours"]))):
        axl = fig.add_subplot(gs[r_i, 0])
        axl.axis("off")
        axl.text(0.42, 0.5, lab, rotation=90, va="center", ha="center",
                 fontsize=12.0, fontweight="bold",
                 color=col if arm == "ours" else "#1B6E54")
        for c_i, (key, blab) in enumerate(strip["budgets"]):
            ax = fig.add_subplot(gs[r_i, c_i + 1])
            rec_f = strip["recs"][(arm, key)]
            kv.draw_keep(ax, im, kv.keep_grid(rec_f))
            n_lost = 0
            for rr in scr["instructions"]:
                okb = box_witnessed(rec_f, rr["bbox"], fw, fh)
                kv.target_box(ax, rr["bbox"], fw, fh, kept=okb, lw=2.0)
                n_lost += 0 if okb else 1
            if r_i == 0:
                ax.set_title("$r{=}" + blab + "\\%$", fontsize=12.4, pad=3)
            n_t = len(scr["instructions"])
            ax.set_xlabel(f"{n_t - n_lost}/{n_t} targets retained",
                          fontsize=10.6, labelpad=2,
                          color=("#B3261E" if n_lost else "#1B6E54"))


def main():
    build()


if __name__ == "__main__":
    main()
