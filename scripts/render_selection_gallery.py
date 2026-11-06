#!/usr/bin/env python
"""Selection gallery (appendix): eight figures that show, from real dumps, what
the selector keeps and what it discards.

Every panel uses the shared convention in `keepviz`: kept cells carry the
original screenshot pixels inside a white contour, dropped cells are dimmed but
remain legible, so the reader can always see the input that a decision was made
about.

  G1  keep gallery across screens at a fixed budget
  G2  budget ladder on one screen, r = 50/25/10/5%
  G3  cache lifecycle matrix: input rail, upper-triangle keep, story panel
  G4  retirement anatomy across several episodes
  G5  nested retirement vs an unconstrained re-election
  G6  detector elements against the token grid
  G7  selector comparison at an identical budget
  G8  failure cases

Data: logs/w36_qual_local (ScreenSpot-v2, 120 instrumented frames),
logs/w44_sweep_local (budget sweep), logs/w59_viz_local (OmniGUI probe subset,
per-frame Module-A replay, population lifecycle stats).
"""
import json
import os
import re
import sys

import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import keepviz as kv  # noqa: E402
from fig_style import C, apply, save, despine  # noqa: E402

SS = os.path.join(HERE, "..", "logs", "w36_qual_local")
SWEEP = os.path.join(HERE, "..", "logs", "w44_sweep_local")
VIZ = os.path.join(HERE, "..", "logs", "w59_viz_local")
W44 = os.path.join(HERE, "..", "docs", "w44_fig_evidence.json")
STATS = os.path.join(HERE, "..", "docs", "w36_qual_stats.json")
FIGDIR = os.path.join(HERE, "..", "artifacts", "figures")
os.makedirs(FIGDIR, exist_ok=True)

ARMS = ("ours", "nocrs", "nomass", "nodiv", "aonly", "uniform", "divprune",
        "cdpruner")


def load_jsonl(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def load_ss():
    dumps = {}
    for a in ARMS:
        p = os.path.join(SS, f"ssv2_{a}_l120.jsonl")
        if os.path.exists(p):
            dumps[a] = load_jsonl(p)
    rows, hdr = {}, None
    for ln in open(os.path.join(SS, "harvest_l120", "chooser.tsv"),
                   encoding="utf-8"):
        parts = ln.rstrip("\n").split("\t")
        if hdr is None:
            hdr = parts
            continue
        rows[int(parts[0])] = dict(zip(hdr, parts))
    for d in json.load(open(os.path.join(SS, "gt120.json"), encoding="utf-8")):
        if d["idx"] in rows:
            rows[d["idx"]]["bbox_xywh"] = d.get("bbox")
            rows[d["idx"]]["instruction"] = d.get("instruction") or \
                rows[d["idx"]].get("instruction", "")
            rows[d["idx"]]["dtype"] = d.get("data_type",
                                            rows[d["idx"]].get("dtype", ""))
    return dumps, rows


def ss_img(rec):
    return Image.open(os.path.join(
        SS, "images", os.path.basename(rec["image_path"]))).convert("RGB")


def ss_mass(idx):
    p = os.path.join(SS, "harvest_l120", f"mass_{idx:03d}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def load_omni():
    by_ep = {}
    for r in load_jsonl(os.path.join(VIZ, "probe_subset.jsonl")):
        by_ep.setdefault(str(r["episode_key"]).split("-")[-1], []).append(r)
    led = json.load(open(os.path.join(VIZ, "episodes.json"), encoding="utf-8"))
    stats = json.load(open(os.path.join(VIZ, "lifecycle_stats.json"),
                           encoding="utf-8"))
    return by_ep, led, stats


def omni_img(ep, step):
    return Image.open(os.path.join(
        VIZ, "images", f"{ep}_{step:02d}.png")).convert("RGB")


def covered(rec, bbox, img, key="local_keep"):
    """Does a kept token's centre land inside the box?"""
    if not bbox:
        return None
    W, H = img.size
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    gh, gw = rec["grid_h"], rec["grid_w"]
    cw, ch = W / gw, H / gh
    for j in rec[key]:
        cx, cy = (j % gw + 0.5) * cw, (j // gw + 0.5) * ch
        if x <= cx <= x + bw and y <= cy <= y + bh:
            return True
    return False


def shorten(s, n=54):
    s = str(s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "\u2026"


def _omni_rec_step(rec):
    p = rec.get("image_path") or ""
    base = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(base) - 1
    except ValueError:
        return int(rec.get("frame_index") or 0)


def action_clauses(instruction):
    parts = re.split(r"[；;]", str(instruction or ""))
    return [p.strip() for p in parts if p.strip()]


def short_action(clause, action_type=None):
    """One-line tap label from an OmniGUI instruction clause. No semicolons."""
    if int(action_type or 0) == 10:
        return "complete"
    cl = str(clause or "").strip()
    if not cl:
        return "tap"
    m = re.search(r"[‘'']([^’']+)[’']", cl)
    if m:
        return "tap " + m.group(1).strip()
    m = re.search(r"(?:click|tap)\s+([^，,。.\s]+)", cl, flags=re.I)
    if m:
        return "tap " + m.group(1).strip()
    return shorten(cl, 18)


# One-line task titles for the G3 header (no semicolons, fig_style rule 4).
G3_TASK = {
    "T0039": "Open Yu'ebao from Me, then the red-packet / Fuka flow",
    "T0110": "Enter the airport as destination, then tap ride now",
    "T0102": "On the fare-estimate page, choose a rider",
}

# Drop a later frame as a near-duplicate when its MAE to the previous kept
# frame is below this fraction of the episode's median adjacent-frame MAE.
G3_NEAR_DUP_FRAC = 0.40


def pick_diverse_steps(ep, recs, led_steps, n=6):
    """Task-order subsequence of at most `n` frames for the G3 matrix.

    1. Drop finished/complete steps (`action_type == 10`).
    2. Drop near-duplicate animation frames: adjacent-pixel MAE below
       `G3_NEAR_DUP_FRAC` of the episode median, but never so many that
       fewer than `n` frames remain.
    3. Keep the length-`n` prefix, so the instruction order (and the
       opening screen) is preserved. Maximising path MAE is the wrong
       objective: it skips structurally necessary pages that share chrome
       with a neighbour (the Me tab vs. the Alipay home screen).
    """
    n_all = len(recs)
    n = min(int(n), n_all)
    thumbs = [np.asarray(omni_img(ep, i).convert("RGB").resize((90, 200)),
                         dtype=np.float32) for i in range(n_all)]
    adj = [float(np.mean(np.abs(thumbs[i] - thumbs[i + 1])))
           for i in range(n_all - 1)]
    med = float(np.median(adj)) if adj else 0.0
    cutoff = G3_NEAR_DUP_FRAC * med

    cands = [i for i in range(n_all)
             if int((led_steps[i] if i < len(led_steps) else {}
                     ).get("action_type") or 0) != 10]
    if not cands:
        cands = list(range(n_all))
    kept = [cands[0]]
    for idx, i in enumerate(cands[1:], start=1):
        mae = float(np.mean(np.abs(thumbs[kept[-1]] - thumbs[i])))
        remain_incl = len(cands) - idx
        can_drop = (len(kept) + remain_incl - 1) >= n
        if mae < cutoff and can_drop:
            continue
        kept.append(i)
    print(f"g3 {ep}: median adj MAE {med:.1f}, near-dup cutoff {cutoff:.1f}, "
          f"selected {kept[:n]}")
    return kept[:n]


# --------------------------------------------------------------------------
# G1: keep gallery -- input / prior field / keep, over several screens
# --------------------------------------------------------------------------
G1_IDX = [int(x) for x in os.environ.get(
    "W59_G1", "62,35,53,78,10,101").split(",")]


def g1(dumps, rows):
    apply(scale=1.55)
    picks = []
    for i in G1_IDX:
        rec = dumps["ours"][i]
        picks.append((i, rec, ss_img(rec), ss_mass(i), rows[i]))
    # keep only one orientation so the grid has no ragged empty bands
    port = [p for p in picks if p[2].size[1] >= p[2].size[0]]
    picks = (port or picks)[:6]
    n = len(picks)
    ar = float(np.median([im.size[1] / im.size[0] for _, _, im, _, _ in picks]))
    cw = (6.9 - 0.10) / n
    fig = plt.figure(figsize=(6.9, 3 * cw * ar + 1.30))
    gs = fig.add_gridspec(3, n, wspace=0.05, hspace=0.30, left=0.052,
                          right=0.992, top=0.965, bottom=0.052)
    row_axes = {0: [], 1: [], 2: []}
    for c, (i, rec, img, mass, meta) in enumerate(picks):
        w, h = img.size
        mask = kv.keep_grid(rec)
        hit = meta["ours"] == "1"

        ax = fig.add_subplot(gs[0, c])
        row_axes[0].append(ax)
        kv.show_image(ax, img)
        kv.target_box(ax, meta["bbox_xywh"], w, h, kept=True, lw=1.5,
                      cross_on_miss=False)
        ax.set_xlabel(shorten(meta.get("instruction", ""), 34), fontsize=6.0,
                      style="italic", labelpad=1.6)

        ax = fig.add_subplot(gs[1, c])
        row_axes[1].append(ax)
        if mass:
            P = np.asarray(mass["P"], float).reshape(mass["grid_h"],
                                                     mass["grid_w"])
            kv.score_heat(ax, img, P, vmax=max(P.max(), 1e-6))
            kv.element_boxes(ax, mass["boxes"], w, h, ec=C["best"], lw=0.5)
            ax.set_xlabel(f"{len(mass['boxes'])} elements", fontsize=6.2,
                          labelpad=1.6)
        else:
            kv.show_image(ax, img)

        ax = fig.add_subplot(gs[2, c])
        row_axes[2].append(ax)
        kv.draw_keep(ax, img, mask, outline_lw=0.7)
        kv.target_box(ax, meta["bbox_xywh"], w, h, kept=hit, lw=1.5)
        kv.keep_badge(ax, mask.sum() / mask.size, hit=hit, fontsize=5.8)
        ax.set_xlabel(meta.get("dtype", ""), fontsize=6.2, labelpad=1.6)

    for r, lab in enumerate(("Input", "LIP prior mass", "TRACE keep")):
        kv.row_label(fig, row_axes[r], lab, fontsize=8.0, dx=0.010)
    fig.text(0.5, 0.010, "Six evaluated ScreenSpot-v2 frames at $r{=}10\\%$. "
             "Kept cells keep the original pixels inside the white contour, "
             "dropped cells stay visible but dimmed.",
             fontsize=7.0, ha="center", va="bottom", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG1_keep_gallery.pdf"))


# --------------------------------------------------------------------------
# G2: budget ladder on one multi-instruction screen
# --------------------------------------------------------------------------
BUDGETS = [("r50", "50"), ("r25", "25"), ("r10", "10"), ("r05", "5")]
G2_ARMS = [("ours", "TRACE"), ("cdpruner", "CDPruner"),
           ("divprune", "DivPrune"), ("uniform", "Uniform")]


def sweep_rec(arm, key, idx):
    if key == "r10":
        return load_jsonl(os.path.join(SS, f"ssv2_{arm}_l120.jsonl"))[idx]
    p = os.path.join(SWEEP, f"ssv2_{arm}_{key}_l120.jsonl")
    return load_jsonl(p)[idx] if os.path.exists(p) else None


def g2():
    apply(scale=1.45)
    W44D = json.load(open(W44, encoding="utf-8"))
    pick = W44D["cross_budget"]["filmstrip_pick"]
    idx = pick["anchor_idx"]
    scr = next(s for s in W44D["multi_screens"] if s["image"] == pick["image"])
    img = Image.open(os.path.join(SS, "images", pick["image"])).convert("RGB")
    w, h = img.size

    arms = [(a, lab) for a, lab in G2_ARMS
            if sweep_rec(a, "r50", idx) is not None]
    nb = len(BUDGETS)
    cw = (6.9 - 0.35) / nb
    fig = plt.figure(figsize=(6.9, len(arms) * cw * (h / w) + 1.05))
    gs = fig.add_gridspec(len(arms), nb, wspace=0.04, hspace=0.24,
                          left=0.062, right=0.992, top=0.955, bottom=0.062)
    for r, (arm, lab) in enumerate(arms):
        axes_r = []
        for c, (key, blab) in enumerate(BUDGETS):
            rec = sweep_rec(arm, key, idx)
            ax = fig.add_subplot(gs[r, c])
            axes_r.append(ax)
            kv.draw_keep(ax, img, kv.keep_grid(rec), outline_lw=0.55)
            lost = 0
            for rr in scr["instructions"]:
                ok = covered(rec, rr["bbox"], img)
                kv.target_box(ax, rr["bbox"], w, h, kept=ok, lw=1.3)
                lost += 0 if ok else 1
            n_t = len(scr["instructions"])
            if r == 0:
                ax.set_title("$r{=}" + blab + "\\%$", fontsize=8.0, pad=2.5)
            ax.set_xlabel(f"{n_t - lost}/{n_t} targets",
                          fontsize=6.6, labelpad=1.5,
                          color=("#B3261E" if lost else "#1B6E54"))
        kv.row_label(fig, axes_r, lab, fontsize=8.0, dx=0.011)
    fig.text(0.5, 0.010, "One screen carrying four instructions. Green boxes "
             "are targets still covered by a kept token, red crossed boxes are "
             "targets the selection lost.",
             fontsize=7.0, ha="center", va="bottom", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG2_budget_ladder.pdf"))


# --------------------------------------------------------------------------
# G3: input rail + upper-triangle cache + lower-left story panel
# --------------------------------------------------------------------------
S_DROP, S_RETIRE, S_SURVIVE, S_ADMIT = 0, 1, 2, 3
STATE_COLORS = {S_RETIRE: kv.LIFECYCLE["retire"],
                S_SURVIVE: kv.LIFECYCLE["survive"],
                S_ADMIT: kv.LIFECYCLE["admit"]}
HIST_N = 5
G3_STORY_BG = "#F6F3EC"
G3_IMG_MAX_W = 400


def _g3_img(ep, step):
    img = omni_img(ep, step)
    w, h = img.size
    if w <= G3_IMG_MAX_W:
        return img
    nw = G3_IMG_MAX_W
    nh = max(1, int(round(h * nw / w)))
    return img.resize((nw, nh), Image.LANCZOS)


def _g3_box(bbox, src_wh, dst_wh):
    if not bbox or not src_wh or src_wh[0] <= 0:
        return bbox
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    return [x * sx, y * sy, bw * sx, bh * sy]


def _g3_story(ax, labels, recs, sel):
    """Lower-left span: step list, colour key, token occupancy sparkline."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(G3_STORY_BG)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#C9C2B6")
        sp.set_linewidth(0.8)
    ax.text(0.055, 0.985, "How to read the matrix", transform=ax.transAxes,
            fontsize=7.6, fontweight="bold", va="top", ha="left")
    ax.text(0.055, 0.948,
            "Left column: the screen at that step.\n"
            "Diagonal: admitted at $k_c$, amber will retire,\n"
            "orange will hold.\n"
            "Right of the diagonal: the same frame held at $k_h$.\n"
            "A row does not change (write-once).\n"
            "A new row is a new page.",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            linespacing=1.32, color="#222222")
    ax.text(0.055, 0.775, "Actions", transform=ax.transAxes,
            fontsize=7.2, fontweight="bold", va="top")
    y = 0.742
    for i, lab in enumerate(labels):
        ax.text(0.07, y, f"{i + 1}   {lab}", transform=ax.transAxes,
                fontsize=6.9, va="top", ha="left")
        y -= 0.032

    ax.text(0.055, 0.535, "Cache colour", transform=ax.transAxes,
            fontsize=7.2, fontweight="bold", va="top")
    chips = [
        (kv.LIFECYCLE["admit"], "will retire"),
        (kv.LIFECYCLE["survive"], "hold (write-once)"),
        ("#4A4A4A", "dropped (ghost)"),
    ]
    cy = 0.502
    for col, lab in chips:
        ax.add_patch(mpatches.Rectangle(
            (0.07, cy - 0.009), 0.048, 0.020,
            transform=ax.transAxes, clip_on=False,
            facecolor=col, edgecolor="#555555", linewidth=0.35))
        ax.text(0.135, cy + 0.001, lab, transform=ax.transAxes,
                fontsize=6.5, va="center", ha="left")
        cy -= 0.030

    kc = [len(recs[f]["keep_cur"]) for f in sel]
    kh = [len(recs[f]["keep_hist_nested"]) for f in sel]
    win = []
    for s, _f in enumerate(sel):
        tot = kc[s]
        for u in range(max(0, s - HIST_N), s):
            tot += kh[u]
        win.append(tot)
    iax = ax.inset_axes([0.14, 0.05, 0.78, 0.28])
    xs = np.arange(1, len(sel) + 1)
    iax.plot(xs, kc, color=kv.LIFECYCLE["admit"], marker="o", ms=3.6,
             lw=1.3, label=r"$k_c$ admit")
    iax.plot(xs, kh, color=kv.LIFECYCLE["survive"], marker="s", ms=3.4,
             lw=1.3, label=r"$k_h$ hold")
    iax.plot(xs, win, color="#333333", ls="--", marker=".", ms=5.0,
             lw=1.2, label="window total")
    iax.set_xticks(list(xs))
    iax.set_xticklabels([str(int(v)) for v in xs])
    iax.set_xlim(0.4, len(sel) + 0.6)
    iax.set_ylim(0, max(win + kc + kh) * 1.22)
    iax.set_xlabel("step", fontsize=6.0, labelpad=1.0)
    iax.set_ylabel("tokens", fontsize=6.0, labelpad=1.5)
    iax.tick_params(labelsize=5.6, length=2.0)
    iax.set_facecolor("#FFFCF7")
    despine(iax)
    iax.legend(loc="upper left", fontsize=5.6, frameon=False,
               handlelength=1.6, borderaxespad=0.15, labelspacing=0.18)
    iax.set_title("Tokens in the window", fontsize=6.6, loc="left", pad=3.0)


def g3(by_ep, led, ep=None, steps=None):
    ep = ep or os.environ.get("W59_G3_EP", "T0039")
    apply(scale=1.30)
    kv.enable_cjk()
    recs = sorted(by_ep[ep], key=_omni_rec_step)
    led_ep = led[ep]
    led_steps = led_ep["steps"]
    n_max = steps or min(int(os.environ.get("W59_G3_STEPS", "6")), len(recs))
    env_sel = os.environ.get("W59_G3_SEL", "").strip()
    if env_sel:
        sel = [int(x) for x in env_sel.split(",") if x.strip() != ""]
    else:
        sel = pick_diverse_steps(ep, recs, led_steps, n=n_max)
    n = len(sel)
    clauses = action_clauses(led_steps[0].get("instruction", "") if led_steps else "")
    labels = []
    for f in sel:
        at = (led_steps[f] if f < len(led_steps) else {}).get("action_type")
        cl = clauses[f] if f < len(clauses) else ""
        labels.append(short_action(cl, at))

    im0 = _g3_img(ep, sel[0])
    ar = im0.size[1] / im0.size[0]
    ncols = n + 1
    cw = (6.9 - 0.38) / ncols
    fig = plt.figure(figsize=(6.9, n * (cw * ar + 0.14) + 1.05))
    gs = fig.add_gridspec(
        n, ncols, width_ratios=[1.18] + [1.0] * n,
        wspace=0.048, hspace=0.22,
        left=0.046, right=0.992, top=0.918, bottom=0.028)
    # Largest rectangle that sits entirely in the empty lower triangle.
    story_k = n // 2

    for i, f in enumerate(sel):
        rec = recs[f]
        img = _g3_img(ep, f)
        w, h = img.size
        gh, gw = rec["grid_h"], rec["grid_w"]
        cur = kv.idx_grid(rec["keep_cur"], gh, gw)
        hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)
        src_wh = (led_steps[f].get("image_wh") or [w, h]) if f < len(led_steps) else [w, h]
        tgt = _g3_box((led_steps[f] if f < len(led_steps) else {}).get("target_xywh"),
                      src_wh, (w, h))

        ax = fig.add_subplot(gs[i, 0])
        kv.show_image(ax, img)
        kv.panel_tag(ax, f"{i + 1}", fontsize=6.2)
        ax.set_xlabel(labels[i], fontsize=5.8, labelpad=1.0)
        if i == 0:
            ax.set_title("input", fontsize=7.2, pad=2.2)

        for j in range(i, n):
            ax = fig.add_subplot(gs[i, j + 1])
            st = np.zeros((gh, gw), dtype=int)
            lag = j - i
            if j == i:
                st[cur & ~hist] = S_ADMIT
                st[hist] = S_SURVIVE
            elif lag <= HIST_N:
                st[hist] = S_SURVIVE
            if lag > HIST_N:
                kv.show_image(ax, img)
                ax.imshow(np.full((2, 2, 4), (0.30, 0.30, 0.30, 0.82)),
                          extent=(0, w, h, 0), zorder=3)
                kv.panel_tag(ax, "left cache", fontsize=5.0)
            else:
                kv.draw_states(ax, img, st, STATE_COLORS, alpha=0.52,
                               ghost=kv.GHOST_SOFT,
                               outline_states=(S_SURVIVE,), outline_lw=0.4)
            if j == i:
                for sp in ax.spines.values():
                    sp.set_linewidth(1.55)
                    sp.set_edgecolor(kv.LIFECYCLE["admit"])
                ax.set_xlabel(f"$k_c$ {int(cur.sum())}", fontsize=5.8,
                              labelpad=1.0, color="#8A5A00")
                if tgt:
                    kv.target_box(ax, tgt, w, h,
                                  kept=covered(rec, tgt, img, "keep_cur"),
                                  lw=1.35)
            elif lag <= HIST_N:
                ax.set_xlabel(f"$k_h$ {int(hist.sum())}", fontsize=5.8,
                              labelpad=1.0, color="#9A4A00")
            if i == 0:
                ax.set_title(f"step {j + 1}", fontsize=7.2, pad=2.2)

    sax = fig.add_subplot(gs[story_k:, 1:story_k + 1])
    _g3_story(sax, labels, recs, sel)

    app = str(led_ep.get("app") or "").replace("alipay", "Alipay").replace("didi", "DiDi")
    task = G3_TASK.get(ep) or shorten(
        str(led_steps[0].get("instruction", "")).replace("；", ", ").replace(";", ", "),
        78)
    fig.text(0.046, 0.982, f"{app} {ep}. {task}.",
             fontsize=7.8, fontweight="bold", va="top")
    fig.text(0.046, 0.952,
             r"Budget $(k_c,\ k_h)=(50\%,\ 10\%)$. "
             "Left column: the input. Upper triangle: cache state of that frame.",
             fontsize=6.8, va="top", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG3_lifecycle_matrix.pdf"))


# --------------------------------------------------------------------------
# G4: retirement anatomy across episodes
# --------------------------------------------------------------------------
G4_PICKS = [t.split(":") for t in os.environ.get(
    "W59_G4", "T0094:0,T0102:2,T0110:1,T0039:3").split(",")]


def g4(by_ep, led):
    apply(scale=1.42)
    picks = [(ep, int(s)) for ep, s in G4_PICKS if ep in by_ep]
    ar = float(np.median([omni_img(ep, s).size[1] / omni_img(ep, s).size[0]
                          for ep, s in picks]))
    cw = (6.9 - 0.32) / 4.0
    fig = plt.figure(figsize=(6.9, len(picks) * cw * ar + 1.0))
    gs = fig.add_gridspec(len(picks), 4, wspace=0.045, hspace=0.26,
                          left=0.058, right=0.992, top=0.952, bottom=0.058)
    for r, (ep, s) in enumerate(picks):
        rec = by_ep[ep][s]
        img = omni_img(ep, s)
        w, h = img.size
        gh, gw = rec["grid_h"], rec["grid_w"]
        cur = kv.idx_grid(rec["keep_cur"], gh, gw)
        hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)
        tgt = led[ep]["steps"][s].get("target_xywh")
        axes_r = []

        ax = fig.add_subplot(gs[r, 0])
        axes_r.append(ax)
        kv.show_image(ax, img)
        if tgt:
            kv.target_box(ax, tgt, w, h, kept=True, lw=1.4,
                          cross_on_miss=False)
        if r == 0:
            ax.set_title("input frame", fontsize=7.6, pad=2.5)
        ax.set_xlabel(f"{gh * gw} tokens", fontsize=6.2, labelpad=1.5)

        for c, (mask, col, title, lab) in enumerate([
                (cur, kv.LIFECYCLE["admit"], "admitted at entry",
                 f"$k_c{{=}}{int(cur.sum())}$"),
                (hist, kv.LIFECYCLE["survive"], "survives retirement",
                 f"$k_h{{=}}{int(hist.sum())}$"),
                (cur & ~hist, kv.LIFECYCLE["retire"], "evicted",
                 f"{int((cur & ~hist).sum())} freed")], start=1):
            ax = fig.add_subplot(gs[r, c])
            axes_r.append(ax)
            kv.draw_keep(ax, img, mask, tint=col, tint_alpha=0.28,
                         ghost=kv.GHOST_SOFT, outline_lw=0.55)
            if tgt:
                kv.target_box(ax, tgt, w, h,
                              kept=covered(rec, tgt, img, "keep_hist_nested")
                              if c == 2 else covered(rec, tgt, img, "keep_cur"),
                              lw=1.4, cross_on_miss=(c != 3))
            if r == 0:
                ax.set_title(title, fontsize=7.6, pad=2.5)
            ax.set_xlabel(lab, fontsize=6.2, labelpad=1.5)
        kv.row_label(fig, axes_r, f"{ep} s{s + 1}", fontsize=7.2, dx=0.010,
                     weight="normal")
    fig.text(0.5, 0.008, "Retirement on four OmniGUI frames. The survivor set "
             "is the rank prefix of the admitted set, so the three keep panels "
             "of each row are strictly nested.",
             fontsize=7.0, ha="center", va="bottom", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG4_retirement_anatomy.pdf"))


# --------------------------------------------------------------------------
# G5: nested retirement vs unconstrained re-election
# --------------------------------------------------------------------------
def g5(by_ep, led, stats):
    apply(scale=1.42)
    picks = [(ep, int(s)) for ep, s in G4_PICKS if ep in by_ep][:3]
    ep, s = picks[0]
    im = omni_img(ep, s)
    ar = im.size[1] / im.size[0]
    cw = (6.9 - 0.10) / 4.0
    head_h, foot_h = 0.60, 0.30
    tot = cw * ar + head_h + foot_h + 0.30
    fig = plt.figure(figsize=(6.9, tot))
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.014, right=0.992,
                          top=1 - head_h / tot, bottom=(foot_h + 0.24) / tot)
    rec = by_ep[ep][s]
    img = im
    w, h = img.size
    gh, gw = rec["grid_h"], rec["grid_w"]
    hist = kv.idx_grid(rec["keep_hist_nested"], gh, gw)
    indep = kv.idx_grid(rec["keep_hist_independent"], gh, gw)

    for c, (mask, col, title, lab) in enumerate([
            (hist, kv.LIFECYCLE["survive"], "nested retirement",
             f"$k_h{{=}}{int(hist.sum())}$, a prefix of the committed order"),
            (indep, "#7B4EA8", "independent re-election",
             f"{int(indep.sum())} tokens, re-scored from scratch"),
            (indep & ~hist, "#B3261E", "would be added",
             f"{int((indep & ~hist).sum())} tokens absent from the prefix"),
            (hist & ~indep, "#0072B2", "would be dropped",
             f"{int((hist & ~indep).sum())} tokens the prefix keeps")]):
        ax = fig.add_subplot(gs[0, c])
        kv.draw_keep(ax, img, mask, tint=col, tint_alpha=0.30,
                     ghost=kv.GHOST_SOFT, outline_lw=0.6)
        ax.set_title(title, fontsize=7.8, pad=2.5)
        ax.set_xlabel(lab, fontsize=6.4, labelpad=1.6)
    fig.text(0.014, 1 - 0.08 / tot, "Nesting is a real constraint, not a free "
             "relabelling", fontsize=8.6, fontweight="bold", va="top")
    fig.text(0.014, 1 - 0.28 / tot,
             f"An unconstrained re-election moves "
             f"{100.0 * stats['mean_independent_divergence']:.1f}\\% of the "
             f"retirement set on average over {stats['n_records']} steps. "
             f"Honouring the prefix is what keeps the history cache "
             f"write-once, so those moves are exactly the compute the method "
             f"declines to spend.",
             fontsize=7.0, va="top", color="#333333")
    fig.text(0.014, foot_h / tot, f"OmniGUI episode {ep} step {s + 1} "
             f"({led[ep]['app']}), $r_c{{=}}50\\%$ entry and "
             f"$r_h{{=}}10\\%$ retirement. Panels three and four are the "
             f"symmetric difference of the first two.",
             fontsize=6.8, style="italic", va="top", color="#555555")
    save(fig, os.path.join(FIGDIR, "figG5_nested_vs_independent.pdf"))


# --------------------------------------------------------------------------
# G6: detector elements against the token grid
# --------------------------------------------------------------------------
def g6(by_ep, led, stats):
    apply(scale=1.42)
    ep = os.environ.get("W59_G6_EP", "T0110")
    s = int(os.environ.get("W59_G6_STEP", "0"))
    rec = by_ep[ep][s]
    img = omni_img(ep, s)
    w, h = img.size
    mass = json.load(open(os.path.join(VIZ, f"mass_{ep}_{s:02d}.json"),
                          encoding="utf-8"))
    gh, gw = rec["grid_h"], rec["grid_w"]
    P = np.asarray(mass["P"], float).reshape(gh, gw)
    union = rec.get("audit", {}).get("miu_token_union") or []
    umask = kv.idx_grid(union, gh, gw)

    cw = (6.9 - 0.30) / 4.0
    fig = plt.figure(figsize=(6.9, cw * (h / w) + 1.45))
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.018, right=0.992,
                          top=0.885, bottom=0.135)

    ax = fig.add_subplot(gs[0, 0])
    kv.show_image(ax, img)
    ax.set_title("input frame", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"token grid {gh}$\\times${gw}", fontsize=6.4, labelpad=1.6)

    ax = fig.add_subplot(gs[0, 1])
    kv.show_image(ax, img)
    kv.element_boxes(ax, mass["boxes"], w, h, ec="#00E5FF", lw=0.8, alpha=0.95)
    ax.set_title("detected elements", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"{len(mass['boxes'])} regions", fontsize=6.4, labelpad=1.6)

    ax = fig.add_subplot(gs[0, 2])
    kv.draw_keep(ax, img, umask, tint=C["best"], tint_alpha=0.26,
                 ghost=kv.GHOST_SOFT, outline_lw=0.5)
    ax.set_title("tokens the elements cover", fontsize=7.8, pad=2.5)
    ax.set_xlabel(f"{int(umask.sum())} of {gh * gw} tokens "
                  f"({100.0 * umask.sum() / max(1, umask.size):.0f}\\%)",
                  fontsize=6.4, labelpad=1.6)

    ax = fig.add_subplot(gs[0, 3])
    kv.score_heat(ax, img, P, vmax=max(P.max(), 1e-6))
    kv.element_boxes(ax, mass["boxes"], w, h, ec="#FFFFFF", lw=0.4, alpha=0.55)
    ax.set_title("LIP mass field", fontsize=7.8, pad=2.5)
    ax.set_xlabel("max-pooled information density", fontsize=6.4, labelpad=1.6)

    fig.text(0.018, 0.975, "From detected elements to a token-level prior",
             fontsize=8.6, fontweight="bold", va="top")
    fig.text(0.018, 0.945, "Element boxes are mapped onto the patch grid and "
             "max-pooled into a density field, so the prior is defined on the "
             "same units the selector spends its budget on.",
             fontsize=7.0, va="top", color="#333333")
    fig.text(0.018, 0.085,
             f"Detector coverage averages "
             f"{100.0 * stats['mean_miu_coverage_ratio']:.1f}\\% of the grid "
             f"over {stats['n_records']} steps, which is why the prior "
             f"constrains the budget without dictating it. Element boxes are "
             f"an offline CPU replay of the same detector policy used online.",
             fontsize=6.8, style="italic", va="top", color="#555555")
    save(fig, os.path.join(FIGDIR, "figG6_detector_to_prior.pdf"))


# --------------------------------------------------------------------------
# G7: selector comparison at an identical budget
# --------------------------------------------------------------------------
G7_ARMS = [("ours", "TRACE"), ("cdpruner", "CDPruner"),
           ("divprune", "DivPrune"), ("nomass", "w/o prior"),
           ("nocrs", "w/o repair"), ("uniform", "Uniform")]
G7_IDX = [int(x) for x in os.environ.get("W59_G7", "62,35").split(",")]


def g7(dumps, rows):
    apply(scale=1.42)
    arms = [(a, l) for a, l in G7_ARMS if a in dumps]
    idxs = [i for i in G7_IDX]
    n = len(arms) + 1
    ar = float(np.median([ss_img(dumps["ours"][i]).size[1] /
                          ss_img(dumps["ours"][i]).size[0] for i in idxs]))
    cw = (6.9 - 0.30) / n
    fig = plt.figure(figsize=(6.9, len(idxs) * (cw * ar + 0.30) + 0.90))
    gs = fig.add_gridspec(len(idxs), n, wspace=0.045, hspace=0.34,
                          left=0.055, right=0.992, top=0.945, bottom=0.055)
    for r, i in enumerate(idxs):
        meta = rows[i]
        img = ss_img(dumps["ours"][i])
        w, h = img.size
        axes_r = []
        ax = fig.add_subplot(gs[r, 0])
        axes_r.append(ax)
        kv.show_image(ax, img)
        kv.target_box(ax, meta["bbox_xywh"], w, h, kept=True, lw=1.5,
                      cross_on_miss=False)
        if r == 0:
            ax.set_title("input", fontsize=7.6, pad=2.5)
        ax.set_xlabel(shorten(meta.get("instruction", ""), 30), fontsize=6.0,
                      style="italic", labelpad=1.5)
        for c, (arm, lab) in enumerate(arms, start=1):
            rec = dumps[arm][i]
            ax = fig.add_subplot(gs[r, c])
            axes_r.append(ax)
            mask = kv.keep_grid(rec)
            ok = covered(rec, meta["bbox_xywh"], img)
            kv.draw_keep(ax, img, mask, outline_lw=0.55)
            kv.target_box(ax, meta["bbox_xywh"], w, h, kept=bool(ok), lw=1.5)
            if r == 0:
                ax.set_title(lab, fontsize=7.6, pad=2.5,
                             fontweight=("bold" if arm == "ours" else "normal"))
            ax.set_xlabel("target kept" if ok else "target lost",
                          fontsize=6.2, labelpad=1.5,
                          color=("#1B6E54" if ok else "#B3261E"))
    fig.text(0.008, 0.988, "Every selector at the same budget $r{=}10\\%$ on "
             "the same frames", fontsize=8.0, fontweight="bold", va="top")
    fig.text(0.5, 0.008, "The differences are structural rather than "
             "quantitative: identical token counts, different regions of the "
             "screen retained.",
             fontsize=7.0, ha="center", va="bottom", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG7_selector_comparison.pdf"))


# --------------------------------------------------------------------------
# G8: failure cases
# --------------------------------------------------------------------------
def g8(dumps, rows):
    apply(scale=1.42)
    # frames where TRACE itself loses the target, ordered by how small the
    # target is: the honest failure mode is thin widgets under a tight budget.
    bad = []
    for i, meta in rows.items():
        if meta.get("ours") != "0" or not meta.get("bbox_xywh"):
            continue
        b = meta["bbox_xywh"]
        bad.append((float(b[2]) * float(b[3]), i))
    bad.sort()
    picks = [i for _, i in bad[:4]]
    if not picks:
        print("g8: no failure frames found, skipped")
        return
    ar = float(np.median([ss_img(dumps["ours"][i]).size[1] /
                          ss_img(dumps["ours"][i]).size[0] for i in picks]))
    cw = (6.9 - 0.20) / len(picks)
    fig = plt.figure(figsize=(6.9, 2 * cw * ar + 1.20))
    gs = fig.add_gridspec(2, len(picks), wspace=0.05, hspace=0.28,
                          left=0.050, right=0.992, top=0.930, bottom=0.065)
    for c, i in enumerate(picks):
        meta = rows[i]
        rec = dumps["ours"][i]
        img = ss_img(rec)
        w, h = img.size
        mask = kv.keep_grid(rec)
        b = meta["bbox_xywh"]

        ax = fig.add_subplot(gs[0, c])
        kv.show_image(ax, img)
        kv.target_box(ax, b, w, h, kept=True, lw=1.5, cross_on_miss=False)
        ax.set_xlabel(shorten(meta.get("instruction", ""), 32), fontsize=6.0,
                      style="italic", labelpad=1.5)
        if c == 0:
            ax.set_title("input", fontsize=7.6, pad=2.5, loc="left")

        ax = fig.add_subplot(gs[1, c])
        kv.draw_keep(ax, img, mask, outline_lw=0.6)
        kv.target_box(ax, b, w, h, kept=False, lw=1.6)
        frac = float(b[2]) * float(b[3]) / (w * h)
        ax.set_xlabel(f"target is {100.0 * frac:.2f}\\% of the screen\n"
                      f"{meta.get('dtype', '')}", fontsize=6.2, labelpad=1.5,
                      color="#B3261E")
        if c == 0:
            ax.set_title("TRACE keep at $r{=}10\\%$", fontsize=7.6, pad=2.5,
                         loc="left")
    fig.text(0.008, 0.988, "Where selection still fails", fontsize=8.0,
             fontweight="bold", va="top")
    fig.text(0.5, 0.008, "The residual failures are thin or low-contrast "
             "widgets whose entire extent falls inside a single patch, so no "
             "budget allocation can separate them from their background.",
             fontsize=7.0, ha="center", va="bottom", color="#333333")
    save(fig, os.path.join(FIGDIR, "figG8_failures.pdf"))


def main():
    kv.enable_cjk()
    which = set(sys.argv[1:]) or None
    dumps, rows = load_ss()
    by_ep, led, stats = load_omni()
    jobs = [("g1", lambda: g1(dumps, rows)),
            ("g2", g2),
            ("g3", lambda: g3(by_ep, led)),
            ("g4", lambda: g4(by_ep, led)),
            ("g5", lambda: g5(by_ep, led, stats)),
            ("g6", lambda: g6(by_ep, led, stats)),
            ("g7", lambda: g7(dumps, rows)),
            ("g8", lambda: g8(dumps, rows))]
    for name, fn in jobs:
        if which and name not in which:
            continue
        fn()


if __name__ == "__main__":
    main()
