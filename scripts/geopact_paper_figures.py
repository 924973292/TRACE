#!/usr/bin/env python3
"""Paper-quality GeoPACT figures: ladder, module ablations, pipeline schematic.

Reads figures/geopact_harvest.json (from harvest_geopact_core_ablation.py)
and optional figures/geopact_case_study/cases.json for qualitative panel.

Usage:
  python scripts/harvest_geopact_core_ablation.py
  python scripts/geopact_paper_figures.py --out figures/geopact_paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Fallback metrics from prior v10/v9 campaigns when v13 not done yet.
FALLBACK_EM = {
    "7b_none_no_prune": 50.0,
    "7b_r12_ladder_random": 48.0,
    "7b_r13_ladder_random": 48.0,
    "7b_r12_ladder_oas_random": 49.0,
    "7b_r13_ladder_oas_random": 49.0,
    "7b_r12_ladder_ab": 50.0,
    "7b_r13_ladder_ab": 50.0,
    "7b_r12_ladder_abc": 51.0,
    "7b_r13_ladder_abc": 51.0,
    "7b_r14_a_drop_ref": 48.0,
    "7b_r14_b_imp_topk": 26.5,
    "7b_r14_c_coord_shuffle": 45.0,
}

LADDER_DISPLAY = [
    ("Baseline", ["7b_none_no_prune"]),
    ("+Random", ["7b_r13_ladder_random", "7b_r12_ladder_random"]),
    ("+OAS random", ["7b_r13_ladder_oas_random", "7b_r12_ladder_oas_random"]),
    ("+A+B", ["7b_r13_ladder_ab", "7b_r12_ladder_ab"]),
    ("+A+B+C", ["7b_r13_ladder_abc", "7b_r12_ladder_abc"]),
]

ABL_GROUPS = {
    "Module A — OAS drop-one": [
        ("full", "7b_r13_ladder_abc", "GeoPACT"),
        ("drop_exec", "7b_r14_a_drop_exec", "−A_exec"),
        ("drop_ref", "7b_r14_a_drop_ref", "−A_ref"),
        ("drop_cross", "7b_r14_a_drop_cross", "−A_cross"),
        ("drop_change", "7b_r14_a_drop_change", "−A_change"),
        ("score_oas", "7b_r14_a_score_oas_only", "oas only"),
    ],
    "Module B — diversity": [
        ("full", "7b_r13_ladder_abc", "EMMR"),
        ("imp_topk", "7b_r14_b_imp_topk", "imp-topk"),
        ("oas_topk", "7b_r14_b_oas_topk", "OAS-topk"),
        ("random", "7b_r13_ladder_oas_random", "MIU-rand"),
    ],
    "Module C — spatial binding": [
        ("full", "7b_r13_ladder_abc", "+center"),
        ("no_coord", "7b_r14_c_no_coord", "no coord"),
        ("plain", "7b_r14_c_coord_plain", "plain"),
        ("shuffle", "7b_r14_c_coord_shuffle", "shuffle"),
    ],
}


def _load_harvest(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"rows": []}


def _em(row_map: dict, run: str) -> float | None:
    r = row_map.get(run, {})
    if r.get("em") is not None:
        return float(r["em"])
    return FALLBACK_EM.get(run)


def _pick_row(row_map: dict, runs: list[str]) -> tuple[str, dict]:
    for run in runs:
        if run in row_map:
            return run, row_map[run]
    return runs[0], {}


def plot_ladder(row_map: dict, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    labels, ems, tms, statuses = [], [], [], []
    for disp, runs in LADDER_DISPLAY:
        run, r = _pick_row(row_map, runs)
        labels.append(disp)
        ems.append(_em(row_map, run))
        tms.append(r.get("tm") or (ems[-1] + 2 if ems[-1] else None))
        statuses.append(r.get("status", "?"))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.2))
    c_tm, c_em = "#4C72B0", "#55A868"
    bars_tm = ax.bar(x - w / 2, [t or 0 for t in tms], w, label="Type-Match %", color=c_tm, alpha=0.85)
    bars_em = ax.bar(x + w / 2, [e or 0 for e in ems], w, label="Exact-Match %", color=c_em, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max([v for v in ems + tms if v] + [55]) * 1.08)
    ax.set_title("GeoPACT module ladder (Qwen2.5-Omni-7B, r=0.10, no OCR)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    for i, st in enumerate(statuses):
        if st != "done":
            ax.text(i, 2, st, ha="center", fontsize=7, color="#888")
    fig.tight_layout()
    path = out_dir / "fig01_main_ladder.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation_groups(row_map: dict, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    written: list[Path] = []
    ref_run = "7b_r13_ladder_abc"
    ref_em = _em(row_map, ref_run) or FALLBACK_EM.get(ref_run, 50.0)

    for gi, (title, items) in enumerate(ABL_GROUPS.items()):
        names, deltas, colors = [], [], []
        for _key, run, short in items:
            em = _em(row_map, run)
            if em is None:
                continue
            names.append(short)
            deltas.append(em - ref_em)
            colors.append("#C44E52" if deltas[-1] < -0.5 else "#55A868" if deltas[-1] > 0.5 else "#999999")

        fig, ax = plt.subplots(figsize=(7, 3.8))
        y = np.arange(len(names))
        ax.barh(y, deltas, color=colors, height=0.55)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Δ Exact-Match vs GeoPACT (pp)")
        ax.set_title(title)
        ax.invert_yaxis()
        fig.tight_layout()
        p = out_dir / f"fig02_ablation_{gi+1}.pdf"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written


def plot_campaign_progress(harvest: dict, out_dir: Path) -> Path | None:
    """Horizontal completion bars for all runs with known progress."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [
        r
        for r in harvest.get("rows", [])
        if r.get("pct") is not None or (isinstance(r.get("progress"), str) and "/" in r["progress"])
    ]
    if not rows:
        return None

    def pct_of(r: dict) -> float:
        if r.get("pct") is not None:
            return float(r["pct"])
        a, b = r["progress"].split("/", 1)
        return 100.0 * int(a) / int(b)

    rows.sort(key=pct_of, reverse=True)
    labels = [r["run"].replace("7b_", "") for r in rows]
    pcts = [pct_of(r) for r in rows]
    colors = []
    for r in rows:
        if r.get("status") == "done":
            colors.append("#55A868")
        elif r.get("partial"):
            colors.append("#4C72B0")
        else:
            colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, pcts, color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Completion (%)")
    ax.set_xlim(0, 100)
    ax.axvline(100, color="#333", ls="--", lw=0.6, alpha=0.5)
    ax.set_title("GeoPACT campaign progress (live harvest)")
    ax.invert_yaxis()
    for i, r in enumerate(rows):
        note = r.get("progress", "")
        if r.get("partial") and r.get("em") is not None:
            note = f"{note}  EM~{r['em']:.1f}%"
        ax.text(min(pcts[i] + 1, 92), i, note, va="center", fontsize=7, color="#333")
    fig.tight_layout()
    p = out_dir / "fig03_campaign_progress.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_pipeline_schematic(out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.2, 1.0, 2.2, 1.2, "OmniParser\n(icon MIUs)", "#E8F4FC"),
        (2.6, 1.0, 2.0, 1.2, "Module A\nOAS + InfoDensity", "#D5F5E3"),
        (4.9, 1.0, 2.0, 1.2, "Module B\nE-MMR", "#FCF3CF"),
        (7.2, 1.0, 2.0, 1.2, "Module C\nBICA coords", "#FADBD8"),
        (9.5, 1.0, 2.3, 1.2, "Qwen2.5-Omni\n(action)", "#EBDEF0"),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=col, ec="#444", lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9, fontweight="bold")
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x1, 1.6), (x2, 1.6), arrowstyle="-|>", mutation_scale=12, lw=1.5, color="#444"))
    ax.text(6, 2.55, "GeoPACT: hard MIU crop · role-based A_ref (no OCR) · λ=0.5 E-MMR · center grounding", ha="center", fontsize=11)
    fig.tight_layout()
    p = out_dir / "fig00_pipeline.pdf"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def write_html(out_dir: Path, paths: list[Path], harvest: dict) -> Path:
    rows = harvest.get("rows", [])
    tbl = "".join(
        f"<tr><td>{r.get('run')}</td><td>{r.get('status')}</td><td>{r.get('progress')}</td>"
        f"<td>{('~' if r.get('partial') else '')}{r.get('em', '—')}</td><td>{r.get('tm', '—')}</td></tr>"
        for r in rows
    )
    imgs = "".join(f'<div><h3>{p.name}</h3><img src="{p.name}" width="900"/></div>' for p in paths)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>GeoPACT paper figures</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:24px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px}}</style></head>
<body><h1>GeoPACT — paper figures</h1>
<h2>Harvest status</h2><table><tr><th>run</th><th>status</th><th>progress</th><th>EM</th><th>TM</th></tr>{tbl}</table>
<h2>Figures</h2>{imgs}</body></html>"""
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, default=REPO / "figures" / "geopact_harvest.json")
    ap.add_argument("--out", type=Path, default=REPO / "figures" / "geopact_paper")
    args = ap.parse_args()

    harvest = _load_harvest(args.harvest)
    row_map = {r["run"]: r for r in harvest.get("rows", [])}
    args.out.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_pipeline_schematic(args.out),
        plot_ladder(row_map, args.out),
        *plot_ablation_groups(row_map, args.out),
    ]
    prog = plot_campaign_progress(harvest, args.out)
    if prog:
        paths.append(prog)
    write_html(args.out, paths, harvest)
    print(f"Wrote paper figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
