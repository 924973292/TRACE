#!/usr/bin/env python3
"""GeoPACT full-pipeline case study: capture + conference-style horizontal figure.

End-to-end panels per example (one row each):
  (1) OmniParser + OCR detections
  (2) Module-A InfoDensity + OAS importance heatmap on all MIUs
  (3) E-MMR hard-crop selection vs ground-truth click region
  (4) BICA spatial binding text injected into the prompt
  (5) Model answer vs ground truth (match badge)

Usage (remote GPU node, OminiEval env):
  source OminiDetect/omini_eval_omniparser.env
  python scripts/geopact_case_study_figure.py capture --doc-indices 120,480 --out figures/geopact_case_study/cases.json
  python scripts/geopact_case_study_figure.py render --cases figures/geopact_case_study/cases.json --out figures/geopact_case_study/geopact_case_study.pdf

  # or auto-pick click samples with OCR:
  python scripts/geopact_case_study_figure.py all --n-cases 2 --out-dir figures/geopact_case_study
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_FULL = "oas_emmr_coord_center_lambda_050"
PRUNE_RATIO = 0.10


def _repo() -> Path:
    return REPO_ROOT


def _load_docs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list in {path}")
    return data


def _serialize_miu(m: dict) -> dict:
    box = m.get("box")
    center = m.get("center")
    out: dict[str, Any] = {
        "miu_id": m.get("miu_id", ""),
        "type": m.get("type", m.get("role", "")),
        "source": m.get("source", ""),
        "text": str(m.get("text") or "")[:120],
        "box": list(box) if box else None,
        "center": list(center) if center else None,
        "det_score": float(m.get("det_score", 0) or 0),
        "oas_imp": float(m.get("oas_imp", 0) or 0),
        "oas_exec": int(m.get("oas_exec", 0) or 0),
        "oas_ref": int(m.get("oas_ref", 0) or 0),
        "oas_state": int(m.get("oas_state", 0) or 0),
        "oas_cross": int(m.get("oas_cross", 0) or 0),
        "info_density": float(m.get("info_density", 0) or 0),
        "info_density_proxy": float(m.get("info_density_proxy", 0) or 0),
        "score": float(m.get("score", 0) or 0),
    }
    return out


def pick_doc_indices(
    docs: list[dict],
    media_dir: Path,
    *,
    n: int,
    scan_limit: int = 800,
    seed_indices: list[int] | None = None,
) -> list[int]:
    if seed_indices:
        return [i for i in seed_indices if 0 <= i < len(docs)]

    picked: list[int] = []
    for i, doc in enumerate(docs[:scan_limit]):
        if doc.get("result_action_type") != 0:
            continue
        touch = doc.get("result_touch_xy") or ""
        if not touch or touch == '""':
            continue
        img_rel = doc.get("image_path") or ""
        if not img_rel:
            continue
        if not (media_dir / img_rel).is_file():
            continue
        instr = (doc.get("instruction_en") or doc.get("instruction") or "").strip()
        if len(instr) < 20:
            continue
        picked.append(i)
        if len(picked) >= n * 12:
            break

    if len(picked) < n:
        return picked[:n]

    # Prefer spread-out indices (different episodes), stay near dataset head for fast capture.
    seen_eps: set[str] = set()
    final: list[int] = []
    for i in picked:
        if i > 24:
            continue
        ep = docs[i].get("episode_id", str(i))
        if ep in seen_eps and len(final) < n - 1:
            continue
        seen_eps.add(ep)
        final.append(i)
        if len(final) >= n:
            break
    if len(final) < n:
        for i in picked:
            if i not in final:
                final.append(i)
            if len(final) >= n:
                break
    return final[:n] if final else picked[:n]


def capture_cases(
    *,
    repo: Path,
    doc_indices: list[int],
    policy: str,
    ratio: float,
    model_path: str,
    attn_impl: str,
    out_json: Path,
    python_bin: str | None = None,
) -> None:
    """Capture via ``lmms_eval`` (stable on cluster); dumps per-doc JSON under *out_json* parent."""
    import subprocess

    dump_dir = out_json.parent / "dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for old in dump_dir.glob("case_doc*.json"):
        old.unlink()

    py = python_bin or os.environ.get("PYTHON_BIN", sys.executable)
    detect_env = repo / "OminiDetect" / "omini_eval_omniparser.env"
    qwen_pkgs = Path(os.environ.get("QWEN_EXTRA_PKGS", ""))

    # lmms_eval only supports contiguous limit from doc 0; map requested indices to limit N.
    max_idx = max(doc_indices)
    limit = max_idx + 1
    if limit > 12:
        print(
            f"WARN: capture limit={limit} is slow; prefer --doc-indices 0,2 (limit<=5).",
            file=sys.stderr,
        )

    model_args = (
        f"pretrained={model_path},attn_implementation={attn_impl},torch_dtype=bfloat16,"
        f"device_map=auto,max_num_frames=4,skip_speaker_loading=True,miu_detector=omniparser,"
        f"visual_prune_ratio={ratio},visual_prune_scope=image,visual_prune_policy={policy}"
    )
    log_root = repo / "logs" / "geopact_case_study_capture"
    cmd = [
        py,
        "-m",
        "lmms_eval",
        "--model",
        "qwen2_5_omni_chat",
        "--model_args",
        model_args,
        "--tasks",
        "task_v2_qwen25_omni_7b",
        "--batch_size",
        "1",
        "--limit",
        str(limit),
        "--log_samples",
        "--output_path",
        str(log_root),
    ]
    env = os.environ.copy()
    env["GEOPACT_CASE_STUDY_DUMP"] = str(dump_dir)
    env["MIU_PRELOAD_EASYOCR"] = env.get("MIU_PRELOAD_EASYOCR", "1")
    env["MIU_OCR_SUBPROC_FALLBACK"] = env.get("MIU_OCR_SUBPROC_FALLBACK", "1")
    env["PYTHONUNBUFFERED"] = "1"
    if qwen_pkgs.is_dir():
        env["PYTHONPATH"] = str(qwen_pkgs) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print(f"Running lmms_eval limit={limit} dump={dump_dir} ...", flush=True)
    proc = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"lmms_eval failed with exit {proc.returncode}")

    media_dir = repo / "data" / "benchmark"
    cases: list[dict] = []
    for doc_idx in doc_indices:
        dump_file = dump_dir / f"case_doc{doc_idx:05d}.json"
        if not dump_file.is_file():
            print(f"WARN: missing dump {dump_file}", file=sys.stderr)
            continue
        case = json.loads(dump_file.read_text(encoding="utf-8"))
        img_rel = case.get("image_path") or ""
        case["image_path"] = str((media_dir / img_rel).relative_to(repo)) if img_rel else ""
        cases.append(case)
        ev = case.get("eval") or {}
        print(
            f"  doc {doc_idx}: selected={len(case.get('selected_mius') or [])} "
            f"exact={ev.get('exact_match')} ocr={(case.get('detect_meta') or {}).get('miu_ocr_count', 0)}",
            flush=True,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json} ({len(cases)} cases)", flush=True)


def _draw_boxes(ax, image, mius: list[dict], *, color_fn, lw: float = 1.5, alpha: float = 0.95):
    import matplotlib.patches as patches

    ax.imshow(image)
    ax.set_axis_off()
    w, h = image.size
    for m in mius:
        box = m.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1 * w, y1 * h),
            (x2 - x1) * w,
            (y2 - y1) * h,
            linewidth=lw,
            edgecolor=color_fn(m),
            facecolor=(*color_fn(m)[:3], 0.12) if len(color_fn(m)) == 4 else (0, 0, 0, 0),
            alpha=alpha,
        )
        ax.add_patch(rect)


def render_figure(cases_path: Path, out_path: Path, *, dpi: int = 200) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import patheffects as pe
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    n_rows = len(cases)
    if n_rows == 0:
        raise ValueError("no cases in JSON")

    # Wide landscape: 5 pipeline stages per row
    fig = plt.figure(figsize=(22, 4.8 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, 5, figure=fig, width_ratios=[1.15, 1.15, 1.15, 1.0, 1.05], wspace=0.08, hspace=0.22)

    col_titles = [
        "(a) OmniParser + OCR",
        "(b) InfoDensity · OAS",
        "(c) E-MMR selection",
        "(d) BICA binding text",
        "(e) Model vs GT",
    ]
    cmap = plt.cm.plasma

    for row, case in enumerate(cases):
        repo = _repo()
        img_path = repo / case["image_path"] if not Path(case["image_path"]).is_absolute() else Path(case["image_path"])
        image = Image.open(img_path).convert("RGB")
        filtered = case.get("filtered_mius") or []
        selected_ids = {m["miu_id"] for m in case.get("selected_mius") or []}
        gt = case.get("gt_coord") or {}

        dens_vals = [float(m.get("info_density") or m.get("info_density_proxy") or 0) for m in filtered]
        oas_vals = [float(m.get("oas_imp", 0)) for m in filtered]
        d_hi = max(dens_vals) if dens_vals else 1.0
        o_hi = max(oas_vals) if oas_vals else 1.0

        def _combo_color(m: dict) -> tuple:
            d = float(m.get("info_density") or m.get("info_density_proxy") or 0) / max(d_hi, 1e-6)
            o = float(m.get("oas_imp", 0)) / max(o_hi, 1e-6)
            t = 0.55 * d + 0.45 * o
            rgba = cmap(min(1.0, max(0.0, t)))
            return (rgba[0], rgba[1], rgba[2], 0.95)

        # (a) all detections — gray + OCR cyan labels
        ax0 = fig.add_subplot(gs[row, 0])
        _draw_boxes(
            ax0,
            image,
            filtered,
            color_fn=lambda m: (0.45, 0.48, 0.52, 0.9) if m.get("source") != "ocr" else (0.1, 0.65, 0.85, 0.95),
            lw=1.2,
        )
        for m in filtered[:48]:
            box = m.get("box")
            if not box:
                continue
            w, h = image.size
            tag = m.get("miu_id", "")[:4]
            txt = (m.get("text") or "")[:10]
            label = f"{tag}" + (f' "{txt}"' if txt else "")
            ax0.text(
                box[0] * w + 2,
                max(10, box[1] * h - 2),
                label,
                fontsize=6,
                color="white",
                va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )
        ax0.set_title(col_titles[0] if row == 0 else "", loc="left", pad=6)
        if row == 0:
            ax0.text(0.02, 0.98, f"n={len(filtered)} MIUs", transform=ax0.transAxes, va="top", fontsize=8, color="#333")

        # (b) density / OAS colormap
        ax1 = fig.add_subplot(gs[row, 1])
        _draw_boxes(ax1, image, filtered, color_fn=_combo_color, lw=2.0)
        ax1.set_title(col_titles[1] if row == 0 else "", loc="left", pad=6)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        if row == n_rows - 1:
            cbar = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.02)
            cbar.set_label("combined score", fontsize=8)
        # top-3 legend
        top3 = sorted(filtered, key=lambda m: -float(m.get("oas_imp", 0)))[:3]
        leg = ", ".join(
            f"{m.get('miu_id','?')} ρ={float(m.get('info_density') or m.get('info_density_proxy') or 0):.2f} "
            f"OAS={float(m.get('oas_imp',0)):.2f}"
            for m in top3
        )
        ax1.text(0.02, 0.02, leg, transform=ax1.transAxes, fontsize=6.5, color="#222", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#ccc"))

        # (c) selected + GT
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(image)
        ax2.set_axis_off()
        w, h = image.size
        import matplotlib.patches as patches

        for m in filtered:
            box = m.get("box")
            if not box:
                continue
            sel = m.get("miu_id") in selected_ids
            if not sel:
                continue
            x1, y1, x2, y2 = box
            ax2.add_patch(
                patches.Rectangle(
                    (x1 * w, y1 * h),
                    (x2 - x1) * w,
                    (y2 - y1) * h,
                    linewidth=3.0,
                    edgecolor="#1B9E77",
                    facecolor=(0.11, 0.62, 0.47, 0.18),
                )
            )
            ax2.text(
                x1 * w + 3,
                y1 * h + 12,
                f"{m.get('miu_id','')} · {float(m.get('oas_imp',0)):.2f}",
                fontsize=7,
                color="#0B5D45",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

        if gt.get("type") == "rect" and gt.get("rect"):
            r = gt["rect"]
            # GT stored as pixel coords in dataset
            x1, y1 = r[0][0] / w, r[0][1] / h
            x2, y2 = r[1][0] / w, r[1][1] / h
            ax2.add_patch(
                patches.Rectangle(
                    (x1 * w, y1 * h),
                    (x2 - x1) * w,
                    (y2 - y1) * h,
                    linewidth=2.5,
                    linestyle="--",
                    edgecolor="#D95F02",
                    facecolor=(0.85, 0.37, 0.01, 0.12),
                )
            )
            ax2.text(x1 * w, y2 * h + 6, "GT click region", color="#D95F02", fontsize=7, fontweight="bold")
        elif gt.get("center"):
            cx, cy = gt["center"]
            ax2.scatter([cx], [cy], s=120, c="#D95F02", marker="x", linewidths=2.5, zorder=5)
        ax2.set_title(col_titles[2] if row == 0 else "", loc="left", pad=6)
        ax2.text(
            0.02,
            0.98,
            f"selected k={len(case.get('selected_mius') or [])} · r={case.get('prune_ratio', 0.1):.0%}",
            transform=ax2.transAxes,
            va="top",
            fontsize=8,
            color="#1B9E77",
            fontweight="bold",
        )

        # (d) binding text
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.set_axis_off()
        coord = (case.get("coord_text") or "").strip()
        if len(coord) > 1200:
            coord = coord[:1200] + "\n…"
        coord_display = coord if coord else "(no coord injection — policy off or empty selection)"
        ax3.text(
            0.0,
            1.0,
            "Injected spatial binding (BICA + center coords):\n\n" + coord_display,
            transform=ax3.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            family="monospace",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F9FC", edgecolor="#C5D0DE"),
        )
        ax3.set_title(col_titles[3] if row == 0 else "", loc="left", pad=6)

        # (e) model answer + match
        ax4 = fig.add_subplot(gs[row, 4])
        ax4.set_axis_off()
        ev = case.get("eval") or {}
        ok = bool(ev.get("exact_match"))
        badge_color = "#1B7F3A" if ok else "#B22222"
        badge = "✓ MATCH" if ok else "✗ MISMATCH"
        ans = (case.get("model_answer") or "").strip()
        if len(ans) > 900:
            ans = ans[:900] + "\n…"
        dist = ev.get("coordinate_distance")
        dist_s = f"{dist:.1f}px" if isinstance(dist, (int, float)) else "—"
        summary = (
            f"{badge}\n"
            f"type_match={ev.get('type_match')} · coord_dist={dist_s}\n\n"
            f"Instruction:\n{(case.get('instruction') or '')[:220]}…\n\n"
            f"Model output:\n{ans}"
        )
        ax4.text(
            0.0,
            1.0,
            summary,
            transform=ax4.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color="#111",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=badge_color, linewidth=2.0),
        )
        ax4.set_title(col_titles[4] if row == 0 else "", loc="left", pad=6)
        ax4.text(
            0.72,
            0.92,
            badge,
            transform=ax4.transAxes,
            fontsize=11,
            fontweight="bold",
            color=badge_color,
            ha="center",
        )

        # Row label
        fig.text(
            0.01,
            1.0 - (row + 0.5) / n_rows,
            f"Ex.{row + 1}\n{case.get('episode_id','')}",
            fontsize=9,
            fontweight="bold",
            va="center",
            ha="left",
            color="#444",
        )

    fig.suptitle(
        "GeoPACT full method — per-step case study (OmniParser → OAS/InfoDensity → E-MMR → BICA → Qwen2.5-Omni)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path} and {png_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="GeoPACT case-study capture + figure")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="Run full GeoPACT on selected docs (GPU)")
    p_cap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p_cap.add_argument("--doc-indices", type=str, default="", help="comma-separated doc indices")
    p_cap.add_argument("--n-cases", type=int, default=2)
    p_cap.add_argument("--policy", default=POLICY_FULL)
    p_cap.add_argument("--ratio", type=float, default=PRUNE_RATIO)
    p_cap.add_argument("--model-path", default="models/Qwen2.5-Omni-7B")
    p_cap.add_argument("--attn-impl", default="flash_attention_2")
    p_cap.add_argument("--out", type=Path, required=True)

    p_render = sub.add_parser("render", help="Build figure from cases JSON")
    p_render.add_argument("--cases", type=Path, required=True)
    p_render.add_argument("--out", type=Path, required=True)
    p_render.add_argument("--dpi", type=int, default=200)

    p_all = sub.add_parser("all", help="capture + render")
    p_all.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p_all.add_argument("--doc-indices", type=str, default="")
    p_all.add_argument("--n-cases", type=int, default=2)
    p_all.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures" / "geopact_case_study")
    p_all.add_argument("--policy", default=POLICY_FULL)
    p_all.add_argument("--model-path", default="models/Qwen2.5-Omni-7B")
    p_all.add_argument("--attn-impl", default="flash_attention_2")

    args = parser.parse_args()
    repo = Path(getattr(args, "repo_root", REPO_ROOT)).resolve()

    if args.cmd == "render":
        render_figure(args.cases.resolve(), args.out.resolve(), dpi=args.dpi)
        return 0

    if args.cmd in {"capture", "all"}:
        docs = _load_docs(repo / "data" / "merged_sorted.json")
        media_dir = repo / "data" / "benchmark"
        if args.doc_indices.strip():
            indices = [int(x.strip()) for x in args.doc_indices.split(",") if x.strip()]
        else:
            indices = pick_doc_indices(docs, media_dir, n=args.n_cases)
        if not indices:
            print("ERROR: no valid doc indices", file=sys.stderr)
            return 1
        out_json = (
            args.out.resolve()
            if args.cmd == "capture"
            else (args.out_dir / "cases.json").resolve()
        )
        capture_cases(
            repo=repo,
            doc_indices=indices,
            policy=args.policy,
            ratio=PRUNE_RATIO if args.cmd == "capture" else getattr(args, "ratio", PRUNE_RATIO),
            model_path=args.model_path,
            attn_impl=args.attn_impl,
            out_json=out_json,
        )
        if args.cmd == "all":
            render_figure(
                out_json,
                (args.out_dir / "geopact_case_study.pdf").resolve(),
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
