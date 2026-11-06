#!/usr/bin/env python3
"""Normalize likaixin/ScreenSpot-Pro HF tree into data/screenspot_pro/{images,annotations,manifest.json}."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

REPO = Path(os.environ.get("OMINI_REPO", Path(__file__).resolve().parents[1]))
HF_ROOT = REPO / "data" / "screenspot_pro_hf"
OUT_ROOT = REPO / "data" / "screenspot_pro"


def _find_dir(name: str) -> Path | None:
    if not HF_ROOT.is_dir():
        return None
    for p in HF_ROOT.rglob(name):
        if p.is_dir():
            return p
    return None


def _resolve_image(images_dst: Path, img_ref: str) -> Path | None:
    if not img_ref:
        return None
    ref = Path(str(img_ref))
    candidates = [
        images_dst / ref,
        images_dst / ref.name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Nested layout: images/blender_windows/screenshot_....png
    for hit in images_dst.rglob(ref.name):
        if hit.is_file():
            return hit
    if ref.parts:
        sub = images_dst / Path(*ref.parts)
        if sub.is_file():
            return sub
    return None


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    images_src = _find_dir("images") or HF_ROOT
    ann_src = _find_dir("annotations") or HF_ROOT

    images_dst = OUT_ROOT / "images"
    ann_dst = OUT_ROOT / "annotations"
    if images_dst.exists():
        shutil.rmtree(images_dst)
    if ann_dst.exists():
        shutil.rmtree(ann_dst)

    if (images_src / "images").is_dir():
        shutil.copytree(images_src / "images", images_dst)
    elif images_src.is_dir():
        shutil.copytree(
            images_src,
            images_dst,
            ignore=shutil.ignore_patterns("annotations", "*.json", "*.md", "*.parquet", "*.html"),
        )

    if (ann_src / "annotations").is_dir():
        shutil.copytree(ann_src / "annotations", ann_dst)
    else:
        ann_dst.mkdir(parents=True, exist_ok=True)
        for jf in HF_ROOT.rglob("*.json"):
            if "annotation" in jf.name.lower() or jf.parent.name == "annotations":
                shutil.copy2(jf, ann_dst / jf.name)

    samples = []
    for jf in sorted(ann_dst.glob("*.json")):
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("_ann_file", jf.name)
                    samples.append(row)
        elif isinstance(payload, dict):
            for key in ("data", "samples", "annotations"):
                if isinstance(payload.get(key), list):
                    for row in payload[key]:
                        if isinstance(row, dict):
                            r = dict(row)
                            r.setdefault("_ann_file", jf.name)
                            samples.append(r)
                    break

    if not samples:
        print(
            "ERROR: 0 ScreenSpot-Pro samples. Download likaixin/ScreenSpot-Pro (images+annotations) first.",
            flush=True,
        )
        return 1

    manifest = {
        "source": "likaixin/ScreenSpot-Pro",
        "num_samples": len(samples),
        "samples": samples,
        "images_dir": "data/screenspot_pro/images",
        "annotations_dir": "data/screenspot_pro/annotations",
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_rows = []
    n_with_img = 0
    for row in samples:
        out = dict(row)
        img = out.get("image_path") or out.get("img_filename") or out.get("file_name") or ""
        resolved = _resolve_image(images_dst, str(img))
        if resolved and resolved.is_file():
            rel = resolved.relative_to(REPO).as_posix()
            out["image_path"] = rel
            n_with_img += 1
        else:
            out["image_path"] = f"data/screenspot_pro/images/{Path(str(img)).name}"
        eval_rows.append(out)

    eval_path = OUT_ROOT / "eval_index.json"
    eval_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"screenspot_pro manifest: {len(samples)} samples -> {OUT_ROOT / 'manifest.json'}")
    print(f"screenspot_pro eval_index: {len(eval_rows)} rows, {n_with_img} with images -> {eval_path}")
    if n_with_img < len(eval_rows) * 0.5:
        print("WARN: fewer than 50% samples have local images; re-run after Tier1 download completes", flush=True)
    return 0 if n_with_img > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
