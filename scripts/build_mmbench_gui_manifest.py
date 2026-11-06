#!/usr/bin/env python3
"""Build MMBench-GUI offline eval indices (L1 QA + L2 grounding).

Source: OpenGVLab/MMBench-GUI
Output:
  data/mmbench_gui/offline_images/
  data/mmbench_gui/l1_eval_index.json
  data/mmbench_gui/l2_eval_index.json
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
HF_REPO = "OpenGVLab/MMBench-GUI"
ZIP_NAME = "MMBench-GUI-OfflineImages.zip"


def _repo_root() -> Path:
    return Path(os.environ.get("OMINI_REPO", Path(__file__).resolve().parents[1]))


def _index_images(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            out[p.name] = p
    return out


def _resolve_image(name: str, index: dict[str, Path], platform: str, images_root: Path) -> str:
    if not name:
        return ""
    if name in index:
        return str(index[name]).replace("\\", "/")
    plat_dir = images_root / platform
    candidate = plat_dir / name
    if candidate.is_file():
        return str(candidate).replace("\\", "/")
    for p in images_root.rglob(name):
        if p.is_file():
            return str(p).replace("\\", "/")
    return ""


def main() -> int:
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    repo = _repo_root()
    hf_cache = repo / "data" / "mmbench_gui_hf"
    out_root = repo / "data" / "mmbench_gui"
    images_root = out_root / "offline_images"
    out_root.mkdir(parents=True, exist_ok=True)

    l1_src = hf_cache / "L1_annotations.json"
    l2_src = hf_cache / "L2_annotations.json"
    zip_path = hf_cache / ZIP_NAME
    if not l1_src.is_file() or not l2_src.is_file() or not zip_path.is_file():
        print("WARN: run cn_mirror_download --only mmbench_gui first", flush=True)
        return 1

    if not images_root.is_dir() or not any(images_root.rglob("*.png")):
        print(f"[EXTRACT] {zip_path} -> {images_root}", flush=True)
        if images_root.exists():
            shutil.rmtree(images_root)
        images_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_root)

    img_index = _index_images(images_root)

    l1_raw = json.loads(l1_src.read_text(encoding="utf-8"))
    l1_rows: list[dict] = []
    for parent in l1_raw:
        platform = str(parent.get("platform", ""))
        app_name = parent.get("app_name", "")
        img_name = str(parent.get("image_path", ""))
        image_size = parent.get("image_size", [0, 0])
        resolved = _resolve_image(img_name, img_index, platform, images_root)
        for gi, group in enumerate(parent.get("groups") or []):
            opts = group.get("options") or {}
            l1_rows.append(
                {
                    "l1_index": parent.get("index"),
                    "group_index": gi,
                    "platform": platform,
                    "app_name": app_name,
                    "image_path": resolved,
                    "image_size": image_size,
                    "question": group.get("question", ""),
                    "options": opts,
                    "answer": str(group.get("answer", "")).strip().upper(),
                    "difficulty": group.get("difficulty", ""),
                }
            )

    l2_raw = json.loads(l2_src.read_text(encoding="utf-8"))
    l2_rows: list[dict] = []
    for row in l2_raw:
        platform = str(row.get("platform", ""))
        img_name = str(row.get("image_path", ""))
        entry = dict(row)
        entry["image_path"] = _resolve_image(img_name, img_index, platform, images_root)
        entry["instruction"] = row.get("instruction", "")
        entry["bbox"] = row.get("bbox", [])
        entry["grounding_type"] = row.get("grounding_type", "")
        l2_rows.append(entry)

    (out_root / "l1_eval_index.json").write_text(
        json.dumps(l1_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "l2_eval_index.json").write_text(
        json.dumps(l2_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "source": HF_REPO,
        "l1_questions": len(l1_rows),
        "l2_samples": len(l2_rows),
        "l1_path": "data/mmbench_gui/l1_eval_index.json",
        "l2_path": "data/mmbench_gui/l2_eval_index.json",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_l1_img = sum(1 for r in l1_rows if r.get("image_path"))
    n_l2_img = sum(1 for r in l2_rows if r.get("image_path"))
    print(f"mmbench_gui: L1={len(l1_rows)} ({n_l1_img} with images) L2={len(l2_rows)} ({n_l2_img}) -> {out_root}")
    return 0 if len(l1_rows) > 1000 and len(l2_rows) > 3000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
