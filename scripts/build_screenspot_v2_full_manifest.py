#!/usr/bin/env python3
"""Build ScreenSpot-V2 full eval index (Mobile + Web + Desktop, 1272 samples).

Source: OS-Copilot/ScreenSpot-v2
Output: {repo}/data/screenspot_v2/{eval_index.json,images/,manifest.json}
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
HF_REPO = "OS-Copilot/ScreenSpot-v2"

DOMAIN_FILES = (
    ("mobile", "screenspot_mobile_v2.json"),
    ("web", "screenspot_web_v2.json"),
    ("desktop", "screenspot_desktop_v2.json"),
)


def _repo_root() -> Path:
    return Path(os.environ.get("OMINI_REPO", Path(__file__).resolve().parents[1]))


def _ensure_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    return hf_hub_download


def main() -> int:
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    repo = _repo_root()
    out = repo / "data" / "screenspot_v2"
    hf_cache = repo / "data" / "screenspot_v2_hf"
    images = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    hf_hub_download = _ensure_hf()
    all_rows: list[dict] = []
    all_names: set[str] = set()

    for domain, json_name in DOMAIN_FILES:
        json_path = hf_cache / json_name
        if not json_path.is_file():
            p = hf_hub_download(
                HF_REPO,
                json_name,
                repo_type="dataset",
                local_dir=str(hf_cache),
                endpoint=HF_MIRROR,
            )
            json_path = Path(p) if not Path(p).is_dir() else hf_cache / json_name
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        for row in rows:
            fname = str(row.get("img_filename", ""))
            if fname:
                all_names.add(fname)
            entry = dict(row)
            entry["domain"] = domain
            entry["platform_domain"] = domain
            all_rows.append(entry)

    zip_path = hf_cache / "screenspotv2_image.zip"
    if not zip_path.is_file():
        p = hf_hub_download(
            HF_REPO,
            "screenspotv2_image.zip",
            repo_type="dataset",
            local_dir=str(hf_cache),
            endpoint=HF_MIRROR,
        )
        zip_path = Path(p) if not Path(p).is_dir() else hf_cache / "screenspotv2_image.zip"

    print(f"[EXTRACT] {zip_path} -> {images} ({len(all_names)} images)", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base not in all_names:
                continue
            if not base.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            dest = images / base
            if dest.is_file():
                continue
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    eval_rows = []
    for row in all_rows:
        fname = str(row.get("img_filename", ""))
        img_path = images / fname
        entry = dict(row)
        entry["image_path"] = str(img_path).replace("\\", "/") if img_path.is_file() else ""
        entry["file_name"] = fname
        entry["instruction"] = row.get("instruction", "")
        entry["bbox"] = row.get("bbox", [])
        eval_rows.append(entry)

    manifest = {
        "source": HF_REPO,
        "num_samples": len(eval_rows),
        "domains": {d: sum(1 for r in eval_rows if r.get("domain") == d) for d, _ in DOMAIN_FILES},
        "samples": eval_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eval_index.json").write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    n_img = sum(1 for r in eval_rows if r.get("image_path"))
    print(f"screenspot_v2: {len(eval_rows)} samples, {n_img} with images -> {out}")
    return 0 if n_img >= 1200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
