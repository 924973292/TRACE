#!/usr/bin/env python3

"""Download + extract AndroidControl screenshots (GUI-Actor-Data, hf-mirror).



Source: cckevinn/GUI-Actor-Data / AndroidControl_images.zip (~49GB)

Output: {repo}/data/android_control/images/*.png

Patches eval_steps.json with image_path for lmms_eval.



Resumable: partial data lives under

  {repo}/data/android_control_hf/.cache/huggingface/

Re-run the same command (or ``scripts/resume_android_control_download.ps1``)

after a disconnect; download continues from the last byte.

"""



from __future__ import annotations



import argparse

import json

import os

import shutil

import subprocess

import sys

import zipfile

from pathlib import Path



HF_MIRROR = "https://hf-mirror.com"

REPO_ID = "cckevinn/GUI-Actor-Data"

ZIP_NAME = "AndroidControl_images.zip"



SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:

    sys.path.insert(0, str(SCRIPT_DIR))



from hf_resumable_download import (  # noqa: E402

    DownloadState,

    configure_hf_download_env,

    download_file_resumable,

)





def _state_path(repo: Path) -> Path:

    return repo / "logs" / "downloads" / "android_control_images.state.json"





def _extract_zip(zip_path: Path, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    seven_zip = shutil.which("7z") or shutil.which("7za")

    if seven_zip:

        print(f"[EXTRACT] 7z -> {out_dir}", flush=True)

        subprocess.run([seven_zip, "x", str(zip_path), f"-o{out_dir}", "-aoa", "-y"], check=True)

        return

    print(f"[EXTRACT] zipfile (slow) -> {out_dir}", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:

        zf.extractall(out_dir)





def _flatten_images(root: Path, images_dir: Path) -> None:

    """Move *.png to images_dir root (match img_url filenames)."""

    images_dir.mkdir(parents=True, exist_ok=True)

    for png in root.rglob("*.png"):

        if png.parent.resolve() == images_dir.resolve():

            continue

        dest = images_dir / png.name

        if dest.is_file():

            continue

        shutil.move(str(png), str(dest))





def _patch_eval_steps(repo: Path, images_dir: Path) -> int:

    steps_path = repo / "data" / "android_control" / "eval_steps.json"

    if not steps_path.is_file():

        build = repo / "scripts" / "build_android_control_manifest.py"

        script_repo = Path(__file__).resolve().parents[1]

        if not build.is_file():

            build = script_repo / "scripts" / "build_android_control_manifest.py"

        if build.is_file():

            subprocess.run([sys.executable, str(build)], cwd=str(repo), check=False)



    if not steps_path.is_file():

        print("[WARN] eval_steps.json missing; run build_android_control_manifest.py first", flush=True)

        return 0



    steps = json.loads(steps_path.read_text(encoding="utf-8"))

    n_ok = 0

    for row in steps:

        name = row.get("img_url") or row.get("image_path") or ""

        if not name:

            continue

        p = images_dir / Path(name).name

        if p.is_file():

            row["image_path"] = f"data/android_control/images/{p.name}".replace("\\", "/")

            n_ok += 1

    steps_path.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[PATCH] eval_steps image_path set for {n_ok}/{len(steps)} steps", flush=True)

    return n_ok





def _download_zip(

    repo: Path,

    *,

    force: bool,

    max_retries: int,

    retry_sleep: float,

) -> Path:

    cache_dir = repo / "data" / "android_control_hf"

    cache_dir.mkdir(parents=True, exist_ok=True)

    state = _state_path(repo)

    configure_hf_download_env(cache_root=cache_dir, download_timeout_s=600, etag_timeout_s=60)



    print(f"[START] {REPO_ID}/{ZIP_NAME} via {os.environ.get('HF_ENDPOINT', HF_MIRROR)}", flush=True)

    print(f"[CACHE] {cache_dir}", flush=True)

    print(f"[STATE] {state}", flush=True)

    print("[TIP] Safe to Ctrl+C; re-run to resume.", flush=True)



    return download_file_resumable(

        repo_id=REPO_ID,

        filename=ZIP_NAME,

        local_dir=cache_dir,

        repo_type="dataset",

        force=force,

        max_retries=max_retries,

        retry_sleep_s=retry_sleep,

        state_path=state,

    )





def main() -> int:

    parser = argparse.ArgumentParser(

        description=__doc__,

        formatter_class=argparse.RawDescriptionHelpFormatter,

    )

    parser.add_argument("--repo-root", default=os.environ.get("OMINI_REPO", r"$REPO"))

    parser.add_argument("--force", action="store_true", help="re-download zip even if present")

    parser.add_argument(

        "--skip-download",

        action="store_true",

        help="Only patch eval_steps paths when images exist; never download zip",

    )

    parser.add_argument(

        "--download-only",

        action="store_true",

        help="Download zip only; skip extract + patch (resume-friendly)",

    )

    parser.add_argument(

        "--extract-only",

        action="store_true",

        help="Extract existing zip only (skip download)",

    )

    parser.add_argument(

        "--max-retries",

        type=int,

        default=0,

        help="Download retries (0=unlimited until success)",

    )

    parser.add_argument(

        "--retry-sleep",

        type=float,

        default=30.0,

        help="Seconds between retries after network error",

    )

    parser.add_argument("--status", action="store_true", help="Print resume state and exit")

    args = parser.parse_args()



    repo = Path(args.repo_root).resolve()

    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

    os.environ["OMINI_REPO"] = str(repo)



    images_dir = repo / "data" / "android_control" / "images"

    cache_dir = repo / "data" / "android_control_hf"

    zip_path = cache_dir / ZIP_NAME

    state_path = _state_path(repo)



    if args.status:

        st = DownloadState.load(state_path)

        if st:

            print(json.dumps(st.__dict__, indent=2))

        elif zip_path.is_file():

            print(f"zip complete: {zip_path} ({zip_path.stat().st_size} bytes)")

        else:

            print("no download state yet")

        return 0



    if not args.force and not args.extract_only and images_dir.is_dir():

        pngs = list(images_dir.glob("*.png"))

        if len(pngs) > 1000:

            print(f"[SKIP] android_control images: {len(pngs)} pngs in {images_dir}", flush=True)

            _patch_eval_steps(repo, images_dir)

            return 0



    if args.skip_download:

        if images_dir.is_dir() and any(images_dir.glob("*.png")):

            n = _patch_eval_steps(repo, images_dir)

            print(f"[PATCH] android_control image_path ({n} steps)", flush=True)

            return 0

        print("[SKIP] android_control images not present (--skip-download)", flush=True)

        return 0



    if not args.extract_only:

        zip_path = _download_zip(

            repo,

            force=args.force,

            max_retries=args.max_retries,

            retry_sleep=args.retry_sleep,

        )

        if args.download_only:

            print(f"[DOWNLOAD-ONLY] zip ready at {zip_path}", flush=True)

            return 0



    if not zip_path.is_file():

        print(f"[ERROR] zip missing: {zip_path}", flush=True)

        return 1



    staging = repo / "data" / "android_control" / "_zip_extract"

    if staging.exists():

        shutil.rmtree(staging)

    _extract_zip(zip_path, staging)

    _flatten_images(staging, images_dir)

    shutil.rmtree(staging, ignore_errors=True)



    n = _patch_eval_steps(repo, images_dir)

    if n <= 0:

        print("[WARN] no images matched eval_steps", flush=True)

        return 1

    print(f"[DONE] android_control images -> {images_dir}", flush=True)

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

