#!/usr/bin/env python3
"""Box-level parity: serial detect (parallel off) vs optimized (parallel on).

Compares MIU count and per-box (source, box, det_score) sets on benchmark samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_DATA = REPO / "data" / "merged_sorted_media_valid.json"
DEFAULT_MEDIA = REPO / "data" / "benchmark"
BOX_TOL = 1e-4


@contextmanager
def _env_overlay(updates: dict[str, str | None]) -> Iterator[None]:
    saved: dict[str, Optional[str]] = {}
    try:
        for key, val in updates.items():
            saved[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[str(key)] = str(val)
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _resolve_image_path(rel: str, media_root: Path) -> Path | None:
    rel = str(rel or "").strip()
    if not rel:
        return None
    if rel.startswith("file://"):
        rel = rel[7:]
    for p in (Path(rel), media_root / rel, REPO / rel):
        if p.is_file():
            return p
    return None


def _load_paths(data_json: Path, media_root: Path, limit: int) -> list[Path]:
    with data_json.open(encoding="utf-8") as f:
        docs = json.load(f)
    paths: list[Path] = []
    for doc in docs:
        if len(paths) >= limit:
            break
        rel = doc.get("image_path") or doc.get("image") or doc.get("screenshot") or ""
        p = _resolve_image_path(rel, media_root)
        if p is not None:
            paths.append(p)
    if not paths:
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for p in sorted(media_root.rglob(pattern)):
                paths.append(p)
                if len(paths) >= limit:
                    return paths
    return paths


def _box_sig(m: dict) -> tuple:
    box = m.get("box") or (0, 0, 0, 0)
    return (
        str(m.get("source", "")),
        tuple(round(float(v), 4) for v in box[:4]),
        round(float(m.get("det_score", 0.0) or 0.0), 4),
    )


def _miu_set(mius: list) -> set[tuple]:
    return {_box_sig(m) for m in mius}


def _source_counts(mius: list) -> Counter:
    return Counter(str(m.get("source", "")) for m in mius)


def _run_detect(pil: Image.Image, *, parallel: bool) -> list:
    from lmms_eval.models.gui_miu_detectors import detect_omniparser_icon_mius_raw

    env = {
        "GEOPACT_YOLO_CIRCULAR_AUX": "1",
        "GEOPACT_YOLO_STRIPE_AUX": "1",
        "GEOPACT_YOLO_OCR_AUX": "0",
        "MIU_DISABLE_OCR": "1",
        "GEOPACT_MODULE_A_DETECT_PARALLEL": "1" if parallel else "0",
        "GEOPACT_MODULE_A_CV_PARALLEL": "1" if parallel else "0",
    }
    with _env_overlay(env):
        raw = detect_omniparser_icon_mius_raw(pil)
    if raw is None:
        return []
    mius, _np = raw
    return mius


def main() -> None:
    ap = argparse.ArgumentParser(description="Module A detect box parity (serial vs parallel)")
    ap.add_argument("--data-json", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = _load_paths(args.data_json, args.media_root, args.limit)
    if not paths:
        raise SystemExit(f"No images found (data={args.data_json}, media={args.media_root})")

    failures: list[dict] = []
    rows: list[dict] = []
    for path in paths:
        pil = Image.open(path).convert("RGB")
        serial = _run_detect(pil, parallel=False)
        optim = _run_detect(pil, parallel=True)
        s_set = _miu_set(serial)
        o_set = _miu_set(optim)
        ok = s_set == o_set
        row = {
            "image": path.name,
            "n_serial": len(serial),
            "n_optim": len(optim),
            "parity": ok,
            "only_serial": len(s_set - o_set),
            "only_optim": len(o_set - s_set),
            "sources_serial": dict(_source_counts(serial)),
            "sources_optim": dict(_source_counts(optim)),
        }
        rows.append(row)
        if not ok:
            failures.append(row)
            print(f"FAIL {path.name}: serial={len(serial)} optim={len(optim)} "
                  f"+serial={row['only_serial']} +optim={row['only_optim']}")
        else:
            print(f"OK   {path.name}: n={len(serial)}")

    n_ok = sum(1 for r in rows if r["parity"])
    report = {
        "n_images": len(rows),
        "n_pass": n_ok,
        "n_fail": len(failures),
        "all_pass": len(failures) == 0,
        "rows": rows,
        "failures": failures,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")

    if failures:
        raise SystemExit(f"Parity failed on {len(failures)}/{len(rows)} images")
    print(f"PASS: {n_ok}/{len(rows)} images box-identical (serial vs parallel)")


if __name__ == "__main__":
    main()
