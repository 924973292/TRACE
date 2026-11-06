#!/usr/bin/env python3
"""Verify local OmniGUI download completeness vs HuggingFace + merged_sorted paths."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def _exists(root: Path, rel: str) -> bool:
    if not rel:
        return False
    return (root / rel.replace("/", os.sep)).is_file()


def _iter_path_values(value: str | list | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def collect_merged_paths(docs: list[dict]) -> dict[str, set[str]]:
    fields = {
        "image_path": set(),
        "video_path": set(),
        "audio_path": set(),
        "history_screenshot_path": set(),
        "action_history_image_path": set(),
    }
    for d in docs:
        for key in ("image_path", "video_path", "audio_path"):
            p = d.get(key) or ""
            if p:
                fields[key].add(p)
        for p in _iter_path_values(d.get("history_screenshot_path")):
            fields["history_screenshot_path"].add(p)
        for ah in d.get("action_history") or []:
            p = ah.get("image_path") or ""
            if p:
                fields["action_history_image_path"].add(p)
    return fields


def count_local_media(benchmark_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {"png": 0, "mp4": 0, "wav": 0, "json": 0, "other": 0}
    for p in benchmark_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in counts:
            counts[ext] += 1
        else:
            counts["other"] += 1
    return counts


def hf_media_counts() -> dict[str, int]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {}
    api = HfApi()
    info = api.repo_info("XIAOCHENLIN00zz/OmniGUI", repo_type="dataset")
    counts: Counter[str] = Counter()
    for s in info.siblings:
        name = s.rfilename
        if not name.startswith("benchmark/"):
            continue
        ext = Path(name).suffix.lower().lstrip(".")
        counts[ext] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="data/merged_sorted.json")
    parser.add_argument("--benchmark-root", default="data/benchmark")
    parser.add_argument("--out", default="data/download_integrity_report.json")
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    json_path = repo / args.json
    benchmark_root = repo / args.benchmark_root

    docs = json.loads(json_path.read_text(encoding="utf-8"))
    path_sets = collect_merged_paths(docs)

    missing: dict[str, list[str]] = {}
    for field, paths in path_sets.items():
        bad = sorted(p for p in paths if not _exists(benchmark_root, p))
        if bad:
            missing[field] = bad

    local_counts = count_local_media(benchmark_root)
    hf_counts = {} if args.skip_hf else hf_media_counts()

    apps = sorted({p.name for p in benchmark_root.iterdir() if p.is_dir()})
    report = {
        "merged_json": str(json_path),
        "benchmark_root": str(benchmark_root),
        "total_docs": len(docs),
        "unique_episodes": len({d.get("episode_id") for d in docs}),
        "apps_present": apps,
        "app_count": len(apps),
        "path_counts_in_json": {k: len(v) for k, v in path_sets.items()},
        "missing_paths": missing,
        "missing_total": sum(len(v) for v in missing.values()),
        "local_media_file_counts": local_counts,
        "hf_benchmark_file_counts": hf_counts,
        "local_vs_hf_delta": {
            ext: local_counts.get(ext, 0) - hf_counts.get(ext, 0)
            for ext in sorted(set(local_counts) | set(hf_counts))
        },
    }

    out_path = repo / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Apps: {len(apps)} ({', '.join(apps[:5])}...)")
    print(f"merged_sorted docs: {report['total_docs']}")
    print("Path refs in JSON:", report["path_counts_in_json"])
    print(f"Missing on disk: {report['missing_total']}")
    for field, paths in missing.items():
        print(f"  {field}: {len(paths)} missing")
        for p in paths[:5]:
            print(f"    - {p}")
        if len(paths) > 5:
            print(f"    ... +{len(paths)-5} more")
    print("Local media counts:", local_counts)
    if hf_counts:
        print("HF benchmark counts:", hf_counts)
        print("Delta (local - HF):", report["local_vs_hf_delta"])
    print(f"Report: {out_path}")

    if report["missing_total"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
