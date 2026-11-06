#!/usr/bin/env python3
"""Download TRACE eval model weights into a target models/ directory.

  export HF_ENDPOINT=https://huggingface.co   # or a mirror
  python scripts/download_omini_models.py --models-dir models trace

Examples:

  python scripts/download_omini_models.py --models-dir models GUI-Owl-1.5-8B-Instruct
  python scripts/download_omini_models.py --models-dir models trace
  python scripts/download_omini_models.py --models-dir models --list

Groups:
  trace     GUI-Owl-1.5-8B + 2B + UI-TARS-1.5-7B (paper carriers)
  owl       GUI-Owl 2B/4B/8B/Think
  gui       owl + UI-TARS + MAI-UI
  core      GUI-Owl-1.5-4B (legacy)
  all       every entry in the manifest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# Same mirror as all existing Omini remote download scripts (not huggingface.co).
HF_MIRROR = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install in your eval venv:\n"
        "  pip install -U huggingface_hub hf_transfer"
    ) from exc


@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    hf_repo: str
    groups: frozenset[str]
    size_gb: float
    note: str = ""


# Manifest of public Hugging Face checkpoints used by TRACE eval.
MODEL_MANIFEST: tuple[ModelSpec, ...] = (
    ModelSpec(
        "GUI-Owl-1.5-2B-Instruct",
        "mPLUG/GUI-Owl-1.5-2B-Instruct",
        frozenset({"owl", "gui", "all", "trace"}),
        4.0,
        "paper 2B copy-without-retune carrier",
    ),
    ModelSpec(
        "GUI-Owl-1.5-4B-Instruct",
        "mPLUG/GUI-Owl-1.5-4B-Instruct",
        frozenset({"core", "owl", "gui", "all"}),
        8.3,
        "★ current mainline (dialogue KV + uniform prune)",
    ),
    ModelSpec(
        "GUI-Owl-1.5-8B-Instruct",
        "mPLUG/GUI-Owl-1.5-8B-Instruct",
        frozenset({"owl", "gui", "all", "trace"}),
        17.0,
        "paper 8B carrier",
    ),
    ModelSpec(
        "GUI-Owl-1.5-8B-Think",
        "mPLUG/GUI-Owl-1.5-8B-Think",
        frozenset({"owl", "gui", "all"}),
        17.0,
    ),
    ModelSpec(
        "UI-TARS-1.5-7B",
        "ByteDance-Seed/UI-TARS-1.5-7B",
        frozenset({"gui", "all", "trace"}),
        16.0,
        "paper UI-TARS carrier",
    ),
    ModelSpec(
        "UI-TARS-2B-SFT",
        "ByteDance-Seed/UI-TARS-2B-SFT",
        frozenset({"gui", "all"}),
        9.2,
        "lmms_eval ui_tars",
    ),
    ModelSpec(
        "MAI-UI-2B",
        "Tongyi-MAI/MAI-UI-2B",
        frozenset({"gui", "all"}),
        4.0,
        "lmms_eval mai_ui",
    ),
    ModelSpec(
        "Qwen2.5-Omni-3B",
        "Qwen/Qwen2.5-Omni-3B",
        frozenset({"omni", "all"}),
        12.0,
        "official OmniGUI upstream / GeoPACT rescue 3B",
    ),
    ModelSpec(
        "Qwen2.5-Omni-7B",
        "Qwen/Qwen2.5-Omni-7B",
        frozenset({"omni", "all"}),
        21.0,
        "official OmniGUI upstream / GeoPACT 7B matrix",
    ),
    ModelSpec(
        "Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        frozenset({"omni", "all"}),
        66.0,
    ),
    ModelSpec(
        "MiniCPM-o-4_5",
        "openbmb/MiniCPM-o-4_5",
        frozenset({"multimodal", "all"}),
        19.0,
        "API server eval (minicpm_o); needs OminiMiniCPM env separately",
    ),
    ModelSpec(
        "VITA-1.5",
        "VITA-MLLM/VITA-1.5",
        frozenset({"multimodal", "all"}),
        19.0,
        "lmms_eval vita",
    ),
    ModelSpec(
        "Baichuan-Omni-1d5",
        "baichuan-inc/Baichuan-Omni-1d5",
        frozenset({"multimodal", "all"}),
        21.0,
        "API eval (baichuan_omni); needs separate API server",
    ),
    ModelSpec(
        "InternViT-300M-448px",
        "OpenGVLab/InternViT-300M-448px",
        frozenset({"aux", "all"}),
        0.6,
        "vision backbone asset; not referenced by current eval scripts",
    ),
)

GROUPS = ("trace", "core", "owl", "gui", "omni", "multimodal", "aux", "all")
BY_LOCAL = {m.local_name: m for m in MODEL_MANIFEST}

# Download queue: GUI-Owl first (4B mainline → 2B → 8B → Think), then everything else.
OWL_DOWNLOAD_ORDER: tuple[str, ...] = (
    "GUI-Owl-1.5-4B-Instruct",
    "GUI-Owl-1.5-2B-Instruct",
    "GUI-Owl-1.5-8B-Instruct",
    "GUI-Owl-1.5-8B-Think",
)
_OWL_RANK = {name: rank for rank, name in enumerate(OWL_DOWNLOAD_ORDER)}
_MANIFEST_RANK = {spec.local_name: rank for rank, spec in enumerate(MODEL_MANIFEST)}


def _sort_download_priority(specs: list[ModelSpec]) -> list[ModelSpec]:
    """GUI-Owl series first; within Owl follow OWL_DOWNLOAD_ORDER; else manifest order."""

    def sort_key(spec: ModelSpec) -> tuple[int, int]:
        if spec.local_name in _OWL_RANK:
            return (0, _OWL_RANK[spec.local_name])
        return (1, _MANIFEST_RANK.get(spec.local_name, 999))

    return sorted(specs, key=sort_key)


def _parse_selection(raw: Sequence[str]) -> list[ModelSpec]:
    if not raw or raw == ("all",):
        return list(MODEL_MANIFEST)
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if part in GROUPS:
                for spec in MODEL_MANIFEST:
                    if part == "all" or part in spec.groups:
                        if spec.local_name not in seen:
                            seen.add(spec.local_name)
                            out.append(spec)
                continue
            spec = BY_LOCAL.get(part)
            if spec is None:
                known = ", ".join(sorted(BY_LOCAL))
                raise SystemExit(f"Unknown model/group {part!r}. Known: {known}; groups: {', '.join(GROUPS)}")
            if spec.local_name not in seen:
                seen.add(spec.local_name)
                out.append(spec)
    return out


def _is_complete(local_dir: str) -> bool:
    """True when config.json exists and all weight shards are present at top level."""
    cfg = os.path.join(local_dir, "config.json")
    if not os.path.isfile(cfg):
        return False

    index = os.path.join(local_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                data = json.load(f)
            shards = sorted(set(data.get("weight_map", {}).values()))
            if not shards:
                return False
            for shard in shards:
                path = os.path.join(local_dir, shard)
                if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                    return False
            return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return False

    for name in os.listdir(local_dir):
        if name.endswith((".safetensors", ".bin")):
            path = os.path.join(local_dir, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return True
    return False


def _enable_hf_transfer(env: dict) -> None:
    endpoint = (env.get("HF_ENDPOINT") or HF_MIRROR).rstrip("/")
    if "hf-mirror.com" in endpoint:
        env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        return
    try:
        import importlib.util

        if importlib.util.find_spec("hf_transfer") is not None:
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        else:
            env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    except Exception:
        env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


def _download_via_mirror(spec: ModelSpec, local_dir: str) -> None:
    """Download via HF_ENDPOINT. Mirror: sequential files; direct HF: parallel snapshot (faster)."""
    env = os.environ.copy()
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT") or HF_MIRROR
    _enable_hf_transfer(env)
    os.environ.update({k: env[k] for k in ("HF_ENDPOINT", "HF_HUB_ENABLE_HF_TRANSFER") if k in env})
    endpoint = env["HF_ENDPOINT"]

    if "hf-mirror.com" not in endpoint.rstrip("/"):
        snapshot_download(
            spec.hf_repo,
            local_dir=local_dir,
            endpoint=endpoint,
            ignore_patterns=["original/*"],
            max_workers=4,
        )
        return

    files = [
        f
        for f in sorted(list_repo_files(spec.hf_repo, repo_type="model"))
        if not f.startswith("original/")
    ]
    total = len(files)
    for i, fname in enumerate(files, 1):
        print(f"  [{i}/{total}] {fname}", flush=True)
        hf_hub_download(
            spec.hf_repo,
            fname,
            local_dir=local_dir,
            endpoint=endpoint,
        )


def download_one(spec: ModelSpec, models_dir: str, *, force: bool) -> None:
    local_dir = os.path.join(models_dir, spec.local_name)
    os.makedirs(local_dir, exist_ok=True)
    if not force and _is_complete(local_dir):
        print(f"[SKIP] {spec.local_name} already present ({local_dir})", flush=True)
        return
    print(
        f"[START] {spec.hf_repo} -> {local_dir}  (~{spec.size_gb} GB)  {spec.note}",
        flush=True,
    )
    print(f"        via HF_ENDPOINT={os.environ.get('HF_ENDPOINT', HF_MIRROR)}", flush=True)
    _download_via_mirror(spec, local_dir)
    print(f"[DONE]  {spec.local_name}", flush=True)


def print_inventory(models_dir: str) -> None:
    print("TRACE models/ inventory:\n")
    print(f"{'Local dir':<32} {'HF repo':<42} {'~GB':>5}  Groups / note")
    print("-" * 110)
    total = 0.0
    for spec in MODEL_MANIFEST:
        groups = ",".join(sorted(spec.groups - {"all"}))
        print(f"{spec.local_name:<32} {spec.hf_repo:<42} {spec.size_gb:>5.1f}  [{groups}] {spec.note}")
        total += spec.size_gb
    print("-" * 110)
    print(f"{'TOTAL (all, no dedup)':<32} {'':<42} {total:>5.1f}")
    print(f"\nTarget models dir: {models_dir}")
    print(
        "\nNot in models/: OmniParser YOLO weights for Module A bilateral prune:\n"
        "  OmniParser-weights/v2.0/icon_detect/model.pt  (HF: microsoft/OmniParser or local mirror)"
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "selection",
        nargs="*",
        default=["trace"],
        help="group name(s), local dir name(s), or comma-separated mix (default: trace)",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("MODELS_DIR", "models"),
        help="destination directory (default: ./models or $MODELS_DIR)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if config+weights exist")
    parser.add_argument("--list", action="store_true", help="print manifest and exit")
    parser.add_argument(
        "--check-complete",
        metavar="MODEL_DIR",
        help="exit 0 if model weights are complete, else exit 1 (for watch/sync scripts)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_complete:
        return 0 if _is_complete(os.path.abspath(args.check_complete)) else 1

    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    if args.list:
        print_inventory(models_dir)
        return 0

    _enable_hf_transfer(os.environ)
    specs = _sort_download_priority(_parse_selection(args.selection))
    est = sum(s.size_gb for s in specs)
    print(f"Downloading {len(specs)} model(s) -> {models_dir}  (~{est:.1f} GB)", flush=True)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", flush=True)
    print("Order: " + " → ".join(s.local_name for s in specs), flush=True)

    errors: list[str] = []
    for spec in specs:
        try:
            download_one(spec, models_dir, force=args.force)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec.local_name}: {exc}")
            print(f"[ERROR] {spec.local_name}: {exc}", flush=True)

    if errors:
        print(f"\nFinished with {len(errors)} error(s).", flush=True)
        return 1
    print("\nAll requested downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
