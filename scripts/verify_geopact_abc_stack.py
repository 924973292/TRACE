#!/usr/bin/env python3
"""Verify GeoPACT Module A / B / C stack before launching ablation fleet.

Primary path: YOLO / OmniParser (GUI-Owl hist_k5 fleet). Canny is optional smoke only.

Checks:
  A — YOLO detection, pentagonal energy, InfoDensity (required)
  B — HES kernel-herding with merged spatial grid (no stride fallback)
  C — SBR coord text injection

Exit 0 only when the YOLO stack is healthy.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from PIL import Image


def _ok(name: str, detail: str = "") -> None:
    print(f"[OK] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""), file=sys.stderr)


def _warn(name: str, detail: str = "") -> None:
    print(f"[WARN] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "", *, required: bool = True) -> None:
        if ok:
            _ok(name, detail)
        elif required:
            _fail(name, detail)
            failures.append(name)
        else:
            _warn(name, detail)

    # ── cv2 / OpenCV (YOLO + pentagonal preprocessing) ─────────────────────
    try:
        import cv2  # noqa: F401

        check("opencv (cv2)", True, getattr(cv2, "__version__", "?"))
    except Exception as exc:
        check(
            "opencv (cv2)",
            False,
            f"{exc} — pip install opencv-python-headless==4.8.1.78 in OminiEval",
        )

    weights = Path(
        os.environ.get(
            "OMNIPARSER_ICON_WEIGHTS",
            str(REPO / "models/omniparser/icon_detect/model.pt"),
        )
    )
    check("YOLO weights (OMNIPARSER_ICON_WEIGHTS)", weights.is_file(), str(weights))

    # ── Real benchmark screenshot ──────────────────────────────────────────
    sample = REPO / "data" / "benchmark" / "douyin" / "media" / "T0144" / "2.png"
    if not sample.is_file():
        merged = json.loads((REPO / "data" / "merged_sorted.json").read_text(encoding="utf-8"))
        for doc in merged:
            rel = doc.get("image_path") or doc.get("image") or ""
            cand = REPO / "data" / "benchmark" / str(rel).replace("\\", "/")
            if cand.is_file():
                sample = cand
                break
    check("benchmark sample image", sample.is_file(), str(sample))
    if not sample.is_file():
        print("FATAL: no sample image", file=sys.stderr)
        return 1

    image = Image.open(sample).convert("RGB")

    # ── Module A: detectors + InfoDensity ───────────────────────────────────
    from lmms_eval.models.gui_miu_detectors import (
        detect_blob_mius,
        detect_canny_morph_mius,
        detect_fusion_fast_mius,
        detect_omniparser_icon_mius,
    )
    from lmms_eval.models.geopact_vl_hook import _run_detector, _sticky_rng

    policy_ab = "oas_ickh_no_coord_yolo"
    try:
        mius_a = _run_detector(image, "omniparser", policy_ab)
        check("Module A — _run_detector(YOLO)", len(mius_a) > 0, f"n={len(mius_a)}")
        if mius_a:
            pent = mius_a[0]
            check(
                "Module A — pentagonal attrs",
                all(k in pent for k in ("pentagonal_scale", "pentagonal_entropy", "pentagonal_contrast")),
                f"keys={sorted(pent.keys())[:8]}...",
            )
            rho = float(mius_a[0].get("info_density", mius_a[0].get("info_density_proxy", 0.0)) or 0.0)
            check("Module A — info_density attached", rho > 0.0, f"sample_rho={rho:.4f}")
    except RuntimeError as exc:
        check("Module A — _run_detector(YOLO)", False, str(exc))
        mius_a = []

    yolo_raw = detect_omniparser_icon_mius(image)
    check(
        "Module A — YOLO in-process",
        yolo_raw is not None and len(yolo_raw) > 0,
        f"n={len(yolo_raw or [])}",
    )

    # Canny / blob / fusion_fast (default fast stack)
    canny_raw: list = []
    try:
        canny_raw = detect_canny_morph_mius(image)
        check("Module A — Canny MIUs", len(canny_raw) > 0, f"n={len(canny_raw)}", required=False)
    except Exception as exc:
        check("Module A — Canny MIUs", False, str(exc), required=False)

    try:
        blob_raw = detect_blob_mius(image)
        check("Module A — blob MIUs", len(blob_raw) > 0, f"n={len(blob_raw)}", required=False)
    except Exception as exc:
        check("Module A — blob MIUs", False, str(exc), required=False)

    try:
        fusion_raw = detect_fusion_fast_mius(image)
        check("Module A — fusion_fast MIUs", len(fusion_raw) > 0, f"n={len(fusion_raw)}")
        if fusion_raw:
            check(
                "Module A — fusion_fast >= canny count",
                len(fusion_raw) >= len(canny_raw or []),
                f"fusion={len(fusion_raw)} canny={len(canny_raw or [])}",
                required=False,
            )
        mius_fusion = _run_detector(image, "fusion_fast", "oas_random_no_coord")
        check("Module A — _run_detector(fusion_fast)", len(mius_fusion) > 0, f"n={len(mius_fusion)}")
    except Exception as exc:
        check("Module A — fusion_fast MIUs", False, str(exc))
        mius_fusion = []

    # ── Module B: HES with MIU mapping (no stride fallback) ────────────────
    from lmms_eval.models.geopact_vl_hook import (
        _hes_select,
        _get_grid,
        _flat_image_token_ranges,
        _llm_tokens_per_image,
        _spatial_merge_size,
        _map_mius_to_tokens,
    )

    grid_h, grid_w = 14, 18
    n_img = grid_h * grid_w
    hidden = 64
    image_embeds = torch.randn(n_img, hidden)
    inputs_embeds = torch.randn(1, 32, hidden)
    image_mask = torch.zeros(32, dtype=torch.bool)
    image_mask[-8:] = True
    image_grid = torch.tensor([[1, grid_h, grid_w]])

    gh, gw, gm = _get_grid(image_grid, n_img)
    check("Module B — grid matches tokens (synthetic raw)", gm, f"grid=({gh},{gw}) n={n_img}")

    if mius_a:
        try:
            keep = _hes_select(
                mius=mius_a,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=image_grid,
                keep_k_tokens=max(1, n_img // 10),
                policy=policy_ab,
            )
            check("Module B — HES keep set", len(keep) > 0, f"kept={len(keep)}/{n_img}")
        except RuntimeError as exc:
            check("Module B — HES", False, str(exc))

    # MIU-level random (oas_random_*): kept tokens should lie in MIU token union
    miu_test = mius_fusion or mius_a
    if miu_test:
        try:
            keep_k = max(1, n_img // 5)
            rng = _sticky_rng("verify:episode", str(sample))
            hes_miu_rand = _hes_select(
                mius=miu_test,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=image_grid,
                keep_k_tokens=keep_k,
                policy="oas_random_no_coord",
                rng=rng,
            )
            mapped_token_idxs, _, _, _ = _map_mius_to_tokens(miu_test, image_embeds, grid_h, grid_w, n_img)
            union_tokens: set[int] = set()
            for group in mapped_token_idxs:
                union_tokens.update(group)
            keep_set = set(hes_miu_rand)
            cov = len(keep_set & union_tokens) / max(len(keep_set), 1)
            check(
                "Module B — MIU-random token coverage",
                cov >= 1.0 - 1e-9 or len(union_tokens) < keep_k,
                f"coverage={cov:.3f} union={len(union_tokens)} keep={len(keep_set)}",
            )
            hes_token_rand = _hes_select(
                mius=miu_test,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=image_grid,
                keep_k_tokens=keep_k,
                policy="random",
                rng=rng,
            )
            jacc = len(set(hes_miu_rand) & set(hes_token_rand)) / max(
                len(set(hes_miu_rand) | set(hes_token_rand)), 1
            )
            check(
                "Module B — MIU-random != token-random",
                jacc < 1.0 - 1e-9,
                f"jaccard={jacc:.3f}",
                required=False,
            )
        except RuntimeError as exc:
            check("Module B — MIU-random", False, str(exc))

    merge = _spatial_merge_size()
    raw_h, raw_w = 80, 36
    n_merged = _llm_tokens_per_image(1, raw_h, raw_w, merge)
    merged_grid = torch.tensor([[1, raw_h, raw_w]])
    gh2, gw2, gm2 = _get_grid(merged_grid, n_merged)
    check(
        "Module B — Qwen3-VL merged grid",
        gm2 and gh2 == raw_h // merge and gw2 == raw_w // merge,
        f"raw=({raw_h},{raw_w}) llm=({gh2},{gw2}) n={n_merged}",
    )
    if mius_a:
        merged_embeds = torch.randn(n_merged, hidden)
        try:
            keep2 = _hes_select(
                mius=mius_a,
                image_embeds=merged_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=merged_grid,
                keep_k_tokens=max(1, n_merged // 5),
                policy=policy_ab,
            )
            check(
                "Module B — HES on merged grid",
                len(keep2) > 0,
                f"kept={len(keep2)}/{n_merged}",
            )
        except RuntimeError as exc:
            check("Module B — HES on merged grid", False, str(exc))

    grid2 = torch.tensor([[1, raw_h, raw_w], [1, raw_h, raw_w]])
    n_two = n_merged * 2
    try:
        ranges2 = _flat_image_token_ranges(grid2, n_two)
        ok_ranges = (
            len(ranges2) == 2
            and ranges2[0] == (0, n_merged)
            and ranges2[1] == (n_merged, n_two)
        )
        check("Module B — multi-frame token ranges", ok_ranges, f"ranges={ranges2}")
    except RuntimeError as exc:
        check("Module B — multi-frame token ranges", False, str(exc))

    try:
        keep_empty = _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=image_grid,
            keep_k_tokens=max(1, n_img // 10),
            policy=policy_ab,
        )
        check(
            "Module B — 0 MIU passthrough (skip A/B/C)",
            len(keep_empty) == n_img,
            f"kept={len(keep_empty)}/{n_img} (full keep)",
        )
    except RuntimeError as exc:
        check("Module B — 0 MIU passthrough (skip A/B/C)", False, str(exc))

    from lmms_eval.models.geopact_vl_hook import (
        _build_coord_text,
        _coord_miu_indices_from_pick_order,
    )

    policy_abc = "oas_ickh_coord_center_yolo"
    if mius_a:
        mapped_token_idxs, _, _, mapped_miu_indices = _map_mius_to_tokens(mius_a, image_embeds, grid_h, grid_w, n_img)
        miu_token_idxs = [[] for _ in mius_a]
        for group, source_idx in zip(mapped_token_idxs, mapped_miu_indices):
            miu_token_idxs[source_idx] = group
        keep_set = set()
        for group in miu_token_idxs[: min(3, len(miu_token_idxs))]:
            keep_set.update(group)
        coord = _build_coord_text(
            selected_miu_indices=list(range(min(3, len(miu_token_idxs)))),
            keep_token_set=keep_set,
            miu_token_idxs=miu_token_idxs,
            mius=mius_a,
            policy=policy_abc,
        )
        check(
            "Module C — coord text",
            bool(
                coord
                and "<|box_start|>" in coord
                and ("miu_" in coord or "A1<|box_start|>" in coord)
            ),
            (coord or "")[:80],
        )
    check(
        "Module C — no_coord ablation",
        _build_coord_text([], set(), [], mius_a, "oas_ickh_no_coord_yolo") is None,
        "returns None",
    )
    pick_order = [0, 5, 10, 2, 7]
    if mius_a and len(mius_a) >= 2:
        mapped_token_idxs, _, _, mapped_miu_indices = _map_mius_to_tokens(
            mius_a, image_embeds, grid_h, grid_w, n_img
        )
        miu_token_idxs_pick = [[] for _ in mius_a]
        for group, source_idx in zip(mapped_token_idxs, mapped_miu_indices):
            miu_token_idxs_pick[source_idx] = group
        ordered = _coord_miu_indices_from_pick_order(
            pick_order, miu_token_idxs_pick, mius_a, topk=3
        )
        check(
            "Module C — pick-order topk",
            len(ordered) <= 3 and all(isinstance(i, int) for i in ordered),
            f"ordered={ordered}",
        )

    # ── Site-packages patches ──────────────────────────────────────────────
    try:
        import inspect
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        src = inspect.getsource(Qwen3VLModel.forward)
        check("Qwen3-VL GeoPACT patch", "GeoPACT" in src, "forward contains GeoPACT hook")
        check("Qwen3-VL deepstack subset", "visual_keep_flat_indices" in src or "index_select" in src)
    except Exception as exc:
        check("Qwen3-VL patch import", False, str(exc))

    if failures:
        print(f"\nABC STACK FAILED ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nABC STACK PASSED — safe to launch GeoPACT ablation fleet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
