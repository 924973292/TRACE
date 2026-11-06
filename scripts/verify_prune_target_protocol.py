#!/usr/bin/env python3
"""Verify the single-step (current) vs multi-step (history) prune-target protocol.

Covers the pure-logic layers that do not need a loaded model:
  - `_sticky_frame_action` decision table
  - `_prune_target` env / model resolution
  - `_dialogue_prefetch_single_frame` history gating
  - `compute_sticky_dialogue_prune_keep_set` current vs history behaviour on a
    multi-frame episode (which frame is pruned, which is kept full, which reuses)
  - `_last_visual_span` for the KV rewrite locator

Run: python scripts/verify_prune_target_protocol.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from lmms_eval.models.geopact_vl_hook import (
    _dialogue_prefetch_single_frame,
    _prune_target,
    _sticky_frame_action,
    compute_sticky_dialogue_prune_keep_set,
)
from lmms_eval.models.geopact_session_kv import _last_visual_span


def _ok(name: str, detail: str = "") -> None:
    print(f"[OK] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""), file=sys.stderr)


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            _ok(name, detail)
        else:
            _fail(name, detail)
            failures.append(name)

    # ── decision table ──────────────────────────────────────────────
    cases = [
        # (target, is_current, has_cached) -> action
        ("current", True, False, "prune"),
        ("current", True, True, "prune"),
        ("current", False, True, "reuse"),
        ("current", False, False, "full"),
        ("history", True, False, "full"),
        ("history", True, True, "full"),
        ("history", False, True, "reuse"),
        ("history", False, False, "prune"),
    ]
    for target, is_cur, cached, expected in cases:
        got = _sticky_frame_action(
            is_current=is_cur, prune_target=target, has_cached_mask=cached
        )
        check(
            f"action[{target},cur={is_cur},cached={cached}]={expected}",
            got == expected,
            f"got={got}",
        )

    # ── _prune_target resolution ────────────────────────────────────
    os.environ.pop("GEOPACT_PRUNE_TARGET", None)
    check("default target current", _prune_target(None) == "current")
    check(
        "model attr history wins",
        _prune_target(SimpleNamespace(_geopact_prune_target="history")) == "history",
    )
    os.environ["GEOPACT_PRUNE_TARGET"] = "history"
    check("env history applies", _prune_target(None) == "history")
    os.environ.pop("GEOPACT_PRUNE_TARGET", None)
    check(
        "invalid model target falls back current",
        _prune_target(SimpleNamespace(_geopact_prune_target="bogus")) == "current",
    )

    # ── prefetch single-frame gating ────────────────────────────────
    os.environ["GEOPACT_SESSION_KV"] = "1"
    os.environ["GEOPACT_SESSION_KV_MODE"] = "dialogue"
    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "1"
    check(
        "prefetch single-frame off under history",
        not _dialogue_prefetch_single_frame(
            SimpleNamespace(_geopact_prune_target="history")
        ),
    )
    check(
        "prefetch single-frame on under current",
        _dialogue_prefetch_single_frame(
            SimpleNamespace(_geopact_prune_target="current")
        ),
    )
    os.environ.pop("GEOPACT_MODULE_A_PREFETCH", None)
    os.environ.pop("GEOPACT_SESSION_KV_MODE", None)
    os.environ.pop("GEOPACT_SESSION_KV", None)

    # ── multi-frame sticky keep sets ────────────────────────────────
    # 3 frames of 20 tokens each; selector keeps 8 (40%).
    torch.manual_seed(0)
    n_frame = 20
    hidden = 8
    n_frames = 3
    image_embeds = torch.randn(n_frame * n_frames, hidden)
    grid = torch.tensor([[1, 4, 5]] * n_frames)
    token_ranges = [(i * n_frame, (i + 1) * n_frame) for i in range(n_frames)]
    # deterministic keep of the *local* first 8 indices for whatever frame is pruned
    keep_local = {0, 1, 2, 3, 4, 5, 6, 7}

    def _fake_prune(*, start, end, **_kw):
        return ({start + i for i in keep_local}, [], [{"box": (0, 0, 1, 1)}], None)

    # history target: current (frame 2) kept full, frames 0/1 pruned + persisted.
    hist_model = SimpleNamespace(
        _geopact_episode_key="ep",
        _geopact_image_paths=["f0.png", "f1.png", "f2.png"],
        _geopact_prune_scope="image_sticky",
        _geopact_prune_target="history",
        _geopact_miu_detector="omniparser",
        _geopact_last_image=object(),
        _geopact_sticky_masks={},
        _geopact_last_prune_diag={},
    )
    with patch(
        "lmms_eval.models.geopact_vl_hook._prune_single_frame", side_effect=_fake_prune
    ):
        keep_hist = compute_sticky_dialogue_prune_keep_set(
            hist_model,
            token_ranges=token_ranges,
            image_embeds=image_embeds,
            image_grid_thw=grid,
            ratio=0.4,
            policy="oas_ickh_coord_center_yolo",
            scope="image_sticky",
        )
    # frame 2 (current) fully kept
    cur_full = set(range(token_ranges[2][0], token_ranges[2][1]))
    check("history: current frame kept full", cur_full <= keep_hist)
    # frames 0/1 pruned to 8 each
    f0_keep = {token_ranges[0][0] + i for i in keep_local}
    f1_keep = {token_ranges[1][0] + i for i in keep_local}
    check("history: frame0 pruned to keep set", {g for g in keep_hist if g < 20} == f0_keep)
    check(
        "history: frame1 pruned to keep set",
        {g for g in keep_hist if 20 <= g < 40} == f1_keep,
    )
    check(
        "history: masks persisted for history frames",
        len(hist_model._geopact_sticky_masks) == 2,
        f"masks={list(hist_model._geopact_sticky_masks.keys())}",
    )

    # current target: only frame 2 pruned; frames 0/1 full (no cached mask).
    cur_model = SimpleNamespace(
        _geopact_episode_key="ep",
        _geopact_image_paths=["f0.png", "f1.png", "f2.png"],
        _geopact_prune_scope="image_sticky",
        _geopact_prune_target="current",
        _geopact_miu_detector="omniparser",
        _geopact_last_image=object(),
        _geopact_sticky_masks={},
        _geopact_last_prune_diag={},
    )
    with patch(
        "lmms_eval.models.geopact_vl_hook._prune_single_frame", side_effect=_fake_prune
    ):
        keep_cur = compute_sticky_dialogue_prune_keep_set(
            cur_model,
            token_ranges=token_ranges,
            image_embeds=image_embeds,
            image_grid_thw=grid,
            ratio=0.4,
            policy="oas_ickh_coord_center_yolo",
            scope="image_sticky",
        )
    f0_full = set(range(0, 20))
    f1_full = set(range(20, 40))
    f2_keep = {token_ranges[2][0] + i for i in keep_local}
    check("current: frame0 kept full", {g for g in keep_cur if g < 20} == f0_full)
    check("current: frame1 kept full", {g for g in keep_cur if 20 <= g < 40} == f1_full)
    check("current: frame2 pruned", {g for g in keep_cur if g >= 40} == f2_keep)

    # ── _last_visual_span ───────────────────────────────────────────
    img = 999
    ids = [1, 2, img, img, img, 3, 4, img, img, 5]
    span = _last_visual_span(ids, img)
    check("last visual span located", span == (7, 9), f"span={span}")
    check("no visual span returns None", _last_visual_span([1, 2, 3], img) is None)
    check("trailing visual span", _last_visual_span([1, img, img], img) == (1, 3))

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s): {failures}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
