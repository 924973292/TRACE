#!/usr/bin/env python3
"""Unit checks for flash-safe G²-RPB (ES-RPB + IoU) and GAEC v2."""
from __future__ import annotations

import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lmms_eval.models.geopact_coord_injection.gaec import (
    _center_compact_from_origpos,
    gaec_v2_enabled,
)
from lmms_eval.models.geopact_coord_injection.slot import (
    InjectionSlot,
    resolve_insert_pos,
)
from lmms_eval.models.geopact_spatial_rpb import (
    _build_es_rpb_weights,
    _build_iou_matrix,
    _iou_boxes,
    apply_embedding_space_rpb,
    extract_hw_from_compact_pos,
    spatial_rpb_enabled,
)


def ok(name: str) -> None:
    print(f"[OK] {name}")


def test_slot_gen_tail_fallback() -> None:
    ins = resolve_insert_pos(
        InjectionSlot.BEFORE_GEN_TAIL,
        compact_seq_len=100,
        gen_tail_len=0,
        last_vision_pos=40,
        ephemeral_mode=True,
    )
    assert ins == 41, ins
    ok("slot BEFORE_GEN_TAIL fallback when gen_tail_len=0")


def test_es_rpb_changes_visual_embs() -> None:
    os.environ["GEOPACT_SPATIAL_RPB"] = "1"
    assert spatial_rpb_enabled()
    seq = 8
    hidden = 16
    compact_embs = torch.randn(1, seq, hidden)
    compact_pos = torch.zeros(3, 1, seq, dtype=torch.long)
    compact_pos[1, 0, :] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    compact_pos[2, 0, :] = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    visual_idx = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    before = compact_embs.clone()
    out = apply_embedding_space_rpb(compact_embs, compact_pos, visual_idx, None)
    diff = (out[0, visual_idx] - before[0, visual_idx]).abs().max().item()
    assert diff > 0, diff
    ok(f"ES-RPB modifies visual embs max_delta={diff:.4f}")


def test_shuffle_degrades_vs_correct_geometry() -> None:
    seq = 6
    compact_pos = torch.zeros(3, 1, seq, dtype=torch.long)
    compact_pos[1, 0, :] = torch.tensor([0, 10, 20, 30, 40, 50])
    compact_pos[2, 0, :] = torch.tensor([0, 1, 2, 3, 4, 5])
    h, w = extract_hw_from_compact_pos(compact_pos)
    visual_idx = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    w_ok = _build_es_rpb_weights(h, w, visual_idx, None, shuffle=False)
    w_shuf = _build_es_rpb_weights(h, w, visual_idx, None, shuffle=True)
    diff = (w_ok - w_shuf).abs().max().item()
    assert diff > 1e-4, diff
    ok(f"shuffle v2 changes ES-RPB weights max_delta={diff:.4f}")


def test_iou_boxes() -> None:
    a = (0.0, 0.0, 0.5, 0.5)
    b = (0.25, 0.25, 0.75, 0.75)
    iou = _iou_boxes(a, b)
    assert 0.0 < iou < 1.0, iou
    mat = _build_iou_matrix([a, b], torch.device("cpu"))
    assert abs(float(mat[0, 1]) - iou) < 1e-6, (float(mat[0, 1]), iou)
    ok(f"IoU matrix iou={iou:.4f}")


def test_iou_bias_changes_weights() -> None:
    os.environ["GEOPACT_RPB_IOU"] = "1"
    h = torch.tensor([0.0, 10.0, 20.0, 30.0])
    w = torch.tensor([0.0, 1.0, 2.0, 3.0])
    visual_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    labels = torch.tensor([-2, 0, 1, 1], dtype=torch.long)
    mius = [
        {"box": (0.0, 0.0, 0.2, 0.2)},
        {"box": (0.5, 0.5, 0.9, 0.9)},
        {"box": (0.55, 0.55, 0.95, 0.95)},
    ]
    w_man = _build_es_rpb_weights(h, w, visual_idx, labels, mius=mius)
    os.environ["GEOPACT_RPB_IOU"] = "0"
    w_no = _build_es_rpb_weights(h, w, visual_idx, labels, mius=mius)
    diff = (w_man - w_no).abs().max().item()
    assert diff > 1e-4, diff
    ok(f"IoU bias changes weights max_delta={diff:.4f}")
    os.environ.pop("GEOPACT_RPB_IOU", None)


def test_gaec_v2_center_from_origpos() -> None:
    os.environ["GEOPACT_GAEC_V2"] = "1"
    assert gaec_v2_enabled()
    seq = 5
    compact_pos = torch.zeros(3, 1, seq, dtype=torch.long)
    compact_pos[1, 0, :] = torch.tensor([10, 20, 30, 40, 50])
    compact_pos[2, 0, :] = torch.tensor([1, 2, 3, 4, 5])
    row_ids = torch.tensor([1, 151655, 151655, 151655, 1])
    visual_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    mius = [{"box": (0.0, 0.0, 0.1, 0.1)}]
    ci = _center_compact_from_origpos(0, mius, visual_idx, compact_pos, row_ids, 151655)
    assert ci == 1, ci
    ok("GAEC v2 center picks nearest origpos visual token")
    os.environ.pop("GEOPACT_GAEC_V2", None)


def test_extract_hw() -> None:
    compact_pos = torch.zeros(3, 1, 4, dtype=torch.long)
    compact_pos[1, 0, :] = torch.tensor([10, 20, 30, 40])
    compact_pos[2, 0, :] = torch.tensor([1, 2, 3, 4])
    h, w = extract_hw_from_compact_pos(compact_pos)
    assert int(h[0]) == 10 and int(w[3]) == 4
    ok("extract_hw_from_compact_pos")


def main() -> int:
    test_slot_gen_tail_fallback()
    test_extract_hw()
    test_es_rpb_changes_visual_embs()
    test_shuffle_degrades_vs_correct_geometry()
    test_iou_boxes()
    test_iou_bias_changes_weights()
    test_gaec_v2_center_from_origpos()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
