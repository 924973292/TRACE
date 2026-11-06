#!/usr/bin/env python3
"""Bitwise equivalence checks for origpos compact gather helpers."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from lmms_eval.models.geopact_vl_hook import (
    _build_keep_mask,
    _gather_compact_prefill_tensors,
    _subset_origpos_mrope,
)


def _legacy_build_keep_mask(image_mask: torch.Tensor, img_seq_positions: torch.Tensor, keep_token_set: set):
    keep_mask = ~image_mask.clone()
    if keep_token_set:
        vi = torch.tensor(
            [int(x) for x in keep_token_set],
            dtype=torch.long,
            device=image_mask.device,
        )
        keep_mask[img_seq_positions.index_select(0, vi)] = True
    return keep_mask


def _legacy_gather(
    input_ids,
    inputs_embeds,
    image_embeds,
    attention_mask,
    keep_mask,
    keep_token_set,
    img_seq_positions,
    image_tok_id,
):
    from lmms_eval.models.geopact_vl_hook import _visual_keep_in_sequence_order_tensor

    compact_ids = input_ids[:, keep_mask]
    compact_embs = inputs_embeds[:, keep_mask]
    if keep_token_set and int(image_embeds.shape[0]) > 0:
        keep_seq = _visual_keep_in_sequence_order_tensor(keep_token_set, img_seq_positions)
        vis_slots = (compact_ids[0] == image_tok_id).nonzero(as_tuple=False).flatten()
        if vis_slots.numel() > 0:
            vis_rows = image_embeds.index_select(0, keep_seq).to(compact_embs.dtype)
            compact_embs = compact_embs.clone()
            compact_embs[0, vis_slots] = vis_rows
    compact_attn = attention_mask[:, keep_mask] if attention_mask is not None else None
    return compact_ids, compact_embs, compact_attn


def test_keep_mask_and_gather(device: str) -> None:
    torch.manual_seed(42)
    seq_len = 128
    n_visual = 40
    image_tok_id = 7

    image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    img_positions = torch.arange(20, 20 + n_visual, device=device)
    image_mask[img_positions] = True

    input_ids = torch.randint(8, 100, (1, seq_len), device=device)
    input_ids[0, img_positions] = image_tok_id
    inputs_embeds = torch.randn(1, seq_len, 64, device=device, dtype=torch.float32)
    image_embeds = torch.randn(n_visual, 64, device=device, dtype=torch.float32)
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

    keep_token_set = {int(x) for x in torch.randperm(n_visual, device=device)[:30].tolist()}

    new_mask = _build_keep_mask(image_mask, img_positions, keep_token_set)
    old_mask = _legacy_build_keep_mask(image_mask, img_positions, keep_token_set)
    assert torch.equal(new_mask, old_mask), "keep_mask mismatch"

    new_out = _gather_compact_prefill_tensors(
        input_ids,
        inputs_embeds,
        image_embeds,
        attention_mask,
        new_mask,
        keep_token_set,
        img_positions,
        image_tok_id,
    )
    old_out = _legacy_gather(
        input_ids,
        inputs_embeds,
        image_embeds,
        attention_mask,
        old_mask,
        keep_token_set,
        img_positions,
        image_tok_id,
    )
    for i, name in enumerate(("compact_ids", "compact_embs", "compact_attn")):
        assert torch.equal(new_out[i], old_out[i]), f"{name} mismatch"


def test_subset_origpos(device: str) -> None:
    torch.manual_seed(7)
    seq_len = 64
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    keep_mask[torch.tensor([5, 12, 33], device=device)] = False
    position_ids = torch.stack(
        [
            torch.arange(seq_len, device=device),
            torch.arange(seq_len, device=device) // 4,
            torch.arange(seq_len, device=device) // 8,
        ],
        dim=0,
    ).unsqueeze(0)
    compact_ids = torch.randint(0, 50, (1, int(keep_mask.sum().item())), device=device)
    compact_attn = torch.ones(1, compact_ids.shape[1], device=device, dtype=torch.long)
    pos, rope = _subset_origpos_mrope(position_ids, keep_mask, compact_ids, compact_attn)
    expected_pos = position_ids[:, :, keep_mask].contiguous()
    assert torch.equal(pos, expected_pos), "subset position_ids mismatch"
    assert rope.dtype == torch.long and rope.numel() > 0


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_keep_mask_and_gather(device)
    test_subset_origpos(device)
    print(f"PASS compact_sequence equivalence on device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
