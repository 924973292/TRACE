#!/usr/bin/env python3
"""Verify dialogue KV MRoPE boundary refresh and suffix alignment invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.geopact_session_kv import (
    _align_suffix_mrope_to_cached_boundary,
    _compact_to_virtual_index_map,
    _refresh_dialogue_boundary_in_state,
    _recompute_cached_boundary_position_ids,
)


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

    img = 999
    text = 1
    virtual = [text, img, img, img, img, text, img, img, text]
    compact = [text, img, img, text, img, img, text]
    mapping = _compact_to_virtual_index_map(compact, virtual, img)
    check(
        "compact_to_virtual maps pruned image blocks",
        mapping == [0, 1, 2, 5, 6, 7, 8],
        f"mapping={mapping}",
    )

    boundary = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.long)
    suffix = torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long)
    aligned = _align_suffix_mrope_to_cached_boundary(suffix, boundary)
    check(
        "suffix aligns to boundary+1 on all MRoPE channels",
        torch.equal(aligned[:, :, :1], boundary + 1),
        f"first={aligned[:, :, 0].tolist()}",
    )

    compress_suffix_pos = torch.tensor([[[4, 5, 6]], [[14, 15, 16]], [[24, 25, 26]]], dtype=torch.long)
    state = {
        "cached_input_ids": compact,
        "cached_mrope_input_ids": virtual,
        "prompt_token_len": len(compact),
    }
    refreshed = _refresh_dialogue_boundary_in_state(
        inner=object(),
        state=state,
        compress_suffix_pos=compress_suffix_pos,
    )
    b = refreshed.get("cached_boundary_position_ids")
    check(
        "compress refresh stores boundary from suffix_pos tail",
        b is not None
        and int(b.shape[-1]) == 1
        and int(b.shape[0]) == 3
        and int(b[2, 0, 0]) == 26,
        f"boundary={None if b is None else b.tolist()}",
    )

    direct = _recompute_cached_boundary_position_ids(
        object(),
        state,
        compress_suffix_pos=compress_suffix_pos,
    )
    check(
        "recompute with compress_suffix_pos matches refresh",
        direct is not None and b is not None and torch.equal(direct, b),
    )

    from lmms_eval.models.geopact_session_kv import _concat_sliding_merged_mrope

    rewrite_pos = torch.tensor([[[10, 11, 12]], [[20, 21, 22]], [[30, 31, 32]]], dtype=torch.long)
    suffix_pos = torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long)
    merged = _concat_sliding_merged_mrope(rewrite_pos, suffix_pos)
    check(
        "merged sliding MRoPE step==1 on three channels",
        torch.equal(merged[:, :, 2:3] + 1, merged[:, :, 3:4]),
        f"joint={merged[:, :, 2:4].tolist()}",
    )

    from lmms_eval.models.geopact_sspc import apply_embedding_zafd

    emb = torch.randn(576, 32)
    out, audit = apply_embedding_zafd(
        emb,
        keep_b=set(range(100)),
        keep_final=set(range(100)),
        frame_start=0,
        n_frame=576,
        grid_h=13,
        grid_w=13,
    )
    check(
        "ZAFD rejects unmatched grid",
        audit.get("sspc_zafd") is False and audit.get("sspc_zafd_reason") == "grid_mismatch",
        f"audit={audit.get('sspc_zafd_reason')}",
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s): {failures}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
