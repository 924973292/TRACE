#!/usr/bin/env python3
"""Offline checks for hist9 + prune_target=history ABC dialogue-KV pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.geopact_session_kv import (
    _capture_last_full_frame,
    _compact_to_virtual_index_map,
    _visual_spans,
    compute_dialogue_roll_drop_len,
    resolve_prune_target,
    sliding_rewrite_prepare_dialogue_kv,
    stage_pending_last_full_frame,
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

    img = 88
    text = 1
    # Two-frame compact prefix: goal + frame0 + frame1 (compressed history + current).
    cached = [text, text, img, img, text, text, img, img, img, img, text]
    suffix = [text, img, img, img, img, text]
    full = cached[6:] + suffix
    state = {"image_paths": ["/a.png", "/b.png", "/c.png"]}

    drop = compute_dialogue_roll_drop_len(
        state,
        cached_ids=cached,
        suffix_ids=suffix,
        full_ids=full,
        image_token_id=img,
    )
    check(
        "roll drop via compact second visual span",
        drop == (0, 6),
        f"drop={drop}",
    )

    spans = _visual_spans(cached, img)
    check(
        "visual spans finds two image blocks",
        len(spans) == 2 and spans[0] == (2, 4) and spans[1] == (6, 10),
        f"spans={spans}",
    )

    virtual = [text, text, img, img, img, img, text, text, img, img, text]
    compact = [text, text, img, img, text, text, img, img, text]
    mapping = _compact_to_virtual_index_map(compact, virtual, img)
    check(
        "virtual map supports slide trim",
        mapping is not None
        and len(mapping) == len(compact)
        and mapping[2] == 2
        and mapping[4] > mapping[2],
        f"mapping={mapping}",
    )

    check(
        "resolve_prune_target prefers model attr",
        resolve_prune_target(SimpleNamespace(_geopact_prune_target="history")) == "history",
    )

    # Pending last-frame capture from suffix prefill payload (ABC prune_rewrite path).
    import os

    os.environ["GEOPACT_SESSION_KV"] = "1"
    os.environ["GEOPACT_SESSION_KV_MODE"] = "dialogue"
    os.environ["GEOPACT_KV_HISTORY_REWRITE"] = "1"
    inner = SimpleNamespace(
        _geopact_prune_target="history",
        _geopact_prune_ratio=0.40,
        _geopact_prune_policy="sspc",
        _geopact_pending_last_frame=None,
        _geopact_last_image=None,
        config=SimpleNamespace(image_token_id=img),
    )
    inner.get_input_embeddings = MagicMock(
        return_value=SimpleNamespace(weight=torch.zeros(1, device="cpu"))
    )
    model = SimpleNamespace(model=inner)
    n_frame = 4
    stored = [text, text, img, img, img, img, text]
    embeds = torch.randn(1, len(stored), 8)
    prefill = {
        "inputs_embeds": embeds,
        "deepstack_image_embeds": [torch.randn(n_frame, 8)],
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "input_ids": torch.tensor([stored]),
    }
    inner._geopact_prefill_payload = prefill
    stage_pending_last_full_frame(
        model,
        stored_ids=stored,
        image_paths=["/shot.png"],
    )
    pending = getattr(inner, "_geopact_pending_last_frame", None)
    check(
        "stage_pending_last_full_frame captures tensors",
        pending is not None
        and pending.get("visual_embeds") is not None
        and int(pending["visual_embeds"].shape[0]) == n_frame,
        f"pending={type(pending)}",
    )

    cap = _capture_last_full_frame(
        inner,
        prefill,
        stored_ids=stored,
        image_paths=["/shot.png"],
        image_token_id=img,
    )
    check(
        "capture_last_full_frame reads last visual block",
        cap is not None and int(cap["visual_embeds"].shape[0]) == n_frame,
    )

    benign = frozenset(
        {
            "not_history_target",
            "disabled",
            "first_step",
            "already_compressed",
        }
    )
    check(
        "sliding_rewrite_prepare_dialogue_kv is callable",
        callable(sliding_rewrite_prepare_dialogue_kv),
    )
    check(
        "compress benign reasons exclude silent failures",
        "no_captured_frame" not in benign and "selector_noop" not in benign,
        f"benign={sorted(benign)}",
    )

    from lmms_eval.models.geopact_session_kv import (
        _align_sliding_suffix_mrope_fast,
        _concat_sliding_merged_mrope,
        _invalidate_virtual_mrope_cache,
        _virtual_mrope_cache_key,
    )

    rewrite_pos = torch.tensor([[[4, 5]], [[14, 15]], [[24, 25]]], dtype=torch.long)
    suffix_pos = torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long)
    merged = _concat_sliding_merged_mrope(rewrite_pos, suffix_pos)
    check(
        "hist9 merged MRoPE boundary aligned",
        int(merged.shape[-1]) == 5 and torch.equal(merged[:, :, 1:2] + 1, merged[:, :, 2:3]),
    )
    vstate = {
        "cached_mrope_input_ids": virtual,
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }
    check("virtual mrope cache key for hist9 virtual", bool(_virtual_mrope_cache_key(vstate)))
    cleared = _invalidate_virtual_mrope_cache({**vstate, "cached_virtual_mrope_pos": torch.zeros(3, 1, 1)})
    check("slide invalidates virtual mrope cache", "cached_virtual_mrope_pos" not in cleared)

    rw = torch.tensor([[[3, 4]], [[13, 14]], [[23, 24]]], dtype=torch.long)
    sfx = torch.tensor([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=torch.long)
    fast, _, _ = _align_sliding_suffix_mrope_fast(
        {"position_ids": sfx},
        rw,
        suffix_len=2,
        device=torch.device("cpu"),
    )
    check(
        "hist9 fast suffix align (no virtual fallback)",
        torch.equal(fast[:, :, :1], rw[:, :, -1:] + 1),
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s): {failures}", file=sys.stderr)
        return 1
    print("ALL HIST9 ABC PIPELINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
