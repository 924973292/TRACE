#!/usr/bin/env python3
"""Offline sliding-rewrite MRoPE / cache / state invariants (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.geopact_session_kv import (
    _align_sliding_suffix_mrope_fast,
    _anchor_prefix_len,
    _anchored_evict_past_key_values,
    _concat_sliding_merged_mrope,
    _get_cached_virtual_full_mrope,
    _invalidate_virtual_mrope_cache,
    _kv_compressed_frame_key,
    _mrope_pos_offset,
    _normalise_kv_compressed_keys,
    _rope_deltas_for_position_ids,
    _rotate_post_rope_keys,
    _shift_kv_compressed_keys_after_slide,
    _store_cached_virtual_full_mrope,
    _virtual_mrope_cache_key,
    _visual_skeleton,
    compute_dialogue_roll_drop_len,
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

    rewrite_pos = torch.tensor([[[10, 11, 12]], [[20, 21, 22]], [[30, 31, 32]]], dtype=torch.long)
    suffix_pos = torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 2, 3]]], dtype=torch.long)
    merged = _concat_sliding_merged_mrope(rewrite_pos, suffix_pos)
    check(
        "merged_pos continuity (rewrite_tail+1 == suffix_head)",
        torch.equal(merged[:, :, 2:3] + 1, merged[:, :, 3:4]),
        f"rewrite_tail={merged[:, :, 2].tolist()} suffix_head={merged[:, :, 3].tolist()}",
    )
    check(
        "merged_pos length",
        int(merged.shape[-1]) == 7,
        f"len={int(merged.shape[-1])}",
    )

    past_len = 12
    prefill_len = 5
    cache_pos = torch.arange(past_len, past_len + prefill_len, dtype=torch.long)
    check(
        "cache_position monotonic",
        int(cache_pos[0]) == past_len
        and int(cache_pos[-1]) == past_len + prefill_len - 1
        and int(cache_pos.shape[0]) == prefill_len,
        f"cache_pos={cache_pos.tolist()}",
    )

    virtual_ids = [1, 2, 88, 88, 88, 88, 1, 1, 88, 88, 1]
    state = {
        "cached_mrope_input_ids": virtual_ids,
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
    }
    key = _virtual_mrope_cache_key(state)
    check("virtual mrope cache key non-empty", bool(key), f"key={key}")

    full_pos = torch.arange(0, len(virtual_ids) * 3, dtype=torch.long).view(3, 1, -1)
    stored = _store_cached_virtual_full_mrope(state, full_pos)
    check(
        "virtual mrope cache stored",
        stored.get("cached_virtual_mrope_key") == key
        and stored.get("cached_virtual_mrope_pos") is not None,
    )
    hit = _get_cached_virtual_full_mrope(stored, device=torch.device("cpu"))
    check("virtual mrope cache hit", hit is not None and int(hit.shape[-1]) == len(virtual_ids))

    invalidated = _invalidate_virtual_mrope_cache(stored)
    check(
        "virtual mrope cache invalidate",
        "cached_virtual_mrope_pos" not in invalidated,
    )
    remiss = _get_cached_virtual_full_mrope(invalidated, device=torch.device("cpu"))
    check("virtual mrope cache miss after invalidate", remiss is None)

    misaligned = torch.tensor([[[50, 51]], [[50, 51]], [[50, 51]]], dtype=torch.long)
    fixed = _concat_sliding_merged_mrope(rewrite_pos, misaligned)
    check(
        "misaligned suffix realigned before concat",
        torch.equal(fixed[:, :, 2:3] + 1, fixed[:, :, 3:4]),
        f"joint={fixed[:, :, 2:4].tolist()}",
    )

    suffix_local = torch.tensor([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long)
    fast_pos, fast_rope, fast_ms = _align_sliding_suffix_mrope_fast(
        {"position_ids": suffix_local},
        rewrite_pos,
        suffix_len=3,
        device=torch.device("cpu"),
    )
    check(
        "fast suffix align matches merged concat head",
        torch.equal(fast_pos[:, :, :1], rewrite_pos[:, :, -1:] + 1),
        f"fast_head={fast_pos[:, :, 0].tolist()} ms={fast_ms}",
    )
    check("fast suffix rope_deltas shape", int(fast_rope.shape[-1]) == 1)

    from lmms_eval.models.geopact_decode_contract import (
        attach_pinned_decode_contract,
        compute_pinned_cache_start,
        contract_complete,
    )

    check(
        "step0 pinned_cache_start",
        compute_pinned_cache_start(past_kv_len=0, prompt_token_count=1825) == 1824,
        "last prompt index = len-1",
    )
    step0_payload = attach_pinned_decode_contract(
        {
            "position_ids": torch.zeros(3, 1, 100, dtype=torch.long),
            "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
        },
        past_kv_len=0,
        logical_prompt_len=100,
    )
    check(
        "attach step0 contract",
        contract_complete(step0_payload)
        and int(step0_payload["_geopact_pinned_cache_start"]) == 99,
    )
    suffix_payload = attach_pinned_decode_contract(
        {
            "position_ids": torch.zeros(3, 1, 1, dtype=torch.long),
            "rope_deltas": torch.zeros(1, 1, dtype=torch.long),
            "past_key_values": type("PKV", (), {"get_seq_length": lambda self: 1800})(),
            "_geopact_pinned_cache_start": 1824,
        },
        past_kv_len=1800,
    )
    check(
        "suffix contract preserves pinned start",
        int(suffix_payload["_geopact_pinned_cache_start"]) == 1824,
    )
    anchor = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.long)
    for step in (1, 2, 5):
        pos = anchor + step
        check(
            f"decode offset monotonic step={step}",
            int(pos[0, 0, 0].item()) == 10 + step,
        )

    key0 = _kv_compressed_frame_key("/tmp/reused.png", 0)
    key2 = _kv_compressed_frame_key("/tmp/reused.png", 2)
    shifted = _shift_kv_compressed_keys_after_slide({key0, key2, "/tmp/legacy.png"})
    check(
        "compressed frame keys are instance-scoped",
        key0 != key2 and key2 in _normalise_kv_compressed_keys({key0, key2}),
    )
    check(
        "compressed frame keys shift after slide",
        key0 not in shifted and ("/tmp/reused.png", 1) in shifted,
        f"shifted={sorted(shifted)}",
    )

    boundary = torch.tensor([[[4]], [[7]], [[9]]], dtype=torch.long)
    slide_rope = _rope_deltas_for_position_ids(boundary, 5)
    check(
        "slide rope_deltas recompute from boundary",
        int(slide_rope.item()) == 5,
        f"rope={slide_rope.tolist()}",
    )

    # --- Bug B: pruned-prefix slide boundary must be proven, never blind ---
    check(
        "visual skeleton collapses image runs",
        _visual_skeleton([1, 9, 9, 9, 2, 9, 3], 9) == [1, -1, 2, -1, 3],
        f"sk={_visual_skeleton([1, 9, 9, 9, 2, 9, 3], 9)}",
    )

    # Window layout: [text 11,12][img][asst 13] | [text 21,22][img][asst 23]
    # cached is ABC-pruned (both runs shorter than the 4-token full runs).
    pruned_cached = [11, 12, 9, 9, 13, 21, 22, 9, 9, 9, 23]
    pruned_state = {
        "image_paths": ["a.png", "b.png"],
        "cached_mrope_input_ids": [11, 12, 9, 9, 9, 9, 13, 21, 22, 9, 9, 9, 9, 23],
    }
    suffix_ids = [31, 32, 9, 9, 9, 9]
    full_ids = [21, 22, 9, 9, 9, 9, 23] + suffix_ids
    drop = compute_dialogue_roll_drop_len(
        pruned_state,
        cached_ids=pruned_cached,
        suffix_ids=suffix_ids,
        full_ids=full_ids,
        image_token_id=9,
    )
    check(
        "pruned roll drop lands on text head (keeps user turn header)",
        drop == (0, 5),
        f"drop={drop} (blind spans[1] slide would give 7 and cut [21,22])",
    )
    drop_bad = compute_dialogue_roll_drop_len(
        pruned_state,
        cached_ids=pruned_cached,
        suffix_ids=[4, 5],
        full_ids=[999, 888],
        image_token_id=9,
    )
    check(
        "unproven pruned boundary rejected (forces full reset)",
        drop_bad is None,
        f"drop={drop_bad}",
    )
    drop_no_full = compute_dialogue_roll_drop_len(
        pruned_state,
        cached_ids=pruned_cached,
        suffix_ids=suffix_ids,
        full_ids=None,
        image_token_id=9,
    )
    check(
        "pruned boundary without full_ids rejected",
        drop_no_full is None,
        f"drop={drop_no_full}",
    )

    # --- Anchored eviction: keep the system anchor, evict the oldest turn ---
    ims = 7
    sys_seg = [ims, 100, 101, 102]
    turn0 = [ims, 110, 9, 9, 111]
    turn1 = [ims, 120, 9, 9, 121]
    turn2 = [ims, 130, 9, 9, 131]
    anch_cached = sys_seg + turn0 + turn1
    anch_full = sys_seg + turn1 + turn2
    check(
        "anchor prefix len locates second im_start",
        _anchor_prefix_len(anch_cached, ims) == len(sys_seg)
        and _anchor_prefix_len(anch_cached, -1) == 0
        and _anchor_prefix_len(sys_seg, ims) == 0,
        f"anchor={_anchor_prefix_len(anch_cached, ims)}",
    )
    anch_state = {"image_paths": ["a.png", "b.png"]}
    boundary = compute_dialogue_roll_drop_len(
        anch_state,
        cached_ids=anch_cached,
        suffix_ids=turn2,
        full_ids=anch_full,
        image_token_id=9,
        im_start_id=ims,
    )
    check(
        "anchored exact eviction proves (anchor, oldest-turn drop)",
        boundary == (len(sys_seg), len(turn0)),
        f"boundary={boundary}",
    )
    boundary_bad_sys = compute_dialogue_roll_drop_len(
        anch_state,
        cached_ids=anch_cached,
        suffix_ids=turn2,
        full_ids=[ims, 999, 998, 997] + turn1 + turn2,
        image_token_id=9,
        im_start_id=ims,
    )
    check(
        "anchored eviction with mutated system prefix rejected",
        boundary_bad_sys is None,
        f"boundary={boundary_bad_sys}",
    )
    # ABC-pruned anchored form: cached turn1 visual run shorter than full's.
    turn1_pruned = [ims, 120, 9, 121]
    anch_cached_pruned = sys_seg + turn0 + turn1_pruned
    anch_state_pruned = {
        "image_paths": ["a.png", "b.png"],
        "cached_mrope_input_ids": sys_seg + turn0 + turn1,
    }
    boundary_pruned = compute_dialogue_roll_drop_len(
        anch_state_pruned,
        cached_ids=anch_cached_pruned,
        suffix_ids=turn2,
        full_ids=anch_full,
        image_token_id=9,
        im_start_id=ims,
    )
    check(
        "anchored skeleton eviction under pruned prefix",
        boundary_pruned == (len(sys_seg), len(turn0)),
        f"boundary={boundary_pruned}",
    )

    # --- Anchored eviction KV op + post-RoPE key rotation ---
    torch.manual_seed(0)
    head_dim = 8
    inv_freq = 1.0 / (100.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    def ref_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ang = positions.view(-1, 1).float() * inv_freq.view(1, -1)
        cos = torch.cos(ang).repeat(1, 2).view(1, 1, -1, head_dim)
        sin = torch.sin(ang).repeat(1, 2).view(1, 1, -1, head_dim)
        half = head_dim // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rot * sin

    raw_k = torch.randn(1, 2, 5, head_dim)
    pos = torch.arange(5)
    delta_shift = 37
    k_at_pos = ref_rope(raw_k, pos)
    k_shifted = _rotate_post_rope_keys(k_at_pos, delta_shift, inv_freq)
    k_expected = ref_rope(raw_k, pos + delta_shift)
    check(
        "post-RoPE key rotation == re-applying RoPE at shifted positions",
        torch.allclose(k_shifted, k_expected, atol=1e-5),
        f"max_err={(k_shifted - k_expected).abs().max().item():.2e}",
    )
    check(
        "zero-delta rotation is identity (same tensor)",
        _rotate_post_rope_keys(k_at_pos, 0, inv_freq) is k_at_pos,
    )

    seq, anchor_n, drop_n = 10, 3, 4
    legacy_k = torch.randn(1, 2, seq, head_dim)
    legacy_v = torch.randn(1, 2, seq, head_dim)
    rot_sys = [torch.randn(1, 2, anchor_n, head_dim)]
    evicted = _anchored_evict_past_key_values(
        ((legacy_k, legacy_v),),
        anchor_len=anchor_n,
        drop_len=drop_n,
        rotated_sys_keys=rot_sys,
    )
    ev_legacy = evicted.to_legacy_cache() if hasattr(evicted, "to_legacy_cache") else evicted
    ev_k, ev_v = ev_legacy[0][0], ev_legacy[0][1]
    check(
        "anchored KV eviction: seq length and segment layout",
        int(ev_k.shape[-2]) == seq - drop_n
        and torch.equal(ev_k[..., :anchor_n, :], rot_sys[0])
        and torch.equal(ev_k[..., anchor_n:, :], legacy_k[..., anchor_n + drop_n :, :])
        and torch.equal(ev_v[..., :anchor_n, :], legacy_v[..., :anchor_n, :])
        and torch.equal(ev_v[..., anchor_n:, :], legacy_v[..., anchor_n + drop_n :, :]),
        f"seq={int(ev_k.shape[-2])}",
    )
    try:
        _anchored_evict_past_key_values(
            ((legacy_k, legacy_v),),
            anchor_len=anchor_n,
            drop_len=drop_n,
            rotated_sys_keys=None,
        )
        anchored_missing_ok = False
    except RuntimeError:
        anchored_missing_ok = True
    check("anchored eviction without rotated sys keys rejected", anchored_missing_ok)

    # --- Bug A: slide position offset continuation ---
    check(
        "mrope pos offset default/read",
        _mrope_pos_offset({}) == 0 and _mrope_pos_offset({"mrope_virt_pos_offset": 7}) == 7,
    )
    off_state_a = {
        "cached_mrope_input_ids": virtual_ids,
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long),
        "mrope_virt_pos_offset": 0,
    }
    off_state_b = dict(off_state_a, mrope_virt_pos_offset=5)
    check(
        "virtual mrope cache key fingerprint includes pos offset",
        _virtual_mrope_cache_key(off_state_a) != _virtual_mrope_cache_key(off_state_b),
    )
    # Runtime slide check mirror: old-frame tail == 0-anchored recompute + const.
    old_full = torch.arange(0, 3 * len(virtual_ids), dtype=torch.long).view(3, 1, -1)
    virt_drop = 4
    pos_offset = 6
    new_raw = old_full[:, :, virt_drop:] - pos_offset
    delta = old_full[:, :, virt_drop:] - new_raw
    check(
        "slide offset delta constant across channels/positions",
        bool((delta == delta.flatten()[0]).all()) and int(delta.flatten()[0]) == pos_offset,
    )
    shifted_state = dict(off_state_b, cached_mrope_input_ids=virtual_ids[virt_drop:])
    stored_tail = _store_cached_virtual_full_mrope(
        shifted_state,
        old_full[:, :, virt_drop:],
    )
    tail_hit = _get_cached_virtual_full_mrope(stored_tail, device=torch.device("cpu"))
    check(
        "post-slide cache stores old-frame tail (offset continuation)",
        tail_hit is not None
        and torch.equal(tail_hit, old_full[:, :, virt_drop:]),
    )

    from lmms_eval.models.geopact_vl_hook import rectify_history_frame_from_full
    from types import SimpleNamespace

    full_embeds = torch.arange(12, dtype=torch.float32).view(4, 3)
    kept, deepstack, local_keep, rectify_stats = rectify_history_frame_from_full(
        SimpleNamespace(),
        cap={
            "visual_embeds": full_embeds,
            "grid_row": torch.tensor([[1, 2, 2]], dtype=torch.long),
        },
        ratio=1.0,
        policy="",
        n_frame=4,
    )
    check(
        "ratio_noop returns full embeds",
        kept is full_embeds
        and deepstack is None
        and local_keep == [0, 1, 2, 3]
        and rectify_stats.get("rectify_reason") == "ratio_noop",
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s): {failures}", file=sys.stderr)
        return 1
    print("ALL SLIDING REWRITE INVARIANTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
