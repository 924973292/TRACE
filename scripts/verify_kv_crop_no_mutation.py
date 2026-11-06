#!/usr/bin/env python3
"""Regression check: _crop_past_key_values must never mutate/alias the source cache.

Root-cause context: transformers>=4.5x DynamicCache.crop() crops IN PLACE and
returns None. The old _crop_past_key_values called it on the live generate
cache from DialogueKVCapture._hook, deleting the assistant-header KV mid-decode
(baseline_full vs baseline_kv step0 drift). This verifier fails if that
behavior ever comes back.

Runs on CPU in seconds; no model weights needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    import torch
    from transformers.cache_utils import DynamicCache

    from lmms_eval.models.geopact_session_kv import _crop_past_key_values

    n_layers, n_heads, seq_len, head_dim = 4, 2, 16, 8
    crop_len = 12

    torch.manual_seed(0)
    src = DynamicCache()
    ref_tensors = []
    for layer_idx in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)
        ref_tensors.append((k.clone(), v.clone()))
        src.update(k, v, layer_idx)

    def _src_layer(idx):
        legacy = src.to_legacy_cache()
        return legacy[idx][0], legacy[idx][1]

    # Case 1: crop shorter than source.
    cropped = _crop_past_key_values(src, crop_len)
    if cropped is None:
        print("FAIL: crop returned None")
        return 1
    if int(cropped.get_seq_length()) != crop_len:
        print(f"FAIL: cropped length {cropped.get_seq_length()} != {crop_len}")
        return 1
    if int(src.get_seq_length()) != seq_len:
        print(
            "FAIL: source cache MUTATED by crop: "
            f"seq_length {src.get_seq_length()} != {seq_len}"
        )
        return 1
    for layer_idx in range(n_layers):
        src_k, src_v = _src_layer(layer_idx)
        ref_k, ref_v = ref_tensors[layer_idx]
        if not torch.equal(src_k, ref_k) or not torch.equal(src_v, ref_v):
            print(f"FAIL: source layer {layer_idx} tensors changed after crop")
            return 1
        crop_legacy = cropped.to_legacy_cache()
        ck, cv = crop_legacy[layer_idx][0], crop_legacy[layer_idx][1]
        if not torch.equal(ck, ref_k[..., :crop_len, :]) or not torch.equal(
            cv, ref_v[..., :crop_len, :]
        ):
            print(f"FAIL: cropped layer {layer_idx} != source slice [:{crop_len}]")
            return 1
        if ck.data_ptr() == src_k.data_ptr():
            print(f"FAIL: cropped layer {layer_idx} aliases source storage")
            return 1

    # Case 2: crop length == source length must still return an independent copy.
    same_len = _crop_past_key_values(src, seq_len)
    if same_len is None or int(same_len.get_seq_length()) != seq_len:
        print("FAIL: equal-length crop broken")
        return 1
    same_legacy = same_len.to_legacy_cache()
    src_k0, _ = _src_layer(0)
    if same_legacy[0][0].data_ptr() == src_k0.data_ptr():
        print("FAIL: equal-length crop aliases source storage")
        return 1
    same_legacy[0][0].add_(1.0)
    src_k0_after, _ = _src_layer(0)
    if not torch.equal(src_k0_after, ref_tensors[0][0]):
        print("FAIL: mutating equal-length crop result changed the source cache")
        return 1

    # Case 3: writing to source after crop must not change the cropped copy.
    cropped_k0_before = cropped.to_legacy_cache()[0][0].clone()
    sk, _ = _src_layer(0)
    sk.add_(5.0)
    if not torch.equal(cropped.to_legacy_cache()[0][0], cropped_k0_before):
        print("FAIL: cropped cache aliases source (source write leaked in)")
        return 1

    # Case 4: owned in-place crop — same object back, truncated, values equal
    # to the clone-crop result (zero-copy fast path for bootstrap/finalize).
    from lmms_eval.models.geopact_session_kv import _crop_past_key_values_owned

    torch.manual_seed(1)
    owned = DynamicCache()
    owned_ref = []
    for layer_idx in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)
        owned_ref.append((k.clone(), v.clone()))
        owned.update(k, v, layer_idx)
    owned_out = _crop_past_key_values_owned(owned, crop_len)
    if owned_out is not owned:
        print("FAIL: owned crop must return the same cache object (in place)")
        return 1
    if int(owned_out.get_seq_length()) != crop_len:
        print(
            f"FAIL: owned crop length {owned_out.get_seq_length()} != {crop_len}"
        )
        return 1
    owned_legacy = owned_out.to_legacy_cache()
    for layer_idx in range(n_layers):
        ref_k, ref_v = owned_ref[layer_idx]
        ok_, ov_ = owned_legacy[layer_idx][0], owned_legacy[layer_idx][1]
        if not torch.equal(ok_, ref_k[..., :crop_len, :]) or not torch.equal(
            ov_, ref_v[..., :crop_len, :]
        ):
            print(f"FAIL: owned crop layer {layer_idx} values != slice [:{crop_len}]")
            return 1
    # Equal-or-longer target must be a no-op (keep current length).
    owned_out2 = _crop_past_key_values_owned(owned_out, seq_len)
    if owned_out2 is not owned_out or int(owned_out2.get_seq_length()) != crop_len:
        print("FAIL: owned crop with target >= length must be a no-op")
        return 1

    print(
        "verify_kv_crop_no_mutation: OK "
        f"(layers={n_layers} seq={seq_len} crop={crop_len}; source untouched, no aliasing; "
        "owned in-place crop verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
