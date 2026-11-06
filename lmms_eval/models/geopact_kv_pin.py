"""Optional KV cache pinning for dialogue eval (BILATERAL_DECODE_PIN)."""
from __future__ import annotations

import os
from typing import Any, Tuple


def bilateral_decode_pin_enabled() -> bool:
    return os.environ.get("BILATERAL_DECODE_PIN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def slide_legacy_kv_tensors(
    legacy: Tuple[Tuple[Any, Any], ...],
    drop_len: int,
) -> Tuple[Tuple[Any, Any], ...]:
    """Slice KV tensors; use views when pinning to avoid clone alloc."""
    if bilateral_decode_pin_enabled():
        return tuple(
            (layer_k[..., drop_len:, :], layer_v[..., drop_len:, :])
            for layer_k, layer_v in legacy
        )
    return tuple(
        (layer_k[..., drop_len:, :].clone(), layer_v[..., drop_len:, :].clone())
        for layer_k, layer_v in legacy
    )
