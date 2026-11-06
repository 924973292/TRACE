"""Block-preallocated KV cache (P3 of the ABC latency anatomy).

``DynamicCache`` re-``torch.cat``s the full K/V of every layer on every decoded
token: O(L^2) memory traffic per step (~92 ms GPU + ~8.4 GB allocation flow per
step measured on hist9). This drop-in subclass preallocates capacity in blocks
(default 1024 tokens) and appends with an in-place narrow ``copy_``, exposing a
narrow *view* of the logical prefix. The kernels consuming K/V see the exact
same values with ``stride(-1) == 1`` (FlashAttention-compatible), so results
are bit-identical to ``DynamicCache``.

Env gates (both read at call sites, default OFF):
  - ``GEOPACT_KV_BLOCK_CACHE=1`` enables the block cache.
  - ``GEOPACT_KV_BLOCK_SIZE``   block granularity in tokens (default 1024).

Compatibility notes (transformers 4.57 layered-cache API):
  - ``get_seq_length`` / ``get_mask_sizes`` report the logical length.
  - ``crop`` adjusts the logical length and re-narrows the views.
  - ``to_legacy_cache`` returns the narrow views; every GeoPACT consumer
    (``_legacy_cache_tensors`` -> clone/slice) copies before mutating.
  - Beam/batch ops and offload first materialize contiguous tensors and drop
    the block buffers (never used in this eval path; correctness fallback).
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer


def block_kv_cache_enabled() -> bool:
    raw = os.environ.get("GEOPACT_KV_BLOCK_CACHE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def block_kv_block_size() -> int:
    try:
        return max(64, int(os.environ.get("GEOPACT_KV_BLOCK_SIZE", "1024")))
    except ValueError:
        return 1024


class GeoPactBlockLayer(DynamicLayer):
    """DynamicLayer with block-preallocated storage and in-place appends."""

    def __init__(self):
        super().__init__()
        self.block_size = block_kv_block_size()
        self._buf_keys: Optional[torch.Tensor] = None
        self._buf_values: Optional[torch.Tensor] = None
        self._len = 0

    def _round_up(self, n: int) -> int:
        b = self.block_size
        return ((int(n) + b - 1) // b) * b

    def _alloc_like(self, ref: torch.Tensor, capacity: int) -> torch.Tensor:
        shape = list(ref.shape)
        shape[-2] = capacity
        return torch.empty(shape, dtype=ref.dtype, device=ref.device)

    def _drop_buffers(self) -> None:
        """Materialize logical prefix as owned tensors; buffers are discarded.

        Used before base-class ops that reassign ``self.keys`` wholesale
        (beam reorder, batch ops, offload). Never hit in single-sample eval.
        """
        if self._buf_keys is None:
            return
        self.keys = self.keys.contiguous()
        self.values = self.values.contiguous()
        self._buf_keys = None
        self._buf_values = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        n = int(key_states.shape[-2])
        if self._buf_keys is None:
            # Fresh layer, or buffers dropped by a materializing op: seed the
            # block buffer from whatever `self.keys` currently holds.
            base_len = 0
            if self.keys is not None and self.keys.numel() > 0:
                base_len = int(self.keys.shape[-2])
            cap = self._round_up(base_len + n)
            buf_k = self._alloc_like(key_states, cap)
            buf_v = self._alloc_like(value_states, cap)
            if base_len:
                buf_k[..., :base_len, :].copy_(self.keys)
                buf_v[..., :base_len, :].copy_(self.values)
            self._buf_keys, self._buf_values = buf_k, buf_v
            self._len = base_len
        elif self._len + n > int(self._buf_keys.shape[-2]):
            cap = self._round_up(self._len + n)
            buf_k = self._alloc_like(key_states, cap)
            buf_v = self._alloc_like(value_states, cap)
            buf_k[..., : self._len, :].copy_(self._buf_keys[..., : self._len, :])
            buf_v[..., : self._len, :].copy_(self._buf_values[..., : self._len, :])
            self._buf_keys, self._buf_values = buf_k, buf_v
        start, end = self._len, self._len + n
        self._buf_keys[..., start:end, :].copy_(key_states)
        self._buf_values[..., start:end, :].copy_(value_states)
        self._len = end
        self.keys = self._buf_keys[..., :end, :]
        self.values = self._buf_values[..., :end, :]
        return self.keys, self.values

    def get_seq_length(self) -> int:
        if self._buf_keys is not None:
            return self._len
        return super().get_seq_length()

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        if self._buf_keys is not None:
            self._len = int(max_length)
            self.keys = self._buf_keys[..., : self._len, :]
            self.values = self._buf_values[..., : self._len, :]
        else:
            super().crop(max_length)

    def reset(self) -> None:
        self._len = 0
        if self._buf_keys is not None:
            self.keys = self._buf_keys[..., :0, :]
            self.values = self._buf_values[..., :0, :]
        else:
            super().reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._drop_buffers()
        super().reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._drop_buffers()
        super().batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._drop_buffers()
        super().batch_select_indices(indices)

    def offload(self) -> None:
        self._drop_buffers()
        super().offload()


class GeoPactBlockKVCache(DynamicCache):
    """DynamicCache drop-in whose layers append in place into block buffers.

    ``config`` is accepted for signature compatibility but layer types are
    always ``GeoPactBlockLayer`` (Qwen3-VL text stack is full-attention only;
    no sliding-window layers exist in this eval path).
    """

    def __init__(
        self,
        ddp_cache_data: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
        config: Any = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        Cache.__init__(
            self,
            layer_class_to_replicate=GeoPactBlockLayer,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        if ddp_cache_data is not None:
            for layer_idx, (key_states, value_states) in enumerate(ddp_cache_data):
                self.update(key_states, value_states, layer_idx)


def make_kv_cache(config: Any = None) -> Any:
    """Factory: block cache when gated on, plain DynamicCache otherwise."""
    if block_kv_cache_enabled():
        return GeoPactBlockKVCache(config=config)
    return DynamicCache(config=config) if config is not None else DynamicCache()
