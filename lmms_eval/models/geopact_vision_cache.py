"""
Per-frame **post-prune** compact cache for GeoPACT (source-compact path).

When a screenshot is the *current* frame we ViT-encode, run HES/sticky selection,
then store:
  - compact_embeds      (K kept LLM visual tokens, prompt sequence order)
  - compact_deepstack   (matching K rows per shallow layer, same order)
  - local_keep_seq      (frame-local flat indices, same order as above)

When the same PNG reappears as *history*, reuse the compact tensors directly —
skip ViT and skip re-running selection on that frame.

MRoPE / position_ids are **never cached**. Each step recomputes full-sequence
MRoPE from ``input_ids`` + **original** ``image_grid_thw`` (origpos path), then
subsets to kept pads. History labels change text before an image block, so
absolute position_ids shift — that is correct. Within-image (h, w) components
for a given ``local_keep_seq[i]`` stay tied to the same grid cell.

Enable: GEOPACT_VISION_EMBED_CACHE=1
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from lmms_eval.models.geopact_vl_hook import (
    _flat_image_token_ranges,
    _llm_tokens_per_image,
    _spatial_merge_size,
)
from lmms_eval.models.simple.qwen_vl_omnigui_common import _norm_geopact_path


def vision_embed_cache_enabled() -> bool:
    return os.environ.get("GEOPACT_VISION_EMBED_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def screenshot_only_cache_enabled() -> bool:
    return os.environ.get("GEOPACT_SCREENSHOT_ONLY_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_compact_cache_store(model) -> dict:
    store = getattr(model, "_geopact_compact_frame_cache", None)
    if store is None:
        store = {}
        model._geopact_compact_frame_cache = store
    return store


def clear_vision_embed_cache(model) -> None:
    store = getattr(model, "_geopact_compact_frame_cache", None)
    if store is not None:
        store.clear()
    if hasattr(model, "_geopact_compact_frame_cache_hits"):
        model._geopact_compact_frame_cache_hits = set()


def _per_image_pixel_lengths(image_grid_thw: torch.Tensor) -> List[int]:
    return [int(image_grid_thw[i].prod().item()) for i in range(int(image_grid_thw.shape[0]))]


def _split_pixel_values(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> Optional[List[torch.Tensor]]:
    lengths = _per_image_pixel_lengths(image_grid_thw)
    if not lengths or sum(lengths) != int(pixel_values.shape[0]):
        return None
    return list(torch.split(pixel_values, lengths, dim=0))


def _grid_row_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a.cpu(), b.cpu()))


def _n_llm_tokens_per_grid_row(grid_row: torch.Tensor) -> int:
    merge = _spatial_merge_size()
    t, h, w = (int(v) for v in grid_row.view(-1).tolist()[:3])
    return _llm_tokens_per_image(t, h, w, merge)


def _move_compact_entry(
    entry: Dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    return {
        "compact_embeds": entry["compact_embeds"].to(device=device, dtype=dtype),
        "compact_deepstack": [
            layer.to(device=device, dtype=dtype) for layer in entry["compact_deepstack"]
        ],
        "local_keep": entry["local_keep"],
        "orig_grid_row": entry["orig_grid_row"],
    }


def save_current_frame_compact_cache(
    model,
    image_path: str,
    *,
    compact_embeds: torch.Tensor,
    compact_deepstack: List[torch.Tensor],
    local_keep_seq: Tuple[int, ...],
    orig_grid_row: torch.Tensor,
) -> None:
    if not image_path:
        return
    if int(compact_embeds.shape[0]) != len(local_keep_seq):
        raise RuntimeError(
            "GeoPACT compact cache save: embed rows != local_keep_seq "
            f"{compact_embeds.shape[0]} vs {len(local_keep_seq)}"
        )
    store = _get_compact_cache_store(model)
    store[_norm_geopact_path(image_path)] = {
        "compact_embeds": compact_embeds.detach(),
        "compact_deepstack": [layer.detach() for layer in compact_deepstack],
        "local_keep_seq": tuple(int(x) for x in local_keep_seq),
        "local_keep": frozenset(int(x) for x in local_keep_seq),
        "orig_grid_row": orig_grid_row.detach().cpu(),
    }


def get_compact_frame_cache(model, image_path: str) -> Optional[Dict[str, Any]]:
    if not image_path:
        return None
    return _get_compact_cache_store(model).get(_norm_geopact_path(image_path))


def _local_keep_seq_from_entry(entry: Dict[str, Any]) -> Tuple[int, ...]:
    seq = entry.get("local_keep_seq")
    if seq:
        return tuple(int(x) for x in seq)
    return tuple(sorted(int(x) for x in entry.get("local_keep", ())))


def _fill_frame_buffer_from_compact(
    n_frame: int,
    entry: Dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Place compact tokens back into a full-frame flat buffer for scatter.

    Only ``local_keep_seq`` positions receive cached embeddings; dropped pads
    stay zero and are removed later by ``keep_mask`` in ``_compact_sequence``.
  MRoPE is applied separately via origpos subset — not from this buffer.
    """
    hidden = int(entry["compact_embeds"].shape[-1])
    buf = torch.zeros(n_frame, hidden, device=device, dtype=dtype)
    local_keep_seq = _local_keep_seq_from_entry(entry)
    compact = entry["compact_embeds"].to(device=device, dtype=dtype)
    if len(local_keep_seq) != int(compact.shape[0]):
        raise RuntimeError(
            f"GeoPACT compact cache mismatch: local_keep_seq={len(local_keep_seq)} "
            f"compact={compact.shape[0]}"
        )
    for row, li in enumerate(local_keep_seq):
        if li < 0 or li >= n_frame:
            raise RuntimeError(
                f"GeoPACT compact cache local index out of range: {li} n_frame={n_frame}"
            )
        buf[li] = compact[row]
    return buf


@torch.inference_mode()
def build_mixed_image_features(
    inner,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_paths: Optional[List[str]] = None,
) -> Tuple[
    torch.Tensor,
    List[List[torch.Tensor]],
    Dict[str, Any],
    Set[int],
]:
    """Build flat image_embeds, reusing post-prune compact cache for history frames.

    Returns (image_embeds, per_frame_deepstack, stats, compact_cache_hit_indices).
  per_frame_deepstack[frame][layer] has shape (n_frame_llm_tokens, hidden); history
  cache hits store only the kept K rows per layer.
    """
    # Per-prefill state only. Reset before every branch, including fallbacks, so
    # frame-relative hit indices can never leak across turns.
    inner._geopact_compact_frame_cache_hits = set()

    stats: Dict[str, Any] = {
        "vision_cache_enabled": False,
        "vision_cache_hits": 0,
        "vision_cache_misses": 0,
        "vision_encode_ms": 0.0,
        "vision_cache_frames": int(image_grid_thw.shape[0]) if image_grid_thw is not None else 0,
        "vision_cache_mode": "compact_post_prune",
    }
    compact_hits: Set[int] = set()

    def _post_vision_encode_ms() -> None:
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if ledger is not None:
            ledger.add_ms("vision_encode_ms", float(stats.get("vision_encode_ms", 0.0)))

    n_frames = int(image_grid_thw.shape[0])
    merge = _spatial_merge_size()
    n_img_total = sum(
        _llm_tokens_per_image(
            int(image_grid_thw[i][0]),
            int(image_grid_thw[i][1]),
            int(image_grid_thw[i][2]),
            merge,
        )
        for i in range(n_frames)
    )
    token_ranges = _flat_image_token_ranges(image_grid_thw, n_img_total)

    if not vision_embed_cache_enabled():
        t0 = time.perf_counter()
        image_embeds_list, deepstack_image_embeds = inner.get_image_features(
            pixel_values, image_grid_thw
        )
        stats["vision_encode_ms"] = (time.perf_counter() - t0) * 1000.0
        _post_vision_encode_ms()
        per_frame = []
        if deepstack_image_embeds:
            offset = 0
            for i in range(n_frames):
                start, end = token_ranges[i]
                n_f = end - start
                per_frame.append(
                    [layer[offset : offset + n_f] for layer in deepstack_image_embeds]
                )
                offset += n_f
        return torch.cat(image_embeds_list, dim=0), per_frame, stats, compact_hits

    paths = list(image_paths or [])
    pixel_slices = _split_pixel_values(pixel_values, image_grid_thw)
    if len(paths) != n_frames or pixel_slices is None or len(pixel_slices) != n_frames:
        t0 = time.perf_counter()
        image_embeds_list, deepstack_image_embeds = inner.get_image_features(
            pixel_values, image_grid_thw
        )
        stats["vision_cache_enabled"] = True
        stats["vision_cache_fallback"] = "path_or_pixel_split_mismatch"
        stats["vision_encode_ms"] = (time.perf_counter() - t0) * 1000.0
        _post_vision_encode_ms()
        per_frame = []
        if deepstack_image_embeds:
            offset = 0
            for i in range(n_frames):
                start, end = token_ranges[i]
                n_f = end - start
                per_frame.append(
                    [layer[offset : offset + n_f] for layer in deepstack_image_embeds]
                )
                offset += n_f
        return torch.cat(image_embeds_list, dim=0), per_frame, stats, compact_hits

    store = _get_compact_cache_store(inner)
    stats["vision_cache_enabled"] = True
    target_device = pixel_values.device
    target_dtype = inner.visual.dtype if hasattr(inner, "visual") else pixel_values.dtype

    frame_buffers: List[torch.Tensor] = []
    frame_deepstack: List[List[torch.Tensor]] = []
    encode_t0 = time.perf_counter()

    for frame_idx, img_path in enumerate(paths):
        start, end = token_ranges[frame_idx]
        n_frame = end - start
        grid_row = image_grid_thw[frame_idx : frame_idx + 1]
        is_current = frame_idx == n_frames - 1
        allow_current_cache = bool(getattr(inner, "_geopact_allow_current_frame_cache", False))
        prune_target = str(getattr(inner, "_geopact_prune_target", "current") or "current").lower()
        # history protocol: the active frame is kept full and must be re-encoded
        # from pixels every step (never served a stale/pruned compact entry).
        skip_current = is_current and (not allow_current_cache or prune_target == "history")
        cached = None if skip_current else get_compact_frame_cache(inner, img_path)

        if cached is not None and _grid_row_equal(cached["orig_grid_row"], grid_row):
            entry = _move_compact_entry(
                cached, device=target_device, dtype=target_dtype
            )
            frame_buffers.append(
                _fill_frame_buffer_from_compact(
                    n_frame, entry, device=target_device, dtype=target_dtype
                )
            )
            frame_deepstack.append(entry["compact_deepstack"])
            compact_hits.add(frame_idx)
            stats["vision_cache_hits"] += 1
            continue

        emb_list, ds = inner.get_image_features(pixel_slices[frame_idx], grid_row)
        frame_emb = emb_list[0]
        frame_ds = ds if ds is not None else []
        frame_buffers.append(frame_emb)
        frame_deepstack.append(frame_ds)
        stats["vision_cache_misses"] += 1

    active_keys = {_norm_geopact_path(p) for p in paths[:-1]}
    for stale_key in list(store.keys()):
        if stale_key not in active_keys:
            del store[stale_key]

    stats["vision_encode_ms"] = (time.perf_counter() - encode_t0) * 1000.0
    _post_vision_encode_ms()
    stats["vision_cache_size"] = len(store)

    image_embeds = torch.cat(frame_buffers, dim=0) if frame_buffers else torch.empty(0)
    inner._geopact_compact_frame_cache_hits = set(compact_hits)
    return image_embeds, frame_deepstack, stats, compact_hits


def write_current_frame_compact_cache(
    model,
    image_paths: List[str],
    token_ranges: List[Tuple[int, int]],
    seq_keep: torch.Tensor,
    image_embeds: torch.Tensor,
    per_frame_deepstack: List[List[torch.Tensor]],
    image_grid_thw: torch.Tensor,
) -> None:
    """Persist post-prune compact tensors for the current (last) frame."""
    if not vision_embed_cache_enabled() or not image_paths or not token_ranges:
        return
    if not per_frame_deepstack:
        return

    frame_idx = len(token_ranges) - 1
    img_path = image_paths[frame_idx] if frame_idx < len(image_paths) else ""
    if not img_path:
        return

    start, end = token_ranges[frame_idx]
    global_keep_set = {int(g) for g in seq_keep.tolist() if start <= int(g) < end}
    if not global_keep_set:
        return

    # Preserve prompt sequence order from hook (matches deepstack / origpos subset).
    ordered_global = [int(g) for g in seq_keep.tolist() if start <= int(g) < end]
    if not ordered_global:
        return
    local_keep_seq = tuple(int(g - start) for g in ordered_global)
    idx = torch.tensor(ordered_global, dtype=torch.long, device=image_embeds.device)
    compact_embeds = image_embeds.index_select(0, idx)
    frame_ds = per_frame_deepstack[frame_idx]
    local_idx = torch.tensor(local_keep_seq, dtype=torch.long, device=frame_ds[0].device)
    compact_deepstack = [layer.index_select(0, local_idx) for layer in frame_ds]
    save_current_frame_compact_cache(
        model,
        img_path,
        compact_embeds=compact_embeds,
        compact_deepstack=compact_deepstack,
        local_keep_seq=local_keep_seq,
        orig_grid_row=image_grid_thw[frame_idx : frame_idx + 1],
    )


def build_deepstack_compact_mixed(
    token_ranges: List[Tuple[int, int]],
    seq_keep: torch.Tensor,
    per_frame_deepstack: List[List[torch.Tensor]],
    compact_cache_hits: Set[int],
    model,
    image_paths: List[str],
) -> Optional[List[torch.Tensor]]:
    """Concat per-frame deepstack compact rows (cached hist + fresh select on current)."""
    if not per_frame_deepstack or not seq_keep.numel():
        return None

    seq_list = [int(x) for x in seq_keep.tolist()]
    n_layers = len(per_frame_deepstack[0])
    out: List[torch.Tensor] = []

    for layer_i in range(n_layers):
        parts: List[torch.Tensor] = []
        for frame_idx, (start, end) in enumerate(token_ranges):
            frame_global = [g for g in seq_list if start <= g < end]
            if frame_idx in compact_cache_hits:
                cached = get_compact_frame_cache(
                    model, image_paths[frame_idx] if frame_idx < len(image_paths) else ""
                )
                if cached is not None:
                    cached_local = _local_keep_seq_from_entry(cached)
                    frame_global = [start + int(idx) for idx in cached_local]
                    if not frame_global:
                        continue
                    layer_compact = cached["compact_deepstack"][layer_i].to(
                        device=per_frame_deepstack[frame_idx][layer_i].device,
                        dtype=per_frame_deepstack[frame_idx][layer_i].dtype,
                    )
                    if int(layer_compact.shape[0]) != len(frame_global):
                        raise RuntimeError(
                            "GeoPACT compact deepstack / seq_keep mismatch "
                            f"frame={frame_idx} layer={layer_i} "
                            f"cached={layer_compact.shape[0]} seq={len(frame_global)}"
                        )
                    parts.append(layer_compact)
                    continue
            if not frame_global:
                continue
            layer_full = per_frame_deepstack[frame_idx][layer_i]
            local_idx = torch.tensor(
                [g - start for g in frame_global],
                dtype=torch.long,
                device=layer_full.device,
            )
            parts.append(layer_full.index_select(0, local_idx))
        if parts:
            out.append(torch.cat(parts, dim=0))
    return out if out else None
