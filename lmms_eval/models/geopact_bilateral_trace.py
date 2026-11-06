"""Semantic trace collector for bilateral dialogue-KV visual pruning debug."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch


def trace_enabled() -> bool:
    return os.environ.get("GEOPACT_BILATERAL_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def trace_dir() -> str:
    return os.environ.get("GEOPACT_BILATERAL_TRACE_DIR", "").strip() or "logs/bilateral_trace"


def _trace_path() -> str:
    arm = os.environ.get("GEOPACT_BILATERAL_TRACE_ARM", "bilateral").strip() or "bilateral"
    episode = os.environ.get("GEOPACT_BILATERAL_TRACE_EPISODE", "unknown").strip() or "unknown"
    out_dir = trace_dir()
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{episode}_{arm}.jsonl")


def keep_hash(indices: List[int]) -> str:
    payload = json.dumps(sorted(int(x) for x in indices), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ids_hash(ids: List[int]) -> str:
    payload = json.dumps([int(x) for x in ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_pos_slice(pos: Optional[torch.Tensor], start: int, end: int) -> Optional[List[List[int]]]:
    if pos is None:
        return None
    if pos.dim() != 3:
        return None
    sl = pos[:, :, start:end].detach().cpu().tolist()
    return sl


def _tensor_pos_token(pos: Optional[torch.Tensor], index: int) -> Optional[List[int]]:
    if pos is None or pos.dim() != 3:
        return None
    if index < 0 or index >= int(pos.shape[-1]):
        return None
    return [int(pos[c, 0, index].item()) for c in range(3)]


def _grid_thw_list(grid: Any) -> Optional[List[List[int]]]:
    if grid is None:
        return None
    if isinstance(grid, torch.Tensor):
        return grid.detach().cpu().tolist()
    return list(grid)


def emit_checkpoint(checkpoint: str, payload: Dict[str, Any]) -> None:
    if not trace_enabled():
        return
    record: Dict[str, Any] = {
        "ts": round(time.time(), 3),
        "checkpoint": checkpoint,
        "arm": os.environ.get("GEOPACT_BILATERAL_TRACE_ARM", "bilateral"),
        "episode_id": os.environ.get("GEOPACT_BILATERAL_TRACE_EPISODE", ""),
        "step_index": int(os.environ.get("GEOPACT_BILATERAL_TRACE_STEP", "0") or 0),
    }
    record.update(payload)
    path = _trace_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def trace_prepare(
    model,
    *,
    compact_stats: Dict[str, Any],
    visual_keep_flat_indices: Optional[torch.Tensor],
    image_grid_thw: Any,
    image_paths: Optional[List[str]] = None,
) -> None:
    inner = getattr(model, "model", model)
    indices: List[int] = []
    if visual_keep_flat_indices is not None:
        indices = [int(x) for x in visual_keep_flat_indices.detach().cpu().tolist()]
    paths = image_paths if image_paths is not None else list(
        getattr(inner, "_geopact_image_paths", None) or []
    )
    sticky_keys = []
    episode_key = str(getattr(inner, "_geopact_episode_key", "") or "")
    try:
        from lmms_eval.models.geopact_vl_hook import _sticky_mask_key

        sticky_keys = [
            {"episode_key": episode_key, "image_path": p, "sticky_key": _sticky_mask_key(episode_key, p)}
            for p in paths
        ]
    except Exception:
        sticky_keys = [{"image_path": p} for p in paths]

    emit_checkpoint(
        "C0_prepare",
        {
            "episode_key": episode_key,
            "step_index": int(getattr(inner, "_geopact_step_index", 0) or 0),
            "image_paths": paths,
            "sticky_keys": sticky_keys,
            "visual_keep_flat_indices": indices,
            "keep_hash": keep_hash(indices) if indices else "",
            "orig_seq_len": int(compact_stats.get("orig_seq_len", 0) or 0),
            "compact_seq_len": int(compact_stats.get("compact_seq_len", 0) or 0),
            "n_visual_total": int(compact_stats.get("n_visual_total", 0) or 0),
            "n_visual_kept": int(compact_stats.get("n_visual_kept", 0) or 0),
            "image_grid_thw": _grid_thw_list(image_grid_thw),
            "compact_backend": str(compact_stats.get("compact_backend", "") or ""),
        },
    )


def trace_sticky_frame(
    *,
    episode_key: str,
    image_path: str,
    is_current: bool,
    local_keep: List[int],
    cache_hit: bool,
    frame_index: int,
) -> None:
    try:
        from lmms_eval.models.geopact_vl_hook import _sticky_mask_key

        sticky_key = _sticky_mask_key(episode_key, image_path)
    except Exception:
        sticky_key = (episode_key, image_path)
    emit_checkpoint(
        "C1_sticky",
        {
            "frame_index": int(frame_index),
            "episode_key": episode_key,
            "image_path": image_path,
            "sticky_key": list(sticky_key) if isinstance(sticky_key, tuple) else sticky_key,
            "is_current": bool(is_current),
            "local_keep": sorted(int(x) for x in local_keep),
            "keep_hash": keep_hash(local_keep) if local_keep else "",
            "cache_hit": bool(cache_hit),
        },
    )


def trace_suffix_inputs(
    *,
    full_image_paths: List[str],
    suffix_image_paths: List[str],
    suffix_no_gen_ids: List[int],
    suffix_len: int,
) -> None:
    emit_checkpoint(
        "C2_suffix_inputs",
        {
            "full_image_paths": full_image_paths,
            "suffix_image_paths": suffix_image_paths,
            "suffix_no_gen_ids": suffix_no_gen_ids,
            "suffix_no_gen_hash": ids_hash(suffix_no_gen_ids) if suffix_no_gen_ids else "",
            "suffix_no_gen_len": len(suffix_no_gen_ids),
            "suffix_len": int(suffix_len),
        },
    )


def trace_finalize(
    *,
    step_index: int,
    prompt_token_len: int,
    cached_ids_len: int,
    cached_mrope_len: int,
    boundary_pos: Optional[torch.Tensor],
    gen_tail_stripped: int,
    image_grid_thw: Any,
    position_ids_gold: Optional[torch.Tensor] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step_index": int(step_index),
        "prompt_token_len": int(prompt_token_len),
        "cached_ids_len": int(cached_ids_len),
        "cached_mrope_len": int(cached_mrope_len),
        "boundary_pos": _tensor_pos_token(boundary_pos, 0),
        "gen_tail_stripped": int(gen_tail_stripped),
        "image_grid_thw": _grid_thw_list(image_grid_thw),
    }
    if position_ids_gold is not None:
        payload["gold_full_pos_len"] = int(position_ids_gold.shape[-1])
        payload["gold_full_pos_last"] = _tensor_pos_token(
            position_ids_gold, int(position_ids_gold.shape[-1]) - 1
        )
    emit_checkpoint("C3_finalize", payload)


def trace_suffix_mrope(
    *,
    path_name: str,
    suffix_pos: torch.Tensor,
    suffix_len: int,
    boundary_pos: Optional[torch.Tensor],
    rope_deltas: Optional[torch.Tensor],
    suffix_no_gen_len: int = 0,
) -> None:
    n = int(suffix_pos.shape[-1])
    user_boundary_idx = max(0, int(suffix_no_gen_len) - 1) if suffix_no_gen_len > 0 else n - 1
    offset = None
    if boundary_pos is not None and n > 0:
        boundary = boundary_pos.to(device=suffix_pos.device, dtype=suffix_pos.dtype)
        first = suffix_pos[:, :, :1]
        offset = (boundary + 1 - first).detach().cpu().tolist()
    emit_checkpoint(
        "C4_suffix_mrope",
        {
            "path": path_name,
            "suffix_len": int(suffix_len),
            "suffix_pos_len": n,
            "suffix_pos_first": _tensor_pos_token(suffix_pos, 0),
            "suffix_pos_last": _tensor_pos_token(suffix_pos, n - 1),
            "suffix_pos_user_boundary": _tensor_pos_token(suffix_pos, user_boundary_idx),
            "boundary_pos": _tensor_pos_token(boundary_pos, 0) if boundary_pos is not None else None,
            "offset": offset,
            "rope_deltas": (
                rope_deltas.detach().cpu().tolist() if isinstance(rope_deltas, torch.Tensor) else None
            ),
            "suffix_pos_full": _tensor_pos_slice(suffix_pos, 0, n),
        },
    )


def trace_batch_prefill(
    *,
    past_len: int,
    prefill_len: int,
    kv_len: int,
    prefill_pos: Optional[torch.Tensor],
) -> None:
    last_idx = max(0, int(prefill_len) - 1) if prefill_len else 0
    emit_checkpoint(
        "C5_batch_prefill",
        {
            "past_len": int(past_len),
            "prefill_len": int(prefill_len),
            "kv_len": int(kv_len),
            "prefill_pos_last": _tensor_pos_token(prefill_pos, last_idx) if prefill_pos is not None else None,
        },
    )


def trace_deepstack(
    *,
    n_vis_mask: int,
    n_ds_rows: int,
    visual_keep_preview: Optional[List[int]] = None,
) -> None:
    preview = list(visual_keep_preview or [])[:8]
    emit_checkpoint(
        "C6_deepstack",
        {
            "n_vis_mask": int(n_vis_mask),
            "n_ds_rows": int(n_ds_rows),
            "visual_keep_preview": preview,
        },
    )


def trace_gold_full_positions(
    *,
    position_ids: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
) -> None:
    """Emit gold full-prompt position_ids for diff script (no-KV arm)."""
    payload: Dict[str, Any] = {
        "gold_full_pos_len": int(position_ids.shape[-1]),
        "gold_full_pos": _tensor_pos_slice(position_ids, 0, int(position_ids.shape[-1])),
    }
    if input_ids is not None:
        payload["gold_input_ids"] = [int(x) for x in input_ids[0].detach().cpu().tolist()]
    emit_checkpoint("C_gold_full", payload)


def _tensor_fingerprint(t: Optional[torch.Tensor], *, stride: int = 97) -> Optional[Dict[str, Any]]:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    flat = t.detach().float().cpu().flatten()
    n = int(flat.numel())
    if n == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "n": 0, "hash": ""}
    sample = flat[:: max(1, stride)]
    payload = json.dumps(
        {
            "mean": round(float(flat.mean().item()), 8),
            "std": round(float(flat.std(unbiased=False).item()), 8),
            "sample": [round(float(x), 6) for x in sample[:64].tolist()],
        },
        separators=(",", ":"),
    )
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "n": n,
        "mean": round(float(flat.mean().item()), 8),
        "std": round(float(flat.std(unbiased=False).item()), 8),
        "hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _position_ids_hash(pos: Optional[torch.Tensor]) -> str:
    if pos is None or not isinstance(pos, torch.Tensor):
        return ""
    sl = _tensor_pos_slice(pos, 0, int(pos.shape[-1]))
    if not sl:
        return ""
    payload = json.dumps(sl, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trace_decode_inputs(
    *,
    gen_payload: Dict[str, Any],
    logical_prompt_len: int,
    early_stop_prompt_len: int,
    decode_ref_len: int,
) -> None:
    pixel_values = gen_payload.get("pixel_values")
    position_ids = gen_payload.get("position_ids")
    image_grid_thw = gen_payload.get("image_grid_thw")
    past_kv = gen_payload.get("past_key_values")
    emit_checkpoint(
        "C7_decode_inputs",
        {
            "logical_prompt_len": int(logical_prompt_len),
            "early_stop_prompt_len": int(early_stop_prompt_len),
            "decode_ref_len": int(decode_ref_len),
            "has_past_kv": past_kv is not None,
            "past_kv_len": (
                int(past_kv.get_seq_length()) if past_kv is not None else 0
            ),
            "attention_mask_len": (
                int(gen_payload["attention_mask"].shape[-1])
                if isinstance(gen_payload.get("attention_mask"), torch.Tensor)
                else 0
            ),
            "pinned_cache_start": gen_payload.get("_geopact_pinned_cache_start"),
            "decode_only": bool(gen_payload.get("_geopact_decode_only_generate")),
            "image_grid_thw": _grid_thw_list(image_grid_thw),
            "pixel_fingerprint": _tensor_fingerprint(pixel_values),
            "position_ids_len": int(position_ids.shape[-1]) if isinstance(position_ids, torch.Tensor) else 0,
            "position_ids_hash": _position_ids_hash(position_ids),
            "input_ids_len": (
                int(gen_payload["input_ids"].shape[-1])
                if isinstance(gen_payload.get("input_ids"), torch.Tensor)
                else 0
            ),
            "has_inputs_embeds": gen_payload.get("inputs_embeds") is not None,
        },
    )


def trace_decode_out(
    *,
    generated_ids: Optional[torch.Tensor],
    gen_timing: Dict[str, Any],
    logical_prompt_len: int,
    early_stop_prompt_len: int,
) -> None:
    out_ids: List[int] = []
    if isinstance(generated_ids, torch.Tensor) and generated_ids.numel() > 0:
        seq = generated_ids[0] if generated_ids.dim() > 1 else generated_ids
        out_ids = [int(x) for x in seq.detach().cpu().tolist()]
    decode_out = max(0, len(out_ids) - int(early_stop_prompt_len)) if out_ids else 0
    if decode_out <= 0 and out_ids and logical_prompt_len > 0 and len(out_ids) > logical_prompt_len:
        decode_out = len(out_ids) - int(logical_prompt_len)
    emit_checkpoint(
        "C9_decode_out",
        {
            "generated_ids_hash": ids_hash(out_ids) if out_ids else "",
            "generated_len": len(out_ids),
            "decode_output_tokens": decode_out,
            "decode_ms": float(
                gen_timing.get(
                    "llm_decode_forward_ms",
                    gen_timing.get("decode_total_ms", gen_timing.get("decode_forward_ms", 0.0)),
                )
                or 0.0
            ),
            "early_stop_fired": bool(
                gen_timing.get("output_tokens_early_stop_cap", 0)
                and gen_timing.get("output_tokens_early_stop_cap", 0)
                < gen_timing.get("max_new_tokens", 4096)
            ),
            "timing_early_stop_prompt_len": float(gen_timing.get("early_stop_prompt_len", 0) or 0),
            "timing_logical_prompt_len": float(gen_timing.get("logical_prompt_len", 0) or 0),
        },
    )
