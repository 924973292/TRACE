"""
Episode-ordered session KV cache: reuse LLM KV for image prefix, rewrite tail each step.

Prompt layout (dialogue incremental KV):
  [System]
  [Screen 0] img0 ... [Screen k] img_k   <- prefix KV (monotonic append within episode)
  Goal / Step / Action History / Analyze  <- tail (re-prefilled every step)

Enable: GEOPACT_SESSION_KV=1  (requires episode-ordered eval; qwen3_vl_gui sorts by
episode_id + step_index). Dialogue mode encodes only in-prompt frames via
``prepare_dialogue_embed_inputs`` and does not require GEOPACT_VISION_EMBED_CACHE.

Dialogue mode (GEOPACT_SESSION_KV_MODE=dialogue): incremental teacher-forcing without
an extra commit forward — after each step, KV is cropped to the user-turn prompt
boundary; the next step prefills only ``GT assistant + current user`` on top.

MRoPE: full-sequence position_ids are computed each step; suffix generate uses
position_ids with stored past_key_values from prior user turns. When visual
pruning is active, suffix MRoPE uses unpruned virtual input_ids + keep_mask
subset (origpos) — never ``get_rope_index`` on compact ids with full grid.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger as eval_logger

_DIALOGUE_DECODE_DROP_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "mrope_source_input_ids",
    "mrope_keep_mask",
    "deepstack_image_embeds",
    "deepstack_visual_embeds",
    "visual_pos_masks",
    "image_mask",
    "geopact_compact_stats",
    "visual_keep_flat_indices",
    "force_rope_recompute",
)

# Keys safe to strip before contract resolution (do not touch rope / decode flags).
_GEOPACT_EARLY_STRIP_KEYS = (
    "geopact_compact_stats",
    "mrope_source_input_ids",
    "mrope_keep_mask",
    "deepstack_image_embeds",
    "image_mask",
    "visual_keep_flat_indices",
    "force_rope_recompute",
)

# Keys that must never be forwarded to ``model.generate``.
_GEOPACT_GENERATE_FORWARD_STRIP_KEYS = _GEOPACT_EARLY_STRIP_KEYS + (
    "rope_deltas",
    "_geopact_rope_deltas",
    "_geopact_decode_only_generate",
    "_geopact_dialogue_suffix_only",
    "_geopact_pinned_cache_start",
    "position_ids",
)

# Backward-compatible alias for callers that strip at generate boundary.
_GEOPACT_INTERNAL_GENERATE_KEYS = _GEOPACT_GENERATE_FORWARD_STRIP_KEYS


def strip_early_geopact_generate_keys(gen_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove bookkeeping keys that do not participate in pinned decode contract."""
    if not isinstance(gen_payload, dict):
        return gen_payload
    out = dict(gen_payload)
    for key in _GEOPACT_EARLY_STRIP_KEYS:
        out.pop(key, None)
    return out


def strip_forward_generate_keys(gen_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all GeoPACT keys before ``model.generate``."""
    if not isinstance(gen_payload, dict):
        return gen_payload
    out = dict(gen_payload)
    for key in _GEOPACT_GENERATE_FORWARD_STRIP_KEYS:
        out.pop(key, None)
    return out


def sanitize_dialogue_generate_payload(gen_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop vision/MRoPE metadata not needed once ``past_key_values`` is populated."""
    if not isinstance(gen_payload, dict):
        return gen_payload
    if gen_payload.get("past_key_values") is None:
        return gen_payload
    out = dict(gen_payload)
    for key in _DIALOGUE_DECODE_DROP_KEYS:
        out.pop(key, None)
    return out


def _uses_embed_prefill(gen_payload: Dict[str, Any]) -> bool:
    """True when ViT scatter already produced ``inputs_embeds`` (dialogue/ABC path)."""
    return isinstance(gen_payload, dict) and gen_payload.get("inputs_embeds") is not None


def _decode_query_len(gen_payload: Dict[str, Any]) -> int:
    ids = gen_payload.get("input_ids")
    if isinstance(ids, torch.Tensor) and ids.ndim >= 2:
        return max(int(ids.shape[-1]), 1)
    embeds = gen_payload.get("inputs_embeds")
    if isinstance(embeds, torch.Tensor) and embeds.ndim >= 2:
        return max(int(embeds.shape[-1]), 1)
    return 1


def _align_decode_attention_mask(gen_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure decode-only generate uses mask length = past_kv_len + query_len."""
    if not isinstance(gen_payload, dict):
        return gen_payload
    if not gen_payload.get("_geopact_decode_only_generate"):
        return gen_payload
    past_kv = gen_payload.get("past_key_values")
    if past_kv is None:
        return gen_payload
    try:
        kv_len = int(past_kv.get_seq_length())
    except Exception:
        return gen_payload
    query_len = _decode_query_len(gen_payload)
    want_len = kv_len + query_len
    mask = gen_payload.get("attention_mask")
    device = None
    dtype = torch.long
    if isinstance(mask, torch.Tensor):
        if int(mask.shape[-1]) == want_len:
            return gen_payload
        device = mask.device
        dtype = mask.dtype
    else:
        ids = gen_payload.get("input_ids")
        if isinstance(ids, torch.Tensor):
            device = ids.device
        elif isinstance(gen_payload.get("inputs_embeds"), torch.Tensor):
            device = gen_payload["inputs_embeds"].device
    if device is None:
        return gen_payload
    out = dict(gen_payload)
    out["attention_mask"] = torch.ones((1, want_len), device=device, dtype=dtype)
    return out


def build_dialogue_generate_payload(
    model,
    gen_payload: Dict[str, Any],
    *,
    logical_prompt_len: int,
    dialogue_suffix_only: bool = False,
    strip_internal: bool = True,
) -> Dict[str, Any]:
    """Single exit for ``model.generate`` kwargs (sanitize, anchor, prefill guard)."""
    from lmms_eval.models.geopact_decode_contract import (
        attach_pinned_decode_contract,
        resolve_rope_for_step,
        should_use_pinned_generate,
        validate_pinned_decode_contract,
    )

    inner = _inner_model(model)
    flow_mode = kv_flow_mode_label(model)
    if gen_payload.get("past_key_values") is not None:
        inner._geopact_prefill_payload = None

    decode_only = bool(gen_payload.get("_geopact_decode_only_generate"))
    out = dict(gen_payload)
    if strip_internal:
        out = strip_early_geopact_generate_keys(out)
    if _uses_embed_prefill(out):
        out.pop("pixel_values", None)
        out.pop("pixel_values_videos", None)
    if out.get("past_key_values") is not None:
        out = sanitize_dialogue_generate_payload(out)
    if out.get("inputs_embeds") is not None:
        out.pop("input_ids", None)

    if out.get("rope_deltas") is None and flow_mode == "none":
        for src in (
            getattr(inner, "_geopact_prefill_payload", None),
            getattr(inner, "_geopact_session_full_prefill_backup", None),
        ):
            if isinstance(src, dict) and src.get("rope_deltas") is not None:
                out["rope_deltas"] = src["rope_deltas"]
                break

    past_len = 0
    past_kv = out.get("past_key_values")
    if past_kv is not None:
        try:
            past_len = int(past_kv.get_seq_length())
        except Exception:
            past_len = 0

    out = resolve_rope_for_step(
        out,
        model_rope=getattr(inner, "rope_deltas", None),
        flow_mode=flow_mode,
        past_kv_len=past_len,
    )

    if should_use_pinned_generate(out):
        if out.get("_geopact_pinned_cache_start") is None:
            prompt_toks = 1 if decode_only else None
            out = attach_pinned_decode_contract(
                out,
                past_kv_len=past_len,
                prompt_token_count=prompt_toks,
                logical_prompt_len=int(logical_prompt_len) if past_len == 0 else None,
            )
        validate_pinned_decode_contract(out)
    else:
        out.pop("position_ids", None)
        out.pop("_geopact_pinned_cache_start", None)
        out.pop("_geopact_rope_deltas", None)

    if out.get("past_key_values") is not None and inner._geopact_prefill_payload is not None:
        raise RuntimeError(
            "GeoPACT generate: past_key_values set but _geopact_prefill_payload is stale"
        )

    if decode_only and out.get("past_key_values") is not None:
        out = _align_decode_attention_mask(out)

    if strip_internal:
        pinned = should_use_pinned_generate(out)
        for key in _GEOPACT_EARLY_STRIP_KEYS + (
            "rope_deltas",
            "_geopact_decode_only_generate",
            "_geopact_dialogue_suffix_only",
        ):
            out.pop(key, None)
        if not pinned:
            for key in (
                "position_ids",
                "_geopact_pinned_cache_start",
                "_geopact_rope_deltas",
            ):
                out.pop(key, None)
    return out

def session_kv_enabled() -> bool:
    return os.environ.get("GEOPACT_SESSION_KV", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def dialogue_kv_enabled() -> bool:
    mode = os.environ.get("GEOPACT_SESSION_KV_MODE", "").strip().lower()
    explicit = os.environ.get("GEOPACT_QWEN_DIALOGUE_CACHE", "").strip().lower()
    return session_kv_enabled() and (
        mode in {"dialogue", "teacher_forcing", "qwen_dialogue"}
        or explicit in {"1", "true", "yes", "on"}
    )


def kv_incremental_flow_enabled() -> bool:
    """Dialogue incremental KV (prefix retain + suffix prefill), with or without ABC."""
    return dialogue_kv_enabled() and session_kv_enabled()


def _visual_prune_active(model) -> bool:
    inner = _inner_model(model)
    ratio = getattr(inner, "_geopact_prune_ratio", None)
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "").strip().lower()
    if ratio is None:
        return False
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return False
    if not (0.0 < r < 0.999):
        return False
    return policy not in {"", "none", "off", "false"}


def kv_history_prune_rewrite_enabled(model) -> bool:
    """ABC history compression rewrite (rectify + merged prefill), not plain incremental KV."""
    inner = _inner_model(model)
    return (
        kv_incremental_flow_enabled()
        and _kv_history_rewrite_enabled()
        and _prune_target_history(inner)
        and _visual_prune_active(model)
    )


def kv_flow_mode_label(model) -> str:
    if kv_history_prune_rewrite_enabled(model):
        return "prune_rewrite"
    if kv_incremental_flow_enabled():
        return "incremental"
    return "none"


def _should_ensure_virtual_mrope_cache(model, state: Dict[str, Any]) -> bool:
    if state.get("_sliding_rewrite_pending") is not None:
        return True
    try:
        from lmms_eval.models.geopact_sspc import sspc_enabled

        if sspc_enabled():
            return True
    except Exception:
        pass
    return kv_history_prune_rewrite_enabled(model)


def _is_qwen25_carrier(m) -> bool:
    """True for the Qwen2.5-VL carrier (UI-TARS-1.5-7B): FCG owns visual/.model/state."""
    return str(getattr(getattr(m, "config", None), "model_type", "")) == "qwen2_5_vl"


def _inner_model(model):
    # Qwen2.5-VL: the FCG itself owns all _geopact state; ``model.model`` would
    # descend into the bare text LLM. Qwen3 keeps the existing descent verbatim.
    if _is_qwen25_carrier(model):
        return model
    return getattr(model, "model", model)


def _carrier_text_model(inner):
    """Text LLM under the carrier inner object.

    Qwen3 composite exposes ``.language_model`` (existing behavior, unchanged);
    the qwen2.5-VL FCG's text LLM is ``.model``; otherwise fall back to inner.
    """
    lm = getattr(inner, "language_model", None)
    if lm is not None:
        return lm
    if _is_qwen25_carrier(inner):
        return getattr(inner, "model", inner)
    return inner


def _clear_geopact_mrope_decode_state(inner) -> None:
    for attr in (
        "rope_deltas",
        "_geopact_generate_position_ids",
        "_geopact_generate_rope_deltas",
        "_geopact_generate_cache_start",
    ):
        if hasattr(inner, attr):
            setattr(inner, attr, None)


def clear_geopact_mrope_decode_state(model) -> None:
    """Reset MRoPE decode hook fields (baseline_full must not reuse stale rope)."""
    _clear_geopact_mrope_decode_state(_inner_model(model))


def clear_session_kv(model) -> None:
    inner = _inner_model(model)
    try:
        from lmms_eval.models.geopact_spatial_relay import clear_relay_state

        clear_relay_state(inner)
    except Exception:
        pass
    inner._geopact_session_kv_state = None
    inner._geopact_session_full_prefill_backup = None
    inner._geopact_dialogue_kv_state = None
    inner._geopact_dialogue_episode_image_paths = None
    inner._geopact_dialogue_suffix_no_gen_input_ids = None
    inner._geopact_dialogue_last_suffix_position_ids = None
    inner._geopact_pending_last_frame = None
    try:
        from lmms_eval.models.geopact_module_a_prefetch import discard_module_a_prefetch

        # Never let a prefetch staged for the previous episode's last frame be
        # consumed by the next episode's first rectify.
        discard_module_a_prefetch(inner)
    except Exception:
        pass
    try:
        from lmms_eval.models.geopact_rectify_prefetch import discard_rectify_prefetch

        discard_rectify_prefetch(inner)
    except Exception:
        pass
    if hasattr(inner, "_geopact_sticky_masks"):
        inner._geopact_sticky_masks = None
    try:
        from lmms_eval.models.geopact_nap import clear_nap_orders

        clear_nap_orders(inner)
    except Exception:
        pass
    if hasattr(inner, "_geopact_sticky_relevance_query"):
        inner._geopact_sticky_relevance_query = None
    if hasattr(inner, "_geopact_sticky_relevance_query_source"):
        inner._geopact_sticky_relevance_query_source = None
    if hasattr(inner, "_geopact_turn_instr_by_path"):
        inner._geopact_turn_instr_by_path = {}
    if hasattr(inner, "_geopact_turn_query_cache"):
        inner._geopact_turn_query_cache = {}
    _clear_geopact_mrope_decode_state(inner)


def backup_session_full_prefill(model, *, metadata_only: bool = False) -> None:
    """Clone prefill payload before generate consumes it (session cache update).

    ``metadata_only=True`` keeps only finalize MRoPE fields (suffix step>0).
    Avoids cloning large ``inputs_embeds`` / deepstack tensors each step.
    """
    if not session_kv_enabled():
        return
    inner = _inner_model(model)
    prefill = getattr(inner, "_geopact_prefill_payload", None)
    if prefill is None:
        inner._geopact_session_full_prefill_backup = None
        return
    if metadata_only:
        backup: Dict[str, Any] = {}
        mrope_src = prefill.get("mrope_source_input_ids")
        if mrope_src is not None and hasattr(mrope_src, "detach"):
            backup["mrope_source_input_ids"] = mrope_src.detach().clone()
        else:
            backup["mrope_source_input_ids"] = mrope_src
        compact_stats = prefill.get("geopact_compact_stats")
        backup["geopact_compact_stats"] = (
            dict(compact_stats) if isinstance(compact_stats, dict) else compact_stats
        )
        inner._geopact_session_full_prefill_backup = backup
        return
    backup = {}
    for key, val in prefill.items():
        if isinstance(val, torch.Tensor):
            backup[key] = val.clone()
        elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
            backup[key] = [t.clone() for t in val]
        else:
            backup[key] = val
    inner._geopact_session_full_prefill_backup = backup


def session_episode_key(doc: Dict) -> str:
    if not isinstance(doc, dict):
        return "unknown-unknown"
    subset = str(doc.get("subset", "unknown") or "unknown")
    episode_id = (
        doc.get("episode_id")
        or doc.get("episode")
        or doc.get("trajectory_id")
        or doc.get("task_id")
        or "unknown"
    )
    episode_id = str(episode_id or "unknown")
    return f"{subset}-{episode_id}"


def session_step_index(doc: Dict) -> int:
    return int(doc.get("step_index", doc.get("step_id", 0)) or 0)


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _image_token_id(inner) -> int:
    return int(getattr(inner.config, "image_token_id", -1) or -1)


def _count_visual_tokens(input_ids: torch.Tensor, start: int, end: int, image_token_id: int) -> int:
    if image_token_id < 0 or end <= start:
        return 0
    return int((input_ids[0, start:end] == image_token_id).sum().item())


def _slice_position_ids(position_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        return position_ids[:, :, start:end].contiguous()
    return position_ids[..., start:end].contiguous()


def _slice_prefill_payload(
    prefill: Dict[str, Any],
    *,
    start: int,
    end: int,
    image_token_id: int,
    visual_row_offset: int = 0,
    visual_row_end: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    input_ids = prefill["input_ids"]
    for key in ("input_ids", "inputs_embeds", "attention_mask"):
        val = prefill.get(key)
        if val is not None:
            out[key] = val[:, start:end].contiguous()
    pos = prefill.get("position_ids")
    if pos is not None:
        out["position_ids"] = _slice_position_ids(pos, start, end)
    if prefill.get("image_grid_thw") is not None:
        out["image_grid_thw"] = prefill["image_grid_thw"]
    image_mask = prefill.get("image_mask")
    if image_mask is not None:
        out["image_mask"] = image_mask[:, start:end].contiguous()
    if prefill.get("rope_deltas") is not None:
        out["rope_deltas"] = prefill["rope_deltas"]
    if prefill.get("geopact_compact_stats") is not None:
        out["geopact_compact_stats"] = dict(prefill["geopact_compact_stats"])
    # Unpruned virtual MRoPE metadata — never slice by compact [start:end].
    for key in ("mrope_source_input_ids", "mrope_keep_mask"):
        if prefill.get(key) is not None:
            out[key] = prefill[key]
    ds = prefill.get("deepstack_image_embeds")
    if ds:
        vis_start = visual_row_offset
        vis_end = visual_row_end
        if vis_end is None:
            vis_end = visual_row_offset + _count_visual_tokens(input_ids, start, end, image_token_id)
        n_vis_slice = int(vis_end - vis_start)
        sliced_layers = []
        for layer_i, layer in enumerate(ds):
            n_avail = int(layer.shape[0])
            if n_avail < vis_end:
                raise RuntimeError(
                    "GeoPACT dialogue KV: deepstack slice out of range "
                    f"layer={layer_i} rows={n_avail} need_end={vis_end}"
                )
            sliced = layer[vis_start:vis_end].contiguous()
            if int(sliced.shape[0]) != n_vis_slice:
                raise RuntimeError(
                    "GeoPACT dialogue KV: deepstack slice row mismatch "
                    f"layer={layer_i} rows={sliced.shape[0]} expected={n_vis_slice}"
                )
            sliced_layers.append(sliced)
        out["deepstack_image_embeds"] = sliced_layers
    out["force_rope_recompute"] = bool(prefill.get("force_rope_recompute"))
    return out


def _slice_active_inputs(active_inputs: Dict[str, Any], *, start: int) -> Dict[str, Any]:
    out = {}
    for key in ("input_ids", "inputs_embeds", "attention_mask"):
        val = active_inputs.get(key)
        if val is not None:
            out[key] = val[:, start:].contiguous()
    pos = active_inputs.get("position_ids")
    if pos is not None:
        out["position_ids"] = _slice_position_ids(pos, start, pos.shape[-1])
    for key in ("image_grid_thw", "rope_deltas"):
        if active_inputs.get(key) is not None:
            out[key] = active_inputs[key]
    return {k: v for k, v in out.items() if v is not None}


def _get_dialogue_state(inner) -> Optional[Dict[str, Any]]:
    return getattr(inner, "_geopact_dialogue_kv_state", None)


def _set_dialogue_state(inner, state: Optional[Dict[str, Any]]) -> None:
    inner._geopact_dialogue_kv_state = state


def _kv_compressed_frame_key(img_path: str, frame_seq: Any) -> Optional[Tuple[str, int]]:
    path = str(img_path or "")
    if not path:
        return None
    try:
        seq = int(frame_seq)
    except Exception:
        return None
    if seq < 0:
        return None
    return (path, seq)


def _kv_compressed_key_from_capture(cap: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    return _kv_compressed_frame_key(
        str(cap.get("img_path") or ""),
        cap.get("frame_seq", cap.get("frame_index", -1)),
    )


def _normalise_kv_compressed_keys(raw: Any) -> set[Tuple[str, int]]:
    out: set[Tuple[str, int]] = set()
    for item in raw or ():
        key = None
        if isinstance(item, (tuple, list)) and len(item) == 2:
            key = _kv_compressed_frame_key(str(item[0] or ""), item[1])
        elif isinstance(item, str):
            # Legacy path-only markers are intentionally not promoted: they were
            # ambiguous when a dataset reuses the same screenshot path.
            key = None
        if key is not None:
            out.add(key)
    return out


def _shift_kv_compressed_keys_after_slide(raw: Any, *, drop_frames: int = 1) -> set[Tuple[str, int]]:
    shifted: set[Tuple[str, int]] = set()
    drop = int(max(drop_frames, 0))
    for path, seq in _normalise_kv_compressed_keys(raw):
        if seq < drop:
            continue
        shifted.add((path, seq - drop))
    return shifted


def _ids_list(input_ids: torch.Tensor) -> List[int]:
    return [int(x) for x in input_ids[0].detach().cpu().tolist()]


def _prefix_matches(full_ids: List[int], prefix_ids: List[int]) -> bool:
    return len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids


def _make_full_attention(
    suffix_attention: Optional[torch.Tensor],
    *,
    past_len: int,
    suffix_len: int,
) -> Optional[torch.Tensor]:
    if suffix_attention is None:
        return None
    full_attn = torch.ones(
        (1, past_len + suffix_len),
        dtype=suffix_attention.dtype,
        device=suffix_attention.device,
    )
    full_attn[:, past_len:] = suffix_attention
    return full_attn


def _cumulative_dialogue_image_grid(
    state: Dict[str, Any],
    suffix_inputs: Dict[str, Any],
) -> torch.Tensor:
    """Merge stored episode grids with the current suffix frame grid."""
    prev_grid = state.get("image_grid_thw")
    suffix_grid = suffix_inputs.get("image_grid_thw")
    if prev_grid is None and suffix_grid is None:
        raise RuntimeError("GeoPACT dialogue KV: missing image_grid_thw for MRoPE recompute")
    if prev_grid is None:
        return suffix_grid
    if suffix_grid is None:
        return prev_grid
    device = suffix_grid.device
    dtype = suffix_grid.dtype
    return torch.cat([prev_grid.to(device=device, dtype=dtype), suffix_grid], dim=0)


def _mrope_source_list(tensor_or_list) -> list[int]:
    if tensor_or_list is None:
        return []
    if isinstance(tensor_or_list, list):
        return [int(x) for x in tensor_or_list]
    return [int(x) for x in tensor_or_list[0].detach().cpu().tolist()]


def _compact_stats_pruned(stats: Optional[Dict[str, Any]]) -> bool:
    if not stats:
        return False
    orig = int(stats.get("orig_seq_len", 0) or 0)
    compact = int(stats.get("compact_seq_len", 0) or 0)
    return orig > 0 and compact > 0 and orig > compact


def _bool_mask_len(mask: Any) -> int:
    if isinstance(mask, torch.Tensor):
        return int(mask.shape[0])
    if isinstance(mask, list):
        return len(mask)
    raise RuntimeError(f"GeoPACT dialogue KV: unsupported keep_mask type {type(mask).__name__}")


def _bool_mask_sum(mask: Any) -> int:
    if isinstance(mask, torch.Tensor):
        return int(mask.to(dtype=torch.bool).sum().item())
    if isinstance(mask, list):
        return int(sum(1 for x in mask if x))
    raise RuntimeError(f"GeoPACT dialogue KV: unsupported keep_mask type {type(mask).__name__}")


def _validate_suffix_prune_mrope_metadata(
    suffix_prefill: Dict[str, Any],
    *,
    suffix_len: int,
    where: str,
) -> Dict[str, Any]:
    """Fail loud when pruned suffix MRoPE metadata is inconsistent (no fixed lengths)."""
    stats = suffix_prefill.get("geopact_compact_stats") or {}
    if not _compact_stats_pruned(stats):
        return {}
    orig = int(stats.get("orig_seq_len", 0) or 0)
    compact = int(stats.get("compact_seq_len", 0) or 0)
    keep_mask = suffix_prefill.get("mrope_keep_mask")
    virt_len = len(_mrope_source_list(suffix_prefill.get("mrope_source_input_ids")))
    audit = {
        "orig_seq_len": orig,
        "compact_seq_len": compact,
        "mrope_source_len": virt_len,
        "suffix_len": int(suffix_len),
        "n_visual_total": int(stats.get("n_visual_total", 0) or 0),
        "n_visual_kept": int(stats.get("n_visual_kept", 0) or 0),
    }
    if orig <= 0 or compact <= 0:
        raise RuntimeError(f"GeoPACT {where}: invalid compact stats {audit}")
    if virt_len > 0 and virt_len != orig:
        raise RuntimeError(
            f"GeoPACT {where}: mrope_source_len={virt_len} != orig_seq_len={orig} audit={audit}"
        )
    if keep_mask is None:
        raise RuntimeError(f"GeoPACT {where}: pruned suffix missing mrope_keep_mask audit={audit}")
    mask_len = _bool_mask_len(keep_mask)
    audit["keep_mask_len"] = mask_len
    if mask_len != orig:
        raise RuntimeError(
            f"GeoPACT {where}: keep_mask_len={mask_len} != orig_seq_len={orig} audit={audit}"
        )
    n_kept = _bool_mask_sum(keep_mask)
    from lmms_eval.models.geopact_coord_injection.lifecycle import coord_spliced_tokens_for_audit

    coord_spliced = coord_spliced_tokens_for_audit(stats)
    audit["keep_mask_kept"] = n_kept
    audit["coord_spliced_tokens"] = coord_spliced
    audit["coord_ephemeral_tokens"] = int(stats.get("coord_ephemeral_tokens", 0) or 0)
    expected_compact = int(n_kept) + coord_spliced
    if expected_compact != compact:
        raise RuntimeError(
            f"GeoPACT {where}: keep_mask_kept={n_kept} + coord_spliced={coord_spliced} "
            f"!= compact_seq_len={compact} audit={audit}"
        )
    if int(suffix_len) != compact:
        raise RuntimeError(
            f"GeoPACT {where}: suffix_len={suffix_len} != compact_seq_len={compact} audit={audit}"
        )
    return audit


def _align_suffix_mrope_to_cached_boundary(
    suffix_pos: torch.Tensor,
    boundary_pos: torch.Tensor,
) -> torch.Tensor:
    """Shift suffix MRoPE (all channels) to continue after the cached prefix boundary."""
    if boundary_pos is None:
        return suffix_pos
    boundary = boundary_pos.to(device=suffix_pos.device, dtype=suffix_pos.dtype)
    if boundary.dim() != 3 or suffix_pos.dim() != 3:
        raise RuntimeError(
            "GeoPACT dialogue KV: boundary/suffix position_ids must be 3D MRoPE "
            f"boundary={None if boundary_pos is None else tuple(boundary_pos.shape)} "
            f"suffix={tuple(suffix_pos.shape)}"
        )
    if int(suffix_pos.shape[-1]) < 1:
        return suffix_pos
    offset = boundary + 1 - suffix_pos[:, :, :1]
    return suffix_pos + offset


def _compact_to_virtual_index_map(
    cached_ids: list,
    virtual_ids: list,
    image_token_id: int,
) -> Optional[list]:
    """Map each compact token index to its virtual MRoPE source index."""
    if not cached_ids or not virtual_ids:
        return None
    mapping: list = []
    vi = 0
    for ci, tok in enumerate(cached_ids):
        while (
            vi < len(virtual_ids)
            and int(virtual_ids[vi]) == int(image_token_id)
            and int(tok) != int(image_token_id)
        ):
            vi += 1
        if vi >= len(virtual_ids):
            return None
        if int(virtual_ids[vi]) != int(tok):
            return None
        mapping.append(vi)
        vi += 1
    return mapping


def _recompute_cached_boundary_position_ids(
    inner,
    state: Dict[str, Any],
    *,
    compress_suffix_pos: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """MRoPE position_ids at the cached prefix user boundary (last committed token)."""
    cached_ids = list(state.get("cached_input_ids") or [])
    prompt_len = int(state.get("prompt_token_len") or len(cached_ids))
    if prompt_len <= 0 or not cached_ids:
        return None
    boundary_idx = min(prompt_len - 1, len(cached_ids) - 1)

    if compress_suffix_pos is not None:
        sp = compress_suffix_pos
        if isinstance(sp, torch.Tensor) and sp.dim() == 3 and int(sp.shape[-1]) >= 1:
            end = int(sp.shape[-1])
            return _slice_position_ids(sp, end - 1, end).detach().cpu().clone()

    cached_mrope = list(state.get("cached_mrope_input_ids") or [])
    full_grid = state.get("image_grid_thw")
    if full_grid is None:
        return None

    from lmms_eval.models.geopact_vl_hook import _precompute_full_mrope

    try:
        embed_fn = inner.get_input_embeddings()
        device = embed_fn.weight.device
    except Exception:
        device = torch.device("cpu")

    prune_on_prefix = bool(cached_mrope) and len(cached_mrope) > len(cached_ids)
    compact_ids = cached_ids[:prompt_len]
    pos_offset = _mrope_pos_offset(state)

    if not prune_on_prefix:
        compact_tensor = torch.tensor([compact_ids], dtype=torch.long, device=device)
        compact_attn = torch.ones((1, len(compact_ids)), dtype=torch.long, device=device)
        grid = full_grid.to(device=device)
        full_pos, _ = _precompute_full_mrope(inner, compact_tensor, grid, compact_attn)
        if pos_offset:
            full_pos = full_pos + pos_offset
        return _slice_position_ids(full_pos, boundary_idx, boundary_idx + 1).detach().cpu().clone()

    image_token_id = _image_token_id(inner)
    virt_ids = list(cached_mrope)
    mapping = _compact_to_virtual_index_map(compact_ids, virt_ids, image_token_id)
    if mapping is None or boundary_idx >= len(mapping):
        return None
    virt_tensor = torch.tensor([virt_ids], dtype=torch.long, device=device)
    virt_attn = torch.ones((1, len(virt_ids)), dtype=torch.long, device=device)
    grid = full_grid.to(device=device)
    full_pos, _ = _precompute_full_mrope(inner, virt_tensor, grid, virt_attn)
    if pos_offset:
        full_pos = full_pos + pos_offset
    virt_idx = int(mapping[boundary_idx])
    return _slice_position_ids(full_pos, virt_idx, virt_idx + 1).detach().cpu().clone()


def _refresh_dialogue_boundary_in_state(
    inner,
    state: Dict[str, Any],
    *,
    compress_suffix_pos: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Recompute and store cached_boundary_position_ids after prefix mutation."""
    out = dict(state)
    boundary = _recompute_cached_boundary_position_ids(
        inner,
        out,
        compress_suffix_pos=compress_suffix_pos,
    )
    if boundary is not None and int(boundary.shape[-1]) == 1:
        out["cached_boundary_position_ids"] = boundary
    else:
        out.pop("cached_boundary_position_ids", None)
    return out


def _mrope_pos_offset(state: Dict[str, Any]) -> int:
    """Cumulative MRoPE position offset after KV slides (original-frame anchor).

    Eviction contract: after a KV slide the surviving keys stay rotated at
    their original absolute positions. Any recompute over the trimmed ids
    anchors at 0, so it must be shifted by this offset back into the original
    coordinate frame before being used next to retained KV.
    """
    try:
        return int(state.get("mrope_virt_pos_offset") or 0)
    except Exception:
        return 0


def _virtual_mrope_cache_key(state: Dict[str, Any]) -> str:
    """Fingerprint virtual MRoPE ids + grid + pos offset for cache invalidation."""
    virtual_ids = list(state.get("cached_mrope_input_ids") or [])
    grid = state.get("image_grid_thw")
    if not virtual_ids or grid is None:
        return ""
    try:
        grid_flat = [int(x) for x in grid.detach().cpu().flatten().tolist()]
    except Exception:
        return ""
    payload = (
        ",".join(str(int(x)) for x in virtual_ids)
        + "|"
        + ",".join(str(x) for x in grid_flat)
        + "|off"
        + str(_mrope_pos_offset(state))
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def _invalidate_virtual_mrope_cache(state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    out.pop("cached_virtual_mrope_key", None)
    out.pop("cached_virtual_mrope_pos", None)
    return out


def _store_cached_virtual_full_mrope(
    state: Dict[str, Any],
    full_pos: torch.Tensor,
) -> Dict[str, Any]:
    out = dict(state)
    key = _virtual_mrope_cache_key(state)
    if not key:
        return out
    out["cached_virtual_mrope_key"] = key
    out["cached_virtual_mrope_pos"] = full_pos.detach().cpu().clone()
    return out


def _get_cached_virtual_full_mrope(
    state: Dict[str, Any],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    expected = _virtual_mrope_cache_key(state)
    if not expected or state.get("cached_virtual_mrope_key") != expected:
        return None
    pos = state.get("cached_virtual_mrope_pos")
    if pos is None or not isinstance(pos, torch.Tensor):
        return None
    return pos.to(device=device)


def _ensure_virtual_mrope_cache(
    inner,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Populate ``cached_virtual_mrope_pos`` when virtual ids + grid are known."""
    key = _virtual_mrope_cache_key(state)
    if not key:
        return state
    if state.get("cached_virtual_mrope_key") == key and state.get("cached_virtual_mrope_pos") is not None:
        return state
    virtual_ids = list(state.get("cached_mrope_input_ids") or [])
    full_grid = state.get("image_grid_thw")
    if not virtual_ids or full_grid is None:
        return state
    from lmms_eval.models.geopact_vl_hook import _precompute_full_mrope

    try:
        embed_fn = inner.get_input_embeddings()
        device = embed_fn.weight.device
    except Exception:
        device = torch.device("cpu")
    virt_tensor = torch.tensor([virtual_ids], dtype=torch.long, device=device)
    virt_attn = torch.ones((1, len(virtual_ids)), dtype=torch.long, device=device)
    full_pos, _ = _precompute_full_mrope(
        inner,
        virt_tensor,
        full_grid.to(device=device),
        virt_attn,
    )
    pos_offset = _mrope_pos_offset(state)
    if pos_offset:
        full_pos = full_pos + pos_offset
    return _store_cached_virtual_full_mrope(state, full_pos)


def _concat_sliding_merged_mrope(
    rewrite_pos: torch.Tensor,
    suffix_pos: torch.Tensor,
) -> torch.Tensor:
    """Align suffix MRoPE to rewrite tail, then concatenate (sliding merged prefill)."""
    boundary = rewrite_pos[:, :, -1:]
    suffix_aligned = _align_suffix_mrope_to_cached_boundary(suffix_pos, boundary)
    if not torch.equal(suffix_aligned[:, :, :1], boundary + 1):
        raise RuntimeError(
            "GeoPACT sliding rewrite: merged MRoPE boundary mismatch "
            f"rewrite_tail={boundary.tolist()} suffix_head={suffix_aligned[:, :, :1].tolist()}"
        )
    return torch.cat([rewrite_pos, suffix_aligned], dim=-1)


def _align_sliding_suffix_mrope_fast(
    suffix_prefill: Dict[str, Any],
    rewrite_pos: torch.Tensor,
    *,
    suffix_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Sliding rewrite: suffix-local MRoPE + boundary align only (no virtual recompute)."""
    mrope_t0 = time.perf_counter()
    suffix_local = suffix_prefill.get("position_ids")
    if suffix_local is None:
        raise RuntimeError(
            "GeoPACT sliding rewrite: suffix prefill missing position_ids "
            "(prepare_dialogue_embed_inputs must run for current frame)"
        )
    if bool(suffix_prefill.get("force_rope_recompute")):
        raise RuntimeError(
            "GeoPACT sliding rewrite: force_rope_recompute forbidden on sliding suffix"
        )
    if int(suffix_local.shape[-1]) != int(suffix_len):
        raise RuntimeError(
            "GeoPACT sliding rewrite: suffix position_ids length mismatch "
            f"got={int(suffix_local.shape[-1])} expected={int(suffix_len)}"
        )
    boundary = rewrite_pos[:, :, -1:].to(device=device)
    suffix_pos = _align_suffix_mrope_to_cached_boundary(
        suffix_local.to(device=device),
        boundary,
    )
    if not torch.equal(suffix_pos[:, :, :1], boundary + 1):
        raise RuntimeError(
            "GeoPACT sliding rewrite: suffix boundary align failed "
            f"boundary={boundary.tolist()} head={suffix_pos[:, :, :1].tolist()}"
        )
    rope_deltas = _rope_deltas_for_position_ids(suffix_pos, int(suffix_len))
    mrope_ms = round((time.perf_counter() - mrope_t0) * 1000.0, 2)
    return suffix_pos, rope_deltas, mrope_ms


def _rope_deltas_for_position_ids(
    position_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Recompute rope_deltas from compact MRoPE position_ids (data-driven seq_len)."""
    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    compact_len = torch.tensor(
        [[int(seq_len)]],
        device=position_ids.device,
        dtype=torch.long,
    )
    return (max_position_ids + 1 - compact_len).contiguous().to(dtype=torch.long)


def _strip_virtual_mrope_gen_tail(
    mrope_list: list[int],
    *,
    compact_full_len: int,
    compact_no_gen_len: int,
    where: str,
) -> tuple[list[int], int]:
    """Drop chat-template generation tail from virtual MRoPE source ids."""
    if not mrope_list or compact_full_len <= 0:
        return list(mrope_list or []), 0
    gen_tail_len = int(compact_full_len) - int(compact_no_gen_len)
    if gen_tail_len <= 0:
        return list(mrope_list), 0
    if gen_tail_len >= len(mrope_list):
        raise RuntimeError(
            f"GeoPACT {where}: generation tail consumes MRoPE source "
            f"tail={gen_tail_len} source_len={len(mrope_list)} "
            f"compact_full={compact_full_len} compact_no_gen={compact_no_gen_len}"
        )
    return list(mrope_list[:-gen_tail_len]), gen_tail_len


def _recompute_dialogue_suffix_mrope(
    model,
    inner,
    state: Dict[str, Any],
    suffix_inputs: Dict[str, Any],
    *,
    past_len: int,
    suffix_len: int,
    suffix_prefill: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Suffix MRoPE for dialogue KV step>0.

    Prefer suffix-local origpos from ``prepare_dialogue_*`` (single-frame grid +
    keep_mask subset already applied), then shift all MRoPE channels to continue
    after the cached prefix boundary. Recomputing on prefix_virtual+suffix_virtual
    with a cumulative multi-image grid disagrees with prefix KV baked at step0.
    """
    from lmms_eval.models.geopact_vl_hook import _precompute_full_mrope, _subset_origpos_mrope_cached

    cached_ids = list(state.get("cached_input_ids") or [])
    suffix_ids = [int(x) for x in suffix_inputs["input_ids"][0].detach().cpu().tolist()]
    suffix_prefill = suffix_prefill or {}

    cached_mrope = list(state.get("cached_mrope_input_ids") or [])
    prefix_mrope = cached_mrope if cached_mrope else cached_ids
    suffix_mrope = _mrope_source_list(suffix_prefill.get("mrope_source_input_ids"))
    if not suffix_mrope:
        suffix_mrope = suffix_ids

    suffix_keep_mask = suffix_prefill.get("mrope_keep_mask")

    prune_on_prefix = bool(cached_mrope) and len(cached_mrope) > len(cached_ids)
    prune_on_suffix = _compact_stats_pruned(suffix_prefill.get("geopact_compact_stats"))
    use_virtual_origpos = prune_on_prefix or prune_on_suffix
    mrope_audit: Dict[str, Any] = {}
    if prune_on_suffix:
        mrope_audit = _validate_suffix_prune_mrope_metadata(
            suffix_prefill,
            suffix_len=suffix_len,
            where="dialogue_suffix_mrope",
        )

    device = suffix_inputs["inputs_embeds"].device
    mrope_t0 = time.perf_counter()

    suffix_local_pos = suffix_prefill.get("position_ids")
    local_path_ok = (
        int(past_len) > 0
        and suffix_local_pos is not None
        and not bool(suffix_prefill.get("force_rope_recompute"))
        and int(suffix_local_pos.shape[-1]) == int(suffix_len)
    )
    if prune_on_suffix and int(past_len) > 0 and not local_path_ok:
        missing: list[str] = []
        if suffix_local_pos is None:
            missing.append("position_ids")
        if bool(suffix_prefill.get("force_rope_recompute")):
            missing.append("force_rope_recompute=True")
        if suffix_local_pos is not None and int(suffix_local_pos.shape[-1]) != int(suffix_len):
            missing.append(
                f"position_ids_len={int(suffix_local_pos.shape[-1])}!=suffix_len={int(suffix_len)}"
            )
        raise RuntimeError(
            "GeoPACT dialogue KV: pruned suffix step>0 requires local MRoPE path; "
            f"missing_or_invalid={missing} audit={mrope_audit}"
        )
    if local_path_ok:
        suffix_pos = suffix_local_pos.to(device=device)
        boundary = state.get("cached_boundary_position_ids")
        if boundary is None and prune_on_prefix and int(past_len) > 0:
            refreshed = _recompute_cached_boundary_position_ids(inner, state)
            if refreshed is not None:
                boundary = refreshed
        if (
            boundary is not None
            and isinstance(boundary, torch.Tensor)
            and int(boundary.shape[-1]) == 1
        ):
            suffix_pos = _align_suffix_mrope_to_cached_boundary(suffix_pos, boundary)
        elif prune_on_prefix and int(past_len) > 0:
            raise RuntimeError(
                "GeoPACT dialogue KV: pruned-prefix suffix step>0 requires "
                "cached_boundary_position_ids or successful boundary recompute "
                f"audit={mrope_audit}"
            )
        prev_rope = state.get("rope_deltas")
        if prev_rope is not None and int(past_len) > 0:
            rope_deltas = prev_rope.to(device=device)
            if isinstance(rope_deltas, torch.Tensor) and rope_deltas.dim() == 1:
                rope_deltas = rope_deltas.unsqueeze(0)
        else:
            rope_deltas = _rope_deltas_for_position_ids(suffix_pos, suffix_len)
        mrope_ms = round((time.perf_counter() - mrope_t0) * 1000.0, 2)
        if int(suffix_pos.shape[-1]) != int(suffix_len):
            raise RuntimeError(
                "GeoPACT dialogue KV: suffix-local MRoPE length mismatch "
                f"suffix_pos={suffix_pos.shape[-1]} expected={suffix_len} audit={mrope_audit}"
            )
        _emit_suffix_mrope_trace(
            inner, suffix_pos, rope_deltas, suffix_len, "local", state
        )
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if ledger is not None:
            ledger.add_ms("dialogue_suffix_mrope_ms", mrope_ms)
            ledger.set_audit(mrope_path="local")
        return suffix_pos, rope_deltas, mrope_ms

    path_name = "virtual" if use_virtual_origpos else "compact"
    if state.get("_sliding_rewrite_active"):
        raise RuntimeError(
            "GeoPACT sliding rewrite: virtual MRoPE fallback forbidden; "
            "suffix must use local position_ids + boundary align "
            f"audit={mrope_audit}"
        )
    if use_virtual_origpos and prune_on_prefix and int(past_len) > 0:
        boundary_probe = state.get("cached_boundary_position_ids")
        if boundary_probe is None and _recompute_cached_boundary_position_ids(inner, state) is None:
            raise RuntimeError(
                "GeoPACT dialogue KV: prune_on_prefix forbids virtual cumulative "
                "MRoPE fallback without cached_boundary_position_ids "
                f"audit={mrope_audit}"
            )
    if use_virtual_origpos:
        full_list = prefix_mrope + suffix_mrope
        full_len = len(full_list)
        virtual_suffix_len = len(suffix_mrope)
        expected_compact = int(past_len) + int(suffix_len)
        if full_len < expected_compact:
            raise RuntimeError(
                "GeoPACT dialogue KV: unpruned MRoPE virtual length shorter than compact "
                f"virtual_full={full_len} past_len+suffix={expected_compact} "
                f"prefix_virtual={len(prefix_mrope)} suffix_virtual={virtual_suffix_len} "
                f"audit={mrope_audit}"
            )
        full_ids = torch.tensor([full_list], device=device, dtype=torch.long)
        full_grid = _cumulative_dialogue_image_grid(state, suffix_inputs)
        full_attn = torch.ones((1, full_len), device=device, dtype=torch.long)
        position_ids, rope_deltas = _precompute_full_mrope(
            inner,
            full_ids,
            full_grid.to(device=device),
            full_attn,
        )
        _pos_offset = _mrope_pos_offset(state)
        if _pos_offset:
            position_ids = position_ids + _pos_offset
            rope_deltas = rope_deltas + _pos_offset
        suffix_start = len(prefix_mrope)
        suffix_pos_full = _slice_position_ids(
            position_ids,
            suffix_start,
            suffix_start + virtual_suffix_len,
        )
        if prune_on_suffix:
            if suffix_keep_mask is None:
                raise RuntimeError(
                    "GeoPACT dialogue KV: pruned suffix requires mrope_keep_mask "
                    f"audit={mrope_audit}"
                )
            mask_len = _bool_mask_len(suffix_keep_mask)
            pos_len = int(suffix_pos_full.shape[-1])
            if mask_len != pos_len:
                raise RuntimeError(
                    "GeoPACT dialogue KV: suffix mrope_keep_mask length mismatch "
                    f"mask={mask_len} pos={pos_len} audit={mrope_audit}"
                )
            suffix_pos, rope_deltas = _subset_origpos_mrope_cached(
                model,
                suffix_pos_full,
                suffix_keep_mask.to(suffix_pos_full.device),
                suffix_inputs["input_ids"],
                suffix_inputs.get("attention_mask"),
            )
        else:
            suffix_pos = suffix_pos_full
            if int(suffix_pos.shape[-1]) != int(suffix_len):
                suffix_pos = _slice_position_ids(suffix_pos, 0, suffix_len)
        boundary = state.get("cached_boundary_position_ids")
        if (
            boundary is not None
            and int(past_len) > 0
            and isinstance(boundary, torch.Tensor)
            and int(boundary.shape[-1]) == 1
        ):
            suffix_pos = _align_suffix_mrope_to_cached_boundary(suffix_pos, boundary)
        prev_rope = state.get("rope_deltas")
        if prev_rope is not None and int(past_len) > 0:
            rope_deltas = prev_rope.to(device=suffix_pos.device)
            if isinstance(rope_deltas, torch.Tensor) and rope_deltas.dim() == 1:
                rope_deltas = rope_deltas.unsqueeze(0)
        elif prune_on_suffix:
            rope_deltas = _rope_deltas_for_position_ids(suffix_pos, suffix_len)
    else:
        full_ids_list = cached_ids + suffix_ids
        full_len = len(full_ids_list)
        expected_len = int(past_len) + int(suffix_len)
        if full_len != expected_len:
            raise RuntimeError(
                "GeoPACT dialogue KV: token length mismatch for MRoPE "
                f"full={full_len} past_len+suffix={expected_len} "
                f"past_len={past_len} suffix_len={suffix_len} audit={mrope_audit}"
            )
        full_ids = torch.tensor([full_ids_list], device=device, dtype=torch.long)
        full_grid = _cumulative_dialogue_image_grid(state, suffix_inputs)
        full_attn = torch.ones((1, full_len), device=device, dtype=torch.long)
        position_ids, rope_deltas = _precompute_full_mrope(
            inner,
            full_ids,
            full_grid.to(device=device),
            full_attn,
        )
        _pos_offset = _mrope_pos_offset(state)
        if _pos_offset:
            position_ids = position_ids + _pos_offset
            rope_deltas = rope_deltas + _pos_offset
        suffix_pos = _slice_position_ids(position_ids, int(past_len), int(past_len) + int(suffix_len))
        prev_rope = state.get("rope_deltas")
        if prev_rope is not None and int(past_len) > 0:
            rope_deltas = prev_rope.to(device=suffix_pos.device)
            if isinstance(rope_deltas, torch.Tensor) and rope_deltas.dim() == 1:
                rope_deltas = rope_deltas.unsqueeze(0)

    mrope_ms = round((time.perf_counter() - mrope_t0) * 1000.0, 2)
    if suffix_pos is None or int(suffix_pos.shape[-1]) != int(suffix_len):
        raise RuntimeError(
            "GeoPACT dialogue KV: suffix MRoPE slice failed "
            f"suffix_pos={None if suffix_pos is None else suffix_pos.shape[-1]} "
            f"expected={suffix_len} virtual_origpos={use_virtual_origpos} audit={mrope_audit}"
        )
    _emit_suffix_mrope_trace(inner, suffix_pos, rope_deltas, suffix_len, path_name, state)
    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    if ledger is not None:
        ledger.add_ms("dialogue_suffix_mrope_ms", mrope_ms)
        ledger.set_audit(mrope_path=path_name)
    return suffix_pos, rope_deltas, mrope_ms


def _emit_suffix_mrope_trace(
    inner,
    suffix_pos: torch.Tensor,
    rope_deltas: Optional[torch.Tensor],
    suffix_len: int,
    path_name: str,
    state: Dict[str, Any],
) -> None:
    try:
        from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_suffix_mrope

        if not trace_enabled():
            return
        suffix_no_gen = list(getattr(inner, "_geopact_dialogue_suffix_no_gen_input_ids", None) or [])
        boundary = state.get("cached_boundary_position_ids")
        if isinstance(boundary, torch.Tensor):
            boundary_t = boundary
        else:
            boundary_t = None
        trace_suffix_mrope(
            path_name=path_name,
            suffix_pos=suffix_pos,
            suffix_len=int(suffix_len),
            boundary_pos=boundary_t,
            rope_deltas=rope_deltas,
            suffix_no_gen_len=len(suffix_no_gen),
        )
    except Exception:
        pass


def _slice_suffix_for_past(
    active_inputs: Dict[str, Any],
    prefill: Dict[str, Any],
    *,
    start: int,
    past_len: int,
    image_token_id: int,
) -> Dict[str, Any]:
    full_len = int(active_inputs["input_ids"].shape[1])
    prefix_vis = _count_visual_tokens(active_inputs["input_ids"], 0, start, image_token_id)
    suffix_prefill = _slice_prefill_payload(
        prefill,
        start=start,
        end=full_len,
        image_token_id=image_token_id,
        visual_row_offset=prefix_vis,
    )
    suffix = _slice_active_inputs(active_inputs, start=start)
    suffix_len = int(suffix["inputs_embeds"].shape[1])
    suffix["past_key_values"] = None
    suffix["cache_position"] = torch.arange(
        past_len,
        past_len + suffix_len,
        device=suffix["inputs_embeds"].device,
        dtype=torch.long,
    )
    suffix["attention_mask"] = _make_full_attention(
        suffix.get("attention_mask"),
        past_len=past_len,
        suffix_len=suffix_len,
    )
    suffix["position_ids"] = suffix_prefill.get("position_ids", suffix.get("position_ids"))
    return suffix, suffix_prefill


def _suffix_prefill_visual_meta(
    suffix: Dict[str, Any],
    suffix_prefill: Optional[Dict[str, Any]],
    *,
    prefill_len: int,
    image_token_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Visual mask + deepstack rows for suffix[:prefill_len] (generation token excluded)."""
    if suffix_prefill is None or prefill_len <= 0:
        return None, None
    image_mask = suffix_prefill.get("image_mask")
    visual_pos_masks = None
    if image_mask is not None:
        visual_pos_masks = image_mask[:, :prefill_len].contiguous()
        if visual_pos_masks.dim() == 3:
            visual_pos_masks = visual_pos_masks[..., 0]
    input_ids = suffix.get("input_ids")
    deepstack_visual_embeds = None
    ds = suffix_prefill.get("deepstack_image_embeds")
    if ds and input_ids is not None and image_token_id >= 0:
        n_vis = _count_visual_tokens(input_ids, 0, prefill_len, image_token_id)
        if n_vis > 0:
            deepstack_visual_embeds = []
            for layer_i, layer in enumerate(ds):
                n_rows = int(layer.shape[0])
                if n_rows != n_vis:
                    raise RuntimeError(
                        "GeoPACT dialogue KV: deepstack row count mismatch for suffix prefill "
                        f"layer={layer_i} rows={n_rows} visual_tokens={n_vis} "
                        "(expected mask-synced compact deepstack, not silent truncation)"
                    )
                deepstack_visual_embeds.append(layer.contiguous())
    if visual_pos_masks is not None and deepstack_visual_embeds is not None:
        n_mask = int(visual_pos_masks.sum().item())
        n_ds = int(deepstack_visual_embeds[0].shape[0])
        if n_mask != n_ds:
            raise RuntimeError(
                "GeoPACT dialogue KV: visual_pos_masks vs deepstack mismatch "
                f"mask={n_mask} deepstack={n_ds}"
            )
        try:
            from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_deepstack

            if trace_enabled():
                preview = None
                vk = (suffix_prefill or {}).get("visual_keep_flat_indices")
                if vk is not None:
                    preview = [int(x) for x in vk.detach().cpu().tolist()]
                trace_deepstack(n_vis_mask=n_mask, n_ds_rows=n_ds, visual_keep_preview=preview)
        except Exception:
            pass
    return visual_pos_masks, deepstack_visual_embeds


@torch.inference_mode()
def _prefill_suffix_before_generate(
    model,
    suffix: Dict[str, Any],
    *,
    past_kv: Any,
    past_len: int,
    suffix_prefill: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Extend cached KV by full suffix in batch; generate bootstraps last token only."""
    suffix_len = int(suffix["inputs_embeds"].shape[1])
    stats: Dict[str, Any] = {"dialogue_kv_batch_prefill_ms": 0.0, "dialogue_kv_batch_prefill_tokens": 0}
    if suffix_len <= 1:
        suffix["_geopact_pinned_cache_start"] = int(past_len)
        suffix["_geopact_decode_only_generate"] = True
        return past_kv, stats

    prefill_len = suffix_len
    device = suffix["inputs_embeds"].device
    position_ids = suffix.get("position_ids")
    rope_deltas = suffix.get("rope_deltas")

    inner = _inner_model(model)
    lm = _carrier_text_model(inner)
    image_token_id = _image_token_id(inner)

    prefill_embeds = suffix["inputs_embeds"][:, :prefill_len].contiguous()
    prefill_cache_pos = torch.arange(
        past_len,
        past_len + prefill_len,
        device=device,
        dtype=torch.long,
    )
    prefill_attn = torch.ones(
        (1, past_len + prefill_len),
        dtype=suffix["attention_mask"].dtype,
        device=device,
    )
    prefill_pos = (
        _slice_position_ids(position_ids, 0, prefill_len) if position_ids is not None else None
    )
    visual_pos_masks, deepstack_visual_embeds = _suffix_prefill_visual_meta(
        suffix,
        suffix_prefill,
        prefill_len=prefill_len,
        image_token_id=image_token_id,
    )

    if rope_deltas is not None:
        inner.rope_deltas = rope_deltas.to(device=prefill_embeds.device)

    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    batch_t0 = time.perf_counter()
    if ledger is not None:
        # Switch out of dialogue_kv_prepare_ms for the LM forward itself.
        ledger.start_bucket("dialogue_kv_batch_prefill_ms")
    lm_kwargs: Dict[str, Any] = {
        "inputs_embeds": prefill_embeds,
        "past_key_values": past_kv,
        "attention_mask": prefill_attn,
        "cache_position": prefill_cache_pos,
        "use_cache": True,
        "return_dict": True,
    }
    if prefill_pos is not None:
        lm_kwargs["position_ids"] = prefill_pos
    # Qwen2.5-VL text LLM does not accept deepstack kwargs; qwen3 path unchanged.
    carrier_has_deepstack = not _is_qwen25_carrier(inner)
    if carrier_has_deepstack and visual_pos_masks is not None:
        lm_kwargs["visual_pos_masks"] = visual_pos_masks
    if carrier_has_deepstack and deepstack_visual_embeds is not None:
        lm_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

    outputs = lm(**lm_kwargs)
    if ledger is not None:
        ledger.stop_bucket("dialogue_kv_batch_prefill_ms")
    batch_prefill_ms = round((time.perf_counter() - batch_t0) * 1000.0, 2)
    running_kv = outputs.past_key_values
    new_len = int(running_kv.get_seq_length()) if running_kv is not None else 0
    expected_len = int(past_len) + int(prefill_len)
    if new_len != expected_len:
        raise RuntimeError(
            "GeoPACT dialogue KV: batch suffix prefill length mismatch "
            f"got={new_len} expected={expected_len}"
        )

    lm_cfg = getattr(lm, "config", getattr(inner, "config", None))
    bootstrap_kv_len = int(past_len) + int(suffix_len) - 1
    crop_t0 = time.perf_counter()
    # In-place: running_kv is the (already-mutated) dialogue state cache that is
    # replaced wholesale at finalize; nothing else consumes the pre-crop tail.
    running_kv = _crop_past_key_values_owned(running_kv, bootstrap_kv_len, config=lm_cfg)
    if running_kv is None:
        raise RuntimeError(
            "GeoPACT dialogue KV: failed to crop KV for decode bootstrap "
            f"target_len={bootstrap_kv_len}"
        )
    if ledger is not None:
        ledger.add_ms(
            "kv_bootstrap_crop_ms", (time.perf_counter() - crop_t0) * 1000.0
        )
        # Suffix slicing / attention-mask rebuild below is prepare-stage work.
        ledger.start_bucket("dialogue_kv_prepare_ms")

    try:
        from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_batch_prefill

        if trace_enabled():
            trace_batch_prefill(
                past_len=int(past_len),
                prefill_len=int(prefill_len),
                kv_len=int(new_len),
                prefill_pos=prefill_pos,
            )
    except Exception:
        pass

    last = suffix_len - 1
    if suffix.get("input_ids") is not None:
        suffix["input_ids"] = suffix["input_ids"][:, last:].contiguous()
        suffix.pop("inputs_embeds", None)
    elif suffix.get("inputs_embeds") is not None:
        suffix["inputs_embeds"] = suffix["inputs_embeds"][:, last:].contiguous()
    kv_len = int(running_kv.get_seq_length()) if running_kv is not None else int(past_len + suffix_len - 1)
    suffix["attention_mask"] = torch.ones(
        (1, kv_len + 1),
        dtype=suffix["attention_mask"].dtype,
        device=device,
    )
    suffix["cache_position"] = torch.arange(
        past_len + last,
        past_len + suffix_len,
        device=device,
        dtype=torch.long,
    )
    if position_ids is not None:
        suffix["position_ids"] = _slice_position_ids(position_ids, last, suffix_len)
    suffix["past_key_values"] = running_kv
    suffix["_geopact_pinned_cache_start"] = int(past_len + last)
    suffix["_geopact_decode_only_generate"] = True
    stats["dialogue_kv_batch_prefill_ms"] = batch_prefill_ms
    stats["dialogue_kv_batch_prefill_tokens"] = int(prefill_len)
    # Forward wall already counted via the dialogue_kv_batch_prefill bucket
    # switch around the LM call above.
    return running_kv, stats


def _legacy_cache_tensors(past_key_values: Any) -> tuple:
    """Return per-layer (k, v) tensors from any HF cache representation."""
    if past_key_values is None:
        return tuple()
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return tuple(
            (layer[0], layer[1])
            for layer in legacy
            if isinstance(layer, (list, tuple)) and len(layer) >= 2
        )
    if isinstance(past_key_values, (list, tuple)):
        return tuple(
            (layer[0], layer[1])
            for layer in past_key_values
            if isinstance(layer, (list, tuple)) and len(layer) >= 2
        )
    return tuple()


def _cache_from_legacy_tensors(
    legacy_tensors: tuple,
    *,
    max_length: Optional[int] = None,
    config: Any = None,
) -> Any:
    """Build a fresh DynamicCache from cloned legacy tensors (never aliases live cache)."""
    if not legacy_tensors:
        return None
    end = int(max_length) if max_length is not None else int(legacy_tensors[0][0].shape[-2])
    cloned_legacy = tuple(
        (k[..., :end, :].clone(), v[..., :end, :].clone()) for k, v in legacy_tensors
    )
    try:
        from lmms_eval.models.geopact_block_kv_cache import make_kv_cache

        cache = make_kv_cache(config=config)
        for layer_idx, (key_states, value_states) in enumerate(cloned_legacy):
            cache.update(key_states, value_states, layer_idx)
        return cache
    except Exception:
        return cloned_legacy


def _clone_past_key_values(past_key_values: Any, *, config: Any = None) -> Any:
    """Deep-copy KV so capture/finalize never mutates the live generate cache."""
    legacy = _legacy_cache_tensors(past_key_values)
    if not legacy:
        raise RuntimeError("GeoPACT dialogue KV: unable to clone past_key_values")
    return _cache_from_legacy_tensors(legacy, config=config)


def _crop_past_key_values(past_key_values: Any, max_length: int, *, config: Any = None) -> Any:
    """Return a NEW KV cache truncated to ``max_length`` tokens (prompt boundary).

    Never mutates or aliases ``past_key_values``. In particular it must NOT call
    ``DynamicCache.crop`` — on transformers>=4.5x that crops the cache IN PLACE
    (returning None), which corrupted the live generate cache when the dialogue
    KV capture hook fired mid-generation (assistant-header KV was deleted for
    the whole decode; root cause of the baseline_full vs baseline_kv drift).
    """
    if past_key_values is None or max_length <= 0:
        return None
    legacy = _legacy_cache_tensors(past_key_values)
    if not legacy:
        raise RuntimeError("GeoPACT dialogue KV: unable to crop past_key_values")
    return _cache_from_legacy_tensors(legacy, max_length=int(max_length), config=config)


def _crop_past_key_values_owned(
    past_key_values: Any,
    max_length: int,
    *,
    config: Any = None,
) -> Any:
    """Truncate a cache the caller exclusively owns, in place (zero-copy views).

    Use ONLY when no other live consumer references ``past_key_values``:
      - decode bootstrap right after the batch suffix prefill (the old dialogue
        state cache is discarded at finalize either way), and
      - post-generate finalize on ``direct_pkv`` / a hook-captured clone.
    The full deep-copy (``_crop_past_key_values``) remains mandatory for the
    mid-generate capture hook, where the source is the live generate cache.

    Values are bit-identical to the clone path (same tensors, sliced views).
    """
    if past_key_values is None or max_length <= 0:
        return None
    crop = getattr(past_key_values, "crop", None)
    if crop is None:
        return _crop_past_key_values(past_key_values, max_length, config=config)
    crop(int(max_length))
    return past_key_values


def slide_kv_on_roll_enabled() -> bool:
    """When true, drop evicted history tokens from KV instead of full prefill."""
    raw = os.environ.get("GEOPACT_KV_SLIDE_ON_ROLL", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _slide_past_key_values(
    past_key_values: Any,
    drop_len: int,
    *,
    config: Any = None,
) -> Any:
    """Drop ``drop_len`` tokens from the left of KV (rolling-window slide)."""
    from lmms_eval.models.geopact_kv_pin import slide_legacy_kv_tensors

    if past_key_values is None or drop_len <= 0:
        return past_key_values
    legacy = _legacy_cache_tensors(past_key_values)
    if not legacy:
        raise RuntimeError("GeoPACT dialogue KV: unable to slide past_key_values")
    seq_len = int(legacy[0][0].shape[-2])
    if drop_len >= seq_len:
        raise RuntimeError(
            f"GeoPACT dialogue KV: slide drop_len={drop_len} >= seq_len={seq_len}"
        )
    slid = slide_legacy_kv_tensors(legacy, int(drop_len))
    return _cache_from_legacy_tensors(slid, config=config)


def _text_rope_inv_freq(inner) -> torch.Tensor:
    """Inverse frequencies of the text LM rotary embedding (head_dim/2)."""
    lm = getattr(inner, "language_model", None)
    if lm is None and _is_qwen25_carrier(inner):
        lm = getattr(inner, "model", None)
    if lm is None:
        lm = getattr(getattr(inner, "model", None), "language_model", None)
    for obj in (lm, getattr(lm, "model", None) if lm is not None else None, inner, getattr(inner, "model", None)):
        if obj is None:
            continue
        rot = getattr(obj, "rotary_emb", None)
        if rot is None:
            rot = getattr(getattr(obj, "model", None), "rotary_emb", None)
        inv = getattr(rot, "inv_freq", None) if rot is not None else None
        if inv is not None:
            return inv
    raise RuntimeError("GeoPACT dialogue KV: text rotary inv_freq unavailable")


def _rotate_post_rope_keys(
    keys: torch.Tensor,
    delta: int,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Shift post-RoPE keys by a constant position delta.

    Valid because RoPE rotations compose (``R(p+d) = R(d) R(p)`` per 2D pair)
    and the anchored-eviction contract proves every MRoPE channel shifts by
    the same delta, so the mrope section layout collapses to a plain rotation
    at each frequency index. Uses unscaled cos/sin: any attention_scaling is
    already baked into the stored keys as a scalar factor and commutes.
    """
    if int(delta) == 0:
        return keys
    orig_dtype = keys.dtype
    k32 = keys.to(torch.float32)
    inv = inv_freq.to(device=keys.device, dtype=torch.float32).flatten()
    if int(inv.numel()) * 2 != int(k32.shape[-1]):
        raise RuntimeError(
            "GeoPACT dialogue KV: inv_freq/head_dim mismatch "
            f"inv={int(inv.numel())} head_dim={int(k32.shape[-1])}"
        )
    ang = float(delta) * inv
    cos = torch.cos(ang).repeat(2).view(1, 1, 1, -1)
    sin = torch.sin(ang).repeat(2).view(1, 1, 1, -1)
    half = int(k32.shape[-1]) // 2
    rot = torch.cat([-k32[..., half:], k32[..., :half]], dim=-1)
    return (k32 * cos + rot * sin).to(orig_dtype)


def _anchored_evict_past_key_values(
    past_key_values: Any,
    *,
    anchor_len: int,
    drop_len: int,
    rotated_sys_keys: Optional[List[torch.Tensor]],
    config: Any = None,
) -> Any:
    """Middle-segment KV eviction: keep ``[0, anchor)`` (keys replaced by the
    provided rotated system keys), drop ``[anchor, anchor+drop)``, keep tail.

    ``torch.cat`` already materialises fresh tensors, so the resulting cache
    never aliases the live one.
    """
    legacy = _legacy_cache_tensors(past_key_values)
    if not legacy:
        raise RuntimeError("GeoPACT dialogue KV: unable to evict past_key_values")
    anchor = int(anchor_len)
    drop = int(drop_len)
    seq_len = int(legacy[0][0].shape[-2])
    if drop <= 0 or anchor < 0 or anchor + drop >= seq_len:
        raise RuntimeError(
            "GeoPACT dialogue KV: bad eviction bounds "
            f"anchor={anchor} drop={drop} seq_len={seq_len}"
        )
    if anchor and (rotated_sys_keys is None or len(rotated_sys_keys) != len(legacy)):
        raise RuntimeError("GeoPACT dialogue KV: rotated system keys missing")

    evicted = []
    for layer_idx, (k, v) in enumerate(legacy):
        if anchor:
            sys_k = rotated_sys_keys[layer_idx]
            if int(sys_k.shape[-2]) != anchor:
                raise RuntimeError(
                    "GeoPACT dialogue KV: rotated system key length mismatch "
                    f"layer={layer_idx} got={int(sys_k.shape[-2])} want={anchor}"
                )
            k_new = torch.cat([sys_k.to(dtype=k.dtype, device=k.device), k[..., anchor + drop :, :]], dim=-2)
            v_new = torch.cat([v[..., :anchor, :], v[..., anchor + drop :, :]], dim=-2)
        else:
            k_new = k[..., drop:, :].clone()
            v_new = v[..., drop:, :].clone()
        evicted.append((k_new, v_new))

    try:
        from lmms_eval.models.geopact_block_kv_cache import make_kv_cache

        cache = make_kv_cache(config=config)
        for layer_idx, (key_states, value_states) in enumerate(evicted):
            cache.update(key_states, value_states, layer_idx)
        return cache
    except Exception:
        return tuple(evicted)


def slide_dialogue_kv_prefix(model, drop_len: int, anchor_len: int = 0) -> Dict[str, Any]:
    """Evict history tokens from dialogue KV (RoPE-relative equivalent to full prefill).

    Anchored eviction (``anchor_len > 0``): keep the system anchor
    ``[0, anchor)``, drop ``[anchor, anchor+drop)`` (the oldest turn), keep the
    tail. Retained tail keys stay rotated at their original absolute MRoPE
    positions; the anchor keys are re-rotated *up* by the proven cumulative
    offset so the whole cache sits uniformly at ``recompute@0 + offset`` — the
    exact frame the position-continuation contract already assumes.

    ``anchor_len == 0`` degenerates to the legacy pure prefix drop.
    """
    stats: Dict[str, Any] = {
        "dialogue_kv_slide_ok": False,
        "dialogue_kv_slide_drop_tokens": int(drop_len),
        "dialogue_kv_slide_anchor_len": int(anchor_len),
    }
    anchor = int(anchor_len)
    if not dialogue_kv_enabled() or drop_len <= 0 or anchor < 0:
        stats["dialogue_kv_slide_reason"] = "disabled_or_zero"
        return stats

    inner = _inner_model(model)
    state = _get_dialogue_state(inner)
    if not state or state.get("past_key_values") is None:
        stats["dialogue_kv_slide_reason"] = "empty_state"
        return stats

    cached_ids = list(state.get("cached_input_ids") or [])
    past_len = int(state.get("prompt_token_len") or len(cached_ids))
    if anchor + int(drop_len) >= past_len:
        stats["dialogue_kv_slide_reason"] = "drop_ge_past_len"
        return stats

    prev_offset = _mrope_pos_offset(state)
    pristine_keys = state.get("sys_pristine_keys")
    pristine_anchor = int(state.get("sys_anchor_len") or 0)
    if anchor:
        if pristine_keys is None:
            if prev_offset != 0:
                # System keys in the live cache are no longer at original
                # positions and we have nothing pristine to re-rotate from.
                stats["dialogue_kv_slide_reason"] = "anchor_pristine_missing"
                return stats
        elif pristine_anchor != anchor:
            stats["dialogue_kv_slide_reason"] = (
                f"anchor_len_changed prev={pristine_anchor} now={anchor}"
            )
            return stats

    lm = _carrier_text_model(inner)
    cache_config = getattr(lm, "config", getattr(inner, "config", None))

    new_len = past_len - int(drop_len)
    new_ids = cached_ids[:anchor] + cached_ids[anchor + int(drop_len) :]
    cached_mrope = list(state.get("cached_mrope_input_ids") or [])
    image_token_id = _image_token_id(inner)
    prune_on_prefix = bool(cached_mrope) and len(cached_mrope) > len(cached_ids)
    if prune_on_prefix:
        mapping = _compact_to_virtual_index_map(
            cached_ids[:past_len],
            cached_mrope,
            image_token_id,
        )
        drop_end = anchor + int(drop_len)
        if mapping is None or drop_end >= len(mapping):
            diag = ""
            if mapping is None:
                ci = vi = 0
                img = int(image_token_id)
                comp = cached_ids[:past_len]
                while ci < len(comp) and vi < len(cached_mrope):
                    if int(cached_mrope[vi]) == img and int(comp[ci]) != img:
                        vi += 1
                        continue
                    if int(cached_mrope[vi]) != int(comp[ci]):
                        break
                    ci += 1
                    vi += 1
                diag = (
                    f" first_div ci={ci} vi={vi} "
                    f"c={comp[max(0, ci - 3) : ci + 4]} "
                    f"v={cached_mrope[max(0, vi - 3) : vi + 4]}"
                )
            stats["dialogue_kv_slide_reason"] = (
                "virtual_map_failed "
                f"map={'none' if mapping is None else len(mapping)} "
                f"len_c={len(cached_ids)} len_v={len(cached_mrope)} "
                f"past={past_len} drop_end={drop_end}" + diag
            )
            return stats
        if anchor and int(mapping[anchor]) != anchor:
            # The system anchor must contain no visual tokens, so compact and
            # virtual indices coincide there.
            stats["dialogue_kv_slide_reason"] = "anchor_virtual_mismatch"
            return stats
        virt_end = int(mapping[drop_end])
        if virt_end <= anchor or virt_end >= len(cached_mrope):
            stats["dialogue_kv_slide_reason"] = "virtual_drop_oob"
            return stats
        new_mrope = cached_mrope[:anchor] + cached_mrope[virt_end:]
        stats["dialogue_kv_slide_virt_drop"] = virt_end - anchor
    elif cached_mrope and len(cached_mrope) >= past_len:
        virt_end = anchor + int(drop_len)
        new_mrope = cached_mrope[:anchor] + cached_mrope[virt_end:]
    else:
        stats["dialogue_kv_slide_reason"] = "mrope_len_mismatch"
        return stats

    # The frame bookkeeping below (image_paths[1:], grid[1:], compressed-key
    # shift) hard-codes drop_frames=1; require the evicted virtual range to
    # contain exactly one image run so a text-only or multi-image turn cannot
    # silently desync grids from ids (would corrupt MRoPE downstream).
    dropped_runs = len(_visual_spans(cached_mrope[anchor:virt_end], image_token_id))
    if dropped_runs != 1:
        stats["dialogue_kv_slide_reason"] = (
            f"drop_frames_mismatch runs={dropped_runs} "
            f"virt_range={anchor}:{virt_end}"
        )
        return stats

    # Position-continuation contract: retained KV keys keep their original
    # absolute MRoPE positions. Capture the old-frame virtual positions before
    # mutation so every post-slide recompute (anchored at 0 on trimmed ids)
    # can be shifted back by a proven constant offset.
    state = _ensure_virtual_mrope_cache(inner, state)
    old_full_pos = _get_cached_virtual_full_mrope(state, device=torch.device("cpu"))
    if old_full_pos is None or int(old_full_pos.shape[-1]) != len(cached_mrope):
        stats["dialogue_kv_slide_reason"] = "pos_offset_unavailable"
        return stats
    old_frame_tail = old_full_pos[:, :, virt_end:].contiguous()
    old_anchor_pos = old_full_pos[:, :, :anchor].contiguous() if anchor else None

    new_state = dict(state)
    new_state["cached_input_ids"] = new_ids
    new_state["cached_mrope_input_ids"] = new_mrope
    new_state["prompt_token_len"] = new_len

    image_paths = list(state.get("image_paths") or [])
    if len(image_paths) > 1:
        new_state["image_paths"] = image_paths[1:]
    # Frame-instance keys are relative to the live dialogue window; when the
    # oldest frame rolls out, remaining compressed markers shift left with it.
    new_state["kv_compressed_paths"] = _shift_kv_compressed_keys_after_slide(
        state.get("kv_compressed_paths") or (),
        drop_frames=1,
    )
    prev_grid = state.get("image_grid_thw")
    if prev_grid is not None:
        try:
            n_rows = int(prev_grid.shape[0])
        except Exception as exc:
            stats["dialogue_kv_slide_reason"] = f"grid_shape_failed:{exc}"
            return stats
        if n_rows <= 1:
            stats["dialogue_kv_slide_reason"] = "grid_too_short"
            return stats
        new_state["image_grid_thw"] = prev_grid[1:].clone()

    # Prove the offset: recompute the trimmed window anchored at 0 and require
    # (a) the anchor region to reproduce its original positions exactly and
    # (b) a constant per-element delta over the tail. Any drift means the
    # eviction boundary is not RoPE-translation-equivalent -> refuse to slide.
    proof_t0 = time.perf_counter()
    try:
        from lmms_eval.models.geopact_vl_hook import _precompute_full_mrope

        try:
            embed_fn = inner.get_input_embeddings()
            mrope_device = embed_fn.weight.device
        except Exception:
            mrope_device = torch.device("cpu")
        new_grid = new_state.get("image_grid_thw")
        if new_grid is None:
            stats["dialogue_kv_slide_reason"] = "pos_offset_no_grid"
            return stats
        virt_tensor = torch.tensor([new_mrope], dtype=torch.long, device=mrope_device)
        virt_attn = torch.ones((1, len(new_mrope)), dtype=torch.long, device=mrope_device)
        new_raw_pos, _ = _precompute_full_mrope(
            inner,
            virt_tensor,
            new_grid.to(device=mrope_device),
            virt_attn,
        )
        new_raw_pos = new_raw_pos.detach().cpu()
        if anchor:
            # Anchor keys currently sit at (0-anchored + previous cumulative
            # offset); the identical system text must recompute to the same
            # 0-anchored positions, so the delta equals the previous offset.
            anchor_delta = old_anchor_pos.to(dtype=new_raw_pos.dtype) - new_raw_pos[:, :, :anchor]
            if anchor_delta.numel() == 0 or not bool(
                (anchor_delta == int(prev_offset)).all()
            ):
                stats["dialogue_kv_slide_reason"] = "anchor_pos_mismatch"
                return stats
        delta = old_frame_tail.to(dtype=new_raw_pos.dtype) - new_raw_pos[:, :, anchor:]
        if delta.numel() == 0 or not bool((delta == delta.flatten()[0]).all()):
            stats["dialogue_kv_slide_reason"] = "pos_offset_mismatch"
            return stats
        pos_offset = int(delta.flatten()[0].item())
        if pos_offset <= 0:
            stats["dialogue_kv_slide_reason"] = f"pos_offset_nonpositive:{pos_offset}"
            return stats
    except Exception as exc:
        stats["dialogue_kv_slide_reason"] = f"pos_offset_failed:{exc}"
        return stats
    slide_pos_proof_ms = round((time.perf_counter() - proof_t0) * 1000.0, 2)

    # Evict KV. With an anchor, capture pristine system keys on the first
    # eviction (offset 0 -> keys still at original positions) and always
    # re-rotate from pristine by the cumulative offset — never rotate rotated
    # keys, so fp error does not compound across rolls.
    slide_rotate_ms = 0.0
    try:
        legacy = _legacy_cache_tensors(state["past_key_values"])
        if not legacy:
            stats["dialogue_kv_slide_reason"] = "legacy_cache_unavailable"
            return stats
        evict_t0 = time.perf_counter()
        rotated_sys_keys: Optional[List[torch.Tensor]] = None
        if anchor:
            if pristine_keys is None:
                pristine_keys = [k[..., :anchor, :].detach().clone() for k, _v in legacy]
                new_state["sys_pristine_keys"] = pristine_keys
                new_state["sys_anchor_len"] = anchor
            inv_freq = _text_rope_inv_freq(inner)
            rotated_sys_keys = [
                _rotate_post_rope_keys(k, pos_offset, inv_freq) for k in pristine_keys
            ]
            slide_rotate_ms = round((time.perf_counter() - evict_t0) * 1000.0, 2)
        slid_pkv = _anchored_evict_past_key_values(
            state["past_key_values"],
            anchor_len=anchor,
            drop_len=int(drop_len),
            rotated_sys_keys=rotated_sys_keys,
            config=cache_config,
        )
        slide_evict_ms = round(
            (time.perf_counter() - evict_t0) * 1000.0 - slide_rotate_ms, 2
        )
    except Exception as exc:
        stats["dialogue_kv_slide_reason"] = f"slide_failed:{exc}"
        return stats
    new_state["past_key_values"] = slid_pkv
    stats["dialogue_kv_slide_pos_proof_ms"] = slide_pos_proof_ms
    stats["dialogue_kv_slide_rotate_ms"] = slide_rotate_ms
    stats["dialogue_kv_slide_evict_ms"] = slide_evict_ms
    try:
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if ledger is not None:
            ledger.set_audit(
                slide_pos_proof_ms=slide_pos_proof_ms,
                slide_rotate_ms=slide_rotate_ms,
                slide_evict_ms=slide_evict_ms,
            )
    except Exception:
        pass

    new_state["mrope_virt_pos_offset"] = pos_offset
    stats["dialogue_kv_slide_pos_offset"] = pos_offset
    # Module C (QASMR): v1 does not track relay frames across anchored
    # eviction (their cache spans / rotation deltas are frame-dependent);
    # drop them conservatively rather than serve mispositioned relays.
    try:
        from lmms_eval.models.geopact_spatial_relay import drop_relays_on_slide

        drop_relays_on_slide(inner, new_state)
    except Exception:
        pass

    new_state.pop("cached_boundary_position_ids", None)
    new_state.pop("_sliding_rewrite_pending", None)
    new_state.pop("_sliding_rewrite_active", None)
    new_state = _invalidate_virtual_mrope_cache(new_state)
    # Post-eviction virtual positions in the KV frame are uniformly
    # ``recompute@0 + cumulative offset``: the anchor keys were re-rotated to
    # exactly that and the tail keeps its old-frame (absolute) positions,
    # which the delta proof showed equal recompute + offset.
    kv_frame_pos = (new_raw_pos + pos_offset).to(dtype=old_frame_tail.dtype).contiguous()
    new_state = _store_cached_virtual_full_mrope(new_state, kv_frame_pos)
    new_state = _refresh_dialogue_boundary_in_state(inner, new_state)
    boundary = new_state.get("cached_boundary_position_ids")
    if isinstance(boundary, torch.Tensor) and int(boundary.shape[-1]) == 1:
        new_state["rope_deltas"] = _rope_deltas_for_position_ids(
            boundary,
            int(new_len),
        ).detach().cpu()
    else:
        new_state.pop("rope_deltas", None)

    # Fail-fast invariants: KV length, ids length and prompt_token_len must
    # stay in lockstep after the slide (silent drift here corrupts decode).
    try:
        slid_kv_len = int(slid_pkv.get_seq_length())
    except Exception:
        slid_kv_len = -1
    if not (slid_kv_len == new_len == len(new_ids)):
        stats["dialogue_kv_slide_reason"] = (
            f"invariant_kv_len_mismatch kv={slid_kv_len} "
            f"prompt={new_len} ids={len(new_ids)}"
        )
        return stats

    _set_dialogue_state(inner, new_state)
    stats["dialogue_kv_slide_ok"] = True
    stats["dialogue_kv_slide_new_len"] = new_len
    stats["dialogue_kv_slide_reason"] = "ok"
    return stats


class DialogueKVCapture:
    """Capture KV at the end of prompt prefill (before decode tokens)."""

    def __init__(self, model, target_kv_len: int):
        self.model = model
        self.target_kv_len = int(target_kv_len)
        self.captured = None
        self._handle = None

    @staticmethod
    def _is_capture_prefill_forward(
        cache_position: Any,
        *,
        target_kv_len: int,
        seq_len: Optional[int] = None,
    ) -> bool:
        """True for multi-token prefill or the last single-token suffix forward."""
        if cache_position is None:
            return True
        if not hasattr(cache_position, "numel"):
            return False
        n = int(cache_position.numel())
        if n > 1:
            return True
        if n == 1:
            # For n==1, cache_position[0] == seq_len_after - 1, so the legacy
            # check `cache_position[0] < target_kv_len` is equivalent to
            # `seq_len <= target_kv_len`. Using the hook's Python seq_len
            # avoids a per-decode-token D2H `.item()` sync.
            if seq_len is not None:
                return int(seq_len) <= int(target_kv_len)
            try:
                return int(cache_position[0].item()) < int(target_kv_len)
            except Exception:
                return False
        return False

    def install(self) -> None:
        inner = _inner_model(self.model)
        lm = _carrier_text_model(inner)
        cache_config = getattr(lm, "config", getattr(inner, "config", None))

        def _hook(_module, _args, _kwargs, output) -> None:
            if self.captured is not None:
                return
            pkv = getattr(output, "past_key_values", None)
            if pkv is None:
                return
            try:
                seq_len = int(pkv.get_seq_length())
            except Exception:
                return
            if seq_len < self.target_kv_len:
                return
            cache_position = _kwargs.get("cache_position")
            if not self._is_capture_prefill_forward(
                cache_position,
                target_kv_len=self.target_kv_len,
                seq_len=seq_len,
            ):
                return
            capture_t0 = time.perf_counter()
            self.captured = _crop_past_key_values(
                pkv,
                self.target_kv_len,
                config=cache_config,
            )
            if self.captured is not None:
                from lmms_eval.models.geopact_step_timing import get_step_ledger

                ledger = get_step_ledger(self.model)
                ledger.add_ms(
                    "dialogue_kv_capture_ms",
                    (time.perf_counter() - capture_t0) * 1000.0,
                )

        self._handle = lm.register_forward_hook(_hook, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def snapshot_from_pkv(self, pkv: Any, *, target_len: Optional[int] = None) -> bool:
        """Explicit KV snapshot (suffix batch prefill path; hook may never fire)."""
        if pkv is None:
            return False
        inner = _inner_model(self.model)
        lm = _carrier_text_model(inner)
        cache_config = getattr(lm, "config", getattr(inner, "config", None))
        target = int(target_len if target_len is not None else self.target_kv_len)
        try:
            seq_len = int(pkv.get_seq_length())
        except Exception:
            return False
        if seq_len < target:
            return False
        self.captured = _crop_past_key_values(pkv, target, config=cache_config)
        return self.captured is not None


def install_dialogue_kv_capture(model, target_kv_len: int) -> DialogueKVCapture:
    capture = DialogueKVCapture(model, target_kv_len)
    capture.install()
    return capture


def prepare_dialogue_generate_payload(
    model,
    doc: Dict,
    suffix_inputs: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Reuse KV through previous user turn; prefill only GT assistant + current user."""
    stats: Dict[str, Any] = {
        "dialogue_kv_enabled": dialogue_kv_enabled(),
        "dialogue_kv_hit": False,
        "dialogue_kv_prefix_tokens": 0,
        "dialogue_kv_suffix_tokens": 0,
        "kv_flow_mode": kv_flow_mode_label(model),
    }
    if not dialogue_kv_enabled():
        return None, stats

    inner = _inner_model(model)
    step_index = session_step_index(doc)
    episode_key = session_episode_key(doc)
    if step_index <= 0:
        stats["dialogue_kv_reason"] = "first_step"
        return None, stats

    state = _get_dialogue_state(inner)
    if not state or state.get("past_key_values") is None:
        stats["dialogue_kv_reason"] = "empty_state"
        return None, stats
    if state.get("episode_key") != episode_key:
        clear_session_kv(model)
        stats["dialogue_kv_reason"] = "episode_changed"
        return None, stats
    if int(state.get("through_user_step", state.get("committed_step", -1))) != step_index - 1:
        stats["dialogue_kv_reason"] = "step_gap"
        return None, stats

    prefill = getattr(inner, "_geopact_prefill_payload", None)
    if prefill is None:
        stats["dialogue_kv_reason"] = "no_prefill_payload"
        return None, stats
    if suffix_inputs.get("inputs_embeds") is None:
        stats["dialogue_kv_reason"] = "missing_inputs_embeds"
        return None, stats

    from lmms_eval.models.geopact_step_timing import get_step_ledger

    ledger = get_step_ledger(model)
    ledger.start_bucket("dialogue_kv_prepare_ms")
    try:
        return _prepare_dialogue_generate_payload_inner(
            model,
            doc,
            suffix_inputs,
            stats,
            inner=inner,
            state=state,
            prefill=prefill,
        )
    finally:
        ledger.stop_bucket("dialogue_kv_prepare_ms")


def _prepare_dialogue_generate_payload_inner(
    model,
    doc: Dict,
    suffix_inputs: Dict[str, Any],
    stats: Dict[str, Any],
    *,
    inner: Any,
    state: Dict[str, Any],
    prefill: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    past_kv = state["past_key_values"]
    past_len = int(past_kv.get_seq_length()) if past_kv is not None else int(
        state.get("prompt_token_len", 0) or 0
    )
    cached_len = len(state.get("cached_input_ids") or [])
    expected_len = int(state.get("prompt_token_len", cached_len) or cached_len)
    if cached_len > 0 and past_len != cached_len:
        clear_session_kv(model)
        stats["dialogue_kv_reason"] = "past_len_mismatch"
        stats["dialogue_kv_past_len"] = past_len
        stats["dialogue_kv_cached_len"] = cached_len
        return None, stats
    if expected_len > 0 and past_len != expected_len:
        clear_session_kv(model)
        stats["dialogue_kv_reason"] = "prompt_token_len_mismatch"
        stats["dialogue_kv_past_len"] = past_len
        stats["dialogue_kv_expected_len"] = expected_len
        return None, stats
    image_token_id = _image_token_id(inner)
    pending = state.get("_sliding_rewrite_pending")
    suffix, suffix_prefill = _slice_suffix_for_past(
        suffix_inputs,
        prefill,
        start=0,
        past_len=past_len,
        image_token_id=image_token_id,
    )
    suffix_len = int(suffix["inputs_embeds"].shape[1])

    if pending:
        rewrite_len = int(pending.get("rewrite_len") or 0)
        vis_start = int(pending.get("vis_start") or 0)
        rewrite_pos = pending.get("position_ids")
        if rewrite_pos is None:
            raise RuntimeError("GeoPACT sliding rewrite: pending missing rewrite position_ids")
        device = suffix["inputs_embeds"].device
        suffix_position_ids, rope_deltas, mrope_ms = _align_sliding_suffix_mrope_fast(
            suffix_prefill or {},
            rewrite_pos,
            suffix_len=suffix_len,
            device=device,
        )
        stats["sliding_rewrite_merged"] = True
        stats["sliding_rewrite_rewrite_len"] = int(rewrite_len)
        stats["sliding_rewrite_suffix_mrope_path"] = "fast_align"
        stats["dialogue_kv_mrope_ms"] = mrope_ms
    else:
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if ledger is not None:
            # The helper self-reports dialogue_suffix_mrope_ms via add_ms;
            # release prepare so the mrope wall is not credited twice.
            ledger.stop_bucket("dialogue_kv_prepare_ms")
        suffix_position_ids, rope_deltas, mrope_ms = _recompute_dialogue_suffix_mrope(
            model,
            inner,
            state,
            suffix_inputs,
            past_len=past_len,
            suffix_len=suffix_len,
            suffix_prefill=suffix_prefill,
        )
        if ledger is not None:
            ledger.start_bucket("dialogue_kv_prepare_ms")
        stats["dialogue_kv_mrope_ms"] = mrope_ms
    suffix["position_ids"] = suffix_position_ids
    if not pending:
        inner._geopact_dialogue_last_suffix_position_ids = suffix_position_ids.detach().cpu().clone()
    if rope_deltas is not None:
        rope_deltas = rope_deltas.to(device=suffix["inputs_embeds"].device)
        suffix["rope_deltas"] = rope_deltas
        inner.rope_deltas = rope_deltas
    suffix["past_key_values"] = past_kv
    # Single D2H copy of suffix ids, reused below (commit + stage paths).
    suffix_ids = suffix_inputs["input_ids"][0].detach().cpu().tolist()
    merged_len = int(suffix_len)
    if pending:
        merged_len += int(pending.get("rewrite_len") or 0)
    if past_len > 0 and merged_len > 1:
        # dialogue_kv_prepare_ms stays active into the prefill helpers; they
        # switch to the lm/crop buckets around the forward and restore it, so
        # payload assembly (merged embeds, visual meta, suffix rebuild) is
        # accounted instead of leaking into misc.
        if pending:
            past_kv, prefill_stats = _prefill_merged_sliding_rewrite_before_generate(
                model,
                suffix,
                pending=pending,
                past_kv=past_kv,
                past_len=past_len,
                suffix_prefill=suffix_prefill,
            )
            stats["sliding_rewrite_lm_forwards"] = 1
        else:
            past_kv, prefill_stats = _prefill_suffix_before_generate(
                model,
                suffix,
                past_kv=past_kv,
                past_len=past_len,
                suffix_prefill=suffix_prefill,
            )
        stats.update(prefill_stats)
        inner._geopact_dialogue_direct_pkv = past_kv
        try:
            stats["dialogue_kv_direct_pkv_len"] = int(past_kv.get_seq_length())
        except Exception:
            pass
        if pending:
            state = _get_dialogue_state(inner) or state
            _finalize_sliding_rewrite_state(
                inner,
                state,
                pending,
                suffix_ids=suffix_ids,
            )
            stats["sliding_rewrite_committed"] = True
    elif past_len > 0 and merged_len == 1:
        inner._geopact_dialogue_direct_pkv = past_kv
        try:
            stats["dialogue_kv_direct_pkv_len"] = int(past_kv.get_seq_length())
        except Exception:
            pass
    try:
        legacy = _legacy_cache_tensors(past_kv)
        if legacy:
            k0, v0 = legacy[0]
            stats["dialogue_kv_layer0_k_shape"] = "x".join(str(int(x)) for x in k0.shape)
            stats["dialogue_kv_layer0_v_shape"] = "x".join(str(int(x)) for x in v0.shape)
            if suffix.get("inputs_embeds") is not None:
                stats["dialogue_kv_inputs_embeds_shape"] = "x".join(
                    str(int(x)) for x in suffix["inputs_embeds"].shape
                )
            if suffix.get("input_ids") is not None:
                stats["dialogue_kv_input_ids_shape"] = "x".join(
                    str(int(x)) for x in suffix["input_ids"].shape
                )
            if suffix.get("attention_mask") is not None:
                stats["dialogue_kv_attention_mask_shape"] = "x".join(
                    str(int(x)) for x in suffix["attention_mask"].shape
                )
            if suffix.get("position_ids") is not None:
                stats["dialogue_kv_position_ids_shape"] = "x".join(
                    str(int(x)) for x in suffix["position_ids"].shape
                )
    except Exception as exc:
        stats["dialogue_kv_shape_debug_error"] = str(exc)
    # Do NOT route dialogue suffix through _geopact_prefill_payload: that forward
    # branch swaps in suffix-local attention_mask/embed metadata and breaks
    # flash-attn varlen KV sizing when past_key_values is already populated.
    cached_ids = list(state.get("cached_input_ids") or [])
    image_paths = list(
        getattr(inner, "_geopact_dialogue_episode_image_paths", None)
        or state.get("image_paths")
        or []
    )
    # Suffix prefill embeds align with suffix_ids only (not prefix+cached).
    stage_pending_last_full_frame(
        model,
        stored_ids=suffix_ids,
        image_paths=image_paths,
    )
    inner._geopact_prefill_payload = None

    stats["dialogue_kv_hit"] = True
    stats["dialogue_kv_reason"] = "hit"
    stats["dialogue_kv_prefix_tokens"] = past_len
    stats["dialogue_kv_suffix_tokens"] = int(suffix_len)
    if suffix.get("inputs_embeds") is not None:
        stats["dialogue_kv_generate_suffix_tokens"] = int(suffix["inputs_embeds"].shape[1])
    elif suffix.get("input_ids") is not None:
        stats["dialogue_kv_generate_suffix_tokens"] = int(suffix["input_ids"].shape[1])
    stats["dialogue_kv_through_user_step"] = int(
        state.get("through_user_step", state.get("committed_step", -1))
    )
    return suffix, stats


def _kv_history_rewrite_enabled() -> bool:
    """KV rewrite for the ``history`` prune target (else full re-prefill fallback)."""
    raw = os.environ.get("GEOPACT_KV_HISTORY_REWRITE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _last_visual_span(ids: List[int], image_token_id: int) -> Optional[Tuple[int, int]]:
    """Return [start, end) of the last contiguous image-token run in ``ids``."""
    if image_token_id < 0 or not ids:
        return None
    end = None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == image_token_id:
            end = i + 1
            break
    if end is None:
        return None
    start = end
    while start - 1 >= 0 and ids[start - 1] == image_token_id:
        start -= 1
    return (start, end)


def _visual_spans(ids: List[int], image_token_id: int) -> List[Tuple[int, int]]:
    """Return [start, end) for each contiguous image-token run left-to-right."""
    if image_token_id < 0 or not ids:
        return []
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(ids)
    while i < n:
        if int(ids[i]) == int(image_token_id):
            start = i
            while i < n and int(ids[i]) == int(image_token_id):
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def resolve_prune_target(model=None) -> str:
    """Unified prune target: model attr first, then GEOPACT_PRUNE_TARGET env.

    ``_geopact_prune_target`` lives on the inner ``Qwen3VLModel``; callers hold
    anything from the lmms wrapper down to that module, so walk the ``.model``
    chain before falling back to the env (a bare outer handle silently reading
    "current" is how NAP stage 2 once degraded to the re-selection path).
    """
    target = ""
    cur = model
    seen = set()
    while cur is not None and id(cur) not in seen:
        target = str(getattr(cur, "_geopact_prune_target", "") or "")
        if target:
            break
        seen.add(id(cur))
        cur = getattr(cur, "model", None)
    if not target:
        target = os.environ.get("GEOPACT_PRUNE_TARGET", "") or ""
    target = target.strip().lower()
    return "history" if target == "history" else "current"


def _prune_target_history(inner) -> bool:
    return resolve_prune_target(inner) == "history"


def stage_pending_last_full_frame(
    model,
    *,
    stored_ids: List[int],
    image_paths: List[str],
) -> None:
    """Lightweight last-frame capture before suffix prefill payload is cleared."""
    if not kv_history_prune_rewrite_enabled(model):
        return
    inner = _inner_model(model)
    prefill = getattr(inner, "_geopact_prefill_payload", None)
    if prefill is None:
        return
    image_token_id = _image_token_id(inner)
    cap = _capture_last_full_frame(
        inner,
        prefill,
        stored_ids=list(stored_ids),
        image_paths=list(image_paths),
        image_token_id=image_token_id,
    )
    if cap is not None:
        inner._geopact_pending_last_frame = cap
        try:
            from lmms_eval.models.geopact_module_a_prefetch import (
                prefetch_module_a_for_history_capture,
            )

            prefetch_module_a_for_history_capture(model, cap)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_rectify_prefetch import (
                rectify_prefetch_at,
                start_rectify_prefetch,
            )

            # Default submit point is post-decode (finalize): submitting here
            # (pre-decode) hides the job behind the whole decode but its
            # kernels/GIL contend with the decode loop (+1-2ms/token) unless
            # the low-priority stream fully absorbs it.
            if rectify_prefetch_at() == "stage":
                start_rectify_prefetch(model, cap)
        except Exception:
            pass


def _visual_skeleton(ids: List[int], image_token_id: int) -> List[int]:
    """Collapse each maximal image-token run to a single ``-1`` marker.

    Skeleton equality proves the text stream (turn headers, coordinates,
    separators) is byte-identical while letting an ABC-pruned window carry
    shorter visual runs than the freshly tokenised full prompt.
    """
    out: List[int] = []
    prev_img = False
    for tok in ids:
        t = int(tok)
        is_img = t == int(image_token_id)
        if is_img:
            if not prev_img:
                out.append(-1)
        else:
            out.append(t)
        prev_img = is_img
    return out


def _anchor_prefix_len(ids: List[int], im_start_id: int) -> int:
    """Length of the system anchor prefix: tokens before the second
    ``<|im_start|>`` (i.e. the system turn plus trailing scaffold). Returns 0
    when the marker cannot be located (no anchored eviction possible)."""
    if im_start_id < 0 or not ids:
        return 0
    seen = 0
    for idx, tok in enumerate(ids):
        if int(tok) == int(im_start_id):
            seen += 1
            if seen == 2:
                return idx
    return 0


def _skeleton_with_starts(
    ids: List[int],
    image_token_id: int,
) -> Tuple[List[int], List[int]]:
    """Visual skeleton plus the original token index of each element start."""
    sk: List[int] = []
    starts: List[int] = []
    prev_img = False
    for idx, tok in enumerate(ids):
        t = int(tok)
        is_img = t == int(image_token_id)
        if is_img:
            if not prev_img:
                sk.append(-1)
                starts.append(idx)
        else:
            sk.append(t)
            starts.append(idx)
        prev_img = is_img
    return sk, starts


def compute_dialogue_roll_drop_len(
    state: Dict[str, Any],
    *,
    cached_ids: List[int],
    suffix_ids: List[int],
    full_ids: Optional[List[int]],
    image_token_id: int,
    im_start_id: int = -1,
) -> Optional[Tuple[int, int]]:
    """Anchored rolling-window eviction boundary for dialogue KV slide.

    Contract (anchored eviction: keep the system anchor, evict the oldest
    turn, keep the tail): the retained window
    ``cached_ids[:anchor] + cached_ids[anchor+drop:] + suffix_ids`` must
    re-create the freshly tokenised prompt ``full_ids`` exactly. With an
    ABC-pruned prefix, visual runs in ``cached_ids`` are shorter than in
    ``full_ids``, so tail equality is proven in skeleton space (image runs
    wildcarded, text tokens exact); the anchor itself contains no visual
    tokens and must match token-exactly.

    Returns ``(anchor_len, drop_len)`` with ``drop_len > 0`` for a proven
    eviction, ``(0, 0)`` for the pure-append case, or ``None`` when the
    boundary could not be proven — the caller must then fall back to a full
    reset. Never slide at an unproven boundary.
    """
    if not cached_ids:
        return None

    if (
        full_ids is not None
        and len(cached_ids) < len(full_ids)
        and cached_ids + suffix_ids == full_ids
    ):
        return (0, 0)
    if full_ids is None:
        return None

    anchor = _anchor_prefix_len(cached_ids, im_start_id)
    if anchor and (
        len(full_ids) < anchor or full_ids[:anchor] != cached_ids[:anchor]
    ):
        anchor = 0

    # Exact path: drop length is uniquely determined by token counts.
    drop = len(cached_ids) + len(suffix_ids) - len(full_ids)
    if 0 < drop and anchor + drop < len(cached_ids):
        if cached_ids[anchor + drop :] + suffix_ids == full_ids[anchor:]:
            return (anchor, drop)
        if anchor and cached_ids[drop:] + suffix_ids == full_ids:
            # Degenerate no-anchor layout (system not retained by protocol).
            return (0, drop)

    image_paths = list(state.get("image_paths") or [])
    if len(image_paths) < 2:
        return None
    cached_mrope = list(state.get("cached_mrope_input_ids") or [])
    pruned_prefix = bool(cached_mrope) and len(cached_mrope) > len(cached_ids)
    if not pruned_prefix or image_token_id < 0:
        return None

    # Skeleton path (ABC-pruned prefix). The anchor was already proven
    # token-exact above (or is 0); prove the tail in skeleton space.
    full_sk = _visual_skeleton(full_ids[anchor:], image_token_id)
    suffix_sk = _visual_skeleton(suffix_ids, image_token_id)
    cached_sk, cached_sk_starts = _skeleton_with_starts(
        cached_ids[anchor:], image_token_id
    )

    for merge_junction in (False, True):
        if merge_junction:
            # cached window ends inside a visual run and suffix starts with
            # one: the runs merge into a single skeleton marker.
            if not (suffix_sk and suffix_sk[0] == -1):
                continue
            tail_sk = suffix_sk[1:]
        else:
            tail_sk = suffix_sk
        j = len(cached_sk) - (len(full_sk) - len(tail_sk))
        if j <= 0 or j >= len(cached_sk):
            continue
        cand = cached_sk[j:]
        if merge_junction and (not cand or cand[-1] != -1):
            continue
        if cand + tail_sk != full_sk:
            continue
        drop = int(cached_sk_starts[j])
        if 0 < drop and anchor + drop < len(cached_ids):
            return (anchor, drop)
    return None


def _build_sliding_rewrite_mrope(
    inner,
    state: Dict[str, Any],
    *,
    vis_start: int,
    n_frame: int,
    tail_ids: List[int],
    local_keep: List[int],
    image_token_id: int,
    device: torch.device,
    cached_full_pos: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """MRoPE for the rewrite segment (rectified visual + text tail)."""
    from lmms_eval.models.geopact_vl_hook import _precompute_full_mrope, _subset_origpos_mrope_cached

    cached_ids = list(state.get("cached_input_ids") or [])
    virtual_ids = list(state.get("cached_mrope_input_ids") or []) or list(cached_ids)
    full_grid = state.get("image_grid_thw")
    if full_grid is None:
        return None, None, 0
    full_grid = full_grid.to(device=device)

    tail_len = len(tail_ids)
    virt_region_start = len(virtual_ids) - tail_len - n_frame
    if virt_region_start < 0:
        return None, None, 0

    full_pos = cached_full_pos
    if full_pos is None:
        full_pos = _get_cached_virtual_full_mrope(state, device=device)
    if full_pos is None:
        virt_tensor = torch.tensor([virtual_ids], dtype=torch.long, device=device)
        virt_attn = torch.ones((1, len(virtual_ids)), dtype=torch.long, device=device)
        full_pos, _ = _precompute_full_mrope(inner, virt_tensor, full_grid, virt_attn)
        # Shift 0-anchored recompute back into the original coordinate frame
        # so rewrite positions stay continuous with the retained KV keys.
        pos_offset = _mrope_pos_offset(state)
        if pos_offset:
            full_pos = full_pos + pos_offset
        stored = _store_cached_virtual_full_mrope(state, full_pos)
        if stored.get("cached_virtual_mrope_key"):
            state["cached_virtual_mrope_key"] = stored["cached_virtual_mrope_key"]
            state["cached_virtual_mrope_pos"] = stored["cached_virtual_mrope_pos"]

    region_pos = _slice_position_ids(full_pos, virt_region_start, len(virtual_ids))
    region_len = n_frame + tail_len
    keep_mask = torch.zeros(region_len, dtype=torch.bool, device=device)
    if local_keep:
        keep_mask[torch.as_tensor(local_keep, dtype=torch.long, device=device)] = True
    keep_mask[n_frame:] = True
    rewrite_len = len(local_keep) + tail_len
    region_ids = torch.tensor(
        [[image_token_id] * len(local_keep) + tail_ids],
        dtype=torch.long,
        device=device,
    )
    rewrite_pos, rope_deltas = _subset_origpos_mrope_cached(
        inner,
        region_pos,
        keep_mask,
        region_ids,
        None,
    )
    if int(rewrite_pos.shape[-1]) != rewrite_len:
        return None, None, 0
    return rewrite_pos, rope_deltas, rewrite_len


@torch.inference_mode()
def sliding_rewrite_prepare_dialogue_kv(
    model,
    doc: Dict,
    *,
    ratio: float,
    policy: str,
) -> Dict[str, Any]:
    """Prepare sliding-window history rewrite (Vision ABC + KV truncate, no LM yet).

    Stores ``_sliding_rewrite_pending`` in dialogue state; the merged rewrite +
    incremental suffix LM forward happens in ``prepare_dialogue_generate_payload``.
    """
    stats: Dict[str, Any] = {"sliding_rewrite_ok": False}
    if not dialogue_kv_enabled() or not _kv_history_rewrite_enabled():
        stats["sliding_rewrite_reason"] = "disabled"
        return stats

    inner = _inner_model(model)
    # Barrier: the async rectify job may still be running (short decode); let it
    # finish before mutating SSPC/step state it might be reading.
    try:
        from lmms_eval.models.geopact_rectify_prefetch import wait_rectify_prefetch

        wait_rectify_prefetch(inner)
    except Exception:
        pass
    try:
        from lmms_eval.models.geopact_sspc import reset_sspc_step_state, sspc_enabled

        if sspc_enabled():
            reset_sspc_step_state(inner)
    except Exception:
        pass

    if not _prune_target_history(inner):
        stats["sliding_rewrite_reason"] = "not_history_target"
        return stats

    step_index = session_step_index(doc)
    if step_index <= 0:
        stats["sliding_rewrite_reason"] = "first_step"
        return stats

    state = _get_dialogue_state(inner)
    if not state or state.get("past_key_values") is None:
        stats["sliding_rewrite_reason"] = "empty_state"
        return stats
    if state.get("episode_key") != session_episode_key(doc):
        stats["sliding_rewrite_reason"] = "episode_changed"
        return stats

    cap = state.get("last_full_frame")
    if not cap:
        stats["sliding_rewrite_reason"] = "no_captured_frame"
        return stats

    img_path = str(cap.get("img_path") or "")
    frame_seq = int(cap.get("frame_seq", -1))
    compressed = _normalise_kv_compressed_keys(state.get("kv_compressed_paths") or ())
    compressed_key = _kv_compressed_key_from_capture(cap)
    if compressed_key is not None and compressed_key in compressed:
        stats["sliding_rewrite_reason"] = "already_compressed"
        stats["sliding_rewrite_ok"] = True
        return stats

    cached_ids = list(state.get("cached_input_ids") or [])
    image_token_id = _image_token_id(inner)
    span = _last_visual_span(cached_ids, image_token_id)
    if span is None:
        stats["sliding_rewrite_reason"] = "no_visual_span"
        return stats
    vis_start, vis_end = span
    n_frame = vis_end - vis_start

    from lmms_eval.models.geopact_vl_hook import rectify_history_frame_from_full
    from lmms_eval.models.geopact_step_timing import get_step_ledger

    ledger = get_step_ledger(model)
    episode_key = str(state.get("episode_key") or "")

    # COC (C v4): certificate-optimal contraction — decide S_hist + the
    # donor->host map from the live contextualized value rows BEFORE rectify
    # (which otherwise applies the value-blind rank-strata rule). The keep set
    # is handed to rectify via a per-frame override on the model; the host map
    # feeds the absorption plan below. Prefetch is bypassed under COC (its
    # precomputed keep set used the strata rule).
    coc_stash = None
    use_coc = False
    try:
        from lmms_eval.models.geopact_c_absorb import coc_enabled, plan_coc_contract

        use_coc = coc_enabled()
        if use_coc:
            coc_t0 = time.perf_counter()
            from lmms_eval.models.geopact_nap import lookup_nap_order
            from lmms_eval.models.geopact_vl_hook import _sticky_mask_key

            c_key = (
                _sticky_mask_key(episode_key, img_path, model=inner)
                if episode_key and img_path
                else None
            )
            c_rec = lookup_nap_order(inner, c_key)
            c_order = list((c_rec or {}).get("order") or ())
            if (
                c_rec is not None
                and str(c_rec.get("kind") or "greedy") == "greedy"
                and len(c_order) == int(n_frame)
            ):
                c_n_full = int(c_rec.get("n_full") or 0) or int(n_frame)
                c_k_hist = max(1, int(round(c_n_full * float(ratio))))
                if c_k_hist < int(n_frame):
                    coc_stash = plan_coc_contract(
                        _legacy_cache_tensors(state["past_key_values"]),
                        vis_start=int(vis_start),
                        n_vis=int(n_frame),
                        scur_sorted=sorted(int(x) for x in c_order),
                        k_hist=c_k_hist,
                    )
            if coc_stash is not None:
                inner._geopact_coc_keep_override = {
                    "key": c_key,
                    "keep_full": list(coc_stash["keep_full"]),
                }
                stats["coc_plan_ms"] = round(
                    (time.perf_counter() - coc_t0) * 1000.0, 2
                )
                stats.update(coc_stash.get("audit") or {})
            else:
                inner._geopact_coc_keep_override = None
    except Exception as exc:
        coc_stash = None
        inner._geopact_coc_keep_override = None
        eval_logger.warning("GeoPACT COC plan failed: %s", exc)

    if ledger is not None:
        ledger.start_bucket("sliding_rewrite_rectify_ms")
    rectify_t0 = time.perf_counter()
    prefetched = None
    if not use_coc:
        try:
            from lmms_eval.models.geopact_rectify_prefetch import (
                try_consume_rectify_prefetch,
            )

            prefetched = try_consume_rectify_prefetch(
                model,
                img_path=img_path,
                frame_seq=frame_seq,
                ratio=float(ratio),
                policy=str(policy or ""),
                episode_key=episode_key,
                n_frame=n_frame,
            )
        except Exception:
            prefetched = None
    if prefetched is not None and prefetched[3].get("rectify_ok"):
        kept_visual, rectified_deepstack, local_keep, rectify_stats = prefetched
    else:
        kept_visual, rectified_deepstack, local_keep, rectify_stats = (
            rectify_history_frame_from_full(
                model,
                cap=cap,
                ratio=float(ratio),
                policy=str(policy or ""),
                episode_key=episode_key,
                n_frame=n_frame,
            )
        )
    rectify_ms = round((time.perf_counter() - rectify_t0) * 1000.0, 2)
    if ledger is not None:
        # start/stop already accumulated the elapsed time; adding rectify_ms
        # again would double-count the bucket in the exclusive sum.
        ledger.stop_bucket("sliding_rewrite_rectify_ms")
    stats["sliding_rewrite_rectify_ms"] = rectify_ms
    stats.update({f"rectify_{k}": v for k, v in rectify_stats.items()})
    if ledger is not None:
        # Surface rectify sub-stage timings/labels in step_profiles via the
        # audit channel (scalars only; no effect on exclusive bucket sums).
        audit = {
            k if k.startswith("rectify_") else f"rectify_{k}": v
            for k, v in rectify_stats.items()
            if isinstance(v, (int, float, bool, str)) and v != ""
        }
        audit["rectify_prefetched"] = bool(prefetched is not None)
        ledger.set_audit(**audit)
    if ledger is not None:
        # Account the prepare-tail work (tail embeds, virtual MRoPE cache, KV
        # crop, state rebuild) instead of letting it fall into misc_ms. The
        # nested mrope bucket auto-closes this one; we re-open it after.
        ledger.start_bucket("sliding_rewrite_ms")
    if rectify_stats.get("rectify_reason") == "ratio_noop":
        if compressed_key is not None:
            compressed.add(compressed_key)
            noop_state = dict(state)
            noop_state["kv_compressed_paths"] = compressed
            _set_dialogue_state(inner, noop_state)
        stats["sliding_rewrite_ok"] = True
        stats["sliding_rewrite_reason"] = "ratio_noop"
        if ledger is not None:
            ledger.stop_bucket("sliding_rewrite_ms")
        return stats
    if not rectify_stats.get("rectify_ok"):
        stats["sliding_rewrite_reason"] = rectify_stats.get("rectify_reason", "rectify_failed")
        if ledger is not None:
            ledger.stop_bucket("sliding_rewrite_ms")
        return stats

    # NAP stage 2: the cached span holds the S_cur subset (row coordinates),
    # while the virtual MRoPE stream still carries the full-frame run. The
    # rectify prefix branch reports the full-frame keep + virtual frame length
    # so the origpos subset picks the original grid positions of S_hist.
    nap_prefix = rectify_stats.get("rectify_selector") == "nap_prefix"
    mrope_n_frame = int(n_frame)
    mrope_local_keep: List[int] = list(local_keep)
    if nap_prefix:
        mrope_n_frame = int(rectify_stats.get("nap_n_full") or n_frame)
        mrope_local_keep = [
            int(x) for x in (rectify_stats.get("nap_full_local_keep") or local_keep)
        ]

    past_kv = state["past_key_values"]
    embed_fn = inner.get_input_embeddings()
    try:
        device = embed_fn.weight.device
    except Exception:
        device = kept_visual.device
    dtype = kept_visual.dtype

    tail_ids = cached_ids[vis_end:]
    tail_embeds = None
    if tail_ids:
        tail_tensor = torch.tensor([tail_ids], dtype=torch.long, device=device)
        tail_embeds = embed_fn(tail_tensor).to(dtype=dtype)

    if tail_embeds is not None:
        rewrite_embeds = torch.cat([kept_visual.unsqueeze(0), tail_embeds], dim=1)
    else:
        rewrite_embeds = kept_visual.unsqueeze(0)
    rewrite_len = int(rewrite_embeds.shape[1])

    state = _ensure_virtual_mrope_cache(inner, state)
    cached_full_pos = _get_cached_virtual_full_mrope(state, device=device)

    if ledger is not None:
        ledger.start_bucket("sliding_rewrite_mrope_ms")
    mrope_t0 = time.perf_counter()
    rewrite_pos, rewrite_rope_deltas, mrope_len = _build_sliding_rewrite_mrope(
        inner,
        state,
        vis_start=vis_start,
        n_frame=mrope_n_frame,
        tail_ids=tail_ids,
        local_keep=mrope_local_keep,
        image_token_id=image_token_id,
        device=device,
        cached_full_pos=cached_full_pos,
    )
    mrope_build_ms = round((time.perf_counter() - mrope_t0) * 1000.0, 2)
    if ledger is not None:
        ledger.stop_bucket("sliding_rewrite_mrope_ms")
    stats["sliding_rewrite_mrope_ms"] = mrope_build_ms
    if ledger is not None:
        ledger.start_bucket("sliding_rewrite_ms")
    if rewrite_pos is None or mrope_len != rewrite_len:
        stats["sliding_rewrite_reason"] = "mrope_failed"
        if ledger is not None:
            ledger.stop_bucket("sliding_rewrite_ms")
        return stats

    lm = _carrier_text_model(inner)
    cache_config = getattr(lm, "config", getattr(inner, "config", None))
    cropped = _crop_past_key_values(past_kv, int(vis_start), config=cache_config)
    if cropped is None:
        stats["sliding_rewrite_reason"] = "crop_failed"
        if ledger is not None:
            ledger.stop_bucket("sliding_rewrite_ms")
        return stats

    deepstack_visual_embeds = None
    if rectified_deepstack:
        deepstack_visual_embeds = []
        for layer in rectified_deepstack:
            deepstack_visual_embeds.append(layer.to(device=device, dtype=dtype))

    new_visual_ids = [image_token_id] * len(local_keep)
    ids_after_rewrite = cached_ids[:vis_start] + new_visual_ids + tail_ids

    new_state = dict(state)
    new_state["past_key_values"] = cropped
    new_state["cached_input_ids"] = list(cached_ids[:vis_start])
    new_state["prompt_token_len"] = int(vis_start)
    new_state.pop("cached_boundary_position_ids", None)
    new_state["_sliding_rewrite_active"] = True
    new_state["_sliding_rewrite_pending"] = {
        "vis_start": int(vis_start),
        "vis_end": int(vis_end),
        "n_frame": int(n_frame),
        "rewrite_len": int(rewrite_len),
        "rewrite_embeds": rewrite_embeds.detach(),
        "position_ids": rewrite_pos.detach(),
        "rope_deltas": rewrite_rope_deltas.detach() if rewrite_rope_deltas is not None else None,
        "deepstack_visual_embeds": deepstack_visual_embeds,
        "n_visual_kept": len(local_keep),
        "tail_ids": list(tail_ids),
        "ids_after_rewrite": ids_after_rewrite,
        "img_path": img_path,
        "frame_seq": int(frame_seq),
        "local_keep": list(local_keep),
    }
    # Module C v3 (NAP-History Absorb): before the frame's cached rows die in
    # the crop, sum the *contextualized* value rows of the dropped tokens onto
    # their absorption hosts (rank-monotone tree resolved against S_hist,
    # balanced loads => kappa -> 1). The in-place host value rewrite happens
    # after the merged re-prefill in _prefill_merged_sliding_rewrite_
    # before_generate; keys / MRoPE / row count / FA2 path stay untouched.
    try:
        from lmms_eval.models.geopact_c_absorb import (
            absorb_lambda_hist,
            absorb_tau,
            c_absorb_enabled,
            collect_donor_value_sums,
            plan_from_host_map,
            plan_history_absorb,
        )

        lam_hist = absorb_lambda_hist()
        absorb_engaged = lam_hist > 0.0 or absorb_tau() is not None
        if c_absorb_enabled() and absorb_engaged and nap_prefix:
            absorb_t0 = time.perf_counter()
            from lmms_eval.models.geopact_nap import lookup_nap_order
            from lmms_eval.models.geopact_vl_hook import _sticky_mask_key

            a_key = (
                _sticky_mask_key(episode_key, img_path, model=inner)
                if episode_key and img_path
                else None
            )
            a_rec = lookup_nap_order(inner, a_key)
            keep_full_a = [int(x) for x in (rectify_stats.get("nap_full_local_keep") or [])]
            scur_a = [int(x) for x in (rectify_stats.get("nap_scur_sorted") or [])]
            plan = None
            if (
                coc_stash is not None
                and keep_full_a
                and scur_a
                and sorted(coc_stash["keep_full"]) == sorted(keep_full_a)
            ):
                # COC path: the value-medoid grouping that chose S_hist also
                # fixes the donor->host map — selection and absorption are
                # optimized against the same certificate.
                plan = plan_from_host_map(
                    keep_full=keep_full_a,
                    scur_sorted=scur_a,
                    vis_start=int(vis_start),
                    host_map=coc_stash["host_map"],
                )
            if plan is None:
                plan = (
                    plan_history_absorb(
                        rec=a_rec,
                        keep_full=keep_full_a,
                        scur_sorted=scur_a,
                        vis_start=int(vis_start),
                    )
                    if a_rec and keep_full_a and scur_a
                    else None
                )
            if plan is not None:
                from lmms_eval.models.geopact_c_absorb import (
                    absorb_value_mode,
                    key_audit_enabled,
                    key_mismatch_audit,
                )

                legacy_pre_crop = _legacy_cache_tensors(past_kv)
                # 'delta'/'oldhost' modes and the drift audit need the OLD
                # (pre-crop) host value rows; collect them in the same pass.
                want_old = absorb_value_mode() != "absolute" or key_audit_enabled()
                sums, mass_t, coll_audit = collect_donor_value_sums(
                    legacy_pre_crop,
                    donor_seq=plan["donor_seq"],
                    host_pos=plan["host_pos"],
                    n_hosts=plan["n_hosts"],
                    host_seq=plan["host_seq"] if want_old else None,
                )
                host_old_layers = coll_audit.pop("host_old", None)

                if key_audit_enabled():
                    # W-A4: measured per-layer post-MRoPE donor/host key
                    # mismatch on the live cache rows (calibration evidence
                    # for the spatial-locality heuristic).
                    stats.update(
                        key_mismatch_audit(
                            legacy_pre_crop,
                            donor_seq=plan["donor_seq"],
                            host_seq=plan["host_seq"],
                            donor_full=plan["donor_full"],
                            host_full=plan["host_full"],
                            grid_w=int(plan.get("grid_w") or 0),
                        )
                    )
                if sums:
                    new_state["_sliding_rewrite_pending"]["absorb"] = {
                        "host_pos": list(plan["host_pos"]),
                        "sums": sums,
                        "mass": mass_t,
                        "lam": float(lam_hist),
                        "host_old": host_old_layers,
                        "mode": absorb_value_mode(),
                    }
                    stats.update(plan["audit"])
                    stats.update(coll_audit)
            else:
                # Not an expected state on absorb arms: the frame entered
                # under a non-greedy/legacy record, so C is silently off for
                # this frame. Warn (once per process) so a mis-wired arm can
                # never masquerade as "absorb ran but had no effect".
                stats["absorb_plan"] = "missing"
                if not getattr(inner, "_geopact_absorb_missing_warned", False):
                    inner._geopact_absorb_missing_warned = True
                    eval_logger.warning(
                        "GeoPACT C-Absorb enabled but no absorb plan for this "
                        "frame (record kind/parents missing) — absorb is OFF "
                        "for such frames; further occurrences logged only in "
                        "stats.absorb_plan"
                    )
            stats["absorb_plan_ms"] = round(
                (time.perf_counter() - absorb_t0) * 1000.0, 2
            )
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"):
            raise
        eval_logger.warning(f"GeoPACT history absorb plan failed: {exc}")
        stats["absorb_plan"] = f"error:{exc}"

    # Module C v5 (CRS Ghost-KV): partition this frame's deleted cache rows
    # into certified regions and stage per-layer ghost K/V means (append-only).
    # Ghost ids enter
    # ids_after_rewrite NOW so the capture boundary that qwen3_vl_gui derives
    # from pending stays 1:1 with the post-splice cache; the physical splice
    # happens after the merged prefill.
    try:
        from lmms_eval.models.geopact_crs_ghost import crs_ghost_enabled, plan_crs_ghosts

        if crs_ghost_enabled() and nap_prefix:
            ghost = plan_crs_ghosts(
                _legacy_cache_tensors(past_kv),
                vis_start=int(vis_start),
                n_vis=int(n_frame),
                local_keep=local_keep,
            )
            if ghost is not None:
                pend = new_state["_sliding_rewrite_pending"]
                n_kept = len(local_keep)
                ids_g = list(pend["ids_after_rewrite"])
                at = int(vis_start) + n_kept
                pend["ids_after_rewrite"] = (
                    ids_g[:at] + [image_token_id] * int(ghost["g"]) + ids_g[at:]
                )
                pend["crs_ghost"] = ghost
                pend["_crs_ghost_g"] = int(ghost["g"])
                stats.update(ghost["audit"])
            else:
                stats["crs_ghost_plan"] = "degenerate"
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"):
            raise
        eval_logger.warning(f"GeoPACT CRS ghost plan failed: {exc}")
        stats["crs_ghost_plan"] = f"error:{exc}"

    # Module C (QASMR): turn the deleted tokens of this frame into KV relays.
    # Two sources, one aggregation:
    #   - cache rows (GEOPACT_C_RELAY_FROM_CACHE=1): the dropped tokens were
    #     really prefilled this/last step, so their contextualized KV rows sit
    #     in the dialogue cache at [vis_start + row]; slice them right before
    #     the crop — no full-frame snapshot clone ever happens.
    #   - legacy snapshot (default): per-layer clone captured at the previous
    #     finalize (spec docs/geopact_module_c_qasmr.md §6.1), consumed and
    #     freed here whether or not relay construction succeeds.
    try:
        from lmms_eval.models.geopact_spatial_relay import (
            build_frame_relays_from_cache_rows,
            build_frame_relays_from_capture,
            relay_from_cache_enabled,
            relay_kv_mode_active,
            store_frame_relays,
        )

        if relay_kv_mode_active():
            relay_t0 = time.perf_counter()
            from lmms_eval.models.geopact_vl_hook import _get_grid

            gh, gw, _grid_ok = _get_grid(cap.get("grid_row"), int(mrope_n_frame))
            if relay_from_cache_enabled():
                keep_rows = {int(i) for i in local_keep}
                dropped_rows = [i for i in range(int(n_frame)) if i not in keep_rows]
                if nap_prefix:
                    scur_sorted = [
                        int(x) for x in (rectify_stats.get("nap_scur_sorted") or [])
                    ]
                    dropped_full = [scur_sorted[i] for i in dropped_rows]
                else:
                    dropped_full = list(dropped_rows)
                qd_export = cap.get("qd_export") or {}
                fr = build_frame_relays_from_cache_rows(
                    _legacy_cache_tensors(past_kv),
                    seq_positions=[int(vis_start) + int(r) for r in dropped_rows],
                    full_positions=dropped_full,
                    n_full=int(mrope_n_frame),
                    grid_h=int(gh),
                    grid_w=int(gw),
                    img_path=img_path,
                    frame_seq=int(frame_seq),
                    qd_export=qd_export,
                )
                stats["sliding_rewrite_relay_source"] = "cache_rows"
            else:
                fr = build_frame_relays_from_capture(
                    cap,
                    local_keep=local_keep,
                    n_frame=int(n_frame),
                    grid_h=int(gh),
                    grid_w=int(gw),
                )
                stats["sliding_rewrite_relay_source"] = "snapshot"
            if fr is not None:
                store_frame_relays(inner, new_state, fr)
                stats["sliding_rewrite_relays"] = int(fr.n_relays)
                stats["sliding_rewrite_relay_cert"] = fr.audit.get("relay_kv_cert_sum")
                stats["sliding_rewrite_relay_dropped"] = fr.audit.get(
                    "relay_dropped_clusters"
                )
                stats["sliding_rewrite_relay_dropped_mass"] = fr.audit.get(
                    "relay_dropped_mass_frac"
                )
                stats["sliding_rewrite_relay_eta_sat"] = fr.audit.get(
                    "relay_cert_eta_sat_frac"
                )
                stats["sliding_rewrite_relay_eta_mean"] = fr.audit.get(
                    "relay_cert_eta_mean"
                )
            else:
                stats["sliding_rewrite_relays"] = 0
                stats["sliding_rewrite_relay_reason"] = "no_snapshot_or_no_deleted"
            stats["sliding_rewrite_relay_ms"] = round(
                (time.perf_counter() - relay_t0) * 1000.0, 2
            )
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"):
            raise
        eval_logger.warning(f"GeoPACT relay build failed: {exc}")
        stats["sliding_rewrite_relay_reason"] = f"error:{exc}"
    finally:
        cap.pop("layer_kv", None)
    _set_dialogue_state(inner, new_state)

    stats["sliding_rewrite_ok"] = True
    stats["sliding_rewrite_reason"] = "pending"
    stats["sliding_rewrite_vis_start"] = int(vis_start)
    stats["sliding_rewrite_len"] = int(rewrite_len)
    stats["sliding_rewrite_kept"] = len(local_keep)
    if ledger is not None:
        ledger.stop_bucket("sliding_rewrite_ms")
    return stats


def _merged_sliding_rewrite_visual_meta(
    pending: Dict[str, Any],
    suffix: Dict[str, Any],
    suffix_prefill: Optional[Dict[str, Any]],
    *,
    prefill_len: int,
    rewrite_len: int,
    image_token_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Visual masks + deepstack for merged rewrite+suffix prefill."""
    rewrite_n_vis = int(pending.get("n_visual_kept") or 0)
    rewrite_ds = pending.get("deepstack_visual_embeds")
    device = pending["rewrite_embeds"].device
    visual_pos_masks = torch.zeros((1, prefill_len), dtype=torch.bool, device=device)
    if rewrite_n_vis > 0:
        visual_pos_masks[0, :rewrite_n_vis] = True

    suffix_part_len = prefill_len - rewrite_len
    suffix_ds_out: Optional[List[torch.Tensor]] = None
    if suffix_prefill is not None and suffix_part_len > 0:
        _, suffix_ds_out = _suffix_prefill_visual_meta(
            suffix,
            suffix_prefill,
            prefill_len=suffix_part_len,
            image_token_id=image_token_id,
        )
        image_mask = suffix_prefill.get("image_mask")
        if image_mask is not None:
            sm = image_mask[:, :suffix_part_len].contiguous()
            if sm.dim() == 3:
                sm = sm[..., 0]
            if int(sm.shape[-1]) >= suffix_part_len:
                visual_pos_masks[0, rewrite_len : rewrite_len + suffix_part_len] = sm[
                    0, :suffix_part_len
                ]

    deepstack_out: Optional[List[torch.Tensor]] = None
    if rewrite_ds or suffix_ds_out:
        deepstack_out = []
        n_rewrite_layers = len(rewrite_ds) if rewrite_ds else 0
        n_suffix_layers = len(suffix_ds_out) if suffix_ds_out else 0
        n_layers = max(n_rewrite_layers, n_suffix_layers)
        for layer_i in range(n_layers):
            parts = []
            if rewrite_ds and layer_i < len(rewrite_ds):
                parts.append(rewrite_ds[layer_i])
            if suffix_ds_out and layer_i < len(suffix_ds_out):
                parts.append(suffix_ds_out[layer_i])
            if parts:
                deepstack_out.append(torch.cat(parts, dim=0))

    if not rewrite_ds and deepstack_out is None:
        return None, None
    return visual_pos_masks, deepstack_out


@torch.inference_mode()
def _prefill_merged_sliding_rewrite_before_generate(
    model,
    suffix: Dict[str, Any],
    *,
    pending: Dict[str, Any],
    past_kv: Any,
    past_len: int,
    suffix_prefill: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Single LM forward for full rewrite segment + suffix (sliding-window rewrite)."""
    rewrite_embeds = pending["rewrite_embeds"]
    rewrite_len = int(pending["rewrite_len"])
    suffix_embeds = suffix["inputs_embeds"]
    device = suffix_embeds.device
    dtype = suffix_embeds.dtype
    merged_embeds = torch.cat(
        [rewrite_embeds.to(device=device, dtype=dtype), suffix_embeds],
        dim=1,
    )
    merged_len = int(merged_embeds.shape[1])
    stats: Dict[str, Any] = {
        "sliding_rewrite_lm_ms": 0.0,
        "sliding_rewrite_lm_tokens": 0,
        "dialogue_kv_batch_prefill_ms": 0.0,
        "dialogue_kv_batch_prefill_tokens": 0,
    }
    if merged_len <= 1:
        suffix["_geopact_pinned_cache_start"] = int(past_len)
        return past_kv, stats

    prefill_len = merged_len
    rewrite_pos = pending["position_ids"].to(device=device)
    suffix_pos = suffix.get("position_ids")
    if suffix_pos is None:
        raise RuntimeError("GeoPACT sliding rewrite: missing suffix position_ids")
    merged_pos = _concat_sliding_merged_mrope(rewrite_pos, suffix_pos.to(device=device))
    merged_pos_prefill = _slice_position_ids(merged_pos, 0, prefill_len)

    inner = _inner_model(model)
    lm = _carrier_text_model(inner)
    image_token_id = _image_token_id(inner)

    prefill_cache_pos = torch.arange(
        past_len,
        past_len + prefill_len,
        device=device,
        dtype=torch.long,
    )
    prefill_attn = torch.ones(
        (1, past_len + prefill_len),
        dtype=suffix["attention_mask"].dtype,
        device=device,
    )
    visual_pos_masks, deepstack_visual_embeds = _merged_sliding_rewrite_visual_meta(
        pending,
        suffix,
        suffix_prefill,
        prefill_len=prefill_len,
        rewrite_len=rewrite_len,
        image_token_id=image_token_id,
    )

    rope_deltas = suffix.get("rope_deltas")
    if rope_deltas is not None:
        inner.rope_deltas = rope_deltas.to(device=device)

    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    batch_t0 = time.perf_counter()
    if ledger is not None:
        # Switch out of dialogue_kv_prepare_ms for the LM forward itself.
        ledger.start_bucket("sliding_rewrite_lm_ms")
    lm_kwargs: Dict[str, Any] = {
        "inputs_embeds": merged_embeds[:, :prefill_len].contiguous(),
        "past_key_values": past_kv,
        "attention_mask": prefill_attn,
        "cache_position": prefill_cache_pos,
        "use_cache": True,
        "return_dict": True,
        "position_ids": merged_pos_prefill,
    }
    # Qwen2.5-VL text LLM does not accept deepstack kwargs; qwen3 path unchanged.
    carrier_has_deepstack = not _is_qwen25_carrier(inner)
    if carrier_has_deepstack and visual_pos_masks is not None:
        lm_kwargs["visual_pos_masks"] = visual_pos_masks
    if carrier_has_deepstack and deepstack_visual_embeds is not None:
        from lmms_eval.models.geopact_vl_prepare import validate_deepstack_alignment

        n_vis = int(visual_pos_masks.sum().item()) if visual_pos_masks is not None else 0
        validate_deepstack_alignment(
            deepstack_visual_embeds,
            visual_pos_masks,
            n_visual_kept=n_vis,
            where="sliding_merged_prefill",
        )
        lm_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

    outputs = lm(**lm_kwargs)
    if ledger is not None:
        ledger.stop_bucket("sliding_rewrite_lm_ms")
    batch_prefill_ms = round((time.perf_counter() - batch_t0) * 1000.0, 2)
    running_kv = outputs.past_key_values
    new_len = int(running_kv.get_seq_length()) if running_kv is not None else 0
    expected_len = int(past_len) + int(prefill_len)
    if new_len != expected_len:
        raise RuntimeError(
            "GeoPACT sliding rewrite: merged prefill length mismatch "
            f"got={new_len} expected={expected_len}"
        )

    # Module C v3 (NAP-History Absorb): rewrite surviving value rows in place
    # with their absorbed donor mass. The kept visual rows sit at cache
    # positions [past_len, past_len + n_kept); keys are untouched.
    absorb = pending.get("absorb")
    if absorb:
        from lmms_eval.models.geopact_c_absorb import apply_history_absorb

        absorb_t0 = time.perf_counter()
        a_stats = apply_history_absorb(
            running_kv,
            seg_start=int(past_len),
            host_pos=absorb["host_pos"],
            sums=absorb["sums"],
            mass=absorb["mass"],
            lam=float(absorb["lam"]),
            host_old=absorb.get("host_old"),
            mode=absorb.get("mode"),
        )
        apply_ms = (time.perf_counter() - absorb_t0) * 1000.0
        a_stats["absorb_apply_ms"] = round(apply_ms, 2)
        stats.update(a_stats)
        pending.pop("absorb", None)
        # Attribute the apply to the sliding-rewrite exclusive bucket (it runs
        # after the lm bucket closed and previously leaked into misc/gap time,
        # hiding part of Module C's cost from the exclusive profile).
        if ledger is not None:
            ledger.add_ms("sliding_rewrite_ms", apply_ms)

    # Module C v5 (CRS Ghost-KV): splice the staged ghost rows right after the
    # frame's kept visual rows. The merged forward above never saw them (its
    # suffix rows carry a one-step lag, doc §5.4); every physical index below
    # (bootstrap crop, cache_position, pinned start) shifts by crs_g. The ids
    # side was already extended at plan time, so capture/finalize boundaries
    # match the spliced cache 1:1.
    crs_g = 0
    ghost = pending.get("crs_ghost")
    if ghost:
        from lmms_eval.models.geopact_crs_ghost import apply_crs_ghosts

        lm_cfg_g = getattr(lm, "config", getattr(inner, "config", None))
        n_kept_vis = int(pending.get("n_visual_kept") or 0)
        running_kv, g_stats = apply_crs_ghosts(
            running_kv,
            insert_at=int(past_len) + n_kept_vis,
            k_layers=ghost["k_layers"],
            v_layers=ghost["v_layers"],
            config=lm_cfg_g,
        )
        stats.update(g_stats)
        if g_stats.get("crs_ghost_applied"):
            crs_g = int(ghost["g"])
        else:
            raise RuntimeError(
                "GeoPACT CRS ghost splice failed after ids were extended: "
                f"{g_stats.get('crs_ghost_reason')}"
            )
        pending.pop("crs_ghost", None)
        if ledger is not None:
            ledger.add_ms(
                "sliding_rewrite_ms", float(g_stats.get("crs_ghost_apply_ms") or 0.0)
            )

    lm_cfg = getattr(lm, "config", getattr(inner, "config", None))
    bootstrap_kv_len = int(past_len) + int(merged_len) - 1 + crs_g
    crop_t0 = time.perf_counter()
    # In-place: running_kv is the rewrite-cropped state cache extended by the
    # merged prefill; finalize commits it via direct_pkv, no other consumer.
    running_kv = _crop_past_key_values_owned(running_kv, bootstrap_kv_len, config=lm_cfg)
    if running_kv is None:
        raise RuntimeError(
            "GeoPACT sliding rewrite: failed to crop KV for decode bootstrap "
            f"target_len={bootstrap_kv_len}"
        )
    if ledger is not None:
        ledger.add_ms(
            "kv_bootstrap_crop_ms", (time.perf_counter() - crop_t0) * 1000.0
        )
        # Tail work (suffix rebuild, position slices) back under prepare.
        ledger.start_bucket("dialogue_kv_prepare_ms")

    last = merged_len - 1
    suffix_only_len = int(suffix_embeds.shape[1])
    last_suffix_idx = suffix_only_len - 1
    if suffix.get("input_ids") is not None:
        suffix["input_ids"] = suffix["input_ids"][:, last_suffix_idx:].contiguous()
        suffix.pop("inputs_embeds", None)
    elif suffix.get("inputs_embeds") is not None:
        suffix["inputs_embeds"] = suffix["inputs_embeds"][:, last_suffix_idx:].contiguous()
    kv_len = int(running_kv.get_seq_length()) if running_kv is not None else int(past_len + crs_g + merged_len - 1)
    suffix["attention_mask"] = torch.ones(
        (1, kv_len + 1),
        dtype=suffix["attention_mask"].dtype,
        device=device,
    )
    suffix["cache_position"] = torch.arange(
        past_len + crs_g + last,
        past_len + crs_g + merged_len,
        device=device,
        dtype=torch.long,
    )
    if merged_pos is not None:
        suffix["position_ids"] = _slice_position_ids(merged_pos, last, merged_len)
        suffix_region = _slice_position_ids(merged_pos, rewrite_len, merged_len)
        inner._geopact_dialogue_last_suffix_position_ids = suffix_region.detach().cpu().clone()
    suffix["past_key_values"] = running_kv
    suffix["_geopact_pinned_cache_start"] = int(past_len + crs_g + last)
    suffix["_geopact_decode_only_generate"] = True

    stats["sliding_rewrite_lm_ms"] = batch_prefill_ms
    stats["sliding_rewrite_lm_tokens"] = int(prefill_len)
    stats["dialogue_kv_batch_prefill_ms"] = batch_prefill_ms
    stats["dialogue_kv_batch_prefill_tokens"] = int(prefill_len)
    # Forward wall is counted once, via the sliding_rewrite_lm bucket switch
    # above; the stats dict keeps the batch-prefill alias for readability only.
    return running_kv, stats


def _finalize_sliding_rewrite_state(
    inner,
    state: Dict[str, Any],
    pending: Dict[str, Any],
    *,
    suffix_ids: List[int],
) -> Dict[str, Any]:
    """Commit rewrite compact ids into dialogue state after merged prefill."""
    ids_after_rewrite = list(pending.get("ids_after_rewrite") or [])
    vis_start = int(pending.get("vis_start") or 0)
    n_kept = int(pending.get("n_visual_kept") or 0)
    tail_len = len(pending.get("tail_ids") or [])
    # CRS Ghost-KV rows (Module C v5) live inside the frame segment: the ids
    # were extended at plan time, the cache at merged-prefill splice time.
    crs_g = int(pending.get("_crs_ghost_g") or 0)
    if ids_after_rewrite and len(ids_after_rewrite) != vis_start + n_kept + crs_g + tail_len:
        raise RuntimeError(
            "GeoPACT sliding rewrite: committed ids length invariant broken "
            f"ids={len(ids_after_rewrite)} vis_start={vis_start} "
            f"kept={n_kept} ghosts={crs_g} tail={tail_len}"
        )
    img_path = str(pending.get("img_path") or "")
    compressed_key = _kv_compressed_frame_key(img_path, pending.get("frame_seq", -1))
    new_state = dict(state)
    new_state["cached_input_ids"] = ids_after_rewrite
    new_state["prompt_token_len"] = len(ids_after_rewrite)
    new_compressed = _normalise_kv_compressed_keys(state.get("kv_compressed_paths") or ())
    if compressed_key is not None:
        new_compressed.add(compressed_key)
    new_state["kv_compressed_paths"] = new_compressed
    new_state.pop("last_full_frame", None)
    new_state.pop("_sliding_rewrite_pending", None)
    new_state.pop("_sliding_rewrite_active", None)
    new_state.pop("cached_boundary_position_ids", None)
    direct_pkv = getattr(inner, "_geopact_dialogue_direct_pkv", None)
    if direct_pkv is not None:
        new_state["past_key_values"] = direct_pkv
    _set_dialogue_state(inner, new_state)
    return new_state




def _capture_last_full_frame(
    inner,
    prefill: Optional[Dict[str, Any]],
    *,
    stored_ids: List[int],
    image_paths: List[str],
    image_token_id: int,
) -> Optional[Dict[str, Any]]:
    """Snapshot the last (current) full-frame visual embeds + deepstack for reuse.

    The captured tensors let the next step prune this frame and rewrite its KV
    segment without reloading pixels or re-running the vision tower.
    """
    if not prefill:
        return None
    embeds = prefill.get("inputs_embeds")
    if embeds is None:
        return None
    span = _last_visual_span(stored_ids, image_token_id)
    if span is None:
        return None
    vis_start, vis_end = span
    n_frame = vis_end - vis_start
    if n_frame <= 0 or int(embeds.shape[1]) < vis_end:
        return None

    visual_embeds = embeds[0, vis_start:vis_end, :].detach().clone()

    # Deepstack is visual-row indexed across all frames; the current frame is the
    # last block of visual rows.
    deepstack = []
    ds = prefill.get("deepstack_image_embeds")
    if ds:
        n_vis_total = _count_visual_tokens(
            prefill.get("input_ids", torch.tensor([stored_ids])),
            0,
            len(stored_ids),
            image_token_id,
        ) if prefill.get("input_ids") is not None else None
        for layer in ds:
            rows = int(layer.shape[0])
            if rows < n_frame:
                return None
            deepstack.append(layer[rows - n_frame : rows, :].detach().clone())

    grid_row = None
    grid = prefill.get("image_grid_thw")
    if grid is not None and int(grid.shape[0]) >= 1:
        grid_row = grid[-1:].detach().clone()

    frame_seq = len(image_paths) - 1 if image_paths else -1
    img_path = image_paths[-1] if image_paths else ""
    pil_image = getattr(inner, "_geopact_last_image", None)
    return {
        "img_path": img_path,
        "frame_seq": int(frame_seq),
        "visual_embeds": visual_embeds,
        "deepstack": deepstack,
        "grid_row": grid_row,
        "pil_image": pil_image,
    }


def finalize_dialogue_kv_after_generate(
    model,
    doc: Dict,
    *,
    prompt_token_len: int,
    prompt_input_ids: List[int],
    capture: Optional[DialogueKVCapture],
    image_grid_thw: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Truncate decode tokens from KV; store prompt boundary for the next step."""
    stats: Dict[str, Any] = {"dialogue_kv_cache_updated": False}
    if not dialogue_kv_enabled():
        return stats

    inner = _inner_model(model)
    episode_key = session_episode_key(doc)
    step_index = session_step_index(doc)
    pkv = capture.captured if capture is not None else None
    capture_source = "hook"
    if pkv is None:
        pkv = getattr(inner, "_geopact_dialogue_direct_pkv", None)
        if pkv is not None:
            capture_source = "direct_pkv"
            inner._geopact_dialogue_direct_pkv = None
    stats["dialogue_kv_capture_source"] = capture_source
    stats["dialogue_kv_capture_hit"] = pkv is not None
    if pkv is None:
        clear_session_kv(model)
        raise RuntimeError(
            "GeoPACT dialogue KV: capture_miss — prefill KV was not captured at prompt boundary"
        )

    if len(prompt_input_ids) < int(prompt_token_len):
        clear_session_kv(model)
        raise RuntimeError(
            "GeoPACT dialogue KV: prompt_input_ids shorter than prompt_token_len "
            f"({len(prompt_input_ids)} < {prompt_token_len})"
        )

    lm = _carrier_text_model(inner)
    # In-place: pkv is either direct_pkv (post-generate, decode finished) or the
    # hook-captured clone; both exclusively owned here. The live generate cache
    # is never passed to this crop.
    cropped = _crop_past_key_values_owned(
        pkv,
        int(prompt_token_len),
        config=getattr(lm, "config", getattr(inner, "config", None)),
    )
    if cropped is None:
        clear_session_kv(model)
        raise RuntimeError(
            f"GeoPACT dialogue KV: crop_failed at prompt_token_len={prompt_token_len}"
        )

    try:
        actual_kv_len = int(cropped.get_seq_length())
    except Exception as exc:
        clear_session_kv(model)
        raise RuntimeError(
            f"GeoPACT dialogue KV: unable to read cropped KV length: {exc}"
        ) from exc
    stored_ids = list(prompt_input_ids[: int(prompt_token_len)])
    if actual_kv_len != int(prompt_token_len):
        clear_session_kv(model)
        raise RuntimeError(
            "GeoPACT dialogue KV: cropped KV length mismatch "
            f"kv={actual_kv_len} prompt_token_len={prompt_token_len}"
        )
    if len(stored_ids) != actual_kv_len:
        clear_session_kv(model)
        raise RuntimeError(
            "GeoPACT dialogue KV: cached_input_ids vs KV length mismatch "
            f"ids={len(stored_ids)} kv={actual_kv_len}"
        )

    episode_image_paths = getattr(inner, "_geopact_dialogue_episode_image_paths", None)
    if not episode_image_paths:
        episode_image_paths = getattr(inner, "_geopact_image_paths", None) or []
    image_token_id = _image_token_id(inner)

    prefill = (
        getattr(inner, "_geopact_session_full_prefill_backup", None)
        or getattr(inner, "_geopact_prefill_payload", None)
    )
    prev_state = _get_dialogue_state(inner) or {}
    prev_mrope = list(prev_state.get("cached_mrope_input_ids") or [])
    step_mrope = _mrope_source_list(
        (prefill or {}).get("mrope_source_input_ids")
    )
    if step_mrope:
        compact_stats = (prefill or {}).get("geopact_compact_stats") or {}
        compact_full_len = int(compact_stats.get("compact_seq_len", 0) or 0)
        prev_prompt_len = (
            int(prev_state.get("prompt_token_len", 0) or 0)
            if step_index > 0 and prev_state
            else 0
        )
        step_compact_no_gen_len = int(actual_kv_len) - prev_prompt_len
        if step_compact_no_gen_len <= 0:
            clear_session_kv(model)
            raise RuntimeError(
                "GeoPACT dialogue KV: invalid compact step length for MRoPE "
                f"actual_kv_len={actual_kv_len} prev_prompt_len={prev_prompt_len}"
            )
        if compact_full_len > 0:
            from lmms_eval.models.geopact_coord_injection.lifecycle import (
                mrope_virtual_compact_len,
            )
            from lmms_eval.models.geopact_coord_injection.record import record_from_stats

            inj_rec = record_from_stats(compact_stats)
            coord_ephemeral = (
                int(inj_rec.token_count)
                if inj_rec.ephemeral and inj_rec.spliced
                else 0
            )
            compact_virt_len = mrope_virtual_compact_len(compact_full_len, inj_rec)
            step_mrope, gen_tail_len = _strip_virtual_mrope_gen_tail(
                step_mrope,
                compact_full_len=compact_virt_len,
                compact_no_gen_len=step_compact_no_gen_len,
                where="dialogue_kv_finalize",
            )
            if gen_tail_len > 0:
                stats["dialogue_kv_mrope_gen_tail_stripped"] = gen_tail_len
            if coord_ephemeral > 0:
                stats["dialogue_kv_coord_ephemeral_stripped"] = coord_ephemeral
        else:
            # No compact stats: the step ran without pruning, so virtual ==
            # compact for this step and any trailing surplus over the captured
            # (no-gen) KV length is exactly the chat-template generation tail.
            # Without this strip the tail pollutes the accumulated virtual
            # stream and breaks the compact<->virtual map that anchored
            # eviction and boundary recomputes rely on.
            step_mrope, gen_tail_len = _strip_virtual_mrope_gen_tail(
                step_mrope,
                compact_full_len=len(step_mrope),
                compact_no_gen_len=step_compact_no_gen_len,
                where="dialogue_kv_finalize_nostats",
            )
            if gen_tail_len > 0:
                stats["dialogue_kv_mrope_gen_tail_stripped"] = gen_tail_len
    if step_index <= 0 or not prev_mrope:
        cached_mrope = step_mrope or stored_ids
    elif step_mrope:
        cached_mrope = prev_mrope + step_mrope
    else:
        cached_mrope = prev_mrope

    prefill_pos = (prefill or {}).get("position_ids")
    boundary_pos = None
    if step_index <= 0:
        if prefill_pos is not None and int(prefill_pos.shape[-1]) >= int(prompt_token_len):
            boundary_idx = int(prompt_token_len) - 1
            boundary_pos = _slice_position_ids(prefill_pos, boundary_idx, boundary_idx + 1)
        if boundary_pos is None:
            gen_pos = getattr(inner, "_geopact_generate_position_ids", None)
            if gen_pos is not None and int(gen_pos.shape[-1]) >= int(prompt_token_len):
                boundary_idx = int(prompt_token_len) - 1
                boundary_pos = _slice_position_ids(gen_pos, boundary_idx, boundary_idx + 1)
        if boundary_pos is None and image_grid_thw is not None:
            tmp_state = {
                "cached_input_ids": stored_ids,
                "prompt_token_len": actual_kv_len,
                "cached_mrope_input_ids": [],
                "image_grid_thw": image_grid_thw,
            }
            boundary_pos = _recompute_cached_boundary_position_ids(inner, tmp_state)
            if boundary_pos is not None:
                stats["dialogue_kv_boundary_source"] = "native_recompute"
    else:
        suffix_pos_stored = getattr(inner, "_geopact_dialogue_last_suffix_position_ids", None)
        suffix_no_gen = list(
            getattr(inner, "_geopact_dialogue_suffix_no_gen_input_ids", None) or []
        )
        if suffix_pos_stored is not None and suffix_no_gen:
            local_end = len(suffix_no_gen) - 1
            if int(suffix_pos_stored.shape[-1]) > local_end:
                boundary_pos = _slice_position_ids(suffix_pos_stored, local_end, local_end + 1)
    inner._geopact_dialogue_last_suffix_position_ids = None

    try:
        from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_finalize

        if trace_enabled():
            trace_finalize(
                step_index=int(step_index),
                prompt_token_len=int(actual_kv_len),
                cached_ids_len=len(stored_ids),
                cached_mrope_len=len(cached_mrope),
                boundary_pos=boundary_pos,
                gen_tail_stripped=int(stats.get("dialogue_kv_mrope_gen_tail_stripped", 0) or 0),
                image_grid_thw=image_grid_thw,
                position_ids_gold=(prefill or {}).get("position_ids"),
            )
    except Exception:
        pass

    # Cumulative slide offset survives across steps within an episode; a full
    # reset clears prev_state and re-anchors the coordinate frame at 0.
    pos_offset = _mrope_pos_offset(prev_state) if step_index > 0 else 0
    state_payload: Dict[str, Any] = {
        "episode_key": episode_key,
        "through_user_step": step_index,
        "prompt_token_len": actual_kv_len,
        "cached_input_ids": stored_ids,
        "cached_mrope_input_ids": cached_mrope,
        "past_key_values": cropped,
        "rope_deltas": getattr(inner, "rope_deltas", None),
        "image_paths": list(episode_image_paths),
        "mrope_virt_pos_offset": pos_offset,
    }
    # Anchored-eviction bookkeeping: pristine (offset-0) system keys survive
    # across steps so later evictions re-rotate from an unrotated source.
    if step_index > 0 and prev_state.get("sys_pristine_keys") is not None:
        state_payload["sys_pristine_keys"] = prev_state.get("sys_pristine_keys")
        state_payload["sys_anchor_len"] = int(prev_state.get("sys_anchor_len") or 0)
    stats["kv_flow_mode"] = kv_flow_mode_label(model)
    prev_key = prev_state.get("cached_virtual_mrope_key")
    if _should_ensure_virtual_mrope_cache(model, state_payload):
        new_key = _virtual_mrope_cache_key(state_payload)
        if new_key and new_key == prev_key and prev_state.get("cached_virtual_mrope_pos") is not None:
            state_payload["cached_virtual_mrope_key"] = prev_key
            state_payload["cached_virtual_mrope_pos"] = prev_state.get("cached_virtual_mrope_pos")
        else:
            state_payload = _ensure_virtual_mrope_cache(inner, state_payload)
    # Carry forward already-compressed history frame instances (KV rewrite).
    state_payload["kv_compressed_paths"] = _normalise_kv_compressed_keys(
        prev_state.get("kv_compressed_paths") or ()
    )
    # Module C (QASMR): active relay frames persist for the episode; the side-
    # car ctx on the language model is refreshed from the committed state.
    if step_index > 0 and prev_state.get("relay_frames"):
        state_payload["relay_frames"] = prev_state.get("relay_frames")
    try:
        from lmms_eval.models.geopact_spatial_relay import (
            install_relay_ctx,
            relay_kv_mode_active,
        )

        if relay_kv_mode_active():
            install_relay_ctx(inner, state_payload)
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"):
            raise
        eval_logger.warning(f"GeoPACT relay ctx install failed: {exc}")
    # History prune target: capture the just-committed (current) screenshot so the
    # next step can compress it in place without re-encoding pixels.
    if kv_history_prune_rewrite_enabled(model):
        pending = getattr(inner, "_geopact_pending_last_frame", None)
        if pending is not None:
            cap = pending
            inner._geopact_pending_last_frame = None
        else:
            cap = _capture_last_full_frame(
                inner,
                prefill,
                stored_ids=stored_ids,
                image_paths=list(episode_image_paths),
                image_token_id=image_token_id,
            )
            if cap is not None:
                try:
                    from lmms_eval.models.geopact_module_a_prefetch import (
                        prefetch_module_a_for_history_capture,
                    )

                    # Full-prefill steps (step0) skip stage_pending; start the
                    # detect here so the next step's rectify joins instead of
                    # running Module A synchronously inside TTFT.
                    prefetch_module_a_for_history_capture(model, cap)
                except Exception:
                    pass
        if cap is not None:
            # Module C (QASMR): snapshot the current frame's real per-layer
            # post-MRoPE KV so the next step's rewrite can build KV relays
            # from it (spec docs/geopact_module_c_qasmr.md §6.1). The frame
            # was prefillled at full token count this step, so ``cropped``
            # holds its exact contextualized KV.
            try:
                from lmms_eval.models.geopact_spatial_relay import (
                    capture_frame_kv_snapshot,
                    relay_kv_snapshot_needed,
                )

                if relay_kv_snapshot_needed() and cap.get("layer_kv") is None:
                    span = _last_visual_span(stored_ids, image_token_id)
                    if span is not None:
                        snap = capture_frame_kv_snapshot(
                            _legacy_cache_tensors(cropped), span[0], span[1]
                        )
                        if snap is not None:
                            cap["layer_kv"] = snap
                            stats["dialogue_kv_relay_snapshot"] = True
            except Exception as exc:
                if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"):
                    raise
                eval_logger.warning(f"GeoPACT relay KV snapshot failed: {exc}")
            state_payload["last_full_frame"] = cap
            try:
                from lmms_eval.models.geopact_rectify_prefetch import (
                    start_rectify_prefetch,
                )

                # Post-decode window (default): the full ABC rectify (Module B
                # + ZAFD) overlaps the next step's CPU-side prep instead of
                # decode. In 'stage' mode the job was already submitted before
                # decode; don't resubmit the same frame.
                handle = getattr(inner, "_geopact_rectify_async", None)
                same_frame = isinstance(handle, dict) and str(
                    handle.get("img_path") or ""
                ) == str(cap.get("img_path") or "") and int(
                    handle.get("frame_seq", -2)
                ) == int(cap.get("frame_seq", -1))
                if not same_frame:
                    start_rectify_prefetch(model, cap, episode_key=episode_key)
            except Exception:
                pass
    if boundary_pos is not None and int(boundary_pos.shape[-1]) == 1:
        state_payload["cached_boundary_position_ids"] = boundary_pos.detach().cpu().clone()
    if image_grid_thw is not None:
        state_payload["image_grid_thw"] = image_grid_thw.detach().cpu()
    _set_dialogue_state(inner, state_payload)
    stats["dialogue_kv_cache_updated"] = True
    stats["dialogue_kv_through_user_step"] = step_index
    stats["dialogue_kv_prompt_tokens"] = actual_kv_len
    stats["dialogue_kv_commit_mode"] = "truncate_after_generate"
    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    if ledger is not None:
        boundary_audit = None
        if boundary_pos is not None and isinstance(boundary_pos, torch.Tensor):
            try:
                boundary_audit = [int(x) for x in boundary_pos.flatten().tolist()]
            except Exception:
                boundary_audit = None
        ledger.set_audit(
            kv_len=int(actual_kv_len),
            pos_offset=int(pos_offset),
            boundary_pos=boundary_audit,
        )
    return stats
