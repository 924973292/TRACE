"""
Pre-generate GeoPACT compact inputs for Qwen3-VL (GUI-Owl path).

Mirrors the proven Qwen2.5-Omni-7B flow (prepare_omnigui_projector_inputs):
  1. Vision encode once
  2. Hook select + compact + origpos MRoPE
  3. Keep original image_grid_thw on the origpos path; compact grid is only for seqpos
  4. Subset deepstack in *sequence* order
  5. generate() receives inputs_embeds + position_ids — no pixel_values

Dialectical notes vs 7B:
  - Fail-loud on MRoPE (7B silent except)
  - Per-frame grid rewrite for multi-image hist_k5 (7B only single synthetic row)
  - Deepstack subset (Qwen3-VL only; 7B has no deepstack)
  - in_forward hook remains as fallback (GEOPACT_COMPACT_BACKEND=in_forward)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch

from lmms_eval.models.geopact_module_a_prefetch import (
    mark_vit_done_for_module_a_prefetch,
    maybe_start_module_a_prefetch,
)
from lmms_eval.models.geopact_vl_hook import (
    _attention_mask_for_rope,
    _begin_precompute_full_mrope,
    _compact_sequence,
    _end_precompute_full_mrope,
    _flat_image_token_ranges,
    _is_dialogue_sticky_embed_prune,
    _module_a_prefetch_enabled,
    _module_a_sideload_enabled,
    _needs_sticky_relevance_query_cache,
    _policy_has,
    _prune_selector_label,
    _resolve_rope_model,
    _rewrite_compact_image_grid,
    _scope_is_sticky,
    _spatial_merge_size,
    _visual_keep_in_sequence_order,
    apply_video_to_compact_prefill,
    cache_sticky_relevance_query,
    compute_sticky_dialogue_prune_keep_set,
    _coord_ephemeral_enabled,
    _coord_policy_active,
)
from lmms_eval.models.geopact_session_kv import (
    dialogue_kv_enabled,
    session_kv_enabled,
)
from lmms_eval.models.geopact_vision_cache import (
    build_deepstack_compact_mixed,
    build_mixed_image_features,
    save_current_frame_compact_cache,
    vision_embed_cache_enabled,
    write_current_frame_compact_cache,
)


def _deepstack_compact_index_select(
    inner,
    per_frame_deepstack: List[List[torch.Tensor]],
    seq_keep: torch.Tensor,
) -> Optional[List[torch.Tensor]]:
    """Fuse per-frame deepstack rows with one cat/layer + shared index_select."""
    if not per_frame_deepstack or not per_frame_deepstack[0] or seq_keep is None:
        return None
    n_frames = len(per_frame_deepstack)
    n_layers = len(per_frame_deepstack[0])
    cache_sig = (
        n_frames,
        n_layers,
        tuple(int(per_frame_deepstack[f][0].shape[0]) for f in range(n_frames)),
    )
    cached = getattr(inner, "_geopact_step_flat_deepstack", None)
    if cached is None or cached.get("sig") != cache_sig:
        flat_layers = [
            torch.cat([per_frame_deepstack[f][li] for f in range(n_frames)], dim=0)
            for li in range(n_layers)
        ]
        inner._geopact_step_flat_deepstack = {"sig": cache_sig, "layers": flat_layers}
    else:
        flat_layers = cached["layers"]
    idx = seq_keep.to(device=flat_layers[0].device, dtype=torch.long)
    return [layer.index_select(0, idx) for layer in flat_layers]


def use_source_compact_backend() -> bool:
    backend = os.environ.get("GEOPACT_COMPACT_BACKEND", "source").strip().lower()
    return backend not in {"0", "false", "off", "in_forward", "legacy", "hook"}


def _dialogue_token_prune_enabled(model) -> bool:
    inner = _inner_vl_model(model)
    if inner is None:
        return False
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if not _is_dialogue_sticky_embed_prune(policy):
        return False
    scope = str(getattr(inner, "_geopact_prune_scope", "") or "").lower()
    return _scope_is_sticky(scope)


def kv_step0_native_enabled() -> bool:
    """Use baseline_full prefill path on dialogue KV full-prefill steps (step0 / segment reset)."""
    return os.environ.get("GEOPACT_KV_STEP0_NATIVE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _dialogue_kv_use_native_full_prefill(model) -> bool:
    """Dialogue KV suffix-only steps keep dialogue_embed; full prefill matches baseline_full."""
    if not dialogue_kv_enabled() or not kv_step0_native_enabled():
        return False
    inner = _inner_vl_model(model)
    if inner is None:
        return False
    if bool(getattr(inner, "_geopact_dialogue_suffix_only_step", False)):
        return False
    return True


def _dialogue_history_embed_only_path(model) -> bool:
    """History-target dialogue KV / ABC: prefill keeps the current frame full.

    History-frame ABC runs in ``sliding_rewrite_prepare_dialogue_kv`` (KV rewrite),
    not via ``compute_sticky_dialogue_prune_keep_set`` during ViT prefill.

    NAP two-stage (GEOPACT_TWO_STAGE=1 + 0<r_cur<1): the current frame is
    pruned at entry too, so the prefill routes through the pruned embed path
    like the ``current`` target; stage-2 contraction still owns the rewrite.
    """
    if not dialogue_kv_enabled():
        return False
    inner = _inner_vl_model(model)
    if inner is None:
        return False
    from lmms_eval.models.geopact_nap import nap_current_prune_active
    from lmms_eval.models.geopact_session_kv import kv_history_prune_rewrite_enabled
    from lmms_eval.models.geopact_vl_hook import _prune_target

    if nap_current_prune_active(inner) and _dialogue_token_prune_enabled(model):
        return False
    if kv_history_prune_rewrite_enabled(model):
        return True
    return str(_prune_target(model) or "").lower() == "history"


def prepare_active_prefill_inputs(
    model,
    inputs,
    *,
    use_pruning: bool,
) -> tuple[Any, Any]:
    """Select pre-LLM prefill path.

    History-target dialogue KV / ABC uses ``prepare_dialogue_embed_inputs`` only;
    ``prepare_dialogue_pruned_embed_inputs`` applies to ``prune_target=current``
    (sticky per-frame mask during prefill). Non-dialogue paths use
    ``prepare_geopact_projector_inputs``.
    """
    active = inputs
    if use_pruning and use_source_compact_backend():
        if _dialogue_history_embed_only_path(model):
            dialogue = prepare_dialogue_embed_inputs(model, inputs)
            if dialogue is not None:
                return dialogue, inputs
        probe_gold_prune = os.environ.get("GEOPACT_PROBE_GOLD_FULL_PRUNE", "").strip() == "1"
        if (
            dialogue_kv_enabled()
            and _dialogue_token_prune_enabled(model)
            and not _dialogue_history_embed_only_path(model)
        ):
            compact = prepare_dialogue_pruned_embed_inputs(model, inputs)
        elif probe_gold_prune and _dialogue_token_prune_enabled(model):
            compact = prepare_dialogue_pruned_embed_inputs(model, inputs)
        else:
            compact = prepare_geopact_projector_inputs(model, inputs)
        if compact is not None:
            return compact, inputs
    if dialogue_kv_enabled():
        if not _dialogue_kv_use_native_full_prefill(model):
            dialogue = prepare_dialogue_embed_inputs(model, inputs)
            if dialogue is not None:
                return dialogue, inputs
        elif use_pruning and use_source_compact_backend():
            compact = prepare_geopact_projector_inputs(model, inputs)
            if compact is not None:
                return compact, inputs
        # Native full-prefill (step0 / segment reset): fall through with pixel_values.
    if vision_embed_cache_enabled() or session_kv_enabled():
        baseline = prepare_baseline_vision_cached_inputs(model, inputs)
        if baseline is not None:
            return baseline, inputs
    return active, inputs


def rewrite_grid_enabled() -> bool:
    return os.environ.get("GEOPACT_REWRITE_GRID", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _is_qwen25_carrier(m) -> bool:
    """True for the Qwen2.5-VL carrier (UI-TARS-1.5-7B): FCG owns visual/.model/state."""
    return str(getattr(getattr(m, "config", None), "model_type", "")) == "qwen2_5_vl"


def _inner_vl_model(model):
    if hasattr(model, "model") and hasattr(model.model, "get_image_features"):
        return model.model
    if hasattr(model, "get_image_features"):
        return model
    # Qwen2.5-VL carrier: no get_image_features anywhere; the FCG itself is the
    # inner object (owns .visual, .model, get_rope_index and all _geopact state).
    if _is_qwen25_carrier(model):
        return model
    if hasattr(model, "model") and _is_qwen25_carrier(model.model):
        return model.model
    return None


def _carrier_build_image_features(inner, pixel_values, image_grid_thw, *, image_paths):
    """Vision features dispatch: qwen3 keeps build_mixed_image_features verbatim.

    Qwen2.5-VL has no get_image_features / deepstack; encode via ``inner.visual``
    directly and bypass the (deepstack-coupled) vision embed cache entirely.
    Returns the same (flat image_embeds, per_frame_deepstack, stats, compact_hits)
    convention; per-frame deepstack is empty for qwen2.5.
    """
    if not _is_qwen25_carrier(inner):
        return build_mixed_image_features(
            inner,
            pixel_values,
            image_grid_thw,
            image_paths=image_paths,
        )
    # Mirror build_mixed_image_features per-prefill reset so stale frame-relative
    # hit indices can never leak across turns.
    inner._geopact_compact_frame_cache_hits = set()
    stats: Dict[str, Any] = {
        "vision_cache_enabled": False,
        "vision_cache_hits": 0,
        "vision_cache_misses": 0,
        "vision_encode_ms": 0.0,
        "vision_cache_frames": int(image_grid_thw.shape[0]) if image_grid_thw is not None else 0,
        "vision_cache_mode": "qwen2_5_vl_no_cache",
    }
    t0 = time.perf_counter()
    image_embeds = inner.visual(
        pixel_values.type(inner.visual.dtype), grid_thw=image_grid_thw
    )
    stats["vision_encode_ms"] = (time.perf_counter() - t0) * 1000.0
    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    if ledger is not None:
        ledger.add_ms("vision_encode_ms", float(stats["vision_encode_ms"]))
    merge = int(inner.config.vision_config.spatial_merge_size)
    expected = sum(
        int(image_grid_thw[i][0]) * int(image_grid_thw[i][1]) * int(image_grid_thw[i][2])
        // (merge**2)
        for i in range(int(image_grid_thw.shape[0]))
    )
    if int(image_embeds.shape[0]) != expected:
        raise RuntimeError(
            "GeoPACT qwen2_5_vl visual: flat embed rows "
            f"{int(image_embeds.shape[0])} != grid-derived tokens {expected}"
        )
    return image_embeds, [], stats, set()


def _carrier_placeholder_mask(inner, input_ids, inputs_embeds, image_embeds):
    """Image placeholder mask: qwen3 uses get_placeholder_mask verbatim.

    Qwen2.5-VL has no get_placeholder_mask; build the mask from image token ids
    in the same (batch, seq, hidden)-expanded layout the qwen3 return provides.
    """
    if hasattr(inner, "get_placeholder_mask"):
        image_mask, _ = inner.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        return image_mask
    image_tok_id = int(getattr(inner.config, "image_token_id", -1) or -1)
    if image_tok_id < 0:
        raise RuntimeError("GeoPACT qwen2_5_vl: config.image_token_id unavailable")
    mask_2d = (input_ids == image_tok_id).to(inputs_embeds.device)
    n_mask = int(mask_2d.sum().item())
    if n_mask != int(image_embeds.shape[0]):
        raise RuntimeError(
            "GeoPACT qwen2_5_vl placeholder mask: image tokens "
            f"{n_mask} != image_embeds rows {int(image_embeds.shape[0])}"
        )
    return mask_2d.unsqueeze(-1).expand_as(inputs_embeds)


def _finalize_carrier_prefill_payload(inner, payload):
    """Qwen2.5-VL forward has no deepstack kwargs; drop the key entirely."""
    if _is_qwen25_carrier(inner):
        payload.pop("deepstack_image_embeds", None)
    return payload


def _normalize_image_mask_1d(image_mask: torch.Tensor) -> torch.Tensor:
    """Match geopact hook layout: (batch, seq) or (batch, seq, 1) -> (seq,) bool."""
    if image_mask.dim() > 1 and int(image_mask.shape[0]) == 1:
        image_mask = image_mask[0]
    if image_mask.dim() > 1:
        image_mask = image_mask.any(dim=tuple(range(1, image_mask.dim())))
    return image_mask.bool()


def validate_deepstack_alignment(
    deepstack_layers: Optional[List[torch.Tensor]],
    image_mask: Optional[torch.Tensor],
    *,
    n_visual_kept: Optional[int] = None,
    where: str,
    carrier_has_deepstack: bool = True,
) -> None:
    """Fail loud when pruned deepstack rows do not match visual token count.

    ``carrier_has_deepstack=False`` (qwen2.5-VL carrier) tolerates an absent
    deepstack instead of raising; default True keeps qwen3 behavior identical.
    """
    if image_mask is None:
        return
    if not carrier_has_deepstack and not deepstack_layers:
        return
    mask_1d = _normalize_image_mask_1d(image_mask)
    n_vis = int(mask_1d.sum().item())
    if n_visual_kept is not None and int(n_visual_kept) != n_vis:
        raise RuntimeError(
            f"GeoPACT {where}: n_visual_kept={n_visual_kept} != image_mask visual_tokens={n_vis}"
        )
    if n_vis == 0:
        if deepstack_layers and int(deepstack_layers[0].shape[0]) != 0:
            raise RuntimeError(
                f"GeoPACT {where}: no visual tokens in mask but deepstack has "
                f"{deepstack_layers[0].shape[0]} rows"
            )
        return
    if not deepstack_layers:
        raise RuntimeError(
            f"GeoPACT {where}: {n_vis} visual tokens but deepstack_image_embeds is missing"
        )
    for layer_i, layer in enumerate(deepstack_layers):
        n_rows = int(layer.shape[0])
        if n_rows != n_vis:
            raise RuntimeError(
                f"GeoPACT {where}: deepstack layer {layer_i} rows={n_rows} "
                f"!= visual_tokens={n_vis}"
            )


def _scatter_image_embeds(inner, input_ids, inputs_embeds, image_embeds, image_mask):
    image_mask = _normalize_image_mask_1d(image_mask)
    seq_len = int(input_ids.shape[1])
    hidden = int(inputs_embeds.shape[-1])
    img_mask_exp = image_mask.unsqueeze(0).unsqueeze(-1).expand(1, seq_len, hidden)
    scattered = inputs_embeds.masked_scatter(
        img_mask_exp, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    )
    compact_image_mask = None
    image_tok_id = int(getattr(inner.config, "image_token_id", -1) or -1)
    if image_tok_id >= 0:
        compact_image_mask = (input_ids[0] == image_tok_id).view(1, -1, 1)
    return scattered, compact_image_mask


def _concat_per_frame_deepstack(per_frame_deepstack: List[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
    if not per_frame_deepstack or not per_frame_deepstack[0]:
        return None
    n_layers = len(per_frame_deepstack[0])
    return [
        torch.cat([per_frame_deepstack[f][li] for f in range(len(per_frame_deepstack))], dim=0)
        for li in range(n_layers)
    ]


@torch.inference_mode()
def prepare_baseline_vision_cached_inputs(model, inputs) -> Optional[Dict[str, Any]]:
    """Baseline (no pruning): cache full-frame ViT outputs + deepstack per PNG."""
    prepare_t0 = time.perf_counter()
    if not vision_embed_cache_enabled():
        return None

    inner = _inner_vl_model(model)
    if inner is None:
        return None

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs.input_ids
    if input_ids is None or input_ids.shape[0] != 1 or input_ids.shape[1] <= 1:
        return None

    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else getattr(inputs, "pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "image_grid_thw", None)
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else getattr(inputs, "attention_mask", None)
    video_grid_thw = inputs.get("video_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "video_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        return None

    image_paths = getattr(inner, "_geopact_image_paths", None) or []
    inputs_embeds = inner.get_input_embeddings()(input_ids)
    image_embeds, per_frame_deepstack, vision_cache_stats, _compact_hits = _carrier_build_image_features(
        inner,
        pixel_values,
        image_grid_thw,
        image_paths=image_paths,
    )
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_mask = _carrier_placeholder_mask(
        inner,
        input_ids,
        inputs_embeds,
        image_embeds,
    )
    scattered, compact_image_mask = _scatter_image_embeds(
        inner, input_ids, inputs_embeds, image_embeds, image_mask
    )

    rope_model = _resolve_rope_model(inner)
    get_rope = getattr(rope_model, "get_rope_index", None) if rope_model is not None else None
    if get_rope is None:
        return None
    attn = _attention_mask_for_rope(attention_mask)
    position_ids, rope_deltas = get_rope(
        input_ids, image_grid_thw, video_grid_thw, attention_mask=attn
    )

    deepstack_full = _concat_per_frame_deepstack(per_frame_deepstack)
    n_img = int(image_embeds.shape[0])
    token_ranges = _flat_image_token_ranges(image_grid_thw, n_img)
    if image_paths and token_ranges:
        frame_idx = len(token_ranges) - 1
        start, end = token_ranges[frame_idx]
        img_path = image_paths[frame_idx] if frame_idx < len(image_paths) else ""
        if img_path and end > start:
            local_keep_seq = tuple(range(end - start))
            idx = torch.arange(start, end, device=image_embeds.device, dtype=torch.long)
            save_current_frame_compact_cache(
                inner,
                img_path,
                compact_embeds=image_embeds.index_select(0, idx),
                compact_deepstack=[
                    layer.index_select(0, idx.to(layer.device)) for layer in (deepstack_full or [])
                ] if deepstack_full else [],
                local_keep_seq=local_keep_seq,
                orig_grid_row=image_grid_thw[frame_idx : frame_idx + 1],
            )

    vision_cache_stats["compact_backend"] = "baseline_vision_cache"
    vision_cache_stats["geopact_prepare_ms"] = round((time.perf_counter() - prepare_t0) * 1000.0, 2)
    vision_cache_stats["pruning_ms"] = 0.0
    vision_cache_stats["orig_seq_len"] = int(input_ids.shape[1])
    vision_cache_stats["compact_seq_len"] = int(input_ids.shape[1])
    vision_cache_stats["tokens_before_pruning"] = int(input_ids.shape[1])
    vision_cache_stats["tokens_after_pruning"] = int(input_ids.shape[1])
    inner._geopact_last_compact_stats = vision_cache_stats

    prefill_payload = {
        "input_ids": input_ids,
        "inputs_embeds": scattered,
        "attention_mask": attention_mask,
        "image_grid_thw": image_grid_thw,
        "image_mask": compact_image_mask,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        "deepstack_image_embeds": deepstack_full,
        "force_rope_recompute": False,
    }
    prefill_payload = apply_video_to_compact_prefill(
        inner, prefill_payload, inputs, hook_active=False
    )
    prefill_payload = _finalize_carrier_prefill_payload(inner, prefill_payload)
    inner._geopact_prefill_payload = prefill_payload

    out = {
        "input_ids": prefill_payload["input_ids"],
        "inputs_embeds": prefill_payload["inputs_embeds"],
        "attention_mask": prefill_payload.get("attention_mask", attention_mask),
        "image_grid_thw": image_grid_thw,
    }
    return {k: v for k, v in out.items() if v is not None}


@torch.inference_mode()
def prepare_dialogue_embed_inputs(model, inputs) -> Optional[Dict[str, Any]]:
    """Dialogue KV: ViT-encode only the frames present in this prompt (no vision cache).

    Step 0 prefills the full user prompt; step>0 prefills the incremental suffix
    (previous GT assistant + current user). Historical screenshots live in LLM KV,
    so we never look up ``GEOPACT_VISION_EMBED_CACHE`` here.
    """
    prepare_t0 = time.perf_counter()
    inner = _inner_vl_model(model)
    if inner is None:
        return None

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs.input_ids
    if input_ids is None or input_ids.shape[0] != 1 or input_ids.shape[1] <= 1:
        return None

    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else getattr(inputs, "pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "image_grid_thw", None)
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else getattr(inputs, "attention_mask", None)
    video_grid_thw = inputs.get("video_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "video_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        return None

    inputs_embeds = inner.get_input_embeddings()(input_ids)
    image_embeds, per_frame_deepstack, vision_stats, _compact_hits = _carrier_build_image_features(
        inner,
        pixel_values,
        image_grid_thw,
        image_paths=[],
    )
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_mask = _carrier_placeholder_mask(
        inner,
        input_ids,
        inputs_embeds,
        image_embeds,
    )
    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    scatter_t0 = time.perf_counter()
    scattered, compact_image_mask = _scatter_image_embeds(
        inner, input_ids, inputs_embeds, image_embeds, image_mask
    )
    if ledger is not None:
        ledger.add_ms("compact_gather_ms", (time.perf_counter() - scatter_t0) * 1000.0)

    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if _needs_sticky_relevance_query_cache(policy) and not bool(
        getattr(inner, "_geopact_dialogue_suffix_only_step", False)
    ):
        cache_sticky_relevance_query(inner)

    rope_model = _resolve_rope_model(inner)
    get_rope = getattr(rope_model, "get_rope_index", None) if rope_model is not None else None
    if get_rope is None:
        return None
    attn = _attention_mask_for_rope(attention_mask)
    mrope_t0 = time.perf_counter()
    position_ids, rope_deltas = get_rope(
        input_ids, image_grid_thw, video_grid_thw, attention_mask=attn
    )
    if ledger is not None:
        ledger.add_ms("origpos_mrope_ms", (time.perf_counter() - mrope_t0) * 1000.0)

    deepstack_full = _concat_per_frame_deepstack(per_frame_deepstack)
    vision_stats["vision_cache_enabled"] = False
    vision_stats["compact_backend"] = "dialogue_embed"
    vision_stats["geopact_prepare_ms"] = round((time.perf_counter() - prepare_t0) * 1000.0, 2)
    if ledger is not None:
        snap = ledger.snapshot_totals()
        accounted = (
            float(snap.get("vision_encode_ms", 0.0) or 0.0)
            + float(snap.get("compact_gather_ms", 0.0) or 0.0)
            + float(snap.get("origpos_mrope_ms", 0.0) or 0.0)
        )
        remainder = float(vision_stats["geopact_prepare_ms"]) - accounted
        if remainder > 0.01:
            ledger.add_ms("vlm_prefill_ms", remainder)
    vision_stats["pruning_ms"] = 0.0
    vision_stats["orig_seq_len"] = int(input_ids.shape[1])
    vision_stats["compact_seq_len"] = int(input_ids.shape[1])
    vision_stats["tokens_before_pruning"] = int(input_ids.shape[1])
    vision_stats["tokens_after_pruning"] = int(input_ids.shape[1])
    inner._geopact_last_compact_stats = vision_stats

    prefill_payload = {
        "input_ids": input_ids,
        "inputs_embeds": scattered,
        "attention_mask": attention_mask,
        "image_grid_thw": image_grid_thw,
        "image_mask": compact_image_mask,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        "deepstack_image_embeds": deepstack_full,
        "force_rope_recompute": False,
        "mrope_source_input_ids": input_ids.detach().clone(),
    }
    prefill_payload = apply_video_to_compact_prefill(
        inner, prefill_payload, inputs, hook_active=False
    )
    prefill_payload = _finalize_carrier_prefill_payload(inner, prefill_payload)
    inner._geopact_prefill_payload = prefill_payload

    out = {
        "input_ids": prefill_payload["input_ids"],
        "inputs_embeds": prefill_payload["inputs_embeds"],
        "attention_mask": prefill_payload.get("attention_mask", attention_mask),
        "image_grid_thw": image_grid_thw,
        "position_ids": prefill_payload.get("position_ids"),
        "rope_deltas": prefill_payload.get("rope_deltas"),
    }
    return {k: v for k, v in out.items() if v is not None}


@torch.inference_mode()
def prepare_dialogue_pruned_embed_inputs(model, inputs) -> Optional[Dict[str, Any]]:
    """Dialogue KV + sticky prune: dialogue_embed ViT/scatter + compact.

    Skips the full GeoPACT hook / vision-cache write path. Requires
    ``visual_prune_scope=image_sticky`` and uniform/random/token_ickh or oas policy.
    """
    prepare_t0 = time.perf_counter()
    inner = _inner_vl_model(model)
    if inner is None:
        return None

    ratio = float(getattr(inner, "_geopact_prune_ratio", 0.0) or 0.0)
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    scope = str(getattr(inner, "_geopact_prune_scope", "") or "")
    if not _dialogue_token_prune_enabled(model):
        return None

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs.input_ids
    if input_ids is None or input_ids.shape[0] != 1 or input_ids.shape[1] <= 1:
        return None

    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else getattr(inputs, "pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "image_grid_thw", None)
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else getattr(inputs, "attention_mask", None)
    video_grid_thw = inputs.get("video_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "video_grid_thw", None)
    if pixel_values is None or image_grid_thw is None:
        return None

    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    inputs_embeds = inner.get_input_embeddings()(input_ids)
    mrope_handle = None
    if not bool(getattr(inner, "_geopact_dialogue_suffix_only_step", False)):
        mrope_handle = _begin_precompute_full_mrope(
            inner,
            input_ids,
            image_grid_thw,
            attention_mask,
            video_grid_thw=video_grid_thw,
        )
    maybe_start_module_a_prefetch(inner, image_grid_thw)
    image_embeds, per_frame_deepstack, vision_stats, _compact_hits = _carrier_build_image_features(
        inner,
        pixel_values,
        image_grid_thw,
        image_paths=[],
    )
    mark_vit_done_for_module_a_prefetch(inner)
    position_ids, rope_deltas = _end_precompute_full_mrope(mrope_handle, ledger=ledger)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_mask = _carrier_placeholder_mask(
        inner,
        input_ids,
        inputs_embeds,
        image_embeds,
    )
    image_mask_1d = _normalize_image_mask_1d(image_mask).to(input_ids.device)
    seq_len = int(input_ids.shape[1])
    hidden = int(inputs_embeds.shape[-1])
    n_img = int(image_embeds.shape[0])
    token_ranges = _flat_image_token_ranges(image_grid_thw, n_img)

    if _needs_sticky_relevance_query_cache(policy) and not bool(
        getattr(inner, "_geopact_dialogue_suffix_only_step", False)
    ):
        cache_sticky_relevance_query(inner)

    inner._geopact_ephemeral_coord_text = None
    prune_t0 = time.perf_counter()
    if ledger is not None:
        ledger.start_bucket("sticky_keep_ms")
    keep_token_set = compute_sticky_dialogue_prune_keep_set(
        inner,
        token_ranges=token_ranges,
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        ratio=ratio,
        policy=policy,
        scope=scope,
        inputs_embeds=inputs_embeds,
        image_mask_1d=image_mask_1d,
    )
    if ledger is not None:
        ledger.stop_bucket("sticky_keep_ms")
        # Module A wall time is NOT an exclusive bucket: when detection runs
        # synchronously (or the async join waits) it happens INSIDE the
        # sticky_keep_ms bucket above; when it prefetched, its wall overlapped
        # the vision_encode bucket. Adding it to the exclusive ledger double-
        # counted Module A (exclusive sum > wall clock). Keep the wall as an
        # informational audit field instead.
        module_a_meta = getattr(inner, "_geopact_last_module_a_meta", None) or {}
        if isinstance(module_a_meta, dict):
            timings = module_a_meta.get("timings_ms") or {}
            total_ms = timings.get("total_ms")
            if total_ms is not None:
                ledger.set_audit(module_a_wall_ms=round(float(total_ms), 2))
    pruning_ms = (time.perf_counter() - prune_t0) * 1000.0

    coord_text = None
    if _coord_ephemeral_enabled() and _coord_policy_active(policy):
        coord_text = getattr(inner, "_geopact_ephemeral_coord_text", None)

    image_tok_id = int(getattr(inner.config, "image_token_id", -1) or -1)
    compact_out = _compact_sequence(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_embeds=image_embeds,
        image_mask=image_mask_1d,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        keep_token_set=keep_token_set,
        seq_len=seq_len,
        hidden=hidden,
        coord_text=coord_text,
        tokenizer=getattr(inner, "_geopact_tokenizer", None),
        embed_fn=inner.get_input_embeddings(),
        image_tok_id=image_tok_id,
        policy=policy,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        model=inner,
        video_grid_thw=video_grid_thw,
    )

    seq_keep = compact_out.get("visual_keep_flat_indices")
    deepstack_compact = None
    if per_frame_deepstack and per_frame_deepstack[0] and seq_keep is not None:
        ds_t0 = time.perf_counter()
        deepstack_compact = _deepstack_compact_index_select(inner, per_frame_deepstack, seq_keep)

    n_kept = len(keep_token_set)
    if per_frame_deepstack and per_frame_deepstack[0] and n_kept > 0:
        if deepstack_compact is None:
            raise RuntimeError(
                "GeoPACT prepare_dialogue_pruned_embed_inputs: pruning kept "
                f"{n_kept} visual tokens but deepstack compact build failed"
            )
        validate_deepstack_alignment(
            deepstack_compact,
            compact_out.get("image_mask"),
            n_visual_kept=n_kept,
            where="prepare_dialogue_pruned_embed_inputs",
        )
        if ledger is not None:
            ledger.add_ms("deepstack_align_ms", (time.perf_counter() - ds_t0) * 1000.0)

    compact_stats = compact_out.get("geopact_compact_stats") or {}
    compact_stats["compact_backend"] = "dialogue_prune_sticky"
    compact_stats["prune_policy"] = policy
    compact_stats["prune_selector"] = _prune_selector_label(policy)
    compact_stats["pruning_ms"] = round(pruning_ms, 2)
    compact_stats["geopact_prepare_ms"] = round((time.perf_counter() - prepare_t0) * 1000.0, 2)
    inner._geopact_last_image_grid_thw = image_grid_thw
    compact_stats["tokens_before_pruning"] = int(compact_stats.get("orig_seq_len", input_ids.shape[1]))
    compact_stats["tokens_after_pruning"] = int(
        compact_stats.get("compact_seq_len", compact_stats["tokens_before_pruning"])
    )
    compact_stats["uses_origpos_grid"] = True
    if not _module_a_sideload_enabled():
        module_a_meta = getattr(inner, "_geopact_last_module_a_meta", None) or {}
        if isinstance(module_a_meta, dict):
            compact_stats["n_mius"] = module_a_meta.get("n_kept")
            timings = module_a_meta.get("timings_ms") or {}
            if timings.get("total_ms") is not None:
                compact_stats["module_a_ms"] = timings.get("total_ms")
        cache_meta = getattr(inner, "_geopact_last_module_a_cache_meta", None) or {}
        if isinstance(cache_meta, dict) and "module_a_cache_hit" in cache_meta:
            compact_stats["module_a_cache_hit"] = cache_meta.get("module_a_cache_hit")
    prune_diag = getattr(inner, "_geopact_last_prune_diag", None) or {}
    if isinstance(prune_diag, dict):
        for key in (
            "keep_k",
            "keep_count",
            "keep_fill_ratio",
            "budget_shortfall",
            "miu_union_token_count",
            "miu_coverage_ratio",
            "n_mius_detected",
            "keep_outside_union_ratio",
            "keep_local_hash",
            "miu_backfill_tokens",
            "cdct_alpha",
            "prune_selector",
            "cdct_fallback",
            "module_a_prefetch",
            "module_a_prefetch_skipped_suffix",
            # Async-overlap evidence: module_a_ms above is the detect wall
            # time even when it ran under the ViT; join_ms is what actually
            # lands on the critical path (W2 latency accounting).
            "module_a_prefetch_async",
            "module_a_overlap_ms",
            "module_a_join_ms",
            "prefetch_reused",
            "prefetch_source",
            # NAP anticipative-coverage evidence (entry-stage swaps).
            "nap_ac_swaps",
            "nap_ac_empty_strata",
            # C-Entry (single-step Module C) engagement evidence.
            "c_entry_reserve",
            "c_entry_dropped",
            "c_entry_clusters_raw",
            "c_entry_folded",
            "c_entry_discarded",
            "c_entry_tokens",
            "c_entry_shortfall",
            "c_entry_mean_cluster",
            # C-Absorb entry-merge extension engagement evidence (W-A3).
            "entry_merge_applied",
            "entry_merge_donors",
            "entry_merge_hosts",
            "entry_merge_kappa",
            "entry_merge_lambda",
        ):
            if key in prune_diag:
                compact_stats[key] = prune_diag[key]
    if _module_a_prefetch_enabled():
        compact_stats["module_a_prefetch"] = getattr(inner, "_geopact_module_a_prefetch", "cpu")
    compact_stats.update(vision_stats)
    vision_stats = compact_stats
    inner._geopact_last_compact_stats = vision_stats

    prefill_payload = {
        "input_ids": compact_out["input_ids"],
        "inputs_embeds": compact_out["inputs_embeds"],
        "attention_mask": compact_out.get("attention_mask", attention_mask),
        "image_grid_thw": image_grid_thw,
        "image_mask": compact_out.get("image_mask"),
        "position_ids": compact_out.get("position_ids"),
        "rope_deltas": compact_out.get("rope_deltas"),
        "deepstack_image_embeds": deepstack_compact,
        "force_rope_recompute": bool(compact_out.get("force_rope_recompute")),
        "visual_keep_flat_indices": seq_keep,
        "mrope_source_input_ids": compact_out.get("mrope_source_input_ids"),
        "mrope_keep_mask": compact_out.get("mrope_keep_mask"),
        "geopact_compact_stats": compact_stats,
    }
    prefill_payload = apply_video_to_compact_prefill(
        inner, prefill_payload, inputs, hook_active=True
    )
    prefill_payload = _finalize_carrier_prefill_payload(inner, prefill_payload)
    inner._geopact_prefill_payload = prefill_payload

    try:
        from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_gold_full_positions, trace_prepare

        if trace_enabled():
            trace_prepare(
                model,
                compact_stats=compact_stats,
                visual_keep_flat_indices=seq_keep,
                image_grid_thw=image_grid_thw,
            )
            if prefill_payload.get("position_ids") is not None:
                trace_gold_full_positions(
                    position_ids=prefill_payload["position_ids"],
                    input_ids=prefill_payload.get("input_ids"),
                )
    except Exception:
        pass

    out = {
        "input_ids": prefill_payload["input_ids"],
        "inputs_embeds": prefill_payload["inputs_embeds"],
        "attention_mask": prefill_payload.get("attention_mask", attention_mask),
        "image_grid_thw": image_grid_thw,
        "position_ids": prefill_payload.get("position_ids"),
        "rope_deltas": prefill_payload.get("rope_deltas"),
    }
    return {k: v for k, v in out.items() if v is not None}


@torch.inference_mode()
def prepare_geopact_projector_inputs(model, inputs) -> Optional[Dict[str, Any]]:
    """Build compact generate kwargs before model.generate().

    Returns a dict suitable for ``generate(**compact)`` with keys:
      input_ids, inputs_embeds, attention_mask, image_grid_thw,
      position_ids, rope_deltas
    plus internal ``_geopact_prefill_payload`` for deepstack replay in forward.

    Returns None when pruning is inactive or batch != 1.
    """
    prepare_t0 = time.perf_counter()
    inner = _inner_vl_model(model)
    if inner is None:
        return None
    hook = getattr(inner, "_geopact_projector_prune_hook", None)
    if hook is None:
        return None

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else inputs.input_ids
    if input_ids is None or input_ids.shape[0] != 1 or input_ids.shape[1] <= 1:
        return None

    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else getattr(inputs, "pixel_values", None)
    image_grid_thw = inputs.get("image_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "image_grid_thw", None)
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else getattr(inputs, "attention_mask", None)
    video_grid_thw = inputs.get("video_grid_thw") if hasattr(inputs, "get") else getattr(inputs, "video_grid_thw", None)

    if pixel_values is None or image_grid_thw is None:
        return None

    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    inputs_embeds = inner.get_input_embeddings()(input_ids)
    image_paths = getattr(inner, "_geopact_image_paths", None) or []
    maybe_start_module_a_prefetch(inner, image_grid_thw)
    image_embeds, per_frame_deepstack, vision_cache_stats, compact_cache_hits = (
        _carrier_build_image_features(
            inner,
            pixel_values,
            image_grid_thw,
            image_paths=image_paths,
        )
    )
    mark_vit_done_for_module_a_prefetch(inner)
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

    image_mask = _carrier_placeholder_mask(
        inner,
        input_ids,
        inputs_embeds,
        image_embeds,
    )

    hook_t0 = time.perf_counter()
    hook_out = hook(
        {
            "model": inner,
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "image_embeds": image_embeds,
            "image_mask": image_mask[0] if image_mask.dim() > 1 else image_mask,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "cache_position": None,
            "position_ids": None,
            "rope_deltas": None,
        }
    )
    if hook_out is None:
        return None
    pruning_ms = (time.perf_counter() - hook_t0) * 1000.0

    n_img = int(image_embeds.shape[0])
    token_ranges = _flat_image_token_ranges(image_grid_thw, n_img)
    seq_keep = hook_out.get("visual_keep_flat_indices")
    if seq_keep is None:
        if image_mask.dim() > 1:
            flat_mask = image_mask[0].any(dim=-1) if image_mask.dim() > 2 else image_mask[0]
        else:
            flat_mask = image_mask
        keep_set = set(range(int(flat_mask.sum().item())))
        if keep_set:
            if image_mask.dim() > 1:
                img_bool = image_mask[0].any(dim=tuple(range(1, image_mask.dim())))
            else:
                img_bool = image_mask.bool()
            img_seq_positions = img_bool.nonzero(as_tuple=False).flatten().tolist()
            seq_order = _visual_keep_in_sequence_order(keep_set, img_seq_positions)
            seq_keep = torch.tensor(seq_order, dtype=torch.long, device=image_embeds.device)
    if seq_keep is not None:
        keep_set = set(int(x) for x in seq_keep.tolist())
    else:
        keep_set = set()

    merge = _spatial_merge_size()
    orig_grid = hook_out.get("image_grid_thw", image_grid_thw)
    if rewrite_grid_enabled():
        compact_grid = _rewrite_compact_image_grid(orig_grid, token_ranges, keep_set, merge)
    else:
        compact_grid = orig_grid
    # Origpos is the canonical Qwen3-VL path: position_ids are computed on the
    # full original prompt and then subset. In that case forward must not mix
    # compact grid metadata with original MRoPE positions.
    uses_origpos = hook_out.get("position_ids") is not None and not bool(
        hook_out.get("force_rope_recompute")
    )
    forward_grid = orig_grid if uses_origpos else compact_grid
    hook_out["image_grid_thw"] = forward_grid

    deepstack_compact = None
    if per_frame_deepstack and seq_keep is not None:
        deepstack_compact = build_deepstack_compact_mixed(
            token_ranges,
            seq_keep,
            per_frame_deepstack,
            compact_cache_hits,
            inner,
            image_paths,
        )
        if deepstack_compact is None:
            if compact_cache_hits:
                raise RuntimeError(
                    "GeoPACT build_deepstack_compact_mixed failed with compact "
                    f"cache hits on frames {sorted(compact_cache_hits)}"
                )
            if per_frame_deepstack[0]:
                deepstack_compact = _deepstack_compact_index_select(
                    inner, per_frame_deepstack, seq_keep
                )

    has_deepstack = bool(per_frame_deepstack and per_frame_deepstack[0])
    n_kept = len(keep_set) if keep_set else n_img
    if has_deepstack and n_kept > 0:
        if deepstack_compact is None:
            raise RuntimeError(
                "GeoPACT prepare_geopact_projector_inputs: pruning kept "
                f"{n_kept} visual tokens but deepstack compact build failed"
            )
        validate_deepstack_alignment(
            deepstack_compact,
            hook_out.get("image_mask"),
            n_visual_kept=n_kept,
            where="prepare_geopact_projector_inputs",
        )
    elif has_deepstack and n_kept == 0:
        raise RuntimeError(
            "GeoPACT prepare_geopact_projector_inputs: model has deepstack but pruning kept 0 visual tokens"
        )

    write_current_frame_compact_cache(
        inner,
        image_paths,
        token_ranges,
        seq_keep,
        image_embeds,
        per_frame_deepstack,
        image_grid_thw,
    )

    compact_stats = hook_out.get("geopact_compact_stats") or {}
    compact_stats["compact_backend"] = "source"
    compact_stats["pruning_ms"] = round(pruning_ms, 2)
    compact_stats["geopact_prepare_ms"] = round((time.perf_counter() - prepare_t0) * 1000.0, 2)
    compact_stats["tokens_before_pruning"] = int(compact_stats.get("orig_seq_len", input_ids.shape[1]))
    compact_stats["tokens_after_pruning"] = int(
        compact_stats.get("compact_seq_len", compact_stats["tokens_before_pruning"])
    )
    compact_stats["grid_rewritten"] = rewrite_grid_enabled() and not uses_origpos
    compact_stats["uses_origpos_grid"] = uses_origpos
    compact_stats["compact_deepstack_layers"] = len(deepstack_compact or [])
    compact_stats["compact_deepstack_rows"] = (
        int(deepstack_compact[0].shape[0]) if deepstack_compact else 0
    )
    compact_stats.update(vision_cache_stats)
    inner._geopact_last_compact_stats = compact_stats

    prefill_payload = {
        "input_ids": hook_out["input_ids"],
        "inputs_embeds": hook_out["inputs_embeds"],
        "attention_mask": hook_out.get("attention_mask", attention_mask),
        "image_grid_thw": forward_grid,
        "image_mask": hook_out.get("image_mask"),
        "position_ids": hook_out.get("position_ids"),
        "rope_deltas": hook_out.get("rope_deltas"),
        "deepstack_image_embeds": deepstack_compact,
        "force_rope_recompute": bool(hook_out.get("force_rope_recompute")),
        "visual_keep_flat_indices": hook_out.get("visual_keep_flat_indices"),
        "mrope_source_input_ids": hook_out.get("mrope_source_input_ids"),
        "mrope_keep_mask": hook_out.get("mrope_keep_mask"),
        "geopact_compact_stats": compact_stats,
    }
    prefill_payload = apply_video_to_compact_prefill(
        inner, prefill_payload, inputs, hook_active=True
    )
    prefill_payload = _finalize_carrier_prefill_payload(inner, prefill_payload)
    inner._geopact_prefill_payload = prefill_payload

    out = {
        "input_ids": prefill_payload["input_ids"],
        "inputs_embeds": prefill_payload["inputs_embeds"],
        "attention_mask": prefill_payload.get("attention_mask", attention_mask),
        "image_grid_thw": forward_grid,
        "position_ids": prefill_payload.get("position_ids"),
        "rope_deltas": prefill_payload.get("rope_deltas"),
    }
    return {k: v for k, v in out.items() if v is not None}
