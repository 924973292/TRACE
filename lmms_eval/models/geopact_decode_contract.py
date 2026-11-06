"""Pinned MRoPE decode contract for GeoPACT origpos generate paths.

Invariant: prefill uses full compact ``position_ids``; each decode step continues
from ``decode_anchor + (cache_position - pinned_cache_start)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from loguru import logger as eval_logger


def strict_mrope_enabled() -> bool:
    raw = os.environ.get("GEOPACT_STRICT_MROPE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def compute_pinned_cache_start(*, past_kv_len: int, prompt_token_count: int) -> int:
    """Global cache index of the last prompt token before decode."""
    past = int(past_kv_len or 0)
    count = int(prompt_token_count or 0)
    if count <= 0:
        return max(past - 1, 0)
    return past + count - 1


def _past_kv_len_from_payload(gen_payload: Dict[str, Any]) -> int:
    past_kv = gen_payload.get("past_key_values")
    if past_kv is None:
        return 0
    try:
        return int(past_kv.get_seq_length())
    except Exception:
        return 0


def _prompt_token_count_from_payload(gen_payload: Dict[str, Any]) -> int:
    pinned = gen_payload.get("_geopact_pinned_cache_start")
    past_len = _past_kv_len_from_payload(gen_payload)
    if pinned is not None:
        try:
            return int(pinned) - past_len + 1
        except Exception:
            pass
    pos = gen_payload.get("position_ids")
    if isinstance(pos, torch.Tensor) and pos.ndim >= 1:
        return int(pos.shape[-1])
    embeds = gen_payload.get("inputs_embeds")
    if isinstance(embeds, torch.Tensor) and embeds.ndim >= 2:
        return int(embeds.shape[1])
    ids = gen_payload.get("input_ids")
    if isinstance(ids, torch.Tensor) and ids.ndim >= 2:
        return int(ids.shape[1])
    return 0


@dataclass
class PinnedDecodeContract:
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    pinned_cache_start: int


def contract_complete(gen_payload: Dict[str, Any]) -> bool:
    if not isinstance(gen_payload, dict):
        return False
    if gen_payload.get("position_ids") is None:
        return False
    rope = gen_payload.get("rope_deltas")
    if gen_payload.get("_geopact_rope_deltas") is None and rope is None:
        return False
    if gen_payload.get("_geopact_pinned_cache_start") is None:
        return False
    return True


def resolve_rope_for_step(
    gen_payload: Dict[str, Any],
    *,
    model_rope: Any = None,
    flow_mode: str = "none",
    past_kv_len: int = 0,
) -> Dict[str, Any]:
    """Promote rope for pinned decode.

    Dialogue KV paths may use ``inner.rope_deltas``. Baseline full re-encode
    (``flow_mode=none``, ``past_kv_len=0``) may use same-step prefill rope
    after ``clear_geopact_mrope_decode_state`` at step boundary — never stale
    suffix rope from a prior incremental step.
    """
    if not isinstance(gen_payload, dict):
        return gen_payload
    out = dict(gen_payload)
    rope = out.get("rope_deltas")
    if rope is not None and out.get("_geopact_rope_deltas") is None:
        out["_geopact_rope_deltas"] = rope
    use_model = flow_mode in ("incremental", "prune_rewrite") or (
        flow_mode == "none" and int(past_kv_len or 0) == 0
    )
    if out.get("_geopact_rope_deltas") is None and use_model and model_rope is not None:
        out["_geopact_rope_deltas"] = model_rope
    return out


def promote_pinned_rope_fields(
    gen_payload: Dict[str, Any],
    *,
    model_rope: Any = None,
    flow_mode: str = "incremental",
    past_kv_len: int = 0,
) -> Dict[str, Any]:
    """Backward-compatible alias for ``resolve_rope_for_step``."""
    return resolve_rope_for_step(
        gen_payload,
        model_rope=model_rope,
        flow_mode=flow_mode,
        past_kv_len=past_kv_len,
    )


def should_use_pinned_generate(gen_payload: Dict[str, Any]) -> bool:
    """Pinned hook when origpos ``position_ids`` and rope (payload or promoted) are present."""
    if not isinstance(gen_payload, dict) or gen_payload.get("position_ids") is None:
        return False
    return (
        gen_payload.get("_geopact_rope_deltas") is not None
        or gen_payload.get("rope_deltas") is not None
    )


def attach_pinned_decode_contract(
    gen_payload: Dict[str, Any],
    *,
    past_kv_len: Optional[int] = None,
    prompt_token_count: Optional[int] = None,
    logical_prompt_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Ensure ``_geopact_pinned_cache_start`` is set when origpos MRoPE is active."""
    if not isinstance(gen_payload, dict):
        return gen_payload
    if gen_payload.get("position_ids") is None:
        return gen_payload

    out = dict(gen_payload)
    rope = out.get("rope_deltas")
    if rope is not None and out.get("_geopact_rope_deltas") is None:
        out["_geopact_rope_deltas"] = rope

    if out.get("_geopact_pinned_cache_start") is not None:
        return out

    past_len = int(past_kv_len) if past_kv_len is not None else _past_kv_len_from_payload(out)
    if prompt_token_count is None:
        if logical_prompt_len is not None and past_len == 0:
            prompt_token_count = int(logical_prompt_len)
        else:
            prompt_token_count = _prompt_token_count_from_payload(out)

    pinned = compute_pinned_cache_start(
        past_kv_len=past_len,
        prompt_token_count=int(prompt_token_count or 0),
    )
    out["_geopact_pinned_cache_start"] = int(pinned)
    if not strict_mrope_enabled() and past_len > 0:
        eval_logger.warning(
            "GeoPACT decode contract: auto-attached pinned_cache_start="
            f"{pinned} past_kv_len={past_len} prompt_tokens={prompt_token_count}"
        )
    return out


def validate_pinned_decode_contract(gen_payload: Dict[str, Any]) -> None:
    """Raise when origpos position_ids are present but the decode anchor is missing."""
    if not should_use_pinned_generate(gen_payload):
        return
    if contract_complete(gen_payload):
        return
    msg = (
        "GeoPACT pinned decode contract incomplete: "
        f"position_ids={gen_payload.get('position_ids') is not None} "
        f"rope_deltas={gen_payload.get('rope_deltas') is not None} "
        f"_geopact_rope_deltas={gen_payload.get('_geopact_rope_deltas') is not None} "
        f"pinned_cache_start={gen_payload.get('_geopact_pinned_cache_start')}"
    )
    if strict_mrope_enabled():
        raise RuntimeError(msg)
    eval_logger.warning(msg)
