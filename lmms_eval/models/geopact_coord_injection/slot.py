"""Declarative injection slot resolution."""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional


class InjectionSlot(str, Enum):
    BEFORE_GEN_TAIL = "before_gen_tail"
    AFTER_LAST_VISION = "after_last_vision"
    SEQUENCE_TAIL = "sequence_tail"


def resolve_slot_from_env(policy: str = "", env: Optional[dict[str, str]] = None) -> InjectionSlot:
    env_map = env if env is not None else os.environ
    raw = (env_map.get("GEOPACT_COORD_SLOT") or "").strip().lower()
    if raw in {s.value for s in InjectionSlot}:
        return InjectionSlot(raw)
    pol = str(policy or "").lower()
    if "coord_slot_before_gen" in pol or "coord_ephemeral" in pol:
        return InjectionSlot.BEFORE_GEN_TAIL
    if "coord_tail" in pol:
        return InjectionSlot.SEQUENCE_TAIL
    if "coord_center" in pol:
        return InjectionSlot.AFTER_LAST_VISION
    # Ephemeral dialogue default
    if (env_map.get("GEOPACT_COORD_EPHEMERAL") or "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return InjectionSlot.BEFORE_GEN_TAIL
    return InjectionSlot.AFTER_LAST_VISION


def resolve_insert_pos(
    slot: InjectionSlot,
    *,
    compact_seq_len: int,
    gen_tail_len: int,
    last_vision_pos: int,
    ephemeral_mode: bool,
) -> int:
    """Return token index at which coord tokens are inserted."""
    if slot == InjectionSlot.BEFORE_GEN_TAIL:
        if gen_tail_len > 0 and gen_tail_len <= compact_seq_len:
            return int(compact_seq_len - gen_tail_len)
        # step-0 / no gen tail: fall back after last vision (GAEC + UDSI safe path)
        if last_vision_pos >= 0:
            return int(last_vision_pos + 1)
        if compact_seq_len > 0:
            return int(compact_seq_len)
        raise RuntimeError(
            "GeoPACT coord splice BEFORE_GEN_TAIL fallback failed: empty compact sequence"
        )
    if slot == InjectionSlot.SEQUENCE_TAIL:
        return int(compact_seq_len)
    # AFTER_LAST_VISION
    if last_vision_pos >= 0:
        return int(last_vision_pos + 1)
    return int(compact_seq_len)


def last_vision_index(compact_ids_row: Any, image_tok_id: int) -> int:
    """Index of last image pad token in compact sequence, or -1."""
    if image_tok_id < 0:
        return -1
    try:
        import torch

        c_img = compact_ids_row == image_tok_id
        if not c_img.any():
            return -1
        return int(c_img.nonzero(as_tuple=False)[-1].item())
    except Exception:
        return -1
