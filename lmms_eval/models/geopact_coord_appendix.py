"""Schema-Deferred Binding (SDB) appendix builder for GeoPACT Module C.

Legacy compatibility layer — new code should use geopact_coord_injection.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

from lmms_eval.models.geopact_coord_injection.pipeline import resolve_coord_format as _resolve_fmt
from lmms_eval.models.geopact_coord_injection.renderer import (
    render_compact_center,
    render_legacy_miu,
    render_sdb,
    render_spatial_box,
    render_spatial_plain,
    render_spatial_ranked,
)

LAST_COORD_BUILD_META: dict[str, Any] = {}


def to_qwen1000(v: float, *, scale: int = 1000) -> int:
    """Quantize [0,1] normalized coord to Qwen [0,1000] integer grid."""
    s = max(1, int(scale))
    return int(round(max(0.0, min(1.0, float(v))) * s))


def miu_center_qwen1000(
    box: Sequence[float],
    *,
    coord_size: Optional[Tuple[int, int]] = None,
) -> tuple[int, int]:
    """Map normalized xyxy box center to output-space integer (cx, cy)."""
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    cw = max(1, int(coord_size[0])) if coord_size else 1000
    ch = max(1, int(coord_size[1])) if coord_size else 1000
    cx = to_qwen1000((x1 + x2) / 2.0, scale=cw)
    cy = to_qwen1000((y1 + y2) / 2.0, scale=ch)
    return cx, cy


def box_to_qwen1000_xyxy(
    box: Sequence[float],
    *,
    coord_size: Optional[Tuple[int, int]] = None,
) -> tuple[int, int, int, int]:
    """Map normalized xyxy box to Qwen [0,1000] integer corners (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    cw = max(1, int(coord_size[0])) if coord_size else 1000
    ch = max(1, int(coord_size[1])) if coord_size else 1000
    return (
        to_qwen1000(x1, scale=cw),
        to_qwen1000(y1, scale=ch),
        to_qwen1000(x2, scale=cw),
        to_qwen1000(y2, scale=ch),
    )


def schema_binding_contract() -> str:
    """Deprecated: contains output-schema words; debug / legacy SDB only."""
    return (
        "Choose action_type first; anchors optional for coord actions ([0,1000]). "
        "Anchors do NOT imply action_type. "
    )


def _box_token(cx: int, cy: int) -> str:
    return f"<|box_start|>({int(cx)},{int(cy)})<|box_end|>"


def build_sdb_appendix(
    anchors: Sequence[tuple[int, int]],
    *,
    rank_prefix: str = "A",
) -> str:
    return render_sdb(anchors, rank_prefix=rank_prefix)


def build_compact_center_appendix(
    anchors: Sequence[tuple[int, int]],
    *,
    rank_prefix: str = "A",
) -> str:
    return render_compact_center(anchors, rank_prefix=rank_prefix)


def build_legacy_miu_lines(
    entries: Sequence[dict[str, Any]],
) -> str:
    return render_legacy_miu(entries)


def build_spatial_plain_appendix(anchors: Sequence[tuple[int, int]]) -> str:
    return render_spatial_plain(anchors)


def build_spatial_ranked_appendix(
    anchors: Sequence[tuple[int, int]],
    *,
    rank_prefix: str = "P",
) -> str:
    return render_spatial_ranked(anchors, rank_prefix=rank_prefix)


def build_spatial_box_appendix(
    anchors: Sequence[tuple[int, int]],
    *,
    rank_prefix: str = "P",
) -> str:
    return render_spatial_box(anchors, rank_prefix=rank_prefix)


def resolve_coord_format(policy: str, env: Optional[dict[str, str]] = None) -> str:
    """Return coord appendix format id."""
    return _resolve_fmt(policy, env=env)


def estimate_token_budget(text: str, tokenizer: Any = None) -> int:
    """Rough token count for verify scripts (tokenizer optional)."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        except Exception:
            pass
    return max(1, len(text) // 4)


def record_coord_build_meta(meta: dict[str, Any]) -> None:
    LAST_COORD_BUILD_META.clear()
    LAST_COORD_BUILD_META.update(meta)


def consume_coord_build_meta() -> dict[str, Any]:
    return dict(LAST_COORD_BUILD_META)
