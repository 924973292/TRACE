"""KV commit / MRoPE lifecycle helpers for coord injection."""
from __future__ import annotations

from typing import Any, List, Sequence

from lmms_eval.models.geopact_coord_injection.record import (
    CoordInjectionMode,
    CoordInjectionRecord,
    record_from_stats,
)


def strip_for_kv_commit(
    compact_ids: Sequence[int],
    *,
    record: CoordInjectionRecord | None = None,
    stats: dict | None = None,
    gen_tail_len: int = 0,
) -> List[int]:
    """Remove ephemeral coord tokens and generation tail for dialogue KV boundary.

    Uses insert_pos when available (precise); falls back to tail subtraction
    only for legacy stats without insert_pos.
    """
    rec = record or record_from_stats(stats or {})
    compact = list(compact_ids)
    end = len(compact)

    if gen_tail_len > 0:
        if gen_tail_len >= end:
            raise RuntimeError(
                f"GeoPACT dialogue KV: generation tail length {gen_tail_len} "
                f">= compact prompt len {end}"
            )
        end -= gen_tail_len

    if rec.ephemeral and rec.spliced and rec.token_count > 0:
        n = int(rec.token_count)
        full = list(compact_ids)
        if gen_tail_len > 0:
            if gen_tail_len >= len(full):
                raise RuntimeError(
                    f"GeoPACT dialogue KV: generation tail length {gen_tail_len} "
                    f">= compact prompt len {len(full)}"
                )
            full = full[: len(compact_ids) - gen_tail_len]
        if rec.insert_pos >= 0:
            ins = int(rec.insert_pos)
            if 0 <= ins and ins + n <= len(full):
                boundary = full[:ins] + full[ins + n :]
                if not boundary:
                    raise RuntimeError(
                        "GeoPACT dialogue KV: empty boundary after coord insert_pos strip"
                    )
                return boundary
            # Stale insert_pos from a prior step — ephemeral coord is before gen tail.
        if n >= len(full):
            raise RuntimeError(
                f"GeoPACT dialogue KV: ephemeral coord token count {n} "
                f">= compact boundary len {len(full)}"
            )
        return full[: len(full) - n]

    boundary = compact[:end]
    if not boundary:
        raise RuntimeError(
            "GeoPACT dialogue KV: empty compact cache boundary after strip"
        )
    return boundary


def mrope_virtual_compact_len(compact_full_len: int, record: CoordInjectionRecord) -> int:
    """Length of compact sequence excluding ephemeral spliced coord tokens."""
    if record.ephemeral and record.spliced and record.token_count > 0:
        return max(0, int(compact_full_len) - int(record.token_count))
    return int(compact_full_len)


def coord_spliced_tokens_for_audit(stats: dict) -> int:
    """Tokens inserted into compact seq (for keep_mask_kept + splice == compact_len)."""
    rec = record_from_stats(stats)
    if rec.spliced:
        return int(rec.token_count)
    return int(stats.get("coord_spliced_tokens", stats.get("coord_ephemeral_tokens", 0)) or 0)
