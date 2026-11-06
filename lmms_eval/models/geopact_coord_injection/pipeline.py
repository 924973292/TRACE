"""Unified Module C build entry (UDSI pipeline)."""
from __future__ import annotations

import os
import random as _random
from typing import Any, Optional, Sequence, Tuple

from lmms_eval.models.geopact_coord_injection.anchor_provider import select_anchors
from lmms_eval.models.geopact_coord_injection.record import CoordInjectionMode, CoordInjectionRecord
from lmms_eval.models.geopact_coord_injection.renderer import render_appendix


def resolve_coord_format(policy: str, env: Optional[dict[str, str]] = None) -> str:
    env_map = env if env is not None else os.environ
    fmt = (env_map.get("GEOPACT_COORD_FORMAT") or "").strip().lower()
    valid = {
        "spatial_plain",
        "spatial_ranked",
        "spatial_box",
        "sdb",
        "compact_center",
        "legacy",
        "sidecar_bbox",
        "sidecar_bbox_role",
        "sidecar_bbox_role_text",
        "sidecar_keep",
    }
    if fmt in valid:
        return fmt
    pol = str(policy or "").lower()
    if "coord_sdb" in pol or pol.endswith("_coord_sdb") or "_coord_sdb_" in pol:
        return "sdb"
    if "coord_ephemeral" in pol:
        return "legacy"
    return "spatial_plain"


def build_coord_injection(
    *,
    selected_miu_indices: Sequence[int],
    keep_token_set: set,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    policy: str,
    coord_size: Optional[Tuple[int, int]] = None,
    miu_order: Optional[Sequence[int]] = None,
    ephemeral: bool = False,
    metadata_only: bool = True,
) -> CoordInjectionRecord:
    """Build spatial appendix record from prune survivors."""
    from lmms_eval.models.geopact_coord_appendix import record_coord_build_meta

    topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
    topk_policy = 0
    for kw in policy.lower().split("_"):
        if kw.startswith("topk") and len(kw) > 4:
            try:
                topk_policy = int(kw[4:])
            except ValueError:
                pass
    topk = topk_policy or topk_env or 0

    shuffle_coords = "coord_shuffle" in (policy or "").lower()
    random_coords = "coord_random" in (policy or "").lower()
    coord_w = max(1, int(coord_size[0])) if coord_size else 1000
    coord_h = max(1, int(coord_size[1])) if coord_size else 1000
    coord_fmt = resolve_coord_format(policy)
    order = list(miu_order if miu_order is not None else selected_miu_indices)

    anchors, surviving = select_anchors(
        policy,
        order,
        miu_token_idxs=miu_token_idxs,
        mius=mius,
        keep_token_set=keep_token_set,
        topk=topk,
        coord_size=(coord_w, coord_h),
        random_coords=random_coords,
        shuffle_coords=shuffle_coords,
    )

    if not anchors:
        record_coord_build_meta({})
        return CoordInjectionRecord.empty()

    text = render_appendix(
        coord_fmt,
        anchors,
        legacy_entries=surviving if coord_fmt == "legacy" else None,
        sidecar_entries=surviving,
    )

    if metadata_only and not ephemeral:
        mode = CoordInjectionMode.METADATA_ONLY
    elif ephemeral:
        mode = CoordInjectionMode.EPHEMERAL
    else:
        mode = CoordInjectionMode.PERSISTENT

    rec = CoordInjectionRecord(
        text=text or "",
        token_count=0,
        insert_pos=-1,
        mode=mode,
        format_id=coord_fmt,
        anchors=tuple(anchors),
        anchor_count=len(anchors),
        ephemeral=ephemeral,
        spliced=False,
    )
    meta = rec.to_stats_dict()
    meta.pop("anchors", None)  # keep LAST_COORD_BUILD_META small
    record_coord_build_meta(meta)
    return rec
