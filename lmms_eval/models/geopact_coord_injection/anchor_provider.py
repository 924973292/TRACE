"""Anchor selection from Module B MIU prune output."""
from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence


def _info_density(m: dict) -> float:
    return float(m.get("info_density", m.get("energy", 0.5)) or 0.5)


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def resolve_anchor_policy(env: Optional[dict[str, str]] = None) -> str:
    env_map = env if env is not None else os.environ
    return (env_map.get("GEOPACT_COORD_ANCHOR_POLICY") or "pick_order_topk").strip().lower()


def _conf_score(m: dict) -> float:
    for key in ("det_score", "info_density", "energy"):
        val = m.get(key)
        if val is not None and val != "":
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                continue
    return _info_density(m)


def _make_surviving_entry(
    m: dict,
    mi: int,
    rank: int,
    *,
    coord_size: tuple[int, int],
    cx: int,
    cy: int,
) -> dict:
    from lmms_eval.models.geopact_coord_appendix import box_to_qwen1000_xyxy

    box = m.get("box", (0.0, 0.0, 1.0, 1.0))
    text = str(m.get("text") or "").strip()[:24]
    return {
        "rank": int(rank),
        "type": m.get("type", "widget"),
        "cx": int(cx),
        "cy": int(cy),
        "box_xyxy": box_to_qwen1000_xyxy(box, coord_size=coord_size),
        "text": text,
        "conf": round(_conf_score(m), 3),
        "energy": float(m.get("energy", 0.5) or 0.5),
        "miu_index": int(mi),
    }


def select_anchors_pick_order_topk(
    order: Sequence[int],
    *,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    keep_token_set: set,
    topk: int,
    coord_size: tuple[int, int],
    random_coords: bool = False,
    shuffle_coords: bool = False,
    rng: Any = None,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Baseline: first topk MIUs in pick order that survive prune."""
    from lmms_eval.models.geopact_coord_appendix import miu_center_qwen1000
    import random as _random

    if topk > 0:
        order = list(order)[:topk]

    surviving: list[dict] = []
    for mi in order:
        if mi >= len(miu_token_idxs):
            continue
        if not any(ti in keep_token_set for ti in miu_token_idxs[mi]):
            continue
        m = mius[mi]
        box = m.get("box", (0.0, 0.0, 1.0, 1.0))
        if random_coords:
            cw, ch = coord_size
            cx = int(_random.random() * cw)
            cy = int(_random.random() * ch)
        else:
            cx, cy = miu_center_qwen1000(box, coord_size=coord_size)
        surviving.append(
            _make_surviving_entry(
                m, mi, len(surviving), coord_size=coord_size, cx=cx, cy=cy
            )
        )

    if shuffle_coords and surviving:
        coords = [(d["cx"], d["cy"]) for d in surviving]
        _random.shuffle(coords)
        for d, (cx, cy) in zip(surviving, coords):
            d["cx"], d["cy"] = cx, cy

    anchors = [(int(d["cx"]), int(d["cy"])) for d in surviving]
    return anchors, surviving


def select_anchors_spatial_diverse_topk(
    order: Sequence[int],
    *,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    keep_token_set: set,
    topk: int,
    coord_size: tuple[int, int],
    min_sep: float = 0.08,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Greedy spatial diversity on normalized box centers."""
    from lmms_eval.models.geopact_coord_appendix import miu_center_qwen1000

    candidates: list[dict] = []
    for mi in order:
        if mi >= len(miu_token_idxs):
            continue
        if not any(ti in keep_token_set for ti in miu_token_idxs[mi]):
            continue
        m = mius[mi]
        box = m.get("box", (0.0, 0.0, 1.0, 1.0))
        cx, cy = miu_center_qwen1000(box, coord_size=coord_size)
        entry = _make_surviving_entry(
            m, mi, len(candidates), coord_size=coord_size, cx=cx, cy=cy
        )
        entry["nx"] = (float(box[0]) + float(box[2])) / 2.0
        entry["ny"] = (float(box[1]) + float(box[3])) / 2.0
        candidates.append(entry)

    k = topk if topk > 0 else len(candidates)
    selected: list[dict] = []
    for cand in candidates:
        if len(selected) >= k:
            break
        ok = True
        for s in selected:
            dist = math.hypot(cand["nx"] - s["nx"], cand["ny"] - s["ny"])
            if dist < min_sep:
                ok = False
                break
        if ok:
            cand["rank"] = len(selected)
            selected.append(cand)

    # Fill remaining slots from candidates if diversity filter was too strict
    if len(selected) < k:
        for cand in candidates:
            if len(selected) >= k:
                break
            if cand not in selected:
                cand = dict(cand)
                cand["rank"] = len(selected)
                selected.append(cand)

    anchors = [(int(d["cx"]), int(d["cy"])) for d in selected]
    return anchors, selected


def select_anchors_keep_survivor_rank(
    order: Sequence[int],
    *,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    keep_token_set: set,
    topk: int,
    coord_size: tuple[int, int],
    min_area: float = 0.0005,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Filter tiny MIU boxes, then pick_order topk."""
    filtered_order = []
    for mi in order:
        if mi >= len(mius):
            continue
        if _box_area(mius[mi].get("box", (0, 0, 1, 1))) >= min_area:
            filtered_order.append(mi)
    return select_anchors_pick_order_topk(
        filtered_order,
        miu_token_idxs=miu_token_idxs,
        mius=mius,
        keep_token_set=keep_token_set,
        topk=topk,
        coord_size=coord_size,
    )


def select_anchors(
    policy: str,
    order: Sequence[int],
    *,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    keep_token_set: set,
    topk: int,
    coord_size: tuple[int, int],
    random_coords: bool = False,
    shuffle_coords: bool = False,
) -> tuple[list[tuple[int, int]], list[dict]]:
    anchor_policy = resolve_anchor_policy()
    common = dict(
        miu_token_idxs=miu_token_idxs,
        mius=mius,
        keep_token_set=keep_token_set,
        topk=topk,
        coord_size=coord_size,
    )
    if anchor_policy == "spatial_diverse_topk":
        return select_anchors_spatial_diverse_topk(order, **common)
    if anchor_policy == "keep_survivor_rank":
        return select_anchors_keep_survivor_rank(order, **common)
    return select_anchors_pick_order_topk(
        order,
        **common,
        random_coords=random_coords,
        shuffle_coords=shuffle_coords,
    )
