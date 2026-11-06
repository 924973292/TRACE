"""GAEC — Geometry-Anchored Ephemeral Carriers (training-free, MRoPE-native)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from lmms_eval.models.geopact_coord_injection.record import (
    CoordInjectionMode,
    CoordInjectionRecord,
)


def coord_mode() -> str:
    return (os.environ.get("GEOPACT_COORD_MODE") or "text").strip().lower()


def gaec_enabled() -> bool:
    return coord_mode() == "gaec"


def gaec_v2_enabled() -> bool:
    return os.environ.get("GEOPACT_GAEC_V2", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gaec_shuffle_carriers() -> bool:
    return os.environ.get("GEOPACT_GAEC_SHUFFLE_POS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gaec_topk() -> int:
    raw = os.environ.get("GEOPACT_GAEC_TOPK", "") or os.environ.get("GEOPACT_COORD_TOPK", "3")
    try:
        return max(1, int(raw or 3))
    except ValueError:
        return 3


@dataclass
class GAECPayload:
    carrier_embs: torch.Tensor
    carrier_ids: torch.Tensor
    carrier_pos: torch.Tensor
    record: CoordInjectionRecord
    anchor_miu_indices: List[int]


def _topk_miu_indices(selected: Sequence[int], topk: int) -> List[int]:
    order = list(selected)
    if topk > 0:
        order = order[:topk]
    return order


def _box_center_norm(m: dict) -> Tuple[float, float]:
    box = m.get("box")
    if box and len(box) >= 4:
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return (float(m.get("cx", 0.5)), float(m.get("cy", 0.5)))


def _center_compact_from_origpos(
    mi: int,
    mius: Sequence[dict],
    visual_compact_idx: torch.Tensor,
    compact_pos: torch.Tensor,
    row_ids: torch.Tensor,
    image_tok_id: int,
) -> Optional[int]:
    """Pick compact visual index whose origpos (h,w) is nearest MIU bbox center."""
    if mi < 0 or mi >= len(mius):
        return None
    cx, cy = _box_center_norm(mius[mi])

    h_all = compact_pos[1, 0, :].to(dtype=torch.float32)
    w_all = compact_pos[2, 0, :].to(dtype=torch.float32)
    vis = visual_compact_idx
    if vis.numel() == 0:
        return None
    hv = h_all.index_select(0, vis)
    wv = w_all.index_select(0, vis)
    h_min = float(hv.min().item())
    h_max = float(hv.max().item())
    w_min = float(wv.min().item())
    w_max = float(wv.max().item())
    h_span = max(h_max - h_min, 1.0)
    w_span = max(w_max - w_min, 1.0)

    best_ci: Optional[int] = None
    best_d = 1e9
    for ord_i in range(int(vis.numel())):
        ci = int(vis[ord_i].item())
        if int(row_ids[ci]) != int(image_tok_id):
            continue
        nh = (float(hv[ord_i].item()) - h_min) / h_span
        nw = (float(wv[ord_i].item()) - w_min) / w_span
        d = (nh - cy) ** 2 + (nw - cx) ** 2
        if d < best_d:
            best_d = d
            best_ci = ci
    return best_ci


def _center_global_token(
    mi: int,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    keep_token_set: set,
    frame_start: int,
    n_frame: int,
) -> Optional[int]:
    """Legacy: pick kept global seq token in MIU closest to bbox center (flat-index heuristic)."""
    if mi < 0 or mi >= len(miu_token_idxs):
        return None
    candidates = [int(t) for t in miu_token_idxs[mi] if int(t) in keep_token_set]
    if not candidates:
        return None
    cx, cy = _box_center_norm(mius[mi] if mi < len(mius) else {})
    best = candidates[0]
    best_d = 1e9
    grid = max(int(n_frame), 1)
    side = max(int(grid**0.5), 1)
    for t in candidates:
        gy = (t // side + 0.5) / max(side, 1)
        gx = (t % side + 0.5) / max(side, 1)
        d = (gx - cx) ** 2 + (gy - cy) ** 2
        if d < best_d:
            best_d = d
            best = t
    return int(best) + int(frame_start)


def prepare_gaec_context(
    model,
    *,
    selected_miu_indices: Sequence[int],
    keep_token_set: set,
    miu_token_idxs: Sequence[Sequence[int]],
    mius: Sequence[dict],
    frame_start: int,
    n_frame: int,
    topk: int = 3,
    shuffle: bool = False,
) -> None:
    """Store GAEC build inputs on model for _compact_sequence."""
    effective_topk = gaec_topk() if gaec_v2_enabled() else topk
    mi_list = _topk_miu_indices(selected_miu_indices, effective_topk)
    if shuffle or gaec_shuffle_carriers():
        import random as _random

        mi_list = list(mi_list)
        _random.shuffle(mi_list)
    model._geopact_gaec_ctx = {
        "miu_indices": mi_list,
        "keep_token_set": set(keep_token_set),
        "miu_token_idxs": miu_token_idxs,
        "mius": mius,
        "frame_start": int(frame_start),
        "n_frame": int(n_frame),
        "gaec_v2": gaec_v2_enabled(),
    }


def build_gaec_carriers(
    *,
    compact_embs: torch.Tensor,
    compact_ids: torch.Tensor,
    compact_pos: torch.Tensor,
    full_position_ids: torch.Tensor,
    keep_mask: torch.Tensor,
    image_tok_id: int,
    gaec_ctx: dict,
    carrier_token_id: Optional[int] = None,
) -> Optional[GAECPayload]:
    """Build ephemeral carrier tokens with vision-grid MRoPE and anchor visual embs."""
    mi_list: List[int] = list(gaec_ctx.get("miu_indices") or [])
    if not mi_list:
        return None

    miu_token_idxs = gaec_ctx.get("miu_token_idxs") or []
    mius = gaec_ctx.get("mius") or []
    keep_token_set = set(gaec_ctx.get("keep_token_set") or set())
    frame_start = int(gaec_ctx.get("frame_start", 0) or 0)
    n_frame = int(gaec_ctx.get("n_frame", 1) or 1)
    use_v2 = bool(gaec_ctx.get("gaec_v2")) or gaec_v2_enabled()

    row_emb = compact_embs[0]
    row_ids = compact_ids[0]
    visual_compact_idx = (row_ids == int(image_tok_id)).nonzero(as_tuple=False).flatten()

    global_to_compact: Dict[int, int] = {}
    kept_globals = keep_mask.nonzero(as_tuple=False).flatten()
    for ci, gi in enumerate(kept_globals.tolist()):
        if ci < int(compact_ids.shape[1]):
            global_to_compact[int(gi)] = int(ci)

    keep_set = set(int(x) for x in keep_token_set)
    carriers: List[torch.Tensor] = []
    carrier_pos_cols: List[torch.Tensor] = []
    anchor_compact_indices: List[int] = []
    pad_id = int(carrier_token_id if carrier_token_id is not None else 0)
    if pad_id == int(image_tok_id):
        pad_id = 0

    for mi in mi_list:
        anchor_ci: Optional[int] = None
        if use_v2 and compact_pos is not None and visual_compact_idx.numel() > 0:
            anchor_ci = _center_compact_from_origpos(
                mi, mius, visual_compact_idx, compact_pos, row_ids, image_tok_id
            )

        if anchor_ci is not None:
            carriers.append(row_emb[anchor_ci].clone())
            center_global = None
            for g, c in global_to_compact.items():
                if c == anchor_ci:
                    center_global = int(g)
                    break
        else:
            pooled: List[torch.Tensor] = []
            if mi < len(miu_token_idxs):
                for g in miu_token_idxs[mi]:
                    gi = int(g)
                    if gi not in keep_set:
                        continue
                    ci = global_to_compact.get(gi)
                    if ci is not None and int(row_ids[ci]) == int(image_tok_id):
                        pooled.append(row_emb[ci])
            if not pooled:
                continue
            if use_v2 and len(pooled) == 1:
                carriers.append(pooled[0].clone())
            else:
                carriers.append(torch.stack(pooled, dim=0).mean(dim=0))

            center_global = _center_global_token(
                mi, miu_token_idxs, mius, keep_set, frame_start, n_frame
            )
            anchor_ci = global_to_compact.get(int(center_global) - frame_start) if center_global else None
            if anchor_ci is None and center_global is not None:
                anchor_ci = global_to_compact.get(int(center_global))

        if center_global is None:
            center_global = _center_global_token(
                mi, miu_token_idxs, mius, keep_set, frame_start, n_frame
            )
        if center_global is None or int(center_global) not in global_to_compact:
            if anchor_ci is not None:
                for g, c in global_to_compact.items():
                    if c == anchor_ci:
                        center_global = int(g)
                        break
            else:
                center_global = int(kept_globals[0].item()) if kept_globals.numel() else 0

        pos_col = full_position_ids[:, 0, int(center_global)].clone()
        carrier_pos_cols.append(pos_col)
        anchor_compact_indices.append(int(anchor_ci) if anchor_ci is not None else -1)

    if not carriers:
        return None

    if gaec_shuffle_carriers() and len(carrier_pos_cols) > 1:
        import random as _random

        perm = list(range(len(carrier_pos_cols)))
        _random.shuffle(perm)
        carrier_pos_cols = [carrier_pos_cols[i] for i in perm]

    cemb = torch.stack(carriers, dim=0).unsqueeze(0).to(
        device=compact_embs.device, dtype=compact_embs.dtype
    )
    n_c = int(cemb.shape[1])
    cid = torch.full(
        (1, n_c),
        pad_id,
        dtype=compact_ids.dtype,
        device=compact_ids.device,
    )
    pos_stack = torch.stack(carrier_pos_cols, dim=-1)
    carrier_pos = pos_stack.unsqueeze(1)

    rec = CoordInjectionRecord(
        text="",
        token_count=n_c,
        insert_pos=-1,
        mode=CoordInjectionMode.EPHEMERAL,
        format_id="gaec_v2" if use_v2 else "gaec",
        anchor_count=len(carriers),
        ephemeral=True,
        spliced=True,
    )
    return GAECPayload(
        carrier_embs=cemb,
        carrier_ids=cid,
        carrier_pos=carrier_pos,
        record=rec,
        anchor_miu_indices=mi_list,
    )


def splice_gaec_into_compact(
    compact_ids: torch.Tensor,
    compact_embs: torch.Tensor,
    compact_attn: Optional[torch.Tensor],
    compact_pos: torch.Tensor,
    compact_rope: torch.Tensor,
    ins: int,
    payload: GAECPayload,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Insert GAEC carriers at ``ins`` with vision-grid MRoPE columns."""
    n_c = int(payload.carrier_embs.shape[1])
    cid_t = payload.carrier_ids.to(compact_ids.device)
    cemb = payload.carrier_embs.to(compact_embs.device, dtype=compact_embs.dtype)

    compact_ids = torch.cat([compact_ids[:, :ins], cid_t, compact_ids[:, ins:]], dim=1)
    compact_embs = torch.cat([compact_embs[:, :ins], cemb, compact_embs[:, ins:]], dim=1)
    if compact_attn is not None:
        ones = torch.ones(1, n_c, dtype=compact_attn.dtype, device=compact_attn.device)
        compact_attn = torch.cat([compact_attn[:, :ins], ones, compact_attn[:, ins:]], dim=1)

    cpos = payload.carrier_pos.to(device=compact_pos.device, dtype=compact_pos.dtype)
    compact_pos = torch.cat([compact_pos[:, :, :ins], cpos, compact_pos[:, :, ins:]], dim=2)
    batch = compact_ids.shape[0]
    seq_row = compact_ids.shape[1]
    deltas = [int(compact_pos[:, b].max().item()) + 1 - seq_row for b in range(batch)]
    compact_rope = torch.tensor(
        deltas,
        device=compact_pos.device,
        dtype=torch.long,
    ).unsqueeze(1)
    return compact_ids, compact_embs, compact_attn, compact_pos, compact_rope
