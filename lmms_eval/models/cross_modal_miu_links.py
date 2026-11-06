"""Cross-modal MIU links for multimodal evidence resonance.

Module A assigns each MIU a modality-native information density.  This module
keeps correspondence separate from density: links are built only from structural
alignment, then audio/video density is propagated onto current-screen image MIUs.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_box(box) -> Optional[tuple[float, float, float, float]]:
    if not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _video_image_alignment(video_box, image_box) -> float:
    v_box = _as_box(video_box)
    i_box = _as_box(image_box)
    if v_box is None or i_box is None:
        return 0.0
    inter = _intersection_area(v_box, i_box)
    if inter <= 0.0:
        return 0.0
    union = _box_area(v_box) + _box_area(i_box) - inter
    iou = inter / max(union, 1e-9)
    coverage = inter / max(_box_area(v_box), 1e-9)
    cx, cy = 0.5 * (v_box[0] + v_box[2]), 0.5 * (v_box[1] + v_box[3])
    center_in = 1.0 if (i_box[0] <= cx <= i_box[2] and i_box[1] <= cy <= i_box[3]) else 0.0
    return float(max(iou, coverage, center_in))


def _time_center(miu: dict) -> float:
    if miu.get("time_center") is not None:
        return max(0.0, min(1.0, _safe_float(miu.get("time_center"), 0.5)))
    t = miu.get("time")
    if isinstance(t, (list, tuple)) and len(t) >= 2:
        return max(0.0, min(1.0, 0.5 * (_safe_float(t[0], 0.0) + _safe_float(t[1], 1.0))))
    return 0.5


def _rank_temporal_links(audio: list[dict], video: list[dict]) -> list[list[float]]:
    if not audio or not video:
        return [[0.0 for _ in video] for _ in audio]
    video_times = [_time_center(v) for v in video]
    links: list[list[float]] = []
    for aud in audio:
        a_time = _time_center(aud)
        order = sorted(range(len(video_times)), key=lambda idx: abs(a_time - video_times[idx]))
        row = [0.0 for _ in video]
        if len(order) == 1:
            row[order[0]] = 1.0
        else:
            denom = max(len(order) - 1, 1)
            for rank, video_idx in enumerate(order):
                row[video_idx] = float(1.0 - rank / denom)
        links.append(row)
    return links


def _rank_normalize(values: Iterable[float]) -> list[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    v_min = min(vals)
    v_max = max(vals)
    if v_max <= 0.0:
        return [0.0 for _ in vals]
    if abs(v_max - v_min) < 1e-12:
        return [1.0 for _ in vals]
    denom = max(v_max - v_min, 1e-12)
    return [float((v - v_min) / denom) for v in vals]


def _node_density(miu: dict) -> float:
    modality = str(miu.get("modality", "image")).lower()
    if modality in {"video", "audio"}:
        return _safe_float(
            miu.get(
                "modality_energy",
                miu.get("interaction_energy", miu.get("info_density", miu.get("info_density_proxy", 0.0))),
            )
        )
    return _safe_float(miu.get("info_density", miu.get("info_density_proxy", 0.0)))


def _aggregate_edges(edges: list[tuple[float, float]], policy: str) -> float:
    if not edges:
        return 0.0
    if "graph_max" in policy or "max_av_support" in policy or "legacy_av_max" in policy:
        return float(max(weight * density for weight, density in edges))
    if "graph_softmax" in policy or "softmax_av_support" in policy:
        logits = [4.0 * max(0.0, float(weight)) for weight, _density in edges]
        max_logit = max(logits)
        exp_vals = [math.exp(v - max_logit) for v in logits]
        denom = max(sum(exp_vals), 1e-12)
        return float(sum((alpha / denom) * weight * density for alpha, (weight, density) in zip(exp_vals, edges)))
    denom = max(sum(max(0.0, weight) for weight, _density in edges), 1e-12)
    return float(sum(max(0.0, weight) * density for weight, density in edges) / denom)


def attach_cross_modal_resonance(mius: list[dict], policy: str = "") -> list[dict]:
    """Attach AV support fields to image MIUs without retaining AV tokens.

    Density is node strength; correspondence is structure.  Video supports image
    MIUs through spatial alignment.  Audio supports image MIUs only through
    audio-to-video temporal synchrony composed with video-to-image alignment.
    """
    policy = str(policy or "").lower()
    image = [m for m in mius if str(m.get("modality", "image")) == "image" and _as_box(m.get("box")) is not None]
    video = [
        m for m in mius
        if str(m.get("modality", "")) == "video"
        and _as_box(m.get("box")) is not None
        and _box_area(_as_box(m.get("box"))) <= 0.75
    ]
    audio = [m for m in mius if str(m.get("modality", "")) == "audio"]

    use_video = "no_video_support" not in policy and "no_av_support" not in policy and "image_only" not in policy
    use_audio = "no_audio_support" not in policy and "no_av_support" not in policy and "image_only" not in policy

    video_image = [
        [_video_image_alignment(v.get("box"), im.get("box")) for im in image]
        for v in video
    ]
    audio_video = _rank_temporal_links(audio, video) if use_audio and use_video else []

    multimodal_raw: list[float] = []
    for image_idx, im in enumerate(image):
        rho_image = _safe_float(im.get("info_density", im.get("info_density_proxy", 0.0)))
        video_edges: list[tuple[float, float]] = []
        video_edge_meta: list[dict[str, float | str]] = []
        if use_video:
            for video_idx, v in enumerate(video):
                align = video_image[video_idx][image_idx]
                if align < 0.05:
                    continue
                density = _node_density(v)
                video_edges.append((align, density))
                video_edge_meta.append(
                    {
                        "source": str(v.get("miu_id", f"video_{video_idx}")),
                        "target": str(im.get("miu_id", f"image_{image_idx}")),
                        "type": "video_image_spatial",
                        "weight": float(align),
                        "density": float(density),
                    }
                )
            if len(video_edges) > 3:
                packed = sorted(zip(video_edges, video_edge_meta), key=lambda item: item[0][0] * item[0][1], reverse=True)[:3]
                video_edges = [item[0] for item in packed]
                video_edge_meta = [item[1] for item in packed]
        video_support = _aggregate_edges(video_edges, policy)
        linked_video_count = len(video_edges)

        audio_edges: list[tuple[float, float]] = []
        audio_edge_meta: list[dict[str, float | str]] = []
        linked_audio_ids: set[str] = set()
        if use_audio and use_video:
            for audio_idx, aud in enumerate(audio):
                for video_idx, _v in enumerate(video):
                    if not audio_video:
                        continue
                    temporal = audio_video[audio_idx][video_idx]
                    if temporal < 0.50:
                        continue
                    composed = temporal * video_image[video_idx][image_idx]
                    if composed < 0.05:
                        continue
                    density = _node_density(aud)
                    audio_edges.append((composed, density))
                    audio_id = str(aud.get("miu_id", f"audio_{audio_idx}"))
                    linked_audio_ids.add(audio_id)
                    audio_edge_meta.append(
                        {
                            "source": audio_id,
                            "via": str(video[video_idx].get("miu_id", f"video_{video_idx}")),
                            "target": str(im.get("miu_id", f"image_{image_idx}")),
                            "type": "audio_video_image_composed",
                            "weight": float(composed),
                            "density": float(density),
                        }
                    )
            if len(audio_edges) > 6:
                packed = sorted(zip(audio_edges, audio_edge_meta), key=lambda item: item[0][0] * item[0][1], reverse=True)[:6]
                audio_edges = [item[0] for item in packed]
                audio_edge_meta = [item[1] for item in packed]
                linked_audio_ids = {str(edge.get("source", "")) for edge in audio_edge_meta if edge.get("source")}
        audio_support = _aggregate_edges(audio_edges, policy)
        linked_audio_count = len(linked_audio_ids)

        rho_multimodal = rho_image + video_support + audio_support
        im["rho_image"] = float(rho_image)
        im["rho_video_support"] = float(video_support)
        im["rho_audio_support"] = float(audio_support)
        im["rho_multimodal"] = float(rho_multimodal)
        im["linked_video_count"] = float(linked_video_count)
        im["linked_audio_count"] = float(linked_audio_count)
        im["cross_graph_video_degree"] = float(len(video_edges))
        im["cross_graph_audio_path_count"] = float(len(audio_edges))
        im["cross_graph_video_edges"] = video_edge_meta
        im["cross_graph_audio_edges"] = audio_edge_meta
        multimodal_raw.append(rho_multimodal)

    normalized = _rank_normalize(multimodal_raw)
    for im, value in zip(image, normalized):
        im["rho_multimodal_rank"] = float(value)
        if "raw_multimodal_density" in policy:
            im["info_density"] = float(im.get("rho_multimodal", im.get("info_density", 0.0)))
        elif "no_info_density" not in policy and "no_multimodal_density" not in policy:
            im["info_density"] = float(value)

    return mius


def cross_modal_summary_meta(mius: list[dict]) -> dict[str, float]:
    image = [m for m in mius if str(m.get("modality", "image")) == "image"]
    if not image:
        return {
            "cross_image_count": 0.0,
            "cross_video_support_mean": 0.0,
            "cross_audio_support_mean": 0.0,
            "cross_linked_video_mean": 0.0,
            "cross_linked_audio_mean": 0.0,
        }
    n = float(len(image))
    return {
        "cross_image_count": n,
        "cross_video_support_mean": float(sum(_safe_float(m.get("rho_video_support")) for m in image) / n),
        "cross_audio_support_mean": float(sum(_safe_float(m.get("rho_audio_support")) for m in image) / n),
        "cross_linked_video_mean": float(sum(_safe_float(m.get("linked_video_count")) for m in image) / n),
        "cross_linked_audio_mean": float(sum(_safe_float(m.get("linked_audio_count")) for m in image) / n),
    }