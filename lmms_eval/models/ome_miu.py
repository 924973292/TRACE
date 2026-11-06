"""Lightweight omni-modal MIU construction for GUI action compression.

The module intentionally keeps each modality extractor simple and explicit:
image MIUs come from GUI components, video MIUs from visual state changes, and
audio MIUs from timestamped onsets/speech-like activity.  Cross-modal structure
is represented by a small set of shared fields (`time`, `box`, `sync_group`,
`linked_component`) so the downstream MMR selector can introduce relevance,
diversity, and complementarity without a large rule system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PIL import Image


Miu = dict


@dataclass(frozen=True)
class OMEConfig:
    max_video_events: int = 8
    max_audio_events: int = 8
    sync_window: float = 0.14


def _as_box_tuple(box) -> Optional[tuple[float, float, float, float]]:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(_box_area(a) + _box_area(b) - inter, 1e-6)


def _center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _role_prior(role: str, modality: str, source: str = "") -> float:
    role = role.lower()
    source = source.lower()
    if source == "skeleton" or role in {"skeleton", "boundary", "container", "panel"}:
        return 0.72
    if modality == "image" and role in {"target", "label", "icon_button", "text", "text_like"}:
        return 0.80
    if modality == "video" and role in {"appear", "disappear", "change", "scroll", "focus", "loading"}:
        return 0.76
    if modality == "audio" and role in {"onset", "speech", "audio_event"}:
        return 0.68
    if role in {"competitor", "state_witness", "av_sync"}:
        return 0.74
    return 0.50


def normalize_image_mius(gui_mius: list[Miu]) -> list[Miu]:
    """Convert existing GUI component dictionaries to the shared OME schema."""
    out: list[Miu] = []
    for idx, miu in enumerate(gui_mius):
        box = _as_box_tuple(miu.get("box"))
        if box is None:
            continue
        source = str(miu.get("source", "image"))
        role = str(miu.get("type", "component"))
        text = str(miu.get("text", "") or "")
        score = float(miu.get("det_score", miu.get("score", 0.5)) or 0.5)
        energy = float(
            miu.get(
                "interaction_energy",
                miu.get("pentagonal_energy", miu.get("modality_energy", score)),
            )
            or 0.0
        )
        role_score = _role_prior(role, "image", source)
        center = _box_center(box)
        area = _box_area(box)
        out.append(
            {
                **miu,
                "miu_id": miu.get("miu_id", f"I{idx:02d}"),
                "modality": "image",
                "role": role,
                "type": role,
                "box": box,
                "time": (1.0, 1.0),
                "time_center": 1.0,
                "text": text[:64],
                "linked_component": miu.get("miu_id", f"I{idx:02d}"),
                "sync_group": "",
                "sync_score": 0.0,
                "det_score": score,
                "modality_score": score,
                "modality_energy": energy,
                "interaction_energy": energy,
                "role_prior": role_score,
                "instruction_alignment": 0.0,
                "action_prior": float(miu.get("action_prior", 0.0) or 0.0),
                "topology_prior": float(miu.get("topology_prior", role_score) or role_score),
                "interactable": float(miu.get("interactable", role_score) or role_score),
                "center": center,
                "area": area,
            }
        )
    return out


def _to_frame_array(frame) -> Optional[np.ndarray]:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    elif isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            return None
    else:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _iter_video_frames(video, max_frames: int = 8) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if isinstance(video, str):
        try:
            import decord  # type: ignore

            reader = decord.VideoReader(video)
            if len(reader) == 0:
                return []
            indices = np.linspace(0, len(reader) - 1, num=min(max_frames, len(reader))).round().astype(int)
            for idx in indices:
                frames.append(reader[int(idx)].asnumpy()[..., :3])
            return frames
        except Exception:
            return []
    if hasattr(video, "detach") and hasattr(video, "shape"):
        try:
            arr = video.detach().cpu().numpy()
        except Exception:
            return []
        if arr.ndim == 4:
            # Qwen official processing returns T,C,H,W float tensors.  We build
            # MIUs from those already-resized frames instead of re-reading video.
            if arr.shape[1] in (1, 3, 4):
                arr = np.transpose(arr[:, :3], (0, 2, 3, 1))
            total = arr.shape[0]
            indices = np.linspace(0, total - 1, num=min(max_frames, total)).round().astype(int)
            for idx in indices:
                frame = arr[int(idx)]
                if frame.dtype != np.uint8:
                    max_val = float(np.nanmax(frame)) if frame.size else 1.0
                    if max_val <= 1.5:
                        frame = frame * 255.0
                clean = _to_frame_array(frame)
                if clean is not None:
                    frames.append(clean)
        return frames
    if isinstance(video, np.ndarray):
        if video.ndim == 4:
            total = video.shape[0]
            indices = np.linspace(0, total - 1, num=min(max_frames, total)).round().astype(int)
            for idx in indices:
                arr = _to_frame_array(video[int(idx)])
                if arr is not None:
                    frames.append(arr)
        elif video.ndim == 3:
            arr = _to_frame_array(video)
            if arr is not None:
                frames.append(arr)
        return frames
    if isinstance(video, (list, tuple)):
        flat = list(video)
        if not flat:
            return []
        indices = np.linspace(0, len(flat) - 1, num=min(max_frames, len(flat))).round().astype(int)
        for idx in indices:
            arr = _to_frame_array(flat[int(idx)])
            if arr is not None:
                frames.append(arr)
    return frames


def _change_box(
    prev: np.ndarray,
    curr: np.ndarray,
    *,
    max_box_area: float = 0.75,
    max_mask_fraction: float = 0.35,
) -> tuple[Optional[tuple[float, float, float, float]], float]:
    if prev.shape != curr.shape:
        curr_img = Image.fromarray(curr).resize((prev.shape[1], prev.shape[0]))
        curr = np.asarray(curr_img)
    prev_gray = prev.astype(np.float32).mean(axis=2)
    curr_gray = curr.astype(np.float32).mean(axis=2)
    diff = np.abs(curr_gray - prev_gray)
    strength = float(diff.mean() / 255.0)
    threshold = max(float(np.percentile(diff, 92)), 12.0)
    mask = diff > threshold
    mask_fraction = float(mask.mean())
    if mask_fraction < 0.0005 or mask_fraction > max_mask_fraction:
        return None, strength
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    box = (
        float(xs.min() / max(w, 1)),
        float(ys.min() / max(h, 1)),
        float((xs.max() + 1) / max(w, 1)),
        float((ys.max() + 1) / max(h, 1)),
    )
    clean_box = _as_box_tuple(box)
    if clean_box is None or _box_area(clean_box) > max_box_area:
        return None, strength
    return clean_box, strength


def build_video_mius(videos, image_mius: list[Miu], config: OMEConfig = OMEConfig()) -> list[Miu]:
    if videos is None:
        return []
    video_items = videos if isinstance(videos, list) else [videos]
    out: list[Miu] = []
    for video_idx, video in enumerate(video_items):
        frames = _iter_video_frames(video, max_frames=max(config.max_video_events + 1, 3))
        if len(frames) < 2:
            continue
        for pair_idx, (prev, curr) in enumerate(zip(frames[:-1], frames[1:])):
            box, strength = _change_box(prev, curr)
            t0 = pair_idx / max(len(frames) - 1, 1)
            t1 = (pair_idx + 1) / max(len(frames) - 1, 1)
            if box is None:
                continue
            else:
                role = "change"
            linked = ""
            if image_mius and box is not None:
                best = max(
                    image_mius,
                    key=lambda m: (_box_iou(box, m["box"]), -_center_distance(box, m["box"])),
                )
                linked = str(best.get("miu_id", ""))
            score = min(1.0, max(0.05, strength * 6.0))
            out.append(
                {
                    "miu_id": f"V{video_idx:02d}_{pair_idx:02d}",
                    "modality": "video",
                    "role": role,
                    "type": role,
                    "source": "frame_diff",
                    "box": box,
                    "time": (t0, t1),
                    "time_center": 0.5 * (t0 + t1),
                    "text": "",
                    "linked_component": linked,
                    "sync_group": "",
                    "sync_score": 0.0,
                    "det_score": score,
                    "modality_score": score,
                    "modality_energy": score,
                    "interaction_energy": score,
                    "role_prior": _role_prior(role, "video"),
                    "instruction_alignment": 0.0,
                    "action_prior": 0.35 + 0.45 * score,
                    "topology_prior": 0.45 + 0.35 * score,
                    "interactable": 0.45,
                    "center": _box_center(box),
                    "area": _box_area(box),
                }
            )
    out = sorted(out, key=lambda m: -float(m.get("modality_score", 0.0)))[: config.max_video_events]
    return sorted(out, key=lambda m: (m["time_center"], m["box"][1], m["box"][0]))


def _iter_audio_arrays(audios) -> Iterable[np.ndarray]:
    if audios is None:
        return []
    if isinstance(audios, np.ndarray):
        return [audios.astype(np.float32).reshape(-1)]
    if isinstance(audios, (list, tuple)):
        arrays: list[np.ndarray] = []
        for item in audios:
            if isinstance(item, np.ndarray):
                arrays.append(item.astype(np.float32).reshape(-1))
        return arrays
    return []


def build_audio_mius(audios, config: OMEConfig = OMEConfig()) -> list[Miu]:
    out: list[Miu] = []
    for audio_idx, audio in enumerate(_iter_audio_arrays(audios)):
        if audio.size < 32:
            continue
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 2.0:
            audio = audio / max(float(np.max(np.abs(audio))), 1e-6)
        n_windows = min(64, max(8, audio.size // 800))
        chunks = np.array_split(audio, n_windows)
        rms = np.array([float(np.sqrt(np.mean(chunk * chunk) + 1e-8)) for chunk in chunks])
        if rms.max() < 1e-4:
            continue
        onset = np.maximum(0.0, np.diff(rms, prepend=rms[:1]))
        combined = 0.65 * (rms / max(rms.max(), 1e-6)) + 0.35 * (onset / max(onset.max(), 1e-6))
        top = np.argsort(-combined)[: config.max_audio_events]
        for rank, win_idx in enumerate(sorted(top.tolist())):
            score = float(combined[win_idx])
            if score < 0.18 and rank > 0:
                continue
            t0 = win_idx / max(n_windows, 1)
            t1 = (win_idx + 1) / max(n_windows, 1)
            role = "onset" if onset[win_idx] >= np.percentile(onset, 75) else "audio_event"
            out.append(
                {
                    "miu_id": f"A{audio_idx:02d}_{win_idx:02d}",
                    "modality": "audio",
                    "role": role,
                    "type": role,
                    "source": "audio_onset",
                    "box": None,
                    "time": (t0, t1),
                    "time_center": 0.5 * (t0 + t1),
                    "text": "",
                    "linked_component": "",
                    "sync_group": "",
                    "sync_score": 0.0,
                    "det_score": score,
                    "modality_score": score,
                    "modality_energy": score,
                    "interaction_energy": score,
                    "role_prior": _role_prior(role, "audio"),
                    "instruction_alignment": 0.0,
                    "action_prior": 0.25 + 0.45 * score,
                    "topology_prior": 0.20 + 0.30 * score,
                    "interactable": 0.0,
                    "center": None,
                    "area": 0.0,
                }
            )
    return sorted(out, key=lambda m: (m["time_center"], -m["modality_score"]))[: config.max_audio_events]


def attach_sync_groups(mius: list[Miu], config: OMEConfig = OMEConfig()) -> list[Miu]:
    audio = [m for m in mius if m.get("modality") == "audio"]
    video = [m for m in mius if m.get("modality") == "video"]
    group_id = 0
    for a in audio:
        best = None
        best_delta = 1e9
        for v in video:
            delta = abs(float(a.get("time_center", 0.0)) - float(v.get("time_center", 0.0)))
            if delta < best_delta:
                best = v
                best_delta = delta
        if best is None or best_delta > config.sync_window:
            continue
        gid = f"S{group_id:02d}"
        group_id += 1
        score = max(0.0, 1.0 - best_delta / max(config.sync_window, 1e-6))
        a["sync_group"] = gid
        a["sync_score"] = score
        a["linked_component"] = best.get("linked_component", "")
        best["sync_group"] = gid
        best["sync_score"] = max(float(best.get("sync_score", 0.0)), score)
        best["modality_score"] = max(float(best.get("modality_score", 0.0)), 0.5 * float(best.get("modality_score", 0.0)) + 0.5 * score)
    return mius


def build_ome_mius(
    image_mius: list[Miu],
    videos=None,
    audios=None,
    config: OMEConfig = OMEConfig(),
) -> tuple[list[Miu], dict[str, float]]:
    image_mius = normalize_image_mius(image_mius)
    video_mius = build_video_mius(videos, image_mius, config=config)
    audio_mius = build_audio_mius(audios, config=config)
    all_mius = attach_sync_groups([*image_mius, *video_mius, *audio_mius], config=config)
    meta = {
        "ome_image_miu_count": float(len(image_mius)),
        "ome_video_miu_count": float(len(video_mius)),
        "ome_audio_miu_count": float(len(audio_mius)),
        "ome_sync_group_count": float(len({m.get("sync_group") for m in all_mius if m.get("sync_group")})),
    }
    return all_mius, meta
