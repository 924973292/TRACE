"""Training-free Topo-Gravity OmniEff message compressor for OmniGUI.

This module implements a lightweight, evaluation-friendly version of the
Topo-Gravity proposal. It operates before model prefill by rewriting the
multimodal messages sent to frozen backbones. The implementation deliberately
keeps outputs in the original input manifold: original images, selected video
frames, original audio when retained, and textual anchor relays.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger as eval_logger

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at import time
    Image = None


AUDIO_KEYWORDS = (
    "audio",
    "sound",
    "voice",
    "speech",
    "listen",
    "hear",
    "music",
    "notification",
    "beep",
    "bell",
    "song",
    "volume",
    "声音",
    "音频",
    "语音",
    "听",
    "音乐",
    "提示音",
    "铃声",
)
VIDEO_KEYWORDS = (
    "video",
    "watch",
    "play",
    "animation",
    "transition",
    "scroll",
    "loading",
    "popup",
    "clip",
    "frame",
    "视频",
    "播放",
    "动画",
    "滑动",
    "滚动",
    "加载",
    "弹窗",
    "出现",
)
TEXT_ACTION_KEYWORDS = (
    "tap",
    "click",
    "button",
    "input",
    "search",
    "checkbox",
    "menu",
    "dialog",
    "list",
    "点击",
    "输入",
    "搜索",
    "按钮",
    "菜单",
    "列表",
    "勾选",
    "弹窗",
)
LOCALIZATION_KEYWORDS = (
    "tap",
    "click",
    "select",
    "choose",
    "find",
    "icon",
    "button",
    "first",
    "点击",
    "选择",
    "找到",
    "图标",
    "按钮",
    "第一",
)
TEMPORAL_REASONING_KEYWORDS = (
    "after",
    "before",
    "until",
    "when",
    "wait",
    "loading",
    "scroll",
    "swipe",
    "next",
    "previous",
    "之后",
    "之前",
    "直到",
    "等待",
    "加载",
    "滚动",
    "滑动",
    "下一",
)
INSTANT_RESPONSE_KEYWORDS = (
    "immediately",
    "quickly",
    "now",
    "as soon as",
    "立刻",
    "立即",
    "马上",
    "快速",
)


@dataclass
class MIU:
    miu_id: str
    modality: str
    role: str
    path: Optional[str] = None
    text: str = ""
    score: float = 0.0
    density: str = "sparse"
    alignment: float = 0.0
    structural: float = 0.0
    temporal: float = 0.0
    risk: float = 0.0
    action_role: str = "context"
    state_delta: str = "stable"
    evidence_source: str = "fallback"
    uncertainty: float = 0.0
    margin: float = 1.0
    bindings: Tuple[Tuple[str, str], ...] = ()
    alias_group: str = ""


@dataclass(frozen=True)
class TopoGravityConfig:
    preset: str
    method_label: str = "Topo-Gravity OmniEff"
    retention_ratio: float = 1.0
    dense_threshold: float = 0.70
    medium_threshold: float = 0.38
    tau: float = 2.5
    image_max_edge_dense: int = 1024
    image_max_edge_medium: int = 768
    history_max_edge_medium: int = 512
    video_medium_frames: int = 2
    keep_residual: bool = True
    preserve_topology: bool = True
    use_gravity: bool = True
    force_visual_only: bool = False
    aggressive_current_image: bool = False
    adaptive_policy: bool = False
    compact_trace: bool = False
    topology_lite: bool = False
    use_category_field: bool = True
    av_anchor_only: bool = False
    strict_token_pruning: bool = False
    semantic_masking: bool = True
    coord_placeholders: bool = True
    event_memory: bool = True
    action_critical_miu: bool = False
    alias_aware: bool = False
    action_margin_pruning: bool = False
    ib_coordinate_lattice: bool = False
    event_marginal_history: bool = False
    visual_first_router: bool = False
    dense_av: bool = False
    fifo_history: bool = False
    residual_families: Tuple[str, ...] = ()


PRESETS: Dict[str, TopoGravityConfig] = {
    "balanced": TopoGravityConfig(preset="balanced"),
    "aggressive": TopoGravityConfig(
        preset="aggressive",
        dense_threshold=0.82,
        medium_threshold=0.52,
        image_max_edge_dense=896,
        image_max_edge_medium=640,
        history_max_edge_medium=384,
        video_medium_frames=1,
        aggressive_current_image=True,
    ),
    "no_residual": TopoGravityConfig(preset="no_residual", keep_residual=False),
    "no_topology": TopoGravityConfig(
        preset="no_topology",
        dense_threshold=0.82,
        medium_threshold=0.52,
        keep_residual=False,
        preserve_topology=False,
        video_medium_frames=1,
        aggressive_current_image=True,
    ),
    "miu_only": TopoGravityConfig(
        preset="miu_only",
        use_gravity=False,
        keep_residual=False,
        video_medium_frames=2,
    ),
    "visual_only": TopoGravityConfig(
        preset="visual_only",
        force_visual_only=True,
        keep_residual=False,
        video_medium_frames=0,
    ),
    "router": TopoGravityConfig(
        preset="router",
        dense_threshold=0.78,
        medium_threshold=0.50,
        image_max_edge_dense=1024,
        image_max_edge_medium=768,
        history_max_edge_medium=448,
        video_medium_frames=2,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        residual_families=("temporal",),
    ),
    "router_fast": TopoGravityConfig(
        preset="router_fast",
        dense_threshold=0.84,
        medium_threshold=0.56,
        image_max_edge_dense=896,
        image_max_edge_medium=640,
        history_max_edge_medium=384,
        video_medium_frames=1,
        keep_residual=False,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        aggressive_current_image=True,
    ),
    "router_oracle": TopoGravityConfig(
        preset="router_oracle",
        dense_threshold=0.78,
        medium_threshold=0.50,
        image_max_edge_dense=1024,
        image_max_edge_medium=768,
        history_max_edge_medium=448,
        video_medium_frames=2,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=True,
        residual_families=("temporal",),
    ),
    "miu_compact": TopoGravityConfig(
        preset="miu_compact",
        use_gravity=False,
        keep_residual=False,
        video_medium_frames=2,
        compact_trace=True,
        topology_lite=True,
    ),
    "av_anchor_compact": TopoGravityConfig(
        preset="av_anchor_compact",
        force_visual_only=True,
        keep_residual=False,
        video_medium_frames=0,
        compact_trace=True,
        topology_lite=True,
        av_anchor_only=True,
    ),
    "geopact_full": TopoGravityConfig(
        preset="geopact_full",
        method_label="GeoPACT-AMTP",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        residual_families=("temporal", "cross_modal", "instant"),
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=True,
        ib_coordinate_lattice=True,
        event_marginal_history=True,
        visual_first_router=True,
    ),
    "geopact_ac_miu": TopoGravityConfig(
        preset="geopact_ac_miu",
        method_label="GeoPACT-ACMIU",
        force_visual_only=True,
        keep_residual=False,
        video_medium_frames=0,
        compact_trace=True,
        topology_lite=True,
        action_critical_miu=True,
        alias_aware=True,
        visual_first_router=True,
    ),
    "geopact_role_priority": TopoGravityConfig(
        preset="geopact_role_priority",
        method_label="GeoPACT-role-priority",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=False,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        visual_first_router=True,
    ),
    "geopact_amtp": TopoGravityConfig(
        preset="geopact_amtp",
        method_label="GeoPACT-AMTP",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=False,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=True,
        ib_coordinate_lattice=True,
        visual_first_router=True,
    ),
    "geopact_no_alias": TopoGravityConfig(
        preset="geopact_no_alias",
        method_label="GeoPACT-no-alias",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=False,
        action_margin_pruning=True,
        ib_coordinate_lattice=True,
        event_marginal_history=True,
        visual_first_router=True,
    ),
    "geopact_no_margin": TopoGravityConfig(
        preset="geopact_no_margin",
        method_label="GeoPACT-no-action-margin",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=False,
        ib_coordinate_lattice=True,
        event_marginal_history=True,
        visual_first_router=True,
    ),
    "geopact_no_ibcl": TopoGravityConfig(
        preset="geopact_no_ibcl",
        method_label="GeoPACT-no-IBCL",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=True,
        ib_coordinate_lattice=False,
        event_marginal_history=True,
        visual_first_router=True,
    ),
    "geopact_fifo_history": TopoGravityConfig(
        preset="geopact_fifo_history",
        method_label="GeoPACT-FIFO-history",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=True,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=True,
        ib_coordinate_lattice=True,
        event_marginal_history=False,
        fifo_history=True,
        visual_first_router=True,
        residual_families=("temporal", "cross_modal", "instant"),
    ),
    "geopact_dense_av": TopoGravityConfig(
        preset="geopact_dense_av",
        method_label="GeoPACT-dense-AV",
        dense_threshold=0.70,
        medium_threshold=0.38,
        keep_residual=True,
        adaptive_policy=False,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        action_critical_miu=True,
        alias_aware=True,
        action_margin_pruning=True,
        ib_coordinate_lattice=True,
        event_marginal_history=True,
        visual_first_router=False,
        dense_av=True,
    ),
    "geopact_coord_only": TopoGravityConfig(
        preset="geopact_coord_only",
        method_label="GeoPACT-Coord",
        force_visual_only=True,
        keep_residual=False,
        video_medium_frames=0,
        compact_trace=True,
        topology_lite=True,
        action_critical_miu=True,
        visual_first_router=True,
    ),
    "geopact_coord_topo": TopoGravityConfig(
        preset="geopact_coord_topo",
        method_label="GeoPACT-CT",
        dense_threshold=0.80,
        medium_threshold=0.52,
        keep_residual=False,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
    ),
    "geopact_no_semantic": TopoGravityConfig(
        preset="geopact_no_semantic",
        method_label="GeoPACT-no-semantic-mask",
        dense_threshold=0.74,
        medium_threshold=0.42,
        keep_residual=True,
        adaptive_policy=False,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        semantic_masking=False,
    ),
    "geopact_no_event": TopoGravityConfig(
        preset="geopact_no_event",
        method_label="GeoPACT-no-event-memory",
        dense_threshold=0.78,
        medium_threshold=0.50,
        keep_residual=False,
        adaptive_policy=True,
        compact_trace=True,
        topology_lite=True,
        use_category_field=False,
        event_memory=False,
    ),
}

_RESIDUAL_QUEUES: Dict[str, List[Tuple[float, str]]] = {}

_DENSITY_COST = {
    "dense": 1.0,
    "medium": 0.45,
    "sparse": 0.08,
}


def _norm(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_retention_ratio(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip().lower().lstrip("r")
            if text.endswith("%"):
                ratio = float(text[:-1]) / 100.0
            else:
                ratio = float(text)
        else:
            ratio = float(value)
    except (TypeError, ValueError):
        return default
    if ratio > 1.0:
        ratio = ratio / 100.0
    return max(0.05, min(1.0, ratio))


def _with_runtime_overrides(config: TopoGravityConfig, lmms_eval_specific_kwargs: Dict[str, Any]) -> TopoGravityConfig:
    ratio = _parse_retention_ratio(
        lmms_eval_specific_kwargs.get("topo_retention_ratio", lmms_eval_specific_kwargs.get("retention_ratio")),
        config.retention_ratio,
    )
    strict_token_pruning = bool(
        lmms_eval_specific_kwargs.get(
            "topo_strict_token_pruning",
            lmms_eval_specific_kwargs.get("strict_token_pruning", False),
        )
    )
    return replace(config, retention_ratio=ratio, strict_token_pruning=strict_token_pruning)


def _contains_any(text: str, keywords: Iterable[str]) -> float:
    text_l = text.lower()
    return 1.0 if any(keyword.lower() in text_l for keyword in keywords) else 0.0


def _iter_media_paths(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_media_paths(item)
        return
    if not isinstance(value, str):
        return
    for item in value.split(","):
        item = item.strip()
        if item:
            yield item


def _resolve_path(path: Optional[str], media_dir: str) -> Optional[str]:
    if isinstance(path, str) and path and media_dir and not os.path.isabs(path):
        return os.path.join(media_dir, path)
    return path


def _safe_stat_size(path: Optional[str]) -> int:
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _cache_root(lmms_eval_specific_kwargs: Dict[str, Any], preset: str) -> str:
    root = lmms_eval_specific_kwargs.get("topo_cache_dir") or os.path.join("data", "topo_cache")
    path = os.path.join(root, preset)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_key(path: str, suffix: str) -> str:
    try:
        stat = os.stat(path)
        raw = f"{os.path.abspath(path)}:{stat.st_mtime_ns}:{stat.st_size}:{suffix}"
    except OSError:
        raw = f"{os.path.abspath(path)}:{suffix}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _downscale_image(path: str, cache_dir: str, max_edge: int) -> str:
    if Image is None or not path or not os.path.exists(path):
        return path
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if max(width, height) <= max_edge:
                return path
            cache_path = os.path.join(cache_dir, f"img_{_cache_key(path, str(max_edge))}_{max_edge}.jpg")
            if os.path.exists(cache_path):
                return cache_path
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            image.save(cache_path, quality=88, optimize=True)
            return cache_path
    except Exception as exc:
        eval_logger.warning(f"Topo-Gravity image downscale failed for {path}: {exc}")
        return path


def _extract_video_keyframes(path: str, cache_dir: str, max_frames: int) -> List[str]:
    if max_frames <= 0 or not path or not os.path.exists(path):
        return []
    try:
        from decord import VideoReader, cpu
    except Exception as exc:
        eval_logger.warning(f"Topo-Gravity video keyframe extraction skipped, decord unavailable: {exc}")
        return []

    out_dir = os.path.join(cache_dir, "video_frames", _cache_key(path, str(max_frames)))
    os.makedirs(out_dir, exist_ok=True)
    expected = [os.path.join(out_dir, f"frame_{idx:02d}.jpg") for idx in range(max_frames)]
    if all(os.path.exists(item) for item in expected):
        return expected

    try:
        reader = VideoReader(path, ctx=cpu(0))
        frame_count = len(reader)
        if frame_count <= 0:
            return []
        sample_count = max(1, min(max_frames, frame_count))
        if sample_count == 1:
            indices = [frame_count - 1]
        else:
            indices = [int(round(x)) for x in [i * (frame_count - 1) / (sample_count - 1) for i in range(sample_count)]]
        frames = reader.get_batch(indices).asnumpy()
        outputs: List[str] = []
        for idx, frame in enumerate(frames):
            out_path = os.path.join(out_dir, f"frame_{idx:02d}.jpg")
            if Image is None:
                continue
            Image.fromarray(frame).convert("RGB").save(out_path, quality=88, optimize=True)
            outputs.append(out_path)
        return outputs
    except Exception as exc:
        eval_logger.warning(f"Topo-Gravity video keyframe extraction failed for {path}: {exc}")
        return []


def _instruction_text(doc: Dict[str, Any]) -> str:
    parts = [str(doc.get("instruction", "") or ""), str(doc.get("instruction_en", "") or "")]
    return "\n".join(part for part in parts if part).strip()


def _episode_key(doc: Dict[str, Any]) -> str:
    return str(doc.get("episode_id") or doc.get("task_id") or doc.get("id") or "global")


def _history_lines(doc: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for action in doc.get("action_history") or []:
        step = action.get("step_index", "?")
        text = str(action.get("action_text", "") or "").strip()
        if text:
            lines.append(f"Step {step}: {text}")
    return lines


def _virtual_miu(
    miu_id: str,
    role: str,
    text: str,
    *,
    action_role: str,
    evidence_source: str,
    state_delta: str = "stable",
    uncertainty: float = 0.25,
    bindings: Tuple[Tuple[str, str], ...] = (),
) -> MIU:
    return MIU(
        miu_id=miu_id,
        modality="text",
        role=role,
        text=text,
        action_role=action_role,
        evidence_source=evidence_source,
        state_delta=state_delta,
        uncertainty=uncertainty,
        bindings=bindings,
    )


def _task_family(doc: Dict[str, Any], config: TopoGravityConfig) -> str:
    if config.use_category_field:
        category = str(doc.get("task_category_full") or doc.get("task_category") or "").lower()
        if "localization" in category:
            return "localization"
        if "semantic" in category:
            return "semantic"
        if "cross" in category:
            return "cross_modal"
        if "temporal" in category:
            return "temporal"
        if "instant" in category:
            return "instant"

    instruction = _instruction_text(doc)
    has_audio = bool(_contains_any(instruction, AUDIO_KEYWORDS))
    has_temporal = bool(_contains_any(instruction, TEMPORAL_REASONING_KEYWORDS) or _contains_any(instruction, VIDEO_KEYWORDS))
    has_instant = bool(_contains_any(instruction, INSTANT_RESPONSE_KEYWORDS))
    has_localization = bool(_contains_any(instruction, LOCALIZATION_KEYWORDS))
    if has_audio and has_temporal:
        return "cross_modal"
    if has_audio:
        return "audio"
    if has_instant:
        return "instant"
    if has_temporal and not has_localization:
        return "temporal"
    if has_localization:
        return "localization"
    return "semantic"


def _residual_enabled(doc: Dict[str, Any], config: TopoGravityConfig) -> bool:
    if not config.event_memory:
        return False
    if not config.keep_residual:
        return False
    if not config.residual_families:
        return True
    return _task_family(doc, config) in set(config.residual_families)


def _build_mius(doc: Dict[str, Any], media_dir: str) -> List[MIU]:
    mius: List[MIU] = []
    for idx, path in enumerate(_iter_media_paths(doc.get("history_screenshot_path"))):
        resolved = _resolve_path(path, media_dir)
        mius.append(
            MIU(
                f"history_image_{idx}",
                "image",
                "history_screenshot",
                resolved,
                action_role="history_witness",
                state_delta="previous_screen",
                evidence_source="action_history",
                uncertainty=0.30,
                bindings=(("after", f"history_image_{max(idx - 1, 0)}"),),
            )
        )

    video_path = doc.get("current_video") or doc.get("video") or doc.get("video_path")
    if video_path:
        mius.append(
            MIU(
                "current_video",
                "video",
                "current_video",
                _resolve_path(video_path, media_dir),
                action_role="state_witness",
                state_delta="transition_clip",
                evidence_source="frame_delta",
                uncertainty=0.35,
                bindings=(("state_transition", "current_screenshot"),),
            )
        )

    audio_path = doc.get("current_audio") or doc.get("audio") or doc.get("audio_path")
    if audio_path:
        if isinstance(audio_path, str):
            mius.append(
                MIU(
                    "current_audio",
                    "audio",
                    "current_audio",
                    _resolve_path(audio_path, media_dir),
                    action_role="modality_witness",
                    state_delta="audio_event",
                    evidence_source="audio_onset",
                    uncertainty=0.40,
                    bindings=(("cross_modal_sync", "current_video"),),
                )
            )
        else:
            mius.append(
                MIU(
                    "current_audio",
                    "audio",
                    "current_audio",
                    text="array_audio",
                    action_role="modality_witness",
                    state_delta="audio_event",
                    evidence_source="audio_array",
                    uncertainty=0.45,
                    bindings=(("cross_modal_sync", "current_video"),),
                )
            )

    image_path = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if image_path:
        if isinstance(image_path, str):
            mius.append(
                MIU(
                    "current_image",
                    "image",
                    "current_screenshot",
                    _resolve_path(image_path, media_dir),
                    action_role="target",
                    evidence_source="fallback_grid",
                    uncertainty=0.20,
                    bindings=(("inside", "screen_container"), ("bounded_by", "screen_boundary")),
                    alias_group="screen_targets",
                )
            )
        else:
            mius.append(
                MIU(
                    "current_image",
                    "image",
                    "current_screenshot",
                    text="image_object",
                    action_role="target",
                    evidence_source="image_object",
                    uncertainty=0.25,
                    bindings=(("inside", "screen_container"), ("bounded_by", "screen_boundary")),
                    alias_group="screen_targets",
                )
            )
        instruction = _instruction_text(doc)
        mius.extend(
            [
                _virtual_miu(
                    "instruction_label",
                    "label",
                    f"instruction phrase anchor: {instruction[:160]}",
                    action_role="label",
                    evidence_source="instruction",
                    uncertainty=0.15 if instruction else 0.50,
                    bindings=(("label_of", "current_image"),),
                ),
                _virtual_miu(
                    "screen_container",
                    "container",
                    "screen/modal/list container frame retained as coordinate frame",
                    action_role="container",
                    evidence_source="fallback_grid",
                    uncertainty=0.25,
                    bindings=(("contains", "current_image"),),
                ),
                _virtual_miu(
                    "screen_boundary",
                    "boundary",
                    "screen and local control boundaries retained for false-adjacency control",
                    action_role="boundary",
                    evidence_source="connected_components",
                    uncertainty=0.30,
                    bindings=(("bounds", "current_image"),),
                ),
            ]
        )

    history_lines = _history_lines(doc)
    if history_lines:
        mius.append(
            _virtual_miu(
                "action_history_state",
                "history_witness",
                "recent action state: " + " | ".join(history_lines[-3:]),
                action_role="history_witness",
                evidence_source="action_history",
                state_delta="unresolved_or_recent",
                uncertainty=0.25,
                bindings=(("action_history", "current_image"),),
            )
        )
    return mius


def _score_mius(mius: List[MIU], doc: Dict[str, Any], config: TopoGravityConfig) -> None:
    instruction = _instruction_text(doc)
    family = _task_family(doc, config)
    duration = float(doc.get("duration") or 0.0)
    action_history = doc.get("action_history") or []

    for miu in mius:
        if not config.use_gravity:
            if miu.role == "current_screenshot":
                miu.density = "dense"
            elif miu.role == "history_screenshot":
                miu.density = "medium"
            elif miu.modality == "video":
                miu.density = "medium"
            elif miu.modality == "audio":
                miu.density = "sparse"
            continue

        if miu.role == "current_screenshot":
            miu.alignment = max(_contains_any(instruction, TEXT_ACTION_KEYWORDS), 0.70)
            miu.structural = 1.0
            miu.temporal = 0.15
            miu.risk = 0.85 if family in {"localization", "semantic"} else 0.65
        elif miu.role == "history_screenshot":
            miu.alignment = 0.35 + 0.15 * bool(action_history)
            miu.structural = 0.50 if action_history else 0.30
            miu.temporal = 0.25
            miu.risk = 0.35
        elif miu.modality == "video":
            miu.alignment = max(_contains_any(instruction, VIDEO_KEYWORDS), 0.25)
            miu.structural = 0.55
            miu.temporal = 0.55 + min(duration / 8.0, 0.35)
            miu.risk = 0.75 if family in {"temporal", "cross_modal"} else 0.35
        elif miu.modality == "audio":
            miu.alignment = max(_contains_any(instruction, AUDIO_KEYWORDS), 0.20)
            miu.structural = 0.35
            miu.temporal = 0.50 + min(duration / 10.0, 0.25)
            miu.risk = 0.75 if family in {"audio", "cross_modal", "instant"} else 0.35
        elif miu.action_role == "label":
            miu.alignment = max(_contains_any(instruction, TEXT_ACTION_KEYWORDS), 0.55)
            miu.structural = 0.85
            miu.temporal = 0.10
            miu.risk = 0.75 if family in {"localization", "semantic"} else 0.45
        elif miu.action_role in {"container", "boundary"}:
            miu.alignment = 0.35
            miu.structural = 0.90
            miu.temporal = 0.10
            miu.risk = 0.70
        elif miu.action_role == "history_witness":
            miu.alignment = 0.35 + 0.20 * bool(action_history)
            miu.structural = 0.45
            miu.temporal = 0.65
            miu.risk = 0.70 if family in {"temporal", "cross_modal", "instant"} else 0.40
        else:
            miu.alignment = 0.20
            miu.structural = 0.20
            miu.temporal = 0.20
            miu.risk = 0.50

        frontend_uncertainty = 1.0 - max(miu.alignment, miu.structural, miu.temporal)
        miu.uncertainty = _norm(max(miu.uncertainty, frontend_uncertainty))
        gravity = (
            0.35 * miu.alignment
            + 0.30 * miu.structural
            + 0.20 * miu.temporal
            + 0.15 * max(miu.risk, miu.uncertainty)
        )
        miu.score = _norm(gravity)

    if config.force_visual_only:
        for miu in mius:
            if miu.modality in {"audio", "video"}:
                miu.density = "sparse"
            elif miu.role == "current_screenshot":
                miu.density = "dense"
            else:
                miu.density = "medium"
        return

    for miu in mius:
        if miu.role == "current_screenshot" and not config.aggressive_current_image:
            miu.density = "dense"
        elif miu.score >= config.dense_threshold:
            miu.density = "dense"
        elif miu.score >= config.medium_threshold:
            miu.density = "medium"
        else:
            miu.density = "sparse"

    if config.adaptive_policy and config.semantic_masking:
        _apply_adaptive_policy(mius, doc, config)


def _apply_adaptive_policy(mius: List[MIU], doc: Dict[str, Any], config: TopoGravityConfig) -> None:
    family = _task_family(doc, config)
    instruction = _instruction_text(doc)
    audio_cue = bool(_contains_any(instruction, AUDIO_KEYWORDS))
    video_cue = bool(_contains_any(instruction, VIDEO_KEYWORDS) or _contains_any(instruction, TEMPORAL_REASONING_KEYWORDS))
    has_history = bool(doc.get("action_history") or doc.get("history_screenshot_path"))

    for miu in mius:
        if miu.role == "current_screenshot":
            if family in {"localization", "semantic"} and config.preset != "router_fast":
                miu.density = "dense"
            else:
                miu.density = "medium"
        elif miu.role == "history_screenshot":
            miu.density = "medium" if has_history else "sparse"
        elif miu.modality == "video":
            if config.av_anchor_only:
                miu.density = "sparse"
            elif family == "temporal":
                miu.density = "medium"
            elif family == "cross_modal" and video_cue:
                miu.density = "medium"
            elif family == "instant" and config.preset != "router_fast":
                miu.density = "medium"
            else:
                miu.density = "sparse"
        elif miu.modality == "audio":
            if config.av_anchor_only:
                miu.density = "sparse"
            elif family in {"audio", "cross_modal"} and audio_cue:
                miu.density = "medium"
            elif config.dense_av:
                miu.density = "medium"
            else:
                miu.density = "sparse"


def _estimate_action_margins(mius: List[MIU], doc: Dict[str, Any], config: TopoGravityConfig) -> None:
    """Training-free action-risk proxy from the paper's pessimistic margin."""
    family = _task_family(doc, config)
    instruction = _instruction_text(doc)
    has_alias_language = any(
        term in instruction.lower()
        for term in ("same", "row", "first", "second", "delete", "choose", "选择", "第", "相同", "删除")
    )
    needs_event = family in {"temporal", "cross_modal", "audio", "instant"} or bool(
        _contains_any(instruction, TEMPORAL_REASONING_KEYWORDS + AUDIO_KEYWORDS + VIDEO_KEYWORDS)
    )
    for miu in mius:
        text_distance = 1.0 - miu.alignment
        visual_distance = 1.0 - miu.structural
        coord_distance = 0.18 if miu.action_role in {"target", "label", "boundary", "container"} else 0.45
        relation_bonus = 0.0 if miu.bindings else 0.18
        state_distance = 0.15 if (needs_event and miu.action_role in {"state_witness", "modality_witness", "history_witness"}) else 0.35
        base_margin = (text_distance + visual_distance + coord_distance + relation_bonus + state_distance) / 5.0
        if has_alias_language and miu.action_role in {"target", "label", "boundary", "container"}:
            base_margin *= 0.65
            if config.alias_aware:
                miu.alias_group = miu.alias_group or "instruction_alias_group"
        pessimistic = base_margin - 0.35 * miu.uncertainty
        miu.margin = _norm(pessimistic)
        miu.risk = _norm(max(miu.risk, 1.0 - miu.margin))
        miu.score = _norm(0.55 * miu.score + 0.45 * miu.risk)


def _apply_action_margin_topology(mius: List[MIU], doc: Dict[str, Any], config: TopoGravityConfig) -> None:
    family = _task_family(doc, config)
    instruction = _instruction_text(doc)
    audio_cue = bool(_contains_any(instruction, AUDIO_KEYWORDS))
    video_cue = bool(_contains_any(instruction, VIDEO_KEYWORDS) or _contains_any(instruction, TEMPORAL_REASONING_KEYWORDS))
    low_margin = sorted(mius, key=lambda miu: (miu.margin, -miu.score))[:4]
    closure_roles = {"target", "label", "container", "boundary"}
    closure_ids = {miu.miu_id for miu in low_margin if miu.action_role in closure_roles}
    for miu in mius:
        if miu.action_role in closure_roles or miu.miu_id in closure_ids:
            miu.density = "dense" if miu.role == "current_screenshot" else "sparse"
        elif miu.action_role == "state_witness":
            triggered = family in {"temporal", "cross_modal", "instant"} or video_cue or miu.margin < 0.45
            miu.density = "medium" if triggered else "sparse"
        elif miu.action_role == "modality_witness":
            triggered = family in {"audio", "cross_modal"} or audio_cue or miu.margin < 0.42
            miu.density = "medium" if triggered else "sparse"
        elif miu.action_role == "history_witness":
            if config.event_marginal_history:
                miu.density = "medium" if (family in {"temporal", "cross_modal", "instant"} or miu.margin < 0.50) else "sparse"
            elif config.fifo_history:
                miu.density = "medium"
            else:
                miu.density = "sparse"
        elif config.dense_av and miu.modality in {"audio", "video"}:
            miu.density = "medium"


def _density_budget(mius: List[MIU]) -> float:
    return sum(_DENSITY_COST.get(miu.density, _DENSITY_COST["sparse"]) for miu in mius)


def _apply_retention_ratio_budget(mius: List[MIU], config: TopoGravityConfig) -> None:
    if config.retention_ratio >= 0.999 or not mius:
        return

    target = max(0.30, len(mius) * config.retention_ratio)
    candidates = sorted(mius, key=lambda miu: (miu.role == "current_screenshot", miu.score))
    while _density_budget(mius) > target:
        changed = False
        for miu in candidates:
            if miu.density == "sparse":
                continue
            if miu.role == "current_screenshot" and miu.density == "medium":
                continue
            miu.density = "medium" if miu.density == "dense" else "sparse"
            changed = True
            if _density_budget(mius) <= target:
                break
        if not changed:
            break


def _should_relay_sparse(miu: MIU, config: TopoGravityConfig) -> bool:
    if not config.preserve_topology:
        return False
    if miu.modality == "text" or miu.action_role in {"label", "container", "boundary", "history_witness"}:
        return True
    if not config.topology_lite:
        return True
    if config.av_anchor_only and miu.modality in {"audio", "video"}:
        return True
    return miu.modality in {"audio", "video"} and max(miu.score, miu.risk) >= 0.45


def _relay_text(miu: MIU, reason: str) -> str:
    path_name = os.path.basename(miu.path or miu.text or miu.miu_id)
    if miu.role == "current_screenshot":
        coord_field = "screen_xy=full; time=current; scale=screen"
    elif miu.role == "history_screenshot":
        coord_field = "screen_xy=history; time=previous; scale=screen"
    elif miu.modality == "video":
        coord_field = "time=transition_clip; scale=frame_sequence"
    elif miu.modality == "audio":
        coord_field = "time=audio_event; scale=waveform"
    else:
        coord_field = "time=current; scale=unknown"
    if miu.action_role in {"label", "container", "boundary", "target"}:
        relation = "label-target-container-boundary closure"
    elif miu.modality in {"video", "audio"}:
        relation = "affordance_flow"
    elif miu.action_role == "history_witness":
        relation = "event-marginal history"
    else:
        relation = "coordinate_anchor"
    lattice_hint = ""
    if miu.action_role in {"label", "container", "boundary"}:
        lattice_hint = "; ib_lattice=relative_same-row/inside/near"
    bindings = ",".join(f"{edge}:{dst}" for edge, dst in miu.bindings[:4]) or "none"
    return (
        f"[GeoPACT relay] role={miu.action_role}; source={miu.evidence_source}; modality={miu.modality}; "
        f"density={miu.density}; anchor={path_name}; coord_field={coord_field}{lattice_hint}; "
        f"margin={miu.margin:.2f}; uncertainty={miu.uncertainty:.2f}; bindings={bindings}; "
        f"relation={relation}; reason={reason}."
    )


def _residue_text(queue_items: List[Tuple[float, str]]) -> str:
    if not queue_items:
        return ""
    anchors = [text for _, text in sorted(queue_items, reverse=True)[:4]]
    return "[Residual MIU Persistence]\n" + "\n".join(anchors)


def _update_residual_queue(doc: Dict[str, Any], mius: List[MIU], enabled: bool) -> None:
    if not enabled:
        return
    episode = _episode_key(doc)
    old_items = [(score * 0.72, text) for score, text in _RESIDUAL_QUEUES.get(episode, []) if score * 0.72 > 0.10]
    new_items: List[Tuple[float, str]] = []
    for miu in mius:
        if miu.modality not in {"video", "audio"}:
            continue
        if miu.density == "dense":
            continue
        unresolved = miu.score * (1.25 if miu.modality == "video" else 1.0)
        new_items.append((unresolved, _relay_text(miu, "unresolved temporal evidence carried from the previous step")))
    merged = sorted(old_items + new_items, reverse=True)[:6]
    if merged:
        _RESIDUAL_QUEUES[episode] = merged


def _append_miu_content(
    content: List[Dict[str, Any]],
    miu: MIU,
    config: TopoGravityConfig,
    cache_dir: str,
) -> None:
    if config.strict_token_pruning:
        if miu.role == "history_screenshot":
            content.append({"type": "text", "text": "[History Screenshot - original media]:"})
            content.append({"type": "image", "url": miu.path})
            return
        if miu.role == "current_screenshot":
            content.append({"type": "text", "text": "[Current Screenshot - original media]:"})
            content.append({"type": "image", "url": miu.path})
            return
        if miu.modality == "video":
            content.append({"type": "text", "text": "[Current Video - original media]:"})
            content.append({"type": "video", "url": miu.path})
            return
        if miu.modality == "audio":
            content.append({"type": "text", "text": "[Current Audio - original media]:"})
            content.append({"type": "audio", "url": {"path": miu.path}})
            return

    if miu.role == "history_screenshot":
        if miu.density == "dense":
            content.append({"type": "text", "text": "[History Screenshot]:"})
            content.append({"type": "image", "url": miu.path})
        elif miu.density == "medium":
            content.append({"type": "text", "text": "[History Screenshot - medium MIU]:"})
            content.append({"type": "image", "url": _downscale_image(miu.path or "", cache_dir, config.history_max_edge_medium)})
        elif _should_relay_sparse(miu, config):
            content.append({"type": "text", "text": _relay_text(miu, "history anchor preserved without dense pixels")})
        return

    if miu.role == "current_screenshot":
        max_edge = config.image_max_edge_dense if miu.density == "dense" else config.image_max_edge_medium
        path = _downscale_image(miu.path or "", cache_dir, max_edge)
        content.append({"type": "text", "text": f"[Current Screenshot - {miu.density} MIU]:"})
        content.append({"type": "image", "url": path})
        return

    if miu.modality == "video":
        if config.force_visual_only:
            if _should_relay_sparse(miu, config):
                content.append({"type": "text", "text": _relay_text(miu, "video omitted by visual-only ablation")})
            return
        if miu.density == "dense":
            content.append({"type": "video", "url": miu.path})
        elif miu.density == "medium":
            frames = _extract_video_keyframes(miu.path or "", cache_dir, config.video_medium_frames)
            if frames:
                content.append({"type": "text", "text": f"[Video MIU keyframes: {len(frames)} selected frames, temporal order preserved]:"})
                for frame in frames:
                    content.append({"type": "image", "url": frame})
            elif _should_relay_sparse(miu, config):
                content.append({"type": "text", "text": _relay_text(miu, "video keyframe extraction unavailable")})
        elif _should_relay_sparse(miu, config):
            content.append({"type": "text", "text": _relay_text(miu, "video compressed to sparse temporal anchor")})
        return

    if miu.modality == "audio":
        if config.force_visual_only:
            if _should_relay_sparse(miu, config):
                content.append({"type": "text", "text": _relay_text(miu, "audio omitted by visual-only ablation")})
            return
        if miu.density == "dense":
            content.append({"type": "audio", "url": {"path": miu.path}})
        elif _should_relay_sparse(miu, config):
            content.append({"type": "text", "text": _relay_text(miu, "audio compressed to event anchor")})
        return


def build_topo_gravity_messages(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]],
    system_prompt: str,
    preset: str = "balanced",
) -> List[Dict[str, Any]]:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    config = _with_runtime_overrides(PRESETS.get(preset, PRESETS["balanced"]), lmms_eval_specific_kwargs)
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "")
    cache_preset = config.preset if config.retention_ratio >= 0.999 else f"{config.preset}_r{int(round(config.retention_ratio * 100))}"
    cache_dir = _cache_root(lmms_eval_specific_kwargs, cache_preset)

    mius = _build_mius(doc, media_dir)
    _score_mius(mius, doc, config)
    if config.action_critical_miu:
        _estimate_action_margins(mius, doc, config)
    if config.action_margin_pruning:
        _apply_action_margin_topology(mius, doc, config)
    _apply_retention_ratio_budget(mius, config)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    content: List[Dict[str, Any]] = []
    if not config.compact_trace:
        operators = "MIU construction, budgeted action-critical cover"
        if config.semantic_masking:
            operators += ", prompt-aware semantic masking"
        if config.preserve_topology:
            operators += ", coordinate placeholders, topology relays"
        if config.keep_residual and config.event_memory:
            operators += ", event continuation memory"
        budget_text = ""
        if config.retention_ratio < 0.999:
            if config.strict_token_pruning:
                budget_text = f"; strict_token_prune_target={config.retention_ratio:.2f}; media_input=original"
            else:
                budget_text = f"; target_retention={config.retention_ratio:.2f}; miu_budget={_density_budget(mius):.2f}"
        content.append(
            {
                "type": "text",
                "text": f"[{config.method_label}] preset={config.preset}; operators={operators}{budget_text}.",
            }
        )

    residual_enabled = _residual_enabled(doc, config)
    if residual_enabled:
        residue = _residue_text(_RESIDUAL_QUEUES.get(_episode_key(doc), []))
        if residue:
            content.append({"type": "text", "text": residue})

    for miu in mius:
        if miu.modality == "text" and _should_relay_sparse(miu, config):
            reason = "typed AC-MIU relation relay"
            if miu.action_role in {"label", "container", "boundary"} and config.ib_coordinate_lattice:
                reason = "IB coordinate lattice preserved as natural-language relation hint"
            content.append({"type": "text", "text": _relay_text(miu, reason)})

    # Preserve the original topology: history -> temporal evidence -> current screenshot -> instruction.
    for role in ("history_screenshot", "current_video", "current_audio", "current_screenshot"):
        for miu in [item for item in mius if item.role == role or item.miu_id == role]:
            _append_miu_content(content, miu, config, cache_dir)

    instruction = _instruction_text(doc)
    action_history = doc.get("action_history", []) or []
    history_text = ""
    if action_history:
        lines = []
        for action in action_history:
            step = action.get("step_index", "?")
            text = action.get("action_text", "")
            lines.append(f"Step {step}: {text}")
        history_text = "\n".join(lines)

    current_step_index = doc.get("step_index", doc.get("step_id", 0))
    miu_summary = ", ".join(f"{miu.role}:{miu.density}:{miu.score:.2f}" for miu in mius)
    prompt_parts = [f"Goal: {instruction}", f"Step: {current_step_index}"]
    if config.method_label.startswith("GeoPACT"):
        prompt_parts.append(
            "Compression Invariants: AC-MIU role coverage, action-margin topology closure, IB coordinate-lattice relays, event-marginal history."
        )
        if config.action_margin_pruning:
            low_margin = min((miu.margin for miu in mius), default=1.0)
            prompt_parts.append(f"Action-Margin Policy: protect low-margin candidates first; min_margin={low_margin:.2f}.")
        if not config.alias_aware and config.action_critical_miu:
            prompt_parts.append("Ablation: alias-aware repeated-control context disabled.")
        if not config.action_margin_pruning and config.action_critical_miu:
            prompt_parts.append("Ablation: action-margin topology pruning disabled; using role-priority retention.")
        if not config.ib_coordinate_lattice and config.action_critical_miu:
            prompt_parts.append("Ablation: IB coordinate lattice disabled; removed spans have weaker relation hints.")
        if not config.semantic_masking:
            prompt_parts.append("Ablation: prompt-aware semantic masking disabled; modality budget is intentionally less task-conditioned.")
        if not config.event_memory:
            prompt_parts.append("Ablation: event continuation memory disabled; do not rely on carried transition relays.")
    if not config.compact_trace:
        prompt_parts.append(f"Compression Trace: {miu_summary}")
    if history_text:
        prompt_parts.append(f"Action History:\n{history_text}")
    prompt_parts.append("Analyze the current screenshot and output the next action as JSON.")
    content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

    messages.append({"role": "user", "content": content})
    _update_residual_queue(doc, mius, residual_enabled)
    return messages
