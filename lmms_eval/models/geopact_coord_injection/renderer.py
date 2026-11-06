"""Spatial-only appendix renderers (no output-schema vocabulary)."""
from __future__ import annotations

from typing import Sequence

SIDECAR_HEADER = "[Pruned view sidecar xy@1000; reference only]"

SIDECAR_FORMATS = frozenset({
    "sidecar_bbox",
    "sidecar_bbox_role",
    "sidecar_bbox_role_text",
    "sidecar_keep",
})


def _box_token(cx: int, cy: int) -> str:
    return f"<|box_start|>({int(cx)},{int(cy)})<|box_end|>"


def _sidecar_level(format_id: str) -> str:
    fmt = (format_id or "").strip().lower()
    if fmt == "sidecar_bbox":
        return "bbox"
    if fmt == "sidecar_bbox_role":
        return "role"
    if fmt == "sidecar_bbox_role_text":
        return "text"
    if fmt == "sidecar_keep":
        return "keep"
    return "bbox"


def _normalize_role(entry: dict) -> str:
    role = str(entry.get("type") or entry.get("role") or "component").strip()
    return role.replace(" ", "_")[:24] or "component"


def _conf_score(entry: dict) -> float:
    for key in ("conf", "det_score", "info_density", "energy"):
        val = entry.get(key)
        if val is not None and val != "":
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                continue
    return 0.0


def _bbox_xyxy(entry: dict) -> tuple[int, int, int, int]:
    box = entry.get("box_xyxy")
    if box and len(box) >= 4:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    return (
        int(entry.get("cx", 0)),
        int(entry.get("cy", 0)),
        int(entry.get("cx", 0)),
        int(entry.get("cy", 0)),
    )


def format_keep_line(entry: dict, *, level: str, rank: int) -> str:
    """One geometry-aware sidecar KEEP row."""
    x1, y1, x2, y2 = _bbox_xyxy(entry)
    parts = [f"[KEEP] R{int(rank) + 1}", f"bbox=[{x1},{y1},{x2},{y2}]"]
    if level in {"role", "text", "keep"}:
        parts.append(f"role={_normalize_role(entry)}")
    if level in {"text", "keep"}:
        raw = str(entry.get("text") or "").strip()
        if raw:
            snippet = raw[:24].replace('"', "'")
            parts.append(f'text="{snippet}"')
        else:
            parts.append("text=-")
    if level == "keep":
        parts.append(f"conf={_conf_score(entry):.1f}")
    return " | ".join(parts)


def render_sidecar(
    entries: Sequence[dict],
    *,
    format_id: str = "sidecar_keep",
) -> str:
    """Geometry-aware textual sidecar (GATS) for pruned survivors."""
    if not entries:
        return ""
    level = _sidecar_level(format_id)
    lines = [SIDECAR_HEADER]
    for d in entries:
        rank = int(d.get("rank", 0))
        lines.append(format_keep_line(d, level=level, rank=rank))
    return "\n".join(lines)


def render_spatial_plain(anchors: Sequence[tuple[int, int]]) -> str:
    if not anchors:
        return ""
    parts = " ".join(f"({int(x)},{int(y)})" for x, y in anchors)
    return f"[Ref points xy@1000] {parts}"


def render_spatial_ranked(anchors: Sequence[tuple[int, int]], *, rank_prefix: str = "P") -> str:
    if not anchors:
        return ""
    parts = " ".join(
        f"{rank_prefix}{i + 1}=({int(x)},{int(y)})" for i, (x, y) in enumerate(anchors)
    )
    return f"[Ref] {parts}"


def render_spatial_box(anchors: Sequence[tuple[int, int]], *, rank_prefix: str = "P") -> str:
    if not anchors:
        return ""
    parts = " ".join(
        f"{rank_prefix}{i + 1}{_box_token(x, y)}" for i, (x, y) in enumerate(anchors)
    )
    return f"[Ref] {parts}"


def render_legacy_miu(entries: Sequence[dict]) -> str:
    lines = []
    for d in entries:
        lines.append(
            f"miu_{d['rank']} {d['type']} {_box_token(d['cx'], d['cy'])}"
        )
    return "\n".join(lines)


def render_sdb(anchors: Sequence[tuple[int, int]], *, rank_prefix: str = "A") -> str:
    """Legacy SDB format (debug only — contains action_type schema words)."""
    if not anchors:
        return ""
    meta = "[Appendix—spatial ref only; not action output] "
    contract = (
        "Choose action_type first; anchors optional for coord actions ([0,1000]). "
        "Anchors do NOT imply action_type. "
    )
    parts = " ".join(
        f"{rank_prefix}{i + 1}{_box_token(x, y)}" for i, (x, y) in enumerate(anchors)
    )
    return meta + contract + parts


def render_compact_center(anchors: Sequence[tuple[int, int]], *, rank_prefix: str = "A") -> str:
    if not anchors:
        return ""
    header = "[Current screenshot candidates. Coordinates normalized to [0,1000] xy.]\n"
    parts = " ".join(
        f"{rank_prefix}{i + 1}{_box_token(x, y)}" for i, (x, y) in enumerate(anchors)
    )
    return header + parts


RENDERERS = {
    "spatial_plain": render_spatial_plain,
    "spatial_ranked": render_spatial_ranked,
    "spatial_box": render_spatial_box,
    "legacy": render_legacy_miu,
    "sdb": render_sdb,
    "compact_center": render_compact_center,
}


def render_appendix(
    format_id: str,
    anchors: Sequence[tuple[int, int]],
    *,
    legacy_entries: Sequence[dict] | None = None,
    sidecar_entries: Sequence[dict] | None = None,
) -> str:
    fmt = (format_id or "spatial_plain").strip().lower()
    if fmt in SIDECAR_FORMATS:
        return render_sidecar(sidecar_entries or legacy_entries or (), format_id=fmt)
    if fmt == "legacy":
        if not legacy_entries:
            return ""
        return render_legacy_miu(legacy_entries)
    fn = RENDERERS.get(fmt, render_spatial_plain)
    return fn(anchors)
