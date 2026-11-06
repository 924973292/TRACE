"""CoordInjectionRecord — single source of truth for Module C splice lifecycle."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Tuple


class CoordInjectionMode(str, Enum):
    METADATA_ONLY = "metadata_only"
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


@dataclass
class CoordInjectionRecord:
    """Describes one coord appendix build + optional token splice."""

    text: str = ""
    token_count: int = 0
    insert_pos: int = -1
    mode: CoordInjectionMode = CoordInjectionMode.METADATA_ONLY
    format_id: str = ""
    anchors: Sequence[Tuple[int, int]] = field(default_factory=tuple)
    anchor_count: int = 0
    ephemeral: bool = False
    spliced: bool = False
    splice_error: Optional[str] = None
    gen_tail_len: int = 0

    def to_stats_dict(self) -> dict[str, Any]:
        """Serialize for geopact_compact_stats / logging."""
        d = asdict(self)
        d["mode"] = self.mode.value
        d["anchors"] = [list(a) for a in self.anchors]
        # Back-compat fields consumed by existing tooling
        d["coord_ephemeral"] = bool(self.ephemeral and self.spliced)
        d["coord_ephemeral_tokens"] = int(self.token_count) if self.ephemeral and self.spliced else 0
        d["coord_spliced_tokens"] = int(self.token_count) if self.spliced else 0
        d["coord_insert_pos"] = int(self.insert_pos)
        d["coord_format"] = str(self.format_id)
        d["coord_anchor_count"] = int(self.anchor_count)
        d["coord_schema_contract"] = self.format_id == "sdb"
        d["coord_splice_error"] = self.splice_error
        return d

    @classmethod
    def empty(cls) -> CoordInjectionRecord:
        return cls()


def record_from_stats(stats: dict[str, Any]) -> CoordInjectionRecord:
    """Reconstruct record from geopact_compact_stats or coord_injection blob."""
    if not stats:
        return CoordInjectionRecord.empty()
    inj = stats.get("coord_injection")
    if isinstance(inj, dict) and inj:
        mode_raw = str(inj.get("mode", "") or CoordInjectionMode.METADATA_ONLY.value)
        try:
            mode = CoordInjectionMode(mode_raw)
        except ValueError:
            mode = CoordInjectionMode.METADATA_ONLY
        anchors_raw = inj.get("anchors") or []
        anchors = tuple(tuple(int(x) for x in a) for a in anchors_raw)
        return CoordInjectionRecord(
            text=str(inj.get("text", "") or ""),
            token_count=int(inj.get("token_count", inj.get("coord_spliced_tokens", 0)) or 0),
            insert_pos=int(inj.get("insert_pos", inj.get("coord_insert_pos", -1)) or -1),
            mode=mode,
            format_id=str(inj.get("format_id", inj.get("coord_format", "")) or ""),
            anchors=anchors,
            anchor_count=int(inj.get("anchor_count", inj.get("coord_anchor_count", 0)) or 0),
            ephemeral=bool(inj.get("ephemeral", inj.get("coord_ephemeral", False))),
            spliced=bool(inj.get("spliced", False)),
            splice_error=inj.get("splice_error", inj.get("coord_splice_error")),
            gen_tail_len=int(inj.get("gen_tail_len", 0) or 0),
        )
    # Legacy flat stats
    token_count = int(stats.get("coord_spliced_tokens", stats.get("coord_ephemeral_tokens", 0)) or 0)
    ephemeral = bool(stats.get("coord_ephemeral", False))
    return CoordInjectionRecord(
        text="",
        token_count=token_count,
        insert_pos=int(stats.get("coord_insert_pos", -1) or -1),
        mode=CoordInjectionMode.EPHEMERAL if ephemeral else CoordInjectionMode.METADATA_ONLY,
        format_id=str(stats.get("coord_format", "") or ""),
        anchor_count=int(stats.get("coord_anchor_count", 0) or 0),
        ephemeral=ephemeral,
        spliced=token_count > 0 and ephemeral,
        splice_error=stats.get("coord_splice_error"),
    )
