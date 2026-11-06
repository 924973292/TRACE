"""Rolling history window for OmniGUI / Qwen novideo eval."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

LOCKED_EVAL_HISTORY_N = 6
# Dataset max action_history is 24; use as practical unlimited cap.
UNLIMITED_HISTORY_SLICE = 10_000


def eval_history_n() -> int:
    """Effective history window size.

    - Unset: locked prod default ``LOCKED_EVAL_HISTORY_N`` (6).
    - ``GEOPACT_EVAL_HISTORY_N=0``: unlimited (no rolling on this dataset).
    - ``GEOPACT_EVAL_HISTORY_N=50``: window 50 (equals unlimited here; max hist is 24).
    """
    raw = os.environ.get("GEOPACT_EVAL_HISTORY_N", "").strip()
    if not raw:
        return LOCKED_EVAL_HISTORY_N
    return int(raw)


def effective_history_slice_n() -> int:
    """Slice width for ``slice_action_history_list`` (large cap when unlimited)."""
    n = eval_history_n()
    if n <= 0:
        return UNLIMITED_HISTORY_SLICE
    return n


def history_window_enabled() -> bool:
    """False when unlimited history — no rolling-window shifts."""
    return eval_history_n() != 0


def _hist_step_key(entry: Dict[str, Any]) -> int:
    return int(entry.get("step_index", entry.get("step_id", 0)) or 0)


def slice_action_history_list(
    history: List[Dict[str, Any]],
    n: int = LOCKED_EVAL_HISTORY_N,
) -> List[Dict[str, Any]]:
    """Keep the most recent *n* history rows (chronological order)."""
    if not history or n <= 0:
        return []
    ordered = sorted(history, key=_hist_step_key)
    return list(ordered[-n:])


def doc_history_entries(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """History rows for rolling-window detection (aligned with qwen_doc_to_messages)."""
    history = doc.get("history") or []
    if isinstance(history, list) and history:
        return list(history)
    action_history = doc.get("action_history")
    if isinstance(action_history, list) and action_history:
        return sorted(
            [{"step_index": int(ah.get("step_index", ah.get("step_id", 0)) or 0)} for ah in action_history],
            key=_hist_step_key,
        )
    return []


def slice_doc_history(
    doc: Dict[str, Any],
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    width = effective_history_slice_n() if n is None else n
    return slice_action_history_list(doc_history_entries(doc), n=width)


def rolling_window_changed(
    prev_doc: Optional[Dict[str, Any]],
    cur_doc: Dict[str, Any],
    n: Optional[int] = None,
) -> bool:
    """True when the oldest step inside the rolling window shifts."""
    if not history_window_enabled():
        return False
    if prev_doc is None:
        return False
    width = effective_history_slice_n() if n is None else n
    cur_step = int(cur_doc.get("step_index", cur_doc.get("step_id", 0)) or 0)
    prev_step = int(prev_doc.get("step_index", prev_doc.get("step_id", 0)) or 0)
    if cur_step != prev_step + 1:
        return False
    cur_hist = doc_history_entries(cur_doc)
    if len(cur_hist) <= width:
        return False
    prev_slice = slice_action_history_list(doc_history_entries(prev_doc), n=width)
    cur_slice = slice_action_history_list(cur_hist, n=width)
    if not prev_slice or not cur_slice:
        return False
    return _hist_step_key(prev_slice[0]) != _hist_step_key(cur_slice[0])


def evicted_history_step_index(
    prev_doc: Dict[str, Any],
    n: Optional[int] = None,
) -> Optional[int]:
    """Oldest history step_index that leaves the window on the next step."""
    width = effective_history_slice_n() if n is None else n
    prev_slice = slice_action_history_list(doc_history_entries(prev_doc), n=width)
    if not prev_slice:
        return None
    return _hist_step_key(prev_slice[0])


def task_supports_session_kv(lmms_eval_specific_kwargs: Optional[Dict[str, Any]]) -> bool:
    """Task-level gate for dialogue SESSION_KV (single-step benchmarks opt out).

    An explicit ``supports_session_kv: true`` wins over dataset defaults so
    multi-step dialogue tasks (AC/M2W merged_sorted) can use the KV engine.
    """
    kw = lmms_eval_specific_kwargs or {}
    if kw.get("supports_session_kv") is False:
        return False
    if kw.get("supports_session_kv") is True:
        return True
    if kw.get("eval_android_control"):
        return False
    if kw.get("screenspot_eval"):
        return False
    if kw.get("mmbench_gui_eval"):
        return False
    return True
