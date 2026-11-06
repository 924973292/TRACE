"""Task-agnostic dialogue SESSION_KV helpers (shared doc_to_messages source).

Ensures full-prefill and incremental suffix paths use the same task-level
``doc_to_messages`` function so GUI-Owl cross-benchmark tasks stay aligned with
official eval prompts when GeoPACT dialogue KV is enabled.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from lmms_eval.tasks.history_window import (
    rolling_window_changed,
    task_supports_session_kv,
)

DocToMessagesFn = Callable[..., List[Dict[str, Any]]]


def invoke_doc_to_messages(
    doc_to_messages: DocToMessagesFn,
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Call utils ``doc_to_messages(doc, kwargs)`` or task-bound ``doc_to_messages(doc)``."""
    import inspect

    try:
        n_params = len(inspect.signature(doc_to_messages).parameters)
    except (TypeError, ValueError):
        return doc_to_messages(doc)  # type: ignore[call-arg]
    if n_params >= 2:
        return doc_to_messages(doc, lmms_eval_specific_kwargs)  # type: ignore[call-arg]
    return doc_to_messages(doc)  # type: ignore[call-arg]


def session_step_index(doc: Dict[str, Any]) -> int:
    return int(doc.get("step_index", doc.get("step_id", 0)) or 0)


def supports_dialogue_session_kv(doc: Dict[str, Any]) -> bool:
    """Heuristic: multi-step docs with episode metadata may use dialogue KV."""
    if session_step_index(doc) <= 0 and not doc.get("action_history"):
        return bool(doc.get("episode_id"))
    return bool(doc.get("episode_id") or doc.get("subset"))


def dialogue_incremental_suffix_allowed(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]],
    doc_to_messages: DocToMessagesFn,
    *,
    prev_doc: Optional[Dict[str, Any]] = None,
    kv_slid_at_roll: bool = False,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """True when step>0 and incremental suffix matches full-prefill message tail."""
    if session_step_index(doc) <= 0:
        return False
    if not task_supports_session_kv(lmms_eval_specific_kwargs):
        return False
    if prev_doc is None:
        return False
    if session_step_index(doc) != session_step_index(prev_doc) + 1:
        return False
    if rolling_window_changed(prev_doc, doc) and not kv_slid_at_roll:
        return False

    if messages is None:
        messages = invoke_doc_to_messages(doc_to_messages, doc, lmms_eval_specific_kwargs)
    return (
        len(messages) >= 2
        and messages[-1].get("role") == "user"
        and messages[-2].get("role") == "assistant"
    )


def dialogue_incremental_suffix_messages(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]],
    doc_to_messages: DocToMessagesFn,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Previous GT assistant + current user turn for incremental dialogue KV."""
    step_index = session_step_index(doc)
    if step_index <= 0:
        raise RuntimeError("incremental dialogue suffix requires step_index > 0")

    if messages is None:
        messages = invoke_doc_to_messages(doc_to_messages, doc, lmms_eval_specific_kwargs)
    current_user = messages[-1]
    if current_user.get("role") != "user":
        raise RuntimeError(
            f"incremental dialogue suffix expects final user turn "
            f"episode={doc.get('episode_id')} step={step_index}"
        )
    if len(messages) >= 2 and messages[-2].get("role") == "assistant":
        return [messages[-2], current_user]

    raise RuntimeError(
        f"incremental dialogue suffix message tail mismatch "
        f"episode={doc.get('episode_id')} step={step_index}"
    )
