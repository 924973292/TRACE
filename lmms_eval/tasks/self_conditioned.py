"""Self-conditioned (student-forcing) multi-step evaluation.

The default multi-step protocol in this repo is teacher forcing: at step t the
prompt carries the GROUND-TRUTH actions of steps 0..t-1, and the dialogue KV
cache commits the ground-truth assistant turn. That measures per-step decisions
under a perfect history and cannot observe error accumulation.

This module closes the loop in action space. When ``GEOPACT_SELFCOND=1``:

  * the history turns of step t carry the model's OWN parsed predictions from
    steps 0..t-1 instead of the ground truth,
  * the dialogue KV commit turn carries the model's own prediction too, so the
    cache holds what the model actually said (which is the self-consistent
    choice under self-conditioning),
  * history SCREENSHOTS stay ground truth, because no GUI dataset ships the
    counterfactual frame that would have followed a divergent action.

The last point is the honest ceiling of this protocol: it is a self-conditioned
replay along a ground-truth visual trajectory. Frames become off-policy after
the first divergence. Both the dense and the pruned arm suffer that same
artifact, so the PAIRED dense-minus-pruned difference is the defensible
quantity.

Why the harness allows this at all: prompts are not pre-rendered. Requests carry
a callable ``doc_to_messages`` and messages are built inside ``generate_until``,
one step at a time, with ``batch_size=1`` and ``GEOPACT_FORCE_EPISODE_ORDER=1``
forcing strict ``(episode, step)`` order. So step t's prompt is constructed
after step t-1's decode has returned.

Env vars
--------
``GEOPACT_SELFCOND=1``           enable self-conditioning
``GEOPACT_SELFCOND_DUMP=<path>`` JSONL audit of predicted vs ground-truth
                                 history at every step
``GEOPACT_SELFCOND_INJECT=<path>``
                                 JSONL of forced actions keyed by
                                 ``(episode_id, step_index)``, used for the
                                 controlled history-contamination experiment
``GEOPACT_SELFCOND_STRICT=1``    raise instead of warn when the committed
                                 assistant text and the next step's history
                                 tail disagree (default on)
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

# (episode_key, step_index) -> record
_STORE: Dict[Tuple[str, int], Dict[str, Any]] = {}
# (episode_key, step_index) -> forced action dict (contamination experiment)
_INJECT: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None
# (episode_key, step_index) -> assistant text this step committed to the KV
_COMMITTED_TEXT: Dict[Tuple[str, int], str] = {}
_LOCK = threading.Lock()
_DUMP_FH = None
_WARNED: set = set()

UNPARSEABLE_ACTION = {"action_type": -1, "_selfcond_unparseable": True}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("GEOPACT_SELFCOND")


def strict() -> bool:
    return _flag("GEOPACT_SELFCOND_STRICT", True)


def episode_key_of(doc: Dict[str, Any]) -> str:
    """Same key the session/KV path uses, so stores line up across modules."""
    subset = str(doc.get("subset", "unknown") or "unknown")
    ep = (
        doc.get("episode_id")
        or doc.get("episode")
        or doc.get("trajectory_id")
        or doc.get("task_id")
        or "unknown"
    )
    return f"{subset}-{str(ep or 'unknown')}"


def step_index_of(doc: Dict[str, Any]) -> int:
    return int(doc.get("step_index", doc.get("step_id", 0)) or 0)


def _short_episode(episode_key: str) -> str:
    """`subset-EPID` -> `EPID`, matching the w43/w45 override key convention."""
    return str(episode_key or "").split("-", 1)[-1]


def _inject_map() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Lazily load the forced-action file for the contamination experiment.

    Rows: {"episode_id": ..., "step_index": int, "action": {...}, "arm": str}
    Keyed on the SHORT episode id so the file is portable across subsets.
    """
    global _INJECT
    if _INJECT is not None:
        return _INJECT
    path = os.environ.get("GEOPACT_SELFCOND_INJECT", "").strip()
    if not path:
        _INJECT = {}
        return _INJECT
    # Do not cache a miss: the inject file is built just before the queue
    # starts, and a cached empty map would silently turn a contamination arm
    # into a clean self-conditioned run.
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GEOPACT_SELFCOND_INJECT={path} is set but the file is missing. "
            "Refusing to start a contamination run that would not contaminate."
        )
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ep = str(row.get("episode_id") or "")
            st = int(row.get("step_index", row.get("step_id", 0)) or 0)
            if ep:
                out[(ep, st)] = row
    if not out:
        raise RuntimeError(f"GEOPACT_SELFCOND_INJECT={path} contains no rows")
    _INJECT = out
    return out


def injected_for(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Forced action row for this doc's step, if the contamination file has one."""
    m = _inject_map()
    if not m:
        return None
    return m.get((_short_episode(episode_key_of(doc)), step_index_of(doc)))


def record(doc: Dict[str, Any], raw_text: str, action: Optional[Dict[str, Any]],
           gt_action_json: str = "") -> Dict[str, Any]:
    """Store the model's own action for this step, to be used as future history.

    An unparseable generation is recorded EXPLICITLY rather than falling back to
    the ground truth. Silently substituting GT here would quietly turn the run
    back into teacher forcing on exactly the steps where the model failed, which
    is the opposite of what this protocol measures.
    """
    ep, st = episode_key_of(doc), step_index_of(doc)
    unparseable = action is None or "action_type" not in action
    forced = injected_for(doc)
    used = dict(UNPARSEABLE_ACTION) if unparseable else dict(action)
    if forced is not None and isinstance(forced.get("action"), dict):
        used = dict(forced["action"])
    rec = {
        "episode_key": ep,
        "episode_id": _short_episode(ep),
        "step_index": st,
        "action": used,
        "predicted_action": None if unparseable else dict(action),
        "unparseable": bool(unparseable),
        "injected": forced is not None,
        "inject_arm": str((forced or {}).get("arm") or ""),
        "raw_text": str(raw_text or "")[:2000],
        "gt_action_json": str(gt_action_json or ""),
    }
    with _LOCK:
        _STORE[(ep, st)] = rec
    _dump(rec)
    return rec


def action_for(episode_key: str, step_index: int) -> Optional[Dict[str, Any]]:
    with _LOCK:
        rec = _STORE.get((str(episode_key), int(step_index)))
    return dict(rec["action"]) if rec else None


def commit_action_for(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Action to commit into the dialogue KV for the CURRENT step."""
    return action_for(episode_key_of(doc), step_index_of(doc))


def history_for(doc: Dict[str, Any],
                gt_history: Optional[List[Dict[str, Any]]] = None
                ) -> Optional[List[Dict[str, Any]]]:
    """Rewrite the ground-truth history so it carries the model's own actions.

    The ground-truth history is used as the TEMPLATE rather than rebuilt from
    scratch: every field except ``action_json`` is copied through, and the step
    list is taken from the template. That keeps frames, step indices and any
    per-step instruction text exactly as the teacher-forced run had them, so the
    only difference between the two protocols is the action text - which is the
    entire point. Rebuilding from ``range(step_index)`` instead would silently
    drop history entries whenever step ids are non-contiguous or the frames live
    under a different key.
    """
    st = step_index_of(doc)
    if st <= 0:
        return []
    ep = episode_key_of(doc)
    template = gt_history if gt_history is not None else _fallback_template(doc)
    if template is None:
        return None
    if not template:
        # No history to rewrite, so the two protocols coincide here.
        return []
    # Coverage is checked per TEMPLATE step below rather than over range(st):
    # step ids need not be contiguous, and only the steps that actually appear
    # in the history have to have been produced by the model.
    out: List[Dict[str, Any]] = []
    for entry in template:
        i = int(entry.get("step_index", 0))
        act = action_for(ep, i)
        if act is None:
            # A history step the model never produced (window starts mid-episode
            # after a resume). Degrade loudly instead of mixing protocols.
            _warn_once(
                f"selfcond: no stored action for {ep} step {i}, falling back to "
                "ground-truth history"
            )
            return None
        new = dict(entry)
        new["action_json"] = json.dumps(_clean(act), ensure_ascii=False)
        out.append(new)
    return out


def _fallback_template(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Minimal template when the caller has no rendered history to hand."""
    return [
        {"step_index": int(ah.get("step_index", 0)),
         **({"image_path": ah["image_path"]} if ah.get("image_path") else {})}
        for ah in sorted(
            doc.get("action_history") or [],
            key=lambda x: x.get("step_index", 0),
        )
        if int(ah.get("step_index", 0)) < step_index_of(doc)
    ]


def _clean(action: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bookkeeping keys so the rendered tool_call matches the GT surface."""
    return {k: v for k, v in action.items() if not str(k).startswith("_")}


def note_committed_text(doc: Dict[str, Any], text: str) -> None:
    """Remember the assistant text rendered for this step's own action.

    Under dialogue KV the assistant turn of step t-1 is not prefilled at step
    t-1: the cache stops at the prompt boundary, and that turn enters the KV at
    step t as part of the incremental suffix ``[prev_assistant, current_user]``.
    Both come from ``doc_to_messages``, so recording the text here lets the next
    step assert that the suffix it is about to prefill is the turn this step
    actually produced.
    """
    with _LOCK:
        _COMMITTED_TEXT[(episode_key_of(doc), step_index_of(doc))] = str(text or "")


def _text_of(msg: Dict[str, Any]) -> str:
    return "".join(
        c.get("text", "") for c in (msg.get("content") or [])
        if isinstance(c, dict)
    )


def check_history_tail(doc: Dict[str, Any], messages: List[Dict[str, Any]],
                       render_expected) -> None:
    """Assert step t's assistant tail carries step t-1's PREDICTED action.

    This is the guard against the failure mode that would silently invalidate
    the whole protocol: if the override does not take effect, the tail still
    holds the ground-truth action and the run is teacher forcing wearing a
    self-conditioned label, with no visible symptom.

    ``render_expected(step_index, action)`` must render the assistant text the
    way the prompt builder does, so the comparison is exact rather than
    semantic.
    """
    st = step_index_of(doc)
    if st <= 0 or len(messages) < 2:
        return
    tail = messages[-2]
    if tail.get("role") != "assistant":
        return
    want_action = action_for(episode_key_of(doc), st - 1)
    if want_action is None:
        return
    expected = render_expected(st - 1, _clean(want_action))
    got = _text_of(tail)
    if got == expected:
        return
    msg = (
        "selfcond: assistant tail does not carry the stored prediction at "
        f"{episode_key_of(doc)} step {st}\n  expected: {expected[:300]!r}\n"
        f"  got:      {got[:300]!r}"
    )
    if strict():
        raise RuntimeError(msg)
    _warn_once(msg)


def _dump(rec: Dict[str, Any]) -> None:
    global _DUMP_FH
    path = os.environ.get("GEOPACT_SELFCOND_DUMP", "").strip()
    if not path:
        return
    with _LOCK:
        if _DUMP_FH is None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _DUMP_FH = open(path, "a", encoding="utf-8")
        _DUMP_FH.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _DUMP_FH.flush()


def _warn_once(msg: str) -> None:
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    try:
        from loguru import logger

        logger.warning(msg)
    except Exception:
        print("[selfcond]", msg)


def stats() -> Dict[str, Any]:
    with _LOCK:
        n = len(_STORE)
        bad = sum(1 for r in _STORE.values() if r.get("unparseable"))
        inj = sum(1 for r in _STORE.values() if r.get("injected"))
    return {"recorded_steps": n, "unparseable": bad, "injected": inj}


def reset() -> None:
    """Test helper: clear all process state."""
    global _INJECT, _DUMP_FH
    with _LOCK:
        _STORE.clear()
        _COMMITTED_TEXT.clear()
        _WARNED.clear()
        _INJECT = None
        if _DUMP_FH is not None:
            _DUMP_FH.close()
            _DUMP_FH = None
