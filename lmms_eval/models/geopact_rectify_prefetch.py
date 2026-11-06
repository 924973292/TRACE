"""Async history-frame rectify prefetch.

The full ABC rectify of a captured history frame (Module A join + Module B
per-turn selection + ZAFD + kept-row gather) depends only on the capture and
that turn's instruction, both frozen before the current step's decode starts.
Running it here on a background thread hides its 100-600 ms behind decode, so
the next step's ``sliding_rewrite_prepare`` only joins a finished result.

All job inputs are detached snapshots; model attribute writes performed by the
job (sticky mask store, prune diag) are keyed by episode/path and benign if the
result is later discarded.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch

_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

RectifyResult = Tuple[
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
    List[int],
    Dict[str, Any],
]


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="rectify_prefetch",
            )
        return _EXECUTOR


def rectify_prefetch_enabled() -> bool:
    # Default OFF: clean single-GPU A/B (hist2 LIMIT=30, sync vs stage) showed the
    # background job taxes decode +1.3-2.1ms/token, exceeding the hidden rectify
    # cost once the blocked-herding optimization cut sync rectify to ~54ms.
    return os.environ.get("GEOPACT_RECTIFY_PREFETCH", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _job_timeout_s() -> int:
    return int(os.environ.get("GEOPACT_RECTIFY_PREFETCH_TIMEOUT", "300"))


def rectify_prefetch_at() -> str:
    """Submit point: 'stage' (pre-decode, default) or 'finalize' (post-decode).

    Measured A/B (hist2 LIMIT=30, identical EM): 'stage' hides the whole job
    behind decode at +0.5ms/token contention (net -58ms/step); 'finalize'
    keeps decode clean but the join barrier eats the saving (net +83ms vs
    stage).
    """
    raw = os.environ.get("GEOPACT_RECTIFY_PREFETCH_AT", "stage").strip().lower()
    return "finalize" if raw == "finalize" else "stage"


def _inner(model):
    return getattr(model, "model", model)


def wait_rectify_prefetch(model) -> None:
    """Block until a pending job finishes (keeps its result consumable)."""
    if model is None:
        return
    handle = getattr(_inner(model), "_geopact_rectify_async", None)
    if not isinstance(handle, dict):
        return
    future: Optional[Future] = handle.get("future")
    if future is not None and not future.done():
        try:
            future.result(timeout=_job_timeout_s())
        except Exception:
            pass


def discard_rectify_prefetch(model) -> None:
    """Drop a pending handle at episode boundary / reset.

    Joins a still-running job first so its stray attribute writes (sspc audit,
    sticky store) cannot race the next episode's state resets.
    """
    if model is None:
        return
    inner = _inner(model)
    handle = getattr(inner, "_geopact_rectify_async", None)
    if handle is None:
        return
    wait_rectify_prefetch(inner)
    inner._geopact_rectify_async = None


_JOB_STREAM: Optional["torch.cuda.Stream"] = None


def _low_priority_stream(device) -> Optional["torch.cuda.Stream"]:
    """Dedicated low-priority stream so job kernels fill decode's idle gaps
    instead of competing with the decode forward on the default stream."""
    global _JOB_STREAM
    if not torch.cuda.is_available():
        return None
    if _JOB_STREAM is None:
        try:
            least, _greatest = torch.cuda.get_stream_priority_range()
        except Exception:
            least = 0
        _JOB_STREAM = torch.cuda.Stream(device=device, priority=least)
    return _JOB_STREAM


_THREAD_PRIO_SET = False


def _lower_thread_priority() -> None:
    """Nice the worker thread once so it yields CPU/GIL to the decode loop."""
    global _THREAD_PRIO_SET
    if _THREAD_PRIO_SET:
        return
    _THREAD_PRIO_SET = True
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except Exception:
        pass


def _rectify_job(
    model,
    cap: Dict[str, Any],
    ratio: float,
    policy: str,
    episode_key: str,
    n_frame: int,
) -> RectifyResult:
    from lmms_eval.models.geopact_vl_hook import rectify_history_frame_from_full

    _lower_thread_priority()

    def _run() -> RectifyResult:
        # The sync path runs under sliding_rewrite_prepare's @inference_mode;
        # keep the same tensor semantics off-thread.
        with torch.inference_mode():
            return rectify_history_frame_from_full(
                model,
                cap=cap,
                ratio=float(ratio),
                policy=str(policy or ""),
                episode_key=str(episode_key or ""),
                n_frame=int(n_frame),
            )

    embeds = cap.get("visual_embeds")
    device = embeds.device if isinstance(embeds, torch.Tensor) else None
    stream = _low_priority_stream(device) if device is not None and device.type == "cuda" else None
    if stream is None:
        return _run()
    # Capture tensors were produced on the default stream; order after it, run
    # on the low-priority stream, and synchronize so returned tensors are ready
    # for any stream the consumer uses.
    stream.wait_stream(torch.cuda.default_stream(device))
    with torch.cuda.stream(stream):
        result = _run()
    stream.synchronize()
    return result


def start_rectify_prefetch(
    model,
    cap: Dict[str, Any],
    *,
    episode_key: str = "",
) -> bool:
    """Submit the full rectify for a freshly captured history frame."""
    if not rectify_prefetch_enabled():
        return False
    if not isinstance(cap, dict):
        return False
    embeds = cap.get("visual_embeds")
    if embeds is None or cap.get("grid_row") is None:
        return False
    inner = _inner(model)
    ratio = float(getattr(inner, "_geopact_prune_ratio", 0.0) or 0.0)
    policy = str(getattr(inner, "_geopact_prune_policy", "") or "")
    if ratio <= 0.0 or ratio >= 1.0 or not policy:
        return False
    if not episode_key:
        state = getattr(inner, "_geopact_dialogue_kv_state", None) or {}
        episode_key = str(state.get("episode_key") or "")
    n_frame = int(embeds.shape[0])
    inner._geopact_rectify_async = {
        "future": _executor().submit(
            _rectify_job, model, cap, ratio, policy, episode_key, n_frame
        ),
        "img_path": str(cap.get("img_path") or ""),
        "frame_seq": int(cap.get("frame_seq", -1)),
        "ratio": ratio,
        "policy": policy,
        "episode_key": episode_key,
        "n_frame": n_frame,
        "started_at": time.perf_counter(),
    }
    return True


def try_consume_rectify_prefetch(
    model,
    *,
    img_path: str,
    frame_seq: int,
    ratio: float,
    policy: str,
    episode_key: str,
    n_frame: int,
) -> Optional[RectifyResult]:
    """Return the prefetched rectify result if it matches this frame exactly.

    On any identity mismatch the handle is dropped and ``None`` returned so the
    caller falls back to the synchronous path (never a silent wrong-frame use).
    """
    inner = _inner(model)
    handle = getattr(inner, "_geopact_rectify_async", None)
    if not isinstance(handle, dict):
        return None
    inner._geopact_rectify_async = None
    future: Optional[Future] = handle.get("future")
    if future is None:
        return None
    if (
        str(handle.get("img_path") or "") != str(img_path or "")
        or int(handle.get("frame_seq", -2)) != int(frame_seq)
        or abs(float(handle.get("ratio", -1.0)) - float(ratio)) > 1e-9
        or str(handle.get("policy") or "") != str(policy or "")
        or int(handle.get("n_frame", -1)) != int(n_frame)
    ):
        return None
    job_ep = str(handle.get("episode_key") or "")
    if job_ep and episode_key and job_ep != str(episode_key):
        return None
    join_t0 = time.perf_counter()
    try:
        kept_visual, rectified_deepstack, local_keep, stats = future.result(
            timeout=_job_timeout_s()
        )
    except Exception:
        return None
    join_ms = (time.perf_counter() - join_t0) * 1000.0
    wall_ms = (time.perf_counter() - float(handle.get("started_at") or join_t0)) * 1000.0
    stats = dict(stats)
    stats["rectify_async"] = True
    stats["rectify_async_join_ms"] = round(join_ms, 2)
    stats["rectify_async_wall_ms"] = round(wall_ms, 2)
    return kept_visual, rectified_deepstack, local_keep, stats
