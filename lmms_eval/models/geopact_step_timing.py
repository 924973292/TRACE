"""Exclusive per-step timing ledger for GeoPACT / dialogue KV profiling.

Buckets are mutually exclusive: only one bucket is active at a time. Use
``begin_step`` at step start and ``finalize_step`` before attaching usage stats.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

EXCLUSIVE_BUCKET_KEYS: tuple[str, ...] = (
    "processor_ms",
    "suffix_construct_ms",
    "episode_setup_ms",
    "vision_encode_ms",
    "module_a_ms",
    "sticky_keep_ms",
    "origpos_mrope_prefull_ms",
    "origpos_mrope_ms",
    "compact_gather_ms",
    "deepstack_align_ms",
    "dialogue_suffix_mrope_ms",
    "dialogue_kv_batch_prefill_ms",
    "dialogue_kv_prepare_ms",
    "dialogue_kv_capture_ms",
    "kv_bootstrap_crop_ms",
    "kv_slide_ms",
    "generate_payload_ms",
    "generate_loop_ms",
    "vlm_prefill_ms",
    "llm_prefill_ms",
    "llm_decode_ms",
    "dialogue_kv_finalize_ms",
    "prompt_tail_ids_ms",
    "suffix_pil_cache_ms",
    "kv_commit_ids_ms",
    "sliding_rewrite_ms",
    "sliding_rewrite_lm_ms",
    "sliding_rewrite_rectify_ms",
    "sliding_rewrite_mrope_ms",
    "module_a_sideload_ms",
    "response_postprocess_ms",
)

_ALL_BUCKETS: tuple[str, ...] = EXCLUSIVE_BUCKET_KEYS + ("misc_ms",)

_ANATOMY_RF = bool(os.environ.get("GEOPACT_ANATOMY", "").strip())

# Diagnostic: attribute inter-bucket gaps (unbucketed wall time) to
# "<prev_bucket>-><next_bucket>" edges in timing_audit.timing_gaps.
_GAP_AUDIT = os.environ.get("GEOPACT_TIMING_GAP_AUDIT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def cuda_sync_enabled() -> bool:
    return os.environ.get("GEOPACT_TIMING_CUDA_SYNC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def mem_audit_enabled() -> bool:
    """GEOPACT_MEM_AUDIT=1: per-step peak/steady GPU memory readings.

    Allocator counters (reset_peak_memory_stats / max_memory_allocated) are
    host-side and need no device synchronize, so this is cheap enough for
    eval runs where the full timing sync is off — the review's memory
    quantification (W4) needs peaks without perturbing latency.
    """
    return os.environ.get("GEOPACT_MEM_AUDIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def kv_cache_bytes_from(cache: Any) -> int:
    """Actual bytes held by a transformers KV cache (DynamicCache & friends).

    Sums numel x element_size over every key/value tensor. Supports both the
    new ``cache.layers[i].keys/.values`` layout and the legacy
    ``key_cache``/``value_cache`` tensor lists. Used by the GEOPACT_MEM_AUDIT
    path so the efficiency table reports *measured* KV bytes (FastV/PDrop
    prune mid-prefill, so analytic prompt-length formulas would overcount).
    """
    import torch

    total = 0
    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        if total:
            return total
    for name in ("key_cache", "value_cache"):
        for t in getattr(cache, name, None) or []:
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total


def install_kv_bytes_probe(model) -> Optional[dict]:
    """Mem-audit helper: stash a reference to the live ``past_key_values``.

    ``generate()`` builds its cache internally and discards it, so a forward
    pre-hook on the top-level model records the kwarg reference; after
    generation the caller reads the actual byte count via
    ``kv_cache_bytes_from(holder["pkv"])``. Installed once per model, only
    when GEOPACT_MEM_AUDIT=1 (reference capture only — no copies, no sync).
    """
    if not mem_audit_enabled():
        return None
    holder = getattr(model, "_geopact_kv_probe_holder", None)
    if holder is not None:
        return holder
    holder = {"pkv": None}

    def _pre_hook(module, args, kwargs):
        pkv = kwargs.get("past_key_values")
        if pkv is not None:
            holder["pkv"] = pkv
        return None

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model._geopact_kv_probe_holder = holder
    return holder


def _maybe_cuda_sync() -> None:
    if not cuda_sync_enabled():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _inner_from_model(model) -> Any:
    return getattr(model, "model", model)


class StepTimingLedger:
    """Mutually exclusive timing buckets for one eval step."""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {k: 0.0 for k in _ALL_BUCKETS}
        self._active: Optional[str] = None
        self._active_t0: float = 0.0
        self._step_t0: float = 0.0
        self._audit: Dict[str, Any] = {}
        self._started = False
        self._rf_ctx = None  # torch.profiler.record_function for anatomy traces
        self._gap_totals: Dict[str, float] = {}
        self._last_close_t: float = 0.0
        self._last_close_name: str = "step_begin"

    def begin_step(self, *, step_t0: Optional[float] = None) -> None:
        self._totals = {k: 0.0 for k in _ALL_BUCKETS}
        self._active = None
        self._active_t0 = 0.0
        self._step_t0 = float(step_t0 if step_t0 is not None else time.perf_counter())
        self._audit = {}
        self._started = True
        self._gap_totals = {}
        self._last_close_t = self._step_t0
        self._last_close_name = "step_begin"

    def _close_active(self) -> None:
        if self._rf_ctx is not None:
            try:
                self._rf_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._rf_ctx = None
        if self._active is None:
            return
        _maybe_cuda_sync()
        now = time.perf_counter()
        elapsed = (now - self._active_t0) * 1000.0
        self._totals[self._active] = self._totals.get(self._active, 0.0) + elapsed
        if _GAP_AUDIT:
            self._last_close_t = now
            self._last_close_name = self._active
        self._active = None
        self._active_t0 = 0.0

    def start_bucket(self, name: str) -> None:
        if name not in EXCLUSIVE_BUCKET_KEYS:
            raise ValueError(f"unknown timing bucket: {name}")
        if not self._started:
            self.begin_step()
        self._close_active()
        _maybe_cuda_sync()
        if _GAP_AUDIT and self._last_close_t > 0.0:
            gap = (time.perf_counter() - self._last_close_t) * 1000.0
            if gap > 0.05:
                edge = f"{self._last_close_name}->{name}"
                self._gap_totals[edge] = self._gap_totals.get(edge, 0.0) + gap
        if _ANATOMY_RF:
            try:
                import torch

                self._rf_ctx = torch.profiler.record_function(f"bucket::{name}")
                self._rf_ctx.__enter__()
            except Exception:
                self._rf_ctx = None
        self._active = name
        self._active_t0 = time.perf_counter()

    def stop_bucket(self, name: Optional[str] = None) -> None:
        if self._active is None:
            return
        if name is not None and self._active != name:
            return
        self._close_active()

    @contextmanager
    def bucket(self, name: str) -> Iterator[None]:
        self.start_bucket(name)
        try:
            yield
        finally:
            self.stop_bucket(name)

    def add_ms(self, name: str, ms: float) -> None:
        """Add pre-measured exclusive time (closes any active bucket first)."""
        if name not in EXCLUSIVE_BUCKET_KEYS:
            return
        if not self._started:
            self.begin_step()
        self._close_active()
        if ms > 0:
            self._totals[name] = self._totals.get(name, 0.0) + float(ms)
        if _GAP_AUDIT:
            now = time.perf_counter()
            if self._last_close_t > 0.0:
                # Wall between the last close and the (approximate) start of
                # this pre-measured region would otherwise vanish from the
                # audit while still landing in misc_ms.
                gap = (now - self._last_close_t) * 1000.0 - max(float(ms), 0.0)
                if gap > 0.05:
                    edge = f"{self._last_close_name}->add:{name}"
                    self._gap_totals[edge] = self._gap_totals.get(edge, 0.0) + gap
            # The pre-measured region ends roughly now; gaps count from here.
            self._last_close_t = now
            self._last_close_name = name

    def set_audit(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is not None:
                self._audit[key] = value

    def snapshot_totals(self) -> Dict[str, float]:
        """Return accumulated bucket totals (active bucket not included)."""
        self._close_active()
        return {k: round(float(self._totals.get(k, 0.0)), 2) for k in _ALL_BUCKETS}

    def finalize_step(self, *, e2e_ms: Optional[float] = None) -> Dict[str, Any]:
        self._close_active()
        if _GAP_AUDIT and self._last_close_t > 0.0:
            tail_gap = (time.perf_counter() - self._last_close_t) * 1000.0
            if tail_gap > 0.05:
                edge = f"{self._last_close_name}->finalize"
                self._gap_totals[edge] = self._gap_totals.get(edge, 0.0) + tail_gap
        if _GAP_AUDIT and self._gap_totals:
            self._audit["timing_gaps"] = {
                k: round(v, 2)
                for k, v in sorted(self._gap_totals.items(), key=lambda kv: -kv[1])
                if v >= 0.2
            }
        if e2e_ms is None:
            e2e_ms = (time.perf_counter() - self._step_t0) * 1000.0 if self._started else 0.0
        exclusive_sum = sum(self._totals.get(k, 0.0) for k in EXCLUSIVE_BUCKET_KEYS)
        misc = max(float(e2e_ms) - exclusive_sum, 0.0)
        self._totals["misc_ms"] = misc
        timings = {k: round(self._totals.get(k, 0.0), 2) for k in _ALL_BUCKETS}
        gap = round(float(e2e_ms) - exclusive_sum - misc, 2)
        timing_e2e_ms = round(float(e2e_ms), 2)
        out = {
            "timings_exclusive_ms": timings,
            "timing_exclusive_sum_ms": round(exclusive_sum, 2),
            "timing_e2e_ms": timing_e2e_ms,
            "timing_gap_ms": gap,
            "timing_audit": dict(self._audit),
        }
        self._started = False
        return out


def get_step_ledger(model) -> StepTimingLedger:
    inner = _inner_from_model(model)
    ledger = getattr(inner, "_geopact_step_timing_ledger", None)
    if ledger is None:
        ledger = StepTimingLedger()
        inner._geopact_step_timing_ledger = ledger
    return ledger


def reset_step_ledger(model) -> StepTimingLedger:
    inner = _inner_from_model(model)
    ledger = StepTimingLedger()
    inner._geopact_step_timing_ledger = ledger
    return ledger


def begin_step_timing(model, *, step_t0: Optional[float] = None) -> StepTimingLedger:
    try:
        from lmms_eval.models.geopact_vl_hook import clear_geopact_mrope_caches

        clear_geopact_mrope_caches(model)
    except Exception:
        pass
    inner = _inner_from_model(model)
    if hasattr(inner, "_geopact_step_flat_deepstack"):
        delattr(inner, "_geopact_step_flat_deepstack")
    ledger = reset_step_ledger(model)
    ledger.begin_step(step_t0=step_t0)
    return ledger


def merge_exclusive_into_usage(usage: Dict[str, Any], ledger_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ledger payload into usage stats for profiling export."""
    timings = ledger_payload.get("timings_exclusive_ms") or {}
    usage = dict(usage)
    usage["timings_exclusive_ms"] = timings
    for key, value in timings.items():
        usage[key] = value
    usage["timing_exclusive_sum_ms"] = ledger_payload.get("timing_exclusive_sum_ms", 0.0)
    usage["timing_gap_ms"] = ledger_payload.get("timing_gap_ms", 0.0)
    timing_e2e = ledger_payload.get("timing_e2e_ms")
    if timing_e2e is not None:
        usage["timing_e2e_ms"] = float(timing_e2e)
        usage["e2e_latency_ms"] = round(float(timing_e2e), 2)
    audit = ledger_payload.get("timing_audit") or {}
    if audit:
        usage["timing_audit"] = audit
        for k, v in audit.items():
            if k not in usage:
                usage[k] = v
    return usage


def vision_prefill_from_exclusive(timings: Dict[str, float]) -> float:
    return round(
        sum(
            float(timings.get(k, 0.0) or 0.0)
            for k in (
                "processor_ms",
                "suffix_construct_ms",
                "vision_encode_ms",
                "sticky_keep_ms",
                "origpos_mrope_ms",
                "compact_gather_ms",
                "deepstack_align_ms",
                "vlm_prefill_ms",
            )
        ),
        2,
    )


def llm_prefill_from_exclusive(timings: Dict[str, float]) -> float:
    return round(
        float(timings.get("dialogue_kv_batch_prefill_ms", 0.0) or 0.0)
        + float(timings.get("dialogue_suffix_mrope_ms", 0.0) or 0.0)
        + float(timings.get("llm_prefill_ms", 0.0) or 0.0),
        2,
    )


def kv_finalize_from_exclusive(timings: Dict[str, float]) -> float:
    return round(float(timings.get("dialogue_kv_finalize_ms", 0.0) or 0.0), 2)
