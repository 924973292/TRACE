"""ABC latency anatomy instrumentation (env-gated, default OFF, zero-semantic).

Modes (comma list in ``GEOPACT_ANATOMY``):
  trace     -- torch.profiler (CPU+CUDA, shapes, memory, stack) per step,
               chrome trace + key_averages dumped per profiled step.
  syncdebug -- torch.cuda.set_sync_debug_mode("warn") + a warnings hook that
               appends every implicit-sync warning WITH python stack to a log.
  pyprofile -- cProfile across profiled steps, pstats dumped at exit.
  memory    -- torch.cuda.memory._record_memory_history snapshot dump.

Step selection: ``GEOPACT_ANATOMY_STEPS`` = "lo:hi" half-open range of the
global chunk counter (default "8:16"). ``GEOPACT_ANATOMY_DIR`` = output dir.

Everything is a no-op unless GEOPACT_ANATOMY is set; the hook call sites in
qwen3_vl_gui add only a dict lookup per step when disabled.
"""

from __future__ import annotations

import atexit
import os
import time
import traceback
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

_STATE: Dict[str, Any] = {
    "modes": None,  # parsed lazily
    "chunk_counter": -1,
    "cprofile": None,
    "sync_hook_installed": False,
    "memory_recording": False,
}


def _modes() -> frozenset:
    if _STATE["modes"] is None:
        raw = os.environ.get("GEOPACT_ANATOMY", "").strip().lower()
        _STATE["modes"] = frozenset(x.strip() for x in raw.split(",") if x.strip())
    return _STATE["modes"]


def anatomy_enabled() -> bool:
    return bool(_modes())


def _out_dir() -> Path:
    d = Path(os.environ.get("GEOPACT_ANATOMY_DIR", "logs/anatomy"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _step_range() -> tuple:
    raw = os.environ.get("GEOPACT_ANATOMY_STEPS", "8:16")
    try:
        lo, hi = raw.split(":", 1)
        return int(lo), int(hi)
    except Exception:
        return 8, 16


def arf(name: str):
    """record_function wrapper: real annotation only while a trace is active."""
    if not _modes():
        return nullcontext()
    try:
        import torch

        return torch.profiler.record_function(f"abc::{name}")
    except Exception:
        return nullcontext()


# ---------------------------------------------------------------------------
# syncdebug
# ---------------------------------------------------------------------------

def _install_sync_hook() -> None:
    if _STATE["sync_hook_installed"]:
        return
    import torch

    torch.cuda.set_sync_debug_mode("warn")
    log_path = _out_dir() / "sync_warnings.log"
    orig_show = warnings.showwarning

    def _hook(message, category, filename, lineno, file=None, line=None):
        try:
            text = str(message)
            if "synchron" in text.lower() or "cudaStreamSynchronize" in text or "cudaDeviceSynchronize" in text:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"=== SYNC step={_STATE['chunk_counter']} t={time.perf_counter():.3f} {text}\n")
                    f.write("".join(traceback.format_stack(limit=24)))
                    f.write("\n")
                return
        except Exception:
            pass
        orig_show(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _hook
    warnings.simplefilter("always")
    _STATE["sync_hook_installed"] = True


# ---------------------------------------------------------------------------
# pyprofile
# ---------------------------------------------------------------------------

def _get_cprofile():
    if _STATE["cprofile"] is None:
        import cProfile

        _STATE["cprofile"] = cProfile.Profile()

        def _dump():
            prof = _STATE["cprofile"]
            if prof is None:
                return
            out = _out_dir() / "pyprofile.pstats"
            prof.dump_stats(str(out))
            import pstats

            with open(_out_dir() / "pyprofile_top200.txt", "w", encoding="utf-8") as f:
                st = pstats.Stats(prof, stream=f)
                st.sort_stats("cumulative").print_stats(200)
                st.sort_stats("tottime").print_stats(200)

        atexit.register(_dump)
    return _STATE["cprofile"]


# ---------------------------------------------------------------------------
# step hooks (called from qwen3_vl_gui)
# ---------------------------------------------------------------------------

def anatomy_step_begin() -> Optional[Dict[str, Any]]:
    """Call at the top of each generate chunk. Returns a token for step_end."""
    modes = _modes()
    if not modes:
        return None
    _STATE["chunk_counter"] += 1
    idx = _STATE["chunk_counter"]
    lo, hi = _step_range()
    active = lo <= idx < hi
    token: Dict[str, Any] = {"idx": idx, "active": active, "modes": modes}

    if "syncdebug" in modes:
        _install_sync_hook()

    if not active:
        return token

    if "trace" in modes:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=os.environ.get("GEOPACT_ANATOMY_STACK", "1") == "1",
            )
            prof.__enter__()
            token["prof"] = prof
        except Exception as exc:  # pragma: no cover - diagnostics only
            token["trace_error"] = repr(exc)

    if "pyprofile" in modes:
        _get_cprofile().enable()
        token["pyprofile"] = True

    if "memory" in modes and not _STATE["memory_recording"]:
        try:
            import torch

            torch.cuda.memory._record_memory_history(max_entries=200000)
            _STATE["memory_recording"] = True
        except Exception:
            pass

    return token


def anatomy_step_end(token: Optional[Dict[str, Any]], *, step_id: int = -1) -> None:
    if not token:
        return
    idx = token["idx"]
    modes = token["modes"]

    if token.get("pyprofile"):
        _STATE["cprofile"].disable()

    prof = token.get("prof")
    if prof is not None:
        try:
            import torch

            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            base = _out_dir() / f"trace_chunk{idx:04d}_step{step_id}"
            prof.export_chrome_trace(str(base) + ".json.gz")
            ka = prof.key_averages(group_by_input_shape=False)
            with open(str(base) + "_keyavg.txt", "w", encoding="utf-8") as f:
                f.write(ka.table(sort_by="cpu_time_total", row_limit=300))
            with open(str(base) + "_keyavg_cuda.txt", "w", encoding="utf-8") as f:
                f.write(ka.table(sort_by="cuda_time_total", row_limit=300))
        except Exception as exc:  # pragma: no cover
            try:
                with open(_out_dir() / "trace_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"chunk={idx} step={step_id} err={exc!r}\n")
            except Exception:
                pass

    if "memory" in modes and _STATE["memory_recording"]:
        lo, hi = _step_range()
        if idx >= hi - 1:
            try:
                import torch

                snap = _out_dir() / f"memory_snapshot_chunk{idx:04d}.pickle"
                torch.cuda.memory._dump_snapshot(str(snap))
                torch.cuda.memory._record_memory_history(enabled=None)
                _STATE["memory_recording"] = False
            except Exception:
                pass
