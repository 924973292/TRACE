"""Persistent subprocess worker for isolated Module A (prefetch path)."""
from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

_CLIENT: Optional["ModuleAWorkerClient"] = None
_CLIENT_LOCK = threading.Lock()


def module_a_subprocess_pool_enabled() -> bool:
    if not os.environ.get("GEOPACT_MODULE_A_SUBPROCESS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_POOL", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def module_a_worker_shm_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_WORKER_SHM", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def worker_yolo_device() -> str:
    explicit = os.environ.get("GEOPACT_MODULE_A_WORKER_YOLO_DEVICE", "").strip()
    if explicit:
        return explicit
    return "0"


def subprocess_prewarm_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_PREWARM", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _subprocess_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "module_a_run_subprocess.py"


def _subprocess_python() -> str:
    return os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_PYTHON", "").strip() or sys.executable


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


class ModuleAWorkerClient:
    """One long-lived Module A child; newline-delimited JSON RPC on stdin/stdout."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._req_id = 0
        self._io_lock = threading.Lock()
        self._start()

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _read_json_line(self, *, what: str, timeout_s: float = 300.0) -> dict:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError(f"GeoPACT Module A worker not running while waiting for {what}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                err = self._read_stderr_tail()
                raise RuntimeError(
                    f"GeoPACT Module A worker exited (code={code}) before {what}: {err}"
                )
            stripped = line.strip()
            if not stripped:
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                _log.warning(
                    "GeoPACT Module A worker ignored non-JSON stdout before %s: %r",
                    what,
                    stripped[:240],
                )
        raise RuntimeError(
            f"GeoPACT Module A worker timed out after {timeout_s:.0f}s waiting for {what}"
        )

    def _start(self) -> None:
        script = _subprocess_script()
        if not script.exists():
            raise RuntimeError(f"GeoPACT: Module A worker script missing: {script}")
        env = os.environ.copy()
        env["MIU_YOLO_DEVICE"] = worker_yolo_device()
        env.setdefault("MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK", "1")
        self._proc = subprocess.Popen(
            [_subprocess_python(), str(script), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._proc.stdout is not None
        ready = self._read_json_line(what="ready", timeout_s=120.0)
        if not ready.get("ready"):
            raise RuntimeError(f"GeoPACT Module A worker not ready: {ready}")

    def _read_stderr_tail(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            lines = self._proc.stderr.readlines()
            return lines[-1].strip() if lines else ""
        except Exception:
            return ""

    def _write_image_payload(
        self,
        pil_image,
        req: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], float]:
        ipc_ms = 0.0
        if module_a_worker_shm_enabled():
            t0 = time.perf_counter()
            try:
                rgb = pil_image.convert("RGB")
                width, height = rgb.size
                arr = np.asarray(rgb, dtype=np.uint8)
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"unexpected RGB shape: {arr.shape!r}")
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                try:
                    dst = np.ndarray(arr.shape, dtype=np.uint8, buffer=shm.buf)
                    dst[:] = arr
                    req["image_shm"] = {
                        "name": shm.name,
                        "width": int(width),
                        "height": int(height),
                    }
                    ipc_ms = (time.perf_counter() - t0) * 1000.0
                    return None, shm.name, ipc_ms
                except Exception:
                    shm.close()
                    shm.unlink()
                    raise
            except Exception as exc:
                _log.debug("Module A SHM encode failed, falling back to PNG: %r", exc)
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = handle.name
            pil_image.convert("RGB").save(tmp_path)
        req["image_path"] = tmp_path
        ipc_ms = (time.perf_counter() - t0) * 1000.0
        return tmp_path, None, ipc_ms

    def detect(
        self,
        pil_image,
        detector: str,
        policy: str,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[dict], Dict[str, Any]]:
        if not self._alive():
            self._start()
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        cleanup_path: Optional[str] = None
        shm_name: Optional[str] = None
        with self._io_lock:
            self._req_id += 1
            req_id = self._req_id
            try:
                req: Dict[str, Any] = {
                    "id": req_id,
                    "detector": detector,
                    "policy": policy,
                }
                if grid_hw is not None:
                    req["grid_hw"] = [int(grid_hw[0]), int(grid_hw[1])]
                cleanup_path, shm_name, ipc_ms = self._write_image_payload(pil_image, req)
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                resp = self._read_json_line(what=f"response id={req_id}")
                if int(resp.get("id", -1)) != req_id:
                    raise RuntimeError(
                        f"GeoPACT Module A worker response id mismatch: "
                        f"expected {req_id}, got {resp.get('id')}"
                    )
                if not resp.get("ok"):
                    raise RuntimeError(
                        str(resp.get("error") or "Module A worker request failed")
                    )
                meta = dict(resp.get("meta") or {})
                timings = dict(meta.get("timings_ms") or {})
                if req.get("image_shm"):
                    timings["ipc_shm_ms"] = round(ipc_ms, 2)
                else:
                    timings["ipc_png_ms"] = round(ipc_ms, 2)
                meta["timings_ms"] = timings
                return list(resp.get("mius") or []), meta
            finally:
                if cleanup_path and os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass
                if shm_name:
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        shm.close()
                        shm.unlink()
                    except Exception:
                        pass

    def shutdown(self) -> None:
        if not self._alive():
            return
        assert self._proc is not None and self._proc.stdin is not None
        try:
            with self._io_lock:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            self._proc.wait(timeout=30)
        except Exception as exc:
            _log.debug("Module A worker shutdown: %r", exc)
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None


def get_module_a_worker() -> ModuleAWorkerClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None or not _CLIENT._alive():
            _CLIENT = ModuleAWorkerClient()
        return _CLIENT


def shutdown_module_a_worker() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.shutdown()
            _CLIENT = None


def prewarm_module_a_worker() -> bool:
    if not subprocess_prewarm_enabled():
        return False
    if not module_a_subprocess_pool_enabled():
        return False
    if not os.environ.get("GEOPACT_MODULE_A_SUBPROCESS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    t0 = time.perf_counter()
    get_module_a_worker()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _log.info("GeoPACT Module A worker prewarmed in %.0f ms", elapsed_ms)
    return True


def run_module_a_subprocess_pooled(
    pil_image,
    detector: str,
    policy: str,
    model=None,
) -> List[dict]:
    grid_hw = None
    if model is not None:
        grid_hw = getattr(model, "_geopact_module_a_grid_hw", None)
    mius, meta = get_module_a_worker().detect(
        pil_image,
        detector,
        policy,
        grid_hw=grid_hw,
    )
    if model is not None:
        model._geopact_last_module_a_meta = meta
        model._geopact_module_a_subprocess_isolated = True
        model._geopact_module_a_subprocess_pooled = True
        timings = meta.get("timings_ms") or {}
        if timings.get("ipc_shm_ms") is not None:
            model._geopact_module_a_ipc_shm_ms = timings.get("ipc_shm_ms")
        if timings.get("ipc_png_ms") is not None:
            model._geopact_module_a_ipc_png_ms = timings.get("ipc_png_ms")
    return mius


def run_module_a_subprocess_oneshot(
    pil_image,
    detector: str,
    policy: str,
    model=None,
) -> List[dict]:
    import subprocess
    import tempfile

    script = _subprocess_script()
    if not script.exists():
        raise RuntimeError(f"GeoPACT: Module A subprocess script missing: {script}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            tmp_path = handle.name
            pil_image.convert("RGB").save(tmp_path)
        cmd = [_subprocess_python(), str(script), tmp_path, "--detector", detector, "--policy", policy]
        grid_hw = None
        if model is not None:
            grid_hw = getattr(model, "_geopact_module_a_grid_hw", None)
        if grid_hw is not None:
            cmd.extend(["--grid-hw", str(int(grid_hw[0])), str(int(grid_hw[1]))])
        env = os.environ.copy()
        env["MIU_YOLO_DEVICE"] = worker_yolo_device()
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("GEOPACT_MODULE_A_SUBPROCESS_TIMEOUT", "300")),
            env=env,
        )
        payload = json.loads(proc.stdout.strip() or "{}")
        mius = list(payload.get("mius") or [])
        meta = dict(payload.get("meta") or {})
        if model is not None:
            model._geopact_last_module_a_meta = meta
            model._geopact_module_a_subprocess_isolated = True
        return mius
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"GeoPACT Module A subprocess failed (exit={exc.returncode}): "
            f"{stderr.splitlines()[-1] if stderr else '<no stderr>'}"
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


atexit.register(shutdown_module_a_worker)
