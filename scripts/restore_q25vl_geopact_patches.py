"""Restore qwen2_5_vl modeling from repo patch baseline (GeoPACT prune slots).

Deploys patches/qwen2_5_vl/modeling_qwen2_5_vl.py into the eval venv
site-packages, mirroring restore_q3vl_geopact_patches.py (shared-FS lock,
pod+PID-unique tmp, retrying os.replace, post-restore import check).
"""
import ast
import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "patches" / "qwen2_5_vl" / "modeling_qwen2_5_vl.py"
LOCK = REPO / "logs" / ".restore_q25vl_geopact_patches.lock"


def resolve_modeling_py() -> Path:
    """Site-packages path for modeling_qwen2_5_vl.py in the eval venv."""
    exe = Path(os.environ.get("EVAL_PYTHON", "").strip() or sys.executable).expanduser()
    exe = exe.absolute() if exe.is_absolute() else exe.resolve()
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    def cand(venv_root: Path) -> Path:
        return (
            venv_root / "lib" / py_ver / "site-packages" / "transformers"
            / "models" / "qwen2_5_vl" / "modeling_qwen2_5_vl.py"
        )

    dst = cand(exe.parent.parent)
    if dst.parent.is_dir():
        return dst
    return dst


@contextlib.contextmanager
def restore_lock():
    handle = None
    try:
        handle = os.open(str(LOCK), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            import fcntl  # type: ignore

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        if handle is not None:
            try:
                try:
                    import fcntl  # type: ignore

                    fcntl.flock(handle, fcntl.LOCK_UN)
                except ImportError:
                    pass
            finally:
                os.close(handle)


def main() -> None:
    with restore_lock():
        dst = resolve_modeling_py()
        print(f"target: {dst}")

        if not SRC.is_file():
            print(f"ERROR: baseline patch missing: {SRC}")
            sys.exit(1)
        if not dst.parent.is_dir():
            print(f"ERROR: transformers qwen2_5_vl package dir missing: {dst.parent}")
            sys.exit(1)

        backup = dst.with_suffix(".py.bak_restore")
        if dst.is_file():
            try:
                shutil.copy2(dst, backup)
                print(f"backup: {backup}")
            except OSError as exc:
                print(f"WARN: backup skipped ({exc})")

        code = SRC.read_text(encoding="utf-8")

        required = [
            "_geopact_projector_prune_hook",
            "_apply_geopact_prune",
            "def apply_rotary_emb(x, cos, sin):",
            # write-once serving path (ported from the qwen3_vl patch)
            "_geopact_prefill_payload",
            "_is_geopact_suffix_prefill_forward",
            "_geopact_generate_position_ids",
            "_geopact_generate_cache_start",
            "GeoPACT hook returned no position_ids",
            "compact MRoPE must come from the hook (origpos).",
            "pinned_cache_start=",
            "_omnigui_last_generation_timing",
            "_omnigui_reset_generation_timing",
        ]
        for marker in required:
            if marker not in code:
                print(f"ERROR: baseline missing marker: {marker!r}")
                sys.exit(1)
        print("geopact prune-slot markers: OK")

        ast.parse(code)
        tmp = dst.with_suffix(f".py.tmp_restore.{os.uname().nodename}.{os.getpid()}")
        tmp.write_text(code, encoding="utf-8")
        try:
            os.replace(tmp, dst)
        except FileNotFoundError:
            tmp.write_text(code, encoding="utf-8")
            os.replace(tmp, dst)
        print(f"restored from {SRC}")

        import importlib
        import time

        last_err = None
        for attempt in range(8):
            try:
                import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as m

                importlib.reload(m)
                assert hasattr(m, "Qwen2_5_VLForConditionalGeneration")
                fcg = m.Qwen2_5_VLForConditionalGeneration
                assert "_apply_geopact_prune" in dir(fcg), "prune slot missing on FCG"
                print("restore_q25vl_geopact_patches: ALL OK")
                last_err = None
                break
            except (FileNotFoundError, OSError, ImportError) as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            raise last_err


if __name__ == "__main__":
    main()
