"""Restore qwen3_vl modeling from repo patch baseline and re-apply GeoPACT patches."""
import ast
import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "patches" / "qwen3_vl" / "modeling_qwen3_vl.py"
# Lock must live on the shared filesystem: queues on *different pods* restore
# into the same NFS venv concurrently, and a /tmp lock is pod-local (the
# resulting os.replace vs read race shows up as ESTALE on the readers).
LOCK = REPO / "logs" / ".restore_q3vl_geopact_patches.lock"


def resolve_qwen3_vl_modeling_py() -> Path:
    """Site-packages path for modeling_qwen3_vl.py in the eval venv."""
    exe = Path(os.environ.get("EVAL_PYTHON", "").strip() or sys.executable).expanduser()
    if not exe.is_absolute():
        exe = exe.resolve()
    else:
        # bin/python often symlinks to /usr/bin/python3.10 — do not resolve past venv.
        exe = exe.absolute()
    venv_root = exe.parent.parent
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    dst = (
        venv_root
        / "lib"
        / py_ver
        / "site-packages"
        / "transformers"
        / "models"
        / "qwen3_vl"
        / "modeling_qwen3_vl.py"
    )
    if dst.parent.is_dir():
        return dst

    return dst

DEEPSTACK_SILENT_SKIP = (
    "    def _deepstack_process(\n"
    "        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor\n"
    "    ):\n"
    "        visual_pos_masks = visual_pos_masks.to(hidden_states.device)\n"
    "        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)\n"
    "        # GeoPACT: skip deepstack when sequence was pruned (shape mismatch guard)\n"
    "        if visual_pos_masks.dim() < 2 or visual_pos_masks.shape[-1] != hidden_states.shape[1]:\n"
    "            return hidden_states\n"
    "        n_vis = int(visual_pos_masks.sum().item())\n"
    "        if visual_embeds.shape[0] != n_vis:\n"
    "            return hidden_states\n"
    "        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds\n"
    "        hidden_states[visual_pos_masks, :] = local_this\n"
    "        return hidden_states"
)

DEEPSTACK_FAIL_LOUD_MARKERS = (
    "GeoPACT deepstack mismatch: embeds=",
    "visual_pos_masks shape",
)

# Legacy destructive patch — strip if present (masks origpos MRoPE failures).
ROPE_FALLBACK_NEW = """                try:
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask=attention_mask_tensor,
                    )
                    self.rope_deltas = rope_deltas
                except (RuntimeError, ValueError):
                    # Pruning changed sequence length; fall back to text-only RoPE
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        None,   # no image grid
                        None,   # no video grid
                        attention_mask=attention_mask_tensor,
                    )
                    self.rope_deltas = rope_deltas"""

ROPE_FALLBACK_OLD = """                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas"""


@contextlib.contextmanager
def restore_lock():
    """Serialize restore writes when multiple eval workers start together."""
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
        dst = resolve_qwen3_vl_modeling_py()
        print(f"target: {dst}")

        if not SRC.is_file():
            print(f"ERROR: baseline patch missing: {SRC}")
            sys.exit(1)
        if not dst.parent.is_dir():
            print(f"ERROR: transformers qwen3_vl package dir missing: {dst.parent}")
            print(f"  EVAL_PYTHON={os.environ.get('EVAL_PYTHON', '')!r}")
            print(f"  sys.executable={sys.executable!r}")
            sys.exit(1)

        backup = dst.with_suffix(".py.bak_restore")
        if dst.is_file():
            try:
                shutil.copy2(dst, backup)
                print(f"backup: {backup}")
            except OSError as exc:
                # Best-effort only: dst may be mid-os.replace from another pod
                # (NFS ESTALE); the restore below rewrites it from SRC anyway.
                print(f"WARN: backup skipped ({exc})")

        code = SRC.read_text(encoding="utf-8")

        required = [
            "visual_keep_flat_indices",
            "_gp_pos = _gp.get",
            "force_rope_recompute",
            "GeoPACT hook returned no position_ids",
            "never fall back to text-only RoPE",
            "_geopact_prefill_payload",
            "_geopact_generate_position_ids",
            "_geopact_generate_cache_start",
            "_is_geopact_suffix_prefill",
            "_is_geopact_suffix_prefill_forward",
            "_past_kv_len <= 0",
            "attention_position_ids = None",
            "past_tokens={_past_tokens} allowed=",
            "_omnigui_last_generation_timing",
            "_omnigui_reset_generation_timing",
            "geopact_parb_attention_forward",
            "should_apply_parb",
            "parb_lift_enabled",
            "try_parb_lift_attention",
            "trp_enabled",
            "try_trp_attention",
            "should_apply_trp",
            "gsc_enabled",
            "try_gsc_attention",
            "should_apply_gsc",
            "correction_enabled",
            "try_correction_attention",
            "should_apply_correction",
            "vsas_enabled",
            "try_vsas_attention_internal",
            "should_apply_vsas",
        ]
        for marker in required:
            if marker not in code:
                print(f"ERROR: baseline missing marker: {marker!r}")
                sys.exit(1)
        print("hook MRoPE v3 markers: OK")

        if DEEPSTACK_SILENT_SKIP in code:
            print("ERROR: baseline still has silent deepstack skip — update patches/qwen3_vl baseline")
            sys.exit(1)
        if not all(marker in code for marker in DEEPSTACK_FAIL_LOUD_MARKERS):
            print("ERROR: baseline missing fail-loud deepstack markers in _deepstack_process")
            sys.exit(1)
        print("deepstack fail-loud: OK")

        if ROPE_FALLBACK_NEW in code:
            code = code.replace(ROPE_FALLBACK_NEW, ROPE_FALLBACK_OLD, 1)
            print("removed legacy text-only RoPE fallback")
        elif ROPE_FALLBACK_OLD in code:
            print("text-only RoPE fallback: absent (OK)")
        else:
            print("WARN: get_rope_index prefill block not found for fallback check")

        ast.parse(code)
        # Unique tmp per pod+pid: fcntl.flock is advisory and has proven
        # unreliable across pods on this NFS mount — a shared ".tmp_restore"
        # name lets another pod os.replace() our tmp away mid-restore
        # (observed as FileNotFoundError on os.replace).
        tmp = dst.with_suffix(f".py.tmp_restore.{os.uname().nodename}.{os.getpid()}")
        tmp.write_text(code, encoding="utf-8")
        try:
            os.replace(tmp, dst)
        except FileNotFoundError:
            # NFS raced us anyway; rewrite once and retry.
            tmp.write_text(code, encoding="utf-8")
            os.replace(tmp, dst)
        print(f"restored from {SRC}")

        import importlib
        import time

        last_err = None
        for attempt in range(8):
            try:
                import transformers.models.qwen3_vl.modeling_qwen3_vl as m

                importlib.reload(m)
                assert hasattr(m, "Qwen3VLModel"), "Qwen3VLModel missing after restore"
                assert "GeoPACT deepstack mismatch" in code
                print("restore_q3vl_geopact_patches: ALL OK")
                last_err = None
                break
            except (FileNotFoundError, OSError, ImportError) as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        if last_err is not None:
            raise last_err


if __name__ == "__main__":
    main()
