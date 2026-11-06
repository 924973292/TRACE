"""P4: fuse RMSNorm kernels in the decode loop via ``torch.compile``.

Anatomy finding: each decode step launches ~8.7k kernels; RMSNorm alone is
5 small elementwise/reduction kernels per call (cast, pow-mean, rsqrt-mul,
weight-mul, cast back) and runs 2x per decoder layer plus q/k norms. Compiling
one shared RMSNorm function (``dynamic=True``) collapses each call to 1-2
fused kernels without touching semantics.

Env gate: ``GEOPACT_COMPILE_NORMS=1`` (default OFF). Patch is applied at the
class level after model load; numerics may differ from eager at float32
reduction-order level, so the A/B gate decides whether it stays enabled.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
from loguru import logger as eval_logger


def compile_norms_enabled() -> bool:
    raw = os.environ.get("GEOPACT_COMPILE_NORMS", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _rms_norm_ref(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm math, identical to HF Qwen3VLTextRMSNorm.forward."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


_COMPILED_RMS = None


def _compiled_rms():
    global _COMPILED_RMS
    if _COMPILED_RMS is None:
        _COMPILED_RMS = torch.compile(_rms_norm_ref, dynamic=True, fullgraph=True)
    return _COMPILED_RMS


def _looks_like_rms_norm(module: torch.nn.Module) -> bool:
    if "RMSNorm" not in type(module).__name__:
        return False
    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    return isinstance(weight, torch.Tensor) and isinstance(eps, float)


def maybe_compile_norms(model: Any) -> Dict[str, Any]:
    """Patch every RMSNorm class in ``model`` to use the compiled kernel.

    Returns an audit dict; on any failure the patch is skipped entirely
    (eager forward retained) and the reason is logged.
    """
    audit: Dict[str, Any] = {"compile_norms": False}
    if not compile_norms_enabled():
        return audit
    try:
        classes = {}
        for _, module in model.named_modules():
            if _looks_like_rms_norm(module):
                classes.setdefault(type(module), 0)
                classes[type(module)] += 1
        if not classes:
            audit["compile_norms_reason"] = "no_rmsnorm_found"
            return audit
        fn = _compiled_rms()
        # Smoke-test the compiled kernel now: torch.compile fails lazily at
        # first call (e.g. no triton in env), which must not surface inside
        # a model forward mid-eval.
        probe_mod = next(m for m in model.modules() if _looks_like_rms_norm(m))
        probe = torch.randn(
            1, 2, probe_mod.weight.shape[-1],
            dtype=probe_mod.weight.dtype, device=probe_mod.weight.device,
        )
        fn(probe, probe_mod.weight, probe_mod.variance_epsilon)

        def _forward(self, hidden_states):
            return fn(hidden_states, self.weight, self.variance_epsilon)

        for cls in classes:
            if getattr(cls, "_geopact_compiled_norm", False):
                continue
            cls._geopact_eager_forward = cls.forward
            cls.forward = _forward
            cls._geopact_compiled_norm = True
        audit["compile_norms"] = True
        audit["compile_norms_classes"] = {c.__name__: n for c, n in classes.items()}
        eval_logger.info(
            "GeoPACT compile_norms: patched {} ({} modules total)",
            ", ".join(c.__name__ for c in classes),
            sum(classes.values()),
        )
    except Exception as exc:
        audit["compile_norms_reason"] = f"error:{exc}"
        eval_logger.warning("GeoPACT compile_norms failed, keeping eager: {}", exc)
    return audit


def undo_compile_norms(model: Any) -> None:
    """Restore eager RMSNorm forwards (used by verify scripts)."""
    seen = set()
    for _, module in model.named_modules():
        cls = type(module)
        if cls in seen:
            continue
        seen.add(cls)
        if getattr(cls, "_geopact_compiled_norm", False):
            cls.forward = cls._geopact_eager_forward
            del cls._geopact_eager_forward
            cls._geopact_compiled_norm = False
