"""
lmms_eval/models/geopact_vl_hook.py

Shared GeoPACT pipeline for Qwen2-VL / Qwen3-VL based GUI agents.

Implements the three-stage GCC → HES → SBR pipeline as described in the
GeoPACT paper, adapted for image-only evaluation (no audio/video).

Every public symbol is designed to be imported by model wrappers
(lmms_eval/models/simple/ui_tars.py, qwen3_vl_gui.py, ...) and passed as
the _geopact_projector_prune_hook to the patched inner model class.

Policy string grammar (visual_prune_policy)
==========================================
Detector:
    [default / empty]  → canny_morph (Canny + morphological close)
    "yolo"             → OmniParser-v2 YOLO icon detector
    "miu_detector=canny" | "canny"  → explicit Canny
    "miu_detector=omniparser" | "omniparser" | "yolo" → YOLO

Stage 1 – GCC:
    [default]           → full pentagonal energy (all 5 attributes)
    "drop_scale"        → disable scale attribute (S-bar)
    "drop_entropy"      → disable entropy attribute (H-bar)
    "drop_contrast"     → disable contrast attribute (C-bar)
    "drop_containment"  → disable containment attribute (R-bar)
    "drop_grid"         → disable grid alignment attribute (G-bar)
    "drop_tau"          → set detector confidence tau=1.0 (ablate gating)

Stage 2 – HES (instruction-conditioned kernel herding):
    [default]           → full kernel herding with density × relevance weighting
    "random"            → random token selection (skips GCC scoring)
    "uniform"           → evenly spaced token stride (skips GCC; sticky-deterministic)
    "token_ickh_diversity" → token-level kernel herding, w_i=1 (dialogue sticky path)
    "token_ickh_relevance" → token-level kernel herding, w_i=r_i only (dialogue sticky)
    "token_ickh_miu_density" / "oas_ickh_cdct_*" → CDCT: w_j=r_j*(1+α*P_j), token ICKH (Module A ρ field)
    "token_instr_topk"  → pure instruction relevance top-K (no herding diversity)
    "token_div_topk"    → pure FPS diversity top-K on vision embeddings (no instruction)
    "imp_topk"          → importance-only top-K (no diversity pressure)
    "no_instr_rel"      → disable instruction text relevance (r_i = 1.0)
    "no_density"        → disable InfoDensity (ρ_i = 1.0)
    "diversity_only"    → weight = 1.0 for all (pure diversity maximisation)

Position encoding:
    [default / "seqpos"] → sequential positions in compacted sequence
    "origpos"            → preserve original spatial MRoPE position_ids

Stage 3 – SBR coord injection:
    [default / "coord_center"] → bundled centre-point tags after last image pad
    "no_coord"          → skip coord injection entirely
    "coord_tail"        → inject at prompt tail (ablation, hurts performance)
    "coord_topk=N"      → only inject top-N MIUs by score (env: GEOPACT_COORD_TOPK)
    "coord_shuffle"     → shuffle coordinate values (causal control)
    "coord_random"      → random coordinates (causal control)

Usage
=====
    from lmms_eval.models.geopact_vl_hook import make_geopact_hook
    hook = make_geopact_hook(
        visual_prune_ratio=0.10,
        visual_prune_policy="oas_ickh_coord_center",
        miu_detector="fusion_fast",      # blob + improved Canny (default fast stack)
    )
    # Register on the inner Model class (Qwen2VLModel / Qwen3VLModel):
    self._model.model._geopact_projector_prune_hook = hook
    # Before each generate() call, put the PIL image on the inner model:
    self._model.model._geopact_last_image = current_pil_image
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random as _random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

_WARNED_ONCE: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED_ONCE:
        _WARNED_ONCE.add(msg)
        _log.warning(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class HESResult(set):
    """Set of kept visual tokens plus MIU order selected by Module B."""

    def __init__(
        self,
        tokens=(),
        selected_miu_indices=None,
        picked_token_order=None,
        miu_backfill_tokens: int = 0,
    ):
        super().__init__(tokens)
        self.selected_miu_indices = list(selected_miu_indices or [])
        self.picked_token_order = list(picked_token_order or [])
        self.miu_backfill_tokens = int(miu_backfill_tokens or 0)


def _policy_has(policy: str, *keywords: str) -> bool:
    p = policy.lower()
    return any(kw.lower() in p for kw in keywords)


_DETECTOR_DIRECTIVE_CANON = {
    "ocr_only": "ocr_only",
    "omniparser": "omniparser",
    "fusion": "fusion_fast",
    "fusion_fast": "fusion_fast",
    "blob": "blob",
    "canny": "canny_morph",
    "canny_morph": "canny_morph",
}


def _get_detector(policy: str, miu_detector: str) -> str:
    """Resolve detector backend from policy string and explicit kwarg.

    Priority: explicit ``miu_detector=<name>`` directive in the policy string
    > legacy policy-substring inference > wrapper kwarg. Legacy substring
    inference (e.g. ``yolo`` -> omniparser) silently overriding an explicit
    kwarg is the historical root cause of no-op detector-swap arms; we keep
    the behavior for arm reproducibility but fail loud when it triggers.
    """
    p = policy.lower()
    m = re.search(r"miu_detector=([a-z_]+)", p)
    if m:
        canon = _DETECTOR_DIRECTIVE_CANON.get(m.group(1))
        if canon:
            return canon
    inferred = None
    if "ocr_only" in p:
        inferred = "ocr_only"
    elif "omniparser" in p or "yolo" in p:
        inferred = "omniparser"
    elif "fusion_fast" in p:
        inferred = "fusion_fast"
    elif "blob" in p:
        inferred = "blob"
    elif "canny" in p:
        inferred = "canny_morph"
    if inferred is not None:
        if miu_detector and inferred != miu_detector:
            _warn_once(
                f"geopact detector: policy substring inference ({inferred!r} from "
                f"policy={policy!r}) overrides explicit miu_detector={miu_detector!r}; "
                "use a 'miu_detector=<name>' policy directive to force the kwarg."
            )
        return inferred
    return miu_detector  # wrapper-level default


def _module_a_sideload_enabled() -> bool:
    """Run full Module A on a pure-B policy arm; B selection stays token-baseline."""
    return os.environ.get("GEOPACT_MODULE_A_SIDELOAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _module_a_sideload_policy(baseline_policy: str) -> str:
    """Policy for Module A compute when sideloading onto pure-B baseline."""
    explicit = os.environ.get("GEOPACT_MODULE_A_SIDELOAD_POLICY", "").strip()
    if explicit:
        return explicit
    return "oas_ickh_cdct_no_coord_yolo"


def _module_a_sideload_mode() -> str:
    """``yolo_only`` = detect only; ``full`` = complete Module A pipeline."""
    return os.environ.get("GEOPACT_MODULE_A_SIDELOAD_MODE", "full").strip().lower()


def _module_a_sideload_timing() -> str:
    """``eval_end`` / ``post_episode`` = queue until eval end (parity-safe); ``post_decode`` = per step."""
    explicit = os.environ.get("GEOPACT_MODULE_A_SIDELOAD_TIMING", "").strip().lower()
    if explicit in ("post_decode", "step", "per_step"):
        return "post_decode"
    return "eval_end"


def _module_a_prefetch_mode() -> Optional[str]:
    """Prepare-time Module A: ``cpu`` (parity-safe prefetch before prune)."""
    mode = os.environ.get("GEOPACT_MODULE_A_PREFETCH", "").strip().lower()
    if mode in {"", "0", "false", "off", "no", "none"}:
        return None
    return mode


def _module_a_prefetch_enabled() -> bool:
    return _module_a_prefetch_mode() is not None


def _module_a_export_weights_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_EXPORT_WEIGHTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dialogue_suffix_only_step(model) -> bool:
    return bool(getattr(model, "_geopact_dialogue_suffix_only_step", False))


def _prefetch_alpha0_cdct(policy: str) -> bool:
    """True when prefetch CDCT runs with α=0 (w_j=r_j; keep must match pure B)."""
    return _module_a_prefetch_enabled() and _policy_uses_cdct(policy) and _cdct_alpha(policy) == 0.0


def _prune_target(model=None) -> str:
    """Which frame the sticky protocol prunes: ``current`` (default) or ``history``."""
    from lmms_eval.models.geopact_session_kv import resolve_prune_target

    return resolve_prune_target(model)


def _prune_target_is_history(model=None) -> bool:
    return _prune_target(model) == "history"


def _sticky_frame_action(
    *,
    is_current: bool,
    prune_target: str,
    has_cached_mask: bool,
    two_stage: bool = False,
) -> str:
    """Resolve per-frame sticky action to ``prune`` / ``reuse`` / ``full``.

    Decision table (see plan):

    | prune_target | is_current | cached | action |
    |--------------|------------|--------|--------|
    | current      | True       | -      | prune  |
    | current      | False      | yes    | reuse  |
    | current      | False      | no     | full   |
    | history      | True       | -      | full   |
    | history      | False      | yes    | reuse  |
    | history      | False      | no     | prune  |

    NAP two-stage overrides one cell: ``history`` + ``is_current`` becomes
    ``prune`` (entry pruning at r_cur); every other row is unchanged.
    """
    history = str(prune_target or "").strip().lower() == "history"
    if is_current:
        if history:
            return "prune" if two_stage else "full"
        return "prune"
    if has_cached_mask:
        return "reuse"
    return "prune" if history else "full"


def _dialogue_prefetch_single_frame(model=None) -> bool:
    """Dialogue KV + prefetch: each step only prunes the current screenshot.

    Only meaningful under the ``current`` prune target; under ``history`` the
    active frame is kept full and history compression is owned by the KV
    rewrite path, so this short-circuit must not fire.
    """
    from lmms_eval.models.geopact_session_kv import dialogue_kv_enabled

    if _prune_target_is_history(model):
        return False
    return dialogue_kv_enabled() and _module_a_prefetch_enabled()


def _module_a_subprocess_enabled() -> bool:
    return os.environ.get("GEOPACT_MODULE_A_SUBPROCESS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _module_a_cache_active(model=None) -> bool:
    """LRU cache for Module A; off on dialogue+prefetch (LLM KV owns cross-step state)."""
    from lmms_eval.models.geopact_module_a_cache import module_a_cache_enabled

    if _dialogue_prefetch_single_frame(model):
        return False
    return module_a_cache_enabled()


def _assert_module_a_sideload_prefetch_exclusive() -> None:
    if _module_a_sideload_enabled() and _module_a_prefetch_enabled():
        raise RuntimeError(
            "GeoPACT: GEOPACT_MODULE_A_SIDELOAD and GEOPACT_MODULE_A_PREFETCH are mutually exclusive"
        )


def _keep_local_hash(keep_local: set) -> str:
    """Stable fingerprint of B keep indices for 04c vs 04i audit."""
    import hashlib

    payload = ",".join(str(int(i)) for i in sorted(int(x) for x in keep_local))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _should_cache_geopact_pil_image(policy: str) -> bool:
    """Whether to load PIL screenshot for Module A (incl. sideload on pure-B arms)."""
    from lmms_eval.models.geopact_qd_selector import bam_skip_module_a
    from lmms_eval.models.geopact_session_kv import dialogue_kv_enabled

    if bam_skip_module_a():
        # no-mass BAM: the screenshot only ever feeds the (skipped) detector;
        # the ~16ms/step PIL decode is pure dead weight (W-L). Sideload still
        # needs it for its post-decode metrics pass.
        return _module_a_sideload_enabled()
    if dialogue_kv_enabled() and _policy_is_token_baseline(policy):
        return _module_a_sideload_enabled() or _module_a_prefetch_enabled()
    return True


def _is_dialogue_sticky_token_prune(policy: str) -> bool:
    """Pure token selectors on dialogue sticky path (no Module A).

    Includes uniform/random and token-level ICKH (``token_ickh_*``).
    Excludes ``oas_*`` MIU paths — use ``_is_dialogue_sticky_embed_prune`` for routing.
    """
    p = policy.lower().strip()
    if _policy_has(
        p,
        "token_ickh_diversity",
        "token_ickh_relevance",
        "token_instr_topk",
        "token_div_topk",
    ):
        return True
    # published training-free baselines (feature-level, no Module A/B/C)
    if _policy_has(
        p, "divprune", "dart", "cdpruner", "visionzip", "vispruner",
        "prumerge", "trim", "visiontrim", "zooprune", "prunesid",
    ):
        return True
    if _policy_has(p, "oas"):
        return False
    if _policy_has(p, "uniform", "uniform_stride"):
        return True
    return p == "random"


def _is_dialogue_sticky_embed_prune(policy: str) -> bool:
    """Policies using lightweight ``prepare_dialogue_pruned_embed_inputs``.

    Token baselines (uniform/random/token_ickh) and ``oas_*`` MIU policies when
    dialogue KV + ``image_sticky`` are enabled.
    """
    p = policy.lower().strip()
    if _policy_has(p, "oas"):
        return True
    return _is_dialogue_sticky_token_prune(policy)


def _needs_sticky_relevance_query_cache(policy: str) -> bool:
    """Whether step-0 should cache global instruction ``q`` for suffix steps."""
    p = policy.lower().strip()
    if _policy_has(p, "token_ickh_relevance", "token_instr_topk"):
        return True
    if _policy_has(p, "cdpruner"):
        return True  # instruction-conditioned DPP baseline needs the query
    if _policy_has(p, "visiontrim"):
        return True  # TGVC text-guided complement needs the query
    if _policy_has(p, "oas") and _policy_has(p, "ickh") and not _policy_has(
        p, "no_instr_rel", "no_instruction", "diversity_only"
    ):
        return True
    return False


def _instr_wire_enabled() -> bool:
    """F1 (v3): wire the per-turn instruction query into every selector path.

    Historically only the dialogue sticky path passed ``instr_query`` down to
    Module B; the hook ``image_sticky`` and non-sticky paths silently fell
    back to the mean-of-all-text embedding (system prompt + history noise).
    Default on; GEOPACT_B_INSTR_WIRE=0 restores the legacy behaviour for
    attribution arms.
    """
    return os.environ.get("GEOPACT_B_INSTR_WIRE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _hook_instr_query_for_frame(model, policy: str, img_path) -> Optional[torch.Tensor]:
    """Per-turn instruction query for hook prune paths (F1), else None."""
    if not _instr_wire_enabled() or not _needs_sticky_relevance_query_cache(policy):
        return None
    try:
        q, source = relevance_query_for_frame(model, img_path)
    except Exception as exc:
        _log.warning("GeoPACT instr wire failed (non-fatal): %s", exc)
        return None
    try:
        diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
        diag["instr_wire_source"] = source
        model._geopact_last_prune_diag = diag
    except Exception:
        pass
    return q


def _policy_is_token_baseline(policy: str) -> bool:
    """Pure token subsampling on dialogue sticky path — skip Module A/B/C."""
    return _is_dialogue_sticky_token_prune(policy)


# ── E3 hybrid arms: published selector × Module A spatial-prior mass ─────────
# GEOPACT_PUB_HYBRID_MASS=1 grafts the query-independent spatial prior onto a
# published baseline's own scoring signal (attribution control: is the prior
# transferable across selectors?). Each selector receives the prior through its
# own score channel: VisPruner (attention), DivPrune (max-min utility), CDPruner
# (DPP relevance), PruneSID (PCA loading). Empty detections degrade gracefully
# to the unmodified published rule (mass ≡ 1). The whitelist lives in
# geopact_pub_baselines so the gate here and the dispatch guard there agree.

from lmms_eval.models.geopact_pub_baselines import (  # noqa: E402
    HYBRID_CAPABLE_SELECTORS as _PUB_HYBRID_SELECTORS,
)


def _pub_hybrid_mass_enabled() -> bool:
    return os.environ.get("GEOPACT_PUB_HYBRID_MASS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _pub_hybrid_alpha() -> float:
    """Prior strength α for hybrid mass m_j = 1 + α·P_j (default = main-table 2)."""
    try:
        return max(0.0, float(os.environ.get("GEOPACT_PUB_HYBRID_ALPHA", "2.0")))
    except ValueError:
        return 2.0


def _pub_hybrid_active(policy: str) -> bool:
    return _pub_hybrid_mass_enabled() and _policy_has(policy, *_PUB_HYBRID_SELECTORS)


# Set once per process by the hybrid path so the audit line does not repeat for
# every frame.
_PUB_HYBRID_LOGGED = False


def _prune_selector_label(policy: str) -> str:
    p = str(policy or "").lower()
    if _policy_uses_cdct(p):
        return "token_ickh_cdct"
    if _policy_has(p, "token_ickh_diversity"):
        return "token_ickh_diversity"
    if _policy_has(p, "token_ickh_relevance"):
        return "token_ickh_relevance"
    if _policy_has(p, "token_instr_topk"):
        return "token_instr_topk"
    if _policy_has(p, "token_div_topk"):
        return "token_div_topk"
    if _policy_has(p, "uniform", "uniform_stride"):
        return "uniform"
    if _policy_has(p, "prunesid"):
        return "prunesid"
    if _policy_has(p, "divprune"):
        return "divprune"
    if _policy_has(p, "cdpruner"):
        return "cdpruner"
    if _policy_has(p, "dart"):
        return "dart"
    if _policy_has(p, "visionzip"):
        return "visionzip"
    if _policy_has(p, "vispruner"):
        return "vispruner"
    if _policy_has(p, "prumerge"):
        return "prumergeplus"
    if _policy_has(p, "zooprune"):
        return "zooprune"
    # NOTE: "visiontrim" contains "trim" — keep this check before trim's.
    if _policy_has(p, "visiontrim"):
        return "visiontrim"
    if _policy_has(p, "trim"):
        return "trim"
    if p == "random":
        return "random"
    return str(policy or "unknown")


def _uniform_keep_indices(n_img: int, keep_k: int) -> set:
    """Evenly spaced visual token indices in raster order (deterministic)."""
    if n_img <= 0:
        return HESResult()
    keep_k = min(max(1, keep_k), n_img)
    if keep_k >= n_img:
        return set(range(n_img))
    if keep_k == 1:
        return {n_img // 2}
    step = (n_img - 1) / (keep_k - 1)
    return {int(round(i * step)) for i in range(keep_k)}


def _clamp_token_keep_k(n_img: int, keep_k_tokens: int) -> int:
    """Shared keep budget clamp for uniform / token-ICKH dialogue sticky selectors."""
    return min(max(1, int(keep_k_tokens)), int(n_img))


def _token_ickh_herding_weights(
    E_norm: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    policy: str,
) -> torch.Tensor:
    """Per-token kernel-herding weights (diversity: ones; relevance: max(0, cos to text))."""
    n_img = int(E_norm.shape[0])
    device = E_norm.device
    if _policy_has(policy, "token_ickh_diversity"):
        return torch.ones(n_img, dtype=E_norm.dtype, device=device)

    mask = image_mask[0] if image_mask.dim() > 1 else image_mask
    mask = mask.to(device=device).bool()
    text_embs = inputs_embeds[0][~mask]
    if text_embs.shape[0] == 0:
        return torch.ones(n_img, dtype=E_norm.dtype, device=device)

    q = F.normalize(text_embs.to(dtype=E_norm.dtype).mean(0, keepdim=True), dim=-1)
    weights = torch.clamp(E_norm @ q.transpose(-1, -2), min=0.0).squeeze(-1)
    if float(weights.sum()) < 1e-8:
        return torch.ones(n_img, dtype=E_norm.dtype, device=device)
    return weights


_ICKH_WORKSPACE: dict[tuple[int, int, int, str], dict[str, Any]] = {}
_ICKH_WORKSPACE_LIMIT = 64


def _ickh_block_size() -> int:
    """Herding picks per greedy iteration (GEOPACT_ICKH_BLOCK, default 1).

    1 = exact sequential kernel herding (semantics identical to the original
    implementation; verified 0 pick flips). Larger blocks are an opt-in
    experiment arm: they trade within-block diversity pressure for ~block×
    fewer GPU iterations and are NOT selection-equivalent.
    """
    try:
        return max(1, int(os.environ.get("GEOPACT_ICKH_BLOCK", "1") or 1))
    except Exception:
        return 1


def _ickh_cuda_graph_enabled() -> bool:
    """CUDA-graph the greedy loop (default on): pure launch-overhead removal.

    The loop is bitwise-identical math; the graph just replays the recorded
    kernel sequence in one launch instead of ~7 launches × keep_k iterations.
    """
    return os.environ.get("GEOPACT_ICKH_CUDA_GRAPH", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _ickh_denom_t1() -> bool:
    """Herding repulsion denominator (GEOPACT_ICKH_DENOM, default 't').

    't'  (default): score = mu - col_sum/t — the historical production loop.
    't1': score = mu - col_sum/(t+1) — the exact equal-weight MMD greedy
          increment / standard herding recurrence. Experiment flag; production
          behavior unchanged by default.
    """
    return os.environ.get("GEOPACT_ICKH_DENOM", "t").strip().lower() in {"t1", "tplus1"}


# W57 latency audit: per-frame HES sub-stage wall clocks. Cleared at
# _hes_select entry, merged into the prune frame timings (prune_hes_*_ms)
# by _prune_single_frame, and exported through the keeps-dump side channel.
_HES_STAGE_MS: Dict[str, float] = {}


def _hes_stage_sync_enabled() -> bool:
    """Insert a CUDA sync before each stage stamp (GEOPACT_HES_STAGE_SYNC).

    Off by default: production timing stays free of extra syncs. Enable only
    on instrumented diagnosis runs where per-stage GPU attribution matters.
    """
    return os.environ.get("GEOPACT_HES_STAGE_SYNC", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _hes_stage_record(key: str, t0: float) -> float:
    """Accumulate elapsed ms since ``t0`` under ``key``; return a fresh t0."""
    if _hes_stage_sync_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    _HES_STAGE_MS[key] = _HES_STAGE_MS.get(key, 0.0) + (now - t0) * 1000.0
    return now


def _geo_floor_k(keep_k: int, n_img: int) -> int:
    """Geometric-floor anchor budget k_g (GEOPACT_B_GEO_FLOOR, default off).

    Two-stage budget: k_g = floor(beta*k)
    deterministic raster anchors, herding fills k - k_g. beta from
    GEOPACT_B_GEO_BETA: 'auto' -> clip(1.5*keep_ratio, 0.25, 0.75), or a fixed
    float. Total budget stays exactly keep_k (anchors count toward it).
    """
    if os.environ.get("GEOPACT_B_GEO_FLOOR", "0").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return 0
    raw = os.environ.get("GEOPACT_B_GEO_BETA", "auto").strip().lower()
    if raw == "auto":
        ratio_eff = keep_k / max(1, n_img)
        beta = min(0.75, max(0.25, 1.5 * ratio_eff))
    else:
        try:
            beta = min(1.0, max(0.0, float(raw)))
        except ValueError:
            beta = 0.5
    return min(keep_k - 1, max(0, int(beta * keep_k)))


def _ickh_greedy_loop(
    ws: dict[str, Any],
    keep_k: int,
    block: int,
    anchors: Optional[torch.Tensor] = None,
    denom_t1: bool = False,
) -> None:
    """Greedy herding picks; reads ws['K'/'mu'], writes ws['order_buf'].

    Self-contained (zeroes its own state) so it can be stream-captured.
    ``anchors``: preselected indices (geometric floor) seeded as S_0 — they
    occupy order_buf[:len(anchors)], contribute to the repulsion term and
    count toward keep_k. ``denom_t1``: use 1/(t+1) repulsion (exact MMD
    greedy) instead of the historical 1/t.
    """
    K = ws["K"]
    col_sum = ws["col_sum"]
    scores = ws["scores"]
    mu = ws["mu"]
    selected_mask = ws["selected_mask"]
    order_buf = ws["order_buf"]

    selected_mask.zero_()
    col_sum.zero_()
    picked = 0
    if anchors is not None and int(anchors.numel()) > 0:
        n_anchor = int(anchors.numel())
        selected_mask.index_fill_(0, anchors, True)
        order_buf[:n_anchor].copy_(anchors)
        col_sum.add_(K.index_select(1, anchors).sum(dim=1))
        picked = n_anchor
    while picked < keep_k:
        m = min(block, keep_k - picked)
        if denom_t1:
            alpha = -1.0 / (picked + 1)
        else:
            # mean of already-picked Gram columns == running sum / picked
            alpha = (-1.0 / picked) if picked else 0.0
        torch.add(mu, col_sum, alpha=alpha, out=scores)
        # NB: python scalar, not a 0-d tensor — tensor fill values trigger a
        # host sync (.item()) which is illegal under CUDA graph capture.
        scores.masked_fill_(selected_mask, float("-inf"))
        if m == 1:
            best = scores.argmax().view(1)
        else:
            # Blocked greedy (opt-in): top-m in one shot, no mutual repulsion
            # within the block. block=1 is exact greedy.
            best = torch.topk(scores, m, largest=True, sorted=True).indices
        selected_mask.index_fill_(0, best, True)
        order_buf[picked : picked + m].copy_(best)
        col_sum.add_(K.index_select(1, best).sum(dim=1))
        picked += m


def _ickh_run_greedy(
    ws: dict[str, Any],
    keep_k: int,
    device: torch.device,
    anchors: Optional[torch.Tensor] = None,
    denom_t1: bool = False,
) -> None:
    """Run the greedy loop, via a cached CUDA graph when possible.

    Experiment variants (anchors / denom_t1) run eagerly — only the exact
    production loop is CUDA-graph cached, so default behavior and its graphs
    are byte-identical to before.
    """
    block = _ickh_block_size()
    if anchors is not None or denom_t1:
        _ickh_greedy_loop(ws, keep_k, block, anchors=anchors, denom_t1=denom_t1)
        return
    if device.type != "cuda" or not _ickh_cuda_graph_enabled():
        _ickh_greedy_loop(ws, keep_k, block)
        return
    gkey = ("graph", keep_k, block)
    graph = ws.get(gkey)
    if graph is False:  # capture failed earlier for this key
        _ickh_greedy_loop(ws, keep_k, block)
        return
    if graph is None:
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                _ickh_greedy_loop(ws, keep_k, block)  # warmup (allocator, cudnn)
            torch.cuda.current_stream(device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    _ickh_greedy_loop(ws, keep_k, block)
            except Exception:
                # Unwind a partially-captured graph so the global RNG/allocator
                # are not left in capture mode (breaks later eager RNG ops).
                try:
                    graph.reset()
                except Exception:
                    pass
                raise
            ws[gkey] = graph
        except Exception:
            ws[gkey] = False
            _ickh_greedy_loop(ws, keep_k, block)
            return
    graph.replay()


def _get_token_ickh_workspace(
    n_img: int,
    hidden: int,
    keep_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Reusable GPU buffers for token ICKH (space-for-time across eval steps).

    Holds full Gram ``K`` (n×n) plus column cache ``col_buf`` (n×keep_k) so each
    greedy step uses a vectorized ``mean(dim=1)`` instead of realloc/sync.
    """
    key = (n_img, hidden, keep_k, str(device))
    ws = _ICKH_WORKSPACE.get(key)
    if ws is not None:
        return ws

    while len(_ICKH_WORKSPACE) >= _ICKH_WORKSPACE_LIMIT:
        _ICKH_WORKSPACE.pop(next(iter(_ICKH_WORKSPACE)))

    ws = {
        "E": torch.empty(n_img, hidden, device=device, dtype=torch.float32),
        "K": torch.empty(n_img, n_img, device=device, dtype=torch.float32),
        "col_sum": torch.empty(n_img, device=device, dtype=torch.float32),
        "scores": torch.empty(n_img, device=device, dtype=torch.float32),
        "mu": torch.empty(n_img, device=device, dtype=torch.float32),
        "weights": torch.empty(n_img, device=device, dtype=torch.float32),
        "selected_mask": torch.zeros(n_img, dtype=torch.bool, device=device),
        "order_buf": torch.empty(keep_k, dtype=torch.long, device=device),
        "neg_inf": torch.tensor(float("-inf"), device=device, dtype=torch.float32),
    }
    _ICKH_WORKSPACE[key] = ws
    return ws


def _token_ickh_herding_weights_into(
    E_norm: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    policy: str,
    out: torch.Tensor,
    instr_query: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Write herding weights into preallocated ``out`` (n,) buffer."""
    n_img = int(E_norm.shape[0])
    if _policy_has(policy, "token_ickh_diversity"):
        out.fill_(1.0)
        return out

    rel = out.view(n_img, 1)
    if instr_query is not None:
        q = instr_query.detach().to(device=E_norm.device, dtype=E_norm.dtype)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=-1)
        if int(q.shape[0]) > 1:
            # Token-level query (T, d): reduce with amax per image token.
            # Writing the (n, T) product into the (n, 1) buffer would corrupt
            # it, so reduce first (same semantics as the BAM likelihood).
            rel.copy_((E_norm @ q.transpose(-1, -2)).amax(dim=-1, keepdim=True))
        else:
            torch.matmul(E_norm, q.transpose(-1, -2), out=rel)
        rel.clamp_(min=0.0)
        if float(out.sum()) < 1e-8:
            out.fill_(1.0)
        return out

    mask = image_mask[0] if image_mask.dim() > 1 else image_mask
    mask = mask.to(device=E_norm.device).bool()
    text_embs = inputs_embeds[0][~mask]
    if text_embs.shape[0] == 0:
        out.fill_(1.0)
        return out

    q = F.normalize(text_embs.to(dtype=E_norm.dtype).mean(0, keepdim=True), dim=-1)
    torch.matmul(E_norm, q.transpose(-1, -2), out=rel)
    rel.clamp_(min=0.0)
    if float(out.sum()) < 1e-8:
        out.fill_(1.0)
    return out


def cache_sticky_relevance_query(inner) -> bool:
    """Embed global task instruction and cache ``q`` for token ICKH relevance.

    Called on dialogue full prefill (step 0 / segment reset), not on suffix steps.
    """
    instr = str(getattr(inner, "_geopact_task_instruction_text", "") or "").strip()
    tokenizer = getattr(inner, "_geopact_tokenizer", None)
    get_embed = getattr(inner, "get_input_embeddings", None)
    embed_fn = get_embed() if callable(get_embed) else None
    if not instr or tokenizer is None or embed_fn is None:
        inner._geopact_sticky_relevance_query = None
        inner._geopact_sticky_relevance_query_source = "missing_instruction"
        return False

    device = next(embed_fn.parameters()).device
    ids = tokenizer(instr, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.numel() == 0:
        inner._geopact_sticky_relevance_query = None
        inner._geopact_sticky_relevance_query_source = "empty_instruction"
        return False

    emb = embed_fn(ids)
    q = F.normalize(emb.float().mean(dim=1), dim=-1)
    inner._geopact_sticky_relevance_query = q.detach()
    inner._geopact_sticky_relevance_query_source = "global_instruction"
    return True


def get_sticky_relevance_query(model) -> Optional[torch.Tensor]:
    """Return cached global-instruction query for token ICKH relevance, if any."""
    return getattr(model, "_geopact_sticky_relevance_query", None)


def _embed_relevance_query_text(inner, text: str) -> Optional[torch.Tensor]:
    """Embed a pure instruction text into a normalized relevance query ``q``.

    Cached per text on the model (cleared at episode boundaries); a step
    typically has one distinct text so this is a dict hit after the first use.
    """
    text = str(text or "").strip()
    if not text:
        return None
    cache = getattr(inner, "_geopact_turn_query_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        inner._geopact_turn_query_cache = cache
    q = cache.get(text)
    if q is not None:
        return q
    tokenizer = getattr(inner, "_geopact_tokenizer", None)
    get_embed = getattr(inner, "get_input_embeddings", None)
    embed_fn = get_embed() if callable(get_embed) else None
    if tokenizer is None or embed_fn is None:
        return None
    device = next(embed_fn.parameters()).device
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.numel() == 0:
        return None
    with torch.no_grad():
        emb = embed_fn(ids)
        q = F.normalize(emb.float().mean(dim=1), dim=-1).detach()
    cache[text] = q
    return q


def _turn_query_tokens_enabled() -> bool:
    """v3 round-3 factor: pass the turn instruction as a token-level query
    matrix (T, d) instead of the mean-pooled vector. Only the BAM likelihood
    consumes multi-row queries (amax over rows = match the best-aligned
    instruction token); mean pooling dilutes short imperative instructions.
    Off by default; enable per arm with GEOPACT_TURN_QUERY_TOKENS=1.
    """
    return os.environ.get("GEOPACT_TURN_QUERY_TOKENS", "0").strip().lower() in {
        "1",
        "true",
        "on",
    }


def _turn_query_meanrow_enabled() -> bool:
    """v3 round-6 factor: prepend the mean-pooled row to the token rows.

    amax over token rows is precise under tight budgets but over-concentrates
    under loose ones; the mean row is the global semantic anchor. Stacking
    [mean; tokens] and letting amax pick per image token is the parameter-free
    combination of both readings. Requires GEOPACT_TURN_QUERY_TOKENS=1.
    """
    return os.environ.get("GEOPACT_TURN_QUERY_MEANROW", "0").strip().lower() in {
        "1",
        "true",
        "on",
    }


def _embed_relevance_query_tokens(inner, text: str) -> Optional[torch.Tensor]:
    """Embed an instruction into normalized per-token query rows (T, d)."""
    text = str(text or "").strip()
    if not text:
        return None
    cache = getattr(inner, "_geopact_turn_query_tok_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        inner._geopact_turn_query_tok_cache = cache
    # Key includes the meanrow flag: an in-process flag sweep (ablation
    # harness) must never reuse a qmix matrix built under the other setting.
    ckey = (text, bool(_turn_query_meanrow_enabled()))
    q = cache.get(ckey)
    if q is not None:
        return q
    tokenizer = getattr(inner, "_geopact_tokenizer", None)
    get_embed = getattr(inner, "get_input_embeddings", None)
    embed_fn = get_embed() if callable(get_embed) else None
    if tokenizer is None or embed_fn is None:
        return None
    device = next(embed_fn.parameters()).device
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if ids.numel() == 0:
        return None
    with torch.no_grad():
        emb = embed_fn(ids)
        q = F.normalize(emb.float()[0], dim=-1).detach()  # (T, d)
        if _turn_query_meanrow_enabled():
            mean_row = F.normalize(emb.float().mean(dim=1), dim=-1).detach()  # (1, d)
            q = torch.cat([mean_row, q], dim=0)
    cache[ckey] = q
    return q


def relevance_query_for_frame(
    inner,
    img_path: Optional[str],
) -> tuple[Optional[torch.Tensor], str]:
    """Relevance query for one frame: that turn's pure instruction.

    Turn t's image tokens must be scored against turn t's instruction (the
    exact text the ``rich_turn`` protocol renders into that user turn, no
    template boilerplate). Falls back to the episode-task sticky query when
    the frame is not in the per-turn map. Returns ``(q, source)`` with source
    in {"turn", "turn_tokens", "global_fallback", "none"} for auditing.
    """
    # QK-relevance arm: register layer-0 W_q/W_k references once so the BAM
    # likelihood can score in the model's attention geometry (review §7.3).
    try:
        from lmms_eval.models.geopact_qd_selector import (
            bam_gqc_enabled,
            bam_qk_enabled,
            register_bam_qk_from_model,
        )

        if bam_qk_enabled() or bam_gqc_enabled():
            register_bam_qk_from_model(inner)
    except Exception:
        pass
    # Ablation knob: GEOPACT_TURN_QUERY=0 forces the episode-global instruction
    # query for every frame (rich_turn per-turn queries disabled).
    if os.environ.get("GEOPACT_TURN_QUERY", "1").strip().lower() in {"0", "false", "off"}:
        turn_map = {}
    else:
        turn_map = getattr(inner, "_geopact_turn_instr_by_path", None) or {}
    if img_path and turn_map:
        p = str(img_path)
        if p.startswith("file://"):
            p = p[7:]
        text = turn_map.get(os.path.normpath(os.path.abspath(p)))
        if text:
            if _turn_query_tokens_enabled():
                q = _embed_relevance_query_tokens(inner, text)
                if q is not None:
                    return q, "turn_tokens"
            q = _embed_relevance_query_text(inner, text)
            if q is not None:
                return q, "turn"
    q = get_sticky_relevance_query(inner)
    if q is None:
        cache_sticky_relevance_query(inner)
        q = get_sticky_relevance_query(inner)
    return q, ("global_fallback" if q is not None else "none")


def _geopact_allow_stride_fallback(policy: str) -> bool:
    """Uniform stride fallback is only valid for token baselines."""
    if os.environ.get("GEOPACT_ALLOW_STRIDE_FALLBACK", "").strip() == "1":
        return True
    return _policy_is_token_baseline(policy)


def _geopact_warn_skip_abc(msg: str) -> None:
    """Log and skip Module A/B/C for this frame (full visual token keep, no coord)."""
    _log.warning("GeoPACT skip A/B/C: %s — keeping all visual tokens, no coord inject", msg)


def _geopact_module_a_warn_skip(msg: str, policy: str) -> None:
    """Module A empty-MIU callback: CDCT falls back to B without spatial prior; legacy skips."""
    if _policy_uses_cdct(policy):
        _log.info(
            "GeoPACT CDCT: %s — B uses w_j=r_j (no spatial prior), pruning at keep_k",
            msg,
        )
        return
    _geopact_warn_skip_abc(msg)


def _hes_passthrough_all(n_img: int, reason: str) -> set:
    """Module B off: retain every visual token for this frame."""
    if n_img > 0:
        _geopact_warn_skip_abc(reason)
    return set(range(n_img))


def _geopact_fail(msg: str, policy: str) -> None:
    """Fail loud on GeoPACT paths; random baseline may degrade with a warning."""
    if _geopact_allow_stride_fallback(policy):
        _log.warning("GeoPACT degrade (random/debug): %s", msg)
        return
    raise RuntimeError(
        f"GeoPACT pipeline failure: {msg}. "
        "Module A/B/C did not run — fix cv2 (opencv-python-headless), YOLO weights, "
        "or patches. Do NOT launch fleet until scripts/verify_geopact_abc_stack.py passes."
    )


def _detect_raw_boxes(pil_image, detector: str, policy: str):
    """Run detector only; return (mius, image_np for pentagonal)."""
        from lmms_eval.models.gui_miu_detectors import (
        detect_blob_mius_raw,
        detect_canny_morph_mius_raw,
        detect_fusion_fast_mius_raw,
        detect_ocr_only_mius_raw,
        detect_omniparser_icon_mius_raw,
        detect_omniparser_yolo_blob_fusion_mius_raw,
    )

    if detector == "ocr_only":
        return detect_ocr_only_mius_raw(pil_image)

    if detector in ("omniparser_blob", "omniparser_yolo_blob", "yolo_blob"):
        raw = detect_omniparser_yolo_blob_fusion_mius_raw(pil_image)
        if raw is None:
            _geopact_fail(
                "omniparser_blob/YOLO returned None — check OMNIPARSER_ICON_WEIGHTS and GPU",
                policy,
            )
            return [], None
        return raw

    if detector == "omniparser":
        _use_blob_aux = os.environ.get("GEOPACT_YOLO_BLOB_AUX", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        raw_fn = (
            detect_omniparser_yolo_blob_fusion_mius_raw
            if _use_blob_aux
            else detect_omniparser_icon_mius_raw
        )
        raw = raw_fn(pil_image)
        if raw is None:
            _geopact_fail(
                "omniparser/YOLO returned None — check OMNIPARSER_ICON_WEIGHTS and GPU",
                policy,
            )
            return [], None
        return raw

    if detector == "fusion_fast":
        return detect_fusion_fast_mius_raw(pil_image)
    if detector == "blob":
        return detect_blob_mius_raw(pil_image)
    return detect_canny_morph_mius_raw(pil_image)


def _run_detector_impl(pil_image, detector: str, policy: str, model=None):
    """Module A: detect → OAS → prefilter → pentagonal → info_density."""
    from lmms_eval.models.module_a_pipeline import run_module_a

    if pil_image is None:
        _geopact_fail("no PIL image on model._geopact_last_image", policy)
        return []

    grid_hw = None
    if model is not None:
        grid_hw = getattr(model, "_geopact_module_a_grid_hw", None)

    mius, meta = run_module_a(
        pil_image,
        detector,
        policy,
        grid_hw=grid_hw,
        detect_raw_fn=lambda: _detect_raw_boxes(pil_image, detector, policy),
        apply_attribute_drops_fn=_apply_attribute_drops,
        policy_is_token_baseline_fn=_policy_is_token_baseline,
        warn_skip_abc_fn=lambda msg: _geopact_module_a_warn_skip(msg, policy),
        fail_fn=_geopact_fail,
    )

    if model is not None:
        model._geopact_last_module_a_meta = meta

    if not _policy_is_token_baseline(policy):
        missing_density = sum(
            1 for m in mius if float(m.get("info_density", m.get("info_density_proxy", 0.0)) or 0.0) <= 0.0
        )
        if missing_density == len(mius) and mius:
            _geopact_fail("Module A InfoDensity missing on all MIUs after attach_module_a_info_density_proxy", policy)

    return mius


def _set_module_a_grid_context(model, image_grid_thw, n_img: int) -> None:
    """Set LLM merge grid on model for grid-aware Module A density (A4)."""
    if model is None:
        return
    grid_h, grid_w, grid_matches = _get_grid(image_grid_thw, n_img)
    if grid_matches:
        model._geopact_module_a_grid_hw = (grid_h, grid_w)
    else:
        model._geopact_module_a_grid_hw = None


def release_pending_yolo_gpu(model=None) -> None:
    """Release YOLO GPU weights deferred during Module A (end of eval step)."""
    if model is None:
        return
    if not getattr(model, "_geopact_yolo_release_pending", False):
        return
    from lmms_eval.models.gui_miu_detectors import maybe_release_yolo_gpu

    maybe_release_yolo_gpu(allow_empty_cache=False)
    model._geopact_yolo_release_pending = False


def _export_module_a_cdct_weights(
    model,
    *,
    mius: list,
    policy: str,
    n_frame: int,
    image_grid_thw,
    episode_key: str = "",
    step_index: Optional[int] = None,
) -> None:
    """Append CDCT P_j field summary for sideload weight audit (04j)."""
    if not _module_a_export_weights_enabled() or model is None:
        return
    export: Dict[str, Any] = {
        "n_mius": int(len(mius or [])),
        "n_frame": int(n_frame),
        "cdct_alpha": _cdct_alpha(policy),
        "policy": str(policy),
        "episode_key": str(episode_key or getattr(model, "_geopact_episode_key", "") or ""),
    }
    if step_index is not None:
        export["step_index"] = int(step_index)
    if n_frame <= 0:
        compact = getattr(model, "_geopact_last_compact_stats", None) or {}
        export["n_frame"] = int(compact.get("n_visual_total") or 0)
        n_frame = int(export["n_frame"])
    if mius and image_grid_thw is not None and n_frame > 0:
        grid_h, grid_w, grid_matches = _get_grid(image_grid_thw, n_frame)
        if grid_matches:
            mapped_groups, _, _, mapped_src = _map_mius_to_tokens(
                mius,
                torch.zeros(n_frame, 1),
                grid_h,
                grid_w,
                n_frame,
                need_embs=False,
            )
            miu_token_idxs = _miu_token_idxs_table(len(mius), mapped_groups, mapped_src)
            P = _build_cdct_density_field(
                mius,
                miu_token_idxs,
                n_frame,
                torch.device("cpu"),
            )
            if P is not None and P.numel() > 0:
                p_cpu = P.detach().float().cpu()
                export.update(
                    {
                        "P_mean": float(p_cpu.mean().item()),
                        "P_max": float(p_cpu.max().item()),
                        "P_nonzero": int((p_cpu > 1e-6).sum().item()),
                        "P_top5_tokens": torch.topk(p_cpu, min(5, p_cpu.numel())).indices.tolist(),
                    }
                )
    path = os.environ.get("GEOPACT_MODULE_A_EXPORT_PATH", "").strip()
    if path:
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(export) + "\n")
    elif _module_a_export_weights_enabled():
        _log.warning("GeoPACT: GEOPACT_MODULE_A_EXPORT_WEIGHTS=1 but GEOPACT_MODULE_A_EXPORT_PATH unset")
    model._geopact_module_a_weight_export = export


def _apply_miu_dropout(mius: list, *, image_id: Optional[str], model=None) -> list:
    """E5a detector-degradation knob: drop GEOPACT_MIU_DROPOUT_FRAC of the
    detections deterministically per image (seeded by GEOPACT_MIU_DROPOUT_SEED
    + a stable image fingerprint), AFTER the Module A cache — the cache keeps
    the full detection set, so one dump serves every dropout level. frac=1.0
    empties the prior entirely (the graceful-degradation floor)."""
    raw = os.environ.get("GEOPACT_MIU_DROPOUT_FRAC", "").strip()
    if not raw or not mius:
        return mius
    try:
        frac = float(raw)
    except ValueError:
        return mius
    if frac <= 0.0:
        return mius
    frac = min(1.0, frac)
    import zlib

    seed_env = os.environ.get("GEOPACT_MIU_DROPOUT_SEED", "0").strip() or "0"
    if image_id:
        fp = str(image_id)
        else:
        # Stable fallback fingerprint: bbox layout of the detection set.
        fp = ";".join(
            ",".join(f"{float(v):.4f}" for v in (m.get("bbox") or m.get("bbox_norm") or []))
            for m in mius
        )
    rng = _random.Random(zlib.crc32(f"{seed_env}|{fp}".encode("utf-8")))
    n_drop = int(round(frac * len(mius)))
    if n_drop <= 0:
        return mius
    drop_idx = set(rng.sample(range(len(mius)), min(n_drop, len(mius))))
    kept = [m for i, m in enumerate(mius) if i not in drop_idx]
    if model is not None:
        diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
        diag["miu_dropout_frac"] = frac
        diag["miu_dropout_dropped"] = len(drop_idx)
        diag["miu_dropout_kept"] = len(kept)
        model._geopact_last_prune_diag = diag
    return kept


def _run_detector(
    pil_image,
    detector: str,
    policy: str,
    model=None,
    *,
    defer_yolo_release: bool = True,
    image_id: Optional[str] = None,
):
    """Run Module A detector + attach InfoDensity (GCC). Fail loud on errors."""
    from lmms_eval.models.geopact_module_a_cache import get_or_run_module_a
    from lmms_eval.models.gui_miu_detectors import maybe_release_yolo_gpu

    _assert_module_a_sideload_prefetch_exclusive()
    prefetch = _module_a_prefetch_mode()
    prev_yolo_dev = os.environ.get("MIU_YOLO_DEVICE")
    if prefetch == "cpu":
        os.environ["MIU_YOLO_DEVICE"] = "cpu"

    def _invoke_impl() -> list:
        return _run_detector_impl(pil_image, detector, policy, model=model)

    pool_subprocess = False
    if _module_a_subprocess_enabled() and prefetch is not None:
        from lmms_eval.models.geopact_module_a_worker import module_a_subprocess_pool_enabled

        pool_subprocess = module_a_subprocess_pool_enabled()

    try:
        if pool_subprocess:
            from lmms_eval.models.geopact_module_a_worker import run_module_a_subprocess_pooled
            from lmms_eval.models.geopact_module_a_prefetch import try_consume_module_a_prefetch

            consumed = try_consume_module_a_prefetch(model, expected_image_id=image_id)
            if consumed is not None:
                mius, _cache_meta = consumed
            else:
                mius = run_module_a_subprocess_pooled(
                    pil_image, detector, policy, model=model
                )
                _cache_meta = {
                    "module_a_subprocess": True,
                    "module_a_subprocess_pool": True,
                    "module_a_cache_hit": False,
                }
        elif _module_a_cache_active(model):
            mius, _cache_meta = get_or_run_module_a(
                pil_image,
                detector,
                policy,
                _invoke_impl,
                model=model,
            )
        else:
            mius = _invoke_impl()
            _cache_meta = {"module_a_cache_hit": False, "module_a_cache_bypass": True}

        if model is not None and isinstance(_cache_meta, dict):
            stats = getattr(model, "_geopact_module_a_cache_stats", None) or {}
            module_meta = getattr(model, "_geopact_last_module_a_meta", None) or {}
            _cache_meta = {
                **_cache_meta,
                **stats,
                **{f"module_a_{k}": v for k, v in module_meta.items() if k != "timings_ms"},
            }
            timings = module_meta.get("timings_ms")
            if isinstance(timings, dict):
                _cache_meta.update({f"module_a_{k}": v for k, v in timings.items()})
            model._geopact_last_module_a_cache_meta = _cache_meta
            if prefetch:
                model._geopact_module_a_prefetch = prefetch
        if defer_yolo_release:
            if model is not None:
                model._geopact_yolo_release_pending = True
        else:
            maybe_release_yolo_gpu(allow_empty_cache=False)
            if model is not None:
                model._geopact_yolo_release_pending = False
        return _apply_miu_dropout(mius, image_id=image_id, model=model)
    finally:
        if prefetch == "cpu":
            if prev_yolo_dev is None:
                os.environ.pop("MIU_YOLO_DEVICE", None)
            else:
                os.environ["MIU_YOLO_DEVICE"] = prev_yolo_dev


def _run_module_a_sideload(model, baseline_policy: str) -> None:
    """Post-decode Module A sidecar: metrics only, zero B/C impact."""
    if not _module_a_sideload_enabled() or model is None:
        return
    pil_image = getattr(model, "_geopact_last_image", None)
    if pil_image is None:
        return
    detector = _get_detector(
        baseline_policy,
        str(getattr(model, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
    )
    mode = _module_a_sideload_mode()
    sideload_pol = (
        baseline_policy if mode == "yolo_only" else _module_a_sideload_policy(baseline_policy)
    )
    compact_stats = getattr(model, "_geopact_last_compact_stats", None) or {}
    n_frame = int(compact_stats.get("n_visual_total") or 0)
    if n_frame > 0:
        grid_thw = getattr(model, "_geopact_last_image_grid_thw", None)
        if grid_thw is not None:
            _set_module_a_grid_context(model, grid_thw, n_frame)
    prev_yolo_dev = os.environ.get("MIU_YOLO_DEVICE")
    if not (prev_yolo_dev or "").strip():
        os.environ["MIU_YOLO_DEVICE"] = "cpu"
    try:
        mius = _run_detector(
            pil_image,
            detector,
            sideload_pol,
            model=model,
            defer_yolo_release=True,
        )
    finally:
        if prev_yolo_dev is None:
            os.environ.pop("MIU_YOLO_DEVICE", None)
        else:
            os.environ["MIU_YOLO_DEVICE"] = prev_yolo_dev
    grid_thw = getattr(model, "_geopact_last_image_grid_thw", None)
    _export_module_a_cdct_weights(
        model,
        mius=mius,
        policy=sideload_pol,
        n_frame=n_frame,
        image_grid_thw=grid_thw,
        episode_key=str(getattr(model, "_geopact_episode_key", "") or ""),
        step_index=getattr(model, "_geopact_step_index", None),
    )
    diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
    module_meta = getattr(model, "_geopact_last_module_a_meta", None) or {}
    diag.update(
        {
            "module_a_sideload": True,
            "module_a_sideload_phase": "post_decode",
            "module_a_sideload_mode": mode,
            "module_a_sideload_policy": sideload_pol,
            "prune_selector": _prune_selector_label(baseline_policy),
            "n_mius_detected": int(module_meta.get("n_kept", 0) or 0),
        }
    )
    model._geopact_last_prune_diag = diag


def _queue_module_a_sideload(model, baseline_policy: str) -> None:
    """Queue sideload for post-episode flush so decode/KV paths stay pure."""
    if model is None:
        return
    pil_image = getattr(model, "_geopact_last_image", None)
    if pil_image is None:
        return
    queue = getattr(model, "_geopact_sideload_queue", None)
    if queue is None:
        queue = []
        model._geopact_sideload_queue = queue
    try:
        pil_copy = pil_image.copy()
    except Exception:
        pil_copy = pil_image
    queue.append(
        {
            "baseline_policy": baseline_policy,
            "pil_image": pil_copy,
            "grid_thw": getattr(model, "_geopact_last_image_grid_thw", None),
            "compact_stats": dict(getattr(model, "_geopact_last_compact_stats", None) or {}),
            "episode_key": str(getattr(model, "_geopact_episode_key", "") or ""),
            "step_index": getattr(model, "_geopact_step_index", None),
        }
    )


def _flush_module_a_sideload_queue(model) -> None:
    """Run queued Module A sideload jobs (episode boundary or eval end)."""
    if model is None or not _module_a_sideload_enabled():
        return
    queue = getattr(model, "_geopact_sideload_queue", None) or []
    if not queue:
        return
    model._geopact_sideload_queue = []
    for job in queue:
        model._geopact_last_image = job.get("pil_image")
        if job.get("grid_thw") is not None:
            model._geopact_last_image_grid_thw = job.get("grid_thw")
        if job.get("compact_stats"):
            model._geopact_last_compact_stats = job.get("compact_stats")
        if job.get("episode_key"):
            model._geopact_episode_key = job.get("episode_key")
        if job.get("step_index") is not None:
            model._geopact_step_index = job.get("step_index")
        _run_module_a_sideload(model, str(job.get("baseline_policy") or ""))
    yolo_dev = os.environ.get("MIU_YOLO_DEVICE", "").strip().lower()
    from lmms_eval.models.gui_miu_detectors import _yolo_runs_on_gpu, maybe_release_yolo_gpu

    if _yolo_runs_on_gpu():
        maybe_release_yolo_gpu(allow_empty_cache=True)


def _dispatch_module_a_sideload(model, baseline_policy: str) -> None:
    timing = _module_a_sideload_timing()
    yolo_dev = os.environ.get("MIU_YOLO_DEVICE", "").strip().lower()
    if timing == "post_decode" and yolo_dev and yolo_dev != "cpu":
        msg = (
            "GeoPACT sideload: post_decode + GPU YOLO risks decode pollution; "
            "use GEOPACT_MODULE_A_SIDELOAD_TIMING=eval_end or MIU_YOLO_DEVICE=cpu"
        )
        if os.environ.get("GEOPACT_STRICT_SIDELOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(msg)
        _log.error(msg)
    if timing == "post_decode":
        _run_module_a_sideload(model, baseline_policy)
    else:
        _queue_module_a_sideload(model, baseline_policy)


def _apply_attribute_drops(mius: list, policy: str) -> list:
    """Zero-out dropped pentagonal attributes so they don't contribute to energy."""
    if not mius:
        return mius
    drop_scale    = _policy_has(policy, "drop_scale")
    drop_entropy  = _policy_has(policy, "drop_entropy")
    drop_contrast = _policy_has(policy, "drop_contrast")
    drop_contain  = _policy_has(policy, "drop_containment", "drop_contain")
    drop_grid     = _policy_has(policy, "drop_grid")
    drop_tau      = _policy_has(policy, "drop_tau")

    if not any([drop_scale, drop_entropy, drop_contrast, drop_contain, drop_grid, drop_tau]):
        return mius  # nothing to change

    patched = []
    for m in mius:
        m2 = dict(m)
        energy = float(m2.get("energy", m2.get("pentagonal_energy", 0.0)) or 0.0)
        tau    = float(m2.get("det_score", 1.0) or 1.0)
        if drop_tau:
            tau = 1.0
        # Reverse-engineer components from the total (approximation: equal weight)
        # For a proper ablation, zero one out and re-sum
        # The energy = tau * (S + H + C + R + G)
        # We zero the dropped attribute by reducing the sum proportionally.
        n_active = 5 - sum([drop_scale, drop_entropy, drop_contrast, drop_contain, drop_grid])
        n_active = max(n_active, 1)
        # Scale energy by fraction of remaining attributes
        m2["energy"] = tau * (energy / 5.0) * n_active if tau > 0 else 0.0
        m2["pentagonal_energy"] = m2["energy"]
        patched.append(m2)
    return patched


def _info_density(m: dict) -> float:
    d = float(m.get("info_density", m.get("info_density_proxy", 0.0)) or 0.0)
    if d <= 0.0:
        d = float(m.get("oas_imp", m.get("energy", 0.0)) or 0.0)
    return d


def _bbox_ceil_enabled() -> bool:
    """Module A v2 fix: strict half-open [x1, x2) box-to-token mapping.

    Legacy (GEOPACT_A_BBOX_CEIL=0) used int(x2*W)+1 which grabs one extra
    row/column whenever x2*W lands exactly on a cell boundary. Default on;
    set 0 to reproduce v1 behavior bit-for-bit (parity arms).
    """
    return os.environ.get("GEOPACT_A_BBOX_CEIL", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _token_indices_for_miu_box(
    box,
    grid_h: int,
    grid_w: int,
    n_img: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Vectorized flat token indices covered by a normalized MIU box."""
    x1, y1, x2, y2 = box
    ci0 = max(0, int(y1 * grid_h))
    cj0 = max(0, int(x1 * grid_w))
    if _bbox_ceil_enabled():
        ci1 = min(grid_h, max(ci0 + 1, math.ceil(y2 * grid_h)))
        cj1 = min(grid_w, max(cj0 + 1, math.ceil(x2 * grid_w)))
    else:
        ci1 = min(grid_h, int(y2 * grid_h) + 1)
        cj1 = min(grid_w, int(x2 * grid_w) + 1)
    if ci0 >= ci1 or cj0 >= cj1:
        return None
    cis = torch.arange(ci0, ci1, device=device, dtype=torch.long)
    cjs = torch.arange(cj0, cj1, device=device, dtype=torch.long)
    ci_g, cj_g = torch.meshgrid(cis, cjs, indexing="ij")
    flat = (ci_g * int(grid_w) + cj_g).reshape(-1)
    # Bounds are known on host: only filter when the box can actually overflow
    # n_img (boolean-mask indexing forces a GPU->CPU sync per MIU otherwise).
    max_flat = (ci1 - 1) * int(grid_w) + (cj1 - 1)
    if max_flat >= int(n_img):
        flat = flat[flat < int(n_img)]
        if flat.numel() == 0:
            return None
    return flat


def _miu_token_idxs_table(
    n_mius: int,
    mapped_groups: list,
    mapped_src: list,
    *,
    frame_offset: int = 0,
) -> list:
    """Scatter mapped token groups into a per-source MIU table."""
    if n_mius <= 0:
        return []
    out: list = [[] for _ in range(int(n_mius))]
    if not mapped_groups:
        return out
    off = int(frame_offset)
    for group, src in zip(mapped_groups, mapped_src):
        si = int(src)
        if not (0 <= si < int(n_mius)) or not group:
            continue
        if off:
            out[si] = [off + int(t) for t in group]
        else:
            out[si] = [int(t) for t in group]
    return out


def _scatter_density_max(
    field: torch.Tensor,
    token_chunks: list[torch.Tensor],
    rho_chunks: list[torch.Tensor],
) -> torch.Tensor:
    """Max-pool normalized MIU densities onto token indices via scatter_reduce."""
    if not token_chunks:
        return field
    all_idx = torch.cat(token_chunks)
    all_rho = torch.cat(rho_chunks)
    field.scatter_reduce_(0, all_idx, all_rho, reduce="amax", include_self=True)
    return field


def _policy_uses_cdct(policy: str) -> bool:
    """CDCT-Herding: token ICKH with Module A max-pooled density prior on weights."""
    return _policy_has(policy, "cdct", "token_ickh_miu_density", "miu_density_field")


def _policy_uses_miu_first_backfill(policy: str) -> bool:
    """MIU whole-box ICKH first; token ICKH backfill when union under-fills keep_k."""
    return _policy_has(policy, "miu_first_backfill", "miu_first_token_backfill")


def _miu_backfill_token_policy(policy: str) -> str:
    """Token policy for Phase-B backfill outside selected MIU unions."""
    if _policy_has(policy, "diversity_only"):
        return "token_ickh_diversity"
    if _policy_has(policy, "no_instr_rel", "no_instruction"):
        return "token_ickh_diversity"
    return "token_ickh_relevance"


def _cdct_alpha(policy: str) -> float:
    """Gravitational field strength α (env GEOPACT_CDCT_ALPHA, default 1.0)."""
    p = str(policy or "").lower()
    for part in p.replace("-", "_").split("_"):
        if part.startswith("alpha") and len(part) > 5:
            try:
                return max(0.0, float(part[5:]))
            except ValueError:
                pass
    try:
        return max(0.0, float(os.environ.get("GEOPACT_CDCT_ALPHA", "1.0")))
    except ValueError:
        return 1.0


def _cdct_token_weight_policy(policy: str) -> str:
    """Relevance vs diversity for the token-r leg of CDCT."""
    if _policy_has(policy, "diversity_only"):
        return "token_ickh_diversity"
    return "token_ickh_relevance"


def _build_cdct_density_field(
    mius: list,
    miu_token_idxs: list,
    n_img: int,
    device: torch.device,
) -> torch.Tensor:
    """Max-pool frame-normalized info_density onto token grid (shape n_img)."""
    n = int(n_img)
    field = torch.zeros(n, device=device, dtype=torch.float32)
    if not mius or not miu_token_idxs:
        return field
    densities = torch.tensor(
        [_info_density(m) for m in mius], device=device, dtype=torch.float32
    )
    max_rho = float(densities.max().item()) if densities.numel() else 0.0
    if max_rho <= 0.0:
        return field
    norm = densities / max_rho
    token_chunks: list[torch.Tensor] = []
    rho_chunks: list[torch.Tensor] = []
    for i, idxs in enumerate(miu_token_idxs):
        if not idxs:
            continue
        valid_idx = [int(t) for t in idxs if 0 <= int(t) < n]
        if not valid_idx:
            continue
        idx_t = torch.as_tensor(valid_idx, device=device, dtype=torch.long)
        token_chunks.append(idx_t)
        rho_chunks.append(norm[i].expand(idx_t.numel()))
    return _scatter_density_max(field, token_chunks, rho_chunks)


def _build_cdct_density_field_from_mapping(
    mius: list,
    mapped_groups: list,
    mapped_src: list,
    n_img: int,
    device: torch.device,
) -> torch.Tensor:
    """Max-pool densities using sparse mapped groups (skip full per-MIU table)."""
    n = int(n_img)
    field = torch.zeros(n, device=device, dtype=torch.float32)
    if not mius or not mapped_groups:
        return field
    densities = torch.tensor(
        [_info_density(m) for m in mius], device=device, dtype=torch.float32
    )
    max_rho = float(densities.max().item()) if densities.numel() else 0.0
    if max_rho <= 0.0:
        return field
    norm = densities / max_rho
    token_chunks: list[torch.Tensor] = []
    rho_chunks: list[torch.Tensor] = []
    for group, src in zip(mapped_groups, mapped_src):
        if not group:
            continue
        si = int(src)
        if not (0 <= si < len(mius)):
            continue
        # Groups are host-side lists: validate in python and keep rho on GPU
        # (norm[si].item() would force one device sync per MIU).
        valid_idx = [int(t) for t in group if 0 <= int(t) < n]
        if not valid_idx:
            continue
        idx_t = torch.as_tensor(valid_idx, device=device, dtype=torch.long)
        token_chunks.append(idx_t)
        rho_chunks.append(norm[si].expand(idx_t.numel()))
    return _scatter_density_max(field, token_chunks, rho_chunks)


def _cdct_reverse_anchor_indices(
    selected_tokens: list,
    miu_token_idxs: list,
    mius: list,
) -> list:
    """Map kept tokens to contributing MIU indices (for Module C coord_text)."""
    return _coord_miu_indices_from_pick_order(
        selected_tokens, miu_token_idxs, mius, topk=0
    )


def _coord_miu_indices_from_pick_order(
    picked_order: list,
    miu_token_idxs: list,
    mius: list,
    *,
    topk: int = 5,
) -> list:
    """Map B token pick order to unique MIU indices (first-seen wins, cap at topk).

    W57: precompute the per-token winner once (max info_density, lowest MIU
    index on ties — identical to the previous per-pick linear scan, which was
    O(len(pick_order) x n_mius x len(idxs)) in pure Python and dominated the
    HES bucket at ~1e6 list-membership checks per frame)."""
    selected_miu_indices: list = []
    seen: set = set()
    limit = int(topk) if int(topk) > 0 else 0
    best_for_tok: dict = {}
    best_rho_for_tok: dict = {}
    for i, idxs in enumerate(miu_token_idxs):
        rho = _info_density(mius[i]) if i < len(mius) else 0.0
        for t in idxs:
            t = int(t)
            # Strict > keeps the lowest MIU index on density ties, matching
            # the original scan order (i ascending, replace only on >).
            if t not in best_for_tok or rho > best_rho_for_tok[t]:
                best_for_tok[t] = i
                best_rho_for_tok[t] = rho
    for tok in picked_order:
        best_i = best_for_tok.get(int(tok), -1)
        if best_i >= 0 and best_i not in seen:
            selected_miu_indices.append(best_i)
            seen.add(best_i)
            if limit > 0 and len(selected_miu_indices) >= limit:
                break
    return selected_miu_indices


# ---------------------------------------------------------------------------
# Stage 2: HES (Heterogeneous Evidence Selection)
# ---------------------------------------------------------------------------

def _hes_token_random(
    n_img: int,
    keep_k_tokens: int,
    rng: Optional[_random.Random] = None,
) -> HESResult:
    """Pure token subsampling — shuffle all visual token indices."""
    idxs = list(range(n_img))
    r = rng if rng is not None else _random
    r.shuffle(idxs)
    return HESResult(idxs[:keep_k_tokens])


def _hes_token_kernel_herding(
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    keep_k_tokens: int,
    policy: str,
    instr_query: Optional[torch.Tensor] = None,
    spatial_prior: Optional[torch.Tensor] = None,
    cdct_alpha: float = 0.0,
) -> HESResult:
    """Token-level ICKH: greedy kernel herding without Module A MIU groups.

    Same ``_hes_select`` dialogue-sticky branch as ``uniform``. The greedy loop
    stays entirely on GPU: a running column sum replaces the O(n·k²) repeated
    ``mean(dim=1)``, ``argmax`` results are kept as device tensors (no per-pick
    ``.item()`` sync), and picked indices transfer to CPU once at the end.
    """
    n_img = int(image_embeds.shape[0])
    keep_k = _clamp_token_keep_k(n_img, keep_k_tokens)
    if keep_k >= n_img:
        picked = list(range(n_img))
        return HESResult(picked, picked_token_order=picked)

    _t = time.perf_counter()
    device = image_embeds.device
    hidden = int(image_embeds.shape[-1])
    ws = _get_token_ickh_workspace(n_img, hidden, keep_k, device)
    _t = _hes_stage_record("hes_ws_ms", _t)

    E = ws["E"]
    K = ws["K"]
    weights = ws["weights"]
    mu = ws["mu"]
    order_buf = ws["order_buf"]

    E.copy_(image_embeds.to(dtype=torch.float32))
    F.normalize(E, dim=-1, out=E)
    torch.mm(E, E.transpose(-1, -2), out=K)
    _t = _hes_stage_record("hes_gram_ms", _t)

    _token_ickh_herding_weights_into(
        E, inputs_embeds, image_mask, policy, weights, instr_query=instr_query
    )
    if spatial_prior is not None and cdct_alpha > 0.0:
        prior = spatial_prior.to(device=weights.device, dtype=weights.dtype)
        if prior.shape[0] == weights.shape[0]:
            weights.mul_(1.0 + float(cdct_alpha) * prior)
    torch.mv(K, weights, out=mu)
    mu.div_(weights.sum())
    _t = _hes_stage_record("hes_weights_ms", _t)

    anchors = None
    geo_k = _geo_floor_k(keep_k, n_img)
    if geo_k > 0:
        anchor_idx = sorted(_uniform_keep_indices(n_img, geo_k))
        anchors = torch.as_tensor(anchor_idx, device=device, dtype=torch.long)
    _ickh_run_greedy(ws, keep_k, device, anchors=anchors, denom_t1=_ickh_denom_t1())
    _t = _hes_stage_record("hes_greedy_ms", _t)

    picked_order = [int(x) for x in order_buf[:keep_k].cpu().tolist()]
    _hes_stage_record("hes_order_ms", _t)
    return HESResult(
        sorted(picked_order),
        picked_token_order=picked_order,
    )


def _hes_token_kernel_herding_subset(
    image_embeds: torch.Tensor,
    candidate_indices: list,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    keep_k_tokens: int,
    policy: str,
    instr_query: Optional[torch.Tensor] = None,
    spatial_prior: Optional[torch.Tensor] = None,
    cdct_alpha: float = 0.0,
) -> list[int]:
    """Token ICKH on a candidate subset; returns global vision token indices."""
    if not candidate_indices:
        return []
    cand = sorted(int(i) for i in candidate_indices if 0 <= int(i) < int(image_embeds.shape[0]))
    if not cand:
        return []
    keep_k = min(max(1, int(keep_k_tokens)), len(cand))
    if keep_k >= len(cand):
        return cand
    sub_embeds = image_embeds[cand]
    prior_sub = None
    if spatial_prior is not None:
        n_img = int(image_embeds.shape[0])
        n_prior = int(spatial_prior.shape[0])
        if len(cand) >= n_img and n_prior == n_img:
            prior_sub = spatial_prior
        elif n_prior > 0:
            idx = torch.as_tensor(cand, device=spatial_prior.device, dtype=torch.long)
            prior_sub = spatial_prior.index_select(0, idx)
    result = _hes_token_kernel_herding(
        sub_embeds,
        inputs_embeds,
        image_mask,
        keep_k,
        policy,
        instr_query=instr_query,
        spatial_prior=prior_sub,
        cdct_alpha=cdct_alpha,
    )
    return [cand[int(i)] for i in result]


def _map_mius_to_tokens_cached(
    mapping_cache: Optional[dict],
    mius: list,
    image_embeds: torch.Tensor,
    grid_h: int,
    grid_w: int,
    n_img: int,
):
    """Memoized `_map_mius_to_tokens` for one prune call (P6: rectify runs it twice).

    Keyed on object identities + grid dims; the cache dict lives for a single
    `_prune_single_frame` invocation, so identity keys are stable and the
    cached tuple is bit-identical to a recompute.
    """
    if mapping_cache is None:
        return _map_mius_to_tokens(
            mius, image_embeds, grid_h, grid_w, n_img, need_embs=False
        )
    key = (id(mius), id(image_embeds), int(grid_h), int(grid_w), int(n_img))
    hit = mapping_cache.get(key)
    if hit is None:
        hit = _map_mius_to_tokens(
            mius, image_embeds, grid_h, grid_w, n_img, need_embs=False
        )
        mapping_cache[key] = hit
    return hit


def _miu_labels_from_table(miu_token_idxs: list, n_img: int, device) -> torch.Tensor:
    """Per-token MIU label tensor (-1 = uncovered; first MIU wins on overlap).

    W57: build host-side then transfer once. The previous in-place CUDA
    variant read ``int(labels[ti])`` per covered token — one device sync per
    token, ~10-40ms/frame on dense screens. Same first-MIU-wins result."""
    n = int(n_img)
    lab = [-1] * n
    for mi, toks in enumerate(miu_token_idxs or []):
        for t in toks:
            ti = int(t)
            if 0 <= ti < n and lab[ti] < 0:
                lab[ti] = mi
    return torch.tensor(lab, dtype=torch.long, device=device)


def _qd_instr_query_fallback(
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    instr_query: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Same relevance-query fallback as the herding weight path: mean text row."""
    if instr_query is not None:
        return instr_query
    mask = image_mask[0] if image_mask.dim() > 1 else image_mask
    mask = mask.to(device=inputs_embeds.device).bool()
    text_embs = inputs_embeds[0][~mask]
    if text_embs.shape[0] == 0:
        return None
    return text_embs.float().mean(0)


def _qd_cdct_select(
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    keep_k: int,
    policy: str,
    *,
    spatial_prior: Optional[torch.Tensor],
    miu_token_idxs_full: list,
    grid_h: int,
    grid_w: int,
    instr_query: Optional[torch.Tensor],
    candidate_indices: Optional[list] = None,
) -> list[int]:
    """Module B v2 (QD selector) dispatch shared by the CDCT full/subset paths.

    Returns the pick order as frame-local flat indices. Also records the B->C
    handoff export (per-token quality, MIU labels, per-MIU uncovered quality)
    into the qd frame context for the rewrite caller to consume.
    """
    from lmms_eval.models.geopact_qd_selector import (
        compensability_kappa,
        miu_residual_quality,
        qd_frame_probe_keys,
        qd_params,
        qd_select_tokens,
        record_qd_frame_export,
    )

    n_img = int(image_embeds.shape[0])
    device = image_embeds.device
    labels = _miu_labels_from_table(miu_token_idxs_full, n_img, device)
    query = None
    if not _policy_has(policy, "diversity_only", "no_instr_rel", "no_instruction"):
        query = _qd_instr_query_fallback(inputs_embeds, image_mask, instr_query)

    params = qd_params()
    kappa = None
    if float(params.get("gamma_c", 0.0)) > 0.0:
        probe = qd_frame_probe_keys()
        if probe is not None and int(probe.shape[1]) == n_img:
            try:
                kappa = compensability_kappa(probe, labels.cpu())
                kappa = kappa.to(device)
            except Exception as exc:
                _log.warning("GeoPACT QD compensability failed (non-fatal): %s", exc)
                kappa = None

    if candidate_indices is not None:
        cand_t = torch.tensor(candidate_indices, dtype=torch.long, device=device)
        emb = image_embeds.index_select(0, cand_t)
        prior_sub = None
        if spatial_prior is not None and int(spatial_prior.shape[0]) == n_img:
            prior_sub = spatial_prior.index_select(0, cand_t)
        labels_sub = labels.index_select(0, cand_t)
        kappa_sub = kappa.index_select(0, cand_t) if kappa is not None else None
        order_local, quality_sub, audit = qd_select_tokens(
            emb,
            keep_k,
            instr_query=query,
            spatial_prior=prior_sub,
            miu_labels=labels_sub,
            grid_h=grid_h,
            grid_w=grid_w,
            flat_indices=cand_t,
            kappa_tokens=kappa_sub,
            params=params,
        )
        pick_order = [int(candidate_indices[i]) for i in order_local]
        quality = torch.zeros(n_img, dtype=torch.float32, device=device)
        quality[cand_t] = quality_sub
    else:
        pick_order, quality, audit = qd_select_tokens(
            image_embeds,
            keep_k,
            instr_query=query,
            spatial_prior=spatial_prior,
            miu_labels=labels,
            grid_h=grid_h,
            grid_w=grid_w,
            kappa_tokens=kappa,
            params=params,
        )

    rho = miu_residual_quality(quality, pick_order, labels)
    record_qd_frame_export(
        {
            "rho": rho,
            "miu_labels": labels.detach().cpu(),
            "audit": audit,
        }
    )
    return pick_order


def _hes_token_cdct_herding(
    mius: list,
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    image_grid_thw,
    keep_k_tokens: int,
    policy: str,
    instr_query: Optional[torch.Tensor] = None,
    mapping_cache: Optional[dict] = None,
) -> HESResult:
    """CDCT-Herding: w_j = r_j * (1 + α P_j) then standard token-level ICKH (μ = Kw/Σw).

    Module A info_density is max-pooled onto the vision token grid; selection stays
    at token granularity (homogeneous keep_k steps). Empty MIUs: spatial prior disabled
    (w_j = r_j only) but keep_k pruning still runs — equivalent to token_ickh_relevance.
    """
    n_img = int(image_embeds.shape[0])
    weight_policy = _cdct_token_weight_policy(policy)
    alpha = _cdct_alpha(policy)

    _t = time.perf_counter()
    miu_token_idxs_full: list = [[] for _ in mius]
    mapped_groups: list = []
    mapped_src: list = []
    if mius:
        grid_h, grid_w, grid_matches = _get_grid(image_grid_thw, n_img)
        if grid_matches:
            mapped_groups, _, _, mapped_src = _map_mius_to_tokens_cached(
                mapping_cache, mius, image_embeds, grid_h, grid_w, n_img
            )
            miu_token_idxs_full = _miu_token_idxs_table(len(mius), mapped_groups, mapped_src)
    _t = _hes_stage_record("hes_map_ms", _t)

    spatial_prior = _build_cdct_density_field_from_mapping(
        mius, mapped_groups, mapped_src, n_img, image_embeds.device
    )
    use_prior = spatial_prior if (mius and alpha > 0.0) else None
    _t = _hes_stage_record("hes_prior_ms", _t)

    from lmms_eval.models.geopact_qd_selector import qd_active as _qd_active

    if _qd_active():
        gh, gw, _gm = _get_grid(image_grid_thw, n_img)
        keep_k = min(max(1, int(keep_k_tokens)), n_img)
        pick_order = _qd_cdct_select(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k,
            policy,
            spatial_prior=spatial_prior if mius else None,
            miu_token_idxs_full=miu_token_idxs_full,
            grid_h=gh,
            grid_w=gw,
            instr_query=instr_query,
        )
        selected_tokens = list(pick_order)
    else:
        result = _hes_token_kernel_herding(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k_tokens,
            weight_policy,
            instr_query=instr_query,
            spatial_prior=use_prior,
            cdct_alpha=alpha if use_prior is not None else 0.0,
        )
        selected_tokens = [int(t) for t in result]
        pick_order = list(getattr(result, "picked_token_order", None) or selected_tokens)
    _t = _hes_stage_record("hes_herd_ms", _t)
    topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
    selected_miu = _coord_miu_indices_from_pick_order(
        pick_order,
        miu_token_idxs_full,
        mius,
        topk=topk_env if topk_env > 0 else 0,
    )
    _hes_stage_record("hes_coord_ms", _t)
    return HESResult(selected_tokens, selected_miu, picked_token_order=pick_order)


def _hes_token_cdct_herding_subset(
    mius: list,
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    image_grid_thw,
    candidate_indices: list,
    keep_k_tokens: int,
    policy: str,
    instr_query: Optional[torch.Tensor] = None,
    mapping_cache: Optional[dict] = None,
) -> HESResult:
    """CDCT-Herding on a candidate subset (SSPC in-budget B selection)."""
    n_img = int(image_embeds.shape[0])
    weight_policy = _cdct_token_weight_policy(policy)
    alpha = _cdct_alpha(policy)
    cand = sorted(int(i) for i in candidate_indices if 0 <= int(i) < n_img)
    if not cand:
        return HESResult()
    keep_k = min(max(1, int(keep_k_tokens)), len(cand))

    miu_token_idxs_full: list = [[] for _ in mius]
    mapped_groups: list = []
    mapped_src: list = []
    if mius:
        grid_h, grid_w, grid_matches = _get_grid(image_grid_thw, n_img)
        if grid_matches:
            mapped_groups, _, _, mapped_src = _map_mius_to_tokens_cached(
                mapping_cache, mius, image_embeds, grid_h, grid_w, n_img
            )
            miu_token_idxs_full = _miu_token_idxs_table(len(mius), mapped_groups, mapped_src)

    spatial_prior = _build_cdct_density_field_from_mapping(
        mius, mapped_groups, mapped_src, n_img, image_embeds.device
    )
    use_prior = spatial_prior if (mius and alpha > 0.0) else None

    from lmms_eval.models.geopact_qd_selector import qd_active as _qd_active

    if _qd_active():
        gh, gw, _gm = _get_grid(image_grid_thw, n_img)
        picked = _qd_cdct_select(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k,
            policy,
            spatial_prior=spatial_prior if mius else None,
            miu_token_idxs_full=miu_token_idxs_full,
            grid_h=gh,
            grid_w=gw,
            instr_query=instr_query,
            candidate_indices=cand,
        )
    else:
        picked = _hes_token_kernel_herding_subset(
            image_embeds,
            cand,
            inputs_embeds,
            image_mask,
            keep_k,
            weight_policy,
            instr_query=instr_query,
            spatial_prior=use_prior,
            cdct_alpha=alpha if use_prior is not None else 0.0,
        )
    pick_order = list(picked)
    topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
    selected_miu = _coord_miu_indices_from_pick_order(
        pick_order,
        miu_token_idxs_full,
        mius,
        topk=topk_env if topk_env > 0 else 0,
    )
    return HESResult(picked, selected_miu, picked_token_order=pick_order)


def _hes_token_instr_topk(
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    keep_k_tokens: int,
    instr_query: Optional[torch.Tensor] = None,
) -> HESResult:
    """Pure instruction relevance top-K — no kernel-herding diversity pressure."""
    n_img = int(image_embeds.shape[0])
    keep_k = _clamp_token_keep_k(n_img, keep_k_tokens)
    if keep_k >= n_img:
        return HESResult(range(n_img))

    E = F.normalize(image_embeds.float(), dim=-1)
    if instr_query is not None:
        q = instr_query.detach().to(device=E.device, dtype=E.dtype)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=-1)
        scores = torch.matmul(E, q.transpose(-1, -2)).squeeze(-1).clamp(min=0.0)
    else:
        mask = image_mask[0] if image_mask.dim() > 1 else image_mask
        mask = mask.to(device=E.device).bool()
        text_embs = inputs_embeds[0][~mask]
        if text_embs.shape[0] == 0:
            scores = torch.ones(n_img, dtype=E.dtype, device=E.device)
        else:
            q = F.normalize(text_embs.to(dtype=E.dtype).mean(0, keepdim=True), dim=-1)
            scores = torch.matmul(E, q.transpose(-1, -2)).squeeze(-1).clamp(min=0.0)

    if float(scores.sum()) < 1e-8:
        return HESResult(_uniform_keep_indices(n_img, keep_k))
    idx = torch.topk(scores, keep_k, largest=True, sorted=False).indices
    return HESResult(idx.detach().cpu().tolist())


def _hes_token_div_topk(
    image_embeds: torch.Tensor,
    keep_k_tokens: int,
) -> HESResult:
    """Pure diversity top-K via farthest-point sampling on normalized vision embeddings."""
    n_img = int(image_embeds.shape[0])
    keep_k = _clamp_token_keep_k(n_img, keep_k_tokens)
    if keep_k >= n_img:
        return HESResult(range(n_img))
    if keep_k <= 1:
        return HESResult([0])

    device = image_embeds.device
    E = F.normalize(image_embeds.float(), dim=-1)
    K = E @ E.T
    selected = [0]
    selected_mask = torch.zeros(n_img, dtype=torch.bool, device=device)
    selected_mask[0] = True
    min_dist = torch.full((n_img,), float("inf"), device=device, dtype=E.dtype)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=E.dtype)

    for _ in range(keep_k - 1):
        last = selected[-1]
        dist = 1.0 - K[:, last]
        min_dist = torch.minimum(min_dist, dist)
        min_dist = min_dist.masked_fill(selected_mask, neg_inf)
        next_t = int(min_dist.argmax().item())
        selected.append(next_t)
        selected_mask[next_t] = True

    return HESResult(selected)


def _hes_miu_random(
    miu_token_idxs: list,
    mapped_miu_indices: list,
    n_img: int,
    keep_k_tokens: int,
    rng: Optional[_random.Random] = None,
) -> HESResult:
    """Random MIU selection: shuffle MIU groups, union tokens until budget.

    If MIU union cannot fill ``keep_k_tokens``, remaining slots are filled
    with token-random from the full image (``miu_random_fallback``).
    """
    r = rng if rng is not None else _random
    groups = list(range(len(miu_token_idxs)))
    r.shuffle(groups)
    keep: set[int] = set()
    selected: list[int] = []
    for gi in groups:
        keep.update(miu_token_idxs[gi])
        selected.append(mapped_miu_indices[gi])
        if len(keep) >= keep_k_tokens:
            break
    keep_list = list(keep)[:keep_k_tokens]
    if len(keep_list) < keep_k_tokens:
        remaining = [i for i in range(n_img) if i not in keep]
        r.shuffle(remaining)
        need = keep_k_tokens - len(keep_list)
        keep_list.extend(remaining[:need])
    return HESResult(keep_list, selected)


def _hes_miu_ickh_with_backfill(
    mius: list,
    miu_token_idxs: list,
    miu_energies: list,
    miu_mean_embs: list,
    mapped_miu_indices: list,
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    hidden: int,
    keep_k_tokens: int,
    policy: str,
    instr_query: Optional[torch.Tensor] = None,
) -> HESResult:
    """Phase A: MIU-level ICKH (whole-box union). Phase B: token ICKH backfill."""
    n_img = int(image_embeds.shape[0])
    keep_k = _clamp_token_keep_k(n_img, keep_k_tokens)

    keep, selected = _kernel_herding(
        mius,
        miu_token_idxs,
        miu_energies,
        miu_mean_embs,
        mapped_miu_indices,
        inputs_embeds,
        image_mask,
        hidden,
        keep_k,
        policy,
        instr_query=instr_query,
    )
    keep_set = set(list(keep)[:keep_k])
    backfill_count = 0

    if len(keep_set) < keep_k:
        need = keep_k - len(keep_set)
        remaining = [i for i in range(n_img) if i not in keep_set]
        backfill_policy = _miu_backfill_token_policy(policy)
        extra = _hes_token_kernel_herding_subset(
            image_embeds,
            remaining,
            inputs_embeds,
            image_mask,
            need,
            backfill_policy,
            instr_query=instr_query,
        )
        backfill_count = len(extra)
        keep_set.update(extra)

    keep_final = list(keep_set)[:keep_k]
    return HESResult(keep_final, selected, miu_backfill_tokens=backfill_count)


def _hes_select(
    mius: list,
    image_embeds: torch.Tensor,   # (n_img_tokens, hidden)
    inputs_embeds: torch.Tensor,  # (1, seq_len, hidden) text-only
    image_mask: torch.Tensor,     # (seq_len,) bool
    image_grid_thw,
    keep_k_tokens: int,
    policy: str,
    rng: Optional[_random.Random] = None,
    instr_query: Optional[torch.Tensor] = None,
    candidate_local: Optional[Set[int]] = None,
    mapping_cache: Optional[dict] = None,
    attn_scores: Optional[torch.Tensor] = None,
) -> set:
    """
    Return keep_token_set (set of visual token indices in [0, n_img)) for the
    selected token budget.

    Implements:
      - random selection
      - importance top-K
      - full kernel herding (GCC density + instruction relevance + diversity)
    """
    _HES_STAGE_MS.clear()
    n_img   = image_embeds.shape[0]
    hidden  = inputs_embeds.shape[-1]

    if n_img == 0:
        return HESResult()

    cand_list: Optional[list[int]] = None
    if candidate_local is not None:
        cand_list = sorted(int(i) for i in candidate_local if 0 <= int(i) < n_img)
        if not cand_list:
            return HESResult()

    # -- uniform evenly-spaced selection (deterministic baseline) --
    if _policy_has(policy, "uniform", "uniform_stride"):
        if cand_list is not None:
            step = max(len(cand_list) // max(1, keep_k_tokens), 1)
            return HESResult(cand_list[::step][:keep_k_tokens])
        return HESResult(_uniform_keep_indices(n_img, keep_k_tokens))

    # -- published training-free baselines (DivPrune / DART / CDPruner /
    # VisionZip / VisPruner / PruMerge+ / TRIM / VisionTrim / ZOO-Prune /
    # PruneSID) --
    if _policy_has(
        policy, "divprune", "dart", "cdpruner", "visionzip", "vispruner",
        "prumerge", "trim", "visiontrim", "zooprune", "prunesid",
    ):
        from lmms_eval.models.geopact_pub_baselines import pub_baseline_select

        # visiontrim's LTAM local term needs the frame's spatial layout.
        pub_grid_hw = None
        if _policy_has(policy, "visiontrim"):
            _gh_pub, _gw_pub, _gm_pub = _get_grid(image_grid_thw, n_img)
            if _gm_pub:
                pub_grid_hw = (_gh_pub, _gw_pub)

        # E3 hybrid: mass m_j = 1 + α·P_j from Module A detections (same
        # max-pooled density field as CDCT); ≡ 1 when detections are empty.
        hybrid_mass: Optional[torch.Tensor] = None
        if _pub_hybrid_active(policy) and mius:
            _gh_hy, _gw_hy, _gm_hy = _get_grid(image_grid_thw, n_img)
            if _gm_hy:
                mapped_groups_hy, _, _, mapped_src_hy = _map_mius_to_tokens_cached(
                    mapping_cache, mius, image_embeds, _gh_hy, _gw_hy, n_img
                )
                prior_hy = _build_cdct_density_field_from_mapping(
                    mius, mapped_groups_hy, mapped_src_hy, n_img, image_embeds.device
                )
                hybrid_mass = 1.0 + _pub_hybrid_alpha() * prior_hy
                # One-shot audit line. Without it, "the parity arm ran the
                # hybrid rule" is an inference from the launcher env rather than
                # an observation, and a silently-inactive graft would leave the
                # cell indistinguishable from the unmodified published rule.
                global _PUB_HYBRID_LOGGED
                if not _PUB_HYBRID_LOGGED:
                    _PUB_HYBRID_LOGGED = True
                    _nz = int((prior_hy > 0).sum().item())
                    print(
                        f"[GEOPACT E3-hybrid ACTIVE] policy={policy} "
                        f"alpha={_pub_hybrid_alpha()} n_tokens={prior_hy.numel()} "
                        f"nonzero_prior={_nz} "
                        f"mass_range=[{float(hybrid_mass.min()):.3f},"
                        f"{float(hybrid_mass.max()):.3f}] detections={len(mius)}",
                        flush=True,
                    )

        if cand_list is not None:
            cand_idx = torch.tensor(cand_list, device=image_embeds.device)
            sub = image_embeds[cand_idx]
            sub_scores = (
                attn_scores.reshape(-1)[cand_idx.to(attn_scores.device)]
                if attn_scores is not None
                else None
            )
            picked = pub_baseline_select(
                policy, sub, keep_k_tokens, instr_query=instr_query,
                attn_scores=sub_scores,
                hybrid_mass=(
                    hybrid_mass[cand_idx.to(hybrid_mass.device)]
                    if hybrid_mass is not None
                    else None
                ),
            )
            return HESResult(cand_list[i] for i in picked)
        return HESResult(
            pub_baseline_select(
                policy, image_embeds, keep_k_tokens, instr_query=instr_query,
                attn_scores=attn_scores, grid_hw=pub_grid_hw,
                hybrid_mass=hybrid_mass,
            )
        )

    # -- token-level ICKH (dialogue sticky; no Module A) --
    if _policy_has(policy, "token_ickh_diversity", "token_ickh_relevance"):
        if cand_list is not None:
            picked = _hes_token_kernel_herding_subset(
                image_embeds, cand_list, inputs_embeds, image_mask,
                keep_k_tokens, policy, instr_query=instr_query,
            )
            return HESResult(picked, picked_token_order=picked)
        return _hes_token_kernel_herding(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k_tokens,
            policy,
            instr_query=instr_query,
        )

    if _policy_has(policy, "token_instr_topk"):
        return _hes_token_instr_topk(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k_tokens,
            instr_query=instr_query,
        )

    if _policy_has(policy, "token_div_topk"):
        return _hes_token_div_topk(image_embeds, keep_k_tokens)

    # -- CDCT: token ICKH + Module A density field (before legacy MIU union path) --
    if _policy_uses_cdct(policy):
        if cand_list is not None:
            return _hes_token_cdct_herding_subset(
                mius=mius,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=image_grid_thw,
                candidate_indices=cand_list,
                keep_k_tokens=keep_k_tokens,
                policy=policy,
                instr_query=instr_query,
                mapping_cache=mapping_cache,
            )
        return _hes_token_cdct_herding(
            mius=mius,
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=image_grid_thw,
            keep_k_tokens=keep_k_tokens,
            policy=policy,
            instr_query=instr_query,
            mapping_cache=mapping_cache,
        )

    # -- pure token random (baseline: skip Module A) --
    if _policy_has(policy, "random") and _policy_is_token_baseline(policy):
        if cand_list is not None:
            r = rng if rng is not None else _random
            shuffled = list(cand_list)
            r.shuffle(shuffled)
            return HESResult(shuffled[:keep_k_tokens])
        return _hes_token_random(n_img, keep_k_tokens, rng)

    # -- oas_random_* with 0 MIUs: fall back to token-random (same budget) --
    if _policy_has(policy, "random") and not mius:
        if cand_list is not None:
            r = rng if rng is not None else _random
            shuffled = list(cand_list)
            r.shuffle(shuffled)
            return HESResult(shuffled[:keep_k_tokens])
        return _hes_token_random(n_img, keep_k_tokens, rng)

    # -- MIU-first + token backfill: 0 MIUs → pure token ICKH (same as CDCT fallback) --
    if _policy_uses_miu_first_backfill(policy) and not mius:
        return _hes_token_kernel_herding(
            image_embeds,
            inputs_embeds,
            image_mask,
            keep_k_tokens,
            _miu_backfill_token_policy(policy),
            instr_query=instr_query,
        )

    # -- No MIUs: skip Module B prune (keep all tokens) --
    if not mius:
        return HESResult(_hes_passthrough_all(n_img, "HES: 0 MIUs"))

    # -- Grid dimensions for spatial MIU→token mapping --
    grid_h, grid_w, grid_matches = _get_grid(image_grid_thw, n_img)

    if not grid_matches:
        _geopact_fail(
            f"HES stride fallback blocked: mius={len(mius)} grid_matches={grid_matches} "
            f"grid=({grid_h},{grid_w}) n_img={n_img}",
            policy,
        )
        if _geopact_allow_stride_fallback(policy):
        stride = max(1, n_img // keep_k_tokens)
            return HESResult(range(0, n_img, stride)[:keep_k_tokens])
        raise RuntimeError("GeoPACT HES: grid mismatch")

    # -- Map MIU boxes → token index sets + projected embeddings --
    miu_token_idxs, miu_energies, miu_mean_embs, mapped_miu_indices = _map_mius_to_tokens(
        mius, image_embeds, grid_h, grid_w, n_img
    )

    if not miu_mean_embs:
        _geopact_fail(
            f"HES stride fallback blocked: no MIU→token mappings (n_mius={len(mius)})",
            policy,
        )
        if _geopact_allow_stride_fallback(policy):
        stride = max(1, n_img // keep_k_tokens)
            return HESResult(range(0, n_img, stride)[:keep_k_tokens])
        raise RuntimeError("GeoPACT HES: MIU token mapping empty")

    # -- MIU-level random (oas_random_*): shuffle MIUs, accumulate tokens --
    if _policy_has(policy, "random") and not _policy_is_token_baseline(policy):
        return _hes_miu_random(
            miu_token_idxs,
            mapped_miu_indices,
            n_img,
            keep_k_tokens,
            rng,
        )

    # -- importance-only top-K (no diversity pressure) --
    if _policy_has(policy, "imp_topk"):
        density = [_info_density(mius[i]) / max(len(miu_token_idxs[i]), 1)
                   for i in range(len(miu_token_idxs))]
        ranked = sorted(range(len(density)), key=lambda i: density[i], reverse=True)
        keep = set()
        selected = []
        for ri in ranked:
            keep.update(miu_token_idxs[ri])
            selected.append(mapped_miu_indices[ri])
            if len(keep) >= keep_k_tokens:
                break
        return HESResult(list(keep)[:keep_k_tokens], selected)

    # -- MIU-first ICKH + token relevance backfill --
    if _policy_uses_miu_first_backfill(policy):
        return _hes_miu_ickh_with_backfill(
            mius,
            miu_token_idxs,
            miu_energies,
            miu_mean_embs,
            mapped_miu_indices,
            image_embeds,
            inputs_embeds,
            image_mask,
            hidden,
            keep_k_tokens,
            policy,
            instr_query=instr_query,
        )

    # -- Full kernel herding --
    keep, selected = _kernel_herding(
        mius, miu_token_idxs, miu_energies, miu_mean_embs,
        mapped_miu_indices, inputs_embeds, image_mask, hidden, keep_k_tokens, policy,
        instr_query=instr_query,
    )
    return HESResult(keep, selected)


def _spatial_merge_size() -> int:
    """Qwen3-VL merges spatial_merge_size×spatial_merge_size vision cells per LLM token."""
    return max(1, int(os.environ.get("GEOPACT_SPATIAL_MERGE", "2")))


def _llm_tokens_per_image(t: int, h: int, w: int, merge: int) -> int:
    merge_sq = merge * merge
    return max(0, int(t) * int(h) * int(w) // merge_sq)


def _get_grid(image_grid_thw, n_img: int):
    """Return (grid_h, grid_w, grid_matches) in **LLM token** grid coordinates."""
    if image_grid_thw is None or image_grid_thw.numel() < 3:
        return 1, 1, False
    merge = _spatial_merge_size()
    thw = image_grid_thw[-1].tolist()
    raw_h, raw_w = int(thw[1]), int(thw[2])
    grid_h, grid_w = raw_h // merge, raw_w // merge
    if grid_h * grid_w == n_img:
        return grid_h, grid_w, True
    # Fallback: raw grid (older checkpoints / tests)
    if raw_h * raw_w == n_img:
        return raw_h, raw_w, True
    return grid_h, grid_w, False


def _flat_image_token_ranges(image_grid_thw, n_img: int) -> list[tuple[int, int]]:
    """Map each image in image_grid_thw to [start, end) indices in flat image_embeds."""
    if image_grid_thw is None or getattr(image_grid_thw, "numel", lambda: 0)() < 3:
        return [(0, n_img)]
    merge = _spatial_merge_size()
    ranges: list[tuple[int, int]] = []
    offset = 0
    for i in range(int(image_grid_thw.shape[0])):
        t, h, w = (int(v) for v in image_grid_thw[i].tolist())
        count = _llm_tokens_per_image(t, h, w, merge)
        if count == 0:
            continue
        end = min(offset + count, n_img)
        if end > offset:
            ranges.append((offset, end))
        offset = end
        if offset >= n_img:
            break
    if not ranges:
        return [(0, n_img)]
    if offset < n_img:
        if len(ranges) == 1:
            # Single-image prompt with coarse grid metadata.
            s0, _ = ranges[0]
            ranges[0] = (s0, n_img)
        else:
            raise RuntimeError(
                f"GeoPACT grid/token mismatch: image_grid_thw rows sum to {offset} "
                f"LLM tokens (merge={merge}) but n_img={n_img}. "
                f"Fix spatial_merge_size or image_grid_thw."
            )
    return ranges


def _scope_is_current_image(scope: str) -> bool:
    return scope in {"image", "current_image", "current", "last_image"}


def _scope_is_sticky(scope: str) -> bool:
    return scope in {"image_sticky", "sticky", "sticky_image"}


def scope_requires_episode_order(scope: str) -> bool:
    """True when eval must process steps in episode order (sticky mask cache)."""
    return _scope_is_sticky(str(scope or "").lower())


def _coord_metadata_only() -> bool:
    """Align with Qwen2.5-Omni OMNIGUI_COORD_METADATA_ONLY: skip coord token splice."""
    if _coord_ephemeral_enabled():
        return False
    for key in ("GEOPACT_COORD_METADATA_ONLY", "OMNIGUI_COORD_METADATA_ONLY"):
        val = os.environ.get(key)
        if val is not None:
            return val.strip().lower() not in {"0", "false", "off", "no"}
    return True


def _coord_ephemeral_enabled() -> bool:
    """Per-step coord text splice; stripped from dialogue KV before next step."""
    return os.environ.get("GEOPACT_COORD_EPHEMERAL", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _coord_policy_active(policy: str) -> bool:
    if _policy_has(policy, "no_coord"):
        return False
    return _coord_ephemeral_enabled() or _policy_has(
        policy, "coord_center", "coord_tail", "coord_ephemeral", "coord_sdb"
    )


def _coord_mode() -> str:
    return (os.environ.get("GEOPACT_COORD_MODE") or "text").strip().lower()


def _should_build_coord_text(policy: str) -> bool:
    """Whether Module C coord string should be computed for this policy."""
    if _policy_has(policy, "no_coord"):
        return False
    if _coord_mode() == "gaec":
        return True
    if _coord_ephemeral_enabled():
        return True
    if _coord_metadata_only():
        # Metadata-only without ephemeral splice: building coord text has no effect.
        return False
    return _policy_has(
        policy, "coord_center", "coord_tail", "coord_ephemeral", "coord_sdb"
    )


def _norm_image_path(path: str) -> str:
    if not path or not isinstance(path, str):
        return ""
    return os.path.normpath(os.path.abspath(path))


def _sticky_instr_key_enabled() -> bool:
    """F3 (v3): include the frame's turn-instruction hash in the sticky key.

    The legacy key (episode, image_path) reuses a cached mask even when the
    same screenshot re-appears under a different instruction. Since the
    per-path turn instruction is stable for normal history frames, the key is
    unchanged there (reuse still works); it only differs when the same path is
    re-shown with a new instruction — exactly the staleness case. Default on;
    GEOPACT_STICKY_INSTR_KEY=0 restores the legacy key.
    """
    return os.environ.get("GEOPACT_STICKY_INSTR_KEY", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }


def _geopact_state_owner(model):
    """Object that owns the ``_geopact_*`` side state for this carrier.

    On the Qwen3 composite, ``model.model`` is the VL model and holds the state.
    On the Qwen2.5-VL carrier (UI-TARS-1.5-7B) the FCG itself owns it, and
    ``model.model`` is the bare text LLM, so the naive descent lands on a
    different object. ``geopact_session_kv._inner_model`` and
    ``geopact_vl_prepare._inner_vl_model`` already special-case this; the NAP
    order store must agree with them or stage 2 looks up a store the entry pass
    never wrote to, and every history frame falls back with
    ``rectify_reason=nap_missing_order``.
    """
    if str(getattr(getattr(model, "config", None), "model_type", "")) == "qwen2_5_vl":
        return model
    return getattr(model, "model", model)


def _sticky_mask_key(episode_key: str, image_path: str, model=None) -> Tuple[str, ...]:
    norm_path = _norm_image_path(image_path)
    if model is not None and _sticky_instr_key_enabled():
        turn_map = getattr(model, "_geopact_turn_instr_by_path", None) or {}
        text = turn_map.get(norm_path)
        if text:
            digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]
            return (episode_key, norm_path, digest)
    return (episode_key, norm_path)


def _sticky_rng(episode_key: str, image_path: str) -> _random.Random:
    key = "\0".join(_sticky_mask_key(episode_key, image_path))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return _random.Random(seed)


def _get_sticky_store(model) -> dict:
    store = getattr(model, "_geopact_sticky_masks", None)
    if store is None:
        store = {}
        model._geopact_sticky_masks = store
    return store


def compute_sticky_dialogue_prune_keep_set(
    model,
    *,
    token_ranges,
    image_embeds: torch.Tensor,
    image_grid_thw,
    ratio: float,
    policy: str,
    scope: Optional[str] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    image_mask_1d: Optional[torch.Tensor] = None,
) -> set:
    """Per-frame image_sticky keep for dialogue sticky prune (token + oas MIU).

    Mirrors hook sticky branch: history frames reuse ``sticky_store``;
    current frame runs HES selector (token baseline or Module A + MIU HES)
    and writes the mask for later steps.
    """
    active_scope = str(
        scope or getattr(model, "_geopact_prune_scope", "") or ""
    ).lower()
    if not _scope_is_sticky(active_scope):
        raise RuntimeError(
            "GeoPACT dialogue prune requires visual_prune_scope=image_sticky, "
            f"got {active_scope!r}"
        )
    if not _is_dialogue_sticky_embed_prune(policy):
        raise RuntimeError(
            "GeoPACT dialogue sticky prune requires uniform/random/token_ickh/"
            f"token_*_topk or oas_* policy, got {policy!r}"
        )
    _assert_module_a_sideload_prefetch_exclusive()

    try:
        from lmms_eval.models.geopact_sspc import reset_sspc_step_state, sspc_enabled

        if sspc_enabled():
            reset_sspc_step_state(model)
    except Exception:
        pass

    episode_key = str(getattr(model, "_geopact_episode_key", "") or "")
    image_paths = getattr(model, "_geopact_image_paths", None) or []
    sticky_store = _get_sticky_store(model)
    single_frame_dialogue = _dialogue_prefetch_single_frame(model)
    prune_target = _prune_target(model)
    from lmms_eval.models.geopact_nap import (
        build_nap_order,
        nap_current_prune_active,
        nap_order_kind,
        ratio_current,
        store_nap_order,
    )

    nap_entry_prune = nap_current_prune_active(model)
    keep_token_set: set = set()
    n_frames = len(token_ranges)
    hidden = int(image_embeds.shape[-1])

    if inputs_embeds is None:
        select_embeds = torch.zeros(
            1, 1, hidden, device=image_embeds.device, dtype=image_embeds.dtype
        )
    else:
        select_embeds = inputs_embeds

    if image_mask_1d is None:
        select_mask = torch.zeros(1, dtype=torch.bool, device=image_embeds.device)
    else:
        select_mask = image_mask_1d
        if select_mask.dim() > 1:
            select_mask = select_mask[0]
        select_mask = select_mask.bool()

    use_instr_query = _needs_sticky_relevance_query_cache(policy)

    def _run_selector(range_idx, start, end, img_path, is_current):
        n_frame = end - start
        # NAP stage 1: the current screenshot enters at the wider r_cur budget;
        # history frames (segment-reset full prefill) keep the r_hist budget.
        frame_ratio = (
            ratio_current() if (nap_entry_prune and is_current) else float(ratio)
        )
        keep_k = max(1, int(round(n_frame * float(frame_ratio))))
        frame_embeds = image_embeds[start:end]
        frame_grid = _frame_grid(image_grid_thw, range_idx, n_frames)
        # Per-turn relevance: frame from turn t is scored against turn t's
        # own instruction (not the episode-global text).
        instr_query: Optional[torch.Tensor] = None
        if use_instr_query:
            instr_query, instr_source = relevance_query_for_frame(model, img_path)
            diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
            diag["instr_source"] = instr_source
            model._geopact_last_prune_diag = diag
        rng = (
            _sticky_rng(episode_key, img_path)
            if _policy_has(policy, "random")
            else None
        )
        pick_order_local: list = []
        if _policy_is_token_baseline(policy) and not _pub_hybrid_active(policy):
            local_keep = _hes_select(
                mius=[],
                image_embeds=frame_embeds,
                inputs_embeds=select_embeds,
                image_mask=select_mask,
                image_grid_thw=frame_grid,
                keep_k_tokens=keep_k,
                policy=policy,
                rng=rng,
                instr_query=instr_query,
                attn_scores=_vz_attn_scores_for_range(
                    model, policy, start, end, int(image_embeds.shape[0])
                ),
            )
            pick_order_local = [
                int(t) for t in (getattr(local_keep, "picked_token_order", None) or [])
            ]
            frame_keep = {start + int(idx) for idx in local_keep}
            _attach_miu_prune_diag(
                model,
                n_frame=n_frame,
                keep_k=keep_k,
                keep_local={int(idx) for idx in local_keep},
                mius=[],
                mapped_token_groups=[list(range(n_frame))],
                policy=policy,
            )
            local_keep = [int(idx) for idx in local_keep]
        else:
            pil_image = getattr(model, "_geopact_last_image", None)
            detector = _get_detector(
                policy,
                str(getattr(model, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
            )
            frame_keep, _, _, frame_coord_text = _prune_single_frame(
                start=start,
                end=end,
                image_embeds=image_embeds,
                inputs_embeds=select_embeds,
                image_mask=select_mask,
                image_grid_thw=frame_grid,
                ratio=float(frame_ratio),
                policy=policy,
                detector=detector,
                pil_image=pil_image,
                rng=rng,
                model=model,
                instr_query=instr_query,
                image_id=str(img_path or "") or None,
            )
            if is_current and frame_coord_text and _coord_ephemeral_enabled():
                model._geopact_ephemeral_coord_text = frame_coord_text
            local_keep = [int(idx) - start for idx in frame_keep]
            pick_order_local = [
                int(t)
                for t in (getattr(model, "_geopact_last_pick_order_local", None) or [])
            ]
        if nap_entry_prune and is_current and episode_key and img_path:
            # NAP: persist the ordered pick sequence so stage 2 contracts the
            # same selection instead of re-running A/B on the S_cur subset.
            kind = nap_order_kind(policy)
            order = build_nap_order(
                keep_local=local_keep,
                pick_order=pick_order_local,
                kind=kind,
                rng=_sticky_rng(episode_key, img_path) if kind == "random" else None,
            )
            if kind == "greedy":
                from lmms_eval.models.geopact_nap import (
                    anticipate_nap_coverage,
                    nap_anticipate_enabled,
                )

                if nap_anticipate_enabled():
                    # Anticipation: entry must leave no stage-2 stratum
                    # without an S_cur member (S_hist ⊂ S_cur is physical).
                    k_hist_ac = max(1, int(round(n_frame * float(ratio))))
                    order, ac_audit = anticipate_nap_coverage(
                        order, n_frame, k_hist_ac
                    )
                    if ac_audit.get("nap_ac_swaps", 0) > 0:
                        new_local = sorted(set(order))
                        frame_keep = {start + int(i) for i in new_local}
                        local_keep = new_local
                    diag_ac = dict(
                        getattr(model, "_geopact_last_prune_diag", {}) or {}
                    )
                    diag_ac.update(ac_audit)
                    model._geopact_last_prune_diag = diag_ac
            crs_nested_applied = False
            if kind == "greedy":
                # Module C v5 multi-step (CRS-N): certify the ordering's
                # k_hist prefix once, at entry — the only injection point a
                # frame ever gets; history contraction is then a pure prefix
                # cut and the certificate survives verbatim (order record
                # pins prefix semantics in contract_nap_order).
                from lmms_eval.models.geopact_crs import (
                    crs_nested_enabled,
                    crs_nested_order,
                )

                if crs_nested_enabled():
                    k_hist_crs = max(1, int(round(n_frame * float(ratio))))
                    _crsn_t0 = time.perf_counter()
                    try:
                        order, keep_changed, crs_audit = crs_nested_order(
                            frame_embeds,
                            [int(t) for t in order],
                            n_full=n_frame,
                            k_hist=k_hist_crs,
                        )
                        _pt_crsn = getattr(
                            model, "_geopact_prune_frame_timings", None
                        )
                        if isinstance(_pt_crsn, dict):
                            _pt_crsn["prune_crsn_ms"] = round(
                                (time.perf_counter() - _crsn_t0) * 1000.0, 2
                            )
                        crs_nested_applied = "crs_nested_g" in crs_audit
                        if keep_changed:
                            new_local = sorted(set(order))
                            frame_keep = {start + int(i) for i in new_local}
                            local_keep = new_local
                        diag_crs = dict(
                            getattr(model, "_geopact_last_prune_diag", {}) or {}
                        )
                        diag_crs.update(crs_audit)
                        model._geopact_last_prune_diag = diag_crs
                        if crs_nested_applied and not getattr(
                            model, "_geopact_crs_nested_logged", False
                        ):
                            # once per process: the prune diag never reaches
                            # the run log, so make first engagement auditable
                            # (loguru — the std-logging channel is dropped)
                            from loguru import logger as _eval_log

                            _eval_log.info(
                                "GeoPACT CRS-N engaged: {}", crs_audit
                            )
                            model._geopact_crs_nested_logged = True
                    except Exception as exc:
                        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in (
                            "1", "true", "yes", "on",
                        ):
                            raise
                        _log.warning("GeoPACT CRS-N entry ordering failed: %s", exc)
            order_extra = None
            if crs_nested_applied:
                order_extra = {"crs_nested": True}
            if kind == "greedy":
                # C-Absorb (W-A1): the same ordering also fixes the absorption
                # hierarchy — build the rank-monotone parent tree once at entry
                # and serialize it with the order record for stage-2 resolve.
                try:
                    from lmms_eval.models.geopact_c_absorb import (
                        absorb_assign_mode,
                        build_absorb_parents,
                        c_absorb_enabled,
                    )

                    if c_absorb_enabled():
                        a_gh, a_gw, _a_ok = _get_grid(frame_grid, n_frame)
                        if absorb_assign_mode() in ("exact", "randexact"):
                            # Exact block assignment is tree-free: donors are
                            # raster-block-mapped at contraction time, so the
                            # rank-monotone tree would never be read. Skip its
                            # O(k) build (~20ms/frame) on the entry critical
                            # path; keep grid dims for the stage-2 planner.
                            order_extra = {
                                **(order_extra or {}),
                                "grid_h": int(a_gh),
                                "grid_w": int(a_gw),
                            }
                        else:
                            parents = build_absorb_parents(
                                order,
                                embeds=frame_embeds,
                                grid_h=a_gh,
                                grid_w=a_gw,
                            )
                            order_extra = {
                                **(order_extra or {}),
                                "absorb_parents": tuple(int(p) for p in parents),
                                "grid_h": int(a_gh),
                                "grid_w": int(a_gw),
                            }
                except Exception as exc:
                    if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in (
                        "1", "true", "yes", "on",
                    ):
                        raise
                    _log.warning("GeoPACT C-Absorb tree build failed: %s", exc)
            store_nap_order(
                model,
                _sticky_mask_key(episode_key, img_path, model=model),
                order=order,
                kind=kind,
                n_full=n_frame,
                extra=order_extra,
            )
            if _independent_probe_enabled():
                # W43-C probe: rerun the same selector at the history budget
                # with no prefix constraint (same features, mass, and
                # instruction). Diag side-state is snapshotted so the probe
                # never leaks into the frame's own audit record.
                try:
                    k_hist_probe = max(1, int(round(n_frame * float(ratio))))
                    diag_snap = getattr(model, "_geopact_last_prune_diag", None)
                    pick_snap = getattr(
                        model, "_geopact_last_pick_order_local", None
                    )
                    if _policy_is_token_baseline(policy) and not _pub_hybrid_active(
                        policy
                    ):
                        indep_local_keep = _hes_select(
                            mius=[],
                            image_embeds=frame_embeds,
                            inputs_embeds=select_embeds,
                            image_mask=select_mask,
                            image_grid_thw=frame_grid,
                            keep_k_tokens=k_hist_probe,
                            policy=policy,
                            rng=(
                                _sticky_rng(episode_key, img_path)
                                if _policy_has(policy, "random")
                                else None
                            ),
                            instr_query=instr_query,
                            attn_scores=_vz_attn_scores_for_range(
                                model, policy, start, end,
                                int(image_embeds.shape[0]),
                            ),
                        )
                        indep_keep = sorted(int(i) for i in indep_local_keep)
                    else:
                        probe_detector = _get_detector(
                            policy,
                            str(
                                getattr(model, "_geopact_miu_detector", "fusion_fast")
                                or "fusion_fast"
                            ),
                        )
                        indep_frame_keep, _, _, _ = _prune_single_frame(
                            start=start,
                            end=end,
                            image_embeds=image_embeds,
                            inputs_embeds=select_embeds,
                            image_mask=select_mask,
                            image_grid_thw=frame_grid,
                            ratio=float(ratio),
                            policy=policy,
                            detector=probe_detector,
                            pil_image=getattr(model, "_geopact_last_image", None),
                            rng=(
                                _sticky_rng(episode_key, img_path)
                                if _policy_has(policy, "random")
                                else None
                            ),
                            model=model,
                            instr_query=instr_query,
                            image_id=str(img_path or "") or None,
                        )
                        indep_keep = sorted(
                            int(idx) - start for idx in indep_frame_keep
                        )
                    model._geopact_last_prune_diag = diag_snap
                    model._geopact_last_pick_order_local = pick_snap
                    model._geopact_indep_probe = {
                        "key": _sticky_mask_key(episode_key, img_path, model=model),
                        "k_c": len(local_keep),
                        "k_h": int(k_hist_probe),
                        "keep_cur": sorted(int(i) for i in local_keep),
                        "keep_hist_nested": sorted(
                            int(x) for x in list(order)[:k_hist_probe]
                        ),
                        "keep_hist_independent": indep_keep,
                    }
                except Exception as exc:
                    if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in (
                        "1", "true", "yes", "on",
                    ):
                        raise
                    _log.warning(
                        "GeoPACT independent probe failed: %s", exc
                    )
        return frame_keep, local_keep

    for range_idx, (start, end) in enumerate(token_ranges):
        n_frame = end - start
        is_current = range_idx == n_frames - 1
        img_path = image_paths[range_idx] if range_idx < len(image_paths) else ""
        cache_key = _sticky_mask_key(episode_key, img_path, model=model)
        cached = sticky_store.get(cache_key)
        # Prefetch single-frame short-circuit only applies to the current-target
        # protocol: history frames reuse their frozen mask (or stay full).
        if single_frame_dialogue and not is_current:
            if cached is not None:
                keep_token_set.update(start + int(idx) for idx in cached)
            else:
                keep_token_set.update(range(start, end))
            continue

        action = _sticky_frame_action(
            is_current=is_current,
            prune_target=prune_target,
            has_cached_mask=cached is not None,
            two_stage=nap_entry_prune,
        )

        if action == "prune":
            frame_keep, local_keep = _run_selector(range_idx, start, end, img_path, is_current)
            keep_token_set.update(frame_keep)
            # Under the current target the active frame is re-selected every step
            # (prefetch owns cross-step state); do not persist its mask. History
            # frames must persist so later steps reuse the same 40%.
            persist = episode_key and img_path and (
                (not is_current) or not _dialogue_prefetch_single_frame(model)
            )
            if persist:
                sticky_store[cache_key] = frozenset(int(idx) for idx in local_keep)
            _trace_sticky(episode_key, img_path, is_current, local_keep, False, range_idx)
            _dump_frame_prune_diag(
                model,
                episode_key=episode_key,
                img_path=img_path,
                frame_index=range_idx,
                is_current=is_current,
                action="prune",
                n_frame=n_frame,
                local_keep=local_keep,
                frame_grid=_frame_grid(image_grid_thw, range_idx, n_frames),
            )
        elif action == "reuse":
            keep_token_set.update(start + int(idx) for idx in cached)
            _trace_sticky(
                episode_key, img_path, is_current, [int(idx) for idx in cached], True, range_idx
            )
            _dump_frame_prune_diag(
                model,
                episode_key=episode_key,
                img_path=img_path,
                frame_index=range_idx,
                is_current=is_current,
                action="reuse",
                n_frame=n_frame,
                local_keep=[int(idx) for idx in cached],
                frame_grid=_frame_grid(image_grid_thw, range_idx, n_frames),
            )
        else:  # full
            keep_token_set.update(range(start, end))
            _trace_sticky(
                episode_key, img_path, is_current, list(range(n_frame)), False, range_idx
            )

    return keep_token_set


def _rectify_nap_prefix_contract(
    model,
    *,
    cap: Dict[str, Any],
    ratio: float,
    episode_key: str,
    n_vis: int,
    full_embeds: torch.Tensor,
    full_deepstack,
    grid_row,
    stats: Dict[str, Any],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
    List[int],
    Dict[str, Any],
]:
    """NAP stage 2: shrink the S_cur rows to the stored ordering's r_hist prefix.

    ``cap['visual_embeds']`` rows are the frame's S_cur subset in sequence
    (ascending full-frame flat index) order. The returned ``local_keep`` is in
    row coordinates (matches the cached ids / KV span); the full-frame keep
    indices and virtual frame length needed for MRoPE subsetting are exposed
    via ``stats['nap_full_local_keep']`` / ``stats['nap_n_full']``.
    """
    from lmms_eval.models.geopact_nap import (
        contract_nap_order,
        lookup_nap_order,
        nap_full_token_count,
    )

    inner = _geopact_state_owner(model)
    img_path = str(cap.get("img_path") or "")
    sticky_key = (
        _sticky_mask_key(episode_key, img_path, model=inner)
        if episode_key and img_path
        else None
    )
    rec = lookup_nap_order(inner, sticky_key)
    if rec is None and os.environ.get("GEOPACT_NAP_KEY_DEBUG", "0").strip() in (
        "1", "true", "yes", "on",
    ):
        _store_dbg = getattr(inner, "_geopact_nap_orders", None) or {}
        _log.warning(
            "NAP key miss: lookup=%r n_vis=%d store_size=%d stored_keys=%r",
            sticky_key,
            n_vis,
            len(_store_dbg),
            list(_store_dbg)[:3],
        )
    n_full = 0
    scur_full: Optional[List[int]] = None
    if rec is not None and len(rec.get("order") or ()) == n_vis:
        scur_full = sorted(int(x) for x in rec["order"])
        n_full = int(rec.get("n_full") or 0)
    else:
        # Order record lost (e.g. process restart mid-episode): the sticky
        # mask still holds S_cur in full-frame coordinates; contract by even
        # stride — deterministic, honest, and never re-runs A/B on subset rows.
        sticky_store = getattr(inner, "_geopact_sticky_masks", None) or {}
        cached_scur = sticky_store.get(sticky_key) if sticky_key else None
        if cached_scur and len(cached_scur) == n_vis:
            scur_full = sorted(int(x) for x in cached_scur)
            rec = {"order": tuple(scur_full), "kind": "stride"}
        stats["rectify_nap_order_fallback"] = (
            "sticky_stride" if scur_full is not None else "missing"
        )
    if scur_full is None or rec is None:
        stats["rectify_reason"] = "nap_missing_order"
        return None, None, [], stats

    if n_full <= 0:
        n_full = nap_full_token_count(grid_row, _spatial_merge_size()) or n_vis
    k_hist = max(1, int(round(n_full * float(ratio))))
    if k_hist >= n_vis:
        stats["rectify_ok"] = True
        stats["rectify_reason"] = "ratio_noop"
        stats["rectify_kept"] = int(n_vis)
        stats["rectify_dropped"] = 0
        stats["rectify_selector"] = "nap_prefix_noop"
        all_keep = list(range(n_vis))
        return (
            full_embeds,
            list(full_deepstack) if full_deepstack else None,
            all_keep,
            stats,
        )

    # COC (C v4) override: sliding_rewrite_prepare_dialogue_kv computes the
    # certificate-optimal keep set from the live cache value rows and stashes
    # it just before calling rectify. Consume-once; validated against this
    # frame's sticky key, budget and S_cur before it can replace the
    # rank-strata contraction.
    contract_mode_used: Optional[str] = None
    keep_full: Optional[List[int]] = None
    coc_over = getattr(inner, "_geopact_coc_keep_override", None)
    if coc_over and sticky_key and coc_over.get("key") == sticky_key:
        inner._geopact_coc_keep_override = None
        cand = [int(x) for x in (coc_over.get("keep_full") or [])]
        scur_set = set(scur_full)
        if len(cand) == k_hist and all(f in scur_set for f in cand):
            keep_full = sorted(cand)
            contract_mode_used = "coc"
    if keep_full is None:
        # W43-B/C2 replay override: an externally prepared history keep for
        # this exact frame replaces the prefix contraction. Tokens outside
        # S_cur cannot be served by a deletion-only cache; they are dropped
        # and counted rather than silently accepted.
        ext_over = _keep_override_lookup(inner, episode_key, img_path)
        if ext_over is not None:
            cand = [int(x) for x in (ext_over.get("keep") or [])]
            scur_set = set(scur_full)
            valid = sorted(x for x in cand if x in scur_set)
            if valid:
                keep_full = valid
                contract_mode_used = "keep_override"
                stats["override_arm"] = str(ext_over.get("arm") or "")
                stats["override_out_of_scur"] = int(len(cand) - len(valid))
                if len(valid) != k_hist:
                    stats["override_budget_mismatch"] = int(len(valid) - k_hist)
                try:
                    from loguru import logger as _eval_log

                    _eval_log.info(
                        "GeoPACT keep-override applied: ep={} img={} arm={} "
                        "k={} out_of_scur={}",
                        episode_key,
                        os.path.basename(img_path),
                        ext_over.get("arm"),
                        len(valid),
                        len(cand) - len(valid),
                    )
                except Exception:
                    pass
            else:
                stats["override_arm"] = str(ext_over.get("arm") or "")
                stats["override_unusable"] = True
    if keep_full is None:
        keep_full = sorted(contract_nap_order(rec, k_hist))
    row_of = {f: i for i, f in enumerate(scur_full)}
    rows = sorted(row_of[f] for f in keep_full if f in row_of)
    keep_full = [scur_full[i] for i in rows]
    if not rows:
        stats["rectify_reason"] = "nap_empty_contract"
        return None, None, [], stats

    embed_fn = inner.get_input_embeddings()
    try:
        device = embed_fn.weight.device
    except Exception:
        device = full_embeds.device
    dtype = full_embeds.dtype
    row_idx = torch.tensor(rows, dtype=torch.long, device=full_embeds.device)
    kept_visual = full_embeds.index_select(0, row_idx).to(device=device, dtype=dtype)
    rectified_deepstack: Optional[List[torch.Tensor]] = None
    if full_deepstack:
        rectified_deepstack = [
            layer.to(device=device, dtype=dtype).index_select(
                0, row_idx.to(layer.device)
            )
            for layer in full_deepstack
        ]

    if sticky_key:
        # Sticky mask advances to S_hist (full-frame coords) so segment-reset
        # full prefills reuse the contracted selection.
        sticky_store = getattr(inner, "_geopact_sticky_masks", None)
        if sticky_store is None:
            sticky_store = {}
            inner._geopact_sticky_masks = sticky_store
        sticky_store[sticky_key] = frozenset(keep_full)

    from lmms_eval.models.geopact_nap import nap_contract_mode

    stats["rectify_ok"] = True
    stats["rectify_reason"] = "ok"
    stats["rectify_selector"] = "nap_prefix"
    stats["rectify_kept"] = len(rows)
    stats["rectify_dropped"] = int(n_vis) - len(rows)
    stats["rectify_zafd"] = False
    stats["nap_n_full"] = int(n_full)
    stats["nap_k_hist"] = int(k_hist)
    stats["nap_order_kind"] = str(rec.get("kind") or "greedy")
    stats["nap_contract_mode"] = contract_mode_used or nap_contract_mode()
    stats["nap_full_local_keep"] = [int(x) for x in keep_full]
    stats["nap_scur_sorted"] = [int(x) for x in scur_full]
    return kept_visual, rectified_deepstack, rows, stats


def rectify_history_frame_from_full(
    model,
    *,
    cap: Dict[str, Any],
    ratio: float,
    policy: str,
    episode_key: str = "",
    n_frame: Optional[int] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[List[torch.Tensor]],
    List[int],
    Dict[str, Any],
]:
    """Vision-side A+B mask selection + optional C ZAFD on a full-frame capture.

    Used by sliding-window KV rewrite: ABC runs on ``cap`` tensors before any LM
    forward; returns rectified compact visual rows ready to splice into suffix.
    """
    stats: Dict[str, Any] = {"rectify_ok": False}
    full_embeds = cap.get("visual_embeds")
    full_deepstack = cap.get("deepstack") or []
    grid_row = cap.get("grid_row")
    if full_embeds is None or grid_row is None:
        stats["rectify_reason"] = "missing_capture_tensors"
        return None, None, [], stats

    n_vis = int(n_frame if n_frame is not None else int(full_embeds.shape[0]))
    if n_vis <= 0 or int(full_embeds.shape[0]) != n_vis:
        stats["rectify_reason"] = "capture_span_mismatch"
        return None, None, [], stats

    # NAP stage 2 (two-stage): the frame entered pruned, so ``cap`` holds the
    # S_cur subset rows and re-running A/B here would both misalign with the
    # full-frame grid and re-decide a different selection. Contract the stored
    # stage-1 ordering to its first k_hist picks instead — the prefix is the
    # greedy solution at the smaller budget (matching pursuit), free of cost.
    from lmms_eval.models.geopact_nap import nap_current_prune_active

    if nap_current_prune_active(model):
        return _rectify_nap_prefix_contract(
            model,
            cap=cap,
            ratio=float(ratio),
            episode_key=episode_key,
            n_vis=n_vis,
            full_embeds=full_embeds,
            full_deepstack=full_deepstack,
            grid_row=grid_row,
            stats=stats,
        )

    keep_k = max(1, int(round(n_vis * float(ratio))))
    if keep_k >= n_vis:
        stats["rectify_ok"] = True
        stats["rectify_reason"] = "ratio_noop"
        stats["rectify_kept"] = int(n_vis)
        stats["rectify_dropped"] = 0
        stats["rectify_selector"] = "ratio_noop_skip"
        all_keep = list(range(n_vis))
        return full_embeds, list(full_deepstack) if full_deepstack else None, all_keep, stats

    # Module C-Lite (BOMR): reserve R relay slots inside the same budget B —
    # B-R original survivors + R balanced-OT relays snapped onto deleted grid
    # positions (spec docs/geopact_module_c_qasmr.md §7). Budget-neutral.
    lite_relays = 0
    try:
        from lmms_eval.models.geopact_spatial_relay import relay_budget, relay_lite_active

        if relay_lite_active():
            lite_relays = max(0, min(int(relay_budget()), keep_k - 1))
    except Exception as exc:
        _log.warning("GeoPACT C-Lite flag check failed: %s", exc)
        lite_relays = 0
    select_k = keep_k - lite_relays
    select_ratio = float(select_k) / float(n_vis) if lite_relays > 0 else float(ratio)

    inner = getattr(model, "model", model)
    embed_fn = inner.get_input_embeddings()
    try:
        device = embed_fn.weight.device
    except Exception:
        device = full_embeds.device
    dtype = full_embeds.dtype

    frame_embeds = full_embeds.to(device=device, dtype=dtype)
    frame_grid = grid_row.to(device=device)

    img_path = str(cap.get("img_path") or "")
    sticky_store = getattr(inner, "_geopact_sticky_masks", None)
    if sticky_store is None:
        sticky_store = {}
        inner._geopact_sticky_masks = sticky_store
    sticky_key = (
        _sticky_mask_key(episode_key, img_path, model=inner)
        if episode_key and img_path
        else ""
    )
    cached_keep = sticky_store.get(sticky_key) if sticky_key else None
    local_keep: Optional[List[int]] = None
    rectified_deepstack_from_cache: Optional[List[torch.Tensor]] = None
    kept_visual_from_cache: Optional[torch.Tensor] = None
    if lite_relays > 0:
        # Cached keep sets / compact embeds contain original rows only; a
        # C-Lite pass must rebuild relay rows, so bypass both reuse paths.
        cached_keep = None
        stats["rectify_lite_reuse_bypassed"] = True
    if cached_keep:
        local_keep = sorted(int(x) for x in cached_keep)
        if local_keep and len(local_keep) < n_vis:
            stats["rectify_selector"] = "sticky_cache_reuse"
        else:
            local_keep = None

    if local_keep is None and img_path and lite_relays == 0:
        try:
            from lmms_eval.models.geopact_vision_cache import (
                _local_keep_seq_from_entry,
                get_compact_frame_cache,
            )

            compact_entry = get_compact_frame_cache(inner, img_path)
            if compact_entry is not None:
                seq = list(_local_keep_seq_from_entry(compact_entry))
                if seq and len(seq) < n_vis:
                    local_keep = [int(x) for x in seq]
                    stats["rectify_selector"] = "compact_cache_reuse"
                    kept_visual_from_cache = compact_entry["compact_embeds"].to(
                        device=device, dtype=dtype
                    )
                    compact_ds = compact_entry.get("compact_deepstack") or []
                    if compact_ds:
                        rectified_deepstack_from_cache = [
                            layer.to(device=device, dtype=dtype) for layer in compact_ds
                        ]
        except Exception as exc:
            _log.warning("GeoPACT rectify compact cache reuse failed: %s", exc)

    if local_keep is None:
        instr_query = None
        if _needs_sticky_relevance_query_cache(policy):
            # The captured frame belongs to the previous turn: prune it with
            # that turn's own instruction (per-turn map), not the current
            # step's global task text.
            instr_query, instr_source = relevance_query_for_frame(inner, img_path)
            stats["rectify_instr_source"] = instr_source
        select_embeds = torch.zeros(1, 1, int(frame_embeds.shape[-1]), device=device, dtype=dtype)
        select_mask = torch.zeros(1, dtype=torch.bool, device=device)

        if _policy_is_token_baseline(policy):
            stats["rectify_selector"] = "token_ickh_baseline"
            local_keep = _hes_select(
                mius=[],
                image_embeds=frame_embeds,
                inputs_embeds=select_embeds,
                image_mask=select_mask,
                image_grid_thw=frame_grid,
                keep_k_tokens=select_k,
                policy=policy,
                rng=None,
                instr_query=instr_query,
            )
            local_keep = sorted(int(x) for x in local_keep)
        else:
            stats["rectify_selector"] = "frame_prune"
            detector = _get_detector(
                policy,
                str(getattr(inner, "_geopact_miu_detector", "fusion_fast") or "fusion_fast"),
            )
            # Module B v2 (QD): give the selector the frame's probe-layer keys
            # so the compensability discount (master spec §2.5) can price
            # deletions by how well QASMR can relay them.
            try:
                from lmms_eval.models.geopact_qd_selector import (
                    qd_active as _qd_active_flag,
                    set_qd_frame_context,
                )

                if _qd_active_flag():
                    probe = None
                    snap = cap.get("layer_kv")
                    if snap is not None:
                        k_all = snap.get("k")
                        if k_all is not None and int(k_all.shape[2]) == n_vis:
                            n_layers = int(k_all.shape[0])
                            probe = k_all[min(n_layers // 2, n_layers - 1)]
                    set_qd_frame_context(probe_keys=probe)
            except Exception as exc:
                _log.warning("GeoPACT QD frame context install failed: %s", exc)
            pil_image = cap.get("pil_image")
            handle = getattr(inner, "_geopact_module_a_async", None)
            if isinstance(handle, dict):
                stats["rectify_prefetch_handle"] = str(handle.get("image_id") or "")
                stats["rectify_prefetch_match"] = bool(
                    img_path and str(handle.get("image_id") or "") == img_path
                )
                fut = handle.get("future")
                stats["rectify_prefetch_done"] = bool(fut is not None and fut.done())
            else:
                stats["rectify_prefetch_handle"] = "none"
            prune_t0 = time.perf_counter()
            frame_keep, _, _, _ = _prune_single_frame(
                start=0,
                end=n_vis,
                image_embeds=frame_embeds,
                inputs_embeds=select_embeds,
                image_mask=select_mask,
                image_grid_thw=frame_grid,
                ratio=float(select_ratio),
                policy=policy,
                detector=detector,
                pil_image=pil_image,
                rng=None,
                model=inner,
                instr_query=instr_query,
                image_id=img_path or None,
            )
            stats["rectify_prune_frame_ms"] = round(
                (time.perf_counter() - prune_t0) * 1000.0, 2
            )
            for pk, pv in (getattr(inner, "_geopact_prune_frame_timings", None) or {}).items():
                stats[f"rectify_{pk}"] = pv
            meta = getattr(inner, "_geopact_last_module_a_cache_meta", None) or {}
            for key in (
                "module_a_prefetch_async_wall_ms",
                "module_a_prefetch_async_join_ms",
                "module_a_prefetch_async_overlap_ms",
                "module_a_prefetch_async",
                "module_a_detect_ms",
                "module_a_total_ms",
            ):
                if key in meta:
                    stats[f"rectify_{key}"] = meta[key]
            local_keep = sorted(int(x) for x in frame_keep)
            # Module B v2 -> C handoff: per-MIU uncovered quality + labels for
            # residual-weighted relay budget allocation (master spec §2.6).
            try:
                from lmms_eval.models.geopact_qd_selector import (
                    clear_qd_frame_context,
                    pop_qd_frame_export,
                )

                qd_export = pop_qd_frame_export()
                clear_qd_frame_context()
                if qd_export is not None:
                    from lmms_eval.models.geopact_qd_selector import qd_selector_kind

                    cap["qd_export"] = qd_export
                    stats["rectify_qd_selector"] = qd_selector_kind()
                    stats["rectify_qd_rho_buckets"] = len(qd_export.get("rho") or {})
                    audit_exp = qd_export.get("audit") or {}
                    for _k in ("bam_uniform_a", "bam_has_mass", "bam_soft_a"):
                        if _k in audit_exp:
                            stats[f"rectify_{_k}"] = bool(audit_exp[_k])
            except Exception as exc:
                _log.warning("GeoPACT QD export handoff failed: %s", exc)

    if not local_keep or len(local_keep) >= n_vis:
        stats["rectify_reason"] = "selector_noop"
        return None, None, [], stats

    if sticky_key:
        sticky_store[sticky_key] = frozenset(local_keep)

    if kept_visual_from_cache is not None:
        stats["rectify_ok"] = True
        stats["rectify_reason"] = "ok"
        stats["rectify_kept"] = len(local_keep)
        stats["rectify_dropped"] = n_vis - len(local_keep)
        stats["rectify_zafd"] = False
        return kept_visual_from_cache, rectified_deepstack_from_cache, local_keep, stats

    zafd_t0 = time.perf_counter()
    try:
        from lmms_eval.models.geopact_sspc import (
            apply_sspc_embedding_zafd_for_model,
            sspc_enabled,
            sspc_zafd_enabled,
        )

        _relay_kv_on = False
        try:
            from lmms_eval.models.geopact_spatial_relay import relay_kv_mode_active

            _relay_kv_on = relay_kv_mode_active()
        except Exception:
            _relay_kv_on = False
        if lite_relays > 0 and sspc_enabled() and sspc_zafd_enabled():
            # C-Lite and ZAFD both redistribute deleted mass; running both
            # would double-count donors. Relays win when both flags are set.
            stats["rectify_zafd_skipped_for_lite"] = True
        elif _relay_kv_on and sspc_enabled() and sspc_zafd_enabled():
            # QASMR KV relays and ZAFD both compensate the deleted set; ZAFD
            # is the falsification baseline and must not stack with C v2.
            stats["rectify_zafd_skipped_for_relay"] = True
        elif sspc_enabled() and sspc_zafd_enabled():
            zafd_ctx = {
                "frame_start": 0,
                "n_frame": int(n_vis),
                "frame_local": True,
            }
            frame_embeds = apply_sspc_embedding_zafd_for_model(
                frame_embeds,
                inner,
                frame_grid,
                zafd_ctx,
                reject_grid_mismatch=True,
            )
            zafd_audit = getattr(inner, "_geopact_sspc_audit", None) or {}
            stats["rectify_zafd"] = bool(zafd_audit.get("sspc_zafd"))
            stats["rectify_zafd_enabled"] = True
            if not bool(zafd_audit.get("sspc_grid_matched", True)):
                stats["rectify_reason"] = "zafd_grid_mismatch"
                return None, None, [], stats
    except Exception as exc:
        stats["rectify_reason"] = f"error:zafd:{exc}"
        return None, None, [], stats
    stats["rectify_zafd_ms"] = round((time.perf_counter() - zafd_t0) * 1000.0, 2)

    # Module C-Lite (BOMR): build R balanced-OT relays over the deleted rows
    # and splice them in as extra kept slots at deleted grid positions. The
    # merged keep list stays within the original budget (B-R survivors + R
    # relays); downstream MRoPE/deepstack handling is unchanged because relay
    # slots are ordinary frame-local indices with custom row contents.
    lite_relay_rows: Optional[dict] = None
    if lite_relays > 0:
        survivors = sorted(set(int(i) for i in local_keep))
        deleted = [i for i in range(n_vis) if i not in set(survivors)]
        # Selectors may return more than select_k survivors (geo floors);
        # relays adapt so the merged keep never exceeds the budget B.
        n_lite = min(int(lite_relays), keep_k - len(survivors), len(deleted))
        if n_lite < 1:
            stats["rectify_lite_relays"] = 0
            stats["rectify_lite_skip_reason"] = "no_budget_left"
        else:
            try:
                from lmms_eval.models.geopact_spatial_relay import (
                    _grid_positions,
                    build_lite_relays,
                )

                gh, gw, _grid_ok = _get_grid(frame_grid, n_vis)
                pos_all = _grid_positions(n_vis, gh, gw)
                del_ds = None
                if full_deepstack:
                    del_idx_t = torch.tensor(deleted, dtype=torch.long)
                    del_ds = [
                        layer.to(device=device, dtype=dtype).index_select(
                            0, del_idx_t.to(device)
                        )
                        for layer in full_deepstack
                    ]
                relay_flat, relay_embeds, relay_ds, _pi, lite_audit = build_lite_relays(
                    frame_embeds.index_select(
                        0, torch.tensor(deleted, dtype=torch.long, device=device)
                    ),
                    pos_all[deleted],
                    deleted,
                    set(survivors),
                    n_relays=n_lite,
                    deepstack=del_ds,
                )
                local_keep = sorted(set(survivors) | set(int(i) for i in relay_flat))
                if len(local_keep) > keep_k:
                    raise RuntimeError(
                        f"C-Lite budget violated: kept={len(local_keep)} budget={keep_k}"
                    )
                lite_relay_rows = {
                    "flat": [int(i) for i in relay_flat],
                    "embeds": relay_embeds.to(device=device, dtype=dtype),
                    "deepstack": relay_ds,
                }
                stats["rectify_lite_relays"] = len(relay_flat)
                for k_a, v_a in lite_audit.items():
                    stats[f"rectify_{k_a}"] = v_a
            except Exception as exc:
                stats["rectify_reason"] = f"error:lite_relay:{exc}"
                return None, None, [], stats

    gather_t0 = time.perf_counter()
    keep_idx = torch.tensor(local_keep, dtype=torch.long, device=device)
    kept_visual = frame_embeds.index_select(0, keep_idx)
    rectified_deepstack = None
    if full_deepstack:
        rectified_deepstack = []
        for layer in full_deepstack:
            layer = layer.to(device=device, dtype=dtype)
            rectified_deepstack.append(layer.index_select(0, keep_idx.to(layer.device)))
    if lite_relay_rows is not None:
        kept_visual = kept_visual.clone()
        pos_map = {int(v): i for i, v in enumerate(local_keep)}
        rows = torch.tensor(
            [pos_map[f] for f in lite_relay_rows["flat"]],
            dtype=torch.long,
            device=device,
        )
        kept_visual[rows] = lite_relay_rows["embeds"]
        relay_ds = lite_relay_rows["deepstack"]
        if rectified_deepstack is not None and relay_ds:
            for li in range(min(len(rectified_deepstack), len(relay_ds))):
                layer = rectified_deepstack[li].clone()
                layer[rows.to(layer.device)] = relay_ds[li].to(
                    device=layer.device, dtype=layer.dtype
                )
                rectified_deepstack[li] = layer
    stats["rectify_gather_ms"] = round((time.perf_counter() - gather_t0) * 1000.0, 2)

    stats["rectify_ok"] = True
    stats["rectify_reason"] = "ok"
    stats["rectify_kept"] = len(local_keep)
    stats["rectify_dropped"] = n_vis - len(local_keep)
    if "rectify_zafd" not in stats:
        stats["rectify_zafd"] = False
    if "rectify_selector" not in stats:
        stats["rectify_selector"] = "unknown"
    return kept_visual, rectified_deepstack, local_keep, stats


def _prune_diag_dump_path() -> str:
    return os.environ.get("GEOPACT_PRUNE_DIAG_DUMP", "").strip()


def _independent_probe_enabled() -> bool:
    """W43-C probe: additionally run the election at k_h with no prefix
    constraint and dump the result beside the nested keep."""
    return os.environ.get("GEOPACT_INDEPENDENT_PROBE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _keep_override_map(model) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
    """W43-B/C2 replay: lazily load GEOPACT_KEEP_OVERRIDE (a JSONL of
    per-frame history keep sets) into a {(episode_id, image_basename): row}
    map. Rows are keyed by the same episode normalization the w43 prep
    scripts use (text after the first '-' of episode_key) plus the frame's
    screenshot basename, which matches the sticky-mask granularity."""
    path = os.environ.get("GEOPACT_KEEP_OVERRIDE", "").strip()
    if not path:
        return None
    if (
        getattr(model, "_geopact_keep_override_path", None) == path
        and getattr(model, "_geopact_keep_override_map", None) is not None
    ):
        return model._geopact_keep_override_map
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ep = str(row.get("episode_id") or row.get("episode_key") or "")
                if "-" in ep and row.get("episode_key"):
                    ep = ep.split("-", 1)[-1]
                img = os.path.basename(str(row.get("image_path") or ""))
                if not ep or not img:
                    continue
                mapping[(ep, img)] = row
        _log.info(
            "GeoPACT keep-override loaded: %d frame rows from %s",
            len(mapping), path,
        )
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            raise
        _log.warning("GeoPACT keep-override load failed (%s): %s", path, exc)
        mapping = {}
    model._geopact_keep_override_path = path
    model._geopact_keep_override_map = mapping
    return mapping


def _keep_override_lookup(model, episode_key: str, img_path: str):
    mapping = _keep_override_map(model)
    if not mapping:
        return None
    ep = str(episode_key or "").split("-", 1)[-1]
    img = os.path.basename(str(img_path or ""))
    return mapping.get((ep, img))


def _dump_frame_prune_diag(
    model,
    *,
    episode_key: str,
    img_path: str,
    frame_index: int,
    is_current: bool,
    action: str,
    n_frame: int,
    local_keep,
    frame_grid,
) -> None:
    """Append one JSONL record per frame decision when GEOPACT_PRUNE_DIAG_DUMP
    is set: keep mask (frame-local indices) + grid dims for mask visualisation,
    plus the crs_*/nap_* audit fields for CRS-N mechanism statistics. The
    regular prune diag never reaches disk, so paper-figure material must be
    collected through this side channel on a small instrumented run."""
    path = _prune_diag_dump_path()
    if not path:
        return
    try:
        grid_h, grid_w, grid_ok = _get_grid(frame_grid, n_frame)
        diag = getattr(model, "_geopact_last_prune_diag", None) or {}
        # W57 latency audit: surface the per-frame prune stage split
        # (prune_detector/hes/map/total) and the Module A overlap/join wall
        # numbers on freshly pruned frames. Reuse rows would carry stale
        # timings from the last pruned frame, so they are skipped.
        if str(action) == "prune":
            _pt = getattr(model, "_geopact_prune_frame_timings", None) or {}
            _am = getattr(model, "_geopact_last_module_a_cache_meta", None) or {}
            diag = {
                **diag,
                **{k: v for k, v in _pt.items() if isinstance(v, (int, float))},
                **{
                    k: v
                    for k, v in _am.items()
                    if isinstance(k, str)
                    and k.startswith("module_a_")
                    and isinstance(v, (int, float, bool))
                },
            }
        audit = {
            k: v
            for k, v in diag.items()
            if isinstance(k, str)
            and k.startswith(
                ("crs_", "nap_", "instr_source", "bam_", "miu_", "n_mius",
                 "prune_", "module_a_")
            )
            and isinstance(v, (int, float, str, bool, type(None)))
        }
        if isinstance(diag.get("miu_token_union"), list):
            audit["miu_token_union"] = [int(i) for i in diag["miu_token_union"]]
        rec = {
            "ts": round(time.time(), 3),
            "episode_key": str(episode_key or ""),
            "image_path": str(img_path or ""),
            "frame_index": int(frame_index),
            "is_current": bool(is_current),
            "action": str(action),
            "n_frame": int(n_frame),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "grid_ok": bool(grid_ok),
            "policy": str(getattr(model, "_geopact_prune_policy", "") or ""),
            "local_keep": sorted(int(i) for i in (local_keep or [])),
            "audit": audit,
        }
        # E8 commitment-error audit: the write-once retirement never re-runs the
        # selector, it contracts to a prefix of the stored nested order, so the
        # order (plus r_hist) is what a later-step history keep must be derived
        # from. Only reaches disk on instrumented dump runs.
        try:
            from lmms_eval.models.geopact_nap import lookup_nap_order

            entry = lookup_nap_order(
                model, _sticky_mask_key(episode_key, img_path, model=model)
            )
            if entry and entry.get("order"):
                rec["nap_order"] = [int(i) for i in entry["order"]]
                rec["nap_order_kind"] = str(entry.get("kind", ""))
        except Exception:
            pass
        rec["ratio_hist"] = float(getattr(model, "_geopact_prune_ratio", 0.0) or 0.0)
        # W43-C probe payload: merged (and consumed) only when the pending
        # probe belongs to this frame's sticky key.
        probe = getattr(model, "_geopact_indep_probe", None)
        if probe and probe.get("key") == _sticky_mask_key(
            episode_key, img_path, model=model
        ):
            model._geopact_indep_probe = None
            rec["k_c"] = int(probe["k_c"])
            rec["k_h"] = int(probe["k_h"])
            rec["keep_cur"] = probe["keep_cur"]
            rec["keep_hist_nested"] = probe["keep_hist_nested"]
            rec["keep_hist_independent"] = probe["keep_hist_independent"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _trace_sticky(episode_key, img_path, is_current, local_keep, cache_hit, frame_index) -> None:
    try:
        from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_sticky_frame

        if trace_enabled():
            trace_sticky_frame(
                episode_key=episode_key,
                image_path=img_path,
                is_current=bool(is_current),
                local_keep=[int(idx) for idx in local_keep],
                cache_hit=bool(cache_hit),
                frame_index=frame_index,
            )
    except Exception:
        pass


def compute_sticky_token_baseline_keep_set(
    model,
    *,
    token_ranges,
    image_embeds: torch.Tensor,
    image_grid_thw,
    ratio: float,
    policy: str,
    scope: Optional[str] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    image_mask_1d: Optional[torch.Tensor] = None,
) -> set:
    """Deprecated alias for :func:`compute_sticky_dialogue_prune_keep_set`."""
    return compute_sticky_dialogue_prune_keep_set(
        model,
        token_ranges=token_ranges,
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        ratio=ratio,
        policy=policy,
        scope=scope,
        inputs_embeds=inputs_embeds,
        image_mask_1d=image_mask_1d,
    )


def _frame_grid(image_grid_thw, range_idx: int, n_frames: int):
    if image_grid_thw is None or getattr(image_grid_thw, "numel", lambda: 0)() < 3:
        return image_grid_thw
    n_rows = int(image_grid_thw.shape[0])
    if n_rows >= n_frames and range_idx < n_rows:
        return image_grid_thw[range_idx : range_idx + 1]
    return image_grid_thw[-1:]


def _vision_patch_size(model) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    return int(getattr(vision_config, "patch_size", 16) or 16)


def _coord_size_for_sbr(model, image_grid_thw) -> Optional[Tuple[int, int]]:
    """Return resized-pixel coordinate canvas for native_mt4 SBR, else None."""
    if str(getattr(model, "_geopact_coordinate_space", "") or "") != "qwen_resized_pixel":
        return None

    geom = getattr(model, "_geopact_current_qwen_geometry", None) or {}
    width = geom.get("qwen_width") or geom.get("resized_width")
    height = geom.get("qwen_height") or geom.get("resized_height")
    if width and height:
        return max(1, int(width)), max(1, int(height))

    if image_grid_thw is None or getattr(image_grid_thw, "numel", lambda: 0)() < 3:
        return None
    row = image_grid_thw[-1].tolist()
    patch = _vision_patch_size(model)
    return max(1, int(row[2]) * patch), max(1, int(row[1]) * patch)


def _vz_attn_scores_for_range(
    model, policy: str, start: int, end: int, n_img_total: int
) -> Optional[torch.Tensor]:
    """Frame slice of the per-token scores captured by the transformers
    patch: the real ViT attention-received (GEOPACT_VZ_ATTN_CAPTURE=1;
    VisionZip / VisPruner / PruMerge+ / VisionTrim official scoring) or the
    ZOO-Prune zeroth-order projector sensitivity
    (GEOPACT_ZOO_SENS_CAPTURE=1). Rows are aligned 1:1 with the concatenated
    ``image_embeds``; a capture/shape mismatch is a hard error so the
    selector can never silently fall back to the deprecated feature proxy."""
    if model is None:
        return None
    if _policy_has(policy, "zooprune"):
        attr = "_geopact_zoo_sens_scores"
    elif _policy_has(policy, "visionzip", "vispruner", "prumerge", "trim"):
        # ("trim" also matches visiontrim, which shares the vz capture.)
        attr = "_geopact_vz_attn_scores"
    else:
        return None
    scores = getattr(model, attr, None)
    if scores is None:
        scores = getattr(getattr(model, "visual", None), attr, None)
    if scores is None:
        from lmms_eval.models.geopact_pub_baselines import (
            vz_attn_capture_enabled,
            zoo_sens_capture_enabled,
        )

        capture_on = (
            zoo_sens_capture_enabled()
            if attr == "_geopact_zoo_sens_scores"
            else vz_attn_capture_enabled()
        )
        if capture_on:
            visual = getattr(model, "visual", None)
            raise RuntimeError(
                f"GeoPACT {attr} capture enabled but no scores found: "
                f"model={type(model).__name__} "
                f"model_attr={getattr(model, attr, 'MISSING')!r} "
                f"visual={type(visual).__name__} "
                f"visual_attr={getattr(visual, attr, 'MISSING')!r} "
                f"visual_has_blocks={hasattr(visual, 'blocks')}"
            )
        return None
    scores = scores.reshape(-1)
    if int(scores.numel()) != int(n_img_total):
        raise RuntimeError(
            "GeoPACT vz-attn capture misaligned: scores rows "
            f"{int(scores.numel())} != visual tokens {int(n_img_total)}"
        )
    return scores[start:end]


def _prune_single_frame(
    *,
    start: int,
    end: int,
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    image_grid_thw,
    ratio: float,
    policy: str,
    detector: str,
    pil_image,
    rng: Optional[_random.Random],
    coord_size: Optional[Tuple[int, int]] = None,
    model=None,
    instr_query: Optional[torch.Tensor] = None,
    image_id: Optional[str] = None,
) -> tuple[set, list, list, Optional[str]]:
    """Return global keep indices, miu_token_idxs, mius, coord_text for one frame."""
    _pt: Dict[str, float] = {}
    _pt_t0 = time.perf_counter()
    if model is not None:
        model._geopact_prune_frame_timings = _pt
    n_frame = end - start
    keep_k = max(1, int(round(n_frame * ratio)))
    frame_embeds = image_embeds[start:end]
    frame_grid = _frame_grid(image_grid_thw, 0, 1)
    grid_h, grid_w, grid_matches = _get_grid(frame_grid, n_frame)

    from lmms_eval.models.geopact_qd_selector import bam_skip_module_a

    mius: list = []
    if (
        pil_image is not None
        and (not _policy_is_token_baseline(policy) or _pub_hybrid_active(policy))
        and not bam_skip_module_a()
    ):
        _set_module_a_grid_context(model, frame_grid, n_frame)
        mius = _run_detector(pil_image, detector, policy, model=model, image_id=image_id)
        _pt["prune_detector_ms"] = round((time.perf_counter() - _pt_t0) * 1000.0, 2)
        if not mius and _policy_uses_cdct(policy):
            try:
                from lmms_eval.models.geopact_sspc import sspc_enabled

                if sspc_enabled() and model is not None:
                    prev = getattr(model, "_geopact_sspc_audit", None) or {}
                    model._geopact_sspc_audit = {**prev, "module_a_empty": 1}
            except Exception:
                pass
        if (
            not mius
            and not _policy_has(policy, "random")
            and not _policy_uses_cdct(policy)
            and not _policy_uses_miu_first_backfill(policy)
            and not _policy_is_token_baseline(policy)
        ):
            # Legacy MIU/oas (non-CDCT): 0 MIUs → skip B/C for this frame.
            # (Hybrid pub arms fall through: empty detections degrade to the
            # unmodified published rule, never to a full keep.)
            keep_local = set(range(n_frame))
            _attach_miu_prune_diag(
                model,
                n_frame=n_frame,
                keep_k=keep_k,
                keep_local=keep_local,
                mius=[],
                mapped_token_groups=[],
                policy=policy,
            )
            return set(range(start, end)), [], [], None

    reserved_local: set = set()
    keep_k_b = keep_k
    sspc_pre_audit: dict = {}
    candidate_local = None
    try:
        from lmms_eval.models.geopact_sspc import resolve_frame_grid, sspc_enabled, sspc_strict_enabled
        from lmms_eval.models.geopact_sspc_prune import plan_sspc_keep_budget, sspc_in_budget

        if sspc_enabled() and model is not None:
            plan_gh, plan_gw, plan_grid_matched = resolve_frame_grid(grid_h, grid_w, n_frame)
            if not plan_grid_matched and sspc_strict_enabled() and sspc_in_budget():
                raise RuntimeError(
                    f"GeoPACT SSPC grid mismatch: n_frame={n_frame} vs grid {grid_h}x{grid_w}"
                )
            if sspc_in_budget():
                reserved_local, keep_k_b, sspc_pre_audit = plan_sspc_keep_budget(
                    keep_k,
                    grid_h=plan_gh,
                    grid_w=plan_gw,
                    n_frame=n_frame,
                    seed=int(start),
                )
                sspc_pre_audit = {
                    **sspc_pre_audit,
                    "sspc_hook_grid_h": grid_h,
                    "sspc_hook_grid_w": grid_w,
                    "sspc_hook_grid_matched": grid_matches,
                }
                candidate_local = set(range(n_frame)) - reserved_local
    except RuntimeError:
        raise
    except Exception as exc:
        _log.warning("GeoPACT SSPC budget plan failed (non-fatal): %s", exc)

    if candidate_local is not None:
        if not candidate_local:
            keep_k_b = 0
        else:
            keep_k_b = min(int(keep_k_b), len(candidate_local))

    # C-Entry (single-step Module C): reserve summary slots inside the keep
    # budget; B selects keep_k_b - R, the dropped set fills R summary tokens
    # after selection (total kept stays exactly keep_k).
    c_entry_r = 0
    if keep_k_b > 1 and keep_k < n_frame:
        from lmms_eval.models.geopact_c_entry import c_entry_enabled, c_entry_reserve

        if c_entry_enabled():
            c_entry_r = min(c_entry_reserve(keep_k), keep_k_b - 1)
            keep_k_b -= c_entry_r

    _pt_hes_t0 = time.perf_counter()
    hes = HESResult()
    # P6: share one MIU->token mapping between HES selection and the
    # coord/diag mapping below (same inputs, bit-identical result).
    mapping_cache: dict = {}
    if keep_k_b <= 0:
        keep_b_local: set = set()
    else:
        hes = _hes_select(
            mius=mius,
            image_embeds=frame_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=frame_grid,
            keep_k_tokens=keep_k_b,
            policy=policy,
            rng=rng,
            instr_query=instr_query,
            candidate_local=candidate_local,
            mapping_cache=mapping_cache,
            attn_scores=_vz_attn_scores_for_range(
                model, policy, start, end, int(image_embeds.shape[0])
            ),
        )
        keep_b_local = set(hes)
    _pt["prune_hes_ms"] = round((time.perf_counter() - _pt_hes_t0) * 1000.0, 2)
    for _sk, _sv in _HES_STAGE_MS.items():
        _pt[f"prune_{_sk}"] = round(float(_sv), 2)
    if model is not None:
        # NAP: frame-local greedy pick sequence for the nested order store.
        model._geopact_last_pick_order_local = [
            int(t) for t in (getattr(hes, "picked_token_order", None) or [])
        ]
    if reserved_local:
        keep_local = keep_b_local | reserved_local
    else:
        keep_local = keep_b_local
    if c_entry_r > 0 and len(keep_local) < n_frame and grid_matches:
        from lmms_eval.models.geopact_c_entry import build_c_entry_summaries

        medoids, summaries, c_audit = build_c_entry_summaries(
            frame_embeds,
            keep_local,
            n_frame,
            grid_h,
            grid_w,
            c_entry_r,
        )
        if medoids and summaries is not None:
            med_t = torch.tensor(medoids, dtype=torch.long, device=frame_embeds.device)
            frame_embeds.index_copy_(0, med_t, summaries.to(frame_embeds.dtype))
            keep_local = set(keep_local) | set(medoids)
        if model is not None:
            sp_audit = dict(getattr(model, "_geopact_spatial_audit", None) or {})
            sp_audit.update(c_audit)
            model._geopact_spatial_audit = sp_audit
            model._geopact_c_entry_audit = dict(c_audit)
    # C-Absorb entry extension (W-A3, independent arm, default off): merge the
    # dropped token *embeddings* into kept ones (ToMe-like; changes keys and
    # the downstream trajectory — no value-only theory claimed here). Also the
    # single-step-compatible form of Module C.
    try:
        from lmms_eval.models.geopact_c_absorb import (
            entry_merge_embeds,
            entry_merge_lambda,
        )

        _lam_entry = entry_merge_lambda()
        if _lam_entry > 0.0 and len(keep_local) < n_frame:
            em_audit = entry_merge_embeds(
                frame_embeds, set(int(i) for i in keep_local), n_frame, _lam_entry
            )
            if model is not None and em_audit.get("entry_merge_applied"):
                # merged into _geopact_last_prune_diag after the diag rebuild
                # below (same pattern as the C-Entry audit).
                model._geopact_entry_merge_audit = dict(em_audit)
    except Exception as exc:
        if os.environ.get("GEOPACT_STRICT", "0").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            raise
        _log.warning("GeoPACT entry merge failed (non-fatal): %s", exc)
    try:
        from lmms_eval.models.geopact_sspc import record_keep_b, record_keep_final, sspc_enabled
        from lmms_eval.models.geopact_sspc_prune import apply_sspc_extra_keep, sspc_in_budget

        if sspc_enabled() and model is not None:
            if not sspc_in_budget():
                keep_local, sspc_extra_audit = apply_sspc_extra_keep(
                    keep_b_local,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    n_frame=n_frame,
                    seed=int(start),
                    keep_k_total=keep_k,
                )
            else:
                _, sspc_extra_audit = apply_sspc_extra_keep(
                    keep_local,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    n_frame=n_frame,
                    seed=int(start),
                    keep_k_total=keep_k,
                )
            record_keep_b(model, {start + int(i) for i in keep_b_local})
            record_keep_final(model, {start + int(i) for i in keep_local})
            prev = getattr(model, "_geopact_sspc_audit", None) or {}
            model._geopact_sspc_audit = {**prev, **sspc_pre_audit, **sspc_extra_audit}
    except Exception as exc:
        _log.warning("GeoPACT SSPC extra-keep failed (non-fatal): %s", exc)
    _pt_map_t0 = time.perf_counter()
    keep_global = {start + idx for idx in keep_local}
    mapped_token_idxs: list = []
    if mius and grid_matches:
        mapped_token_idxs, _, _, mapped_miu_indices = _map_mius_to_tokens_cached(
            mapping_cache, mius, frame_embeds, grid_h, grid_w, n_frame
        )
        miu_token_idxs = _miu_token_idxs_table(
            len(mius), mapped_token_idxs, mapped_miu_indices, frame_offset=start
        )
    else:
        miu_token_idxs = []
    _pt["prune_map_ms"] = round((time.perf_counter() - _pt_map_t0) * 1000.0, 2)

    miu_backfill = int(getattr(hes, "miu_backfill_tokens", 0) or 0)
    _attach_miu_prune_diag(
        model,
        n_frame=n_frame,
        keep_k=keep_k,
        keep_local=keep_local,
        mius=mius,
        mapped_token_groups=mapped_token_idxs,
        policy=policy,
        cdct_fallback="no_spatial_prior" if _policy_uses_cdct(policy) and not mius else None,
        miu_backfill_tokens=miu_backfill,
    )
    _attach_module_a_prefetch_diag(model, policy)
    if c_entry_r > 0 and model is not None:
        diag_ce = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
        diag_ce.update(getattr(model, "_geopact_c_entry_audit", None) or {})
        model._geopact_last_prune_diag = diag_ce
    if model is not None and getattr(model, "_geopact_entry_merge_audit", None):
        diag_em = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
        diag_em.update(model._geopact_entry_merge_audit)
        model._geopact_last_prune_diag = diag_em
        model._geopact_entry_merge_audit = None

    selected_miu_indices = list(getattr(hes, "selected_miu_indices", []) or [])
    pick_order_global = [
        start + int(t) for t in (getattr(hes, "picked_token_order", None) or [])
    ]
    if miu_token_idxs and pick_order_global:
        topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
        selected_miu_indices = _coord_miu_indices_from_pick_order(
            pick_order_global,
            miu_token_idxs,
            mius,
            topk=topk_env if topk_env > 0 else 0,
        )
    elif miu_token_idxs and not selected_miu_indices:
        selected_miu_indices = sorted(
            range(len(miu_token_idxs)),
            key=lambda i: _info_density(mius[i]) / max(len(miu_token_idxs[i]), 1),
            reverse=True,
        )

    coord_text = None
    if _should_build_coord_text(policy):
        if _coord_mode() == "gaec":
            from lmms_eval.models.geopact_coord_injection.gaec import prepare_gaec_context

            topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
            prepare_gaec_context(
                model,
                selected_miu_indices=selected_miu_indices,
                keep_token_set=keep_global,
                miu_token_idxs=miu_token_idxs,
                mius=mius,
                frame_start=start,
                n_frame=n_frame,
                topk=topk_env if topk_env > 0 else 3,
            )
        else:
            coord_text = _build_coord_text(
                selected_miu_indices=selected_miu_indices,
                keep_token_set=keep_global,
                miu_token_idxs=miu_token_idxs,
                mius=mius,
                policy=policy,
                coord_size=coord_size,
                miu_order=selected_miu_indices,
            )
    _attach_compact_prune_ctx(
        model,
        start=start,
        n_frame=n_frame,
        miu_token_idxs=miu_token_idxs,
        mius=mius,
        policy=policy,
    )
    _pt["prune_total_ms"] = round((time.perf_counter() - _pt_t0) * 1000.0, 2)
    return keep_global, miu_token_idxs, mius, coord_text


def _attach_miu_prune_diag(
    model,
    *,
    n_frame: int,
    keep_k: int,
    keep_local: set,
    mius: list,
    mapped_token_groups: list,
    policy: str = "",
    cdct_fallback: Optional[str] = None,
    miu_backfill_tokens: int = 0,
) -> None:
    """Record prune coverage stats on model for compact_stats / audit scripts."""
    if model is None:
        return
    miu_union: set = set()
    for group in mapped_token_groups or []:
        miu_union.update(int(i) for i in group)
    keep_count = len(keep_local)
    if mius and mapped_token_groups:
        union_count = len(miu_union)
        coverage = float(union_count) / float(max(1, n_frame))
        outside = len(set(keep_local) - miu_union)
        outside_ratio = float(outside) / float(max(1, keep_count))
    else:
        union_count = n_frame
        coverage = 1.0
        outside_ratio = 0.0
    diag = {
        "keep_k": int(keep_k),
        "keep_count": int(keep_count),
        "keep_fill_ratio": round(keep_count / max(1, keep_k), 4),
        "budget_shortfall": max(0, int(keep_k) - int(keep_count)),
        "miu_union_token_count": int(union_count),
        "miu_coverage_ratio": round(coverage, 4),
        "n_mius_detected": int(len(mius or [])),
        "keep_outside_union_ratio": round(outside_ratio, 4),
        "keep_local_hash": _keep_local_hash(keep_local),
    }
    if _prune_diag_dump_path() and mius and mapped_token_groups:
        # E6 failure binning needs to test GT-target-vs-detection overlap; the
        # union token set only reaches disk on instrumented dump runs.
        diag["miu_token_union"] = sorted(int(i) for i in miu_union)
    if policy and _policy_is_token_baseline(policy):
        diag["prune_selector"] = _prune_selector_label(policy)
    if _policy_uses_cdct(policy):
        diag["cdct_alpha"] = _cdct_alpha(policy)
        diag["prune_selector"] = "token_ickh_cdct"
        if cdct_fallback:
            diag["cdct_fallback"] = cdct_fallback
    # Truthful selector label on the v3 stack: with GEOPACT_B_SELECTOR=bam the
    # CDCT naming above is vestigial routing, not the selector that ran. Also
    # surface the BAM audit (uniform-a / mass engaged / QK / GQC) in the LIVE
    # prune diag — previously only the rectify path recorded it, so an
    # empty-detector "m == 1" frame was invisible in step audits.
    try:
        from lmms_eval.models.geopact_qd_selector import (
            pop_qd_frame_export,
            qd_selector_kind,
        )

        if qd_selector_kind() == "bam":
            diag["prune_selector"] = "bam_v3"
            exp = pop_qd_frame_export()
            audit_exp = (exp or {}).get("audit") or {}
            # Guard against a stale export from an earlier frame (paths that
            # skip selection entirely never record one for this frame).
            if int(audit_exp.get("bam_n") or -1) == int(n_frame):
                for _k in (
                    "bam_uniform_a",
                    "bam_has_mass",
                    "bam_soft_a",
                    "bam_qk_relevance",
                    "bam_gqc",
                    "bam_no_diversity",
                ):
                    if _k in audit_exp:
                        diag[_k] = bool(audit_exp[_k])
                # numeric alpha diagnostics (auto prior-strength design data)
                for _k in (
                    "bam_prior_g",
                    "bam_rel_g",
                    "bam_pa_corr",
                    "bam_pa_overlap",
                    "bam_alpha_star",
                    "bam_alpha_auto",
                    "bam_mass_alpha",
                    "bam_keep_k",
                    "bam_n",
                ):
                    if _k in audit_exp:
                        diag[_k] = float(audit_exp[_k])
    except Exception:
        pass
    if _policy_uses_miu_first_backfill(policy):
        diag["prune_selector"] = "miu_first_token_backfill"
        diag["miu_backfill_tokens"] = int(miu_backfill_tokens)
    model._geopact_last_prune_diag = diag


def _attach_module_a_prefetch_diag(model, policy: str) -> None:
    if model is None or not _module_a_prefetch_enabled():
        return
    diag = dict(getattr(model, "_geopact_last_prune_diag", {}) or {})
    diag["module_a_prefetch"] = _module_a_prefetch_mode()
    if _policy_uses_cdct(policy):
        alpha = _cdct_alpha(policy)
        diag["cdct_alpha"] = alpha
        if alpha == 0.0:
            diag["module_a_prefetch_alpha0_cdct"] = True
    if getattr(model, "_geopact_module_a_subprocess_isolated", False):
        diag["module_a_subprocess"] = True
    if getattr(model, "_geopact_module_a_subprocess_pooled", False):
        diag["module_a_subprocess_pool"] = True
    if getattr(model, "_geopact_module_a_prefetch_async_join_ms", None) is not None:
        diag["module_a_prefetch_async"] = True
        diag["module_a_join_ms"] = getattr(
            model, "_geopact_module_a_prefetch_async_join_ms"
        )
        overlap = getattr(model, "_geopact_module_a_prefetch_async_overlap_ms", None)
        if overlap is not None:
            diag["module_a_overlap_ms"] = overlap
    meta = getattr(model, "_geopact_last_module_a_meta", None) or {}
    if isinstance(meta, dict) and int(meta.get("n_kept") or meta.get("n_raw") or 0) > 0:
        diag["module_a_spatial_prior_computed"] = True
    model._geopact_last_prune_diag = diag


def _map_mius_to_tokens(mius, image_embeds, grid_h, grid_w, n_img, need_embs=True):
    """W57b: ``need_embs=False`` skips the per-MIU ``index_select().mean``
    chain (~2 GPU launches x n_mius per frame). Only the deprecated MIU-level
    herding consumes the mean embeddings; every CDCT/BAM-era caller reads the
    token groups and source indices alone."""
    miu_token_idxs, miu_energies, miu_mean_embs, mapped_miu_indices = [], [], [], []
    device = image_embeds.device
    idx_tensors: list = []
    for source_idx, m in enumerate(mius):
        box = m.get("box", (0.0, 0.0, 1.0, 1.0))
        idx_t = _token_indices_for_miu_box(box, grid_h, grid_w, n_img, device)
        if idx_t is None:
            continue
        if need_embs:
            miu_mean_embs.append(image_embeds.index_select(0, idx_t).mean(0))
        miu_energies.append(float(m.get("energy", m.get("pentagonal_energy", 0.5)) or 0.5))
        idx_tensors.append(idx_t)
        mapped_miu_indices.append(source_idx)
    if idx_tensors:
        # One batched GPU->CPU transfer instead of a sync per MIU.
        lengths = [int(t.numel()) for t in idx_tensors]
        flat = torch.cat(idx_tensors).cpu().tolist()
        off = 0
        for length in lengths:
            miu_token_idxs.append([int(x) for x in flat[off : off + length]])
            off += length
    return miu_token_idxs, miu_energies, miu_mean_embs, mapped_miu_indices


_LEGACY_MIU_HERDING_WARNED = False


def _kernel_herding(
    mius, miu_token_idxs, miu_energies, miu_mean_embs,
    mapped_miu_indices, inputs_embeds, image_mask, hidden, keep_k_tokens, policy,
    instr_query: Optional[torch.Tensor] = None,
):
    # DEPRECATED (GeoPACT v2): the 1/(t(t+1)) repulsion of this MIU-level loop
    # over-penalizes early picks. Kept only as a negative control; no v2 arm
    # may reach it.
    global _LEGACY_MIU_HERDING_WARNED
    if not _LEGACY_MIU_HERDING_WARNED:
        _LEGACY_MIU_HERDING_WARNED = True
        _log.warning(
            "GeoPACT: legacy MIU _kernel_herding is deprecated; "
            "use CDCT/token paths (policy=%s)",
            policy,
        )
    n_mius = len(miu_mean_embs)
    E      = torch.stack(miu_mean_embs)       # (n_mius, hidden)
    E_norm = F.normalize(E, dim=-1)

    # Instruction text embedding q (cached global task goal on dialogue suffix steps)
    if instr_query is not None:
        q = instr_query.detach().to(device=E_norm.device, dtype=E_norm.dtype)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=-1)
    else:
    text_embs = inputs_embeds[0][~image_mask]   # (n_text, hidden)
    if text_embs.shape[0] > 0 and not _policy_has(policy, "no_instr_rel", "no_instruction"):
        q = F.normalize(text_embs.mean(0, keepdim=True), dim=-1)  # (1, hidden)
    else:
        q = torch.zeros(1, hidden, dtype=E_norm.dtype, device=E_norm.device)

    # r_i = max(0, <ē_i, q>)
    relevance = torch.clamp((E_norm @ q.T).squeeze(-1), min=0.0)

    # ρ_i = InfoDensity / n_i
    use_density = not _policy_has(policy, "no_density", "diversity_only")
    if use_density:
        density = torch.tensor(
            [_info_density(mius[i]) / max(len(miu_token_idxs[i]), 1)
             for i in range(n_mius)],
            dtype=E.dtype, device=E.device,
        )
        if density.max() > 1e-8:
            density = density / density.max()
    else:
        density = torch.ones(n_mius, dtype=E.dtype, device=E.device)

    # w_i = ρ_i × r_i  (or ρ_i when no_instr_rel)
    if _policy_has(policy, "diversity_only"):
        weights = torch.ones(n_mius, dtype=E.dtype, device=E.device)
    else:
        weights = density * relevance
    if weights.sum() < 1e-8:
        weights = density + 1e-8

    # μ_i = (K w)_i / Σ w_j
    K  = E_norm @ E_norm.T
    mu = (K @ weights) / weights.sum()

    # Greedy kernel herding
    selected: list = []
    tokens_so_far: set = set()
    neg_inf = float("-inf")
    for _ in range(n_mius):
        if len(tokens_so_far) >= keep_k_tokens:
            break
        if selected:
            s_t    = torch.tensor(selected, dtype=torch.long, device=K.device)
            sim    = K[:, s_t].mean(dim=1)
            scores = mu - sim / (len(selected) + 1)
        else:
            scores = mu.clone()
        for prev in selected:
            scores[prev] = neg_inf
        best = int(scores.argmax().item())
        selected.append(best)
        tokens_so_far.update(miu_token_idxs[best])

    return set(list(tokens_so_far)[:keep_k_tokens]), [mapped_miu_indices[i] for i in selected]


# ---------------------------------------------------------------------------
# Stage 3: SBR (Spatial Binding Restoration)
# ---------------------------------------------------------------------------

def _gaec_carrier_token_id(model, tokenizer) -> int:
    """Non-image token id for GAEC carriers (must not match image_token_id)."""
    image_tok = int(getattr(model.config, "image_token_id", -1) or -1)
    for candidate in (
        getattr(model.config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None,
        getattr(model.config, "pad_token_id", None),
    ):
        if candidate is not None and int(candidate) != image_tok:
            return int(candidate)
    return 0


def _attach_compact_prune_ctx(
    model,
    *,
    start: int,
    n_frame: int,
    miu_token_idxs: list,
    mius: list,
    policy: str,
) -> None:
    """Stash MIU + CDCT density for ES-RPB / residue in _compact_sequence."""
    if model is None:
        return
    density_flat = None
    if mius and miu_token_idxs and _policy_uses_cdct(policy):
        local_groups = [
            [int(t) - int(start) for t in grp if int(start) <= int(t) < int(start) + int(n_frame)]
            for grp in miu_token_idxs
        ]
        try:
            import torch as _torch

            density_flat = _build_cdct_density_field(
                mius,
                local_groups,
                int(n_frame),
                _torch.device("cpu"),
            )
        except Exception:
            density_flat = None
    model._geopact_compact_prune_ctx = {
        "frame_start": int(start),
        "n_frame": int(n_frame),
        "miu_token_idxs": miu_token_idxs,
        "mius": mius,
        "cdct_density_flat": density_flat,
    }


def _build_coord_text(
    selected_miu_indices: list,
    keep_token_set: set,
    miu_token_idxs: list,
    mius: list,
    policy: str,
    coord_size: Optional[Tuple[int, int]] = None,
    *,
    miu_order: Optional[list] = None,
) -> Optional[str]:
    """Build Module C coord appendix via UDSI pipeline."""
    from lmms_eval.models.geopact_coord_injection.pipeline import build_coord_injection

    if _policy_has(policy, "no_coord"):
        from lmms_eval.models.geopact_coord_appendix import record_coord_build_meta

        record_coord_build_meta({})
        return None

    rec = build_coord_injection(
        selected_miu_indices=selected_miu_indices,
        keep_token_set=keep_token_set,
        miu_token_idxs=miu_token_idxs,
        mius=mius,
        policy=policy,
        coord_size=coord_size,
        miu_order=miu_order,
        ephemeral=_coord_ephemeral_enabled(),
        metadata_only=_coord_metadata_only(),
    )
    return rec.text or None


# ---------------------------------------------------------------------------
# Compact sequence builder
# ---------------------------------------------------------------------------

def _attention_mask_for_rope(attention_mask):
    """Normalize attention_mask for Qwen3VLModel.get_rope_index."""
    if attention_mask is None:
        return None
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask.get("full_attention", attention_mask)
    if attention_mask.ndim == 4:
        attention_mask = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
        if attention_mask.dtype.is_floating_point:
            attention_mask = attention_mask / torch.finfo(attention_mask.dtype).min
            attention_mask = (1.0 - attention_mask).int()
    return attention_mask


def _resolve_rope_model(model):
    """Return the module that owns get_rope_index (Qwen3VLModel / Qwen2VLModel)."""
    if model is None:
        return None
    if hasattr(model, "get_rope_index"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "get_rope_index"):
        return inner
    return None


def _mrope_full_cache_fingerprint(input_ids, image_grid_thw, video_grid_thw) -> tuple:
    """Content-aware cache key for full-sequence MRoPE within one prefill."""
    parts: list = [_tensor_content_fingerprint(input_ids)]
    parts.append(
        _tensor_content_fingerprint(image_grid_thw) if image_grid_thw is not None else None
    )
    parts.append(
        _tensor_content_fingerprint(video_grid_thw) if video_grid_thw is not None else None
    )
    return tuple(parts)


def _tensor_content_fingerprint(tensor: torch.Tensor) -> tuple:
    """Cheap stable fingerprint; avoids stale hits after allocator reuse."""
    t = tensor.detach()
    if t.numel() == 0:
        return (tuple(int(x) for x in t.shape), 0, ())
    flat = t.reshape(-1)
    n = int(flat.numel())
    sample_idx = (0, n // 2, n - 1) if n >= 3 else tuple(range(n))
    samples = tuple(int(flat[i].item()) for i in sample_idx)
    return (tuple(int(x) for x in t.shape), int(t.sum().item()), samples)


def _subset_mrope_cache_fingerprint(position_ids, keep_mask: torch.Tensor) -> tuple:
    keep_on_device = keep_mask.to(position_ids.device)
    return (
        _tensor_content_fingerprint(position_ids),
        _tensor_content_fingerprint(keep_on_device),
    )


def clear_geopact_mrope_caches(model) -> None:
    """Drop MRoPE structure caches (call on episode / prompt boundary changes)."""
    inner = getattr(model, "model", model)
    for attr in ("_geopact_mrope_full_cache", "_geopact_mrope_subset_cache"):
        if hasattr(inner, attr):
            delattr(inner, attr)


@dataclass
class _MropePrecomputeHandle:
    """Async or sync full-sequence MRoPE precompute (origpos path)."""

    model: Any
    stream: Any = None
    start_event: Any = None
    end_event: Any = None
    sync_done: bool = False
    overlapped: bool = False
    position_ids: Optional[torch.Tensor] = None
    rope_deltas: Optional[torch.Tensor] = None
    wall_t0: float = 0.0


def _mrope_vit_overlap_enabled(model, input_ids) -> bool:
    """Overlap full MRoPE with ViT only when the prompt is long enough to benefit."""
    if os.environ.get("GEOPACT_MROPE_VIT_OVERLAP", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return False
    inner = getattr(model, "model", model)
    if bool(getattr(inner, "_geopact_dialogue_suffix_only_step", False)):
        return False
    if input_ids is None:
        return False
    min_seq = int(os.environ.get("GEOPACT_MROPE_OVERLAP_MIN_SEQ", "1024"))
    return int(input_ids.shape[-1]) >= min_seq


def _mrope_side_stream(model, device: torch.device) -> Any:
    """Reuse one side stream per inner model (avoid per-step stream alloc)."""
    inner = getattr(model, "model", model)
    stream = getattr(inner, "_geopact_mrope_side_stream", None)
    if stream is None or getattr(stream, "device", None) != device:
        stream = torch.cuda.Stream(device=device)
        inner._geopact_mrope_side_stream = stream
    return stream


def _begin_precompute_full_mrope(
    model,
    input_ids,
    image_grid_thw,
    attention_mask,
    *,
    video_grid_thw=None,
) -> _MropePrecomputeHandle:
    """Start full MRoPE on a side CUDA stream (overlaps with ViT when available)."""
    handle = _MropePrecomputeHandle(model=model, wall_t0=time.perf_counter())
    use_cuda_stream = False
    try:
        use_cuda_stream = (
            torch.cuda.is_available()
            and input_ids is not None
            and getattr(input_ids, "is_cuda", False)
            and _mrope_vit_overlap_enabled(model, input_ids)
        )
    except Exception:
        use_cuda_stream = False

    if not use_cuda_stream:
        handle.sync_done = True
        handle.position_ids, handle.rope_deltas = _precompute_full_mrope(
            model,
            input_ids,
            image_grid_thw,
            attention_mask,
            video_grid_thw=video_grid_thw,
        )
        return handle

    handle.overlapped = True
    handle.stream = _mrope_side_stream(model, input_ids.device)
    handle.start_event = torch.cuda.Event(enable_timing=True)
    handle.end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(handle.stream):
        handle.start_event.record(handle.stream)
        handle.position_ids, handle.rope_deltas = _precompute_full_mrope(
            model,
            input_ids,
            image_grid_thw,
            attention_mask,
            video_grid_thw=video_grid_thw,
        )
        handle.end_event.record(handle.stream)
    return handle


def _end_precompute_full_mrope(
    handle: Optional[_MropePrecomputeHandle],
    *,
    ledger=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Join async MRoPE precompute; record ``origpos_mrope_prefull_ms`` when ledger set."""
    if handle is None:
        return None, None

    prefull_ms = 0.0
    if handle.sync_done or handle.stream is None:
        prefull_ms = (time.perf_counter() - handle.wall_t0) * 1000.0
        else:
        default = torch.cuda.current_stream(device=handle.position_ids.device)
        default.wait_stream(handle.stream)
        if handle.position_ids is not None:
            handle.position_ids.record_stream(default)
        if handle.rope_deltas is not None:
            handle.rope_deltas.record_stream(default)
        if handle.start_event is not None and handle.end_event is not None:
            prefull_ms = float(handle.start_event.elapsed_time(handle.end_event))
        else:
            prefull_ms = (time.perf_counter() - handle.wall_t0) * 1000.0

    if ledger is not None and prefull_ms > 0:
        if handle.overlapped:
            ledger.add_ms("origpos_mrope_prefull_ms", prefull_ms)
        else:
            ledger.add_ms("origpos_mrope_ms", prefull_ms)

    return handle.position_ids, handle.rope_deltas


def _precompute_full_mrope(
    model,
    input_ids,
    image_grid_thw,
    attention_mask,
    *,
    video_grid_thw=None,
):
    """Qwen3-VL computes MRoPE inside forward *after* the GeoPACT hook.

    For origpos pruning we must build full-sequence position_ids first, then
    subset kept tokens. prepare_inputs_for_generation forces position_ids=None.
    """
    inner = getattr(model, "model", model)
    fp = _mrope_full_cache_fingerprint(input_ids, image_grid_thw, video_grid_thw)
    cached = getattr(inner, "_geopact_mrope_full_cache", None)
    if cached is not None and cached.get("fp") == fp:
        return cached["position_ids"], cached["rope_deltas"]

    rope_model = _resolve_rope_model(model)
    get_rope = getattr(rope_model, "get_rope_index", None) if rope_model is not None else None
    if get_rope is None or input_ids is None:
        raise RuntimeError(
            "GeoPACT origpos: model.get_rope_index unavailable "
            f"(model={type(model).__name__})"
        )
    attn = _attention_mask_for_rope(attention_mask)
    with torch.no_grad():
        position_ids, rope_deltas = get_rope(
            input_ids, image_grid_thw, video_grid_thw, attention_mask=attn
        )
    if position_ids is None:
        raise RuntimeError("GeoPACT origpos: get_rope_index returned position_ids=None")
    if position_ids.shape[-1] != input_ids.shape[1]:
        raise RuntimeError(
            "GeoPACT origpos: MRoPE length mismatch "
            f"pos={position_ids.shape[-1]} ids={input_ids.shape[1]}"
        )
    inner._geopact_mrope_full_cache = {
        "fp": fp,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
    }
    return position_ids, rope_deltas


def _subset_origpos_mrope_cached(
    model,
    position_ids,
    keep_mask: torch.Tensor,
    compact_ids: torch.Tensor,
    attention_mask=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subset MRoPE with per-prefill cache keyed by full position tensor + keep pattern."""
    inner = getattr(model, "model", model)
    fp = _subset_mrope_cache_fingerprint(position_ids, keep_mask)
    cached = getattr(inner, "_geopact_mrope_subset_cache", None)
    if cached is not None and cached.get("fp") == fp:
        return cached["compact_pos"], cached["compact_rope"]
    compact_pos, compact_rope = _subset_origpos_mrope(
        position_ids, keep_mask, compact_ids, attention_mask
    )
    inner._geopact_mrope_subset_cache = {
        "fp": fp,
        "compact_pos": compact_pos,
        "compact_rope": compact_rope,
    }
    return compact_pos, compact_rope


def _visual_keep_in_sequence_order_tensor(
    keep_token_set: set,
    img_seq_positions: torch.Tensor,
) -> torch.Tensor:
    """Flat visual indices in prompt sequence order (tensor, no Python list)."""
    n_visual = int(img_seq_positions.numel())
    if not keep_token_set:
        return torch.empty(0, dtype=torch.long, device=img_seq_positions.device)
    vi = torch.tensor(
        [int(x) for x in keep_token_set],
        dtype=torch.long,
        device=img_seq_positions.device,
    )
    bad = vi[(vi < 0) | (vi >= n_visual)]
    if bad.numel() > 0:
        preview = bad[:8].tolist()
        raise RuntimeError(
            "GeoPACT visual keep index out of range: "
            f"bad={preview} n_bad={int(bad.numel())} n_visual={n_visual}. "
            "keep_token_set must contain flat image-token indices, not input_ids sequence positions."
        )
    seq_pos = img_seq_positions.index_select(0, vi)
    order = seq_pos.argsort()
    return vi.index_select(0, order)


def _visual_keep_in_sequence_order(keep_token_set: set, img_seq_positions: list) -> list:
    """Order flat visual indices by prompt sequence (required for deepstack alignment)."""
    n_visual = len(img_seq_positions)
    bad = sorted(int(vi) for vi in keep_token_set if int(vi) < 0 or int(vi) >= n_visual)
    if bad:
        preview = bad[:8]
        raise RuntimeError(
            "GeoPACT visual keep index out of range: "
            f"bad={preview} n_bad={len(bad)} n_visual={n_visual}. "
            "keep_token_set must contain flat image-token indices, not input_ids sequence positions."
        )
    pairs = [
        (int(img_seq_positions[vi]), int(vi))
        for vi in keep_token_set
    ]
    pairs.sort(key=lambda item: item[0])
    return [vi for _, vi in pairs]


def _rewrite_compact_image_grid(
    image_grid_thw: torch.Tensor,
    token_ranges: list,
    keep_token_set: set,
    merge_size: int,
) -> torch.Tensor:
    """Rewrite grid_thw so token count matches kept visual tokens (7B-style, per frame)."""
    merge = max(int(merge_size or 2), 1)
    rows = []
    for range_idx, (start, end) in enumerate(token_ranges):
        kept = sum(1 for k in keep_token_set if start <= int(k) < end)
        if kept <= 0:
            kept = max(1, int(round((end - start) * 0.05)))
        if image_grid_thw.shape[0] == 1 and len(token_ranges) > 1:
            src = image_grid_thw[0]
        else:
            src = image_grid_thw[min(range_idx, int(image_grid_thw.shape[0]) - 1)]
        rows.append([1, merge, int(kept) * merge])
    grid = torch.tensor(rows, dtype=image_grid_thw.dtype, device=image_grid_thw.device)
    merge_sq = merge * merge
    for i, (start, end) in enumerate(token_ranges):
        kept = sum(1 for k in keep_token_set if start <= int(k) < end)
        if kept <= 0:
            kept = 1
        expected = int(grid[i].prod().item()) // merge_sq
        if expected != kept:
            raise RuntimeError(
                f"GeoPACT grid rewrite mismatch frame={i}: grid_tokens={expected} kept={kept}"
            )
    return grid


def _subset_origpos_mrope(
    position_ids,
    keep_mask: torch.Tensor,
    compact_ids: torch.Tensor,
    attention_mask=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subset full-sequence MRoPE to kept tokens; recompute rope_deltas for compact row."""
    if keep_mask.shape[0] != position_ids.shape[-1]:
        raise RuntimeError(
            "GeoPACT origpos: keep_mask length mismatch "
            f"mask={keep_mask.shape[0]} pos={position_ids.shape[-1]}"
        )
    compact_pos = position_ids[:, :, keep_mask.to(position_ids.device)].contiguous()
    if compact_pos.numel() == 0:
        raise RuntimeError("GeoPACT origpos: empty compact_pos after subset")

    if attention_mask is not None:
        compact_len = attention_mask.sum(-1, keepdim=True).to(compact_pos.device)
    else:
        compact_len = torch.tensor(
            [[compact_ids.shape[1]]],
            device=compact_pos.device,
            dtype=torch.long,
        )
    max_position_ids = compact_pos.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    compact_rope = (max_position_ids + 1 - compact_len).contiguous().to(dtype=torch.long)
    return compact_pos, compact_rope


def _build_keep_mask(
    image_mask: torch.Tensor,
    img_seq_positions: torch.Tensor,
    keep_token_set: set,
) -> torch.Tensor:
    """Build per-sequence keep mask: all text tokens + selected visual flat indices."""
    keep_mask = ~image_mask
    if not keep_token_set:
        return keep_mask
    vi = torch.tensor(
        [int(x) for x in keep_token_set],
        dtype=torch.long,
        device=image_mask.device,
    )
    n_visual = int(img_seq_positions.numel())
    bad_keep = vi[(vi < 0) | (vi >= n_visual)]
    if bad_keep.numel() > 0:
        preview = bad_keep[:8].tolist()
        raise RuntimeError(
            "GeoPACT keep_mask: visual keep index out of range: "
            f"bad={preview} n_bad={int(bad_keep.numel())} n_visual={n_visual}. "
            "Expected flat image-token indices; absolute input_ids positions are invalid here."
        )
    keep_mask = keep_mask.clone()
    keep_mask[img_seq_positions.index_select(0, vi)] = True
    return keep_mask


def _gather_compact_prefill_tensors(
    input_ids,
    inputs_embeds,
    image_embeds,
    attention_mask,
    keep_mask: torch.Tensor,
    keep_token_set: set,
    img_seq_positions: torch.Tensor,
    image_tok_id: int,
):
    """Gather compact prefill tensors with one shared keep_mask (origpos-safe)."""
    dev = input_ids.device
    if inputs_embeds.device != dev:
        inputs_embeds = inputs_embeds.to(dev)
    if image_embeds.device != dev:
        image_embeds = image_embeds.to(device=dev, dtype=inputs_embeds.dtype)
    mask = keep_mask.to(device=dev, dtype=torch.bool)
    compact_ids = input_ids[:, mask]
    compact_embs = inputs_embeds[:, mask]
    if keep_token_set and int(image_embeds.shape[0]) > 0:
        keep_seq = _visual_keep_in_sequence_order_tensor(keep_token_set, img_seq_positions)
        vis_slots = (compact_ids[0] == image_tok_id).nonzero(as_tuple=False).flatten()
        if vis_slots.numel() > 0:
            vis_rows = image_embeds.index_select(0, keep_seq.to(device=dev)).to(compact_embs.dtype)
            compact_embs = compact_embs.clone()
            compact_embs[0, vis_slots] = vis_rows
    if attention_mask is not None:
        compact_attn = attention_mask.to(dev)[:, mask]
    else:
        compact_attn = None
    return compact_ids, compact_embs, compact_attn


def _normalize_image_mask_1d(image_mask: torch.Tensor, seq_len: int, *, where: str) -> torch.Tensor:
    """Normalize Qwen placeholder masks to a single (seq_len,) bool row.

    Qwen variants may expose image masks as (seq,), (1, seq), (seq, hidden),
    or (1, seq, hidden). Never reduce the sequence dimension accidentally.
    """
    if image_mask.dim() == 1:
        out = image_mask
    elif image_mask.dim() == 2:
        if int(image_mask.shape[0]) == 1 and int(image_mask.shape[1]) == seq_len:
            out = image_mask[0]
        elif int(image_mask.shape[0]) == seq_len:
            out = image_mask.any(dim=1)
        elif int(image_mask.shape[1]) == 1 and int(image_mask.shape[0]) == seq_len:
            out = image_mask[:, 0]
        else:
            raise RuntimeError(
                f"GeoPACT {where}: unsupported image_mask shape {tuple(image_mask.shape)} "
                f"for seq_len={seq_len}"
            )
    elif image_mask.dim() >= 3:
        if int(image_mask.shape[0]) == 1 and int(image_mask.shape[1]) == seq_len:
            out = image_mask[0].any(dim=tuple(range(1, image_mask.dim() - 1)))
        elif int(image_mask.shape[0]) == seq_len:
            out = image_mask.any(dim=tuple(range(1, image_mask.dim())))
        else:
            raise RuntimeError(
                f"GeoPACT {where}: unsupported image_mask shape {tuple(image_mask.shape)} "
                f"for seq_len={seq_len}"
            )
    else:
        raise RuntimeError(f"GeoPACT {where}: scalar image_mask is invalid")

    out = out.bool()
    if int(out.shape[0]) != seq_len:
        raise RuntimeError(
            f"GeoPACT {where}: normalized image_mask length {out.shape[0]} != seq_len {seq_len}"
        )
    return out


def _extend_mrope_for_inserted_tokens(
    compact_pos: torch.Tensor,
    compact_ids: torch.Tensor,
    ins: int,
    n_c: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extend MRoPE positions for SBR coord tokens and refresh rope_deltas."""
    new_pos = int(compact_pos.max()) + 1
    extra = torch.arange(
        new_pos,
        new_pos + n_c,
        device=compact_pos.device,
        dtype=compact_pos.dtype,
    )
    extra3 = extra.view(1, 1, -1).expand(3, compact_pos.shape[1], -1)
    compact_pos = torch.cat([compact_pos[:, :, :ins], extra3, compact_pos[:, :, ins:]], dim=2)
    batch = compact_ids.shape[0]
    seq_row = compact_ids.shape[1]
    deltas = [int(compact_pos[:, b].max().item()) + 1 - seq_row for b in range(batch)]
    compact_rope = torch.tensor(
        deltas,
        device=compact_pos.device,
        dtype=torch.long,
    ).unsqueeze(1)
    return compact_pos, compact_rope


def _compact_sequence(
    input_ids,
    inputs_embeds,
    image_embeds,
    image_mask,
    attention_mask,
    image_grid_thw,
    keep_token_set: set,
    seq_len: int,
    hidden: int,
    coord_text: Optional[str],
    tokenizer,
    embed_fn,
    image_tok_id: int,
    policy: str,
    position_ids=None,
    rope_deltas=None,
    model=None,
    *,
    video_grid_thw=None,
):
    """Scatter image_embeds, build compact sequence, optionally inject SBR coords."""
    if input_ids.shape[0] != 1:
        raise RuntimeError(
            f"GeoPACT source-compact currently requires batch_size=1, got {input_ids.shape[0]}"
        )
    dev = input_ids.device
    inputs_embeds = inputs_embeds.to(dev)
    image_embeds = image_embeds.to(device=dev, dtype=inputs_embeds.dtype)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    image_mask = _normalize_image_mask_1d(image_mask, seq_len, where="_compact_sequence").to(dev)

    ledger = getattr(model, "_geopact_step_timing_ledger", None) if model is not None else None
    if model is not None:
        # Correction contexts are prefill-shape dependent. Clear stale contexts
        # before rebuilding them after all compact-sequence splices are done.
        try:
            from lmms_eval.models.geopact_correction import clear_correction_ctx

            clear_correction_ctx(model)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_vsas import clear_vsas_ctx

            clear_vsas_ctx(model)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_sspc import clear_sspc_attach_ctx

            clear_sspc_attach_ctx(model)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_gsc import clear_gsc_ctx

            clear_gsc_ctx(model)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_trp import clear_trp_ctx

            clear_trp_ctx(model)
        except Exception:
            pass
        try:
            from lmms_eval.models.geopact_parb import clear_parb_ctx

            clear_parb_ctx(model)
        except Exception:
            pass
    compact_t0 = time.perf_counter()

    img_seq_positions = image_mask.nonzero(as_tuple=False).flatten()

    try:
        from lmms_eval.models.geopact_sspc import apply_sspc_embedding_zafd_for_model, sspc_enabled, sspc_zafd_enabled

        if model is not None and sspc_enabled() and sspc_zafd_enabled():
            prune_ctx_zafd = getattr(model, "_geopact_compact_prune_ctx", None)
            image_embeds = apply_sspc_embedding_zafd_for_model(
                image_embeds,
                model,
                image_grid_thw,
                prune_ctx_zafd,
                reject_grid_mismatch=True,
            )
    except Exception as exc:
        from lmms_eval.models.geopact_sspc import sspc_strict_enabled

        if sspc_strict_enabled():
            raise
        _log.warning("GeoPACT SSPC ZAFD failed (non-fatal): %s", exc)

    keep_mask = _build_keep_mask(image_mask, img_seq_positions, keep_token_set)
    compact_ids, compact_embs, compact_attn = _gather_compact_prefill_tensors(
        input_ids,
        inputs_embeds,
        image_embeds,
        attention_mask,
        keep_mask,
        keep_token_set,
        img_seq_positions,
        image_tok_id,
    )

    if ledger is not None:
        ledger.add_ms("compact_gather_ms", (time.perf_counter() - compact_t0) * 1000.0)

    # Preserve original MRoPE positions for kept tokens (default).
    # Recomputing RoPE from full image_grid_thw on a shorter sequence breaks
    # coordinate accuracy (TM may hold, EM collapses). Use seqpos only if requested.
    compact_pos   = None
    compact_rope  = None
    force_rope_recompute = _policy_has(policy, "seqpos")
    mrope_t0 = time.perf_counter()
    if not force_rope_recompute:
        if position_ids is None and model is not None:
            position_ids, rope_deltas = _precompute_full_mrope(
                model,
                input_ids,
                image_grid_thw,
                attention_mask,
                video_grid_thw=video_grid_thw,
            )
        if position_ids is None:
            raise RuntimeError(
                "GeoPACT origpos: full position_ids missing before compact subset"
            )
        compact_pos, compact_rope = _subset_origpos_mrope_cached(
            model, position_ids, keep_mask, compact_ids, compact_attn
        )
    if ledger is not None and not force_rope_recompute:
        ledger.add_ms("origpos_mrope_ms", (time.perf_counter() - mrope_t0) * 1000.0)
        if compact_pos is not None:
            ledger.set_audit(mrope_path="origpos")
    elif ledger is not None and force_rope_recompute:
        ledger.set_audit(mrope_path="seqpos")

    def _any_spatial_consumer_enabled() -> bool:
        """flat_to_compact is only read by GSC/TRP/PARB/spatial-RPB/CDCT."""
        from lmms_eval.models.geopact_gsc import gsc_enabled
        from lmms_eval.models.geopact_parb import parb_enabled
        from lmms_eval.models.geopact_spatial_rpb import (
            cdct_residue_enabled,
            spatial_rpb_enabled,
        )
        from lmms_eval.models.geopact_trp import trp_enabled

        return (
            gsc_enabled()
            or trp_enabled()
            or parb_enabled()
            or spatial_rpb_enabled()
            or cdct_residue_enabled()
        )

    keep_seq_t = _visual_keep_in_sequence_order_tensor(keep_token_set, img_seq_positions)
    spatial_audit: dict = {}
    prune_ctx = getattr(model, "_geopact_compact_prune_ctx", None) if model is not None else None
    if prune_ctx is not None and _any_spatial_consumer_enabled():
        from lmms_eval.models.geopact_spatial_rpb import compact_visual_indices

        # Single D2H copy each; the previous per-element .item() loop cost
        # ~2 GPU syncs per kept token (~24ms/step at c50) and its output was
        # discarded whenever every spatial consumer below is disabled.
        vis_idx = compact_visual_indices(compact_ids, image_tok_id)
        keep_seq_list = keep_seq_t.tolist()
        vis_idx_list = vis_idx.tolist()
        flat_to_compact = {
            int(flat_i): int(vis_idx_list[ord_i])
            for ord_i, flat_i in enumerate(keep_seq_list[: len(vis_idx_list)])
        }
        ctx_p = dict(prune_ctx)
        ctx_p["flat_to_compact"] = flat_to_compact
        grid_thw = getattr(model, "_geopact_last_image_grid_thw", None) if model is not None else None

        from lmms_eval.models.geopact_gsc import attach_gsc_ctx, gsc_enabled

        if gsc_enabled():
            gsc_ctx = attach_gsc_ctx(
                model,
                compact_ids=compact_ids,
                compact_pos=compact_pos,
                image_tok_id=image_tok_id,
                prune_ctx=ctx_p,
                visual_keep_flat=keep_seq_t,
                image_grid_thw=grid_thw,
            )
            if gsc_ctx is not None:
                spatial_audit = dict(gsc_ctx.audit)
                if model is not None:
                    model._geopact_spatial_audit = spatial_audit
        else:
            from lmms_eval.models.geopact_trp import attach_trp_ctx, trp_enabled

            if trp_enabled():
                trp_ctx = attach_trp_ctx(
                    model,
                    compact_ids=compact_ids,
                    compact_pos=compact_pos,
                    image_tok_id=image_tok_id,
                    prune_ctx=ctx_p,
                    visual_keep_flat=keep_seq_t,
                )
                if trp_ctx is not None:
                    spatial_audit = dict(trp_ctx.audit)
                    if model is not None:
                        model._geopact_spatial_audit = spatial_audit
            else:
                from lmms_eval.models.geopact_parb import attach_parb_ctx, parb_enabled

                if parb_enabled():
                    parb_ctx = attach_parb_ctx(
                        model,
                        compact_ids=compact_ids,
                        compact_pos=compact_pos,
                        image_tok_id=image_tok_id,
                        prune_ctx=ctx_p,
                        visual_keep_flat=keep_seq_t,
                    )
                    if parb_ctx is not None:
                        spatial_audit = dict(parb_ctx.audit)
                        if model is not None:
                            model._geopact_spatial_audit = spatial_audit
                else:
                    from lmms_eval.models.geopact_spatial_rpb import (
                        apply_spatial_enhancements,
                        cdct_residue_enabled,
                        spatial_rpb_enabled,
                    )

                    if spatial_rpb_enabled() or cdct_residue_enabled():
                        ctx = dict(prune_ctx)
                        ctx["flat_to_compact"] = flat_to_compact
                        compact_embs, spatial_audit = apply_spatial_enhancements(
                            compact_embs,
                            compact_pos,
                            compact_ids,
                            image_tok_id,
                            prune_ctx=ctx,
                            visual_keep_flat=keep_seq_t,
                        )
                        if model is not None:
                            model._geopact_spatial_audit = spatial_audit

    # SBR: inject coord text immediately after last surviving image pad
    from lmms_eval.models.geopact_coord_injection.record import (
        CoordInjectionMode,
        CoordInjectionRecord,
    )
    from lmms_eval.models.geopact_coord_injection.slot import (
        last_vision_index,
        resolve_insert_pos,
        resolve_slot_from_env,
    )

    coord_metadata_only = _coord_metadata_only()
    coord_metadata_chars = int(len(coord_text or ""))
    coord_spliced_tokens = 0
    coord_ephemeral_tokens = 0
    coord_insert_pos = -1
    coord_ephemeral_flag = False
    coord_splice_error: Optional[str] = None
    coord_record = CoordInjectionRecord.empty()

    gaec_done = False
    gaec_ctx = getattr(model, "_geopact_gaec_ctx", None) if model is not None else None
    if gaec_ctx and compact_pos is not None and position_ids is not None:
        from lmms_eval.models.geopact_coord_injection.gaec import (
            build_gaec_carriers,
            gaec_enabled,
            splice_gaec_into_compact,
        )

        if gaec_enabled():
            ephemeral = _coord_ephemeral_enabled()
            coord_ephemeral_flag = ephemeral
            slot = resolve_slot_from_env(policy)
            try:
                payload = build_gaec_carriers(
                    compact_embs=compact_embs,
                    compact_ids=compact_ids,
                    compact_pos=compact_pos,
                    full_position_ids=position_ids,
                    keep_mask=keep_mask,
                    image_tok_id=image_tok_id,
                    gaec_ctx=gaec_ctx,
                    carrier_token_id=_gaec_carrier_token_id(model, tokenizer),
                )
                if payload is not None:
                    seq_len_compact = int(compact_ids.shape[1])
                    gen_tail_len = int(getattr(model, "_geopact_gen_prompt_tail_len", 0) or 0)
                    lv = last_vision_index(compact_ids[0], image_tok_id)
                    ins = resolve_insert_pos(
                        slot,
                        compact_seq_len=seq_len_compact,
                        gen_tail_len=gen_tail_len,
                        last_vision_pos=lv,
                        ephemeral_mode=ephemeral,
                    )
                    n_c = int(payload.carrier_embs.shape[1])
                    coord_insert_pos = int(ins)
                    coord_spliced_tokens = n_c
                    coord_ephemeral_tokens = n_c if ephemeral else 0
                    (
                        compact_ids,
                        compact_embs,
                        compact_attn,
                        compact_pos,
                        compact_rope,
                    ) = splice_gaec_into_compact(
                        compact_ids,
                        compact_embs,
                        compact_attn,
                        compact_pos,
                        compact_rope,
                        ins,
                        payload,
                    )
                    coord_record = payload.record
                    coord_record.insert_pos = int(ins)
                    coord_record.gen_tail_len = gen_tail_len
                    coord_record.ephemeral = ephemeral
                    coord_record.spliced = True
                    if model is not None:
                        model._geopact_last_coord_injection = coord_record
                        gaec_audit = dict(getattr(model, "_geopact_spatial_audit", None) or {})
                        gaec_audit.update(
                            {
                                "gaec": True,
                                "gaec_v2": coord_record.format_id == "gaec_v2",
                                "gaec_carriers": n_c,
                                "gaec_anchor_mius": list(payload.anchor_miu_indices),
                            }
                        )
                        model._geopact_spatial_audit = gaec_audit
                    gaec_done = True
            except Exception as exc:
                _log.warning("GeoPACT GAEC injection failed (non-fatal): %s", exc)
                coord_splice_error = str(exc)
                if model is not None:
                    model._geopact_last_coord_injection = CoordInjectionRecord.empty()

    if not gaec_done and coord_text and coord_metadata_only:
        try:
            from lmms_eval.models.geopact_coord_appendix import consume_coord_build_meta

            meta = consume_coord_build_meta()
        except Exception:
            meta = {}
        coord_record = CoordInjectionRecord(
            text=coord_text,
            mode=CoordInjectionMode.METADATA_ONLY,
            format_id=str(meta.get("coord_format", "") or ""),
            anchor_count=int(meta.get("coord_anchor_count", 0) or 0),
            ephemeral=False,
            spliced=False,
        )
    elif not gaec_done and coord_text and tokenizer is not None and embed_fn is not None:
        ephemeral = _coord_ephemeral_enabled()
        coord_ephemeral_flag = ephemeral
        slot = resolve_slot_from_env(policy)
        try:
            cid = tokenizer.encode(coord_text, add_special_tokens=False)
            cid_t = torch.tensor(cid, dtype=compact_ids.dtype, device=compact_ids.device)
            with torch.no_grad():
                cemb = embed_fn(cid_t).unsqueeze(0).to(
                    device=compact_embs.device, dtype=compact_embs.dtype
                )

            seq_len_compact = int(compact_ids.shape[1])
            gen_tail_len = int(getattr(model, "_geopact_gen_prompt_tail_len", 0) or 0)
            lv = last_vision_index(compact_ids[0], image_tok_id)
            ins = resolve_insert_pos(
                slot,
                compact_seq_len=seq_len_compact,
                gen_tail_len=gen_tail_len,
                last_vision_pos=lv,
                ephemeral_mode=ephemeral,
            )
            n_c = int(cemb.shape[1])
            coord_insert_pos = int(ins)
            coord_spliced_tokens = n_c
            coord_ephemeral_tokens = n_c if ephemeral else 0
            compact_ids = torch.cat(
                [compact_ids[:, :ins], cid_t.unsqueeze(0), compact_ids[:, ins:]], dim=1
            )
            compact_embs = torch.cat(
                [compact_embs[:, :ins], cemb, compact_embs[:, ins:]], dim=1
            )
            if compact_attn is not None:
                ones = torch.ones(1, n_c, dtype=compact_attn.dtype, device=compact_attn.device)
                compact_attn = torch.cat(
                    [compact_attn[:, :ins], ones, compact_attn[:, ins:]], dim=1
                )
            if compact_pos is not None:
                compact_pos, compact_rope = _extend_mrope_for_inserted_tokens(
                    compact_pos, compact_ids, ins, n_c
                )
            try:
                from lmms_eval.models.geopact_coord_appendix import consume_coord_build_meta

                meta = consume_coord_build_meta()
        except Exception:
                meta = {}
            mode = (
                CoordInjectionMode.EPHEMERAL
                if ephemeral
                else CoordInjectionMode.PERSISTENT
            )
            coord_record = CoordInjectionRecord(
                text=coord_text,
                token_count=n_c,
                insert_pos=ins,
                mode=mode,
                format_id=str(meta.get("coord_format", "") or ""),
                anchor_count=int(meta.get("coord_anchor_count", 0) or 0),
                ephemeral=ephemeral,
                spliced=True,
                gen_tail_len=gen_tail_len,
            )
            if model is not None:
                model._geopact_last_coord_injection = coord_record
        except Exception as exc:
            _log.warning("GeoPACT SBR coord injection failed (non-fatal): %s", exc)
            coord_spliced_tokens = 0
            coord_ephemeral_tokens = 0
            coord_insert_pos = -1
            coord_ephemeral_flag = False
            coord_splice_error = str(exc)
            if model is not None:
                model._geopact_ephemeral_coord_text = None
                model._geopact_last_coord_injection = CoordInjectionRecord.empty()

    # Keep original image_grid_thw — changing it causes get_rope_index to crash
    # when the compact token count is too small for the spatial merge.
    # The model will recompute RoPE positions using original grid dims which is
    # acceptable since kept tokens preserve their original spatial coordinates.

    compact_image_mask = None
    if image_tok_id >= 0:
        # Match get_placeholder_mask layout (batch, seq, 1) so forward's
        # `image_mask[..., 0]` yields (batch, seq), not a length-1 vector.
        compact_image_mask = (compact_ids[0] == image_tok_id).view(1, -1, 1)

    visual_keep_flat_indices = keep_seq_t if keep_seq_t.numel() > 0 else None

    # Attach attention-time correction contexts only after GAEC / coord splices.
    # Otherwise ctx.seq_len can differ from the actual prefill query length and
    # should_apply_* silently skips Module C.
    def _any_ctx_consumer_enabled() -> bool:
        """This flat_to_compact feeds SSPC/VSAS/correction/GSC/TRP/PARB only."""
        from lmms_eval.models.geopact_correction import correction_enabled
        from lmms_eval.models.geopact_gsc import gsc_enabled
        from lmms_eval.models.geopact_parb import parb_enabled
        from lmms_eval.models.geopact_sspc import sspc_enabled
        from lmms_eval.models.geopact_trp import trp_enabled
        from lmms_eval.models.geopact_vsas import vsas_enabled

        return (
            sspc_enabled()
            or vsas_enabled()
            or correction_enabled()
            or gsc_enabled()
            or trp_enabled()
            or parb_enabled()
        )

    if prune_ctx is not None and _any_ctx_consumer_enabled():
        from lmms_eval.models.geopact_spatial_rpb import compact_visual_indices

        vis_idx = compact_visual_indices(compact_ids, image_tok_id)
        # Single D2H copy each (the per-element .item() loop costs ~12ms/step
        # at c50 in pure GPU syncs).
        keep_seq_list = keep_seq_t.tolist()
        vis_idx_list = vis_idx.tolist()
        frame_start = int((prune_ctx or {}).get("frame_start") or 0)
        flat_to_compact = {}
        for ord_i, flat_i in enumerate(keep_seq_list[: len(vis_idx_list)]):
            ci = int(vis_idx_list[ord_i])
            flat_to_compact[int(flat_i)] = ci
            flat_to_compact[int(flat_i) - frame_start] = ci
        ctx_p = dict(prune_ctx)
        ctx_p["flat_to_compact"] = flat_to_compact
        grid_thw = getattr(model, "_geopact_last_image_grid_thw", None) if model is not None else None

        try:
            from lmms_eval.models.geopact_sspc import attach_sspc_ctx, sspc_enabled

            if sspc_enabled():
                sspc_ctx = attach_sspc_ctx(model)
                if sspc_ctx is not None:
                    spatial_audit = dict(sspc_ctx.audit)
                    if model is not None:
                        model._geopact_spatial_audit = spatial_audit
            from lmms_eval.models.geopact_vsas import (
                attach_vsas_ctx,
                splice_vsas_into_compact,
                vsas_enabled,
                vsas_mode,
            )

            if not sspc_enabled() and vsas_enabled():
                vsas_ctx = attach_vsas_ctx(
                    model,
                    compact_ids=compact_ids,
                    compact_pos=compact_pos,
                    image_tok_id=image_tok_id,
                    prune_ctx=ctx_p,
                )
                if vsas_ctx is not None:
                    spatial_audit = dict(vsas_ctx.audit)
                    if model is not None:
                        model._geopact_spatial_audit = spatial_audit
                    if vsas_mode() == "seq" and compact_pos is not None:
                        (
                            compact_ids,
                            compact_embs,
                            compact_attn,
                            compact_pos,
                            compact_rope,
                            vsas_ctx,
                        ) = splice_vsas_into_compact(
                            compact_ids,
                            compact_embs,
                            compact_attn,
                            compact_pos,
                            compact_rope,
                            vsas_ctx,
                            image_tok_id,
                        )
                        if model is not None:
                            lang = getattr(model, "language_model", None)
                            if lang is not None:
                                lang._geopact_vsas_ctx = vsas_ctx
                            model._geopact_vsas_ctx = vsas_ctx
                            model._geopact_spatial_audit = dict(vsas_ctx.audit)
            elif not sspc_enabled() and not vsas_enabled():
                from lmms_eval.models.geopact_correction import attach_correction_ctx, correction_enabled

                if correction_enabled():
                    correction_ctx = attach_correction_ctx(
                        model,
                        compact_ids=compact_ids,
                        compact_pos=compact_pos,
                        image_tok_id=image_tok_id,
                        prune_ctx=ctx_p,
                        visual_keep_flat=keep_seq_t,
                        image_grid_thw=grid_thw,
                    )
                    if correction_ctx is not None:
                        spatial_audit = dict(correction_ctx.audit)
                        if model is not None:
                            model._geopact_spatial_audit = spatial_audit
                else:
                    from lmms_eval.models.geopact_gsc import attach_gsc_ctx, gsc_enabled

                    if gsc_enabled():
                        gsc_ctx = attach_gsc_ctx(
                            model,
                            compact_ids=compact_ids,
                            compact_pos=compact_pos,
                            image_tok_id=image_tok_id,
                            prune_ctx=ctx_p,
                            visual_keep_flat=keep_seq_t,
                            image_grid_thw=grid_thw,
                        )
                        if gsc_ctx is not None:
                            spatial_audit = dict(gsc_ctx.audit)
                            if model is not None:
                                model._geopact_spatial_audit = spatial_audit
                    else:
                        from lmms_eval.models.geopact_trp import attach_trp_ctx, trp_enabled

                        if trp_enabled():
                            trp_ctx = attach_trp_ctx(
                                model,
                                compact_ids=compact_ids,
                                compact_pos=compact_pos,
                                image_tok_id=image_tok_id,
                                prune_ctx=ctx_p,
                                visual_keep_flat=keep_seq_t,
                            )
                            if trp_ctx is not None:
                                spatial_audit = dict(trp_ctx.audit)
                                if model is not None:
                                    model._geopact_spatial_audit = spatial_audit
                        else:
                            from lmms_eval.models.geopact_parb import attach_parb_ctx, parb_enabled

                            if parb_enabled():
                                parb_ctx = attach_parb_ctx(
                                    model,
                                    compact_ids=compact_ids,
                                    compact_pos=compact_pos,
                                    image_tok_id=image_tok_id,
                                    prune_ctx=ctx_p,
                                    visual_keep_flat=keep_seq_t,
                                )
                                if parb_ctx is not None:
                                    spatial_audit = dict(parb_ctx.audit)
                                    if model is not None:
                                        model._geopact_spatial_audit = spatial_audit
        except Exception as exc:
            _log.warning("GeoPACT correction ctx attach failed (non-fatal): %s", exc)

    coord_meta = {}
    if coord_text:
        try:
            from lmms_eval.models.geopact_coord_appendix import consume_coord_build_meta

            coord_meta = consume_coord_build_meta()
        except Exception:
            coord_meta = {}
    if coord_record.anchor_count <= 0 and coord_meta:
        coord_record.anchor_count = int(coord_meta.get("coord_anchor_count", 0) or 0)
    if not coord_record.format_id and coord_meta:
        coord_record.format_id = str(coord_meta.get("coord_format", "") or "")
    if coord_splice_error:
        coord_record.splice_error = coord_splice_error

    inj_stats = coord_record.to_stats_dict()
    inj_stats["text"] = coord_record.text[:120] if coord_record.text else ""

    compact_sspc_stats: dict = {}
    try:
        from lmms_eval.models.geopact_sspc import sspc_enabled, sspc_stats_for_compact

        if model is not None and sspc_enabled():
            compact_sspc_stats = sspc_stats_for_compact(model)
    except Exception:
        compact_sspc_stats = {}

    return {
        "input_ids":       compact_ids,
        "inputs_embeds":   compact_embs,
        "attention_mask":  compact_attn,
        "image_grid_thw":  image_grid_thw,   # original, not modified
        "image_mask":      compact_image_mask,
        "position_ids":    compact_pos,
        "rope_deltas":     compact_rope,
        "force_rope_recompute": force_rope_recompute,
        "visual_keep_flat_indices": visual_keep_flat_indices,
        "mrope_source_input_ids": input_ids.detach().clone(),
        "mrope_keep_mask": keep_mask.detach().clone(),
        "geopact_compact_stats": {
            "orig_seq_len": int(seq_len),
            "compact_seq_len": int(compact_ids.shape[1]),
            "n_visual_total": int(image_embeds.shape[0]),
            "n_visual_kept": int(keep_seq_t.numel()),
            "compact_gather_backend": "fused_index_select",
            "origpos": not force_rope_recompute,
            "has_position_ids": compact_pos is not None,
            "coord_metadata_only": coord_metadata_only,
            "coord_metadata_chars": coord_metadata_chars if coord_text else 0,
            "coord_metadata_lines": len([ln for ln in str(coord_text or "").splitlines() if ln.strip()]),
            "coord_ephemeral": bool(coord_ephemeral_flag and coord_spliced_tokens > 0),
            "coord_ephemeral_tokens": int(coord_ephemeral_tokens),
            "coord_spliced_tokens": int(coord_spliced_tokens),
            "coord_ephemeral_chars": coord_metadata_chars if coord_spliced_tokens > 0 else 0,
            "coord_insert_pos": int(coord_insert_pos),
            "coord_format": str(coord_record.format_id or coord_meta.get("coord_format", "") or ""),
            "coord_schema_contract": bool(
                coord_record.format_id == "sdb"
                or coord_meta.get("coord_schema_contract", False)
            ),
            "coord_anchor_count": int(
                coord_record.anchor_count or coord_meta.get("coord_anchor_count", 0) or 0
            ),
            "coord_splice_error": coord_splice_error,
            "coord_injection": inj_stats,
            "spatial_rpb": bool(spatial_audit.get("spatial_rpb")),
            "cdct_residue": bool(spatial_audit.get("cdct_residue")),
            "es_rpb_lambda": spatial_audit.get("es_rpb_lambda"),
            "cdct_residue_alpha": spatial_audit.get("cdct_residue_alpha"),
            **compact_sspc_stats,
        },
    }


def _keep_token_set_with_compact_cache_hits(
    keep_token_set: set,
    token_ranges,
    compact_hits: set,
    model,
) -> set:
    """Force cache-hit frames to use the cached post-prune token order."""
    if not compact_hits:
        return keep_token_set

    image_paths = getattr(model, "_geopact_image_paths", None) or []
    out = set()
    for frame_idx, (start, end) in enumerate(token_ranges):
        if frame_idx in compact_hits:
            img_path = image_paths[frame_idx] if frame_idx < len(image_paths) else ""
            try:
                from lmms_eval.models.geopact_vision_cache import (
                    _local_keep_seq_from_entry,
                    get_compact_frame_cache,
                )

                cached = get_compact_frame_cache(model, img_path)
                if cached is not None:
                    out.update(start + int(idx) for idx in _local_keep_seq_from_entry(cached))
                    continue
            except Exception as exc:
                _log.warning("GeoPACT compact cache keep reuse failed: %s", exc)
        out.update(int(g) for g in keep_token_set if start <= int(g) < end)
    return out


def _subset_compact_tensors_by_mask(
    compact: dict,
    keep_mask: torch.Tensor,
) -> dict:
    """Drop sequence positions from a post-image-compact dict (e.g. video prune)."""
    device = compact["input_ids"].device
    keep_mask = keep_mask.to(device=device, dtype=torch.bool)
    seq_len = int(compact["input_ids"].shape[1])
    if int(keep_mask.shape[0]) != seq_len:
        raise RuntimeError(
            f"GeoPACT subset: keep_mask len {keep_mask.shape[0]} != seq_len {seq_len}"
        )

    compact_ids = compact["input_ids"][:, keep_mask]
    compact_embs = compact["inputs_embeds"][:, keep_mask]
    compact_attn = compact.get("attention_mask")
    if compact_attn is not None:
        compact_attn = compact_attn[:, keep_mask]

    compact_pos = compact.get("position_ids")
    compact_rope = compact.get("rope_deltas")
    if compact_pos is not None:
        compact_pos, compact_rope = _subset_origpos_mrope(
            compact_pos, keep_mask, compact_ids, compact_attn
        )

    out = dict(compact)
    out["input_ids"] = compact_ids
    out["inputs_embeds"] = compact_embs
    if compact_attn is not None:
        out["attention_mask"] = compact_attn
    if compact_pos is not None:
        out["position_ids"] = compact_pos
        out["rope_deltas"] = compact_rope
    return out


def video_prune_active(inner, *, hook_active: bool) -> bool:
    """True when current-step video tokens should be pruned (never cached)."""
    flag = os.environ.get("GEOPACT_PRUNE_VIDEO", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if not hook_active:
        return False
    ratio = float(getattr(inner, "_geopact_prune_ratio", 1.0) or 1.0)
    return ratio < 0.999


def _merge_image_video_deepstack(
    deepstack_image: Optional[list],
    deepstack_video: Optional[list],
    image_mask_1d: torch.Tensor,
    video_mask_1d: torch.Tensor,
) -> Optional[list]:
    """Joint deepstack rows in sequence order (image | video visual positions)."""
    if deepstack_image is None and deepstack_video is None:
        return None
    visual_pos = image_mask_1d | video_mask_1d
    n_vis = int(visual_pos.sum().item())
    if n_vis == 0:
        return None
    img_joint = image_mask_1d[visual_pos]
    vid_joint = video_mask_1d[visual_pos]
    n_img = int(img_joint.sum().item())
    n_vid = int(vid_joint.sum().item())
    if deepstack_image is not None and n_img > 0 and int(deepstack_image[0].shape[0]) != n_img:
        raise RuntimeError(
            f"GeoPACT video deepstack: image rows {deepstack_image[0].shape[0]} != {n_img}"
        )
    if deepstack_video is not None and n_vid > 0 and int(deepstack_video[0].shape[0]) != n_vid:
        raise RuntimeError(
            f"GeoPACT video deepstack: video rows {deepstack_video[0].shape[0]} != {n_vid}"
        )
    n_layers = len(deepstack_image or deepstack_video or [])
    out = []
    for li in range(n_layers):
        img_layer = (deepstack_image or [None] * n_layers)[li]
        vid_layer = (deepstack_video or [None] * n_layers)[li]
        ref = img_layer if img_layer is not None else vid_layer
        joint = ref.new_zeros(n_vis, ref.shape[-1])
        if img_layer is not None and n_img > 0:
            joint[img_joint, :] = img_layer
        if vid_layer is not None and n_vid > 0:
            joint[vid_joint, :] = vid_layer
        out.append(joint)
    return out


@torch.inference_mode()
def apply_video_to_compact_prefill(
    inner,
    prefill: dict,
    inputs,
    *,
    hook_active: bool,
) -> dict:
    """Encode current-step video, optional random prune (no cache), scatter into compact prefill."""
    if os.environ.get("GEOPACT_SCREENSHOT_ONLY_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return prefill
    pixel_values_videos = (
        inputs.get("pixel_values_videos")
        if hasattr(inputs, "get")
        else getattr(inputs, "pixel_values_videos", None)
    )
    video_grid_thw = (
        inputs.get("video_grid_thw")
        if hasattr(inputs, "get")
        else getattr(inputs, "video_grid_thw", None)
    )
    if pixel_values_videos is None or video_grid_thw is None:
        return prefill

    get_video = getattr(inner, "get_video_features", None)
    if get_video is None:
        return prefill

    video_feat_list, deepstack_video_list = get_video(pixel_values_videos, video_grid_thw)
    video_embeds = torch.cat(video_feat_list, dim=0).to(
        prefill["inputs_embeds"].device, prefill["inputs_embeds"].dtype
    )

    input_ids = prefill["input_ids"]
    inputs_embeds = prefill["inputs_embeds"]
    video_tok_id = int(getattr(inner.config, "video_token_id", -1) or -1)
    if video_tok_id < 0:
        return prefill

    vid_positions = (input_ids[0] == video_tok_id).nonzero(as_tuple=False).flatten().tolist()
    n_vid = len(vid_positions)
    if n_vid == 0:
        return prefill
    if int(video_embeds.shape[0]) != n_vid:
        raise RuntimeError(
            f"GeoPACT video token mismatch: placeholders={n_vid} embeds={video_embeds.shape[0]}"
        )

    ratio = float(getattr(inner, "_geopact_prune_ratio", 1.0) or 1.0)
    prune = video_prune_active(inner, hook_active=hook_active)
    keep_local = list(range(n_vid))
    if prune:
        keep_k = max(1, int(round(n_vid * ratio)))
        episode_key = str(getattr(inner, "_geopact_episode_key", "") or "")
        rng = _sticky_rng(episode_key, "__geopact_current_video__")
        keep_local = sorted(rng.sample(range(n_vid), min(keep_k, n_vid)))
        drop_positions = [vid_positions[i] for i in range(n_vid) if i not in keep_local]
        if drop_positions:
            seq_len = int(input_ids.shape[1])
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
            keep_mask[torch.tensor(drop_positions, device=input_ids.device, dtype=torch.long)] = False
            prefill = _subset_compact_tensors_by_mask(prefill, keep_mask)
            input_ids = prefill["input_ids"]
            inputs_embeds = prefill["inputs_embeds"]
            vid_positions = (input_ids[0] == video_tok_id).nonzero(as_tuple=False).flatten().tolist()
            video_embeds = video_embeds.index_select(
                0, torch.tensor(keep_local, device=video_embeds.device, dtype=torch.long)
            )
            if deepstack_video_list:
                idx = torch.tensor(keep_local, device=deepstack_video_list[0].device, dtype=torch.long)
                deepstack_video_list = [layer.index_select(0, idx) for layer in deepstack_video_list]

    _, video_mask = inner.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
    )
    video_mask_1d = _normalize_image_mask_1d(video_mask, int(input_ids.shape[1]), where="video_scatter")
    hidden = int(inputs_embeds.shape[-1])
    vid_mask_exp = video_mask_1d.unsqueeze(0).unsqueeze(-1).expand(1, int(input_ids.shape[1]), hidden)
    inputs_embeds = inputs_embeds.masked_scatter(
        vid_mask_exp, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    )
    prefill["inputs_embeds"] = inputs_embeds

    image_tok_id = int(getattr(inner.config, "image_token_id", -1) or -1)
    image_mask_1d = torch.zeros(int(input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
    if image_tok_id >= 0:
        image_mask_1d = image_mask_1d | (input_ids[0] == image_tok_id)
    image_mask_1d = image_mask_1d | video_mask_1d

    ds_image = prefill.get("deepstack_image_embeds")
    ds_video = deepstack_video_list if deepstack_video_list else None
    img_only = torch.zeros_like(image_mask_1d)
    if image_tok_id >= 0:
        img_only = img_only | (input_ids[0] == image_tok_id)
    joint_ds = _merge_image_video_deepstack(ds_image, ds_video, img_only, video_mask_1d)
    if joint_ds is not None:
        prefill["deepstack_image_embeds"] = joint_ds
    prefill["image_mask"] = image_mask_1d.view(1, -1, 1)
    prefill["video_grid_thw"] = video_grid_thw

    stats = getattr(inner, "_geopact_last_compact_stats", None) or {}
    if not isinstance(stats, dict):
        stats = {}
    stats["video_tokens_total"] = int(n_vid)
    stats["video_tokens_kept"] = int(video_embeds.shape[0])
    stats["video_pruned"] = bool(prune)
    inner._geopact_last_compact_stats = stats
    if "geopact_compact_stats" in prefill:
        prefill["geopact_compact_stats"].update(
            {
                "video_tokens_total": stats["video_tokens_total"],
                "video_tokens_kept": stats["video_tokens_kept"],
                "video_pruned": stats["video_pruned"],
            }
        )
    return prefill


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_geopact_hook(
    visual_prune_ratio: float,
    visual_prune_policy: str,
    miu_detector: str = "fusion_fast",
    visual_prune_scope: str = "image",
):
    """
    Return a closure that implements GCC + HES + SBR.

    Parameters
    ----------
    visual_prune_ratio : float
        Fraction of image tokens to keep (0 < r < 1).
    visual_prune_policy : str
        Policy string controlling detector, HES variant, position encoding,
        SBR placement, and ablation drops.
    miu_detector : str
        Default detector backend if not overridden by policy string.
        ``fusion_fast`` (default), ``canny_morph``, ``blob``, or ``omniparser``.
    """
    ratio    = float(visual_prune_ratio)
    policy   = str(visual_prune_policy or "").strip()
    detector = _get_detector(policy, miu_detector)
    scope    = str(visual_prune_scope or "image").lower()

    def _finalize(model, result: dict) -> dict:
        stats = result.get("geopact_compact_stats")
        if stats is not None:
            cache_meta = getattr(model, "_geopact_last_module_a_cache_meta", None)
            if isinstance(cache_meta, dict):
                for key in ("module_a_cache_hit", "module_a_cache_key", "hits", "misses"):
                    if key in cache_meta:
                        stats[f"module_a_cache_{key}" if key in {"hits", "misses"} else key] = cache_meta[key]
            model._geopact_last_compact_stats = stats
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug("GeoPACT compact: %s", stats)
        return result

    def _hook(payload: dict):
        model          = payload["model"]
        model._geopact_prune_ratio = ratio
        model._geopact_prune_policy = policy
        input_ids      = payload["input_ids"]          # (1, seq_len)
        inputs_embeds  = payload["inputs_embeds"]      # (1, seq_len, hidden) text-only
        image_embeds   = payload["image_embeds"]       # (n_img, hidden)
        image_mask     = payload["image_mask"]         # (seq_len,) bool
        attention_mask = payload.get("attention_mask")
        image_grid_thw = payload.get("image_grid_thw")
        video_grid_thw = payload.get("video_grid_thw")
        position_ids   = payload.get("position_ids")
        rope_deltas    = payload.get("rope_deltas")

        if image_grid_thw is None or image_embeds is None:
            return None

        n_img   = image_embeds.shape[0]
        seq_len = input_ids.shape[1]
        hidden  = inputs_embeds.shape[-1]
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"GeoPACT source-compact currently requires batch_size=1, got {input_ids.shape[0]}"
            )
        image_mask = _normalize_image_mask_1d(image_mask, seq_len, where="hook")
        if n_img == 0:
            return None

        try:
            from lmms_eval.models.geopact_sspc import reset_sspc_step_state, sspc_enabled, sspc_strict_enabled

            if sspc_enabled():
                reset_sspc_step_state(model)
        except Exception:
            pass

        active_scope = str(getattr(model, "_geopact_prune_scope", scope) or scope).lower()
        try:
            from lmms_eval.models.geopact_sspc import sspc_enabled, sspc_strict_enabled

            if (
                sspc_enabled()
                and sspc_strict_enabled()
                and not _scope_is_sticky(active_scope)
            ):
                raise RuntimeError(
                    "GeoPACT SSPC requires visual_prune_scope=image_sticky, "
                    f"got {active_scope!r}"
                )
        except RuntimeError:
            raise
        except Exception:
            pass
        token_ranges = _flat_image_token_ranges(image_grid_thw, n_img)

        # ── image_sticky: per-frame prune with cross-step mask cache ─────
        if _scope_is_sticky(active_scope):
            episode_key = str(getattr(model, "_geopact_episode_key", "") or "")
            image_paths = getattr(model, "_geopact_image_paths", None) or []
            sticky_store = _get_sticky_store(model)
            prune_target = _prune_target(model)
            keep_token_set: set = set()
            coord_text = None

            compact_hits = getattr(model, "_geopact_compact_frame_cache_hits", None) or set()

            for range_idx, (start, end) in enumerate(token_ranges):
                n_frame = end - start
                is_current = range_idx == len(token_ranges) - 1
                img_path = image_paths[range_idx] if range_idx < len(image_paths) else ""
                cache_key = _sticky_mask_key(episode_key, img_path, model=model)
                frame_grid = _frame_grid(image_grid_thw, range_idx, len(token_ranges))

                if range_idx in compact_hits:
                    from lmms_eval.models.geopact_vision_cache import (
                        _local_keep_seq_from_entry,
                        get_compact_frame_cache,
                    )

                    cached_compact = get_compact_frame_cache(model, img_path)
                    if cached_compact is not None:
                        keep_token_set.update(
                            start + int(idx)
                            for idx in _local_keep_seq_from_entry(cached_compact)
                        )
                        continue

                # Published-baseline c/h mode (baseline_full multi-step arms):
                # the current frame is pruned at GEOPACT_PUBBASE_CURRENT while
                # history frames use the regular ratio; the current-frame mask
                # is NOT cached so the frame is re-pruned at the history budget
                # once it rolls into history (matches the (c, h) grid budget).
                pub_cur_ratio = None
                if _policy_has(
                    policy, "divprune", "dart", "cdpruner", "visionzip", "vispruner",
                    "prumerge", "trim", "visiontrim", "zooprune", "prunesid",
                ):
                    from lmms_eval.models.geopact_pub_baselines import (
                        pubbase_current_keep_ratio,
                    )

                    pub_cur_ratio = pubbase_current_keep_ratio()

                cached = sticky_store.get(cache_key)
                action = _sticky_frame_action(
                    is_current=is_current,
                    prune_target=prune_target,
                    has_cached_mask=cached is not None,
                    two_stage=pub_cur_ratio is not None,
                )

                if action == "prune":
                    pil_image = getattr(model, "_geopact_last_image", None)
                    if not is_current and _pub_hybrid_active(policy) and img_path:
                        # Hybrid arms consult Module A: a history frame being
                        # re-pruned at the h-budget must be detected on ITS OWN
                        # screenshot, not the current step's.
                        try:
                            from PIL import Image as _PILImage

                            pil_image = _PILImage.open(img_path).convert("RGB")
                        except Exception:
                            pil_image = None
                    rng = _sticky_rng(episode_key, img_path) if _policy_has(policy, "random") else None
                    frame_ratio = (
                        pub_cur_ratio
                        if (is_current and pub_cur_ratio is not None)
                        else ratio
                    )
                    frame_keep, _, _, frame_coord = _prune_single_frame(
                        start=start,
                        end=end,
                        image_embeds=image_embeds,
                        inputs_embeds=inputs_embeds,
                        image_mask=image_mask,
                        image_grid_thw=frame_grid,
                        ratio=frame_ratio,
                        policy=policy,
                        detector=detector,
                        pil_image=pil_image,
                        rng=rng,
                        coord_size=_coord_size_for_sbr(model, frame_grid),
                        model=model,
                        instr_query=_hook_instr_query_for_frame(model, policy, img_path),
                        image_id=str(img_path or "") or None,
                    )
                    if is_current:
                        coord_text = frame_coord
                    keep_token_set.update(frame_keep)
                    local_keep = frozenset(idx - start for idx in frame_keep)
                    if episode_key and img_path and not (
                        is_current and pub_cur_ratio is not None
                    ):
                        sticky_store[cache_key] = local_keep
                elif action == "reuse":
                    keep_token_set.update(start + idx for idx in cached)
                else:  # full
                    keep_token_set.update(range(start, end))

            tokenizer = getattr(model, "_geopact_tokenizer", None)
            embed_fn = getattr(model, "get_input_embeddings", lambda: None)()
            image_tok_id = int(getattr(model.config, "image_token_id", -1) or -1)

            return _finalize(model, _compact_sequence(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                image_mask=image_mask,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                keep_token_set=keep_token_set,
                seq_len=seq_len,
                hidden=hidden,
                coord_text=coord_text,
                tokenizer=tokenizer,
                embed_fn=embed_fn,
                image_tok_id=image_tok_id,
                policy=policy,
                position_ids=position_ids,
                rope_deltas=rope_deltas,
                model=model,
                video_grid_thw=video_grid_thw,
            ))

        prune_start, prune_end = token_ranges[-1]
        n_scope = prune_end - prune_start
        scoped_current = _scope_is_current_image(active_scope) and len(token_ranges) > 1
        if scoped_current:
            keep_k = max(1, int(round(n_scope * ratio)))
            select_embeds = image_embeds[prune_start:prune_end]
            select_grid = image_grid_thw[-1:] if int(image_grid_thw.shape[0]) > 1 else image_grid_thw
            select_n = n_scope
        else:
        keep_k = max(1, int(round(n_img * ratio)))
            select_embeds = image_embeds
            select_grid = image_grid_thw
            select_n = n_img

        # C-Entry (single-step Module C): reserve summary slots inside the
        # keep budget; B selects keep_k - R, the dropped set fills R summary
        # tokens after selection (total kept stays exactly keep_k).
        from lmms_eval.models.geopact_c_entry import c_entry_enabled, c_entry_reserve

        c_entry_r = 0
        if c_entry_enabled() and keep_k < select_n:
            c_entry_r = min(c_entry_reserve(keep_k), max(0, keep_k - 1))
            keep_k = keep_k - c_entry_r

        # ── Stage 1: GCC ──────────────────────────────────────────────────
        pil_image = getattr(model, "_geopact_last_image", None)
        mius: list = []
        if pil_image is not None and (
            not _policy_is_token_baseline(policy) or _pub_hybrid_active(policy)
        ):
            from lmms_eval.models.geopact_module_a_prefetch import _current_image_id

            _set_module_a_grid_context(model, select_grid, select_n)
            mius = _run_detector(
                pil_image, detector, policy, model=model, image_id=_current_image_id(model)
            )

        # ── Stage 2: HES ──────────────────────────────────────────────────
        if (
            not mius
            and not _policy_is_token_baseline(policy)
            and not _policy_uses_cdct(policy)
            and not _policy_uses_miu_first_backfill(policy)
        ):
            keep_local = set(range(select_n))
            selected_miu_indices = []
        else:
            _paths_ns = getattr(model, "_geopact_image_paths", None) or []
            hes = _hes_select(
            mius=mius,
                image_embeds=select_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
                image_grid_thw=select_grid,
            keep_k_tokens=keep_k,
            policy=policy,
                instr_query=_hook_instr_query_for_frame(
                    model, policy, _paths_ns[-1] if _paths_ns else None
                ),
                attn_scores=_vz_attn_scores_for_range(
                    model, policy,
                    prune_start if scoped_current else 0,
                    prune_end if scoped_current else n_img,
                    n_img,
                ),
            )
            keep_local = set(hes)
            selected_miu_indices = list(getattr(hes, "selected_miu_indices", []) or [])
        if c_entry_r > 0 and len(keep_local) < select_n:
            from lmms_eval.models.geopact_c_entry import build_c_entry_summaries

            _gh_ce, _gw_ce, _gm_ce = _get_grid(select_grid, select_n)
            if _gm_ce:
                medoids, summaries, c_audit = build_c_entry_summaries(
                    select_embeds,
                    keep_local,
                    select_n,
                    _gh_ce,
                    _gw_ce,
                    c_entry_r,
                )
                if medoids and summaries is not None:
                    med_t = torch.tensor(
                        medoids, dtype=torch.long, device=select_embeds.device
                    )
                    select_embeds.index_copy_(
                        0, med_t, summaries.to(select_embeds.dtype)
                    )
                    keep_local.update(medoids)
                if model is not None:
                    sp_audit = dict(getattr(model, "_geopact_spatial_audit", None) or {})
                    sp_audit.update(c_audit)
                    model._geopact_spatial_audit = sp_audit
        if scoped_current:
            keep_token_set = {prune_start + idx for idx in keep_local}
            for start, end in token_ranges[:-1]:
                keep_token_set.update(range(start, end))
        else:
            keep_token_set = keep_local

        compact_hits = getattr(model, "_geopact_compact_frame_cache_hits", None) or set()
        keep_token_set = _keep_token_set_with_compact_cache_hits(
            keep_token_set,
            token_ranges,
            compact_hits,
            model,
        )

        # Reconstruct selected MIU indices (needed for SBR) from keep_token_set
        grid_h, grid_w, grid_matches = _get_grid(select_grid, select_n)
        if mius and grid_matches:
            mapped_token_idxs, _, _, mapped_miu_indices = _map_mius_to_tokens(
                mius, select_embeds, grid_h, grid_w, select_n, need_embs=False
            )
            miu_token_idxs = _miu_token_idxs_table(
                len(mius),
                mapped_token_idxs,
                mapped_miu_indices,
                frame_offset=prune_start if scoped_current else 0,
            )
        else:
            miu_token_idxs = []

        if miu_token_idxs:
            if not selected_miu_indices:
            selected_miu_indices = sorted(
                    range(len(miu_token_idxs)),
                key=lambda i: _info_density(mius[i]) / max(len(miu_token_idxs[i]), 1),
                reverse=True,
            )

        # ── Stage 3: SBR ──────────────────────────────────────────────────
        coord_text = None
        if _should_build_coord_text(policy):
            if _coord_mode() == "gaec":
                from lmms_eval.models.geopact_coord_injection.gaec import prepare_gaec_context

                topk_env = int(os.environ.get("GEOPACT_COORD_TOPK", "0") or 0)
                prepare_gaec_context(
                    model,
                    selected_miu_indices=selected_miu_indices,
                    keep_token_set=keep_token_set,
                    miu_token_idxs=miu_token_idxs,
                    mius=mius,
                    frame_start=prune_start if scoped_current else 0,
                    n_frame=select_n,
                    topk=topk_env if topk_env > 0 else 3,
                )
            else:
        coord_text = _build_coord_text(
            selected_miu_indices=selected_miu_indices,
            keep_token_set=keep_token_set,
                    miu_token_idxs=miu_token_idxs,
                    mius=mius,
                    policy=policy,
                    coord_size=_coord_size_for_sbr(model, select_grid),
                )
        _attach_compact_prune_ctx(
            model,
            start=prune_start if scoped_current else 0,
            n_frame=select_n,
            miu_token_idxs=miu_token_idxs,
            mius=mius,
            policy=policy,
        )

        tokenizer = getattr(model, "_geopact_tokenizer", None)
        embed_fn  = getattr(model, "get_input_embeddings", lambda: None)()
        image_tok_id = int(getattr(model.config, "image_token_id", -1) or -1)

        return _finalize(model, _compact_sequence(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_mask=image_mask,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            keep_token_set=keep_token_set,
            seq_len=seq_len,
            hidden=hidden,
            coord_text=coord_text,
            tokenizer=tokenizer,
            embed_fn=embed_fn,
            image_tok_id=image_tok_id,
            policy=policy,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            model=model,
            video_grid_thw=video_grid_thw,
        ))

    return _hook
