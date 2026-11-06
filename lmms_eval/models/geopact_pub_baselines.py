"""Published training-free pruning baselines (paper comparison arms).

All baselines run under the ``baseline_full`` serving paradigm (full re-prefill
every step, ``GEOPACT_SESSION_KV=0``) so that the comparison isolates the token
*selection* policy; KV-rewrite serving is our own contribution and is never
enabled for these arms.

Two integration surfaces:

1. Feature-level selectors (DivPrune / CDPruner / VisionZip / VisPruner /
   PruMerge+ / TRIM / VisionTrim / ZOO-Prune / PruneSID) — dispatched from
   ``geopact_vl_hook._hes_select`` via
   ``visual_prune_policy``. They score ViT ``image_embeds`` rows per
   frame, exactly like the uniform/random token baselines. DivPrune and
   CDPruner's official protocols are representation-agnostic and match
   ours exactly.
   VisionZip / VisPruner adaptations are documented on the respective
   functions (both need ViT-internal [CLS]/mean attention, which Qwen3VL's
   windowed no-CLS ViT does not expose at the hook surface; we substitute
   the paper-specified no-CLS fallback "average attention received"
   computed from the same features the other baselines score).

1b. DART — official protocol prunes *inside the LLM* before layer K=2
   (ZichenWen1/DART Qwen2-VL integration), and per the 2026-07-20 fidelity
   decision we implement it exactly there: ``dart_prefill_ctx`` /
   ``dart_key_l1_norms`` / ``dart_keep_index`` run inline in the
   transformers patch alongside FastV. Pivots = top L1-norm of the
   layer-(K-1) post-RoPE key states (official defaults: 4 image + 4 text
   pivots); the retained set is built per-pivot — each pivot grabs its
   ``TOKEN_TOPK`` least-cosine-similar tokens (on the final-RMSNorm'ed
   layer-(K-1) hidden states) from the remaining image pool; text pivots
   are excluded from the retained set at the end. The older feature-level
   ``dart_select`` variant is DEPRECATED for paper tables (kept only for
   its unit-test oracle and historical arms).

2. FastV — requires layer-K LLM attention, so it is applied inline inside
   ``Qwen3VLTextModel.forward`` (transformers patch). Original-paper protocol:
   score = attention the visual tokens receive from the *last* prompt token at
   layer ``drop_k - 1`` (mean over heads, softmax in fp32); the lowest-scored
   tokens of each image segment are physically dropped before layer ``drop_k``
   runs. No instruction-segment query masking (that is a HistPrune variant,
   not FastV).

3. PDrop / PyramidDrop — same inline transformers-patch surface as FastV,
   but multi-stage: the LLM depth is cut into S=4 equal stages (official
   default) and at the end of each of the 3 stage boundaries the image
   tokens are ranked by the attention they receive from the last
   instruction token (query x key of that layer — identical scoring
   primitive to FastV here) and pruned by a per-stage retention ratio
   ``lambda``. Budget convention: ``GEOPACT_PDROP_KEEP_CUR`` (and
   ``_HIST``) give the *final-stage* keep ratio r; per-stage
   ``lambda = r^(1/3)`` so the retained fraction decays exponentially to r
   by the last stage (official pyramid shape, e.g. official
   image_token_ratio_list [0.5, 0.25, 0.125] == lambda 0.5). Note this
   leaves PDrop with strictly *more* average compute than FastV at the
   same nominal r (its shallow stages keep more tokens) — a deliberately
   conservative handicap in the baselines' favour.

Budget convention (aligned with our method's grids):

* single-step: the (single) current image keeps ``keep_cur`` fraction;
* multi-step: current frame keeps ``keep_cur``, every history frame keeps
  ``keep_hist`` — both frames are pruned, matching our (c, h) budget exactly.

Env flags::

    GEOPACT_FASTV=1                 enable inline FastV
    GEOPACT_FASTV_DROP_K=2          prune before this decoder layer (default 2)
    GEOPACT_FASTV_KEEP_CUR=0.5      keep ratio for the current (last) image
    GEOPACT_FASTV_KEEP_HIST=0.1     keep ratio for history images (default: cur)
    GEOPACT_PUBBASE_CURRENT=0.5     hook-path baselines: also prune the current
                                    frame at this keep ratio (multi-step c/h)
    GEOPACT_DART_PIVOTS=8           DART pivot count (paper default; env
                                    override kept for the ablation arms
                                    already run at 10, see dart_num_pivots())
    GEOPACT_PDROP=1                 enable inline PyramidDrop
    GEOPACT_PDROP_KEEP_CUR=0.5      final-stage keep ratio, current image
    GEOPACT_PDROP_KEEP_HIST=0.1     final-stage keep ratio, history images
    GEOPACT_PDROP_LAYERS=9,18,27    stage-boundary layers (default: equal
                                    quarters of the LLM depth, official S=4)
    GEOPACT_VISIONZIP_DOM_FRAC      dominant fraction of the keep budget
                                    (default 54/64, official 54:10 split)
    GEOPACT_VISPRUNER_IMP_RATIO     important fraction of the keep budget
                                    (default 0.5, official scripts' default)
    GEOPACT_VISIONTRIM_DVTS_FRAC    VisionTrim DVTS fraction of the keep
                                    budget (default 48/64, official 48:16)
    GEOPACT_ZOO_SENS_CAPTURE=1      ZOO-Prune: vision tower computes the
                                    zeroth-order projector sensitivity
    GEOPACT_ZOO_NUM_DIRS=64         ZOO-Prune perturbation directions m
    GEOPACT_ZOO_INTENS=0.01         ZOO-Prune perturbation magnitude h
    GEOPACT_PRUNESID_TAU_SCALE      PruneSID NMS threshold multiplier
                                    (default 1.0 = official adaptive tau;
                                    M3 best-tuned-baseline sweep)
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

_log = logging.getLogger("geopact.pub_baselines")


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------

def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val


def fastv_enabled() -> bool:
    return os.environ.get("GEOPACT_FASTV", "").strip().lower() in {"1", "true", "yes", "on"}


def fastv_drop_k() -> int:
    raw = os.environ.get("GEOPACT_FASTV_DROP_K", "").strip()
    return int(raw) if raw else 2


def fastv_keep_cur() -> float:
    val = _env_float("GEOPACT_FASTV_KEEP_CUR")
    if val is None:
        raise ValueError("GEOPACT_FASTV=1 requires GEOPACT_FASTV_KEEP_CUR")
    return min(max(val, 0.0), 1.0)


def fastv_keep_hist() -> float:
    val = _env_float("GEOPACT_FASTV_KEEP_HIST")
    if val is None:
        return fastv_keep_cur()
    return min(max(val, 0.0), 1.0)


def pubbase_current_keep_ratio() -> Optional[float]:
    """Multi-step c/h for the hook-path baselines: keep ratio for the current
    frame (None → current frame follows the default sticky protocol)."""
    val = _env_float("GEOPACT_PUBBASE_CURRENT")
    if val is None or not (0.0 < val < 1.0):
        return None
    return val


def pdrop_enabled() -> bool:
    return os.environ.get("GEOPACT_PDROP", "").strip().lower() in {"1", "true", "yes", "on"}


def pdrop_keep_cur() -> float:
    val = _env_float("GEOPACT_PDROP_KEEP_CUR")
    if val is None:
        raise ValueError("GEOPACT_PDROP=1 requires GEOPACT_PDROP_KEEP_CUR")
    return min(max(val, 0.0), 1.0)


def pdrop_keep_hist() -> float:
    val = _env_float("GEOPACT_PDROP_KEEP_HIST")
    if val is None:
        return pdrop_keep_cur()
    return min(max(val, 0.0), 1.0)


def pdrop_drop_layers(num_layers: int) -> List[int]:
    """Stage-boundary layers. Official protocol: S=4 equal stages, rank&drop
    after each of the first three (LLaVA-1.5 32 layers -> [8, 16, 24]);
    generalised here to equal quarters of the actual LLM depth."""
    raw = os.environ.get("GEOPACT_PDROP_LAYERS", "").strip()
    if raw:
        layers = sorted({int(x) for x in raw.replace(";", ",").split(",") if x.strip()})
        return [l for l in layers if 1 <= l < num_layers]
    q = max(1, num_layers // 4)
    return [q, 2 * q, 3 * q]


def visionzip_dominant_frac() -> float:
    val = _env_float("GEOPACT_VISIONZIP_DOM_FRAC")
    if val is None or not (0.0 < val <= 1.0):
        return 54.0 / 64.0
    return val


def vz_attn_capture_enabled() -> bool:
    """GEOPACT_VZ_ATTN_CAPTURE=1: the vision tower captures the *real*
    last-block softmax attention-received score per merged token (official
    VisionZip/VisPruner/PruMerge+ scoring signal; 2026-07-20 fidelity ruling:
    the scoring signal is method core and must not be proxied)."""
    return os.environ.get("GEOPACT_VZ_ATTN_CAPTURE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


@torch.no_grad()
def vz_attn_received_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens,
    merge_size: int,
    chunk: int = 512,
) -> torch.Tensor:
    """Streaming attention-received column sums (official VisionZip protocol).

    Reproduces the official ``qwen2_5vl_visionzip.py`` reduction exactly:
    ``softmax(qk^T/sqrt(d))`` per attention segment, mean over heads, sum
    over the query dim (= attention each key receives), then mean over each
    ``merge_size**2`` consecutive-row PatchMerger group. Computed as a
    chunked column-sum so the T x T matrix is never materialized (Qwen3VL's
    full-sequence ViT attention would need ~4 GB/head eagerly on large GUI
    screenshots).

    q, k: (heads, seq, head_dim) post-RoPE; cu_seqlens: per-image segment
    boundaries (attention never crosses segments). Returns (seq //
    merge_size**2,) fp32 scores on q's device.
    """
    heads, seq_len, head_dim = q.shape
    q = q.float()
    k = k.float()
    scale = float(head_dim) ** -0.5
    scores = torch.zeros(seq_len, dtype=torch.float32, device=q.device)
    bounds = [int(b) for b in (cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s:
            continue
        k_seg_t = k[:, s:e, :].transpose(1, 2)  # (heads, hd, seg)
        seg_scores = torch.zeros(e - s, dtype=torch.float32, device=q.device)
        for qs in range(s, e, max(1, int(chunk))):
            qe = min(qs + max(1, int(chunk)), e)
            probs = torch.softmax(torch.matmul(q[:, qs:qe, :], k_seg_t) * scale, dim=-1)
            seg_scores += probs.sum(dim=(0, 1))
        scores[s:e] = seg_scores / float(heads)
    group = int(merge_size) * int(merge_size)
    if group <= 1:
        return scores
    if seq_len % group != 0:
        raise RuntimeError(
            f"vz_attn_received_from_qk: seq {seq_len} not divisible by merge group {group}"
        )
    return scores.view(seq_len // group, group).mean(dim=-1)


def prumerge_outlier_frac() -> float:
    val = _env_float("GEOPACT_PRUMERGE_OUTLIER_FRAC")
    if val is None or not (0.0 < val <= 1.0):
        return 0.25
    return val


def visiontrim_dvts_frac() -> float:
    """GEOPACT_VISIONTRIM_DVTS_FRAC: dominant (DVTS) fraction of the keep
    budget. Official VisionTrim (Yu et al., ICLR'26) eval scripts use
    DVTS_token_num=48 + TGVC_token_num=16 at the 64-token budget -> 0.75."""
    val = _env_float("GEOPACT_VISIONTRIM_DVTS_FRAC")
    if val is None or not (0.0 < val <= 1.0):
        return 48.0 / 64.0
    return val


def zoo_sens_capture_enabled() -> bool:
    """GEOPACT_ZOO_SENS_CAPTURE=1: the vision tower computes ZOO-Prune's
    zeroth-order projector sensitivity per merged token (official scoring
    signal — method core, same no-proxy ruling as the VisionZip family)."""
    return os.environ.get("GEOPACT_ZOO_SENS_CAPTURE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def zoo_num_dirs() -> int:
    """GEOPACT_ZOO_NUM_DIRS: number of random perturbation directions m
    (official run_zoo_prune.sh NOISERECOV_NUM=64)."""
    raw = os.environ.get("GEOPACT_ZOO_NUM_DIRS", "").strip()
    return int(raw) if raw else 64


def zoo_intensity() -> float:
    """GEOPACT_ZOO_INTENS: perturbation magnitude h (official
    run_zoo_prune.sh NOISERECOV_INTENS=0.01)."""
    val = _env_float("GEOPACT_ZOO_INTENS")
    return 0.01 if val is None or val <= 0.0 else val


def zoo_seed() -> int:
    """GEOPACT_ZOO_SEED: seed for the perturbation directions. The official
    code draws unseeded ``randn`` per image; we fix the generator so every
    run (and every image) sees the same m directions — a determinism-only
    deviation (the sensitivity estimator is an expectation over directions;
    fixing the sample changes nothing about the method)."""
    raw = os.environ.get("GEOPACT_ZOO_SEED", "").strip()
    return int(raw) if raw else 1234


@torch.no_grad()
def zoo_sensitivity_from_merger(
    hidden_states: torch.Tensor,
    merger,
    num_dirs: Optional[int] = None,
    intensity: Optional[float] = None,
    chunk: int = 8,
) -> torch.Tensor:
    """ZOO-Prune (Kim et al., CVPR'26) zeroth-order projector sensitivity.

    Official protocol (AIM-SKKU/ZOO-Prune ``llava_arch.py``, followed
    exactly): draw m random unit directions u_j in the pre-projector feature
    space, apply the SAME u_j to every token, push the +-h perturbed
    features through the (frozen) modality projector, and score each output
    token by the mean central-difference norm ``mean_j ||proj(x + h u_j) -
    proj(x - h u_j)|| / (2h)``. Defaults m=64, h=0.01 (official
    run_zoo_prune.sh NOISERECOV_NUM / NOISERECOV_INTENS).

    Qwen3VL adaptation: the modality projector is the vision tower's
    PatchMerger (LayerNorm -> 2x2 group flatten -> MLP); each output row
    aggregates one merge group, so the sensitivity lands per *merged* token
    — exactly the rows the hook-path selector scores (same alignment as the
    VisionZip-family attention capture). Directions are drawn from a fixed
    generator (see ``zoo_seed``).

    ``hidden_states``: (seq, d) pre-merger features; returns (seq //
    merge_group,) fp32 scores on the same device.
    """
    m = int(num_dirs if num_dirs is not None else zoo_num_dirs())
    h = float(intensity if intensity is not None else zoo_intensity())
    seq_len, d = hidden_states.shape
    device = hidden_states.device
    proj_dtype = next(merger.parameters()).dtype
    x = hidden_states.to(proj_dtype)

    gen = torch.Generator(device=device)
    gen.manual_seed(zoo_seed())
    # Draw all m directions in one call (chunk-size independent), then chunk
    # only the perturbed forward passes to bound memory.
    u_all = torch.randn(m, d, generator=gen, device=device, dtype=torch.float32)
    u_all = (u_all / (u_all.norm(dim=-1, keepdim=True) + 1e-12)).to(proj_dtype)

    acc: Optional[torch.Tensor] = None
    for cs in range(0, m, max(1, int(chunk))):
        ce = min(cs + max(1, int(chunk)), m)
        c = ce - cs
        u = u_all[cs:ce]
        pert = x.unsqueeze(0) + h * u.unsqueeze(1)          # (c, seq, d) +h
        proj_plus = merger(pert.reshape(c * seq_len, d))     # (c*seq/g, out)
        pert = x.unsqueeze(0) - h * u.unsqueeze(1)
        proj_minus = merger(pert.reshape(c * seq_len, d))
        delta = (proj_plus.float() - proj_minus.float()).view(c, -1, proj_plus.shape[-1])
        sens = delta.norm(dim=-1) / (2.0 * h)                # (c, n_out)
        acc = sens.sum(dim=0) if acc is None else acc + sens.sum(dim=0)
    assert acc is not None
    return acc / float(m)


def vispruner_important_ratio() -> float:
    val = _env_float("GEOPACT_VISPRUNER_IMP_RATIO")
    if val is None or not (0.0 <= val <= 1.0):
        return 0.5
    return val


def dart_num_pivots() -> int:
    """GEOPACT_DART_PIVOTS: pivot count. Official DART (Wen et al. 2025,
    EMNLP) uses 8 pivot tokens "in all experiments" (paper Table 1 caption);
    default corrected 2026-07-17 (was 10 — a pre-verification placeholder;
    the paper's own Table 8/Fig 5 robustness ablation shows pivot-selection
    criteria and counts in [8, 20] vary by <3%, so already-completed arms at
    10 are not re-run)."""
    raw = os.environ.get("GEOPACT_DART_PIVOTS", "").strip()
    return int(raw) if raw else 8


# ---------------------------------------------------------------------------
# in-LLM DART (official protocol; supersedes the deprecated feature-level
# ``dart_select`` variant for all paper-table arms, 2026-07-20 decision)
# ---------------------------------------------------------------------------

def dart_inllm_enabled() -> bool:
    return os.environ.get("GEOPACT_DART", "").strip().lower() in {"1", "true", "yes", "on"}


def dart_drop_k() -> int:
    """Prune before this decoder layer. Official Qwen2-VL integration: K=2
    (``lmms-eval/models/qwen2_vl.py`` default ``pruned_layer=2``)."""
    raw = os.environ.get("GEOPACT_DART_DROP_K", "").strip()
    return int(raw) if raw else 2


def dart_keep_cur() -> float:
    val = _env_float("GEOPACT_DART_KEEP_CUR")
    if val is None:
        raise ValueError("GEOPACT_DART=1 requires GEOPACT_DART_KEEP_CUR")
    return min(max(val, 0.0), 1.0)


def dart_keep_hist() -> float:
    val = _env_float("GEOPACT_DART_KEEP_HIST")
    if val is None:
        return dart_keep_cur()
    return min(max(val, 0.0), 1.0)


def dart_pivot_img() -> int:
    """Official ``pivot_image_token`` default: 4 (Qwen2-VL eval scripts)."""
    raw = os.environ.get("GEOPACT_DART_PIVOT_IMG", "").strip()
    return int(raw) if raw else 4


def dart_pivot_txt() -> int:
    """Official ``pivot_text_token`` default: 4 (Qwen2-VL eval scripts)."""
    raw = os.environ.get("GEOPACT_DART_PIVOT_TXT", "").strip()
    return int(raw) if raw else 4


def dart_prefill_ctx(
    *,
    past_key_values,
    hidden_states: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
) -> Optional[dict]:
    """Return an in-LLM DART context when the current forward is an eligible
    full prefill (same eligibility rules as FastV)."""
    if not dart_inllm_enabled():
        return None
    if visual_pos_masks is None or hidden_states.shape[0] != 1:
        return None
    if hidden_states.shape[1] <= 1:
        return None
    past_len = 0
    if past_key_values is not None:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = 0
    if past_len > 0:
        return None
    vmask = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
    if int(vmask.sum().item()) == 0:
        return None
    drop_k = dart_drop_k()
    if drop_k < 1:
        return None
    return {
        "drop_k": drop_k,
        "keep_cur": dart_keep_cur(),
        "keep_hist": dart_keep_hist(),
        "pivot_img": dart_pivot_img(),
        "pivot_txt": dart_pivot_txt(),
        "vmask": vmask.bool(),
        "k_l1": None,
    }


@torch.no_grad()
def dart_key_l1_norms(
    decoder_layer,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Per-token L1 norm of the layer's post-RoPE key states, heads flattened.

    Official DART scores pivots on the layer-(K-1) key states after RoPE
    (``modeling_qwen2_vl_self.py``: k_states from the layer output tuple,
    ``permute(0,2,1,3).reshape(b, s, -1)`` then ``norm(p=1, dim=-1)``).
    The official tensor has ``repeat_kv`` applied (each KV head duplicated
    ``groups`` times); duplication multiplies every token's L1 norm by the
    same constant, so ranking is identical without it."""
    hs = decoder_layer.input_layernorm(hidden_states)
    attn = decoder_layer.self_attn
    bsz, seq, _ = hs.shape
    head_dim = attn.head_dim
    k = attn.k_proj(hs).view(bsz, seq, -1, head_dim)
    k = attn.k_norm(k).transpose(1, 2)  # (b, KVH, s, hd)
    cos, sin = position_embeddings
    k = _apply_rope(k, cos, sin)
    # flatten heads -> per-token vector, L1 over it (official reshape+norm)
    k_flat = k.transpose(1, 2).reshape(bsz, seq, -1).float()
    return k_flat.norm(p=1, dim=-1)[0]  # (seq,)


def dart_keep_index_from_parts(
    *,
    k_l1: torch.Tensor,
    hidden_norm: torch.Tensor,
    vmask: torch.Tensor,
    keep_cur: float,
    keep_hist: float,
    pivot_img: int,
    pivot_txt: int,
    device,
) -> torch.Tensor:
    """Official DART retained-set construction, generalised to multi-image
    (c/h) budgets by running the official per-image procedure per segment.

    Official single-image mechanics (get_retained_image_token, followed
    line-by-line): image pivots = top ``pivot_img`` K-L1 tokens of the image;
    text pivots = top ``pivot_txt`` K-L1 tokens of the query text (tokens
    after the image); ``TOKEN_TOPK = int(L * keep / (pivot_img + pivot_txt))``;
    iterating over the initial pivot snapshot, each pivot grabs its
    ``TOKEN_TOPK`` least-cosine-similar (final-RMSNorm hidden states) tokens
    from the remaining image pool; text pivots are dropped from the retained
    set at the end. All text tokens are always kept.

    Multi-image generalisation (declared): the procedure runs independently
    per image segment with that segment's keep ratio (current frame
    ``keep_cur``, history ``keep_hist``); the text-pivot pool is the
    instruction suffix (non-visual tokens after the last image), matching the
    official "query tokens after the image" semantics.
    """
    seq = int(k_l1.shape[0])
    vmask = vmask.to(k_l1.device)
    keep = torch.ones(seq, dtype=torch.bool, device=k_l1.device)
    segs = visual_segments(vmask)
    if not segs:
        return torch.arange(seq, device=device)

    hs = hidden_norm.float()
    last_end = segs[-1][1]
    txt_pool = torch.nonzero(~vmask, as_tuple=False).squeeze(-1)
    txt_suffix = txt_pool[txt_pool >= last_end]
    if int(txt_suffix.numel()) == 0:
        txt_suffix = txt_pool

    for seg_idx, (s, e) in enumerate(segs):
        is_current = seg_idx == len(segs) - 1
        ratio = keep_cur if is_current else keep_hist
        n = e - s
        if round(n * float(ratio)) >= n:
            continue  # segment keeps full budget -> official no-op

        p_img = min(int(pivot_img), n)
        p_txt = min(int(pivot_txt), int(txt_suffix.numel()))
        n_pivots = max(1, p_img + p_txt)
        token_topk = int(n * float(ratio) / n_pivots)

        seg_l1 = k_l1[s:e]
        img_piv = (torch.topk(seg_l1, k=p_img, largest=True).indices + s).tolist()
        if p_txt > 0:
            txt_l1 = k_l1[txt_suffix]
            txt_piv = txt_suffix[torch.topk(txt_l1, k=p_txt, largest=True).indices].tolist()
        else:
            txt_piv = []

        retained = set(img_piv)
        valid = [i for i in range(s, e) if i not in retained]
        for item in img_piv + txt_piv:  # official snapshot iteration order
            if not valid or token_topk <= 0:
                break
            vecs = hs[valid, :]
            neg_cos = -F.cosine_similarity(hs[item, :].unsqueeze(0), vecs, dim=-1)
            k_grab = min(token_topk, len(valid))
            top = torch.topk(neg_cos, k=k_grab).indices.tolist()
            grabbed = [valid[i] for i in top]
            retained.update(grabbed)
            valid = [i for i in valid if i not in set(grabbed)]

        keep[s:e] = False
        keep[sorted(retained)] = True

    return torch.nonzero(keep, as_tuple=False).squeeze(-1).to(device)


def dart_keep_index(ctx: dict, hidden_norm_b: torch.Tensor, device) -> torch.Tensor:
    """ctx + final-RMSNorm'ed hidden states (b=1) -> kept sequence positions."""
    return dart_keep_index_from_parts(
        k_l1=ctx["k_l1"],
        hidden_norm=hidden_norm_b[0],
        vmask=ctx["vmask"],
        keep_cur=ctx["keep_cur"],
        keep_hist=ctx["keep_hist"],
        pivot_img=ctx["pivot_img"],
        pivot_txt=ctx["pivot_txt"],
        device=device,
    )


# ---------------------------------------------------------------------------
# feature-level selectors (per-frame, rows = visual tokens of one frame)
# ---------------------------------------------------------------------------

def _prunesid_pca_group(features: torch.Tensor, min_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """PSCA grouping (official ``pca_group``, prunesid_qwen/modeling_qwen2_vl.py).

    ``standard = sigmoid(features.float()).T`` ([D, N]); PCA over the token
    columns; each token joins the principal component with its largest
    absolute loading. Official code calls ``torch.pca_lowrank`` (randomized
    SVD, unseeded); we compute the same PCA with an exact deterministic SVD
    of the column-centred matrix — identical math (pca_lowrank centres by the
    column mean too), no run-to-run RNG drift.
    """
    standard = torch.sigmoid(features.to(torch.float32)).permute(1, 0)
    standard = standard - standard.mean(dim=0, keepdim=True)
    q = min(int(min_components), standard.shape[0], standard.shape[1])
    _u, _s, vh = torch.linalg.svd(standard, full_matrices=False)
    v = torch.abs(vh[:q].permute(1, 0))  # [N, q] token loadings
    belong = torch.argmax(v, dim=1)
    return v, belong


def _prunesid_nms(similarity: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
    """Official greedy NMS: pop the max-score token, suppress everything more
    similar than ``threshold``, repeat while any positive score remains."""
    scores = scores.clone()
    keep: List[int] = []
    while float(scores.sum()) > 0:
        max_idx = int(scores.argmax())
        scores[max_idx] = 0
        keep.append(max_idx)
        scores[similarity[max_idx] > threshold] = 0
    return keep


# Selectors that define a hybrid (published rule × spatial-prior mass) variant.
# Single source of truth: geopact_vl_hook imports this so the hook-side gate and
# the dispatch-side guard can never disagree about which arms are legal.
HYBRID_CAPABLE_SELECTORS = ("vispruner", "divprune", "cdpruner", "prunesid")


def _policy_defines_hybrid(policy: str) -> bool:
    p = str(policy or "").lower()
    # "visiontrim" must not match "trim"-style substrings; all four names here
    # are unambiguous, so a plain containment test is safe.
    return any(s in p for s in HYBRID_CAPABLE_SELECTORS)


def prunesid_select(
    embeds: torch.Tensor,
    keep_k: int,
    hybrid_mass: Optional[torch.Tensor] = None,
) -> List[int]:
    """PruneSID (Fang et al., ICLR'26): PSCA grouping + intra-group NMS.

    Line-by-line port of the official Qwen2-VL integration
    (``prunesid/prunesid_qwen/modeling_qwen2_vl.py``,
    ``Qwen2VisionTransformerPretrainedModel_prunesid.forward``), which scores
    the *merger output* rows — exactly the representation our hook surface
    provides, so no signal substitution is needed. Pipeline: (1) PSCA groups
    tokens by their dominant principal component (q = max(keep/4, 4));
    (2) per-group greedy NMS with the adaptive threshold
    ``tau = max(keep/(N/18), 1) * sim_mean`` where ``sim_mean`` is the
    official upper-triangle-zeroed mean of the full cosine matrix (their
    ``triu(0).mean()`` divides by N^2 — replicated verbatim); (3) per-group
    quotas: lower bound 1, upper bound ``5*ceil(keep/64)`` capped by group
    and NMS-survivor counts, remainder shared proportionally to survivor
    counts via the official cumsum-round-diff scheme, greedy fill by largest
    group. Adaptations: the per-image
    ``max(round(...), 16)`` floor and the N<=16 passthrough are replaced by
    the shared exact-``keep_k`` budget convention of all our baseline arms
    (the floor never binds at our grid budgets); exact SVD replaces
    ``pca_lowrank`` (see ``_prunesid_pca_group``). The official information-
    aware dynamic per-image ratio is not in the released Qwen2 code path and
    is therefore not replicated (fixed per-frame budget, matching both the
    official release and our budget-parity protocol).

    ``hybrid_mass`` (E3 attribution control, GEOPACT_PUB_HYBRID_MASS=1): the
    spatial prior multiplies the PCA loading magnitudes that PruneSID ranks
    within each group, so a detected token wins its intra-group NMS ties and
    its quota slot ahead of an equally-loaded background token. Grouping,
    adaptive threshold and quota arithmetic are untouched — the prior only
    re-ranks inside PruneSID's own score channel. ``hybrid_mass=None`` runs the
    published rule bit-identically.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    device = embeds.device

    lengths, belong = _prunesid_pca_group(embeds, max(int(keep_k / 4), 4))
    if hybrid_mass is not None:
        mass = hybrid_mass.float().reshape(-1).to(lengths.device)
        if int(mass.numel()) != n:
            raise ValueError(
                f"prunesid hybrid_mass length {int(mass.numel())} != n tokens {n}"
            )
        lengths = lengths * mass.clamp_min(0.0).unsqueeze(1)
    lengths = lengths.clone()
    scores = lengths.clone()
    n_comp = lengths.shape[1]

    # zero every loading outside the token's own group (official masking)
    comp_ids = torch.arange(n_comp, device=device).unsqueeze(0).expand(n, -1)
    own = belong.unsqueeze(1).expand(-1, n_comp) == comp_ids
    lengths[~own] = 0
    scores[~own] = 0

    normalized = F.normalize(embeds.float(), p=2, dim=-1)
    sim = torch.mm(normalized, normalized.t())
    sim_mean = float(sim.triu(diagonal=0).mean())  # official N^2 denominator
    ratio = max(keep_k / (n / 18.0), 1.0)
    threshold = ratio * sim_mean
    # M3 fairness sweep: GEOPACT_PRUNESID_TAU_SCALE multiplies the official
    # adaptive NMS threshold (default 1.0 = published rule) — PruneSID's one
    # free hyperparameter for the best-tuned-baseline comparison.
    _tau_scale = _env_float("GEOPACT_PRUNESID_TAU_SCALE")
    if _tau_scale is not None and _tau_scale > 0.0:
        threshold = threshold * _tau_scale

    given = torch.arange(n, 0, -1, device=device, dtype=scores.dtype)
    group_idxs: List[torch.Tensor] = []
    for g in range(n_comp):
        gi = torch.nonzero(belong == g, as_tuple=False).squeeze(-1)
        if gi.numel() == 0:
            group_idxs.append(gi)
            continue
        g_sim = sim[gi][:, gi]
        g_scores = lengths[gi, g]
        kept_local = _prunesid_nms(g_sim, g_scores, threshold)
        kept = gi[torch.tensor(kept_local, dtype=torch.long, device=device)]
        scores[kept, g] = given[: kept.numel()]
        group_idxs.append(kept)

    nms_counts = torch.tensor([gi.numel() for gi in group_idxs],
                              device=device, dtype=torch.float32)
    group_counts = own.sum(dim=0).to(torch.float32)

    lower = torch.minimum(torch.ones_like(group_counts), group_counts)
    upper = torch.full_like(group_counts, 5.0 * math.ceil(keep_k / 64.0))
    upper = torch.minimum(upper, group_counts)
    upper = torch.minimum(upper, nms_counts)
    while float(upper.sum()) < keep_k:
        upper = torch.minimum(upper + 1, group_counts)

    other = max(0.0, keep_k - float(lower.sum()))
    norm_counts = nms_counts / nms_counts.sum().clamp_min(1e-12)
    cum = torch.cumsum(norm_counts, dim=0)
    dist = (cum * other).round()
    dist = dist - torch.cat([torch.zeros(1, device=device), dist[:-1]])
    quota = torch.minimum(dist + lower, upper)

    order = torch.argsort(nms_counts, descending=True)
    target = other + float(lower.sum())
    fill = 0
    while float(quota.sum()) < target and fill < n_comp:
        gi = order[fill]
        quota[gi] += min(float(upper[gi] - quota[gi]), target - float(quota.sum()))
        fill += 1

    sort_idx = torch.argsort(scores, descending=True, dim=0)
    picked: List[torch.Tensor] = []
    for g in range(n_comp):
        picked.append(sort_idx[: int(quota[g]), g])
    out = torch.cat(picked, dim=0)[:keep_k]
    return sorted(out.tolist())


def divprune_select(
    embeds: torch.Tensor,
    keep_k: int,
    hybrid_mass: Optional[torch.Tensor] = None,
) -> List[int]:
    """DivPrune: greedy max-min diversity selection (d = 1 - cosine).

    Seed = token with the largest nearest-neighbour distance; then repeatedly
    add the token whose minimum distance to the selected set is largest.

    ``hybrid_mass`` (E3 attribution control, GEOPACT_PUB_HYBRID_MASS=1):
    per-token mass m_j = 1 + α·P_j from the Module A spatial prior; the greedy
    utility becomes m_j · d_min(j, S) (mass-weighted max-min objective). With
    ``hybrid_mass=None`` the published rule runs bit-identically.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    feats = F.normalize(embeds.float(), p=2, dim=-1)
    sim = torch.mm(feats, feats.t()).clamp(-1.0, 1.0)

    dist = 1.0 - sim
    dist.fill_diagonal_(float("inf"))

    if hybrid_mass is not None:
        mass = hybrid_mass.float().reshape(-1).to(embeds.device)
        if int(mass.numel()) != n:
            raise ValueError(
                f"divprune hybrid_mass length {int(mass.numel())} != n tokens {n}"
            )
        nn_dist = dist.min(dim=1).values.clamp_min(0.0)
        seed = int(torch.argmax(mass * nn_dist).item())
        selected = torch.zeros(n, dtype=torch.bool, device=embeds.device)
        selected[seed] = True
        max_sim_to_sel = sim[:, seed].clone()
        for _ in range(keep_k - 1):
            min_dist = (1.0 - max_sim_to_sel).clamp_min(0.0)
            cand = (mass * min_dist).masked_fill(selected, float("-inf"))
            best = int(torch.argmax(cand).item())
            selected[best] = True
            max_sim_to_sel = torch.maximum(max_sim_to_sel, sim[:, best])
        return sorted(torch.nonzero(selected, as_tuple=False).squeeze(-1).tolist())

    seed = int(torch.argmax(dist.min(dim=1).values).item())

    selected = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    selected[seed] = True
    max_sim_to_sel = sim[:, seed].clone()
    for _ in range(keep_k - 1):
        cand = max_sim_to_sel.masked_fill(selected, float("inf"))
        best = int(torch.argmin(cand).item())
        selected[best] = True
        max_sim_to_sel = torch.maximum(max_sim_to_sel, sim[:, best])
    return sorted(torch.nonzero(selected, as_tuple=False).squeeze(-1).tolist())


def dart_select(embeds: torch.Tensor, keep_k: int, num_pivots: Optional[int] = None) -> List[int]:
    """DEPRECATED feature-level DART variant (2026-07-20). The paper-table
    DART arms now use the official in-LLM protocol (``dart_prefill_ctx`` /
    ``dart_keep_index``, transformers-patch inline, layer K=2) — see module
    docstring §1b. This function is retained only for the historical
    ``visual_prune_policy=dart`` arms and its unit-test oracle; do not queue
    new paper arms through it.

    Variant mechanics: pivots by L2 norm of the merger-output features and a
    *global* keep of the tokens with the lowest max-cosine to any pivot
    (official: layer-2 K-state L1-norm pivots + per-pivot round-robin)."""
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    p = min(int(num_pivots if num_pivots is not None else dart_num_pivots()), n)
    feats = embeds.float()
    norms = feats.norm(dim=-1)
    _, piv = torch.topk(norms, k=p, largest=True)

    feats_n = F.normalize(feats, p=2, dim=-1)
    sim = torch.mm(feats_n, feats_n[piv].t())  # [n, p]
    max_sim = sim.max(dim=-1).values

    if keep_k <= p:
        piv_scores = max_sim[piv]
        _, sub = torch.topk(piv_scores, k=keep_k, largest=False)
        return sorted(piv[sub].tolist())

    keep_mask = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    keep_mask[piv] = True
    cand_scores = max_sim.masked_fill(keep_mask, float("inf"))
    _, extra = torch.topk(cand_scores, k=keep_k - p, largest=False)
    keep_mask[extra] = True
    return sorted(torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist())


def cdpruner_select(
    embeds: torch.Tensor,
    keep_k: int,
    instr_query: Optional[torch.Tensor] = None,
    hybrid_mass: Optional[torch.Tensor] = None,
) -> List[int]:
    """CDPruner: instruction-conditioned DPP (kernel L~ = diag(r~) S diag(r~)).

    Official protocol (Zeng et al., NeurIPS'25, Eq. 5-7): relevance r_i is
    the raw cosine similarity between visual token i and the instruction
    embedding, then **min-max normalised** to [0, 1] (r~ = (r - min(r)) /
    (max(r) - min(r))) — NOT an exponential/softmax sharpening. With no
    instruction the kernel degrades to a pure-diversity DPP (r~ == 1). Greedy
    MAP via the fast Cholesky update (Chen et al., 2018).

    (2026-07-17 official-repo verification: an earlier draft of this function
    used ``rel = exp(theta * cos)`` — a made-up sharpening temperature with
    no basis in the paper. Fixed to match Eq. 6 exactly; every CDPruner arm
    started before the fix landed was deprecated and the full grid re-queued
    under the corrected kernel in ``logs/xds8b_pub2/``.)

    ``hybrid_mass`` (E3 attribution control, GEOPACT_PUB_HYBRID_MASS=1): the
    spatial prior enters through CDPruner's *own* relevance channel,
    r~ <- r~ * m, rescaled back to max 1 so the relevance/diversity balance of
    the DPP kernel keeps its published scale and only the *ranking* inside the
    relevance term changes. The greedy Cholesky MAP is untouched.
    ``hybrid_mass=None`` runs the published rule bit-identically.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    feats = F.normalize(embeds.float(), p=2, dim=-1)
    sim = torch.mm(feats, feats.t()).clamp(-1.0, 1.0)

    if instr_query is not None:
        q = instr_query.float().reshape(-1)
        q = F.normalize(q, p=2, dim=-1).to(feats.device)
        cos = torch.mv(feats, q).clamp(-1.0, 1.0)
        r_min, r_max = cos.min(), cos.max()
        span = (r_max - r_min).clamp_min(1e-8)
        rel = (cos - r_min) / span
    else:
        rel = torch.ones(n, device=feats.device)

    if hybrid_mass is not None:
        mass = hybrid_mass.float().reshape(-1).to(feats.device)
        if int(mass.numel()) != n:
            raise ValueError(
                f"cdpruner hybrid_mass length {int(mass.numel())} != n tokens {n}"
            )
        rel = rel * mass.clamp_min(0.0)
        rel = rel / rel.max().clamp_min(1e-8)

    # L_ij = r_i * S_ij * r_j; greedy MAP with incremental Cholesky.
    d2 = rel * rel  # diag(L), S_ii = 1
    cis = torch.zeros(keep_k, n, device=feats.device)
    picked: List[int] = []
    active = torch.ones(n, dtype=torch.bool, device=feats.device)
    for t in range(keep_k):
        scores = d2.masked_fill(~active, float("-inf"))
        j = int(torch.argmax(scores).item())
        dj = torch.sqrt(torch.clamp(d2[j], min=1e-12))
        l_row = rel[j] * sim[j] * rel  # L[j, :]
        if t == 0:
            e = l_row / dj
        else:
            e = (l_row - torch.mv(cis[:t].t(), cis[:t, j])) / dj
        cis[t] = e
        d2 = d2 - e * e
        active[j] = False
        picked.append(j)
    return sorted(picked)


def _attn_received_scores(embeds: torch.Tensor) -> torch.Tensor:
    """DEPRECATED feature-space proxy of "attention received" (2026-07-20
    fidelity ruling: the scoring signal is method core — VisionZip /
    VisPruner / PruMerge+ paper arms must use the *real* last-vision-block
    attention captured by the transformers patch, see
    ``vz_attn_received_from_qk``). Kept only for TRIM's no-instruction
    fallback and historical/unit-test arms.

    A = softmax(X X^T / sqrt(d)) row-wise (fp32), score_j = mean_i A[i, j].
    """
    feats = embeds.float()
    d = feats.shape[-1]
    logits = torch.mm(feats, feats.t()) / (float(d) ** 0.5)
    probs = torch.softmax(logits, dim=-1)
    return probs.mean(dim=0)


def _resolve_attn_scores(
    embeds: torch.Tensor,
    attn_scores: Optional[torch.Tensor],
    method: str,
) -> torch.Tensor:
    """Return the official attention-received score vector for a frame.

    When the capture env is on (all paper-table VisionZip/VisPruner/
    PruMerge+ arms), a missing/mismatched score vector is a hard error —
    silently falling back to the feature proxy would reintroduce the exact
    fidelity violation the 2026-07-20 ruling forbids."""
    n = int(embeds.shape[0])
    if attn_scores is not None:
        if int(attn_scores.numel()) != n:
            raise RuntimeError(
                f"{method}: attn_scores rows {int(attn_scores.numel())} != embeds rows {n}"
            )
        return attn_scores.reshape(-1).float().to(embeds.device)
    if vz_attn_capture_enabled():
        raise RuntimeError(
            f"{method}: GEOPACT_VZ_ATTN_CAPTURE=1 but no captured attention "
            "scores reached the selector (plumbing bug — refusing the proxy fallback)"
        )
    return _attn_received_scores(embeds)


def visionzip_select(
    embeds: torch.Tensor,
    keep_k: int,
    attn_scores: Optional[torch.Tensor] = None,
) -> List[int]:
    """VisionZip (Yang et al., CVPR'25): dominant + contextual tokens.

    Official protocol: (1) dominant tokens = top-attention tokens — scored
    by the model's *real* last-vision-block attention received (official
    ``qwen2_5vl_visionzip.py`` no-CLS reduction: head-mean softmax, summed
    over the query dim, 2x2 merge-group averaged), captured by the
    transformers patch and passed in as ``attn_scores`` (mandatory for
    paper-table arms, 2026-07-20 fidelity ruling); (2) the remaining tokens
    are uniformly split into *target* and *merge* sets and merge tokens are
    averaged into their most key-similar target, producing contextual
    tokens. Budget split follows the official 54:10 dominant:contextual
    ratio, rescaled to our keep budget.

    Adaptation note:
    our serving path is a position-preserving index-select over real tokens
    (required for the KV-consistent compact gather + deepstack rows), so
    synthetic merged embeddings cannot be injected. We therefore keep the
    *target* tokens of the contextual split un-merged — the tokens that
    would have received the merges. Target construction follows the
    official ``clip_encoder.py`` uniform stride over the non-dominant
    remainder in original order; official uses the integer stride
    ``arange(0, m, m//ctx)[:ctx]`` while we use the float stride
    ``floor(i*m/ctx)`` (identical whenever ``m % ctx == 0``, and an equally
    uniform cover otherwise — both are "uniformly split" per the paper).
    The dominant component (84% of the budget) is untouched by this
    adaptation and matches the official ranking exactly.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    dom_k = min(max(1, int(round(keep_k * visionzip_dominant_frac()))), keep_k)
    ctx_k = keep_k - dom_k

    scores = _resolve_attn_scores(embeds, attn_scores, "visionzip")
    order = torch.argsort(scores, descending=True)
    dominant = order[:dom_k].tolist()
    if ctx_k <= 0:
        return sorted(dominant)

    dom_mask = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    dom_mask[order[:dom_k]] = True
    # official target/merge construction: uniform split of the non-dominant
    # remainder in original (raster) order; targets are the kept side.
    remainder = torch.nonzero(~dom_mask, as_tuple=False).squeeze(-1)
    m = int(remainder.numel())
    if m <= ctx_k:
        return sorted(dominant + remainder.tolist())
    step = m / float(ctx_k)
    targets = [int(remainder[min(int(i * step), m - 1)]) for i in range(ctx_k)]
    return sorted(set(dominant + targets))


def vispruner_select(
    embeds: torch.Tensor,
    keep_k: int,
    attn_scores: Optional[torch.Tensor] = None,
    hybrid_mass: Optional[torch.Tensor] = None,
) -> List[int]:
    """VisPruner (Zhang et al., ICCV'25): important + diverse tokens.

    Official protocol (llava_arch.py, followed exactly): (1) rank all tokens
    by visual attention — the model's *real* last-vision-block attention
    received (captured by the transformers patch; mandatory ``attn_scores``
    for paper-table arms, 2026-07-20 fidelity ruling) — and keep the top
    ``important_ratio * keep_k`` as important tokens; (2) from the remainder
    (kept in saliency order), iteratively remove duplicates — split
    odd/even, cosine-match each even-half token to its most similar
    odd-half token, drop the ``k<=8`` highest-similarity even-half tokens
    per iteration — until exactly ``(1-r)*keep_k`` diverse tokens remain.
    Default important_ratio 0.5 (official scripts).

    ``hybrid_mass`` (E3 attribution control, GEOPACT_PUB_HYBRID_MASS=1):
    per-token mass m_j = 1 + α·P_j from the Module A spatial prior multiplied
    into the attention saliency before ranking (attention × mass); the
    duplicate-removal stage is unchanged. ``hybrid_mass=None`` = published rule.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    imp_k = min(max(0, int(round(keep_k * vispruner_important_ratio()))), keep_k)
    div_k = keep_k - imp_k

    scores = _resolve_attn_scores(embeds, attn_scores, "vispruner")
    if hybrid_mass is not None:
        mass = hybrid_mass.float().reshape(-1).to(scores.device)
        if int(mass.numel()) != n:
            raise ValueError(
                f"vispruner hybrid_mass length {int(mass.numel())} != n tokens {n}"
            )
        scores = scores.float() * mass
    order = torch.argsort(scores, descending=True)
    important = order[:imp_k]
    residual = order[imp_k:]

    if div_k <= 0:
        return sorted(important.tolist())

    feats_n = F.normalize(embeds.float(), p=2, dim=-1)
    while int(residual.numel()) > div_k:
        r_len = int(residual.numel())
        n_rm = min(8, r_len - div_k)
        a_idx, b_idx = residual[::2], residual[1::2]
        if n_rm <= 0 or int(b_idx.numel()) == 0 or n_rm > int(a_idx.numel()):
            break
        sim = torch.mm(feats_n[a_idx], feats_n[b_idx].t()).max(dim=-1).values
        keep_a = torch.argsort(sim, descending=True)[n_rm:]
        residual = torch.cat([a_idx[keep_a], b_idx])
    return sorted(important.tolist() + residual[:div_k].tolist())


def prumergeplus_select(
    embeds: torch.Tensor,
    keep_k: int,
    attn_scores: Optional[torch.Tensor] = None,
) -> List[int]:
    """PruMerge+ (Shang et al., ICCV'25): attention outliers + spatially
    uniform supplement (+ cluster merging).

    Official protocol: (1) rank tokens by visual attention — the model's
    *real* last-vision-block attention received (captured by the
    transformers patch; mandatory ``attn_scores`` for paper-table arms,
    2026-07-20 fidelity ruling) — and pick the IQR *outliers* as the
    informative set (adaptive, typically ~6% of tokens); (2) the "+"
    variant supplements with **spatially uniform** sampled tokens up to the
    target budget; (3) merge each pruned token into its k-NN kept cluster
    center (weighted average).

    Fixed-budget adaptation:
    our protocol requires an exact keep_k, so the adaptive IQR outlier set
    is rendered as the top ``GEOPACT_PRUMERGE_OUTLIER_FRAC`` (default 0.25,
    matching the official outlier:supplement proportion at the paper's 25%
    operating point) of the budget by attention received, and the uniform
    spatial supplement fills the remaining budget in raster order — the
    paper's own "+" design point. The merge step is replaced by keeping the
    cluster centers un-merged (position-preserving index-select serving
    path; same adaptation as VisionZip, applied uniformly to all
    merge-family baselines)."""
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    out_k = min(max(1, int(round(keep_k * prumerge_outlier_frac()))), keep_k)
    sup_k = keep_k - out_k

    scores = _resolve_attn_scores(embeds, attn_scores, "prumergeplus")
    order = torch.argsort(scores, descending=True)
    outliers = order[:out_k].tolist()
    if sup_k <= 0:
        return sorted(outliers)

    out_mask = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    out_mask[order[:out_k]] = True
    remainder = torch.nonzero(~out_mask, as_tuple=False).squeeze(-1)
    m = int(remainder.numel())
    if m <= sup_k:
        return sorted(outliers + remainder.tolist())
    step = m / float(sup_k)
    supplement = [int(remainder[min(int(i * step), m - 1)]) for i in range(sup_k)]
    return sorted(set(outliers + supplement))


def trim_select(
    embeds: torch.Tensor,
    keep_k: int,
    instr_query: Optional[torch.Tensor] = None,
    attn_scores: Optional[torch.Tensor] = None,
) -> List[int]:
    """TRIM (Song et al., COLING'25): text-relevance token reduction.

    Official protocol: score every visual token by its CLIP-space similarity
    to the text query, keep the IQR *outliers* of the similarity
    distribution, and aggregate all pruned tokens into one extra token.

    Fixed-budget adaptation:
    (1) the adaptive IQR count is replaced by the exact ``keep_k`` budget —
    top-k by the same relevance score (IQR typically retains ~5-8%; every
    budget we evaluate, r10-r50, is a superset, so this only *adds* tokens
    in relevance order); (2) similarity is computed against the instruction
    embedding through the same ``instr_query`` channel CDPruner uses (same
    text encoder surface, apples-to-apples); (3) the single aggregated token
    cannot be injected on our position-preserving serving path and is
    dropped — same merge->representative rule applied to VisionZip and
    PruMerge+. TRIM is the *pure text-relevance single-criterion*
    representative (contrast: CDPruner = relevance x diversity DPP); with
    no instruction available it falls back to attention received (same
    no-text fallback family as the other feature-level arms)."""
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    if instr_query is not None:
        feats = F.normalize(embeds.float(), p=2, dim=-1)
        q = instr_query.float().reshape(-1)
        q = F.normalize(q, p=2, dim=-1).to(feats.device)
        scores = torch.mv(feats, q)
    elif attn_scores is not None:
        scores = attn_scores.reshape(-1).float().to(embeds.device)
    else:
        scores = _attn_received_scores(embeds)
    order = torch.argsort(scores, descending=True)
    return sorted(order[:keep_k].tolist())


def visiontrim_local_importance(
    embeds: torch.Tensor,
    grid_h: int,
    grid_w: int,
    window: int = 3,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 0.5,
) -> torch.Tensor:
    """VisionTrim LTAM local importance (official ``build_token_affinity``,
    vectorised; per-index equal to the official nested-loop version).

    For every token (i, j) and each neighbour in its ``window x window``
    neighbourhood N(i,j) (self included): ``kappa = -||F_ij - F_n||^2 /
    (w1 sigma_feat) - w3 ||P_ij - P_n||^2 / (w2 sigma_pos)`` with scalar
    ``sigma_feat = Var(F)`` / ``sigma_pos = Var(P)`` over the whole frame
    (official semantics, unbiased torch default). Importance is the affinity
    row *mean over window^2 slots* — official zero-pads missing border
    neighbours, so the divisor is always window^2 — followed by a softmax
    over tokens. Defaults w1=1, w2=1, w3=0.5, window=3 (official).
    Returns (n,) fp32 softmax scores.
    """
    n = int(embeds.shape[0])
    if grid_h * grid_w != n:
        raise RuntimeError(
            f"visiontrim LTAM: grid {grid_h}x{grid_w} != tokens {n}"
        )
    feats = embeds.float().view(grid_h, grid_w, -1)
    sigma_feat = feats.var().clamp_min(1e-12)
    device = embeds.device
    ys = torch.arange(grid_h, dtype=torch.float32, device=device)
    xs = torch.arange(grid_w, dtype=torch.float32, device=device)
    pos = torch.stack(
        [ys.view(-1, 1).expand(grid_h, grid_w).reshape(-1),
         xs.view(1, -1).expand(grid_h, grid_w).reshape(-1)],
        dim=1,
    )
    sigma_pos = pos.var().clamp_min(1e-12)

    pad = window // 2
    kappa_sum = torch.zeros(grid_h, grid_w, dtype=torch.float32, device=device)
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            # valid centre region whose (di, dj) neighbour is in-grid
            r0, r1 = max(0, -di), grid_h - max(0, di)
            c0, c1 = max(0, -dj), grid_w - max(0, dj)
            if r0 >= r1 or c0 >= c1:
                continue
            cur = feats[r0:r1, c0:c1]
            nbr = feats[r0 + di:r1 + di, c0 + dj:c1 + dj]
            feat_term = ((cur - nbr) ** 2).sum(dim=-1) / (w1 * sigma_feat)
            pos_term = float(di * di + dj * dj) / (w2 * float(sigma_pos))
            kappa_sum[r0:r1, c0:c1] += -feat_term - w3 * pos_term
    importance = kappa_sum.reshape(-1) / float(window * window)
    return torch.softmax(importance, dim=-1)


def _visiontrim_combine_adaptive(
    global_scores: torch.Tensor, local_scores: torch.Tensor
) -> torch.Tensor:
    """Official ``combine_importance(..., method='adaptive')``: softmax-
    normalise both signals, then weight each by the *other's* variance
    (lower variance = more certain = higher weight)."""
    a = torch.softmax(global_scores.float(), dim=-1)
    b = torch.softmax(local_scores.float(), dim=-1)
    var_a = a.var()
    var_b = b.var()
    denom = (var_a + var_b).clamp_min(1e-12)
    return (var_b / denom) * a + (var_a / denom) * b


def visiontrim_select(
    embeds: torch.Tensor,
    keep_k: int,
    attn_scores: Optional[torch.Tensor] = None,
    instr_query: Optional[torch.Tensor] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
) -> List[int]:
    """VisionTrim (Yu et al., ICLR'26): DVTS dominant tokens + TGVC
    text-guided complement.

    Official protocol (hanxunyu/VisionTrim): (1) **DVTS** scores every token
    by a global-local view — global = ViT [CLS] attention, local = the LTAM
    window-affinity importance — fused by Adaptive Variance-based Weighting
    (``combine_importance(method='adaptive')``), and keeps the top
    ``DVTS_token_num``; (2) **TGVC** ranks the pruned remainder by CLIP-space
    text-to-image similarity to the instruction, takes the top
    ``TGVC_token_num`` as complement targets and iteratively merges the rest
    into them (3 text-guided assignment rounds). Budget split follows the
    official 48:16 DVTS:TGVC ratio, rescaled to our keep budget
    (``visiontrim_dvts_frac``).

    Adaptations:
    (a) Qwen3VL's ViT has no [CLS] token — the global signal is the *real*
    last-vision-block attention received captured by the transformers patch
    (mandatory ``attn_scores``; identical no-CLS mapping used for the
    VisionZip family, 2026-07-20 fidelity ruling). Note the released repo's
    final code line overwrites the paper's adaptive fusion with a plain
    0.5/0.5 mix of cls-attention and attention-received — two signals that
    coincide under the no-CLS mapping; we implement the paper's (and the
    repo's LTAM module's) global-local adaptive fusion, which preserves the
    method's stated contribution. (b) text relevance for TGVC is computed
    against the instruction embedding through the same ``instr_query``
    channel CDPruner/TRIM use (same text surface, apples-to-apples); with no
    instruction the complement falls back to attention-received order (same
    no-text fallback family as TRIM). (c) our serving path is a
    position-preserving index-select, so the TGVC merge rounds are rendered
    as keeping the complement *targets* un-merged — the same
    merge->representative rule applied to VisionZip / PruMerge+ / TRIM.
    (d) frames whose token count does not match the reported grid (no
    spatial layout available) drop the local LTAM term and rank by the
    global signal alone.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    dvts_k = min(max(1, int(round(keep_k * visiontrim_dvts_frac()))), keep_k)
    tgvc_k = keep_k - dvts_k

    global_scores = _resolve_attn_scores(embeds, attn_scores, "visiontrim")
    # Scale fidelity: the official global signal is a [CLS] attention *row*,
    # i.e. a probability distribution (sums to 1, per-token values ~1/n). Our
    # no-CLS capture is an attention-received column sum whose raw magnitude
    # is O(queries); feeding that into the adaptive fusion's softmax yields a
    # near-one-hot vector with huge variance, which zeroes the global weight
    # and leaves LTAM smoothness alone (selects background). Normalising to a
    # distribution restores the official scale (rankings are unaffected).
    gs = global_scores.reshape(-1).float()
    gs = gs / gs.sum().clamp_min(1e-12)
    if grid_hw is not None and int(grid_hw[0]) * int(grid_hw[1]) == n:
        local_scores = visiontrim_local_importance(
            embeds, int(grid_hw[0]), int(grid_hw[1])
        )
        metric = _visiontrim_combine_adaptive(gs, local_scores)
    else:
        metric = gs

    order = torch.argsort(metric, descending=True)
    dominant = order[:dvts_k].tolist()
    if tgvc_k <= 0:
        return sorted(dominant)

    dom_mask = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    dom_mask[order[:dvts_k]] = True
    remainder = torch.nonzero(~dom_mask, as_tuple=False).squeeze(-1)
    if int(remainder.numel()) <= tgvc_k:
        return sorted(dominant + remainder.tolist())
    if instr_query is not None:
        feats = F.normalize(embeds[remainder].float(), p=2, dim=-1)
        q = F.normalize(instr_query.float().reshape(-1), p=2, dim=-1).to(feats.device)
        rel = torch.mv(feats, q)
    else:
        rel = global_scores.reshape(-1).float()[remainder]
    top = torch.argsort(rel, descending=True)[:tgvc_k]
    complement = remainder[top].tolist()
    return sorted(dominant + complement)


def zooprune_select(
    embeds: torch.Tensor,
    keep_k: int,
    sens_scores: Optional[torch.Tensor] = None,
) -> List[int]:
    """ZOO-Prune (Kim et al., CVPR'26): zeroth-order sensitivity + SADS.

    Official protocol (AIM-SKKU/ZOO-Prune ``llava_arch.py``, followed
    exactly): token importance = zeroth-order projector sensitivity
    (``zoo_sensitivity_from_merger``, computed in the transformers patch —
    the scoring signal is method core, no proxy); selection = Sensitivity-
    Aware Diverse Selection (SADS): min-max normalise the sensitivity to a
    weight in [0, 1], seed with the most sensitive token, then greedily add
    the token maximising ``min-cosine-distance-to-selected x sensitivity
    weight``. Distances are computed on the same post-projector features the
    other feature-level arms score (official: ``image_features[0]``). Our
    greedy loop updates the min-distance vector incrementally — per-index
    identical to the official ``index_select(...).min(dim=0)`` form.

    ``sens_scores`` is mandatory: a missing vector under
    GEOPACT_ZOO_SENS_CAPTURE=1 is a plumbing bug and any fallback would
    silently swap out the method's core signal.
    """
    n = int(embeds.shape[0])
    keep_k = min(max(1, int(keep_k)), n)
    if keep_k >= n:
        return list(range(n))
    if sens_scores is None:
        raise RuntimeError(
            "zooprune: no zeroth-order sensitivity scores reached the "
            "selector (GEOPACT_ZOO_SENS_CAPTURE must be on for zooprune "
            "arms; refusing a proxy fallback)"
        )
    if int(sens_scores.numel()) != n:
        raise RuntimeError(
            f"zooprune: sens_scores rows {int(sens_scores.numel())} != embeds rows {n}"
        )
    s = sens_scores.reshape(-1).float().to(embeds.device)
    s_min, s_max = s.min(), s.max()
    if bool(s_max > s_min):
        sens_weight = (s - s_min) / (s_max - s_min + 1e-8)
    else:
        sens_weight = torch.ones_like(s)

    feats = F.normalize(embeds.float(), p=2, dim=-1)
    sim = torch.mm(feats, feats.t()).clamp(-1.0, 1.0)
    dist = 1.0 - sim

    selected = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    first = int(torch.argmax(sens_weight).item())
    selected[first] = True
    min_dist = dist[first].clone()
    for _ in range(keep_k - 1):
        scores = (min_dist * sens_weight).masked_fill(selected, float("-inf"))
        pick = int(torch.argmax(scores).item())
        selected[pick] = True
        min_dist = torch.minimum(min_dist, dist[pick])
    return sorted(torch.nonzero(selected, as_tuple=False).squeeze(-1).tolist())


def pub_baseline_select(
    policy: str,
    embeds: torch.Tensor,
    keep_k: int,
    instr_query: Optional[torch.Tensor] = None,
    attn_scores: Optional[torch.Tensor] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
    hybrid_mass: Optional[torch.Tensor] = None,
) -> List[int]:
    """Dispatch helper used by ``_hes_select``. Returns local keep indices.

    ``attn_scores``: per-token scores for the frame (same row order as
    ``embeds``) captured by the transformers patch — the real ViT
    attention-received under GEOPACT_VZ_ATTN_CAPTURE=1 (mandatory for
    visionzip / vispruner / prumergeplus / visiontrim paper arms, 2026-07-20
    fidelity ruling) or the zeroth-order projector sensitivity under
    GEOPACT_ZOO_SENS_CAPTURE=1 (mandatory for zooprune arms).
    ``grid_hw``: (grid_h, grid_w) of the frame's token layout (visiontrim's
    LTAM local term).
    ``hybrid_mass``: E3 spatial-prior mass (GEOPACT_PUB_HYBRID_MASS=1). Four
    selectors define a hybrid rule, each grafting the prior onto its *own*
    score channel: divprune (max-min utility), vispruner (attention saliency),
    cdpruner (DPP relevance), prunesid (PCA loading). Any other policy fails
    loud so a mis-labelled arm can never silently run the unmodified
    baseline."""
    p = str(policy or "").lower()
    if hybrid_mass is not None and not _policy_defines_hybrid(p):
        raise ValueError(
            "hybrid_mass is only defined for "
            f"{'/'.join(HYBRID_CAPABLE_SELECTORS)}, got {policy!r}"
        )
    if "prunesid" in p:
        return prunesid_select(embeds, keep_k, hybrid_mass=hybrid_mass)
    if "divprune" in p:
        return divprune_select(embeds, keep_k, hybrid_mass=hybrid_mass)
    if "cdpruner" in p:
        return cdpruner_select(embeds, keep_k, instr_query=instr_query,
                               hybrid_mass=hybrid_mass)
    if "dart" in p:
        return dart_select(embeds, keep_k)
    if "zooprune" in p:
        return zooprune_select(embeds, keep_k, sens_scores=attn_scores)
    # NOTE: "visiontrim" contains "trim" — keep this check before trim's.
    if "visiontrim" in p:
        return visiontrim_select(
            embeds, keep_k, attn_scores=attn_scores, instr_query=instr_query,
            grid_hw=grid_hw,
        )
    if "visionzip" in p:
        return visionzip_select(embeds, keep_k, attn_scores=attn_scores)
    if "vispruner" in p:
        return vispruner_select(
            embeds, keep_k, attn_scores=attn_scores, hybrid_mass=hybrid_mass
        )
    if "prumergeplus" in p or "prumerge" in p:
        return prumergeplus_select(embeds, keep_k, attn_scores=attn_scores)
    if "trim" in p:
        return trim_select(embeds, keep_k, instr_query=instr_query, attn_scores=attn_scores)
    raise ValueError(f"unknown published-baseline policy: {policy!r}")


# ---------------------------------------------------------------------------
# FastV (inline, transformers patch side)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (b, heads, s, hd); cos/sin: (b, s, hd)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def visual_segments(visual_mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
    """Contiguous [start, end) runs of visual positions (one run per image)."""
    idx = torch.nonzero(visual_mask_1d, as_tuple=False).squeeze(-1).tolist()
    segs: List[Tuple[int, int]] = []
    run_start = None
    prev = None
    for i in idx:
        if run_start is None:
            run_start = i
        elif i != prev + 1:
            segs.append((run_start, prev + 1))
            run_start = i
        prev = i
    if run_start is not None:
        segs.append((run_start, prev + 1))
    return segs


def fastv_prefill_ctx(
    *,
    past_key_values,
    hidden_states: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
) -> Optional[dict]:
    """Return a FastV context dict when the current forward is an eligible
    full prefill (batch 1, no cached tokens, visual tokens present)."""
    if not fastv_enabled():
        return None
    if visual_pos_masks is None or hidden_states.shape[0] != 1:
        return None
    if hidden_states.shape[1] <= 1:
        return None
    past_len = 0
    if past_key_values is not None:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = 0
    if past_len > 0:
        return None
    vmask = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
    if int(vmask.sum().item()) == 0:
        return None
    drop_k = fastv_drop_k()
    if drop_k < 1:
        return None
    return {
        "drop_k": drop_k,
        "keep_cur": fastv_keep_cur(),
        "keep_hist": fastv_keep_hist(),
        "vmask": vmask.bool(),
        "scores": None,
    }


def pdrop_prefill_ctx(
    *,
    past_key_values,
    hidden_states: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
    num_layers: int,
) -> Optional[dict]:
    """Return a PyramidDrop context when the current forward is an eligible
    full prefill. Same eligibility rules as FastV; ``drop_layers`` is the
    list of pending stage boundaries and ``keep_cur``/``keep_hist`` are the
    *per-stage* retention ratios lambda = r_final^(1/3)."""
    if not pdrop_enabled():
        return None
    if visual_pos_masks is None or hidden_states.shape[0] != 1:
        return None
    if hidden_states.shape[1] <= 1:
        return None
    past_len = 0
    if past_key_values is not None:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = 0
    if past_len > 0:
        return None
    vmask = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
    if int(vmask.sum().item()) == 0:
        return None
    drop_layers = pdrop_drop_layers(num_layers)
    if not drop_layers:
        return None
    n_stages = len(drop_layers)
    lam_cur = pdrop_keep_cur() ** (1.0 / n_stages)
    lam_hist = pdrop_keep_hist() ** (1.0 / n_stages)
    return {
        "drop_layers": list(drop_layers),
        "keep_cur": lam_cur,
        "keep_hist": lam_hist,
        "vmask": vmask.bool(),
        "scores": None,
    }


@torch.no_grad()
def fastv_last_token_scores(
    decoder_layer,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Attention that every key position receives from the last prompt token
    at this layer (original FastV scoring; mean over heads, fp32 softmax)."""
    hs = decoder_layer.input_layernorm(hidden_states)
    attn = decoder_layer.self_attn
    bsz, seq, _ = hs.shape
    head_dim = attn.head_dim

    q = attn.q_proj(hs[:, -1:, :]).view(bsz, 1, -1, head_dim)
    q = attn.q_norm(q).transpose(1, 2)  # (b, H, 1, hd)
    k = attn.k_proj(hs).view(bsz, seq, -1, head_dim)
    k = attn.k_norm(k).transpose(1, 2)  # (b, KVH, seq, hd)

    cos, sin = position_embeddings
    q = _apply_rope(q, cos[:, -1:, :], sin[:, -1:, :])
    k = _apply_rope(k, cos, sin)

    groups = attn.num_key_value_groups
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)

    logits = torch.matmul(q.float(), k.float().transpose(2, 3)) * attn.scaling
    probs = torch.softmax(logits, dim=-1)  # last token attends to all keys
    return probs.mean(dim=1)[0, 0]  # (seq,)


def fastv_keep_index(ctx: dict, device) -> torch.Tensor:
    """Sequence positions to keep: all text tokens + per-segment top-score
    visual tokens (current segment budget ``keep_cur``, history ``keep_hist``)."""
    scores: torch.Tensor = ctx["scores"]
    vmask: torch.Tensor = ctx["vmask"].to(scores.device)
    seq = int(scores.shape[0])
    keep = torch.ones(seq, dtype=torch.bool, device=scores.device)
    segs = visual_segments(vmask)
    for seg_idx, (s, e) in enumerate(segs):
        is_current = seg_idx == len(segs) - 1
        ratio = ctx["keep_cur"] if is_current else ctx["keep_hist"]
        n = e - s
        k = min(max(1, int(round(n * float(ratio)))), n)
        if k >= n:
            continue
        seg_scores = scores[s:e]
        _, top = torch.topk(seg_scores, k=k, largest=True)
        keep[s:e] = False
        keep[s + top] = True
    return torch.nonzero(keep, as_tuple=False).squeeze(-1).to(device)


# ---------------------------------------------------------------------------
# IVC-Prune (Sun et al., ICLR'26) — inline, transformers patch side
# ---------------------------------------------------------------------------
#
# Official FireRedTeam/IVC-Prune Qwen2.5-VL protocol
# (transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py,
# ``IVCP_forward_flashattention``): at layer k-1 capture that layer's value
# states; at layer k (single selection) keep
#   (1) IVC tokens — top positions by the per-token mean of the RoPE cos and
#       sin embeddings over the image span (real-axis / 90-degree rotation
#       references; official k_c = 10% per score at 50% retention),
#   (2) foreground tokens — two-stage value-vector similarity: text-suffix
#       queries -> image keys (softmax value "attention", mean over heads),
#       seeds = top 1%, then seed queries -> image keys; final score = mean of
#       both score sets; fill the remaining budget in score order,
# then physically prune hidden states while PRESERVING the original position
# ids, and retroactively prune the KV cache of all earlier layers.
#
# Adaptations: the released code hard-codes
# the 50% retention point (target 0.5, k_c 0.1 per score); our grids need
# r in {50,25,10,5}%, so both scale proportionally (per-score IVC fraction
# 0.2*r, seed fraction stays at the official 1%, target exactly r per image
# segment) — this preserves the official IVC:foreground budget split (10:40)
# at every rung and satisfies the exact-budget parity protocol. Multi-image
# steps prune every segment at its own (c, h) budget like FastV/DART. The
# selection layer k follows the official 36-layer Qwen2.5-VL-3B setting
# (ivcp_k=22, VLMEvalKit config) since GUI-Owl-1.5-8B's decoder is 36 layers;
# override via GEOPACT_IVCP_K.

def ivcp_enabled() -> bool:
    return os.environ.get("GEOPACT_IVCP", "").strip() == "1"


def ivcp_k() -> int:
    raw = os.environ.get("GEOPACT_IVCP_K", "").strip()
    try:
        return max(1, int(raw)) if raw else 22
    except ValueError:
        return 22


def ivcp_keep_cur() -> float:
    val = _env_float("GEOPACT_IVCP_KEEP_CUR")
    return float(val) if val is not None else 0.5


def ivcp_keep_hist() -> float:
    val = _env_float("GEOPACT_IVCP_KEEP_HIST")
    return float(val) if val is not None else ivcp_keep_cur()


def ivcp_prefill_ctx(
    *,
    past_key_values,
    hidden_states: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
) -> Optional[dict]:
    """Return an IVC-Prune context when the current forward is an eligible
    full prefill (batch 1, no cached tokens, visual tokens present) — same
    eligibility rules as FastV/DART."""
    if not ivcp_enabled():
        return None
    if visual_pos_masks is None or hidden_states.shape[0] != 1:
        return None
    if hidden_states.shape[1] <= 1:
        return None
    past_len = 0
    if past_key_values is not None:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = 0
    if past_len > 0:
        return None
    vmask = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
    if int(vmask.sum().item()) == 0:
        return None
    return {
        "k": ivcp_k(),
        "keep_cur": ivcp_keep_cur(),
        "keep_hist": ivcp_keep_hist(),
        "vmask": vmask.bool(),
        "value_states": None,
    }


@torch.no_grad()
def ivcp_value_states(decoder_layer, hidden_states: torch.Tensor) -> torch.Tensor:
    """Value states of this layer's input (b=1) -> (KVH, seq, hd).

    The official patch captures the layer-(k-1) value states inside its
    attention forward (``output_value_states=True``, after ``repeat_kv``);
    values carry no RoPE, so recomputing ``v_proj`` of the layer input is the
    identical tensor. Head-mean over kv heads equals the official head-mean
    over the repeated full heads.
    """
    hs = decoder_layer.input_layernorm(hidden_states)
    attn = decoder_layer.self_attn
    bsz, seq, _ = hs.shape
    v = attn.v_proj(hs).view(bsz, seq, -1, attn.head_dim).transpose(1, 2)
    return v[0]  # (KVH, seq, hd)


def _ivcp_value_attention(
    value_states: torch.Tensor,
    query_idx: torch.Tensor,
    img_start: int,
    img_len: int,
) -> torch.Tensor:
    """Official ``calculate_avg_similarity_for_subset_einsum`` (b=1): softmax
    value similarity of the query rows against the image-span keys, averaged
    over heads -> (n_query, img_len)."""
    v = value_states  # (H, S, D)
    head_dim = int(v.shape[-1])
    vq = v[:, query_idx, :]
    vk = v[:, img_start:img_start + img_len, :].transpose(1, 2)
    sim = torch.einsum("hqd,hdk->hqk", vq, vk) / math.sqrt(head_dim)
    weights = F.softmax(sim, dim=-1, dtype=torch.float32).to(v.dtype)
    return weights.mean(dim=0)


def ivcp_keep_index(
    ctx: dict,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    device,
) -> torch.Tensor:
    """Kept sequence positions per the official selection at layer k.

    Missing value-state capture is a hard failure (method-core signal)."""
    value_states = ctx.get("value_states")
    if value_states is None:
        raise RuntimeError(
            "GEOPACT_IVCP=1 requires the layer-(k-1) value-state capture; "
            "got none at the selection layer (check GEOPACT_IVCP_K)."
        )
    vmask: torch.Tensor = ctx["vmask"]
    seq = int(vmask.shape[0])
    cos, sin = position_embeddings
    segs = visual_segments(vmask)
    if not segs:
        return torch.arange(seq, device=device)

    # official text queries: everything past the (last) image end + 2
    last_end = segs[-1][1]
    q_start = min(last_end + 2, seq)
    text_query_idx = torch.arange(q_start, seq, device=value_states.device)
    if int(text_query_idx.numel()) == 0:
        text_query_idx = torch.arange(last_end, seq, device=value_states.device)
    if int(text_query_idx.numel()) == 0:
        raise RuntimeError("IVC-Prune: no text suffix after the image span.")

    keep = torch.ones(seq, dtype=torch.bool, device=value_states.device)
    for seg_idx, (s, e) in enumerate(segs):
        is_current = seg_idx == len(segs) - 1
        ratio = float(ctx["keep_cur"] if is_current else ctx["keep_hist"])
        n = e - s
        target = min(max(1, int(round(n * ratio))), n)
        if target >= n:
            continue

        text_scores = _ivcp_value_attention(value_states, text_query_idx, s, n)
        text_mean = text_scores.float().mean(dim=0)
        n_seed = max(1, int(round(n * 0.01)))
        seeds_local = torch.topk(text_mean, k=min(n_seed, n)).indices
        seed_scores = _ivcp_value_attention(
            value_states, seeds_local + s, s, n
        )
        final_mean = torch.cat(
            (text_scores.float(), seed_scores.float()), dim=0
        ).mean(dim=0)

        # IVC tokens: per-token mean of the RoPE cos / sin over the segment,
        # every non-sequence dim reduced (official mean(dim=(0, 2))).
        seg_cos = cos[..., s:e, :].float()
        seg_sin = sin[..., s:e, :].float()
        red = tuple(i for i in range(seg_cos.dim()) if seg_cos.shape[i] != n or i == seg_cos.dim() - 1)
        cos_mean = seg_cos.mean(dim=red)
        sin_mean = seg_sin.mean(dim=red)
        n_pe = min(max(0, int(round(n * 0.2 * ratio))), n)
        pe_local = torch.cat((
            torch.topk(cos_mean, k=n_pe).indices,
            torch.topk(sin_mean, k=n_pe).indices,
        )) if n_pe > 0 else torch.empty(0, dtype=torch.long, device=text_mean.device)

        priority = torch.unique(torch.cat((pe_local, seeds_local)))
        remaining = target - int(priority.numel())
        if remaining > 0:
            ranked = torch.topk(final_mean, k=n).indices
            fill_mask = ~torch.isin(ranked, priority)
            extra = ranked[fill_mask][:remaining]
            selected = torch.cat((priority, extra))
        else:
            # scaled fractions keep |priority| <= target at every grid rung;
            # truncate by foreground score if a degenerate case ever exceeds it
            order = torch.argsort(final_mean[priority], descending=True)
            selected = priority[order][:target]

        keep[s:e] = False
        keep[selected + s] = True

    return torch.nonzero(keep, as_tuple=False).squeeze(-1).to(device)


def ivcp_prune_past_key_values(past_key_values, keep_idx: torch.Tensor, current_layer_idx: int) -> None:
    """Official ``prune_past_key_values``: retroactively subset the KV cache
    of every layer before the selection layer (single-selection strategy)."""
    if past_key_values is None:
        return
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is None or value_cache is None:
        return
    for layer_idx in range(min(len(key_cache), current_layer_idx)):
        k = key_cache[layer_idx]
        if k is not None and k.numel() > 0:
            key_cache[layer_idx] = k[:, :, keep_idx.to(k.device), :]
        v = value_cache[layer_idx]
        if v is not None and v.numel() > 0:
            value_cache[layer_idx] = v[:, :, keep_idx.to(v.device), :]
    if hasattr(past_key_values, "_seen_tokens"):
        past_key_values._seen_tokens = int(keep_idx.numel())


def fastv_apply_prune(
    *,
    ctx: dict,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_position_ids: Optional[torch.Tensor],
    attention_mask,
    cache_position: Optional[torch.Tensor],
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds: Optional[list],
    keep_idx: Optional[torch.Tensor] = None,
):
    """Physically subset the prefill state to the keep set (FastV scoring by
    default; in-LLM DART passes its own precomputed ``keep_idx``).

    Returns (hidden_states, position_embeddings, attention_position_ids,
    attention_mask, cache_position, visual_pos_masks, deepstack_visual_embeds,
    n_dropped).
    """
    if keep_idx is None:
        keep_idx = fastv_keep_index(ctx, hidden_states.device)
    seq = int(hidden_states.shape[1])
    n_keep = int(keep_idx.numel())
    if n_keep >= seq:
        return (
            hidden_states, position_embeddings, attention_position_ids,
            attention_mask, cache_position, visual_pos_masks,
            deepstack_visual_embeds, 0,
        )

    hidden_states = hidden_states.index_select(1, keep_idx)
    cos, sin = position_embeddings
    position_embeddings = (
        cos.index_select(1, keep_idx.to(cos.device)),
        sin.index_select(1, keep_idx.to(sin.device)),
    )
    if attention_position_ids is not None:
        attention_position_ids = attention_position_ids.index_select(
            -1, keep_idx.to(attention_position_ids.device)
        )
    if cache_position is not None:
        cache_position = cache_position.index_select(0, keep_idx.to(cache_position.device))
    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
        if attention_mask.dim() == 4:
            m_idx = keep_idx.to(attention_mask.device)
            attention_mask = attention_mask.index_select(2, m_idx).index_select(3, m_idx)
        else:
            attention_mask = attention_mask.index_select(-1, keep_idx.to(attention_mask.device))

    if visual_pos_masks is not None:
        vmask_full = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
        vmask_full = vmask_full.bool()
        # flat index (within the visual-token ordering) of every kept visual row
        vis_order = torch.cumsum(vmask_full.long(), dim=0) - 1
        kept_vis_mask = vmask_full[keep_idx]
        kept_vis_flat = vis_order[keep_idx[kept_vis_mask]]
        new_vmask = kept_vis_mask.unsqueeze(0)
        if deepstack_visual_embeds is not None:
            flat_idx = kept_vis_flat.to(deepstack_visual_embeds[0].device)
            deepstack_visual_embeds = [
                e.index_select(0, flat_idx) for e in deepstack_visual_embeds
            ]
        visual_pos_masks = new_vmask.to(visual_pos_masks.device)

    return (
        hidden_states, position_embeddings, attention_position_ids,
        attention_mask, cache_position, visual_pos_masks,
        deepstack_visual_embeds, seq - n_keep,
    )
