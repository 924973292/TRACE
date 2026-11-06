#!/usr/bin/env python3
"""Verify token-level ICKH and oas MIU routing on dialogue sticky path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from lmms_eval.models.geopact_vl_hook import (
    _build_cdct_density_field,
    _cdct_alpha,
    _hes_select,
    _hes_token_kernel_herding,
    _is_dialogue_sticky_embed_prune,
    _is_dialogue_sticky_token_prune,
    _map_mius_to_tokens,
    _needs_sticky_relevance_query_cache,
    _policy_has,
    _policy_is_token_baseline,
    _policy_uses_cdct,
    _policy_uses_miu_first_backfill,
    _prune_single_frame,
    _token_ickh_herding_weights_into,
    _uniform_keep_indices,
    cache_sticky_relevance_query,
    compute_sticky_dialogue_prune_keep_set,
    get_sticky_relevance_query,
)


def _hes_token_kernel_herding_ref(
    image_embeds: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    keep_k_tokens: int,
    policy: str,
):
    """Reference loop (CPU sync per pick) — regression only."""
    import torch.nn.functional as F

    n_img = int(image_embeds.shape[0])
    keep_k = min(max(1, int(keep_k_tokens)), n_img)
    if keep_k >= n_img:
        return set(range(n_img))

    E_norm = F.normalize(image_embeds.float(), dim=-1)
    K = E_norm @ E_norm.T
    device = image_embeds.device

    if "token_ickh_diversity" in policy.lower():
        weights = torch.ones(n_img, dtype=torch.float32, device=device)
    else:
        mask = image_mask[0] if image_mask.dim() > 1 else image_mask
        mask = mask.bool()
        text_embs = inputs_embeds[0][~mask]
        if text_embs.shape[0] > 0:
            q = F.normalize(text_embs.float().mean(0, keepdim=True), dim=-1)
            weights = torch.clamp((E_norm @ q.T).squeeze(-1), min=0.0)
        else:
            weights = torch.ones(n_img, dtype=torch.float32, device=device)
        if float(weights.sum()) < 1e-8:
            weights = torch.ones(n_img, dtype=torch.float32, device=device)

    mu = (K @ weights) / weights.sum()
    selected: list[int] = []
    neg_inf = float("-inf")
    for _ in range(keep_k):
        if selected:
            s_t = torch.tensor(selected, dtype=torch.long, device=device)
            scores = mu - K[:, s_t].mean(dim=1)
        else:
            scores = mu.clone()
        for prev in selected:
            scores[prev] = neg_inf
        best = int(scores.argmax().item())
        selected.append(best)
    return set(selected)


def _ok(name: str, detail: str = "") -> None:
    print(f"[OK] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""), file=sys.stderr)


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            _ok(name, detail)
        else:
            _fail(name, detail)
            failures.append(name)

    for pol in (
        "uniform",
        "random",
        "token_ickh_diversity",
        "token_ickh_relevance",
        "token_instr_topk",
        "token_div_topk",
    ):
        check(
            f"dialogue sticky prune: {pol}",
            _is_dialogue_sticky_token_prune(pol),
            f"_is_dialogue_sticky_token_prune={_is_dialogue_sticky_token_prune(pol)}",
        )

    check(
        "oas_ickh excluded from token-only sticky",
        not _is_dialogue_sticky_token_prune("oas_ickh_coord_center_yolo"),
    )
    check(
        "oas_ickh on dialogue embed prune path",
        _is_dialogue_sticky_embed_prune("oas_ickh_coord_center_yolo"),
    )
    check(
        "token_ickh still token_baseline alias",
        _policy_is_token_baseline("token_ickh_diversity"),
    )
    check(
        "oas_ickh not token_baseline",
        not _policy_is_token_baseline("oas_ickh_coord_center_yolo"),
    )
    check(
        "oas_ickh needs sticky relevance cache",
        _needs_sticky_relevance_query_cache("oas_ickh_coord_center_yolo"),
    )
    check(
        "token_ickh_diversity no sticky relevance cache",
        not _needs_sticky_relevance_query_cache("token_ickh_diversity"),
    )

    oas_nd_yolo = "oas_ickh_no_density_no_coord_yolo"
    check(
        "oas no_density yolo on dialogue embed prune path",
        _is_dialogue_sticky_embed_prune(oas_nd_yolo),
    )
    check(
        "oas no_density yolo policy has no_density",
        _policy_has(oas_nd_yolo, "no_density"),
    )
    check(
        "oas no_density yolo needs sticky relevance cache",
        _needs_sticky_relevance_query_cache(oas_nd_yolo),
    )
    check(
        "oas no_density yolo not token_baseline",
        not _policy_is_token_baseline(oas_nd_yolo),
    )

    oas_cdct = "oas_ickh_cdct_no_coord_yolo"
    check("cdct policy recognized", _policy_uses_cdct(oas_cdct))
    check(
        "cdct on dialogue embed prune path",
        _is_dialogue_sticky_embed_prune(oas_cdct),
    )
    check(
        "cdct needs sticky relevance cache",
        _needs_sticky_relevance_query_cache(oas_cdct),
    )
    check("cdct default alpha", abs(_cdct_alpha(oas_cdct) - 1.0) < 1e-6)

    # Dialogue sticky oas branch delegates to _prune_single_frame (mock detect).
    from unittest.mock import patch
    from types import SimpleNamespace

    mock_model = SimpleNamespace(
        _geopact_episode_key="ep1",
        _geopact_image_paths=["/tmp/frame.png"],
        _geopact_prune_scope="image_sticky",
        _geopact_miu_detector="omniparser",
        _geopact_last_image=object(),
        _geopact_sticky_masks={},
    )
    emb_sticky = torch.randn(20, 8)
    grid_sticky = torch.tensor([[1, 4, 5]])
    with patch(
        "lmms_eval.models.geopact_vl_hook._prune_single_frame",
        return_value=({3, 7, 11, 15}, [], [{"box": (0, 0, 1, 1)}], None),
    ) as mock_prune:
        keep = compute_sticky_dialogue_prune_keep_set(
            mock_model,
            token_ranges=[(0, 20)],
            image_embeds=emb_sticky,
            image_grid_thw=grid_sticky,
            ratio=0.2,
            policy="oas_ickh_coord_center_yolo",
            scope="image_sticky",
        )
    check("oas dialogue sticky calls _prune_single_frame", mock_prune.called)
    check("oas dialogue sticky keep set", keep == {3, 7, 11, 15})

    torch.manual_seed(0)
    n_img = 48
    hidden = 32
    seq_len = 20
    image_embeds = torch.randn(n_img, hidden)
    inputs_embeds = torch.randn(1, seq_len, hidden)
    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    image_mask[8:16] = True
    # Make text direction align with first vision tokens for relevance test.
    inputs_embeds[0, 8:12] = image_embeds[:4] + 0.1

    keep_k = max(1, n_img // 5)
    grid = torch.tensor([[1, 6, 8]])

    uniform = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="uniform",
        )
    )
    div1 = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_ickh_diversity",
        )
    )
    div2 = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_ickh_diversity",
        )
    )
    rel = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_ickh_relevance",
        )
    )

    check("uniform keep count", len(uniform) == keep_k, f"got {len(uniform)}")
    check("diversity keep count", len(div1) == keep_k, f"got {len(div1)}")
    check("relevance keep count", len(rel) == keep_k, f"got {len(rel)}")
    check("diversity deterministic", div1 == div2)
    check(
        "diversity != uniform (non-degenerate)",
        div1 != uniform,
        f"uniform={sorted(uniform)[:6]} div={sorted(div1)[:6]}",
    )
    check(
        "relevance != diversity (non-degenerate)",
        rel != div1,
        f"rel={sorted(rel)[:6]} div={sorted(div1)[:6]}",
    )

    # Expected uniform indices for sanity
    expected_uniform = _uniform_keep_indices(n_img, keep_k)
    check("uniform matches helper", uniform == expected_uniform)

    # Vectorized ICKH must match reference greedy picks (larger n).
    torch.manual_seed(7)
    n_big = 256
    hidden_big = 64
    emb = torch.randn(n_big, hidden_big)
    in_emb = torch.randn(1, 24, hidden_big)
    im_mask = torch.zeros(24, dtype=torch.bool)
    im_mask[4:20] = True
    in_emb[0, 4:12] = emb[:8] + 0.05
    for pol in ("token_ickh_diversity", "token_ickh_relevance"):
        kk = max(1, n_big // 4)
        fast = set(
            _hes_token_kernel_herding(emb, in_emb, im_mask, kk, pol)
        )
        ref = _hes_token_kernel_herding_ref(emb, in_emb, im_mask, kk, pol)
        check(f"{pol} matches reference @ n={n_big}", fast == ref, f"delta={len(fast ^ ref)}")

    # Relevance: cached global instruction q overrides suffix text.
    from types import SimpleNamespace

    class _MiniEmbed(torch.nn.Embedding):
        def __init__(self):
            super().__init__(128, hidden_big)

    mini = _MiniEmbed()
    torch.nn.init.normal_(mini.weight, std=0.02)
    inner = SimpleNamespace(
        _geopact_task_instruction_text="Navigate to Wi-Fi settings and connect.",
        _geopact_tokenizer=lambda text, return_tensors="pt", add_special_tokens=False: SimpleNamespace(
            input_ids=torch.tensor([[11, 12, 13, 14, 15]])
        ),
        get_input_embeddings=lambda: mini,
        _geopact_sticky_relevance_query=None,
        _geopact_sticky_relevance_query_source=None,
    )
    check("cache relevance query", cache_sticky_relevance_query(inner))
    q = get_sticky_relevance_query(inner)
    check("cached q present", q is not None)
    suffix_emb = torch.randn(1, 24, hidden_big)
    suffix_emb[0, 8:16] = emb[:8] * 0.1
    w_q = torch.empty(n_big)
    w_suffix = torch.empty(n_big)
    E_big = F.normalize(emb.to(torch.float32), dim=-1)
    _token_ickh_herding_weights_into(
        E_big, suffix_emb, im_mask, "token_ickh_relevance", w_q, instr_query=q
    )
    _token_ickh_herding_weights_into(
        E_big, suffix_emb, im_mask, "token_ickh_relevance", w_suffix, instr_query=None
    )
    check(
        "cached q != suffix-text weights",
        not torch.allclose(w_q, w_suffix, atol=1e-5),
    )
    rel_cached = set(
        _hes_token_kernel_herding(
            emb, suffix_emb, im_mask, kk, "token_ickh_relevance", instr_query=q
        )
    )
    rel_suffix = set(
        _hes_token_kernel_herding(
            emb, suffix_emb, im_mask, kk, "token_ickh_relevance", instr_query=None
        )
    )
    check(
        "cached q can change relevance keep set vs suffix",
        rel_cached != rel_suffix,
        f"cached={sorted(rel_cached)[:6]} suffix={sorted(rel_suffix)[:6]}",
    )

    instr_topk = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_instr_topk",
            instr_query=F.normalize(torch.randn(1, hidden), dim=-1),
        )
    )
    div_topk = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_div_topk",
        )
    )
    check("token_instr_topk keep count", len(instr_topk) == keep_k, f"got {len(instr_topk)}")
    check("token_div_topk keep count", len(div_topk) == keep_k, f"got {len(div_topk)}")
    check(
        "token_instr_topk != token_ickh_relevance",
        instr_topk != rel,
        f"instr_topk={sorted(instr_topk)[:6]} ickh_rel={sorted(rel)[:6]}",
    )
    check(
        "token_div_topk != token_ickh_diversity",
        div_topk != div1,
        f"div_topk={sorted(div_topk)[:6]} ickh_div={sorted(div1)[:6]}",
    )

    # CDCT: max-pool density + token budget + alpha=0 degeneracy.
    import os
    from lmms_eval.models.geopact_vl_hook import _map_mius_to_tokens

    mock_mius = [
        {"box": (0.0, 0.0, 1.0, 1.0), "info_density": 0.2},
        {"box": (0.2, 0.2, 0.45, 0.45), "info_density": 1.0},
    ]
    gh, gw, n_cdct = 4, 5, 20
    mapped_groups, _, _, mapped_src = _map_mius_to_tokens(
        mock_mius, image_embeds, gh, gw, n_cdct
    )
    miu_full = [[] for _ in mock_mius]
    for group, src in zip(mapped_groups, mapped_src):
        miu_full[int(src)] = list(group)
    P = _build_cdct_density_field(mock_mius, miu_full, n_cdct, image_embeds.device)
    if miu_full[1]:
        peak_tok = int(miu_full[1][0])
        check(
            "cdct max-pool child density wins overlap",
            float(P[peak_tok].item()) >= 0.99,
            f"P[{peak_tok}]={float(P[peak_tok].item())}",
        )
    cdct_keep = set(
        _hes_select(
            mius=mock_mius,
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy=oas_cdct,
            instr_query=F.normalize(torch.randn(1, hidden), dim=-1),
        )
    )
    check("cdct fills keep_k", len(cdct_keep) == keep_k, f"got {len(cdct_keep)}")
    _prev_alpha = os.environ.get("GEOPACT_CDCT_ALPHA")
    os.environ["GEOPACT_CDCT_ALPHA"] = "0"
    try:
        q_fix = F.normalize(torch.randn(1, hidden), dim=-1)
        cdct_a0 = set(
            _hes_select(
                mius=mock_mius,
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=grid,
                keep_k_tokens=keep_k,
                policy=oas_cdct,
                instr_query=q_fix,
            )
        )
        rel_same_q = set(
            _hes_select(
                mius=[],
                image_embeds=image_embeds,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                image_grid_thw=grid,
                keep_k_tokens=keep_k,
                policy="token_ickh_relevance",
                instr_query=q_fix,
            )
        )
        check("cdct alpha=0 matches token_ickh_relevance", cdct_a0 == rel_same_q)
    finally:
        if _prev_alpha is None:
            os.environ.pop("GEOPACT_CDCT_ALPHA", None)
        else:
            os.environ["GEOPACT_CDCT_ALPHA"] = _prev_alpha

    # CDCT with 0 MIUs: must prune (no spatial prior) — same as token_ickh_relevance.
    q_zero = F.normalize(torch.randn(1, hidden), dim=-1)
    cdct_empty = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy=oas_cdct,
            instr_query=q_zero,
        )
    )
    rel_empty = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_ickh_relevance",
            instr_query=q_zero,
        )
    )
    check("cdct 0-MIU prunes to keep_k", len(cdct_empty) == keep_k, f"got {len(cdct_empty)}")
    check("cdct 0-MIU matches token_ickh_relevance", cdct_empty == rel_empty)

    from unittest.mock import patch
    from types import SimpleNamespace

    mock_model = SimpleNamespace(_geopact_last_prune_diag={})
    frame_embeds = image_embeds[:n_img]
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[],
    ):
        keep_global, _, _, _ = _prune_single_frame(
            start=0,
            end=n_img,
            image_embeds=frame_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy=oas_cdct,
            detector="omniparser",
            pil_image=object(),
            rng=None,
            model=mock_model,
            instr_query=q_zero,
        )
    check(
        "prune_single_frame CDCT 0-MIU keeps keep_k",
        len(keep_global) == keep_k,
        f"got {len(keep_global)}",
    )
    diag = getattr(mock_model, "_geopact_last_prune_diag", {}) or {}
    check("prune_single_frame CDCT diag fallback", diag.get("cdct_fallback") == "no_spatial_prior")
    check("prune_single_frame CDCT diag n_mius", diag.get("n_mius_detected") == 0)

    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[],
    ):
        legacy_global, _, _, _ = _prune_single_frame(
            start=0,
            end=n_img,
            image_embeds=frame_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy="oas_ickh_no_density_no_coord_yolo",
            detector="omniparser",
            pil_image=object(),
            rng=None,
            model=SimpleNamespace(_geopact_last_prune_diag={}),
            instr_query=q_zero,
        )
    check(
        "prune_single_frame legacy 0-MIU full keep",
        len(legacy_global) == n_img,
        f"got {len(legacy_global)}",
    )

    # MIU-first + token backfill (04x hybrid B).
    miu_backfill_pol = (
        "oas_ickh_miu_first_token_backfill_no_density_no_coord_yolo"
    )
    check("miu_first_backfill policy recognized", _policy_uses_miu_first_backfill(miu_backfill_pol))
    check(
        "miu_first_backfill on dialogue embed prune path",
        _is_dialogue_sticky_embed_prune(miu_backfill_pol),
    )
    check(
        "miu_first_backfill needs sticky relevance cache",
        _needs_sticky_relevance_query_cache(miu_backfill_pol),
    )
    check(
        "miu_first_backfill not token_baseline",
        not _policy_is_token_baseline(miu_backfill_pol),
    )

    # Sparse MIUs: union under-fills keep_k → backfill must reach keep_k.
    sparse_mius = [
        {"box": (0.0, 0.0, 0.12, 0.12), "info_density": 1.0, "energy": 1.0},
        {"box": (0.15, 0.15, 0.22, 0.22), "info_density": 0.8, "energy": 0.8},
    ]
    gh_s, gw_s, n_sparse = 6, 8, n_img
    mapped_sparse, _, _, _ = _map_mius_to_tokens(
        sparse_mius, image_embeds, gh_s, gw_s, n_sparse
    )
    union_sparse: set[int] = set()
    for group in mapped_sparse:
        union_sparse.update(int(i) for i in group)
    check(
        "sparse MIU union smaller than keep_k (backfill scenario)",
        len(union_sparse) < keep_k,
        f"union={len(union_sparse)} keep_k={keep_k}",
    )
    q_hybrid = F.normalize(torch.randn(1, hidden), dim=-1)
    hybrid_keep = set(
        _hes_select(
            mius=sparse_mius,
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy=miu_backfill_pol,
            instr_query=q_hybrid,
        )
    )
    check(
        "miu_first_backfill fills keep_k",
        len(hybrid_keep) == keep_k,
        f"got {len(hybrid_keep)}",
    )
    outside_union = hybrid_keep - union_sparse
    check(
        "miu_first_backfill can pick tokens outside MIU union",
        len(outside_union) > 0,
        f"outside={len(outside_union)}",
    )

    # 0-MIU fallback: same as token_ickh_relevance (not legacy full-keep).
    hybrid_empty = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy=miu_backfill_pol,
            instr_query=q_hybrid,
        )
    )
    rel_hybrid_q = set(
        _hes_select(
            mius=[],
            image_embeds=image_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            keep_k_tokens=keep_k,
            policy="token_ickh_relevance",
            instr_query=q_hybrid,
        )
    )
    check("miu_first_backfill 0-MIU prunes to keep_k", len(hybrid_empty) == keep_k)
    check("miu_first_backfill 0-MIU matches token_ickh_relevance", hybrid_empty == rel_hybrid_q)

    hybrid_model = SimpleNamespace(_geopact_last_prune_diag={})
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[],
    ):
        hybrid_global, _, _, _ = _prune_single_frame(
            start=0,
            end=n_img,
            image_embeds=frame_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy=miu_backfill_pol,
            detector="omniparser",
            pil_image=object(),
            rng=None,
            model=hybrid_model,
            instr_query=q_hybrid,
        )
    check(
        "prune_single_frame miu_first_backfill 0-MIU keeps keep_k",
        len(hybrid_global) == keep_k,
        f"got {len(hybrid_global)}",
    )
    hybrid_diag = getattr(hybrid_model, "_geopact_last_prune_diag", {}) or {}
    check(
        "prune_single_frame miu_first_backfill diag selector",
        hybrid_diag.get("prune_selector") == "miu_first_token_backfill",
    )

    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=sparse_mius,
    ):
        hybrid_sparse_model = SimpleNamespace(_geopact_last_prune_diag={})
        hybrid_sparse_global, _, _, _ = _prune_single_frame(
            start=0,
            end=n_img,
            image_embeds=frame_embeds,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy=miu_backfill_pol,
            detector="omniparser",
            pil_image=object(),
            rng=None,
            model=hybrid_sparse_model,
            instr_query=q_hybrid,
        )
    check(
        "prune_single_frame miu_first_backfill sparse MIUs fills keep_k",
        len(hybrid_sparse_global) == keep_k,
        f"got {len(hybrid_sparse_global)}",
    )
    sparse_diag = getattr(hybrid_sparse_model, "_geopact_last_prune_diag", {}) or {}
    check(
        "prune_single_frame miu_first_backfill records backfill tokens",
        int(sparse_diag.get("miu_backfill_tokens", 0) or 0) > 0,
        f"backfill={sparse_diag.get('miu_backfill_tokens')}",
    )
    check(
        "prune_single_frame miu_first_backfill keep_fill_ratio=1",
        float(sparse_diag.get("keep_fill_ratio", 0.0) or 0.0) >= 0.999,
        f"fill={sparse_diag.get('keep_fill_ratio')}",
    )

    from lmms_eval.models.geopact_vl_hook import (
        _dispatch_module_a_sideload,
        _flush_module_a_sideload_queue,
        _keep_local_hash,
        _module_a_sideload_enabled,
        _module_a_sideload_timing,
        _run_module_a_sideload,
    )

    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD", None)
    pure_model = SimpleNamespace(
        _geopact_episode_key="ep_pure",
        _geopact_image_paths=["pure.png"],
        _geopact_last_image=object(),
        _geopact_miu_detector="omniparser",
        _geopact_last_prune_diag={},
    )
    keep_pure = compute_sticky_dialogue_prune_keep_set(
        pure_model,
        token_ranges=[(0, n_img)],
        image_embeds=image_embeds,
        image_grid_thw=grid,
        ratio=keep_k / max(1, n_img),
        policy="token_ickh_relevance",
        scope="image_sticky",
        inputs_embeds=inputs_embeds,
        image_mask_1d=image_mask,
    )

    os.environ["GEOPACT_MODULE_A_SIDELOAD"] = "1"
    check("module_a sideload env enabled", _module_a_sideload_enabled())
    sideload_model = SimpleNamespace(
        _geopact_episode_key="ep_sideload",
        _geopact_image_paths=["sideload.png"],
        _geopact_last_image=object(),
        _geopact_miu_detector="omniparser",
        _geopact_last_prune_diag={},
        _geopact_last_compact_stats={"n_visual_total": n_img},
        _geopact_last_image_grid_thw=grid,
    )
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8}],
    ) as mock_detect:
        keep_sideload = compute_sticky_dialogue_prune_keep_set(
            sideload_model,
            token_ranges=[(0, n_img)],
            image_embeds=image_embeds,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy="token_ickh_relevance",
            scope="image_sticky",
            inputs_embeds=inputs_embeds,
            image_mask_1d=image_mask,
        )
    check(
        "sideload prune path does not call detector during B",
        not mock_detect.called,
        "detector must run post-decode only",
    )
    check(
        "sideload B keep matches pure token sticky path",
        keep_sideload == keep_pure,
        f"sideload={keep_sideload} pure={keep_pure}",
    )
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8}],
    ) as mock_post:
        os.environ.pop("MIU_YOLO_DEVICE", None)
        os.environ.pop("GEOPACT_MODULE_A_SIDELOAD_TIMING", None)
        check("default sideload timing is eval_end", _module_a_sideload_timing() == "eval_end")
        _dispatch_module_a_sideload(sideload_model, "token_ickh_relevance")
        check("eval_end dispatch queues sideload", len(getattr(sideload_model, "_geopact_sideload_queue", [])) == 1)
        check("eval_end dispatch does not run detector yet", not mock_post.called)
        _flush_module_a_sideload_queue(sideload_model)
    check("post-eval flush runs detector", mock_post.called)
    check(
        "post-decode sideload defaults CPU YOLO",
        mock_post.call_args.kwargs.get("defer_yolo_release") is True,
    )
    os.environ["GEOPACT_MODULE_A_SIDELOAD_TIMING"] = "post_decode"
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8}],
    ) as mock_decode:
        _dispatch_module_a_sideload(sideload_model, "token_ickh_relevance")
    check("post_decode dispatch runs detector immediately", mock_decode.called)

    sideload_model._geopact_sideload_queue = []
    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD_TIMING", None)
    os.environ["MIU_YOLO_DEVICE"] = "0"
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=[{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8}],
    ):
        with patch(
            "lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu",
        ) as mock_release:
            _dispatch_module_a_sideload(sideload_model, "token_ickh_relevance")
            _flush_module_a_sideload_queue(sideload_model)
    check("GPU eval_end flush calls maybe_release_yolo_gpu once", mock_release.call_count == 1)

    inline_model = SimpleNamespace(_geopact_last_prune_diag={})
    os.environ.pop("MIU_YOLO_DEVICE", None)
    with patch(
        "lmms_eval.models.geopact_module_a_cache.get_or_run_module_a",
        return_value=([{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8}], {}),
    ):
        with patch(
            "lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu",
        ) as mock_inline_release:
            from lmms_eval.models.geopact_vl_hook import _run_detector, release_pending_yolo_gpu

            _run_detector(None, "omniparser", "oas_ickh_cdct_no_coord_yolo", model=inline_model)
            check(
                "inline GPU YOLO defers release until step end",
                mock_inline_release.call_count == 0
                and getattr(inline_model, "_geopact_yolo_release_pending", False),
            )
            release_pending_yolo_gpu(inline_model)
            check(
                "release_pending_yolo_gpu flushes deferred YOLO release",
                not getattr(inline_model, "_geopact_yolo_release_pending", False)
                and mock_inline_release.call_count == 1
                and mock_inline_release.call_args.kwargs.get("allow_empty_cache") is False,
            )

    check(
        "keep_local_hash stable",
        _keep_local_hash(set(range(keep_k))) == _keep_local_hash(set(range(keep_k))),
    )
    sideload_diag = getattr(sideload_model, "_geopact_last_prune_diag", {}) or {}
    check("post-decode sideload diag flag", sideload_diag.get("module_a_sideload") is True)
    check(
        "post-decode sideload phase",
        sideload_diag.get("module_a_sideload_phase") == "post_decode",
    )
    check(
        "post-decode sideload selector",
        sideload_diag.get("prune_selector") == "token_ickh_relevance",
    )

    os.environ["GEOPACT_MODULE_A_SIDELOAD"] = "1"
    os.environ["GEOPACT_MODULE_A_SIDELOAD_MODE"] = "yolo_only"
    from lmms_eval.models.module_a_pipeline import run_module_a

    mius, meta = run_module_a(
        object(),
        "omniparser",
        "token_ickh_relevance",
        grid_hw=None,
        detect_raw_fn=lambda: ([{"box": (0, 0, 0.1, 0.1)}], None),
        apply_attribute_drops_fn=lambda m, p: m,
        policy_is_token_baseline_fn=lambda p: True,
        warn_skip_abc_fn=lambda msg: None,
        fail_fn=lambda msg, p: None,
    )
    check("yolo_only sideload mode returns detect mius", len(mius) == 1)
    check("yolo_only sideload meta", meta.get("sideload_mode") == "yolo_only")
    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD_MODE", None)
    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD", None)

    # CPU prefetch (L2): alpha=0 runs Module A; CDCT w_j=r_j → keep parity with pure B.
    from lmms_eval.models.geopact_vl_hook import (
        _assert_module_a_sideload_prefetch_exclusive,
        _module_a_prefetch_enabled,
        _module_a_prefetch_mode,
        _prefetch_alpha0_cdct,
    )

    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD", None)
    os.environ.pop("GEOPACT_MODULE_A_PREFETCH", None)
    check("prefetch disabled by default", not _module_a_prefetch_enabled())
    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "cpu"
    check("prefetch cpu mode", _module_a_prefetch_mode() == "cpu")

    prefetch_model = SimpleNamespace(
        _geopact_episode_key="ep_prefetch",
        _geopact_image_paths=["prefetch.png"],
        _geopact_last_image=object(),
        _geopact_miu_detector="omniparser",
        _geopact_last_prune_diag={},
        _geopact_last_compact_stats={"n_visual_total": n_img},
        _geopact_last_image_grid_thw=grid,
    )
    mock_miu = [{"box": (0.1, 0.1, 0.5, 0.5), "info_density": 0.8, "energy": 0.6}]
    _prev_alpha = os.environ.get("GEOPACT_CDCT_ALPHA")
    os.environ["GEOPACT_CDCT_ALPHA"] = "0"
    try:
        with patch(
            "lmms_eval.models.geopact_vl_hook._run_detector",
            return_value=mock_miu,
        ) as mock_prefetch_detect:
            keep_prefetch = compute_sticky_dialogue_prune_keep_set(
                prefetch_model,
                token_ranges=[(0, n_img)],
                image_embeds=image_embeds,
                image_grid_thw=grid,
                ratio=keep_k / max(1, n_img),
                policy=oas_cdct,
                scope="image_sticky",
                inputs_embeds=inputs_embeds,
                image_mask_1d=image_mask,
            )
        check("prefetch alpha=0 runs detector (spatial prior)", mock_prefetch_detect.called)
        check(
            "prefetch alpha=0 keep matches token_ickh_relevance",
            keep_prefetch == keep_pure,
            f"prefetch={keep_prefetch} pure={keep_pure}",
        )
        prefetch_diag = getattr(prefetch_model, "_geopact_last_prune_diag", {}) or {}
        check("prefetch diag flag", prefetch_diag.get("module_a_prefetch") == "cpu")
        check("prefetch alpha=0 diag", float(prefetch_diag.get("cdct_alpha", -1)) == 0.0)
        check("prefetch alpha=0 cdct flag", _prefetch_alpha0_cdct(oas_cdct))
    finally:
        if _prev_alpha is None:
            os.environ.pop("GEOPACT_CDCT_ALPHA", None)
        else:
            os.environ["GEOPACT_CDCT_ALPHA"] = _prev_alpha

    os.environ["GEOPACT_MODULE_A_SIDELOAD"] = "1"
    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "cpu"
    try:
        _assert_module_a_sideload_prefetch_exclusive()
        check("sideload+prefetch mutual exclusion", False, "expected RuntimeError")
    except RuntimeError:
        check("sideload+prefetch mutual exclusion", True)
    os.environ.pop("GEOPACT_MODULE_A_SIDELOAD", None)
    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "cpu"

    with patch(
        "lmms_eval.models.geopact_module_a_cache.get_or_run_module_a",
        side_effect=lambda pil, det, pol, fn, model=None: (fn(), {}),
    ):
        with patch(
            "lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu",
        ) as mock_prefetch_release:
            prev_yolo = os.environ.get("MIU_YOLO_DEVICE")
            os.environ.pop("MIU_YOLO_DEVICE", None)
            yolo_during: list = []

            def _capture_impl(*_args, **_kwargs):
                yolo_during.append(os.environ.get("MIU_YOLO_DEVICE"))
                return mock_miu

            from lmms_eval.models.geopact_vl_hook import _run_detector

            with patch(
                "lmms_eval.models.geopact_vl_hook._run_detector_impl",
                side_effect=_capture_impl,
            ):
                _run_detector(object(), "omniparser", oas_cdct, model=prefetch_model)
            check("prefetch forces CPU YOLO during detect", yolo_during == ["cpu"])
            check(
                "prefetch release skips empty_cache",
                mock_prefetch_release.call_args.kwargs.get("allow_empty_cache") is False,
            )
            if prev_yolo is None:
                os.environ.pop("MIU_YOLO_DEVICE", None)
            else:
                os.environ["MIU_YOLO_DEVICE"] = prev_yolo

    suffix_model = SimpleNamespace(
        _geopact_episode_key="ep_suffix",
        _geopact_image_paths=["suffix.png"],
        _geopact_last_image=object(),
        _geopact_miu_detector="omniparser",
        _geopact_last_prune_diag={},
        _geopact_dialogue_suffix_only_step=True,
        _geopact_sticky_masks={},
    )
    os.environ["GEOPACT_MODULE_A_PREFETCH"] = "cpu"
    os.environ["GEOPACT_CDCT_ALPHA"] = "0"
    check("prefetch alpha=0 cdct mode", _prefetch_alpha0_cdct(oas_cdct))
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=mock_miu,
    ) as mock_suffix_detect:
        keep_suffix = compute_sticky_dialogue_prune_keep_set(
            suffix_model,
            token_ranges=[(0, n_img)],
            image_embeds=image_embeds,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy=oas_cdct,
            scope="image_sticky",
            inputs_embeds=inputs_embeds,
            image_mask_1d=image_mask,
        )
    check("suffix prefetch alpha=0 runs detector", mock_suffix_detect.called)
    check(
        "suffix prefetch alpha=0 keep matches token_ickh_relevance",
        keep_suffix == keep_pure,
        f"suffix={keep_suffix} pure={keep_pure}",
    )

    os.environ["GEOPACT_SESSION_KV"] = "1"
    os.environ["GEOPACT_SESSION_KV_MODE"] = "dialogue"
    os.environ["GEOPACT_CDCT_ALPHA"] = "1"
    sticky_alpha1 = SimpleNamespace(
        _geopact_episode_key="ep_sticky",
        _geopact_image_paths=["seen.png"],
        _geopact_last_image=object(),
        _geopact_miu_detector="omniparser",
        _geopact_last_prune_diag={},
        _geopact_dialogue_suffix_only_step=True,
        _geopact_sticky_masks={("ep_sticky", "seen.png"): frozenset({1, 2, 3})},
        _geopact_last_compact_stats={"n_visual_total": n_img},
        _geopact_last_image_grid_thw=grid,
    )
    check("dialogue prefetch alpha=1 always runs detector", not _prefetch_alpha0_cdct(oas_cdct))
    with patch(
        "lmms_eval.models.geopact_vl_hook._run_detector",
        return_value=mock_miu,
    ) as mock_dialogue_detect:
        compute_sticky_dialogue_prune_keep_set(
            sticky_alpha1,
            token_ranges=[(0, n_img)],
            image_embeds=image_embeds,
            image_grid_thw=grid,
            ratio=keep_k / max(1, n_img),
            policy=oas_cdct,
            scope="image_sticky",
            inputs_embeds=inputs_embeds,
            image_mask_1d=image_mask,
        )
    check("dialogue prefetch alpha=1 runs detector despite sticky", mock_dialogue_detect.called)

    import json
    from types import SimpleNamespace

    subprocess_payload = {"mius": mock_miu, "meta": {"timings_ms": {"total_ms": 1.0}, "detector": "omniparser"}}
    with patch.dict(
        os.environ,
        {
            "GEOPACT_MODULE_A_SUBPROCESS": "1",
            "GEOPACT_MODULE_A_SUBPROCESS_POOL": "1",
            "GEOPACT_MODULE_A_PREFETCH": "cpu",
        },
        clear=False,
    ):
        with patch(
            "lmms_eval.models.geopact_module_a_worker.get_module_a_worker"
        ) as mock_worker_get:
            worker = SimpleNamespace(
                detect=lambda *_a, **_k: (mock_miu, subprocess_payload["meta"])
            )
            mock_worker_get.return_value = worker
            with patch("lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu"):
                from lmms_eval.models.geopact_vl_hook import _run_detector
                from PIL import Image

                out = _run_detector(
                    Image.new("RGB", (8, 8)), "omniparser", oas_cdct, model=prefetch_model
                )
    check("prefetch subprocess pool returns mius", len(out) == 1)
    check("prefetch subprocess pool invoked", mock_worker_get.called)
    check(
        "prefetch subprocess pool isolation flag",
        getattr(prefetch_model, "_geopact_module_a_subprocess_isolated", False),
    )
    check(
        "prefetch subprocess pool diag flag",
        getattr(prefetch_model, "_geopact_module_a_subprocess_pooled", False),
    )

    oneshot_json = json.dumps(subprocess_payload)
    with patch.dict(
        os.environ,
        {
            "GEOPACT_MODULE_A_SUBPROCESS": "1",
            "GEOPACT_MODULE_A_SUBPROCESS_POOL": "0",
            "GEOPACT_MODULE_A_PREFETCH": "cpu",
        },
        clear=False,
    ):
        with patch(
            "subprocess.run",
            return_value=SimpleNamespace(stdout=oneshot_json, stderr="", returncode=0),
        ) as mock_oneshot:
            with patch("lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu"):
                from lmms_eval.models.geopact_vl_hook import _run_detector
                from PIL import Image

                out2 = _run_detector(
                    Image.new("RGB", (8, 8)), "omniparser", oas_cdct, model=prefetch_model
                )
    check("prefetch subprocess oneshot returns mius", len(out2) == 1)
    check("prefetch subprocess oneshot invoked", mock_oneshot.called)

    with patch.dict(
        os.environ,
        {
            "GEOPACT_MODULE_A_SUBPROCESS": "1",
            "GEOPACT_MODULE_A_SUBPROCESS_POOL": "1",
            "GEOPACT_MODULE_A_PREFETCH": "cpu",
            "GEOPACT_MODULE_A_PREFETCH_ASYNC": "1",
        },
        clear=False,
    ):
        from lmms_eval.models.geopact_module_a_prefetch import module_a_prefetch_async_enabled

        check("prefetch async enabled with pool", module_a_prefetch_async_enabled())
        with patch(
            "lmms_eval.models.geopact_module_a_prefetch.try_consume_module_a_prefetch",
            return_value=(mock_miu, {"module_a_prefetch_async": True}),
        ) as mock_consume:
            with patch(
                "lmms_eval.models.geopact_module_a_worker.run_module_a_subprocess_pooled"
            ) as mock_pool:
                with patch("lmms_eval.models.gui_miu_detectors.maybe_release_yolo_gpu"):
                    from lmms_eval.models.geopact_vl_hook import _run_detector
                    from PIL import Image

                    out3 = _run_detector(
                        Image.new("RGB", (8, 8)), "omniparser", oas_cdct, model=prefetch_model
                    )
        check("prefetch async consume returns mius", len(out3) == 1)
        check("prefetch async consume invoked", mock_consume.called)
        check("prefetch async skips sync pool", not mock_pool.called)

    with patch.dict(
        os.environ,
        {
            "GEOPACT_MODULE_A_SUBPROCESS": "1",
            "GEOPACT_MODULE_A_SUBPROCESS_POOL": "1",
            "GEOPACT_MODULE_A_PREFETCH": "cpu",
            "GEOPACT_MODULE_A_WORKER_SHM": "1",
        },
        clear=False,
    ):
        from lmms_eval.models.geopact_module_a_worker import module_a_worker_shm_enabled

        check("worker shm enabled by default", module_a_worker_shm_enabled())

    os.environ.pop("GEOPACT_SESSION_KV", None)
    os.environ.pop("GEOPACT_SESSION_KV_MODE", None)
    if _prev_alpha is None:
        os.environ.pop("GEOPACT_CDCT_ALPHA", None)
    else:
        os.environ["GEOPACT_CDCT_ALPHA"] = _prev_alpha

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nAll token ICKH sticky checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
