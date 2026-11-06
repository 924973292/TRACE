"""Module A — parameter-free MIU interaction energy density.

Every MIU carries one interaction energy ``E``. Module A exposes the exact
density used by Module B as:

    info_density_i = E_i / n_tokens_i

For YOLO GUI MIUs, ``E`` is the pentagonal native interaction energy.

**Legacy** (``oas_ickh_cdct_no_coord_yolo`` and ablation tokens):

    E_i = conf_i * (S_i + H_i + C_i + G_i + R_i)

**Module A v2** (``oas_ickh_cdct_no_coord_yolo_v2``, GP-validated structural sum):

    E_i = S_i + H_i + C_i + G_i + R_i

``n_tokens`` uses the vision merge grid when ``grid_hw`` is available.

``R`` is row/column center alignment (tab bars, bottom nav, vertical rails).
Policy ``drop_pent_resonance`` removes R; ``conf_attr_additive`` restores the
older additive form ``S + H + C + G + R + f(conf)``.

For video/audio MIUs, ``E`` is their modality-native ``modality_energy`` and is
used directly by default because AV token counts are synthetic projector
heuristics. The projector path may attach diagnostics, but it never rewrites the
Module-A score; Module B reads ``miu["info_density"]`` directly.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Mapping, Sequence


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


_PENTAGONAL_COMPONENT_KEYS: dict[str, str] = {
    "scale": "pentagonal_scale",
    "entropy": "pentagonal_entropy",
    "contrast": "pentagonal_contrast",
    "containment": "pentagonal_containment",
    "resonance": "pentagonal_resonance",
}


def _policy_has(policy: str, *needles: str) -> bool:
    policy = str(policy or "").lower()
    return any(needle in policy for needle in needles)


def _has_pentagonal_attrs(miu: dict) -> bool:
    return all(key in miu for key in _PENTAGONAL_COMPONENT_KEYS.values())


def _drop_pentagonal_component(policy: str, component: str) -> bool:
    if component == "resonance":
        return _policy_has(
            policy,
            "drop_pent_resonance",
            "drop_pentagonal_resonance",
            "no_pent_resonance",
            "no_pentagonal_resonance",
        )
    return _policy_has(
        policy,
        f"drop_pent_{component}",
        f"drop_pentagonal_{component}",
        f"no_pent_{component}",
        f"no_pentagonal_{component}",
    )


def _no_confidence_gate(policy: str) -> bool:
    return _policy_has(
        policy,
        "no_conf_gate",
        "no_confidence_gate",
        "drop_conf_gate",
        "no_detector_gate",
        "no_pent_gate",
    )


def _use_confidence_gate(policy: str) -> bool:
    return _policy_has(
        policy,
        "old_conf_gate",
        "conf_gate_old",
        "conf_as_gate",
        "det_conf_gate",
    ) and not _no_confidence_gate(policy)


def _is_module_a_v2(policy: str) -> bool:
    """Structural energy E = sum(S,H,C,G,R); policy token ``_yolo_v2`` or ``struct_energy``."""
    policy = str(policy or "").lower()
    if _policy_has(policy, "module_a_v2", "struct_energy"):
        return True
    return "_yolo_v2" in policy or policy.endswith("_v2")


def _drop_confidence_attribute(policy: str) -> bool:
    return _policy_has(
        policy,
        "no_conf_attr",
        "drop_conf_attr",
        "no_confidence_attr",
        "drop_confidence_attr",
        "no_confidence_attribute",
        "no_conf",
    )


def _confidence_attribute(det_score: float, policy: str = "") -> float:
    if _drop_confidence_attribute(policy):
        return 0.0
    det = max(0.0, min(1.0, float(det_score)))
    alpha = 4.0
    return float(math.log1p(alpha * det) / math.log1p(alpha))


def _miu_confidence_attribute(miu: dict, policy: str = "") -> float:
    if _drop_confidence_attribute(policy):
        return 0.0
    if "pentagonal_confidence_attr" in miu:
        return max(0.0, _safe_float(miu.get("pentagonal_confidence_attr"), 0.0))
    return _confidence_attribute(_safe_float(miu.get("det_score", 0.0)), policy)


def _pentagonal_ablation_active(policy: str) -> bool:
    return (
        _is_module_a_v2(policy)
        or _no_confidence_gate(policy)
        or _use_confidence_gate(policy)
        or _drop_confidence_attribute(policy)
        or any(
            _drop_pentagonal_component(policy, component)
            for component in _PENTAGONAL_COMPONENT_KEYS
        )
    )


def _pentagonal_energy_additive(policy: str) -> bool:
    return _policy_has(
        policy,
        "conf_attr_additive",
        "additive_conf",
        "pent_additive_conf",
    )


def _pentagonal_energy_from_attrs(miu: dict, policy: str = "") -> float:
    pentagonal_sum = 0.0
    for component, key in _PENTAGONAL_COMPONENT_KEYS.items():
        if _drop_pentagonal_component(policy, component):
            continue
        pentagonal_sum += _safe_float(miu.get(key, 0.0))
    det_score = max(0.0, min(1.0, _safe_float(miu.get("det_score", 0.0))))
    if _is_module_a_v2(policy):
        return float(pentagonal_sum)
    if _drop_confidence_attribute(policy):
        if _policy_has(policy, "conf_attr_only", "confidence_attr_only"):
            return float(_miu_confidence_attribute(miu, policy))
        return float(pentagonal_sum)
    if _pentagonal_energy_additive(policy) or _no_confidence_gate(policy):
        confidence_attr = _miu_confidence_attribute(miu, policy)
        if _policy_has(policy, "conf_attr_only", "confidence_attr_only"):
            return float(confidence_attr)
        return float(pentagonal_sum + confidence_attr)
    if _use_confidence_gate(policy):
        return float(det_score * pentagonal_sum)
    # Default: conf × (S + H + C + G + R).
    return float(det_score * pentagonal_sum)


def _use_raw_energy_density(policy: str) -> bool:
    """Skip /n_tokens: info_density = E (ablation: large-box bias)."""
    return _policy_has(policy, "raw_energy", "no_div_token")


def _force_area_heuristic_token_count(policy: str) -> bool:
    """Use area×48 denominator even when grid_hw is available."""
    return _policy_has(policy, "div_area_heuristic", "area_heuristic_denom")


def estimate_miu_pad_count(miu: dict, *, pads_per_unit_area: float = 48.0) -> int:
    """Heuristic pad count from normalized box area (Module A proxy)."""
    box = miu.get("box")
    if not box or len(box) < 4:
        return 1
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    area = max(0.0, (x2 - x1) * (y2 - y1))
    return max(1, int(round(area * pads_per_unit_area)))


def _flag_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "off"}


def estimate_miu_grid_token_count(miu: dict, grid_h: int, grid_w: int) -> int:
    """Token count aligned with ``_map_mius_to_tokens`` (LLM merge grid).

    F2 (v3): the density denominator must count exactly the cells that
    ``_token_indices_for_miu_box`` assigns to the MIU, otherwise
    info_density = E / n_tokens is normalized by a different support than the
    one the prior field P_j is scattered onto. Default follows the hook's
    half-open ceil mapping (GEOPACT_A_BBOX_CEIL, default on);
    GEOPACT_A_DENSITY_ALIGN=0 restores the legacy ``int(y2*gh)+1`` count for
    attribution arms.
    """
    box = miu.get("box")
    if not box or len(box) < 4:
        return 1
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    gh, gw = max(1, int(grid_h)), max(1, int(grid_w))
    ci0 = max(0, int(y1 * gh))
    cj0 = max(0, int(x1 * gw))
    if _flag_on("GEOPACT_A_DENSITY_ALIGN") and _flag_on("GEOPACT_A_BBOX_CEIL"):
        ci1 = min(gh, max(ci0 + 1, math.ceil(y2 * gh)))
        cj1 = min(gw, max(cj0 + 1, math.ceil(x2 * gw)))
    else:
        ci1 = min(gh, int(y2 * gh) + 1)
        cj1 = min(gw, int(x2 * gw) + 1)
    return max(1, (ci1 - ci0) * (cj1 - cj0))


def miu_interaction_energy(miu: dict, policy: str = "") -> float:
    """Return the per-MIU interaction energy ``E``.

    When YOLO-only pentagonal attributes are present, GUI energy defaults to
    four non-parametric GUI interaction attributes plus a soft confidence
    attribute. Resonance is diagnostic by default and the old multiplicative
    detector-confidence gate is available only by explicit policy:

        Legacy: E = det * (S + H + C + G + R)
        v2 (_yolo_v2): E = S + H + C + G + R
        Ablation additive: E = S + H + C + G + R + f(det_score)

    Non-image MIUs use their modality-native energy. Legacy fallbacks stay in
    place only for older detectors that do not yet attach explicit energy.
    """
    if _has_pentagonal_attrs(miu):
        return _pentagonal_energy_from_attrs(miu, policy)

    if "interaction_energy" in miu:
        return _safe_float(miu.get("interaction_energy", 0.0))

    if "modality_energy" in miu:
        return _safe_float(miu.get("modality_energy", 0.0))

    modality = str(miu.get("modality") or "image").lower()
    if modality != "image":
        return _safe_float(miu.get("modality_score", miu.get("det_score", 0.0)))

    if "energy" in miu:
        return _safe_float(miu.get("energy", 0.0))

    has_attrs = ("distinctiveness" in miu) or ("interactable" in miu) or ("has_text" in miu)
    if not has_attrs:
        return (
            0.45 * _safe_float(miu.get("action_prior", 0.0))
            + 0.35 * _safe_float(miu.get("det_score", 0.0))
            + 0.20 * _safe_float(miu.get("topology_prior", miu.get("role_prior", 0.0)))
        )

    # A_sal: detection confidence used as the overall gate.
    det = _safe_float(miu.get("det_score", 0.0))
    # Key GUI attribute bits, used directly (no soft gating).
    a_act = _safe_float(miu.get("interactable", miu.get("action_prior", 0.0)))
    a_text = _safe_float(miu.get("has_text", 0.0))
    a_dist = _safe_float(miu.get("distinctiveness", 1.0), default=1.0)
    return float(det * (a_act + a_text + a_dist))


def miu_saliency_proxy(miu: dict) -> float:
    """Backward-compatible alias for older callers."""
    return miu_interaction_energy(miu)


def info_density_raw(energy: float, token_count: int, modality: str = "image", policy: str = "") -> float:
    """Core formula: image E / token_count; AV raw energy by default."""
    modality = str(modality or "image").lower()
    policy = str(policy or "").lower()
    if modality in {"video", "audio"} and "av_div_token_count" not in policy:
        if "av_capped_denominator" in policy or "av_token_cap" in policy:
            cap = 8 if modality == "video" else 12
            return float(energy) / float(max(1, min(cap, int(token_count))))
        return float(energy)
    if _use_raw_energy_density(policy):
        return float(energy)
    return float(energy) / float(max(1, int(token_count)))


def info_density_proxy(miu: dict) -> float:
    """Module A InfoDensity before projector (stored on MIU dict)."""
    n_tok = int(miu.get("est_token_count") or estimate_miu_pad_count(miu))
    return info_density_raw(miu_interaction_energy(miu), n_tok, str(miu.get("modality", "image")))


def info_density_from_projector(
    saliency_mean: float,
    token_count: int,
    emb_variance: float,
    *,
    saliency_max: float,
    variance_max: float,
) -> float:
    """Diagnostic projector density; never used to overwrite Module A."""
    del saliency_max, variance_max, emb_variance
    return info_density_raw(saliency_mean, token_count)


def attach_module_a_info_density_proxy(
    mius: list[dict],
    policy: str = "",
    *,
    grid_hw: tuple[int, int] | None = None,
) -> list[dict]:
    """Attach strict ``interaction_energy`` and ``info_density`` to MIUs."""
    policy = (policy or "").lower()
    if "no_info_density" in policy:
        for miu in mius:
            miu["interaction_energy"] = miu_interaction_energy(miu, policy)
            miu["info_density_proxy"] = 0.0
            miu["info_density_raw"] = 0.0
            miu["info_density"] = 0.0
        return mius
    use_grid = grid_hw is not None and len(grid_hw) == 2 and not _force_area_heuristic_token_count(policy)
    grid_h, grid_w = (int(grid_hw[0]), int(grid_hw[1])) if use_grid else (0, 0)
    for miu in mius:
        if use_grid:
            miu["est_token_count"] = estimate_miu_grid_token_count(miu, grid_h, grid_w)
            miu["est_token_count_source"] = "grid"
        else:
            miu["est_token_count"] = int(miu.get("est_token_count") or estimate_miu_pad_count(miu))
            miu["est_token_count_source"] = "area_heuristic"
        energy = miu_interaction_energy(miu, policy)
        proxy = info_density_raw(energy, miu["est_token_count"], str(miu.get("modality", "image")), policy)
        miu["interaction_energy"] = energy
        miu["info_density_proxy"] = proxy
        miu["info_density_raw"] = proxy
        miu["info_density"] = proxy
    return mius


def refine_module_a_info_density_projector(
    mius: Sequence[dict],
    saliency_means: Sequence[float],
    token_counts: Sequence[int],
    emb_variances: Sequence[float],
) -> None:
    """Attach projector diagnostics without changing strict Module-A density."""
    if not mius:
        return
    density_raw_list = [
        info_density_raw(saliency_means[i], token_counts[i])
        for i in range(len(mius))
    ]
    sal_max = max(density_raw_list) if density_raw_list else 1.0
    var_max = max(float(v) for v in emb_variances) if emb_variances else 1.0
    for i, miu in enumerate(mius):
        miu["info_density_projector"] = info_density_from_projector(
            saliency_means[i],
            token_counts[i],
            emb_variances[i],
            saliency_max=sal_max,
            variance_max=var_max,
        )
        miu["info_density_projector_variance"] = float(emb_variances[i])
        miu["miu_token_count"] = int(token_counts[i])


def module_a_summary_meta(mius: Iterable[dict]) -> dict[str, float]:
    items = list(mius)
    if not items:
        return {
            "info_density_count": 0.0,
            "interaction_energy_mean": 0.0,
            "info_density_mean": 0.0,
            "info_density_proxy_mean": 0.0,
        }
    n = float(len(items))
    return {
        "info_density_count": n,
        "interaction_energy_mean": float(
            sum(_safe_float(m.get("interaction_energy")) for m in items) / n
        ),
        "info_density_mean": float(sum(_safe_float(m.get("info_density")) for m in items) / n),
        "info_density_proxy_mean": float(
            sum(_safe_float(m.get("info_density_proxy")) for m in items) / n
        ),
    }


def mmr_importance_from_miu(miu: dict, policy: str = "") -> float:
    """What Module B reads: ``info_density`` from Module A."""
    policy = (policy or "").lower()
    if (
        "no_info_density" in policy
        or "no_multimodal_density" in policy
        or "no_miu_density" in policy
        or "diversity_only" in policy
    ):
        return 1.0
    if "score_oas_only" in policy:
        return _safe_float(miu.get("oas_imp", 0.0))
    if "density_proxy_only" in policy:
        return _safe_float(miu.get("info_density_proxy", miu.get("oas_imp", 0.0)))
    dens = _safe_float(miu.get("info_density", 0.0))
    if dens > 0.0:
        return dens
    return _safe_float(miu.get("info_density_proxy", miu.get("oas_imp", 0.0)))
