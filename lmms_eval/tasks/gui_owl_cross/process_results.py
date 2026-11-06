"""AITW + AndroidControl process_results for GUI-Owl cross-benchmark."""

from __future__ import annotations

from typing import Any, Dict, List

from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action
from lmms_eval.tasks.gui_owl_cross.aitw_action_match import aitw_actions_match, omni_to_aitw_action
from lmms_eval.tasks.gui_owl_cross.utils import (
    gui_owl_aitw_doc_to_messages,
    gui_owl_android_control_doc_to_messages,
    load_aitw_step_docs,
    load_android_control_test_docs,
)


def aitw_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.screenspot_v2_mobile.utils import screenspot_v2_mobile_doc_to_visual

    return screenspot_v2_mobile_doc_to_visual(doc, lmms_eval_specific_kwargs)


def android_control_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.gui_owl_cross.utils import _resolve_media_image_path

    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/android_control")
    img_path = doc.get("image_path") or doc.get("img_url")
    if not img_path:
        return []
    resolved = _resolve_media_image_path(str(img_path), media_dir)
    if not resolved:
        return []
    from PIL import Image

    return [Image.open(resolved).convert("RGB")]


def android_control_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def android_control_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return str(doc.get("action", doc.get("gt_action", "")))


def aitw_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def aitw_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return gui_owl_aitw_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def aitw_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return str(doc.get("action", doc.get("gt_action", "")))


def aitw_process_result(doc, result, lmms_eval_specific_kwargs=None):
    pred_text = result[0] if result else ""
    pred_action = omni_to_aitw_action(parse_omni_action(pred_text))
    gt_action = doc.get("gt_action") or doc.get("action") or {}
    if isinstance(gt_action, str):
        gt_action = omni_to_aitw_action(parse_omni_action(gt_action))
    screen = (
        int(doc.get("screen_width", 1080)),
        int(doc.get("screen_height", 2400)),
    )
    match = aitw_actions_match(pred_action, gt_action, screen_size=screen)
    payload = {"match": match, "subset": doc.get("subset", "standard")}
    return {"aitw_action_match": payload}


def aitw_aggregate_action_match(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("match") else 0.0 for r in results) / len(results)


def android_control_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return gui_owl_android_control_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def android_control_dialogue_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.gui_owl_cross.utils import gui_owl_android_control_dialogue_doc_to_messages

    return gui_owl_android_control_dialogue_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def android_control_process_result(doc, result, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.gui_owl_cross.android_control_eval import score_android_control_step

    pred_text = result[0] if result else ""
    scored = score_android_control_step(doc, pred_text)
    return {"android_control_exact_match": scored}


def android_control_omni_process_result(doc, result, lmms_eval_specific_kwargs=None):
    """AC action-space scoring with OmniGUI metric keys (TM/EM/GP/SR + latency)."""
    from lmms_eval.tasks.agentcpm_gui.utils_eval_extensions import (
        _attach_generation_usage,
        _omni_metric_result_map,
        _split_generation_output,
    )
    from lmms_eval.tasks.agentcpm_gui.performance_profile import normalize_step_profile
    from lmms_eval.tasks.gui_owl_cross.android_control_eval import score_android_control_step

    pred_text, generation_usage = _split_generation_output(result[0] if result else "")
    scored = score_android_control_step(doc, pred_text)
    eval_result = {
        "subset": doc.get("subset", "android_control_high"),
        "episode_id": doc.get("episode_id", "unknown"),
        "step_id": doc.get("step_index", doc.get("step_id", 0)),
        "step_index": doc.get("step_index", doc.get("step_id", 0)),
        "type_match": bool(scored.get("type_match", False)),
        "exact_match": bool(scored.get("exact_match", False)),
        "pred_action_type": scored.get("pred_action_type"),
        "gt_action_type": scored.get("gt_action_type"),
    }
    eval_result = _attach_generation_usage(eval_result, generation_usage)
    if generation_usage:
        normalized = normalize_step_profile({**eval_result, **generation_usage})
        eval_result = {**eval_result, **normalized}
        eval_result["generation_usage"] = {**generation_usage, **normalized}
    return _omni_metric_result_map(eval_result)


def mind2web_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = dict(lmms_eval_specific_kwargs or {})
    lmms_eval_specific_kwargs.setdefault("media_dir", "data/mind2web")
    return android_control_doc_to_visual(doc, lmms_eval_specific_kwargs)


def mind2web_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return ""


def mind2web_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    return str(doc.get("gt_action", doc.get("action", "")))


def mind2web_dialogue_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.gui_owl_cross.utils import gui_owl_mind2web_dialogue_doc_to_messages

    return gui_owl_mind2web_dialogue_doc_to_messages(doc, lmms_eval_specific_kwargs or {})


def mind2web_omni_process_result(doc, result, lmms_eval_specific_kwargs=None):
    """Mind2Web SeeClick metrics (Ele.Acc/Op.F1/Step SR) + OmniGUI TM/EM/GP keys."""
    from lmms_eval.tasks.agentcpm_gui.utils_eval_extensions import (
        _attach_generation_usage,
        _omni_metric_result_map,
        _split_generation_output,
    )
    from lmms_eval.tasks.agentcpm_gui.performance_profile import normalize_step_profile
    from lmms_eval.tasks.gui_owl_cross.mind2web_eval import score_mind2web_step

    pred_text, generation_usage = _split_generation_output(result[0] if result else "")
    scored = score_mind2web_step(doc, pred_text)
    eval_result = {
        "subset": doc.get("subset", "mind2web"),
        "episode_id": doc.get("episode_id", "unknown"),
        "step_id": doc.get("step_index", doc.get("step_id", 0)),
        "step_index": doc.get("step_index", doc.get("step_id", 0)),
        "type_match": bool(scored.get("type_match", False)),
        "exact_match": bool(scored.get("exact_match", False)),
        "pred_action_type": scored.get("pred_action_type"),
        "gt_action_type": scored.get("gt_action_type"),
    }
    eval_result = _attach_generation_usage(eval_result, generation_usage)
    if generation_usage:
        normalized = normalize_step_profile({**eval_result, **generation_usage})
        eval_result = {**eval_result, **normalized}
        eval_result["generation_usage"] = {**generation_usage, **normalized}
    out = _omni_metric_result_map(eval_result)
    m2w_payload = {
        "ele_acc": bool(scored.get("ele_acc", False)),
        "op_f1": float(scored.get("op_f1", 0.0)),
        "step_sr": bool(scored.get("step_sr", False)),
        "subset": eval_result["subset"],
    }
    out["m2w_ele_acc"] = m2w_payload
    out["m2w_op_f1"] = m2w_payload
    out["m2w_step_sr"] = m2w_payload
    return out


def mind2web_aggregate_ele_acc(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("ele_acc") else 0.0 for r in results) / len(results)


def mind2web_aggregate_op_f1(results):
    if not results:
        return 0.0
    return sum(float(r.get("op_f1", 0.0)) for r in results) / len(results)


def mind2web_aggregate_step_sr(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("step_sr") else 0.0 for r in results) / len(results)


def android_control_aggregate_exact_match(results):
    if not results:
        return 0.0
    return sum(1.0 if r.get("exact_match") else 0.0 for r in results) / len(results)


def aitw_aggregate_by_subset(results):
    """Per-subset action-match rates for AITW standard test."""
    buckets: Dict[str, List[float]] = {}
    for row in results:
        subset = str(row.get("subset", "standard"))
        buckets.setdefault(subset, []).append(1.0 if row.get("match") else 0.0)
    out: Dict[str, float] = {}
    for subset, vals in buckets.items():
        out[subset] = sum(vals) / len(vals) if vals else 0.0
    if out:
        out["macro_avg"] = sum(out.values()) / len(out)
    return out.get("macro_avg", 0.0) if out else 0.0
