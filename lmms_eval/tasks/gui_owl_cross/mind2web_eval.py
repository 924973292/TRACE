"""Mind2Web (SeeClick / HistPrune paradigm) scoring for GUI-Owl mobile_use outputs.

Metrics per the SeeClick / Mind2Web standard:
  - Ele.Acc : predicted point falls inside the GT element bbox (pixel space).
  - Op.F1   : token-level F1 between predicted and GT operation strings
              ("CLICK" / "TYPE value" / "SELECT value").
  - Step SR : Ele.Acc hit AND Op.F1 == 1.0.

OmniGUI-protocol migration:
  - type_match  = action type equality (TM)
  - exact_match = Step SR (EM; feeds GP/SR aggregations)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

M2W_OPS = ("click", "type", "select")


def _pred_tool_call(pred_text: str) -> Dict[str, Any]:
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", pred_text or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    args = data.get("arguments") or data.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


_BARE_JSON_RE = re.compile(r"\{[^{}]*\"(?:action|action_type)\"[^{}]*\}", re.S)

# UI-TARS native DSL, as emitted under UITARS_NATIVE_MULTISTEP with the
# Mind2Web action space: click/type/select carrying a <point>x y</point> target.
_DSL_RE = re.compile(
    r"(click|type|select)\s*\([^()]*?"
    r"(?:point|coordinate|start_box)\s*=\s*['\"]?(?:<point>)?[\[\(]?\s*(\d+)[\s,]+(\d+)",
    re.IGNORECASE,
)
_DSL_CONTENT_RE = re.compile(r"(?:content|text)\s*=\s*['\"](.*?)['\"]", re.S)


def _pred_bare_json(pred_text: str) -> Dict[str, Any]:
    """Read a flat action dict emitted without <tool_call> tags.

    The UI-TARS glue lowers its DSL to a flat dict, so a Mind2Web arm on that
    backbone never presents the tool-call surface this module was written for.
    """
    for m in reversed(list(_BARE_JSON_RE.finditer(pred_text or ""))):
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        verb = data.get("action")
        if not isinstance(verb, str):
            # Numeric OmniActionType, mapped by the caller's fallback instead.
            continue
        verb = verb.lower().strip()
        alias = {"input": "type", "tap": "click", "left_single": "click"}
        verb = alias.get(verb, verb)
        if verb not in M2W_OPS:
            continue
        out: Dict[str, Any] = {"action_type": verb}
        coord = data.get("coordinate") or data.get("point")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                out["coordinate"] = [float(coord[0]), float(coord[1])]
            except (TypeError, ValueError):
                pass
        text = data.get("text", data.get("content"))
        if text:
            out["text"] = str(text)
        return out
    return {}


def _pred_uitars_dsl(pred_text: str) -> Dict[str, Any]:
    """Read the last `Action: verb(point=..., content=...)` line."""
    matches = list(_DSL_RE.finditer(pred_text or ""))
    if not matches:
        return {}
    m = matches[-1]
    verb = m.group(1).lower()
    if verb not in M2W_OPS:
        return {}
    out: Dict[str, Any] = {
        "action_type": verb,
        "coordinate": [float(m.group(2)), float(m.group(3))],
    }
    tail = _DSL_CONTENT_RE.search(pred_text[m.start():])
    if tail and tail.group(1):
        out["text"] = tail.group(1)
    return out


def guiowl_pred_to_m2w_action(pred_text: str) -> Dict[str, Any]:
    """Parse GUI-Owl output into {action_type, coordinate([0,1000]), text}."""
    args = _pred_tool_call(pred_text)
    if not args:
        # Flat dict, then native DSL, before the legacy numeric fallback.
        for reader in (_pred_bare_json, _pred_uitars_dsl):
            got = reader(pred_text)
            if got:
                return got
        try:
            from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action

            parsed = parse_omni_action(pred_text or "")
        except Exception:
            parsed = None
        if not parsed:
            return {}
        at = parsed.get("action_type")
        type_map = {0: "click", 7: "type"}
        action = type_map.get(int(at)) if at is not None else None
        if action is None:
            return {}
        out: Dict[str, Any] = {"action_type": action}
        if parsed.get("coordinate"):
            out["coordinate"] = list(parsed["coordinate"][:2])
        if parsed.get("text"):
            out["text"] = str(parsed["text"])
        return out

    action = str(args.get("action", "")).lower().strip()
    if action not in M2W_OPS:
        return {}
    out = {"action_type": action}
    coord = args.get("coordinate")
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        out["coordinate"] = [float(coord[0]), float(coord[1])]
    text = args.get("text")
    if text:
        out["text"] = str(text)
    return out


def _op_repr(action_type: str, text: str) -> str:
    op = str(action_type or "").upper().strip()
    val = str(text or "").strip()
    return f"{op} {val}".strip()


def _token_f1(pred: str, gt: str) -> float:
    pt = pred.lower().split()
    gtok = gt.lower().split()
    if not pt or not gtok:
        return 1.0 if pt == gtok else 0.0
    common: Dict[str, int] = {}
    for t in pt:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gtok:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall = overlap / len(gtok)
    return 2 * precision * recall / (precision + recall)


def _point_in_bbox_px(
    coord_norm: Optional[List[float]],
    bbox_px: Optional[List[float]],
    img_size: Tuple[int, int],
) -> bool:
    if not coord_norm or not bbox_px or len(bbox_px) < 4:
        return False
    w, h = img_size
    x = float(coord_norm[0]) / 1000.0 * max(w, 1)
    y = float(coord_norm[1]) / 1000.0 * max(h, 1)
    x1, y1, x2, y2 = (float(v) for v in bbox_px[:4])
    return x1 <= x <= x2 and y1 <= y <= y2


def score_mind2web_step(doc: Dict[str, Any], pred_text: str) -> Dict[str, Any]:
    """Score one Mind2Web step (SeeClick metrics + OmniGUI TM/EM)."""
    gt = doc.get("gt_action") or doc.get("action") or {}
    pred = guiowl_pred_to_m2w_action(pred_text)
    size = doc.get("img_size") or [1280, 720]
    img_size = (int(size[0] or 1280), int(size[1] or 720))

    gt_type = str(gt.get("action_type", "")).lower()
    pred_type = str(pred.get("action_type", "")).lower()
    type_match = bool(pred_type) and pred_type == gt_type

    ele_hit = _point_in_bbox_px(pred.get("coordinate"), gt.get("bbox_px"), img_size)

    pred_op = _op_repr(pred_type, pred.get("text", ""))
    gt_op = _op_repr(gt_type, gt.get("text", ""))
    op_f1 = _token_f1(pred_op, gt_op) if pred_type else 0.0

    step_sr = bool(ele_hit and op_f1 >= 1.0 - 1e-9)
    return {
        "ele_acc": ele_hit,
        "op_f1": op_f1,
        "step_sr": step_sr,
        "type_match": type_match,
        "exact_match": step_sr,
        "pred_action_type": pred_type or None,
        "gt_action_type": gt_type or None,
    }
