"""AndroidControl test-split scoring for GUI-Owl mobile_use outputs.

Maps GUI-Owl predictions to AndroidControl JSON actions and applies
exact_match per GUI-Actor / AndroidControl high-level test protocol.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional, Tuple

from lmms_eval.tasks.agentcpm_gui.utils import parse_json_output, text_similarity

# Normalized coordinate L2 threshold (matches agentcpm_gui utils android_control path).
TAP_DISTANCE_THRESHOLD_AC = 0.04

_SWIPE_TO_SCROLL = {
    3: "up",
    4: "down",
    5: "left",
    6: "right",
}

_OMNI_TO_AC_TYPE = {
    0: "click",
    2: "long_press",
    7: "type",
    8: "navigate_back",
    9: "navigate_home",
    -1: "wait",
}


def _norm_coord(val: float, axis_size: int) -> float:
    v = float(val)
    if v > 1.5:
        return v / max(axis_size, 1)
    return v


def _extract_open_app(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle mobile_use action=open when parse_json_output returns raw tool_call fields."""
    if parsed.get("action_type") is not None:
        return None
    action = str(parsed.get("action", "")).lower()
    if action != "open":
        return None
    app = parsed.get("text") or parsed.get("app_name") or parsed.get("content") or ""
    if not app:
        args = parsed.get("arguments") or {}
        if isinstance(args, dict):
            app = args.get("text") or args.get("app_name") or ""
    if not app:
        return None
    return {"action_type": "open_app", "app_name": str(app).strip()}


def _parse_open_from_tool_call(pred_text: str) -> Dict[str, Any]:
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
    if not isinstance(args, dict):
        return {}
    if str(args.get("action", "")).lower() != "open":
        return {}
    app = args.get("text") or args.get("app_name") or args.get("content") or ""
    if not app:
        return {}
    return {"action_type": "open_app", "app_name": str(app).strip()}


def guiowl_pred_to_ac_action(pred_text: str) -> Dict[str, Any]:
    """Convert GUI-Owl model output to AndroidControl-style action dict."""
    open_ac = _parse_open_from_tool_call(pred_text)
    if open_ac:
        return open_ac

    parsed = parse_json_output(pred_text or "")
    if not parsed:
        return {}

    open_ac = _extract_open_app(parsed)
    if open_ac:
        return open_ac

    at = parsed.get("action_type")
    if at is None:
        return {}

    try:
        at_int = int(at)
    except (TypeError, ValueError):
        return {}

    if at_int in (3, 4, 5, 6):
        direction = _SWIPE_TO_SCROLL.get(at_int, "up")
        return {"action_type": "scroll", "direction": direction}

    ac_type = _OMNI_TO_AC_TYPE.get(at_int)
    if not ac_type:
        return {}

    out: Dict[str, Any] = {"action_type": ac_type}

    coord = parsed.get("coordinate") or parsed.get("start_coordinate")
    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        scale = 1000.0 if max(float(coord[0]), float(coord[1])) > 1.5 else 1.0
        out["x"] = float(coord[0]) / scale
        out["y"] = float(coord[1]) / scale

    if ac_type == "type":
        text = parsed.get("text") or parsed.get("content") or ""
        if text:
            out["text"] = str(text)

    return out


def _normalize_ac_type(action_type: str) -> str:
    t = str(action_type or "").lower().strip()
    aliases = {
        "input_text": "type",
        "navigate_back": "navigate_back",
        "navigate_home": "navigate_home",
        "press_back": "navigate_back",
        "press_home": "navigate_home",
        "tap": "click",
    }
    return aliases.get(t, t)


def _coord_match(
    pred_x: float,
    pred_y: float,
    gt_x: float,
    gt_y: float,
    *,
    screen_size: Tuple[int, int],
) -> bool:
    w, h = screen_size
    px, py = _norm_coord(pred_x, w), _norm_coord(pred_y, h)
    gx, gy = _norm_coord(gt_x, w), _norm_coord(gt_y, h)
    dist = math.hypot(px - gx, py - gy)
    return dist <= TAP_DISTANCE_THRESHOLD_AC


def ac_actions_match(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    *,
    screen_size: Tuple[int, int] = (1080, 2400),
) -> bool:
    """Return True when prediction matches AndroidControl GT action."""
    if not pred or not gt:
        return False

    p_type = _normalize_ac_type(pred.get("action_type", ""))
    g_type = _normalize_ac_type(gt.get("action_type", ""))
    if not p_type or not g_type:
        return False
    if p_type != g_type:
        return False

    if g_type in ("click", "long_press"):
        for key in ("x", "y"):
            if key not in gt:
                return False
        if "x" not in pred or "y" not in pred:
            return False
        return _coord_match(
            float(pred["x"]),
            float(pred["y"]),
            float(gt["x"]),
            float(gt["y"]),
            screen_size=screen_size,
        )

    if g_type == "scroll":
        pd = str(pred.get("direction", "")).lower()
        gd = str(gt.get("direction", "")).lower()
        return pd == gd

    if g_type == "type":
        pt = str(pred.get("text", "")).lower().strip()
        gt_text = str(gt.get("text", "")).lower().strip()
        if not pt or not gt_text:
            return False
        return pt in gt_text or gt_text in pt or text_similarity(pt, gt_text) >= 0.8

    if g_type == "open_app":
        pa = str(pred.get("app_name", pred.get("text", ""))).lower().strip()
        ga = str(gt.get("app_name", "")).lower().strip()
        if not pa or not ga:
            return False
        return pa in ga or ga in pa or text_similarity(pa, ga) >= 0.8

    if g_type in ("navigate_back", "navigate_home", "wait"):
        return True

    return False


_SCROLL_DIR_TO_OMNI = {
    "up": 3,
    "down": 4,
    "left": 5,
    "right": 6,
}


def _ac_norm_to_mille(val: float) -> int:
    v = float(val)
    if v <= 1.5:
        return int(round(v * 1000.0))
    return int(round(v))


def ac_action_to_omni_action(
    action: Dict[str, Any],
    *,
    screen_size: Tuple[int, int] = (1080, 2400),
) -> Dict[str, Any]:
    """Map AndroidControl GT JSON to OmniGUI action_type dict for teacher forcing."""
    at = _normalize_ac_type(action.get("action_type", ""))
    if at == "click":
        return {
            "action_type": 0,
            "coordinate": [
                _ac_norm_to_mille(action.get("x", 0)),
                _ac_norm_to_mille(action.get("y", 0)),
            ],
        }
    if at == "long_press":
        return {
            "action_type": 2,
            "coordinate": [
                _ac_norm_to_mille(action.get("x", 0)),
                _ac_norm_to_mille(action.get("y", 0)),
            ],
        }
    if at == "type":
        return {"action_type": 7, "text": str(action.get("text", "") or "")}
    if at == "scroll":
        direction = str(action.get("direction", "down")).lower()
        omni_at = _SCROLL_DIR_TO_OMNI.get(direction, 4)
        cx, cy = 500, 500
        deltas = {"up": (0, -300), "down": (0, 300), "left": (-300, 0), "right": (300, 0)}
        dx, dy = deltas.get(direction, (0, 300))
        return {
            "action_type": omni_at,
            "coordinate": [cx, cy],
            "end_coordinate": [cx + dx, cy + dy],
        }
    if at == "navigate_back":
        return {"action_type": 8}
    if at == "navigate_home":
        return {"action_type": 9}
    if at == "wait":
        return {"action_type": -1}
    return {"action_type": -1}


def ac_gt_to_teacher_assistant_text(
    action: Dict[str, Any],
    *,
    screen_size: Tuple[int, int] = (1080, 2400),
) -> str:
    """Convert AndroidControl GT to GUI-Owl mobile_use assistant turn (teacher forcing)."""
    from lmms_eval.tasks.agentcpm_gui.gui_owl_native_messages import format_native_assistant_turn

    at = _normalize_ac_type(action.get("action_type", ""))
    if at == "open_app":
        app = str(action.get("app_name", "") or "").strip()
        tc = {"name": "mobile_use", "arguments": {"action": "open", "text": app}}
        summary = f'Open "{app}"' if app else "Open app"
        return (
            f"Action: {summary}\n"
            f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
        )
    omni = ac_action_to_omni_action(action, screen_size=screen_size)
    return format_native_assistant_turn(omni)


def score_android_control_step(doc: Dict[str, Any], pred_text: str) -> Dict[str, Any]:
    """Score one AndroidControl eval step."""
    gt = doc.get("action") or doc.get("gt_action") or {}
    if isinstance(gt, str):
        gt = {}
    pred = guiowl_pred_to_ac_action(pred_text)
    size = doc.get("img_size") or [1080, 2400]
    screen = (int(size[0] or 1080), int(size[1] or 2400))
    exact = ac_actions_match(pred, gt, screen_size=screen)
    return {
        "exact_match": exact,
        "type_match": _normalize_ac_type(pred.get("action_type", ""))
        == _normalize_ac_type(gt.get("action_type", "")),
        "pred_action_type": pred.get("action_type"),
        "gt_action_type": gt.get("action_type"),
    }
