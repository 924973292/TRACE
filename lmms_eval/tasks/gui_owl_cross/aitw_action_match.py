"""Google AITW action-matching metric (ported from android_in_the_wild README)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

TAP_DISTANCE_THRESHOLD = 0.14  # fraction of screen diagonal / width per Google AITW
DUAL_POINT_EQUAL_THRESH = 0.04


def _as_point(val: Any) -> Optional[Tuple[float, float]]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    return None


def _is_tap(touch: Tuple[float, float], lift: Tuple[float, float]) -> bool:
    return math.hypot(touch[0] - lift[0], touch[1] - lift[1]) <= DUAL_POINT_EQUAL_THRESH


def _swipe_axis(touch: Tuple[float, float], lift: Tuple[float, float]) -> str:
    dx = abs(lift[0] - touch[0])
    dy = abs(lift[1] - touch[1])
    return "horizontal" if dx >= dy else "vertical"


def _tap_match(pred: Tuple[float, float], gt: Tuple[float, float], *, screen_size: Tuple[int, int] = (1080, 2400)) -> bool:
    w, h = screen_size
    dist = math.hypot(pred[0] - gt[0], pred[1] - gt[1])
    # Google: within 14% of screen (use max dimension as proxy)
    return dist <= TAP_DISTANCE_THRESHOLD * max(w, h)


def aitw_actions_match(pred: Dict[str, Any], gt: Dict[str, Any], *, screen_size: Tuple[int, int] = (1080, 2400)) -> bool:
    """Return True when pred action matches gt under AITW offline action-matching rules."""
    p_type = str(pred.get("action_type", pred.get("type", ""))).lower()
    g_type = str(gt.get("action_type", gt.get("type", ""))).lower()
    if p_type != g_type and p_type and g_type:
        return False

    p_touch = _as_point(pred.get("touch") or pred.get("touch_point") or pred.get("coordinate"))
    p_lift = _as_point(pred.get("lift") or pred.get("lift_point") or pred.get("end_coordinate"))
    g_touch = _as_point(gt.get("touch") or gt.get("touch_point") or gt.get("coordinate"))
    g_lift = _as_point(gt.get("lift") or gt.get("lift_point") or gt.get("end_coordinate"))

    # Non dual-point: type match suffices
    if g_touch is None and g_lift is None:
        return p_type == g_type or (not p_type and not g_type)

    if g_touch is None or g_lift is None:
        return p_touch is not None and g_touch is not None and _tap_match(p_touch, g_touch, screen_size=screen_size)

    if p_touch is None or p_lift is None:
        return False

    g_is_tap = _is_tap(g_touch, g_lift)
    p_is_tap = _is_tap(p_touch, p_lift)
    if g_is_tap != p_is_tap:
        return False
    if g_is_tap:
        return _tap_match(p_touch, g_touch, screen_size=screen_size)
    return _swipe_axis(p_touch, p_lift) == _swipe_axis(g_touch, g_lift)


def omni_to_aitw_action(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map parse_omni_action dict to AITW-style action for matching."""
    if not action:
        return {}
    out: Dict[str, Any] = {"action_type": action.get("action_type")}
    if action.get("coordinate"):
        out["touch"] = action["coordinate"]
        out["lift"] = action["coordinate"]
    if action.get("start_coordinate"):
        out["touch"] = action["start_coordinate"]
    if action.get("end_coordinate"):
        out["lift"] = action["end_coordinate"]
    return out
