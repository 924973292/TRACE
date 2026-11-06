"""Shared helpers for GUI-Owl cross-benchmark offline eval tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lmms_eval.tasks.agentcpm_gui.gui_owl_native_messages import (
    GUI_OWL_NATIVE_SYSTEM_PROMPT,
)


# Mind2Web web-agent variant of the GUI-Owl tool schema: same mobile_use
# tool-call surface the model was trained on, action space restricted to the
# SeeClick / HistPrune paradigm ops (CLICK / TYPE / SELECT), each targeting an
# element via a normalized [0,1000] coordinate.
GUI_OWL_MIND2WEB_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a mouse and keyboard to interact with a web page, and take screenshots.
* This is an interface to a web page shown in the screenshot. You can click elements, type into inputs, and select options from dropdowns.
* Coordinates use a normalized 1000x1000 coordinate space, regardless of the screenshot's real pixel resolution: [0,0] is top-left and [1000,1000] is bottom-right.
* Make sure to click elements at the center of the target element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `click`: Click the element at coordinate (x, y).
* `type`: Type the specified text into the input element at coordinate (x, y).
* `select`: Select the specified option text from the dropdown element at coordinate (x, y).", "enum": ["click", "type", "select"], "type": "string"}, "coordinate": {"description": "(x, y): normalized coordinates in the 0-1000 screen space; x increases left-to-right and y increases top-to-bottom. Required by every action.", "type": "array"}, "text": {"description": "The text to type or the option to select. Required only by `action=type` and `action=select`.", "type": "string"}}, "required": ["action", "coordinate"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.'''


# Official GUI-Owl grounding system prompt, verbatim from
# Mobile-Agent-v3.5/grounding_and_kb/eval_grounding_benchmarks.py
# (`only_two_action_system_prompt`). The "screen's resolution is 1000x1000"
# line is the official trick to elicit [0,1000]-normalized coordinates.
GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.
* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''


# Official UI-TARS-1.5 grounding prompt, verbatim from
# bytedance/UI-TARS codes/ui_tars/prompt.py (GROUNDING_DOUBAO), including the
# stray double quote after </point> that ships in the official eval. The
# instruction is embedded in a single user turn with no system message,
# matching the official single-step grounding script. Enabled via
# UITARS_NATIVE_GROUNDING=1 so GUI-Owl arms are untouched.
UITARS_GROUNDING_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with "
    "screenshots. You need to perform the next action to complete the task. "
    "\n\n## Output Format\n\nAction: ...\n\n\n## Action Space\n"
    "click(point='<point>x1 y1</point>'')\n\n## User Instruction\n{instruction}"
)


def _uitars_native_grounding_enabled() -> bool:
    import os

    return os.environ.get("UITARS_NATIVE_GROUNDING", "0") == "1"


def _lmms_text(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _lmms_image(path: str) -> Dict[str, str]:
    return {"type": "image", "url": path}


def gui_owl_grounding_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Official GUI-Owl grounding protocol: computer_use two-action system
    prompt, user = [image, raw benchmark instruction] with no wrapper text."""
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/screenspot_v2_mobile/images")
    img_path = doc.get("image_path") or doc.get("img_filename") or doc.get("file_name")
    if img_path and not Path(str(img_path)).is_file():
        img_path = str(Path(media_dir) / Path(str(img_path)).name)

    instruction = (
        doc.get("instruction")
        or doc.get("instruction_cn")
        or doc.get("prompt")
        or doc.get("text")
        or ""
    )
    content: List[Dict[str, Any]] = []
    if img_path and Path(str(img_path)).is_file():
        content.append(_lmms_image(str(img_path)))
    if _uitars_native_grounding_enabled():
        content.append(
            _lmms_text(UITARS_GROUNDING_PROMPT.format(instruction=str(instruction)))
        )
        return [{"role": "user", "content": content}]
    content.append(_lmms_text(str(instruction)))
    return [
        {"role": "system", "content": [_lmms_text(GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT)]},
        {"role": "user", "content": content},
    ]


def gui_owl_aitw_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AITW single-step: screenshot + goal/step instruction, no history (official offline BC)."""
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/aitw/images")
    img_path = _resolve_media_image_path(str(doc.get("image_path") or ""), media_dir)

    instruction = doc.get("instruction") or doc.get("goal") or doc.get("step_instruction") or ""
    user_text = f"Instruction: Today's date is: (eval).\nTask: {instruction}"
    content: List[Dict[str, Any]] = [_lmms_text(user_text)]
    if img_path:
        content.insert(0, _lmms_image(str(img_path)))
    else:
        raise RuntimeError(
            f"gui_owl step message: screenshot not found (image_path={doc.get('image_path')}, media_dir={media_dir})"
        )
    return [
        {"role": "system", "content": [_lmms_text(GUI_OWL_NATIVE_SYSTEM_PROMPT)]},
        {"role": "user", "content": content},
    ]


def gui_owl_android_control_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AndroidControl high-level step: image + step instruction."""
    return gui_owl_aitw_doc_to_messages(doc, lmms_eval_specific_kwargs)


def _resolve_media_image_path(img_path: str, media_dir: str) -> Optional[str]:
    """Resolve manifest-relative paths like images/foo.png under media_dir."""
    if not img_path:
        return None
    p = Path(str(img_path))
    if p.is_file():
        return str(p)
    if media_dir:
        for cand in (
            Path(media_dir) / p,
            Path(media_dir) / "images" / p.name,
            Path(media_dir) / p.name,
        ):
            if cand.is_file():
                return str(cand)
    return None


def _resolve_ac_image_path(doc: Dict[str, Any], media_dir: str) -> Optional[str]:
    return _resolve_media_image_path(str(doc.get("image_path") or ""), media_dir)


def gui_owl_android_control_dialogue_doc_to_messages(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """AndroidControl with teacher-forced GT history for dialogue SESSION_KV."""
    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/android_control/images")
    step_index = int(doc.get("step_index", doc.get("step_id", 0)) or 0)
    action_history = list(doc.get("action_history") or [])

    if step_index <= 0 and not action_history:
        return gui_owl_android_control_doc_to_messages(doc, lmms_eval_specific_kwargs)

    from lmms_eval.tasks.gui_owl_cross.android_control_eval import ac_gt_to_teacher_assistant_text

    instruction = doc.get("instruction") or doc.get("goal") or doc.get("step_instruction") or ""
    user_prefix = f"Instruction: Today's date is: (eval).\nTask: {instruction}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [_lmms_text(GUI_OWL_NATIVE_SYSTEM_PROMPT)]},
    ]

    from lmms_eval.tasks.history_window import slice_action_history_list, effective_history_slice_n

    # Rolling history window (GEOPACT_EVAL_HISTORY_N, e.g. hist9) — protocol
    # aligned with OmniGUI multi-step eval. AC history can reach depth 23.
    sorted_hist = slice_action_history_list(
        action_history, n=effective_history_slice_n()
    )
    for idx, hist in enumerate(sorted_hist):
        hist_img = _resolve_media_image_path(str(hist.get("image_path") or ""), media_dir)
        assistant_text = (
            hist.get("action_text")
            or hist.get("output")
            or ac_gt_to_teacher_assistant_text(hist.get("action") or {})
        )
        if idx == 0:
            user_content: List[Dict[str, Any]] = [_lmms_text(user_prefix)]
            if hist_img and Path(str(hist_img)).is_file():
                user_content.append(_lmms_image(str(hist_img)))
            messages.append({"role": "user", "content": user_content})
        else:
            user_content = []
            if hist_img and Path(str(hist_img)).is_file():
                user_content.append(_lmms_image(str(hist_img)))
            if user_content:
                messages.append({"role": "user", "content": user_content})
        messages.append(
            {"role": "assistant", "content": [_lmms_text(str(assistant_text))]},
        )

    current_img = _resolve_ac_image_path(doc, media_dir)
    current_user: List[Dict[str, Any]] = []
    if not sorted_hist:
        current_user.append(_lmms_text(user_prefix))
    if current_img:
        current_user.append(_lmms_image(current_img))
    messages.append({"role": "user", "content": current_user})
    return messages


def m2w_action_to_uitars_dsl(action: Dict[str, Any]) -> str:
    """Render one Mind2Web GT action in the UI-TARS `Action:` DSL.

    Teacher-forced history is normally supplied as a precomputed `action_text`
    in mobile_use <tool_call> form. Under the native prompt the model is asked
    for `Action: verb(...)`, so that history has to be restated in the same DSL
    or the model sees one contract in the system turn and another in its own
    past turns.
    """
    at = str(action.get("action_type", "") or "").strip().lower()
    coord = action.get("coordinate") or [0, 0]
    try:
        x, y = int(round(float(coord[0]))), int(round(float(coord[1])))
    except (TypeError, ValueError, IndexError):
        x, y = 0, 0
    point = f"<point>{x} {y}</point>"
    text = str(action.get("text", "") or "").replace("'", "")
    if at in {"type", "input", "7"}:
        return f"type(point='{point}', content='{text}')"
    if at in {"select", "selectoption", "select_option"}:
        return f"select(point='{point}', content='{text}')"
    return f"click(point='{point}')"


def gui_owl_mind2web_dialogue_doc_to_messages(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Mind2Web with teacher-forced GT history (SeeClick paradigm, web tool schema).

    Reuses the AC dialogue structure (system + teacher-forced turns + current
    screenshot) with the Mind2Web CLICK/TYPE/SELECT system prompt, unless
    UITARS_NATIVE_MULTISTEP selects the UI-TARS DSL contract instead.
    """
    from lmms_eval.tasks.agentcpm_gui.utils import (
        UITARS_MIND2WEB_PROMPT,
        _uitars_native_multistep_enabled,
        _qwen_prompt_thought_enabled,
    )

    kwargs = dict(lmms_eval_specific_kwargs or {})
    kwargs.setdefault("media_dir", "data/mind2web")

    native = _uitars_native_multistep_enabled()
    if native:
        # Restate history in the DSL before the builder consumes action_text.
        doc = dict(doc)
        rewritten = []
        for hist in list(doc.get("action_history") or []):
            hist = dict(hist)
            dsl = m2w_action_to_uitars_dsl(hist.get("action") or {})
            step = hist.get("step_index", 0)
            hist["action_text"] = (
                f"Thought: Executing step {step}.\nAction: {dsl}"
                if _qwen_prompt_thought_enabled()
                else f"Action: {dsl}"
            )
            rewritten.append(hist)
        doc["action_history"] = rewritten

    messages = gui_owl_android_control_dialogue_doc_to_messages(doc, kwargs)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            "role": "system",
            "content": [
                _lmms_text(
                    UITARS_MIND2WEB_PROMPT if native else GUI_OWL_MIND2WEB_SYSTEM_PROMPT
                )
            ],
        }
    return messages


_OFFICIAL_PAREN_RE = re.compile(r"\((\d+),\s*(\d+)\)")
_OFFICIAL_BRACKET_RE = re.compile(r"\[(\d+),\s*(\d+)\]")


def parse_click_point_official(output: str) -> Tuple[float, float]:
    """Official GUI-Owl grounding extraction (eval_grounding_benchmarks.py):
    scan for (x, y) pairs first, then [x, y]; take the LAST match; if nothing
    matches, default to (0, 0). Returns coordinates in [0, 1000] space."""
    text = output or ""
    matches = _OFFICIAL_PAREN_RE.findall(text)
    if not matches:
        matches = _OFFICIAL_BRACKET_RE.findall(text)
    if not matches:
        return 0.0, 0.0
    x, y = matches[-1]
    return float(x), float(y)


def point_in_bbox_1000(point_1000: Tuple[float, float], bbox_pixels: List[float], width: int, height: int) -> bool:
    """Official scoring: rescale the GT pixel bbox [x1,y1,x2,y2] into the
    virtual 1000x1000 space (floor on the top-left, ceil on the bottom-right)
    and test point-in-rectangle inclusively."""
    import math

    if not bbox_pixels or len(bbox_pixels) < 4 or width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox_pixels[:4])
    if x2 <= x1 or y2 <= y1:  # x1,y1,w,h form
        x2, y2 = x1 + float(bbox_pixels[2]), y1 + float(bbox_pixels[3])
    gx1 = math.floor(x1 * 1000.0 / width)
    gy1 = math.floor(y1 * 1000.0 / height)
    gx2 = math.ceil(x2 * 1000.0 / width)
    gy2 = math.ceil(y2 * 1000.0 / height)
    px, py = point_1000
    return gx1 <= px <= gx2 and gy1 <= py <= gy2


def parse_click_point_normalized(output: str, *, scale: float = 1000.0) -> Optional[Tuple[float, float]]:
    """Parse GUI-Owl click coordinate; return normalized [0,1] x,y."""
    try:
        from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action
    except Exception:
        parse_omni_action = None  # type: ignore

    if parse_omni_action is not None:
        action = parse_omni_action(output or "")
        if action and action.get("coordinate"):
            x, y = action["coordinate"][:2]
            return float(x) / scale, float(y) / scale

    m = re.search(r"coordinate[\"']?\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", output or "")
    if m:
        return float(m.group(1)) / scale, float(m.group(2)) / scale
    m2 = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", output or "")
    if m2:
        x, y = float(m2.group(1)), float(m2.group(2))
        if max(x, y) > 1.5:
            return x / scale, y / scale
        return x, y
    return None


def point_in_bbox_norm(point: Tuple[float, float], bbox: List[float]) -> bool:
    """bbox: [x1,y1,x2,y2] in same normalized space as point."""
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def load_screenspot_pro_docs(manifest_path: str = "data/screenspot_pro/manifest.json") -> List[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("samples", []))


def load_aitw_step_docs(steps_path: str = "data/aitw/steps/standard_test_steps.json") -> List[Dict[str, Any]]:
    path = Path(steps_path)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_android_control_test_docs(json_path: str = "data/android_control/test-AndroidControl.json") -> List[Dict[str, Any]]:
    path = Path(json_path)
    if not path.is_file():
        return []
    episodes = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Dict[str, Any]] = []
    if not isinstance(episodes, list):
        return docs
    for ep_idx, episode in enumerate(episodes):
        if not isinstance(episode, list):
            continue
        for step_idx, step in enumerate(episode):
            if not isinstance(step, dict):
                continue
            row = dict(step)
            row["episode_id"] = row.get("episode_id", f"ac_{ep_idx}")
            row["step_index"] = step_idx
            docs.append(row)
    return docs


# ---------------------------------------------------------------------------
# OmniGUI 8B task: thin re-exports into agentcpm_gui (same protocol, different metadata)
# ---------------------------------------------------------------------------

def omni_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.agentcpm_gui.utils import omni_doc_to_visual as _fn

    return _fn(doc, lmms_eval_specific_kwargs)


def omni_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.agentcpm_gui.utils import omni_doc_to_target as _fn

    return _fn(doc, lmms_eval_specific_kwargs)


def gui_owl_native_multiturn_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.agentcpm_gui.utils_eval_extensions import (
        gui_owl_native_multiturn_doc_to_messages as _fn,
    )

    return _fn(doc, lmms_eval_specific_kwargs)


def omni_process_results(doc, results, lmms_eval_specific_kwargs=None):
    from lmms_eval.tasks.agentcpm_gui.utils_eval_extensions import omni_process_results as _fn

    return _fn(doc, results, lmms_eval_specific_kwargs)


def _omni_agg(name: str):
    def _wrapper(results):
        from lmms_eval.tasks.agentcpm_gui import utils_eval_extensions as ext

        return getattr(ext, name)(results)

    return _wrapper


omni_aggregate_type_match = _omni_agg("omni_aggregate_type_match")
omni_aggregate_exact_match = _omni_agg("omni_aggregate_exact_match")
omni_aggregate_success_rate = _omni_agg("omni_aggregate_success_rate")
omni_aggregate_goal_progress = _omni_agg("omni_aggregate_goal_progress")
