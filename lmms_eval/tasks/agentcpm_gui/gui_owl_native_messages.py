"""Official GUI-Owl-1.5 Mobile-Agent prompt migration for OmniGUI.

This module deliberately treats the earlier local native prompt as invalid.
The source of truth for the system prompt and output schema is X-PLUG/MobileAgent
``Mobile-Agent-v3.5/mobile_use/utils.py`` at commit ``8cf3966e``.
The evaluation setting intentionally expands the official default
``build_messages(..., history_n=4)`` to a locked rolling ``history_n=6`` window.

Official online GUI-Owl appends model outputs to history. OmniGUI is a
step-level benchmark, so history assistant turns are teacher-forced from GT
actions into the same ``Action: ... <tool_call>`` mobile_use output space.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger
from PIL import Image


OFFICIAL_SOURCE_URL = (
    "https://raw.githubusercontent.com/X-PLUG/MobileAgent/"
    "8cf3966e/Mobile-Agent-v3.5/mobile_use/utils.py"
)
OFFICIAL_SOURCE_COMMIT = "8cf3966e"
OFFICIAL_DEFAULT_HISTORY_N = 4
LOCKED_EVAL_HISTORY_N = 6
OFFICIAL_HISTORY_N = LOCKED_EVAL_HISTORY_N
OFFICIAL_MODEL_NAME = "GUI-Owl-1.5-2B-Instruct"

# Exact raw SYSTEM_PROMPT literal from Mobile-Agent-v3.5/mobile_use/utils.py.
# Keep this immutable for provenance. The actual OmniGUI native_mt4 prompt below
# patches only the coordinate wording, because this benchmark is scored in
# normalized [0, 1000] coordinates rather than real screenshot pixels.
OFFICIAL_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", \\"camera\\", \\"clear\\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `interact`: Resolve the blocking window by interacting with the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "interact", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`,and `action=interact`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''

def _patch_prompt_to_normalized_1000(prompt: str) -> str:
    return (
        prompt.replace(
            "* The screen's resolution is 1000x1000.",
            "* Coordinates use a normalized 1000x1000 coordinate space, regardless of the screenshot's real pixel resolution: [0,0] is top-left and [1000,1000] is bottom-right.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
            "(x, y): normalized coordinates in the 0-1000 screen space; x increases left-to-right and y increases top-to-bottom. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
            "(x, y): normalized end coordinates in the 0-1000 screen space. Required only by `action=swipe`.",
        )
    )


def _patch_prompt_to_resized_pixel(prompt: str) -> str:
    return (
        prompt.replace(
            "* The screen's resolution is 1000x1000.",
            "* Coordinates use the actual resized pixel space of the screenshot after Qwen vision preprocessing. The current screenshot size will be stated in the user message.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
            "(x, y): pixel coordinates in the resized screenshot stated in the current user message; x increases left-to-right and y increases top-to-bottom. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
            "(x, y): resized-pixel end coordinates in the screenshot size stated in the current user message. Required only by `action=swipe`.",
        )
    )


GUI_OWL_NATIVE_COORDINATE_SYSTEM = "normalized_0_1000"
GUI_OWL_NATIVE_SYSTEM_PROMPT = _patch_prompt_to_normalized_1000(OFFICIAL_SYSTEM_PROMPT)
GUI_OWL_NATIVE_RESIZED_PIXEL_COORDINATE_SYSTEM = "qwen_resized_pixel"
GUI_OWL_NATIVE_RESIZED_PIXEL_SYSTEM_PROMPT = _patch_prompt_to_resized_pixel(OFFICIAL_SYSTEM_PROMPT)
DEFAULT_HISTORY_N = LOCKED_EVAL_HISTORY_N

ENGLISH_APPS = {
    "amazon", "instagram", "snapchat", "spotify", "x", "youtube", "tiktok",
    "duolingo", "googletranslate", "imdb", "ted", "vimeo", "redbulltv", "tasty",
}


def _sorted_action_history(action_history: List[Dict]) -> List[Dict]:
    return sorted(action_history, key=lambda ah: ah.get("step_index", 0))


def _task_instruction(doc: Dict) -> str:
    app = doc.get("app", "")
    if app in ENGLISH_APPS:
        return doc.get("instruction_en", doc.get("instruction", ""))
    return doc.get("instruction", "")


def _action_history_box_coord(x: int, y: int, img_w: int, img_h: int) -> List[int]:
    """Return norm1000 coords from merged_sorted action_history start_box/end_box.

    OmniGUI ``action_history[].action_text`` stores click/swipe boxes in the same
    normalized [0, 1000] space as model outputs — not raw screenshot pixels.
    Re-scaling by image width/height would double-normalize (e.g. T0001
    click(500,810) must stay [500,810], not become [926,675] on 540x1200).
    """
    return [int(x), int(y)]


def _swipe_action_type_from_coords(start: List[int], end: List[int]) -> int:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > abs(dy):
        return 6 if dx > 0 else 5
    return 4 if dy > 0 else 3


def parse_action_history_text(action_text: str, *, img_w: int, img_h: int) -> Optional[Dict[str, Any]]:
    """Parse OmniGUI action_history text into the unified action dict."""
    text = (action_text or "").strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"wait()", "wait"} or lower.startswith("wait("):
        return {"action_type": -1}
    if lower.startswith("task_complete") or "complete" in lower:
        return {"action_type": 10}
    if lower.startswith("task_impossible") or "impossible" in lower:
        return {"action_type": 11}
    if lower.startswith("back") or lower.startswith("press_back"):
        return {"action_type": 8}
    if lower.startswith("home") or lower.startswith("press_home"):
        return {"action_type": 9}

    m = re.search(r"click\s*\(\s*start_box\s*=\s*'\((\d+)\s*,\s*(\d+)\)'\s*\)", text, re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return {"action_type": 0, "coordinate": _action_history_box_coord(x, y, img_w, img_h)}

    m = re.search(
        r"(swipe|scroll)[^(]*\(\s*start_box\s*=\s*'\((\d+)\s*,\s*(\d+)\)'\s*,\s*end_box\s*=\s*'\((\d+)\s*,\s*(\d+)\)'\s*\)",
        text,
        re.I,
    )
    if m:
        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        start = _action_history_box_coord(x1, y1, img_w, img_h)
        end = _action_history_box_coord(x2, y2, img_w, img_h)
        swipe_type = _swipe_action_type_from_coords(start, end)
        return {
            "action_type": swipe_type,
            "coordinate": start,
            "start_coordinate": start,
            "end_coordinate": end,
        }

    m = re.search(r"type\s*\([^)]*text\s*=\s*['\"]([^'\"]*)['\"]", text, re.I)
    if m:
        return {"action_type": 7, "text": m.group(1)}
    m = re.search(r"type\s*\(\s*['\"]([^'\"]*)['\"]", text, re.I)
    if m:
        return {"action_type": 7, "text": m.group(1)}

    try:
        from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action

        parsed = parse_omni_action(text)
        if parsed and "action_type" in parsed:
            return parsed
    except Exception:
        pass
    return None


def omni_action_to_mobile_use_tool_call(action: Dict[str, Any]) -> Dict[str, Any]:
    """Map OmniGUI action_type dict to GUI-Owl mobile_use tool_call payload."""
    at = int(action.get("action_type", -1))
    args: Dict[str, Any] = {}

    def _coord(key: str = "coordinate") -> Optional[List[int]]:
        val = action.get(key) or action.get("point")
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return [int(float(val[0])), int(float(val[1]))]
        return None

    if at in (0, 1):
        args["action"] = "click"
        c = _coord()
        if c:
            args["coordinate"] = c
    elif at == 2:
        args["action"] = "long_press"
        c = _coord()
        if c:
            args["coordinate"] = c
        dur = action.get("duration")
        if dur is not None:
            try:
                args["time"] = float(dur) / 1000.0 if float(dur) > 20 else float(dur)
            except (TypeError, ValueError):
                args["time"] = 2.0
        else:
            args["time"] = 2.0
    elif at in (3, 4, 5, 6):
        args["action"] = "swipe"
        start = _coord("start_coordinate") or _coord("coordinate")
        end = _coord("end_coordinate")
        if start:
            args["coordinate"] = start
        if end:
            args["coordinate2"] = end
    elif at == 7:
        args["action"] = "type"
        args["text"] = str(action.get("text", "") or "")
    elif at == 8:
        args["action"] = "system_button"
        args["button"] = "Back"
    elif at == 9:
        args["action"] = "system_button"
        args["button"] = "Home"
    elif at == 10:
        args["action"] = "terminate"
        args["status"] = "success"
    elif at == 11:
        args["action"] = "terminate"
        args["status"] = "failure"
    else:
        args["action"] = "wait"
        args["time"] = 2.0

    return {"name": "mobile_use", "arguments": args}


def mobile_use_action_summary(action: Dict[str, Any]) -> str:
    """Short Action line for synthetic history outputs."""
    tc = omni_action_to_mobile_use_tool_call(action)
    act = tc["arguments"].get("action", "wait")
    if act == "click":
        c = tc["arguments"].get("coordinate")
        return f"Tap at {c}" if c else "Tap on the target element"
    if act == "swipe":
        start = tc["arguments"].get("coordinate")
        end = tc["arguments"].get("coordinate2")
        return f"Swipe from {start} to {end}" if start and end else "Swipe on the screen"
    if act == "type":
        return f'Type "{tc["arguments"].get("text", "")}"'
    if act == "system_button":
        return f'Press {tc["arguments"].get("button", "Back")}'
    if act == "terminate":
        return "Finish the task" if tc["arguments"].get("status") == "success" else "Task impossible"
    if act == "long_press":
        return "Long press on the target element"
    if act == "wait":
        return "Wait for the screen to update"
    return str(act)


def format_native_assistant_turn(action: Dict[str, Any]) -> str:
    tc = omni_action_to_mobile_use_tool_call(action)
    summary = mobile_use_action_summary(action)
    return (
        f"Action: {summary}\n"
        f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"
    )


def _official_previous_action_summary(output_text: str, model_name: str) -> str:
    """Mirror official build_messages summary extraction for older turns."""
    text = output_text
    if model_name.endswith(".mem"):
        if "<tool_call>" in text:
            text = text.split("<tool_call>")[0].strip()
    else:
        if "Action:" in text and "<tool_call>" in text:
            text = text.split("Action:")[1].split("<tool_call>")[0].strip()
    return text


def _official_instruction_prompt(
    instruction: str,
    history_output: List[Dict[str, str]],
    *,
    model_name: str,
    history_n: int,
) -> str:
    """Port of official Mobile-Agent-v3.5 build_messages instruction block."""
    current_step = len(history_output)
    history_start_idx = max(0, current_step - history_n)

    previous_actions = []
    for i in range(history_start_idx):
        if i < len(history_output):
            text = _official_previous_action_summary(history_output[i]["output"], model_name)
            previous_actions.append(f"Step {i + 1}: {text}")

    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    today = datetime.today()
    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    formatted_date = today.strftime("%Y-%m-%d") + " " + weekday_names[today.weekday()]
    date_info = f"Today's date is: {formatted_date}."

    return (
        f"Please generate the next move according to the UI screenshot, "
        f"instruction and previous actions.\n\n"
        f"Instruction: {date_info}{instruction}\n\n"
        f"Previous actions:\n{previous_actions_str}"
    )


def _resolve_media_path(path: str, media_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    if media_dir:
        return os.path.join(media_dir, path)
    return path


def _lmms_text(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _lmms_image(path: str) -> Dict[str, str]:
    return {"type": "image", "url": path}


def build_messages_compatible(
    image_path: str,
    instruction: str,
    history_output: List[Dict[str, str]],
    model_name: str,
    history_n: int = LOCKED_EVAL_HISTORY_N,
    system_prompt: str = GUI_OWL_NATIVE_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Official build_messages layout with the locked rolling history window."""
    history_n = LOCKED_EVAL_HISTORY_N
    instruction_prompt = _official_instruction_prompt(
        instruction,
        history_output,
        model_name=model_name,
        history_n=history_n,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [_lmms_text(system_prompt)]}
    ]

    history_len = min(history_n, len(history_output))
    if history_len > 0:
        for idx, item in enumerate(history_output[-history_n:]):
            if idx == 0:
                messages.append({
                    "role": "user",
                    "content": [_lmms_text(instruction_prompt), _lmms_image(item["image"])],
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [_lmms_image(item["image"])],
                })
            messages.append({
                "role": "assistant",
                "content": [_lmms_text(item["output"])],
            })
        messages.append({
            "role": "user",
            "content": [_lmms_image(image_path)],
        })
    else:
        messages.append({
            "role": "user",
            "content": [_lmms_text(instruction_prompt), _lmms_image(image_path)],
        })

    return messages


def _history_output_from_doc(
    doc: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    *,
    media_dir: str,
) -> List[Dict[str, str]]:
    history_output: List[Dict[str, str]] = []

    for ah in _sorted_action_history(action_history):
        image_path = ah.get("image_path")
        if not image_path:
            continue
        resolved = _resolve_media_path(str(image_path), media_dir)
        if resolved and not os.path.isabs(resolved):
            resolved = os.path.abspath(resolved)
        if not os.path.isfile(resolved):
            eval_logger.warning(
                f"Skip missing official history screenshot: episode={doc.get('episode_id')} "
                f"step={ah.get('step_index', 0)} path={resolved}"
            )
            continue

        try:
            with Image.open(resolved) as im:
                img_w, img_h = int(im.width), int(im.height)
        except Exception:
            img_w = int(doc.get("image_width", 540) or 540)
            img_h = int(doc.get("image_height", 1200) or 1200)

        action = parse_action_history_text(ah.get("action_text", ""), img_w=img_w, img_h=img_h)
        if action is None:
            action = {"action_type": -1}
        history_output.append({"output": format_native_assistant_turn(action), "image": resolved})

    return history_output


def gui_owl_native_doc_to_messages_impl(
    doc: Dict,
    lmms_eval_specific_kwargs: Optional[Dict],
    *,
    multiturn: bool,
) -> List[Dict]:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    media_dir = lmms_eval_specific_kwargs.get("media_dir", "data/benchmark")
    coordinate_type = str(lmms_eval_specific_kwargs.get("coordinate_type", GUI_OWL_NATIVE_COORDINATE_SYSTEM))
    system_prompt = (
        GUI_OWL_NATIVE_RESIZED_PIXEL_SYSTEM_PROMPT
        if coordinate_type == GUI_OWL_NATIVE_RESIZED_PIXEL_COORDINATE_SYSTEM
        else GUI_OWL_NATIVE_SYSTEM_PROMPT
    )
    requested_history_n = int(lmms_eval_specific_kwargs.get("max_history_turns", LOCKED_EVAL_HISTORY_N))
    if requested_history_n != LOCKED_EVAL_HISTORY_N:
        eval_logger.warning(
            f"GUI-Owl locked eval ignores max_history_turns={requested_history_n}; "
            f"using history_n={LOCKED_EVAL_HISTORY_N}"
        )

    current_image = doc.get("current_image") or doc.get("image") or doc.get("image_path")
    if isinstance(current_image, str):
        current_image = _resolve_media_path(current_image, media_dir)
        if current_image and not os.path.isabs(current_image):
            current_image = os.path.abspath(current_image)
    elif current_image:
        current_image = _resolve_media_path(str(current_image), media_dir)
        if current_image and not os.path.isabs(current_image):
            current_image = os.path.abspath(current_image)

    if current_image and not os.path.isfile(str(current_image)):
        eval_logger.warning(
            f"Missing current screenshot: episode={doc.get('episode_id')} path={current_image}"
        )

    action_history = doc.get("action_history", []) or []
    history_output = _history_output_from_doc(doc, action_history, media_dir=media_dir) if multiturn else []

    return build_messages_compatible(
        str(current_image or ""),
        _task_instruction(doc),
        history_output,
        OFFICIAL_MODEL_NAME,
        history_n=LOCKED_EVAL_HISTORY_N,
        system_prompt=system_prompt,
    )


def gui_owl_native_current_doc_to_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """GUI-Owl official mobile_use prompt, current screenshot only."""
    return gui_owl_native_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs, multiturn=False
    )


def gui_owl_native_multiturn_doc_to_messages(
    doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """GUI-Owl official mobile_use + locked history_n=6 multi-turn migration."""
    return gui_owl_native_doc_to_messages_impl(
        doc, lmms_eval_specific_kwargs, multiturn=True
    )
