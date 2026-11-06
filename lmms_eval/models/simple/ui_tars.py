"""
UI-TARS wrapper for OmniGUI image-only evaluation.

Supported checkpoints:
- UI-TARS-2B-SFT (model_type=qwen2_vl)
- UI-TARS-1.5-7B (model_type=qwen2_5_vl) — cross-backbone transfer carrier.

The GeoPACT prune hook consumer differs between the two patched modeling
files: the qwen2_vl patch reads `_geopact_*` attributes off the composite
`Qwen2VLModel` (fcg.model), while the qwen2_5_vl patch reads them off the
`Qwen2_5_VLForConditionalGeneration` itself. `_geopact_target()` resolves the
correct object from config.model_type.

Multi-round GeoPACT session-KV (write-once KVR) orchestration is ported from
`qwen3_vl_gui.py`. It activates only under the same env gates the GUI-Owl
wrapper uses (`GEOPACT_SESSION_KV=1`, `GEOPACT_FORCE_EPISODE_ORDER=1`,
`GEOPACT_VISION_EMBED_CACHE`, sticky prune scopes, `GUIOWL_NATIVE_MULTITURN`);
with all gates off, `generate_until` runs the original single-step body
unchanged. The write-once serving internals (suffix-prefill detection, decode
pin, `_geopact_prefill_payload`) live in the patched qwen2_5_vl modeling file,
so unlike the qwen3 wrapper no `prepare_inputs_for_generation` monkeypatch is
needed here — `_call_geopact_generate` only installs the pinned-decode
attributes on the FCG.

Requires is_simple=False — messages come from utils.doc_to_messages only.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.geopact_vl_hook import (
    _dispatch_module_a_sideload,
    _flush_module_a_sideload_queue,
    _module_a_sideload_enabled,
    _should_cache_geopact_pil_image,
    make_geopact_hook,
    release_pending_yolo_gpu,
    scope_requires_episode_order,
)
from lmms_eval.models.geopact_step_timing import (
    begin_step_timing,
    cuda_sync_enabled,
    get_step_ledger,
    install_kv_bytes_probe,
    kv_cache_bytes_from,
    mem_audit_enabled,
    merge_exclusive_into_usage,
)
from lmms_eval.models.geopact_vl_prepare import (
    _dialogue_kv_use_native_full_prefill,
    prepare_active_prefill_inputs,
)
from lmms_eval.models.geopact_session_kv import (
    DialogueKVCapture,
    backup_session_full_prefill,
    build_dialogue_generate_payload,
    clear_geopact_mrope_decode_state,
    clear_session_kv,
    dialogue_kv_enabled,
    finalize_dialogue_kv_after_generate,
    install_dialogue_kv_capture,
    kv_flow_mode_label,
    prepare_dialogue_generate_payload,
    session_kv_enabled,
    session_episode_key,
    session_step_index,
    slide_dialogue_kv_prefix,
    slide_kv_on_roll_enabled,
    stage_pending_last_full_frame,
    strip_forward_generate_keys,
)
from lmms_eval.models.geopact_vision_cache import vision_embed_cache_enabled
from lmms_eval.models.geopact_decode_contract import (
    attach_pinned_decode_contract,
    contract_complete,
    resolve_rope_for_step,
    should_use_pinned_generate,
    validate_pinned_decode_contract,
)
from lmms_eval.models.simple.qwen_vl_generation_timing import (
    call_qwen_vl_generate_with_timing,
    resolve_early_stop_prompt_len,
)
from lmms_eval.models.simple.qwen_vl_omnigui_common import (
    assert_messages_request,
    build_qwen_vl_usage_stats,
    cache_geopact_image,
    canonicalize_omni_action_output,
    convert_messages_to_qwen,
    extract_episode_image_paths_from_doc,
    format_response_with_usage,
    join_module_a_prewarm,
    log_messages_debug,
    log_vision_preprocess_debug,
    prepare_geopact_episode_context,
    prewarm_module_a_subprocess_in_eval_process,
    read_processor_vision_bounds,
    set_geopact_episode_image_context,
    start_module_a_prewarm_background,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("qwen_vl_utils not found; install via `pip install qwen-vl-utils`.")
    process_vision_info = None


def _native_multiturn_eval_enabled() -> bool:
    return os.environ.get("GUIOWL_NATIVE_MULTITURN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _force_episode_order_enabled() -> bool:
    return os.environ.get("GEOPACT_FORCE_EPISODE_ORDER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ss_latency_log_enabled() -> bool:
    return os.environ.get("GEOPACT_SS_LATENCY_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_UITARS_COORD_OPEN_BRACKET = re.compile(r"\[(\s*\d+\s*,\s*\d+\s*)\)")
_UITARS_COORD_OPEN_PAREN = re.compile(r"\((\s*\d+\s*,\s*\d+\s*)\]")


def _uitars_repair_action_text(ans: str) -> str:
    """Deterministic cleanup of UI-TARS-1.5 action text before Omni parsing.

    The Qwen2.5-VL chat template leaks a leading ``assistant`` role header
    into the decoded slice, the model repeats whole turns when no stop term
    fires, and its box-token habit closes JSON coordinate lists with ``)``
    (e.g. ``"coordinate": [518, 288)``). All three break ``parse_omni_action``.
    Applied identically on every arm (dense and pruned), so gate equivalence
    and retention comparisons are unaffected.
    """
    text = (ans or "").strip()
    while text.startswith("assistant"):
        text = text[len("assistant"):].lstrip("\n :")
    cut = text.find("\nassistant")
    if cut > 0:
        text = text[:cut]
    text = _UITARS_COORD_OPEN_BRACKET.sub(r"[\1]", text)
    text = _UITARS_COORD_OPEN_PAREN.sub(r"[\1]", text)
    return text.strip()


_UITARS_COORD_FIELD = re.compile(
    r'("(?:coordinate|start_coordinate|end_coordinate)"\s*:\s*)\[(\d+)\s*,\s*(\d+)\]'
)


def _uitars_unwrap_tool_call_json(text: str) -> str:
    """Flatten a bare ``mobile_use`` tool-call object into a flat action dict.

    ``parse_omni_action`` only routes through ``_convert_tool_call_to_action``
    when the payload is wrapped in ``<tool_call>`` tags. UI-TARS-1.5 emits the
    same object *unwrapped* (measured on Mind2Web: 9% of dense responses), and
    the flat-JSON path finds no recognised action key in ``{"name": ...,
    "arguments": ...}``, so the whole step is discarded as unparseable. Lifting
    ``arguments`` to the top level restores it without touching the scorer.
    Applied identically on every arm, so retention comparisons are unaffected.
    """
    if not text or "arguments" not in text:
        return text
    stripped = text.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return text
    try:
        obj = json.loads(stripped[start : end + 1])
    except (ValueError, TypeError):
        return text
    if not isinstance(obj, dict):
        return text
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return text
    if not isinstance(args, dict) or not args:
        return text
    flat = dict(args)
    # Format 2: {"name": "click", "arguments": {"coordinate": [...]}} carries the
    # verb in `name`; Format 1 carries it in `arguments.action`.
    if "action" not in flat:
        name = obj.get("name")
        if isinstance(name, str) and name and name != "mobile_use":
            flat["action"] = name
    if "action" not in flat:
        return text
    return json.dumps(flat, ensure_ascii=False)


def _uitars_coord_rescale_enabled() -> bool:
    return os.environ.get("UITARS_COORD_RESCALE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _uitars_rescale_coords_to_1000(text: str, grid_thw: Any, patch_size: int = 14) -> str:
    """Rescale UI-TARS-1.5 absolute-pixel coordinates into [0,1000] space.

    UI-TARS-1.5 (Qwen2.5-VL) grounds in the smart-resized input-image pixel
    space, while the shared GUI-Owl scorers (``point_in_bbox_1000`` /
    ``evaluate_omni_step``) consume the virtual 1000x1000 space. Smart resize
    preserves aspect ratio, so dividing by the resized dims of the current
    (last) image matches the original-image normalization the ground truth
    uses. Uniform across arms; disable with UITARS_COORD_RESCALE=0.
    """
    if not text or grid_thw is None:
        return text
    try:
        rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
        if not rows:
            return text
        last = rows[-1]
        if isinstance(last, (int, float)):  # flat [t, h, w]
            last = rows
        _, gh, gw = (int(v) for v in last[-3:])
        resized_h = gh * patch_size
        resized_w = gw * patch_size
        if resized_h <= 0 or resized_w <= 0:
            return text
    except Exception:
        return text

    def _sub(m: "re.Match[str]") -> str:
        x = min(max(round(int(m.group(2)) * 1000.0 / resized_w), 0), 1000)
        y = min(max(round(int(m.group(3)) * 1000.0 / resized_h), 0), 1000)
        return f"{m.group(1)}[{x}, {y}]"

    return _UITARS_COORD_FIELD.sub(_sub, text)


# UI-TARS-1.5 native DSL, tolerant to a truncated closing paren:
#   click(start_box='(x,y)') / click(coordinate = [x, y]) / click(point='(x,y)')
#   click(point='<point>x y</point>')  (official 1.5 grounding format,
#   space-separated inside <point> tags)
_UITARS_DSL_XY = (
    r"(?:start_box|start_point|point|coordinate)\s*=\s*['\"]?"
    r"(?:<point>)?[\[\(]?\s*(\d+)[\s,]+(\d+)"
)
_UITARS_DSL_END_XY = (
    r"(?:end_box|end_point)\s*=\s*['\"]?(?:<point>)?[\[\(]?\s*(\d+)[\s,]+(\d+)"
)
_UITARS_DSL_CLICKISH = re.compile(
    r"(click|left_single|left_double|double_click|right_single|long_press)\s*\([^()]*?"
    + _UITARS_DSL_XY,
    re.IGNORECASE,
)
_UITARS_DSL_DRAGISH = re.compile(r"(drag|swipe|scroll)\s*\(", re.IGNORECASE)
_UITARS_DSL_TYPE = re.compile(
    r"type\s*\([^()]*?(?:content|text)\s*=\s*['\"](.*?)['\"]\s*\)?",
    re.IGNORECASE | re.DOTALL,
)
# Mind2Web web action space (UITARS_MIND2WEB_PROMPT): type and select both carry
# a target point, unlike the mobile `type(content='')`. Captured separately so
# the mobile behaviour above is untouched when no point is present.
_UITARS_DSL_WEB_TYPISH = re.compile(
    r"(type|select)\s*\([^()]*?" + _UITARS_DSL_XY + r"[^()]*?"
    r"(?:content|text)\s*=\s*['\"](.*?)['\"]",
    re.IGNORECASE | re.DOTALL,
)
_UITARS_DSL_SELECT = re.compile(
    r"select\s*\([^()]*?" + _UITARS_DSL_XY,
    re.IGNORECASE,
)
_UITARS_DSL_SIMPLE = re.compile(
    r"(press_back|press_home|finished|terminate|wait|call_user|impossible|infeasible)\s*\(",
    re.IGNORECASE,
)
_UITARS_DSL_DIRECTION = re.compile(r"direction\s*=\s*['\"]?(up|down|left|right)", re.IGNORECASE)


def _uitars_dsl_to_omni_json(text: str) -> str:
    """Convert a UI-TARS-1.5 native DSL action into Omni-parseable JSON.

    Deterministic scoring glue applied identically on every arm: when the
    model answers in its trained ``Thought ... Action: verb(...)`` style
    instead of the prompted JSON tool call, extract the LAST DSL call and
    re-emit it as a flat action dict for ``parse_omni_action``. Coordinates
    stay in the model's absolute pixel space; the [0,1000] rescale runs
    afterwards on the emitted JSON fields. Returns the input unchanged when
    no DSL action is found or the text already parses as an action JSON.
    """
    if not text:
        return text
    if "{" in text:
        try:
            from lmms_eval.tasks.agentcpm_gui.utils import parse_omni_action

            if parse_omni_action(text) is not None:
                return text
        except Exception:
            pass

    # Collect every candidate action with its position; keep the LAST one
    # (the model's final Action line supersedes actions mentioned in thought).
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    for mm in _UITARS_DSL_CLICKISH.finditer(text):
        action = {
            "click": "click",
            "left_single": "click",
            "left_double": "double_click",
            "double_click": "double_click",
            "right_single": "click",
            "long_press": "long_press",
        }[mm.group(1).lower()]
        candidates.append((mm.start(), {
            "action": action,
            "coordinate": [int(mm.group(2)), int(mm.group(3))],
        }))

    for dm in _UITARS_DSL_DRAGISH.finditer(text):
        # Argument bodies nest quotes/parens (start_box='(x,y)'), so scan a
        # fixed window after the verb instead of bracket-matching.
        body = text[dm.end():dm.end() + 160]
        start = re.search(_UITARS_DSL_XY, body, re.IGNORECASE)
        end = re.search(_UITARS_DSL_END_XY, body, re.IGNORECASE)
        direction = _UITARS_DSL_DIRECTION.search(body)
        if start and end:
            candidates.append((dm.start(), {
                "action": "swipe",
                "start_coordinate": [int(start.group(1)), int(start.group(2))],
                "end_coordinate": [int(end.group(1)), int(end.group(2))],
            }))
        elif direction:
            payload: Dict[str, Any] = {"action": f"swipe_{direction.group(1).lower()}"}
            if start:
                payload["coordinate"] = [int(start.group(1)), int(start.group(2))]
            candidates.append((dm.start(), payload))

    # Web forms first: type/select with a point. Recorded before the mobile
    # type pattern so the later dedup by position keeps the richer payload.
    web_spans: List[Tuple[int, int]] = []
    for wm in _UITARS_DSL_WEB_TYPISH.finditer(text):
        verb = wm.group(1).lower()
        payload_web: Dict[str, Any] = {
            "action": "select" if verb == "select" else "input",
            "coordinate": [int(wm.group(2)), int(wm.group(3))],
            "text": wm.group(4),
        }
        candidates.append((wm.start(), payload_web))
        web_spans.append((wm.start(), wm.end()))

    def _inside_web(pos: int) -> bool:
        return any(s <= pos < e for s, e in web_spans)

    for tm in _UITARS_DSL_TYPE.finditer(text):
        if _inside_web(tm.start()):
            continue
        candidates.append((tm.start(), {"action": "input", "text": tm.group(1)}))

    # select without a content field still names a target element.
    for sm2 in _UITARS_DSL_SELECT.finditer(text):
        if _inside_web(sm2.start()):
            continue
        candidates.append((sm2.start(), {
            "action": "select",
            "coordinate": [int(sm2.group(1)), int(sm2.group(2))],
        }))

    for sm in _UITARS_DSL_SIMPLE.finditer(text):
        verb = sm.group(1).lower()
        if verb == "press_back":
            act: Dict[str, Any] = {"action": "back"}
        elif verb == "press_home":
            act = {"action": "home"}
        elif verb in ("finished", "terminate"):
            act = {"action": "terminate", "status": "success"}
        elif verb in ("impossible", "infeasible"):
            act = {"action": "terminate", "status": "failure"}
        else:  # wait / call_user
            act = {"action": "wait"}
        candidates.append((sm.start(), act))

    if not candidates:
        return text
    candidates.sort(key=lambda c: c[0])
    chosen = candidates[-1][1]

    # `select` has no OmniActionType, so parse_omni_action returns None for a
    # flat select dict and canonicalize_omni_action_output then infers TAP from
    # the bare coordinate, turning every SELECT into a CLICK. GUI-Owl avoids
    # this by leaving select inside its mobile_use tool call, where the same
    # None makes canonicalization a no-op and the Mind2Web scorer reads the verb
    # directly. Emit the identical carrier so both backbones are scored by one
    # route, rather than teaching the shared parser a verb that would change
    # canonical output on every published arm.
    if chosen.get("action") == "select":
        args = {k: v for k, v in chosen.items() if k != "action"}
        args["action"] = "select"
        tc = {"name": "mobile_use", "arguments": args}
        return f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>"

    return json.dumps(chosen, ensure_ascii=False)


@register_model("ui_tars")
class UITARS(lmms):
    """UI-TARS-2B-SFT wrapper (Qwen2-VL) for OmniGUI image-only evaluation."""

    is_simple = False

    def __init__(
        self,
        pretrained: str = "ByteDance-Seed/UI-TARS-2B-SFT",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        visual_prune_ratio: Optional[float] = None,
        visual_prune_policy: Optional[str] = None,
        visual_prune_scope: Optional[str] = "image",
        visual_prune_target: Optional[str] = "current",
        miu_detector: Optional[str] = "canny_morph",
        attach_usage_stats: bool = True,
        early_stop_on_action: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.visual_prune_ratio = visual_prune_ratio
        self.visual_prune_policy = visual_prune_policy
        self.visual_prune_scope = visual_prune_scope
        target = str(visual_prune_target or "current").strip().lower()
        self.visual_prune_target = "history" if target == "history" else "current"
        self.miu_detector = miu_detector or "canny_morph"
        self.attach_usage_stats = attach_usage_stats
        self.early_stop_on_action = early_stop_on_action
        self._explicit_min_pixels = min_pixels
        self._explicit_max_pixels = max_pixels

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # UI-TARS-1.5 emits long thought chains and DSL actions; let the shared
        # early-stop criteria also halt on a complete DSL call (uniform on all
        # arms; harmless for models that emit Omni JSON, which stops earlier).
        os.environ.setdefault("UITARS_DSL_EARLY_STOP", "1")

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForVision2Seq.from_pretrained(pretrained, **model_kwargs).eval()

        # P4 (env-gated, default off): fuse RMSNorm kernels via torch.compile.
        from lmms_eval.models.geopact_compile_norms import maybe_compile_norms

        maybe_compile_norms(self._model)

        prewarm_handle = None
        post_model_t0 = time.perf_counter()
        if self._use_visual_pruning():
            prewarm_handle = start_module_a_prewarm_background()
        self._geopact_module_a_prewarm_handle = prewarm_handle

        processor_kwargs = {"use_fast": False}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(pretrained, **processor_kwargs)
        bounds = read_processor_vision_bounds(self.processor)
        self.min_pixels = min_pixels if min_pixels is not None else bounds.get("min_pixels")
        self.max_pixels = max_pixels if max_pixels is not None else bounds.get("max_pixels")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        target_module = self._geopact_target()
        target_module._geopact_last_image = None
        target_module._geopact_prune_scope = str(self.visual_prune_scope or "image").lower()
        target_module._geopact_prune_target = self.visual_prune_target
        target_module._geopact_miu_detector = str(self.miu_detector or "canny_morph")
        if self._use_visual_pruning():
            target_module._geopact_tokenizer = self._tokenizer
            target_module._geopact_prune_ratio = float(self.visual_prune_ratio)
            target_module._geopact_prune_policy = str(self.visual_prune_policy or "")
            target_module._geopact_projector_prune_hook = make_geopact_hook(
                visual_prune_ratio=visual_prune_ratio,
                visual_prune_policy=visual_prune_policy or "",
                miu_detector=self.miu_detector,
                visual_prune_scope=self.visual_prune_scope or "image",
            )
        else:
            target_module._geopact_projector_prune_hook = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        if self.rank == 0:
            hook = getattr(target_module, "_geopact_projector_prune_hook", None)
            eval_logger.info(
                f"UITARS init: model_type={getattr(self._config, 'model_type', '?')} "
                f"vision_bounds={bounds} prune_target={self.visual_prune_target} "
                f"geopact_hook={'set' if hook is not None else 'None'} "
                f"hook_module={type(target_module).__name__}"
            )

        if prewarm_handle is not None:
            join_module_a_prewarm(prewarm_handle, where="init")
        elif self._use_visual_pruning():
            prewarm_module_a_subprocess_in_eval_process()
        if self._use_visual_pruning() and self.rank == 0:
            eval_logger.info(
                "GeoPACT: UITARS post-model init wall {:.0f} ms",
                (time.perf_counter() - post_model_t0) * 1000.0,
            )

    def _ensure_module_a_prewarm_done(self) -> None:
        handle = getattr(self, "_geopact_module_a_prewarm_handle", None)
        if handle is not None and not getattr(handle, "joined", False):
            join_module_a_prewarm(handle, where="generate")
            self._geopact_module_a_prewarm_handle = None

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if not hasattr(self, "accelerator"):
            return self._model
        cached = self.__dict__.get("_geopact_unwrapped_model")
        if cached is None:
            cached = self.accelerator.unwrap_model(self._model)
            self.__dict__["_geopact_unwrapped_model"] = cached
        return cached

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not implemented for UITARS")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round not implemented for UITARS")

    def flatten(self, input):
        return [j for i in input for j in i]

    def _geopact_target(self):
        """Object the patched modeling file reads `_geopact_*` attributes from.

        qwen2_5_vl patch: Qwen2_5_VLForConditionalGeneration (the fcg itself).
        qwen2_vl patch: the composite Qwen2VLModel at fcg.model.
        """
        fcg = self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model
        model_type = str(getattr(getattr(fcg, "config", None), "model_type", "") or "")
        if model_type == "qwen2_5_vl":
            return fcg
        return getattr(fcg, "model", fcg)

    def _is_qwen25_carrier(self) -> bool:
        return str(getattr(self._config, "model_type", "") or "") == "qwen2_5_vl"

    def _use_visual_pruning(self) -> bool:
        # ratio == 1.0 is intentionally allowed: the cross-backbone no-op gate
        # exercises the full hook/scatter/RoPE path with a keep-all selection
        # and requires output equivalence against the dense run.
        return (
            self.visual_prune_ratio is not None
            and 0.0 < self.visual_prune_ratio <= 1.0
            and self.visual_prune_policy not in {None, "", "none", "off", "false"}
        )

    # ── GeoPACT session-KV orchestration (ported from qwen3_vl_gui) ──────────

    def _skip_sliding_suffix_pil_cache(self) -> bool:
        """Sliding rewrite: rectify reuses last_full_frame PIL; skip redundant load."""
        return kv_flow_mode_label(self.model) == "prune_rewrite"

    @staticmethod
    def _cap_gui_max_new_tokens(requested: int, *, early_stop_on_action: bool) -> int:
        """Decode length ceiling for GUI action JSON (early-stop is the primary exit).

        Default follows frozen baseline protocol (``max_new_tokens=4096``). Optional
        ``GEOPACT_GUI_MAX_NEW_TOKENS`` lowers the ceiling for perf experiments only.
        """
        req = int(requested)
        if not early_stop_on_action:
            return req
        env_cap = os.environ.get("GEOPACT_GUI_MAX_NEW_TOKENS", "").strip()
        if not env_cap:
            return req
        return min(req, int(env_cap))

    @staticmethod
    def _geopact_generate_audit_thresholds() -> tuple[float, float]:
        """(max_decode_tokens, max_generation_ms) before rank-0 warning."""
        max_toks = float(os.environ.get("GEOPACT_GENERATE_AUDIT_MAX_TOKENS", "512"))
        max_ms = float(os.environ.get("GEOPACT_GENERATE_AUDIT_MAX_MS", "15000"))
        return max_toks, max_ms

    def _hf_safe_generate_kwargs(self, gen_kwargs: dict) -> dict:
        return strip_forward_generate_keys(dict(gen_kwargs))

    def _call_geopact_generate(self, **gen_kwargs):
        """Pinned GeoPACT decode: install the write-once attrs on the carrier.

        Unlike the qwen3 wrapper (which monkeypatches
        ``prepare_inputs_for_generation``), the patched qwen2_5_vl modeling file
        consumes ``_geopact_generate_position_ids`` /
        ``_geopact_generate_rope_deltas`` / ``_geopact_generate_cache_start``
        natively in both ``prepare_inputs_for_generation`` (prefill / suffix
        prefill pin) and ``forward`` (decode anchor continuation), so the
        wrapper only sets and clears the attributes around ``generate``.
        """
        from lmms_eval.models.geopact_block_kv_cache import (
            GeoPactBlockKVCache,
            block_kv_cache_enabled,
        )

        target = self._geopact_target()
        # P3: seed generate with a block-preallocated cache so decode appends
        # in place instead of re-cat'ing the whole KV every token (env-gated).
        if block_kv_cache_enabled() and gen_kwargs.get("past_key_values") is None:
            gen_kwargs["past_key_values"] = GeoPactBlockKVCache()
        flow_mode = kv_flow_mode_label(self.model)
        past_kv = gen_kwargs.get("past_key_values")
        past_len = 0
        if past_kv is not None:
            try:
                past_len = int(past_kv.get_seq_length())
            except Exception:
                past_len = 0
        draft = resolve_rope_for_step(
            attach_pinned_decode_contract(dict(gen_kwargs)),
            model_rope=getattr(target, "rope_deltas", None),
            flow_mode=flow_mode,
            past_kv_len=past_len,
        )
        if not should_use_pinned_generate(draft):
            return self.model.generate(**self._hf_safe_generate_kwargs(draft))
        validate_pinned_decode_contract(draft)
        if not contract_complete(draft):
            return self.model.generate(**self._hf_safe_generate_kwargs(draft))

        gen_kwargs = draft
        pinned_position_ids = gen_kwargs.pop("position_ids", None)
        pinned_cache_start = gen_kwargs.pop("_geopact_pinned_cache_start", None)
        rope_deltas = gen_kwargs.pop("_geopact_rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = gen_kwargs.pop("rope_deltas", None)
        else:
            gen_kwargs.pop("rope_deltas", None)
        if rope_deltas is None and pinned_position_ids is not None:
            rope_deltas = getattr(target, "rope_deltas", None)
        if rope_deltas is None or pinned_position_ids is None or pinned_cache_start is None:
            return self.model.generate(**self._hf_safe_generate_kwargs(gen_kwargs))

        original_rope = getattr(target, "rope_deltas", None)
        rope_deltas = rope_deltas.to(pinned_position_ids.device)
        try:
            target.rope_deltas = rope_deltas
            target._geopact_generate_position_ids = pinned_position_ids
            target._geopact_generate_rope_deltas = rope_deltas
            target._geopact_generate_cache_start = int(pinned_cache_start)
            return self.model.generate(**gen_kwargs)
        finally:
            target.rope_deltas = original_rope
            target._geopact_generate_position_ids = None
            target._geopact_generate_rope_deltas = None
            target._geopact_generate_cache_start = None

    def _session_lmms_kwargs(self, doc_to_messages_fn=None) -> dict:
        base = {"media_dir": os.environ.get("OMNIGUI_MEDIA_DIR", "data/benchmark")}
        task = getattr(doc_to_messages_fn, "__self__", None)
        task_kw = getattr(task, "lmms_eval_specific_kwargs", None)
        if isinstance(task_kw, dict) and task_kw:
            merged = dict(base)
            merged.update(task_kw)
            return merged
        return base

    def _prev_episode_doc(self, task_name: str, split_name: str, doc: Dict) -> Optional[Dict]:
        step = session_step_index(doc)
        if step <= 0:
            return None
        ep_key = session_episode_key(doc)
        want_step = step - 1
        split_docs = self.task_dict[task_name][split_name]
        rows = split_docs.values() if isinstance(split_docs, dict) else split_docs
        for row in rows:
            if not isinstance(row, dict):
                continue
            if session_episode_key(row) != ep_key:
                continue
            if session_step_index(row) == want_step:
                return row
        return None

    def _active_inputs_from_qwen_messages(
        self,
        qwen_messages: List[Dict],
        doc: Dict,
        *,
        add_generation_prompt: bool,
        skip_episode_context: bool = False,
    ):
        """Processor + optional GeoPACT compact / vision-cache prefill."""
        inner = self._geopact_target()
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if (
            not skip_episode_context
            and (
                self._use_visual_pruning()
                or vision_embed_cache_enabled()
                or session_kv_enabled()
            )
        ):
            if ledger is not None:
                ledger.start_bucket("episode_setup_ms")
            try:
                prepare_geopact_episode_context(
                    self.model, doc, qwen_messages, target=inner
                )
            finally:
                if ledger is not None:
                    ledger.stop_bucket("episode_setup_ms")
        if self._use_visual_pruning() and _should_cache_geopact_pil_image(
            str(self.visual_prune_policy or "")
        ):
            if ledger is not None:
                ledger.start_bucket("suffix_pil_cache_ms")
            try:
                cache_geopact_image(self.model, qwen_messages, target=inner)
            finally:
                if ledger is not None:
                    ledger.stop_bucket("suffix_pil_cache_ms")
            try:
                from lmms_eval.models.geopact_module_a_prefetch import (
                    early_start_module_a_prefetch,
                )

                # W-L: submit the current frame's detect now, so its runway is
                # template render + tokenize + ViT instead of ViT only. The
                # consume path re-validates image_id + real grid and falls
                # back to a fresh submit on mismatch.
                early_start_module_a_prefetch(self.model)
            except Exception:
                pass
        if ledger is not None:
            ledger.start_bucket("processor_ms")
        try:
            # Jinja chat-template render is processor work (~15ms on long
            # dialogues); keep it inside the bucket.
            texts = [
                self.processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            ]
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info([qwen_messages])
            else:
                image_inputs, video_inputs = None, None

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
        finally:
            if ledger is not None:
                ledger.stop_bucket("processor_ms")

        if add_generation_prompt:
            if ledger is not None:
                ledger.start_bucket("prompt_tail_ids_ms")
            try:
                inner._geopact_gen_prompt_tail_len = self._generation_prompt_tail_len(
                    qwen_messages
                )
            finally:
                if ledger is not None:
                    ledger.stop_bucket("prompt_tail_ids_ms")
        else:
            inner._geopact_gen_prompt_tail_len = 0

        active_inputs, _raw_inputs = prepare_active_prefill_inputs(
            self.model,
            inputs,
            use_pruning=self._use_visual_pruning(),
        )
        return active_inputs, inputs

    def _input_ids_from_qwen_messages(
        self,
        qwen_messages: List[Dict],
        *,
        add_generation_prompt: bool,
    ) -> List[int]:
        """Token ids for a prompt boundary without running the model vision encoder."""
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info([qwen_messages])
        else:
            image_inputs, video_inputs = None, None
        encoded = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return [int(x) for x in encoded["input_ids"][0].detach().cpu().tolist()]

    def _cached_generation_prompt_tail_len(self, qwen_messages: List[Dict]) -> int:
        """Reuse tail len from the latest processor encode when available."""
        inner = self._geopact_target()
        cached = getattr(inner, "_geopact_gen_prompt_tail_len", None)
        if cached is not None:
            tail = int(cached)
            if tail >= 0:
                return tail
        return self._generation_prompt_tail_len(qwen_messages)

    def _generation_prompt_tail_len(self, qwen_messages: List[Dict]) -> int:
        """Generation-prompt tail token length via text diff (no image re-preprocess).

        The chat-template generation prompt is a pure-text suffix appended after
        the rendered conversation, so ``with_gen == no_gen + tail_str`` and the
        tail tokenizes independently (template boundaries are full special
        tokens). We tokenize only the tail string and cache per tail text;
        any structural surprise falls back to the full double-processor path.
        """
        try:
            with_gen_text = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            no_gen_text = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=False
            )
            if with_gen_text.startswith(no_gen_text):
                tail_str = with_gen_text[len(no_gen_text):]
                if not tail_str:
                    return 0
                cache = getattr(self, "_geopact_gen_tail_len_cache", None)
                if cache is None:
                    cache = {}
                    self._geopact_gen_tail_len_cache = cache
                tail = cache.get(tail_str)
                if tail is None:
                    tail = len(
                        self.tokenizer(tail_str, add_special_tokens=False).input_ids
                    )
                    cache[tail_str] = tail
                return int(tail)
        except Exception:
            pass
        return self._generation_prompt_tail_len_full(qwen_messages)

    def _generation_prompt_tail_len_full(self, qwen_messages: List[Dict]) -> int:
        """Legacy processor-only delta (two full encodes); fallback/verify path."""
        with_gen = self._input_ids_from_qwen_messages(
            qwen_messages,
            add_generation_prompt=True,
        )
        no_gen = self._input_ids_from_qwen_messages(
            qwen_messages,
            add_generation_prompt=False,
        )
        tail = len(with_gen) - len(no_gen)
        if tail < 0:
            raise RuntimeError(
                "GeoPACT dialogue KV: generation tail length negative "
                f"with_gen={len(with_gen)} no_gen={len(no_gen)}"
            )
        return tail

    def _coord_injection_record(self, active_inputs: Dict):
        from lmms_eval.models.geopact_coord_injection.record import (
            CoordInjectionRecord,
            record_from_stats,
        )

        compact_len = 0
        if active_inputs.get("input_ids") is not None:
            compact_len = int(active_inputs["input_ids"].shape[1])

        stats = active_inputs.get("geopact_compact_stats")
        if isinstance(stats, dict) and stats:
            rec = record_from_stats(stats)
            if rec.spliced or rec.token_count > 0:
                return rec

        inner = self._geopact_target()
        last_stats = getattr(inner, "_geopact_last_compact_stats", None) or {}
        if isinstance(last_stats, dict) and last_stats:
            rec = record_from_stats(last_stats)
            if compact_len <= 0 or (
                rec.insert_pos < 0 or rec.insert_pos + rec.token_count <= compact_len
            ):
                if rec.spliced or rec.token_count > 0:
                    return rec

        rec = getattr(inner, "_geopact_last_coord_injection", None)
        if rec is not None and getattr(rec, "spliced", False):
            if compact_len <= 0 or (
                rec.insert_pos < 0 or rec.insert_pos + int(rec.token_count) <= compact_len
            ):
                return rec
        return CoordInjectionRecord.empty()

    def _compact_ids_at_user_boundary(
        self,
        active_inputs: Dict,
        qwen_messages: List[Dict],
    ) -> List[int]:
        """Compact token ids through the user turn (strip generation-prompt tail).

        Pruning removes image pads in the middle; the text-only generation tail
        length is unchanged vs the processor layout, so we can strip it from the
        compact ``input_ids`` without a second ViT encode.
        """
        from lmms_eval.models.geopact_coord_injection.lifecycle import strip_for_kv_commit

        if active_inputs.get("input_ids") is None:
            raise RuntimeError(
                "GeoPACT dialogue KV: compact cache boundary requires input_ids"
            )
        compact = [
            int(x) for x in active_inputs["input_ids"][0].detach().cpu().tolist()
        ]
        tail = self._cached_generation_prompt_tail_len(qwen_messages)
        record = self._coord_injection_record(active_inputs)
        return strip_for_kv_commit(
            compact,
            record=record,
            gen_tail_len=tail,
        )

    def _prepare_dialogue_suffix_inputs(
        self,
        doc: Dict,
        lmms_kwargs: dict,
        doc_to_messages_fn,
        *,
        full_messages: Optional[List[Dict]] = None,
    ):
        """Build incremental suffix: previous GT assistant + current user turn."""
        from lmms_eval.tasks.eval_message_registry import dialogue_incremental_suffix_messages

        inner = self._geopact_target()
        ledger = getattr(inner, "_geopact_step_timing_ledger", None)
        if ledger is not None:
            ledger.start_bucket("suffix_construct_ms")
        try:
            messages = dialogue_incremental_suffix_messages(
                doc,
                lmms_kwargs,
                doc_to_messages_fn,
                messages=full_messages,
            )
            qwen_messages = convert_messages_to_qwen(
                messages,
                min_pixels=self._explicit_min_pixels,
                max_pixels=self._explicit_max_pixels,
            )
            full_image_paths = extract_episode_image_paths_from_doc(doc, lmms_kwargs)
            set_geopact_episode_image_context(
                self.model, doc, full_image_paths, lmms_kwargs, target=inner
            )
            inner._geopact_dialogue_episode_image_paths = full_image_paths
            suffix_image_paths = list(full_image_paths[-1:]) if full_image_paths else []
            inner._geopact_image_paths = suffix_image_paths
        finally:
            if ledger is not None:
                ledger.stop_bucket("suffix_construct_ms")

        if (
            self._use_visual_pruning()
            and _should_cache_geopact_pil_image(str(self.visual_prune_policy or ""))
            and not self._skip_sliding_suffix_pil_cache()
        ):
            if ledger is not None:
                ledger.start_bucket("suffix_pil_cache_ms")
            try:
                cache_geopact_image(self.model, qwen_messages, target=inner)
            finally:
                if ledger is not None:
                    ledger.stop_bucket("suffix_pil_cache_ms")

        inner._geopact_dialogue_suffix_only_step = True
        active_inputs, _ = self._active_inputs_from_qwen_messages(
            qwen_messages,
            doc,
            add_generation_prompt=True,
            skip_episode_context=True,
        )
        inner._geopact_image_paths = full_image_paths
        if not isinstance(active_inputs, dict):
            active_inputs = dict(active_inputs.items())
        if ledger is not None:
            ledger.start_bucket("kv_commit_ids_ms")
        try:
            suffix_no_gen_ids = self._compact_ids_at_user_boundary(
                active_inputs,
                qwen_messages,
            )
        finally:
            if ledger is not None:
                ledger.stop_bucket("kv_commit_ids_ms")
        inner._geopact_dialogue_suffix_no_gen_input_ids = suffix_no_gen_ids
        try:
            from lmms_eval.models.geopact_bilateral_trace import trace_enabled, trace_suffix_inputs

            if trace_enabled():
                suffix_len = int(active_inputs["inputs_embeds"].shape[1])
                trace_suffix_inputs(
                    full_image_paths=full_image_paths,
                    suffix_image_paths=suffix_image_paths,
                    suffix_no_gen_ids=suffix_no_gen_ids,
                    suffix_len=suffix_len,
                )
        except Exception:
            pass
        if active_inputs.get("inputs_embeds") is None:
            raise RuntimeError(
                "GeoPACT dialogue KV: suffix path requires inputs_embeds "
                "(dialogue embed prefill failed; check pixel_values / image_grid_thw)"
            )
        return active_inputs

    @staticmethod
    def _norm_media_path(path: Any) -> str:
        p = str(path or "")
        if p.startswith("file://"):
            p = p[len("file://") :]
        return os.path.abspath(p) if p else ""

    def _dialogue_frame_token_counts(self) -> Optional[Dict[str, int]]:
        """Per-frame unpruned LLM token counts from dialogue-state grids."""
        inner = self._geopact_target()
        state = getattr(inner, "_geopact_dialogue_kv_state", None) or {}
        paths = list(state.get("image_paths") or [])
        grid = state.get("image_grid_thw")
        if not paths or grid is None or int(grid.shape[0]) != len(paths):
            return None
        merge = int(getattr(self.processor.image_processor, "merge_size", 2) or 2)
        counts: Dict[str, int] = {}
        for path, row in zip(paths, grid):
            t, h, w = (int(x) for x in row.tolist())
            n = (t * h * w) // (merge * merge)
            if n <= 0:
                return None
            counts[self._norm_media_path(path)] = n
        return counts

    def _cheap_prompt_ids(
        self,
        qwen_messages: List[Dict],
        frame_counts: Dict[str, int],
        *,
        placeholder_tokens: int = 4,
    ) -> Optional[List[int]]:
        """Prompt token ids without pixel processing.

        Chat-template text is tokenised directly (one image token per frame),
        then each image token is expanded to the frame's known LLM token count.
        Frames absent from ``frame_counts`` (the new current frame) get a fixed
        placeholder run — it appears identically in suffix and full renders, so
        the roll-boundary arithmetic and tail-equality proof are unaffected.
        """
        inner = self._geopact_target()
        image_token_id = int(getattr(inner.config, "image_token_id", -1) or -1)
        if image_token_id < 0:
            return None
        image_paths = [
            item.get("image")
            for msg in qwen_messages
            for item in (msg.get("content") or [])
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = [int(x) for x in self.tokenizer(text).input_ids]
        out: List[int] = []
        img_i = 0
        for tok in ids:
            if tok == image_token_id:
                if img_i >= len(image_paths):
                    return None
                n = frame_counts.get(
                    self._norm_media_path(image_paths[img_i]),
                    placeholder_tokens,
                )
                out.extend([tok] * n)
                img_i += 1
            else:
                out.append(tok)
        if img_i != len(image_paths):
            return None
        return self._compact_ids_from_token_ids(out, qwen_messages=qwen_messages)

    def _compute_roll_drop_token_len(
        self,
        doc: Dict,
        lmms_kwargs: dict,
        doc_to_messages_fn,
        *,
        full_messages: Optional[List[Dict]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Anchored eviction boundary ``(anchor_len, drop_len)`` so that
        ``cached[:anchor] + cached[anchor+drop:] + suffix_ids == full prompt ids``."""
        from lmms_eval.models.geopact_session_kv import compute_dialogue_roll_drop_len
        from lmms_eval.tasks.eval_message_registry import (
            dialogue_incremental_suffix_messages,
            invoke_doc_to_messages,
        )

        inner = self._geopact_target()
        state = getattr(inner, "_geopact_dialogue_kv_state", None) or {}
        cached = list(state.get("cached_input_ids") or [])
        if not cached:
            return None

        if full_messages is None:
            full_messages = invoke_doc_to_messages(doc_to_messages_fn, doc, lmms_kwargs)
        suffix_messages = dialogue_incremental_suffix_messages(
            doc, lmms_kwargs, doc_to_messages_fn, messages=full_messages
        )
        suffix_qwen = convert_messages_to_qwen(
            suffix_messages,
            min_pixels=self._explicit_min_pixels,
            max_pixels=self._explicit_max_pixels,
        )
        full_qwen = convert_messages_to_qwen(
            full_messages,
            min_pixels=self._explicit_min_pixels,
            max_pixels=self._explicit_max_pixels,
        )

        # Fast path: expand image runs from dialogue-state grids instead of
        # running the full processor (pixel preprocessing) on every window
        # image just to obtain token ids (~2s/roll otherwise, unbucketed).
        full_ids: Optional[List[int]] = None
        suffix_ids: Optional[List[int]] = None
        if os.environ.get("GEOPACT_CHEAP_ROLL_IDS", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }:
            frame_counts = self._dialogue_frame_token_counts()
            if frame_counts is not None:
                full_ids = self._cheap_prompt_ids(full_qwen, frame_counts)
                suffix_ids = self._cheap_prompt_ids(suffix_qwen, frame_counts)
        if full_ids is None or suffix_ids is None:
            full_ids = self._input_ids_from_qwen_messages(
                full_qwen,
                add_generation_prompt=False,
            )
            full_ids = self._compact_ids_from_token_ids(full_ids, qwen_messages=full_qwen)
            suffix_ids = self._input_ids_from_qwen_messages(
                suffix_qwen,
                add_generation_prompt=False,
            )
            suffix_ids = self._compact_ids_from_token_ids(
                suffix_ids, qwen_messages=suffix_qwen
            )

        image_token_id = int(getattr(inner.config, "image_token_id", -1) or -1)
        try:
            im_start_id = int(self.tokenizer.convert_tokens_to_ids("<|im_start|>"))
        except Exception:
            im_start_id = -1
        return compute_dialogue_roll_drop_len(
            state,
            cached_ids=cached,
            suffix_ids=suffix_ids,
            full_ids=full_ids,
            image_token_id=image_token_id,
            im_start_id=im_start_id,
        )

    def _compact_ids_from_token_ids(
        self,
        token_ids: List[int],
        *,
        qwen_messages: List[Dict],
    ) -> List[int]:
        """Strip generation tail from processor token ids (no second processor pass)."""
        from lmms_eval.models.geopact_coord_injection.lifecycle import strip_for_kv_commit

        tail = self._cached_generation_prompt_tail_len(qwen_messages)
        return strip_for_kv_commit(
            list(token_ids),
            record=None,
            gen_tail_len=tail,
        )

    def _dialogue_cumulative_image_grid(
        self,
        step_idx: int,
        active_dict: dict,
        suffix_dict: Optional[dict],
    ) -> Optional["torch.Tensor"]:
        """Episode image_grid_thw through the current user turn (for dialogue KV MRoPE)."""
        inner = self._geopact_target()
        prev_state = getattr(inner, "_geopact_dialogue_kv_state", None) or {}
        prev_grid = prev_state.get("image_grid_thw")
        suffix_grid = (suffix_dict or {}).get("image_grid_thw")
        # Incremental suffix path only when we have both cached prefix grid and suffix grid.
        if step_idx > 0 and suffix_dict is not None and prev_grid is not None:
            if suffix_grid is None:
                return prev_grid if isinstance(prev_grid, torch.Tensor) else torch.tensor(prev_grid)
            if not isinstance(prev_grid, torch.Tensor):
                prev_grid = torch.tensor(prev_grid)
            return torch.cat([prev_grid.to(device=suffix_grid.device), suffix_grid], dim=0)

        backup = getattr(inner, "_geopact_session_full_prefill_backup", None)
        if backup and backup.get("image_grid_thw") is not None:
            return backup["image_grid_thw"]
        return active_dict.get("image_grid_thw")

    def _session_request_sort_key(self, args) -> tuple:
        _ctx, _doc_fn, _gen_kw, doc_id, task_name, split_name = args
        doc = self.task_dict[task_name][split_name][doc_id]
        return (session_episode_key(doc), session_step_index(doc))

    def _needs_episode_order(self) -> bool:
        if _force_episode_order_enabled():
            return True
        if session_kv_enabled():
            return True
        if vision_embed_cache_enabled() or _native_multiturn_eval_enabled():
            return True
        return scope_requires_episode_order(str(self.visual_prune_scope or "image"))

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # Session-KV / episode-ordered orchestration only under the same env
        # gates qwen3_vl_gui uses; default single-step path stays untouched.
        if not self._needs_episode_order():
            return self._generate_until_single(requests)
        if session_kv_enabled() and not self._is_qwen25_carrier():
            raise RuntimeError(
                "GEOPACT_SESSION_KV on ui_tars requires the qwen2_5_vl carrier "
                "(UI-TARS-1.5-7B); the qwen2_vl patch has no write-once serving "
                f"path (model_type={getattr(self._config, 'model_type', '?')})"
            )
        return self._generate_until_geopact(requests)

    def _generate_until_single(self, requests: List[Instance]) -> List[str]:
        """Original UI-TARS single-step path (no GeoPACT session orchestration)."""
        res = []
        debug_logged = False

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="UI-TARS")
        re_ords = utils.Collator(
            [r.args for r in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            assert_messages_request(chunk[0], "UITARS")
            step_t0 = time.perf_counter()

            contexts, doc_to_messages_fns, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [u for u in until if u != "\n\n"]

            qwen_messages_batch = []
            for i, _ctx in enumerate(contexts):
                doc = self.task_dict[task_name][split_name][doc_id[i]]
                messages = doc_to_messages_fns[i](doc)
                qwen_messages = convert_messages_to_qwen(
                    messages,
                    min_pixels=self._explicit_min_pixels,
                    max_pixels=self._explicit_max_pixels,
                )
                qwen_messages_batch.append(qwen_messages)
                if self._use_visual_pruning():
                    cache_geopact_image(self.model, qwen_messages, target=self._geopact_target())
                if not debug_logged and self.rank == 0:
                    log_messages_debug("UITARS", messages, qwen_messages)
                    debug_logged = True

            texts = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in qwen_messages_batch
            ]
            if process_vision_info is not None:
                image_inputs, video_inputs = process_vision_info(qwen_messages_batch)
            else:
                image_inputs, video_inputs = None, None

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
            if contexts:
                sample_doc = self.task_dict[task_name][split_name][doc_id[0]]
                log_vision_preprocess_debug(
                    "UITARS",
                    sample_doc,
                    inputs,
                    explicit_min_pixels=self._explicit_min_pixels,
                    explicit_max_pixels=self._explicit_max_pixels,
                )
            preprocess_ms = (time.perf_counter() - step_t0) * 1000.0

            prompt_len = inputs.input_ids.shape[-1]
            temperature = gen_kwargs.get("temperature", 0)
            do_sample = temperature > 0
            max_new_toks = gen_kwargs.get("max_new_tokens", 4096)

            peak_mem_mb = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            generate_t0 = time.perf_counter()
            with torch.inference_mode():
                cont, gen_timing = call_qwen_vl_generate_with_timing(
                    self.model,
                    prompt_len,
                    self.model.generate,
                    tokenizer=self.tokenizer,
                    early_stop_on_action=self.early_stop_on_action,
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_new_toks,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=gen_kwargs.get("top_p", None) if do_sample else None,
                    use_cache=self.use_cache,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if os.environ.get("UITARS_DEBUG_RAW_DECODE", "").strip() == "1":
                for _i, _a in enumerate(answers):
                    if not (_a or "").strip():
                        _raw = self.processor.batch_decode(
                            [generated_ids[_i]], skip_special_tokens=False
                        )[0]
                        eval_logger.warning(
                            "UITARS empty answer: "
                            f"n_tokens={len(generated_ids[_i])} "
                            f"raw_head={_raw[:300]!r} raw_tail={_raw[-150:]!r} "
                            f"ids_head={generated_ids[_i][:24].tolist()}"
                        )

            for i, (ans, ctx) in enumerate(zip(answers, contexts)):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                ans = _uitars_repair_action_text(ans)
                ans = _uitars_unwrap_tool_call_json(ans)
                ans = _uitars_dsl_to_omni_json(ans)
                if _uitars_coord_rescale_enabled():
                    ans = _uitars_rescale_coords_to_1000(
                        ans, getattr(inputs, "image_grid_thw", None)
                    )
                ans, raw_original = canonicalize_omni_action_output(ans)
                ans_out = ans
                if self.attach_usage_stats:
                    n_out = len(generated_ids[i]) if i < len(generated_ids) else 0
                    per_prompt_len = int(
                        inputs.attention_mask[i].sum().item()
                        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None
                        else prompt_len
                    )
                    usage = build_qwen_vl_usage_stats(
                        step_t0=step_t0,
                        preprocess_ms=preprocess_ms,
                        generate_t0=generate_t0,
                        gen_timing=gen_timing,
                        prompt_len=per_prompt_len,
                        output_tokens=n_out,
                        peak_mem_mb=peak_mem_mb,
                    )
                    if raw_original:
                        usage["raw_model_output"] = raw_original
                    ans_out = format_response_with_usage(ans, usage)
                res.append(ans_out)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans_out)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def _generate_until_geopact(self, requests: List[Instance]) -> List[str]:
        """Episode-ordered GeoPACT path (session-KV / sticky pruning / vision cache).

        Ported from qwen3_vl_gui.generate_until; model-object plumbing adapted:
        wherever the qwen3 wrapper descends to ``self.model.model``, this uses
        ``self._geopact_target()`` (the FCG for qwen2_5_vl). No deepstack.
        """
        self._ensure_module_a_prewarm_done()
        res = []
        debug_logged = False

        def _collate(x):
            if self._needs_episode_order():
                return self._session_request_sort_key(x)
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="UI-TARS")
        # Messages-task args: (ctx, doc_to_messages, gen_kwargs, doc_id, task, split)
        # Session KV / image_sticky: episode-ordered steps (no length-based shuffle).
        re_ords = utils.Collator(
            [r.args for r in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=not self._needs_episode_order(),
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            assert_messages_request(chunk[0], "UITARS")
            _anatomy_token = None
            if os.environ.get("GEOPACT_ANATOMY"):
                from lmms_eval.models.geopact_anatomy import anatomy_step_begin

                _anatomy_token = anatomy_step_begin()
            step_t0 = time.perf_counter()
            _step_ledger0 = begin_step_timing(self.model, step_t0=step_t0)
            inner = self._geopact_target()
            # step_timing stores the ledger on ``model.model``; session_kv /
            # vl_prepare read it off the carrier target (the FCG for
            # qwen2_5_vl). Mirror the same object so every layer sees one ledger.
            inner._geopact_step_timing_ledger = _step_ledger0
            # Doc fetch (arrow row decode) + session kwargs is setup work.
            _step_ledger0.start_bucket("episode_setup_ms")

            contexts, doc_to_messages_fns, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [u for u in until if u != "\n\n"]

            sample_doc = self.task_dict[task_name][split_name][doc_id[0]]
            doc_to_messages_fn = doc_to_messages_fns[0]
            if not session_kv_enabled():
                clear_geopact_mrope_decode_state(self.model)
            step_idx = session_step_index(sample_doc) if session_kv_enabled() else 0
            session_kwargs = self._session_lmms_kwargs(doc_to_messages_fn)
            prev_doc = self._prev_episode_doc(task_name, split_name, sample_doc)
            rolling_window_reset = False
            kv_slid_at_roll = False
            dialogue_suffix_allowed = False
            cached_full_messages = None
            from lmms_eval.tasks.history_window import task_supports_session_kv

            if (
                dialogue_kv_enabled()
                and session_kv_enabled()
                and step_idx > 0
                # Tasks that opt out via supports_session_kv:false (e.g. the
                # low-level AndroidControl per-step protocol) must never enter
                # the incremental-suffix path: their message tails do not end
                # with [assistant, user], so full prefill every step.
                and task_supports_session_kv(session_kwargs)
            ):
                from lmms_eval.tasks.eval_message_registry import (
                    dialogue_incremental_suffix_allowed,
                    invoke_doc_to_messages,
                )
                from lmms_eval.tasks.history_window import rolling_window_changed

                _suffix_setup_ledger = get_step_ledger(self.model)
                _suffix_setup_ledger.start_bucket("episode_setup_ms")
                cached_full_messages = invoke_doc_to_messages(
                    doc_to_messages_fn, sample_doc, session_kwargs
                )
                _suffix_setup_ledger.stop_bucket("episode_setup_ms")

                if rolling_window_changed(prev_doc, sample_doc):
                    slide_fail_reason = "slide_disabled"
                    if slide_kv_on_roll_enabled():
                        _roll_ledger = get_step_ledger(self.model)
                        _roll_ledger.start_bucket("kv_slide_ms")
                        _drop_proof_t0 = time.perf_counter()
                        try:
                            roll_boundary = self._compute_roll_drop_token_len(
                                sample_doc,
                                session_kwargs,
                                doc_to_messages_fn,
                                full_messages=cached_full_messages,
                            )
                        finally:
                            _roll_ledger.set_audit(
                                slide_drop_proof_ms=round(
                                    (time.perf_counter() - _drop_proof_t0) * 1000.0, 2
                                )
                            )
                            _roll_ledger.stop_bucket("kv_slide_ms")
                        if roll_boundary is None:
                            slide_fail_reason = "drop_len_unproven"
                        elif roll_boundary[1] <= 0:
                            # Proven pure append (cached + suffix == full): the
                            # window rolled but no cached tokens need eviction.
                            kv_slid_at_roll = True
                            if self.rank == 0:
                                eval_logger.info(
                                    "GeoPACT dialogue KV: rolling window pure append "
                                    "(no eviction needed)"
                                )
                        else:
                            anchor_len, drop_len = roll_boundary
                            _slide_ledger = get_step_ledger(self.model)
                            _slide_ledger.start_bucket("kv_slide_ms")
                            try:
                                slide_stats = slide_dialogue_kv_prefix(
                                    self.model, drop_len, anchor_len
                                )
                            finally:
                                _slide_ledger.stop_bucket("kv_slide_ms")
                            if slide_stats.get("dialogue_kv_slide_ok"):
                                kv_slid_at_roll = True
                                get_step_ledger(self.model).set_audit(
                                    dialogue_kv_slide_anchor_len=int(anchor_len),
                                    dialogue_kv_slide_drop_tokens=int(drop_len),
                                )
                                if self.rank == 0:
                                    eval_logger.info(
                                        "GeoPACT dialogue KV: anchored eviction roll "
                                        f"anchor={anchor_len} drop_tokens={drop_len} "
                                        f"pos_offset={slide_stats.get('dialogue_kv_slide_pos_offset')} "
                                        f"new_len={slide_stats.get('dialogue_kv_slide_new_len')}"
                                    )
                            else:
                                slide_fail_reason = str(
                                    slide_stats.get("dialogue_kv_slide_reason", "slide_failed")
                                )
                    if not kv_slid_at_roll:
                        # No third state on roll: either a proven KV slide or an
                        # explicit full reset (slow but correct).
                        rolling_window_reset = True
                        get_step_ledger(self.model).set_audit(
                            dialogue_kv_slide_fallback_reset=1,
                            dialogue_kv_slide_fallback_reason=slide_fail_reason,
                        )
                        if self.rank == 0:
                            eval_logger.warning(
                                "GeoPACT dialogue KV: rolling window reset "
                                "(dialogue_kv_slide_fallback_reset) at "
                                f"episode={session_episode_key(sample_doc)} step={step_idx} "
                                f"reason={slide_fail_reason}"
                            )
                        clear_session_kv(self.model)

                _suffix_setup_ledger.start_bucket("episode_setup_ms")
                try:
                    dialogue_suffix_allowed = dialogue_incremental_suffix_allowed(
                        sample_doc,
                        session_kwargs,
                        doc_to_messages_fn,
                        prev_doc=prev_doc,
                        kv_slid_at_roll=kv_slid_at_roll,
                        messages=cached_full_messages,
                    )
                finally:
                    _suffix_setup_ledger.stop_bucket("episode_setup_ms")
                if not dialogue_suffix_allowed:
                    if self.rank == 0:
                        eval_logger.info(
                            "GeoPACT dialogue KV: segment reset at "
                            f"episode={session_episode_key(sample_doc)} step={step_idx} "
                            "(full prefill, cleared stale KV)"
                        )
                    clear_session_kv(self.model)
                elif kv_flow_mode_label(self.model) == "prune_rewrite":
                    # Sliding-window rewrite: Vision ABC + KV truncate; merged LM
                    # forward happens in prepare_dialogue_generate_payload.
                    from lmms_eval.models.geopact_session_kv import (
                        sliding_rewrite_prepare_dialogue_kv,
                    )

                    rewrite_stats = sliding_rewrite_prepare_dialogue_kv(
                        self.model,
                        sample_doc,
                        ratio=float(self.visual_prune_ratio or 0.0),
                        policy=str(self.visual_prune_policy or ""),
                    )
                    reason = rewrite_stats.get("sliding_rewrite_reason", "")
                    if self.rank == 0 and reason not in (
                        "not_history_target",
                        "disabled",
                    ):
                        eval_logger.info(f"GeoPACT sliding-rewrite: {rewrite_stats}")
                    rewrite_ok = bool(rewrite_stats.get("sliding_rewrite_ok"))
                    benign_reasons = frozenset(
                        {
                            "not_history_target",
                            "disabled",
                            "first_step",
                            "already_compressed",
                            "ratio_noop",
                        }
                    )
                    if not rewrite_ok and reason not in benign_reasons:
                        dialogue_suffix_allowed = False
            dialogue_suffix_only = (
                dialogue_kv_enabled()
                and session_kv_enabled()
                and step_idx > 0
                and dialogue_suffix_allowed
            )
            if dialogue_suffix_only and len(contexts) != 1:
                raise RuntimeError(
                    "GeoPACT dialogue KV requires batch_size=1 when step_index > 0"
                )

            suffix_dict = None
            batched_messages = []
            qwen_messages_batch = []
            if not dialogue_suffix_only:
                for i, _ctx in enumerate(contexts):
                    doc = self.task_dict[task_name][split_name][doc_id[i]]
                    messages = doc_to_messages_fns[i](doc)
                    batched_messages.append(messages)
                    qwen_messages = convert_messages_to_qwen(
                        messages,
                        min_pixels=self._explicit_min_pixels,
                        max_pixels=self._explicit_max_pixels,
                    )
                    qwen_messages_batch.append(qwen_messages)
                    if (
                        self._use_visual_pruning()
                        or vision_embed_cache_enabled()
                        or session_kv_enabled()
                    ):
                        _setup_ledger = get_step_ledger(self.model)
                        _setup_ledger.start_bucket("episode_setup_ms")
                        try:
                            prepare_geopact_episode_context(
                                self.model, doc, qwen_messages, session_kwargs,
                                target=inner,
                            )
                            if self._use_visual_pruning() and _should_cache_geopact_pil_image(
                                str(self.visual_prune_policy or "")
                            ):
                                cache_geopact_image(self.model, qwen_messages, target=inner)
                        finally:
                            _setup_ledger.stop_bucket("episode_setup_ms")
                    if not debug_logged and self.rank == 0:
                        log_messages_debug("UITARS", messages, qwen_messages)
                        debug_logged = True

            compact_stats = None
            inputs = None
            processor_prompt_len = 0
            if dialogue_suffix_only:
                suffix_dict = self._prepare_dialogue_suffix_inputs(
                    sample_doc,
                    session_kwargs,
                    doc_to_messages_fn,
                    full_messages=cached_full_messages,
                )
                active_inputs = suffix_dict
                compact_stats = getattr(inner, "_geopact_last_compact_stats", None)
                if self.rank == 0 and compact_stats:
                    if compact_stats.get("compact_backend") == "dialogue_embed":
                        eval_logger.info(f"GeoPACT dialogue-embed (suffix): {compact_stats}")
                    elif compact_stats.get("compact_backend") == "dialogue_prune_sticky":
                        eval_logger.info(f"GeoPACT dialogue-prune-sticky (suffix): {compact_stats}")
                    elif compact_stats.get("vision_cache_enabled"):
                        eval_logger.info(f"GeoPACT baseline-vision-cache (suffix): {compact_stats}")
                        eval_logger.info(
                            "GeoPACT vision-cache: "
                            f"hits={compact_stats.get('vision_cache_hits', 0)} "
                            f"misses={compact_stats.get('vision_cache_misses', 0)} "
                            f"encode_ms={compact_stats.get('vision_encode_ms', 0):.1f}"
                        )
                _sfx_ledger = get_step_ledger(self.model)
                # .item() blocks on the async suffix embed prefill (vision
                # encode + scatter); attribute that host wait to suffix work.
                _sfx_ledger.start_bucket("suffix_construct_ms")
                try:
                    processor_prompt_len = int(suffix_dict["attention_mask"][0].sum().item())
                finally:
                    _sfx_ledger.stop_bucket("suffix_construct_ms")
            else:
                ledger = get_step_ledger(self.model)
                if ledger is not None:
                    ledger.start_bucket("processor_ms")
                try:
                    texts = [
                        self.processor.apply_chat_template(
                            m, tokenize=False, add_generation_prompt=True
                        )
                        for m in qwen_messages_batch
                    ]
                    if process_vision_info is not None:
                        image_inputs, video_inputs = process_vision_info(qwen_messages_batch)
                    else:
                        image_inputs, video_inputs = None, None

                    inputs = self.processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    if self._use_visual_pruning():
                        inner._geopact_coordinate_space = "normalized_0_1000"
                        inner._geopact_current_qwen_geometry = {}
                    inputs = inputs.to("cuda" if self.device_map == "auto" else self.device)
                    if contexts:
                        log_vision_preprocess_debug(
                            "UITARS",
                            sample_doc,
                            inputs,
                            explicit_min_pixels=self._explicit_min_pixels,
                            explicit_max_pixels=self._explicit_max_pixels,
                        )
                finally:
                    if ledger is not None:
                        ledger.stop_bucket("processor_ms")
                if isinstance(inputs, dict) and inputs.get("attention_mask") is not None:
                    processor_prompt_len = int(inputs["attention_mask"][0].sum().item())
                elif getattr(inputs, "attention_mask", None) is not None:
                    processor_prompt_len = int(inputs.attention_mask[0].sum().item())
                else:
                    processor_prompt_len = 0

                active_inputs, _raw_inputs = prepare_active_prefill_inputs(
                    self.model,
                    inputs,
                    use_pruning=self._use_visual_pruning(),
                )
                compact_stats = getattr(inner, "_geopact_last_compact_stats", None)
                if self.rank == 0 and compact_stats:
                    backend = compact_stats.get("compact_backend", "")
                    if backend == "source":
                        eval_logger.info(f"GeoPACT source-compact (prune): {compact_stats}")
                    elif backend == "dialogue_embed":
                        eval_logger.info(f"GeoPACT dialogue-embed: {compact_stats}")
                    elif backend == "dialogue_prune_sticky":
                        eval_logger.info(f"GeoPACT dialogue-prune-sticky: {compact_stats}")
                    elif backend == "baseline_vision_cache":
                        eval_logger.info(f"GeoPACT baseline-vision-cache: {compact_stats}")
                    if compact_stats.get("vision_cache_enabled"):
                        eval_logger.info(
                            "GeoPACT vision-cache: "
                            f"hits={compact_stats.get('vision_cache_hits', 0)} "
                            f"misses={compact_stats.get('vision_cache_misses', 0)} "
                            f"encode_ms={compact_stats.get('vision_encode_ms', 0):.1f}"
                        )
                if dialogue_kv_enabled():
                    inner._geopact_dialogue_episode_image_paths = list(
                        getattr(inner, "_geopact_image_paths", None) or []
                    )

            session_stats: dict = {}
            session_gen_payload = None
            dialogue_cache_stats: dict = {}
            dialogue_kv_capture = None
            dialogue_prompt_ids: Optional[List[int]] = None
            dialogue_cache_prompt_ids: Optional[List[int]] = None
            active_dict = (
                active_inputs
                if isinstance(active_inputs, dict)
                else dict(inputs.items())
            )
            # Payload region starts here: session-kv backup, prefix id assembly
            # and the generate payload build below are all payload work.
            get_step_ledger(self.model).start_bucket("generate_payload_ms")
            if session_kv_enabled():
                if not dialogue_kv_enabled():
                    raise RuntimeError(
                        "GeoPACT_SESSION_KV requires GEOPACT_SESSION_KV_MODE=dialogue"
                    )
                backup_session_full_prefill(
                    self.model,
                    metadata_only=bool(dialogue_suffix_only),
                )
                session_stats = {"session_kv_enabled": True, "session_kv_hit": False}

                if step_idx > 0 and dialogue_suffix_allowed:
                    if suffix_dict is None:
                        raise RuntimeError(
                            "GeoPACT dialogue KV: suffix inputs missing for step>0"
                        )
                    dlg_state = getattr(inner, "_geopact_dialogue_kv_state", None) or {}
                    sliding_pending = dlg_state.get("_sliding_rewrite_pending")
                    if sliding_pending and sliding_pending.get("ids_after_rewrite"):
                        prefix_ids = list(sliding_pending["ids_after_rewrite"])
                        past_len_stored = len(prefix_ids)
                    else:
                        prefix_ids = list(dlg_state.get("cached_input_ids") or [])
                        past_len_stored = int(
                            dlg_state.get("prompt_token_len") or len(prefix_ids)
                        )
                    suffix_no_gen_ids = list(
                        getattr(
                            inner,
                            "_geopact_dialogue_suffix_no_gen_input_ids",
                            None,
                        )
                        or []
                    )
                    if not suffix_no_gen_ids:
                        raise RuntimeError(
                            "GeoPACT dialogue KV: missing suffix no-generation ids"
                        )
                    dialogue_cache_prompt_ids = prefix_ids + suffix_no_gen_ids
                    if not sliding_pending and len(prefix_ids) != past_len_stored:
                        raise RuntimeError(
                            "GeoPACT dialogue KV: prefix cached_input_ids length "
                            f"{len(prefix_ids)} != stored prompt_token_len "
                            f"{past_len_stored}"
                        )
                    if len(dialogue_cache_prompt_ids) != past_len_stored + len(
                        suffix_no_gen_ids
                    ):
                        raise RuntimeError(
                            "GeoPACT dialogue KV: cache prompt id length mismatch "
                            f"prefix={len(prefix_ids)} suffix_no_gen="
                            f"{len(suffix_no_gen_ids)} total="
                            f"{len(dialogue_cache_prompt_ids)} expected="
                            f"{past_len_stored + len(suffix_no_gen_ids)}"
                        )
                    dialogue_capture_len = len(dialogue_cache_prompt_ids)
                    dialogue_kv_capture = DialogueKVCapture(
                        self.model, dialogue_capture_len
                    )
                    session_gen_payload, session_stats = prepare_dialogue_generate_payload(
                        self.model,
                        sample_doc,
                        suffix_dict,
                    )
                    # Payload-assembly tail (capture install, suffix id D2H,
                    # logging) counts as payload work, not misc.
                    get_step_ledger(self.model).start_bucket("generate_payload_ms")
                    if session_gen_payload is None:
                        raise RuntimeError(
                            "GeoPACT dialogue KV cache miss at "
                            f"episode={sample_doc.get('episode_id')} "
                            f"step={step_idx}: {session_stats}"
                        )
                    dialogue_kv_capture.install()
                    suffix_ids = [
                        int(x) for x in suffix_dict["input_ids"][0].detach().cpu().tolist()
                    ]
                    dialogue_prompt_ids = prefix_ids + suffix_ids
                else:
                    if rolling_window_reset:
                        session_stats["dialogue_kv_reason"] = "rolling_window_reset"
                    elif step_idx > 0 and not dialogue_suffix_allowed:
                        session_stats["dialogue_kv_reason"] = "segment_reset"
                    elif (
                        not dialogue_suffix_only
                        and _dialogue_kv_use_native_full_prefill(self.model)
                        and isinstance(active_dict, dict)
                        and active_dict.get("inputs_embeds") is None
                    ):
                        session_stats["dialogue_kv_reason"] = "native_first_step"
                    else:
                        session_stats["dialogue_kv_reason"] = "first_step"

                if self.rank == 0:
                    eval_logger.info(f"GeoPACT session-kv: {session_stats}")

                if session_gen_payload is None and getattr(
                    inner, "_geopact_session_full_prefill_backup", None
                ) is not None:
                    inner._geopact_prefill_payload = inner._geopact_session_full_prefill_backup

            prompt_len = int(active_dict["input_ids"].shape[-1])
            if (
                dialogue_kv_enabled()
                and session_kv_enabled()
                and (step_idx == 0 or not dialogue_suffix_allowed)
                and qwen_messages_batch
            ):
                dialogue_cache_prompt_ids = self._compact_ids_at_user_boundary(
                    active_dict,
                    qwen_messages_batch[0],
                )
                # Stage the current-frame capture (and its Module A prefetch)
                # before generate so detection overlaps decode; otherwise the
                # finalize-time capture starts the detect after decode and the
                # next step's rectify blocks on the join (~300-600ms).
                if dialogue_cache_prompt_ids and os.environ.get(
                    "GEOPACT_PRESTAGE_STEP0_CAPTURE", "1"
                ).strip().lower() not in {"0", "false", "off", "no"}:
                    try:
                        stage_pending_last_full_frame(
                            self.model,
                            stored_ids=dialogue_cache_prompt_ids,
                            image_paths=list(
                                getattr(
                                    inner,
                                    "_geopact_dialogue_episode_image_paths",
                                    None,
                                )
                                or []
                            ),
                        )
                    except Exception:
                        pass
            logical_prompt_len = (
                len(dialogue_prompt_ids)
                if dialogue_prompt_ids is not None
                else prompt_len
            )
            dialogue_capture_len = (
                len(dialogue_cache_prompt_ids)
                if dialogue_cache_prompt_ids is not None
                else logical_prompt_len
            )
            native_kv_full_prefill = (
                dialogue_kv_enabled()
                and session_kv_enabled()
                and not dialogue_suffix_only
                and _dialogue_kv_use_native_full_prefill(self.model)
                and isinstance(active_dict, dict)
                and active_dict.get("inputs_embeds") is None
            )
            if dialogue_kv_enabled() and session_kv_enabled():
                if isinstance(active_dict, dict) and active_dict.get("inputs_embeds") is None:
                    if not native_kv_full_prefill:
                        raise RuntimeError(
                            "GeoPACT dialogue KV requires scattered inputs_embeds on the active prompt"
                        )
                if dialogue_kv_capture is None:
                    dialogue_kv_capture = install_dialogue_kv_capture(
                        self.model, dialogue_capture_len
                    )
            temperature = gen_kwargs.get("temperature", 0)
            do_sample = temperature > 0
            max_new_toks = self._cap_gui_max_new_tokens(
                gen_kwargs.get("max_new_tokens", 4096),
                early_stop_on_action=self.early_stop_on_action,
            )

            peak_mem_mb = 0.0
            avg_mem_mb = 0.0
            mem_t0 = 0
            step_ledger = get_step_ledger(self.model)
            step_ledger.stop_bucket()
            step_ledger.start_bucket("generate_payload_ms")
            if torch.cuda.is_available() and (cuda_sync_enabled() or mem_audit_enabled()):
                if cuda_sync_enabled():
                    torch.cuda.synchronize()
                # allocator counters are host-side: safe without a sync
                torch.cuda.reset_peak_memory_stats()
                mem_t0 = torch.cuda.memory_allocated()
            _kv_probe = install_kv_bytes_probe(self.model)
            processor_ms = float(step_ledger.snapshot_totals().get("processor_ms", 0.0)) + float(
                step_ledger.snapshot_totals().get("suffix_construct_ms", 0.0)
            )
            raw_gen_payload: dict = {}
            if dialogue_suffix_only:
                if session_gen_payload is None:
                    raise RuntimeError(
                        "GeoPACT dialogue KV: step>0 must generate via suffix cache payload"
                    )
                raw_gen_payload = dict(session_gen_payload)
                decode_ref_ids = session_gen_payload.get("input_ids")
                if decode_ref_ids is None:
                    device = (
                        session_gen_payload.get("inputs_embeds").device
                        if session_gen_payload.get("inputs_embeds") is not None
                        else active_dict["input_ids"].device
                    )
                    decode_ref_ids = torch.empty((1, 0), dtype=torch.long, device=device)
            elif session_gen_payload is not None:
                raw_gen_payload = dict(session_gen_payload)
                decode_ref_ids = session_gen_payload.get("input_ids")
                if decode_ref_ids is None:
                    decode_ref_ids = active_dict["input_ids"]
            else:
                decode_ref_ids = active_dict.get("input_ids")
                if isinstance(active_inputs, dict):
                    raw_gen_payload = dict(active_inputs)
                else:
                    raw_gen_payload = dict(inputs.items())

            flow_mode = kv_flow_mode_label(self.model)
            if (
                raw_gen_payload.get("inputs_embeds") is None
                and (flow_mode == "none" or native_kv_full_prefill)
            ):
                gen_payload = self._hf_safe_generate_kwargs(dict(raw_gen_payload))
                gen_fn = self.model.generate
            else:
                gen_payload = build_dialogue_generate_payload(
                    self.model,
                    raw_gen_payload,
                    logical_prompt_len=int(logical_prompt_len),
                    dialogue_suffix_only=bool(dialogue_suffix_only),
                )
                gen_fn = (
                    self._call_geopact_generate
                    if should_use_pinned_generate(gen_payload)
                    else self.model.generate
                )
            decode_ref_len = 0
            if decode_ref_ids is not None:
                try:
                    decode_ref_len = int(decode_ref_ids.shape[-1])
                except Exception:
                    decode_ref_len = 0
            early_stop_prompt_len = resolve_early_stop_prompt_len(
                logical_prompt_len=int(logical_prompt_len),
                gen_kwargs=gen_payload,
                decode_ref_len=decode_ref_len,
            )
            strict_early = os.environ.get("GEOPACT_STRICT_EARLY_STOP", "1").strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }
            if (
                gen_payload.get("past_key_values") is not None
                and int(early_stop_prompt_len) > 4
            ):
                msg = (
                    "GeoPACT early-stop miswired on dialogue KV decode: "
                    f"early_stop_prompt_len={early_stop_prompt_len} "
                    f"decode_ref_len={decode_ref_len} logical_prompt_len={logical_prompt_len} "
                    f"dialogue_suffix={bool(dialogue_suffix_only)}"
                )
                if strict_early:
                    raise RuntimeError(msg)
                eval_logger.error(msg)
            decode_diag = {
                "early_stop_prompt_len": int(early_stop_prompt_len),
                "decode_ref_len": int(decode_ref_len),
                "logical_prompt_len": int(logical_prompt_len),
                "dialogue_suffix_only": bool(dialogue_suffix_only),
            }
            step_ledger.stop_bucket("generate_payload_ms")
            try:
                from lmms_eval.models.geopact_bilateral_trace import trace_decode_inputs, trace_enabled

                if trace_enabled():
                    trace_decode_inputs(
                        gen_payload=gen_payload,
                        logical_prompt_len=int(logical_prompt_len),
                        early_stop_prompt_len=int(early_stop_prompt_len),
                        decode_ref_len=int(decode_ref_len),
                    )
            except Exception:
                pass
            generate_t0 = time.perf_counter()
            with torch.inference_mode():
                cont, gen_timing = call_qwen_vl_generate_with_timing(
                    self.model,
                    logical_prompt_len,
                    gen_fn,
                    tokenizer=self.tokenizer,
                    early_stop_on_action=self.early_stop_on_action,
                    early_stop_prompt_len=early_stop_prompt_len,
                    timing_target=inner,
                    **gen_payload,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_new_toks,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=gen_kwargs.get("top_p", None) if do_sample else None,
                    use_cache=self.use_cache,
                )
            try:
                from lmms_eval.models.geopact_bilateral_trace import trace_decode_out, trace_enabled

                if trace_enabled():
                    trace_decode_out(
                        generated_ids=cont,
                        gen_timing=gen_timing,
                        logical_prompt_len=int(logical_prompt_len),
                        early_stop_prompt_len=int(early_stop_prompt_len),
                    )
            except Exception:
                pass
            release_pending_yolo_gpu(inner)
            if self.rank == 0:
                dec_toks = float(
                    gen_timing.get(
                        "decode_output_tokens",
                        gen_timing.get("output_tokens_early_stop_cap", 0.0),
                    )
                )
                gen_ms = float(gen_timing.get("generation_ms", 0.0))
                audit_toks, audit_ms = self._geopact_generate_audit_thresholds()
                if dec_toks > audit_toks or gen_ms > audit_ms:
                    eval_logger.warning(
                        "GeoPACT generate slow-path audit: "
                        f"decode_tokens={dec_toks:.0f} generation_ms={gen_ms:.1f} "
                        f"early_stop_prompt_len={early_stop_prompt_len} "
                        f"logical_prompt_len={logical_prompt_len} "
                        f"dialogue_suffix={bool(dialogue_suffix_only)} "
                        f"past_kv={gen_payload.get('past_key_values') is not None}"
                    )
            step_ledger = get_step_ledger(self.model)
            step_ledger.set_audit(
                output_token_len=int(
                    float(
                        gen_timing.get(
                            "decode_output_tokens",
                            gen_timing.get("output_tokens_early_stop_cap", 0.0),
                        )
                        or 0.0
                    )
                )
            )
            _vlm_fw_ms = float(gen_timing.get("vlm_prefill_ms", 0.0))
            _llm_pf_fw_ms = float(
                gen_timing.get("llm_prefill_forward_ms", gen_timing.get("prefill_forward_ms", 0.0))
            )
            _llm_dec_fw_ms = float(
                gen_timing.get(
                    "llm_decode_forward_ms",
                    gen_timing.get("decode_total_ms", gen_timing.get("decode_forward_ms", 0.0)),
                )
            )
            step_ledger.add_ms("vlm_prefill_ms", _vlm_fw_ms)
            step_ledger.add_ms("llm_prefill_ms", _llm_pf_fw_ms)
            step_ledger.add_ms("llm_decode_ms", _llm_dec_fw_ms)
            # generate() wall time not spent inside model forwards: sampling,
            # prepare_inputs, stopping criteria, per-step syncs.
            _gen_wall_ms = float(gen_timing.get("generation_ms", 0.0))
            if _gen_wall_ms > 0:
                step_ledger.add_ms(
                    "generate_loop_ms",
                    max(_gen_wall_ms - _vlm_fw_ms - _llm_pf_fw_ms - _llm_dec_fw_ms, 0.0),
                )
            if torch.cuda.is_available() and (cuda_sync_enabled() or mem_audit_enabled()):
                if cuda_sync_enabled():
                    torch.cuda.synchronize()
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                avg_mem_mb = (mem_t0 + torch.cuda.memory_allocated()) / 2.0 / (1024.0 * 1024.0)
            step_kv_cache_bytes = 0
            if _kv_probe is not None and _kv_probe.get("pkv") is not None:
                # measured KV bytes of this step's live cache (post-decode
                # steady state; FastV/PDrop mid-prefill pruning included).
                step_kv_cache_bytes = kv_cache_bytes_from(_kv_probe["pkv"])
                _kv_probe["pkv"] = None

            kv_finalize_t0 = time.perf_counter()
            if session_kv_enabled() and dialogue_kv_enabled():
                if dialogue_cache_prompt_ids is None:
                    if step_idx != 0 and dialogue_suffix_allowed:
                        raise RuntimeError(
                            "GeoPACT dialogue KV: missing dialogue cache prompt ids"
                        )
                    if not qwen_messages_batch:
                        raise RuntimeError(
                            "GeoPACT dialogue KV: missing messages for cache boundary ids"
                        )
                    dialogue_cache_prompt_ids = self._compact_ids_at_user_boundary(
                        active_dict,
                        qwen_messages_batch[0],
                    )
                dialogue_grid = self._dialogue_cumulative_image_grid(
                    step_idx,
                    active_dict,
                    suffix_dict,
                )
                if dialogue_cache_prompt_ids is not None and int(dialogue_capture_len) != len(
                    dialogue_cache_prompt_ids
                ):
                    raise RuntimeError(
                        "GeoPACT dialogue KV: capture_len vs cache prompt ids mismatch "
                        f"capture_len={dialogue_capture_len} ids={len(dialogue_cache_prompt_ids)}"
                    )
                try:
                    dialogue_cache_stats = finalize_dialogue_kv_after_generate(
                        self.model,
                        sample_doc,
                        prompt_token_len=dialogue_capture_len,
                        prompt_input_ids=dialogue_cache_prompt_ids,
                        capture=dialogue_kv_capture,
                        image_grid_thw=dialogue_grid,
                    )
                finally:
                    if dialogue_kv_capture is not None:
                        dialogue_kv_capture.remove()
                if self.rank == 0:
                    eval_logger.info(f"GeoPACT dialogue-kv cache-update: {dialogue_cache_stats}")
            step_ledger.add_ms(
                "dialogue_kv_finalize_ms",
                (time.perf_counter() - kv_finalize_t0) * 1000.0,
            )

            if _module_a_sideload_enabled() and self._use_visual_pruning():
                sideload_t0 = time.perf_counter()
                _dispatch_module_a_sideload(inner, str(self.visual_prune_policy or ""))
                step_ledger.add_ms(
                    "module_a_sideload_ms",
                    (time.perf_counter() - sideload_t0) * 1000.0,
                )

            post_t0 = time.perf_counter()
            cont_len = int(cont.shape[-1])
            ref_len = int(decode_ref_ids.shape[-1])
            embeds_generate = (
                isinstance(active_dict, dict)
                and (
                    active_dict.get("inputs_embeds") is not None
                    or session_gen_payload is not None
                    or gen_payload.get("past_key_values") is not None
                )
            )
            # inputs_embeds / past_kv generate usually returns decode-only token ids.
            if embeds_generate and cont_len <= ref_len:
                generated_ids = [cont[0]]
            elif cont_len > int(logical_prompt_len):
                generated_ids = [cont[0, int(logical_prompt_len) :]]
            else:
                generated_ids = [
                    out[len(inp) :] for inp, out in zip(decode_ref_ids, cont)
                ]
            answers = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if os.environ.get("UITARS_DEBUG_RAW_DECODE", "").strip() == "1":
                for _i, _a in enumerate(answers):
                    if not _uitars_repair_action_text(_a or ""):
                        _raw = self.processor.batch_decode(
                            [generated_ids[_i]], skip_special_tokens=False
                        )[0]
                        eval_logger.warning(
                            "UITARS empty-after-repair (geopact path): "
                            f"n_tokens={len(generated_ids[_i])} "
                            f"raw_head={_raw[:400]!r} raw_tail={_raw[-200:]!r} "
                            f"ids_head={generated_ids[_i][:24].tolist()}"
                        )
            step_ledger.add_ms(
                "response_postprocess_ms",
                (time.perf_counter() - post_t0) * 1000.0,
            )
            step_e2e_ms = (time.perf_counter() - step_t0) * 1000.0
            _join_ms = getattr(
                inner,
                "_geopact_module_a_prefetch_async_join_ms",
                None,
            )
            if _join_ms is not None:
                _prune_diag = getattr(inner, "_geopact_last_prune_diag", None) or {}
                step_ledger.set_audit(
                    module_a_join_ms=round(float(_join_ms), 2),
                    module_a_overlap_ms=getattr(
                        inner,
                        "_geopact_module_a_prefetch_async_overlap_ms",
                        None,
                    ),
                    prefetch_reused=_prune_diag.get("prefetch_reused"),
                    prefetch_source=_prune_diag.get("prefetch_source"),
                    prefetch_grid_match=_prune_diag.get("prefetch_grid_match"),
                )
            ledger_payload = step_ledger.finalize_step(e2e_ms=step_e2e_ms)

            for i, (ans, ctx) in enumerate(zip(answers, contexts)):
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                ans = _uitars_repair_action_text(ans)
                ans = _uitars_unwrap_tool_call_json(ans)
                ans = _uitars_dsl_to_omni_json(ans)
                if _uitars_coord_rescale_enabled():
                    _grid = (
                        active_dict.get("image_grid_thw")
                        if isinstance(active_dict, dict)
                        else getattr(active_inputs, "image_grid_thw", None)
                    )
                    ans = _uitars_rescale_coords_to_1000(ans, _grid)
                ans, raw_original = canonicalize_omni_action_output(ans)
                ans_out = ans
                if self.attach_usage_stats:
                    n_out = len(generated_ids[i]) if i < len(generated_ids) else 0
                    per_prompt_len = int(logical_prompt_len)
                    ref_ids = (
                        active_dict.get("input_ids")
                        if isinstance(active_dict, dict)
                        else getattr(active_inputs, "input_ids", None)
                    )
                    from lmms_eval.models.simple.qwen_vl_omnigui_common import count_qwen_vision_tokens

                    vision_tok_count = count_qwen_vision_tokens(ref_ids, self.model)
                    hist_tok_count = max(per_prompt_len - vision_tok_count, 0)
                    usage = build_qwen_vl_usage_stats(
                        step_t0=step_t0,
                        preprocess_ms=processor_ms,
                        generate_t0=generate_t0,
                        gen_timing=gen_timing,
                        prompt_len=per_prompt_len,
                        output_tokens=n_out,
                        peak_mem_mb=peak_mem_mb,
                        avg_mem_mb=avg_mem_mb,
                        compact_stats=compact_stats,
                        vision_tokens=vision_tok_count,
                        history_tokens=hist_tok_count,
                        processor_prompt_len=processor_prompt_len,
                        exclusive_timings={
                            **(ledger_payload.get("timings_exclusive_ms") or {}),
                            "timing_e2e_ms": ledger_payload.get("timing_e2e_ms"),
                        },
                        timing_audit=ledger_payload.get("timing_audit"),
                    )
                    usage = merge_exclusive_into_usage(usage, ledger_payload)
                    if compact_stats:
                        suffix_tok = int(compact_stats.get("compact_seq_len", processor_prompt_len))
                        usage["processor_prompt_tokens"] = int(processor_prompt_len)
                        usage["suffix_prompt_tokens"] = suffix_tok
                        usage["compact_prompt_tokens"] = int(per_prompt_len)
                        usage["visual_tokens_kept"] = int(compact_stats.get("n_visual_kept", 0))
                        usage["visual_tokens_total"] = int(compact_stats.get("n_visual_total", 0))
                        usage["input_tokens"] = int(per_prompt_len)
                    if session_stats:
                        usage.update({
                            k: v
                            for k, v in session_stats.items()
                            if isinstance(v, (int, float, bool, str))
                        })
                    if dialogue_cache_stats:
                        usage.update({
                            k: v
                            for k, v in dialogue_cache_stats.items()
                            if isinstance(v, (int, float, bool, str))
                        })
                    usage.update(decode_diag)
                    if mem_audit_enabled():
                        usage["kv_cache_bytes"] = int(step_kv_cache_bytes)
                        usage["kv_cache_mb"] = round(
                            step_kv_cache_bytes / (1024.0 * 1024.0), 2
                        )
                        if peak_mem_mb > 0:
                            usage["kv_frac_of_peak_mem"] = round(
                                step_kv_cache_bytes
                                / (1024.0 * 1024.0)
                                / float(peak_mem_mb),
                                4,
                            )
                    if raw_original:
                        usage["raw_model_output"] = raw_original
                    if _ss_latency_log_enabled():
                        # Single-step latency ledger: grounding tasks (SSv2 /
                        # SSPro / MMBench) discard usage in process_results, so
                        # emit a parseable per-doc line for the timing harvest.
                        try:
                            eval_logger.info(
                                "SS_LATENCY "
                                + json.dumps(
                                    {
                                        k: usage.get(k)
                                        for k in (
                                            "e2e_latency_ms",
                                            "ttft_ms",
                                            "vision_prefill_ms",
                                            "vlm_prefill_ms",
                                            "llm_prefill_ms",
                                            "decode_latency_ms",
                                            "pruning_ms",
                                            "geopact_prepare_ms",
                                            "input_tokens",
                                            "output_tokens",
                                            "visual_tokens_kept",
                                            "visual_tokens_total",
                                            "peak_gpu_memory_mb",
                                            "kv_cache_bytes",
                                            "kv_cache_mb",
                                            "kv_frac_of_peak_mem",
                                        )
                                        if usage.get(k) is not None
                                    }
                                )
                            )
                        except Exception:
                            pass
                    ans_out = format_response_with_usage(ans, usage)
                res.append(ans_out)
                self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), ans_out)
                if session_kv_enabled():
                    inner._geopact_dialogue_suffix_only_step = False
                pbar.update(1)
            if _anatomy_token is not None:
                from lmms_eval.models.geopact_anatomy import anatomy_step_end

                anatomy_step_end(_anatomy_token, step_id=int(step_idx))

        pbar.close()
        if _module_a_sideload_enabled() and self._use_visual_pruning():
            _flush_module_a_sideload_queue(self._geopact_target())
        return re_ords.get_original(res)
