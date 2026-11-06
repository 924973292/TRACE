"""Generation timing for Qwen2-VL / Qwen3-VL via forward hooks + TTFT criteria."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from lmms_eval.tasks.agentcpm_gui.utils import (
    omni_action_is_complete_for_early_stop,
    parse_omni_action,
)


def resolve_early_stop_prompt_len(
    *,
    logical_prompt_len: int,
    gen_kwargs: Dict[str, Any],
    decode_ref_len: int = 0,
) -> int:
    """Prompt offset for stopping criteria when prefix tokens live in ``past_key_values``.

    Dialogue-KV suffix decode passes only the last suffix token in ``input_ids``;
    using ``logical_prompt_len`` (~prefix+suffix) disables early-stop and can run
  to ``max_new_tokens``. When ``past_key_values`` is set, use the decode slice length.
    """
    if gen_kwargs.get("past_key_values") is None:
        return int(logical_prompt_len)
    if decode_ref_len > 0:
        return int(decode_ref_len)
    input_ids = gen_kwargs.get("input_ids")
    if input_ids is not None:
        try:
            return int(input_ids.shape[-1])
        except Exception:
            pass
    return 0


# Argument bodies legitimately nest one level of parentheses — UI-TARS's own
# box format is click(start_box='(100,200)') — so a flat [^()]* body never
# closes on those calls and decode runs to max_new_tokens. Allow a single level
# of nesting: (?:[^()]|\([^()]*\))*. A miss only wastes decode time, but here it
# wasted it on the most common legacy form.
_UITARS_DSL_ACTION_COMPLETE = re.compile(
    r"(?:^|\n)\s*(?:Action\s*:\s*)?"
    r"(?:click|left_single|left_double|double_click|right_single|long_press|"
    r"drag|swipe|scroll|type|press_back|press_home|press_enter|hotkey|"
    r"back|home|finished|terminate|wait|open_app|call_user)\s*"
    r"\((?:[^()]|\([^()]*\))*\)",
    re.IGNORECASE,
)


def _uitars_dsl_early_stop_enabled() -> bool:
    return os.environ.get("UITARS_DSL_EARLY_STOP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def make_omni_action_early_stop_criteria(
    tokenizer: Any,
    prompt_len: int,
    *,
    min_decode_tokens: int = 8,
) -> Optional[StoppingCriteria]:
    """Stop greedy decode once a complete OmniGUI action JSON is present.

    With UITARS_DSL_EARLY_STOP=1 (set by the ui_tars wrapper), also stop once
    a complete UI-TARS DSL action call (e.g. ``click(start_box='(x,y)')``)
    has been emitted, since UI-TARS-1.5 produces long thought chains before
    its action and never emits Omni JSON on multi-step tasks.
    """

    class _OmniActionEarlyStopCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self._stopped = False
            self._last_checked_len = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self._stopped:
                return True
            try:
                cur_len = int(input_ids.shape[-1])
            except Exception:
                return False
            if cur_len <= prompt_len + min_decode_tokens:
                return False
            if cur_len == self._last_checked_len:
                return False
            self._last_checked_len = cur_len

            try:
                gen_ids = input_ids[0, prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                return False
            if _uitars_dsl_early_stop_enabled() and _UITARS_DSL_ACTION_COMPLETE.search(text):
                self._stopped = True
                return True
            if "}" not in text:
                return False

            try:
                action = parse_omni_action(text)
            except Exception:
                action = None
            if action and omni_action_is_complete_for_early_stop(action):
                self._stopped = True
                return True
            return False

    return _OmniActionEarlyStopCriteria()


def _merge_stopping_criteria(
    gen_kwargs: Dict[str, Any],
    extra: list[StoppingCriteria],
) -> None:
    existing = gen_kwargs.get("stopping_criteria")
    if existing is None:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(extra)
        return
    try:
        items = list(existing)
    except TypeError:
        items = [existing]
    gen_kwargs["stopping_criteria"] = StoppingCriteriaList(items + extra)


class QwenVLGenerationTiming:
    """Measure TTFT, vision prefill, LLM prefill, and decode from module forward hooks.

    Metric semantics (Qwen2/3-VL, sequential pipeline):
      - vlm_prefill_ms: first ``visual`` forward (ViT + in-module patch merger).
      - llm_prefill_forward_ms: first ``language_model`` forward on the *full*
        multimodal prompt (all image + text tokens through decoder layers).
      - prefill_ms: vlm_prefill_ms + llm_prefill_forward_ms (approximately).

    It is normal for llm_prefill_forward_ms >= vlm_prefill_ms: the LM pass
    processes ~all input tokens (e.g. 1.8k) while vision only encodes patches.
    """

    def __init__(self, model: Any, *, timing_target: Any = None) -> None:
        self.model = model
        # Object the patched modeling file records `_omnigui_*` timing on.
        # Default (None) keeps the historical `model.model` descent (qwen3 /
        # qwen2_vl composite); the qwen2_5_vl carrier records on the FCG itself.
        self._timing_target = timing_target
        self._handles: list = []
        self.timing: Dict[str, float] = {}
        self._cuda = torch.cuda.is_available()
        self._visual_calls = 0
        self._lm_calls = 0

    def _inner(self) -> Any:
        if self._timing_target is not None:
            return self._timing_target
        return getattr(self.model, "model", self.model)

    def reset(self) -> None:
        self.timing = {
            "forward_calls": 0.0,
            "decode_forward_calls": 0.0,
            "vlm_prefill_ms": 0.0,
            "llm_prefill_forward_ms": 0.0,
            "llm_decode_forward_ms": 0.0,
        }
        self._visual_calls = 0
        self._lm_calls = 0

    def _elapsed_ms(self, t0: float) -> float:
        if self._cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def install(self) -> None:
        inner = self._inner()
        visual = getattr(inner, "visual", None)
        lm = getattr(inner, "language_model", None)

        if visual is not None:

            def visual_pre(_module, _args) -> None:
                if self._visual_calls == 0:
                    self._visual_t0 = time.perf_counter()

            def visual_post(_module, _args, _output) -> None:
                if self._visual_calls == 0 and hasattr(self, "_visual_t0"):
                    self.timing["vlm_prefill_ms"] += self._elapsed_ms(self._visual_t0)
                self._visual_calls += 1

            self._handles.append(visual.register_forward_pre_hook(visual_pre))
            self._handles.append(visual.register_forward_hook(visual_post))

        if lm is not None:

            def lm_pre(_module, _args, _kwargs=None) -> None:
                self._lm_t0 = time.perf_counter()

            def lm_post(_module, _args, _output) -> None:
                if not hasattr(self, "_lm_t0"):
                    return
                ms = self._elapsed_ms(self._lm_t0)
                self.timing["forward_calls"] += 1.0
                if self._lm_calls == 0:
                    self.timing["llm_prefill_forward_ms"] += ms
                else:
                    self.timing["llm_decode_forward_ms"] += ms
                    self.timing["decode_forward_calls"] += 1.0
                self._lm_calls += 1

            self._handles.append(lm.register_forward_pre_hook(lm_pre, with_kwargs=True))
            self._handles.append(lm.register_forward_hook(lm_post))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _merge_site_package_timing(self, timing: Dict[str, float]) -> None:
        inner = self._inner()
        source_timing = getattr(inner, "_omnigui_last_generation_timing", None)
        if not isinstance(source_timing, dict):
            return
        hook_recorded = float(timing.get("forward_calls", 0.0)) >= 1.0
        protected = {
            "llm_prefill_forward_ms",
            "llm_decode_forward_ms",
            "prefill_forward_ms",
            "decode_forward_ms",
        }
        for key, value in source_timing.items():
            if not isinstance(value, (int, float)):
                continue
            if hook_recorded and key in protected:
                continue
            timing[key] = float(value)

    def run(
        self,
        prompt_len: int,
        generate_fn: Callable[..., Any],
        *,
        tokenizer: Any = None,
        early_stop_on_action: bool = True,
        early_stop_prompt_len: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> Tuple[Any, Dict[str, float]]:
        self.reset()
        self.install()

        inner = self._inner()
        reset_timing = getattr(inner, "_omnigui_reset_generation_timing", None)
        if callable(reset_timing):
            try:
                reset_timing(clear_projector=True)
            except TypeError:
                reset_timing()

        timing = self.timing
        stop_prompt_len = (
            int(early_stop_prompt_len)
            if early_stop_prompt_len is not None
            else int(prompt_len)
        )
        timing["early_stop_prompt_len"] = float(stop_prompt_len)
        timing["logical_prompt_len"] = float(prompt_len)
        decode_only = gen_kwargs.get("past_key_values") is not None
        ttft_start_event = None
        ttft_end_event = None
        generate_wall_t0 = time.perf_counter()

        class _FirstTokenTimingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                nonlocal ttft_end_event
                if "_ttft_seen" not in timing:
                    try:
                        current_len = int(input_ids.shape[-1])
                    except Exception:
                        current_len = 0
                    if current_len > stop_prompt_len:
                        timing["_ttft_seen"] = 1.0
                        if torch.cuda.is_available():
                            ttft_end_event = torch.cuda.Event(enable_timing=True)
                            ttft_end_event.record()
                        else:
                            timing["ttft_ms"] = float(
                                (time.perf_counter() - generate_wall_t0) * 1000.0
                            )
                return False

        first_token_criteria = _FirstTokenTimingCriteria()
        criteria_items: list[StoppingCriteria] = [first_token_criteria]
        if early_stop_on_action and tokenizer is not None:
            early_stop = make_omni_action_early_stop_criteria(tokenizer, stop_prompt_len)
            if early_stop is not None:
                criteria_items.append(early_stop)
        _merge_stopping_criteria(gen_kwargs, criteria_items)

        try:
            if torch.cuda.is_available():
                ttft_start_event = torch.cuda.Event(enable_timing=True)
                ttft_start_event.record()
            generate_wall_t0 = time.perf_counter()

            cont = generate_fn(**gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if ttft_start_event is not None and ttft_end_event is not None:
                    timing["ttft_ms"] = float(ttft_start_event.elapsed_time(ttft_end_event))
            else:
                timing.setdefault(
                    "ttft_ms", float((time.perf_counter() - generate_wall_t0) * 1000.0)
                )

            timing["generation_ms"] = float(
                (time.perf_counter() - generate_wall_t0) * 1000.0
            )
            self._merge_site_package_timing(timing)

            vlm_ms = float(
                timing.get(
                    "source_image_feature_prefill_ms",
                    timing.get("vlm_prefill_ms", 0.0),
                )
            )
            llm_prefill_ms = float(
                timing.get(
                    "llm_prefill_forward_ms",
                    timing.get("prefill_forward_ms", 0.0),
                )
            )
            decode_ms = float(
                timing.get(
                    "llm_decode_forward_ms",
                    timing.get("decode_forward_ms", 0.0),
                )
            )
            timing["vlm_prefill_ms"] = vlm_ms
            timing["llm_prefill_forward_ms"] = llm_prefill_ms
            timing["prefill_ms"] = float(
                timing.get(
                    "model_prefill_total_ms",
                    vlm_ms + llm_prefill_ms,
                )
            )
            timing["decode_total_ms"] = decode_ms
            timing["decode_forward_ms"] = decode_ms
            out_len = 0
            try:
                if decode_only:
                    out_len = int(cont.shape[-1])
                else:
                    out_len = max(int(cont.shape[-1]) - int(prompt_len), 0)
            except Exception:
                pass
            timing["decode_output_tokens"] = float(out_len)
            timing["output_tokens_early_stop_cap"] = float(out_len)
            return cont, dict(timing)
        finally:
            self.remove()


def call_qwen_vl_generate_with_timing(
    model: Any,
    prompt_len: int,
    generate_fn: Callable[..., Any],
    *,
    tokenizer: Any = None,
    early_stop_on_action: bool = True,
    early_stop_prompt_len: Optional[int] = None,
    timing_target: Any = None,
    **gen_kwargs: Any,
) -> Tuple[Any, Dict[str, float]]:
    return QwenVLGenerationTiming(model, timing_target=timing_target).run(
        prompt_len,
        generate_fn,
        tokenizer=tokenizer,
        early_stop_on_action=early_stop_on_action,
        early_stop_prompt_len=early_stop_prompt_len,
        **gen_kwargs,
    )
