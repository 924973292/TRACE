#!/usr/bin/env python3
"""Regression: dialogue-KV decode early-stop wiring + Omni action completeness."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from lmms_eval.models.simple.qwen_vl_generation_timing import (
    make_omni_action_early_stop_criteria,
    resolve_early_stop_prompt_len,
)
from lmms_eval.tasks.agentcpm_gui.utils import (
    omni_action_is_complete_for_early_stop,
    parse_omni_action,
)


class _FakeTokenizer:
    def __init__(self, text: str) -> None:
        self._text = text

    def decode(self, ids, skip_special_tokens=True):
        return self._text


class _PrefixTokenizer:
    """One token id per character so criteria can be stepped token-by-token."""

    def __init__(self, text: str) -> None:
        self._text = text

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            n = int(ids.numel())
        else:
            n = len(list(ids))
        return self._text[:n]


def test_resolve_early_stop_with_past_kv():
    logical = 1114
    decode_ref = torch.tensor([[42]], dtype=torch.long)
    gen_kwargs = {"past_key_values": object(), "input_ids": decode_ref}
    got = resolve_early_stop_prompt_len(
        logical_prompt_len=logical,
        gen_kwargs=gen_kwargs,
        decode_ref_len=int(decode_ref.shape[-1]),
    )
    assert got == 1, f"expected decode-only offset 1, got {got}"


def test_resolve_early_stop_full_prefill():
    logical = 1116
    gen_kwargs = {"input_ids": torch.zeros(1, logical, dtype=torch.long)}
    got = resolve_early_stop_prompt_len(
        logical_prompt_len=logical,
        gen_kwargs=gen_kwargs,
        decode_ref_len=logical,
    )
    assert got == logical, f"expected full prompt len {logical}, got {got}"


def test_early_stop_fires_on_short_decode_sequence():
    tokenizer = _FakeTokenizer('{"action_type": 0, "coordinate": [100, 200]}')
    crit = make_omni_action_early_stop_criteria(tokenizer, prompt_len=1, min_decode_tokens=2)
    assert crit is not None
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    scores = None
    stopped = False
    for end in range(2, ids.shape[-1] + 1):
        if crit(ids[:, :end], scores):
            stopped = True
            break
    assert stopped, "early_stop must trigger on decode-only ids when prompt_len=1"


def test_early_stop_blocked_with_wrong_logical_len():
    tokenizer = _FakeTokenizer('{"action_type": 0, "coordinate": [100, 200]}')
    crit = make_omni_action_early_stop_criteria(tokenizer, prompt_len=1114, min_decode_tokens=8)
    ids = torch.tensor([[1] + list(range(2, 62))], dtype=torch.long)
    assert not crit(ids, None), "wrong logical prompt_len must not stop within 60 decode tokens"


def _assert_complete(text: str) -> None:
    action = parse_omni_action(text)
    assert action is not None, text
    assert omni_action_is_complete_for_early_stop(action), (text, action)


def _assert_incomplete(text: str) -> None:
    action = parse_omni_action(text)
    assert not omni_action_is_complete_for_early_stop(action), (text, action)


def test_completeness_complete_action_shapes():
    _assert_complete('{"action_type":0,"coordinate":[100,200]}')
    _assert_complete('{"action_type":7,"text":"hello"}')
    _assert_complete(
        '<tool_call>{"name":"mobile_use","arguments":{"action":"system_button","button":"Back"}}</tool_call>'
    )
    _assert_complete(
        '<tool_call>{"name":"mobile_use","arguments":{"action":"terminate","status":"success"}}</tool_call>'
    )
    _assert_complete(
        '<tool_call>{"name":"mobile_use","arguments":{"action":"swipe","coordinate":[500,800],"coordinate2":[500,200]}}</tool_call>'
    )


def test_completeness_incomplete_action_shapes():
    _assert_incomplete('{"action_type":0}')
    _assert_incomplete('{"action_type":7}')
    _assert_incomplete('{"action_type":3,"coordinate":[100,200]}')
    _assert_incomplete('{"action_type":0,"coordinate":[100')


def test_early_stop_waits_for_complete_json_not_first_brace():
    text = '{"action_type":0,"coordinate":[100,200]}'
    tokenizer = _PrefixTokenizer(text)
    crit = make_omni_action_early_stop_criteria(tokenizer, prompt_len=1, min_decode_tokens=2)
    prompt = torch.tensor([[0]], dtype=torch.long)
    gen_ids = torch.tensor([list(range(1, len(text) + 2))], dtype=torch.long)
    ids = torch.cat([prompt, gen_ids], dim=-1)
    stopped_at = None
    for end in range(prompt.shape[-1] + 2, ids.shape[-1] + 1):
        if crit(ids[:, :end], None):
            stopped_at = end - prompt.shape[-1]
            break
    assert stopped_at is not None, "must eventually stop on complete action JSON"
    assert stopped_at >= len(text), f"stopped too early at decode token {stopped_at}"


def test_early_stop_never_fires_on_incomplete_tap():
    text = '{"action_type":0,"coordinate":[100'
    tokenizer = _PrefixTokenizer(text)
    crit = make_omni_action_early_stop_criteria(tokenizer, prompt_len=1, min_decode_tokens=2)
    prompt = torch.tensor([[0]], dtype=torch.long)
    gen_ids = torch.tensor([list(range(1, len(text) + 2))], dtype=torch.long)
    ids = torch.cat([prompt, gen_ids], dim=-1)
    for end in range(prompt.shape[-1] + 2, ids.shape[-1] + 1):
        assert not crit(ids[:, :end], None), "must not stop on truncated tap JSON"


def main() -> int:
    test_resolve_early_stop_with_past_kv()
    test_resolve_early_stop_full_prefill()
    test_early_stop_fires_on_short_decode_sequence()
    test_early_stop_blocked_with_wrong_logical_len()
    test_completeness_complete_action_shapes()
    test_completeness_incomplete_action_shapes()
    test_early_stop_waits_for_complete_json_not_first_brace()
    test_early_stop_never_fires_on_incomplete_tap()
    print("verify_dialogue_kv_early_stop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
