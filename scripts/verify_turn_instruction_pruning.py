#!/usr/bin/env python3
"""Verify Module B pruning uses the per-turn synchronized pure instruction.

Contract (rich_turn protocol): turn t's image tokens must be scored against
turn t's own instruction — the exact pure text rendered into that user turn
(no template boilerplate) — for both the full-prefill window prune and the
history rectify path.

Checks:
  1. frame->instruction extraction aligns 1:1 with episode image paths;
  2. the model-side map keys (normalized paths) resolve to the right text;
  3. texts are pure (no template markers);
  4. protocol consistency: map text == the Instruction line rich_turn renders;
  5. relevance_query_for_frame returns source="turn" with a text-keyed cache
     (same text -> cached tensor, different text -> different query), and
     falls back to the episode-task query for unknown frames.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ["GEOPACT_QWEN_PROMPT_VARIANT"] = "rich_turn"
os.environ["GEOPACT_EVAL_HISTORY_N"] = "9"
for k in ("GEOPACT_PROMPT_BOILER", "GEOPACT_PROMPT_THOUGHT", "GEOPACT_TASK_MODE"):
    os.environ.pop(k, None)

import torch
import torch.nn as nn

from lmms_eval.models.geopact_vl_hook import relevance_query_for_frame
from lmms_eval.models.simple.qwen_vl_omnigui_common import (
    _norm_geopact_path,
    extract_episode_image_paths_from_doc,
    extract_turn_instructions_from_doc,
    set_geopact_episode_image_context,
)
from lmms_eval.tasks.agentcpm_gui.utils import _qwen_step_user_text

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"OK   {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([[(ord(c) % 97) + 1 for c in text[:32]]], dtype=torch.long)

        class R:
            input_ids = ids

        return R()


class _FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self._embed = nn.Embedding(128, 8)
        self._geopact_tokenizer = _FakeTokenizer()

    def get_input_embeddings(self):
        return self._embed


def main() -> int:
    doc = {
        "subset": "verify",
        "episode_id": "E1",
        "step_index": 2,
        "task_description": "Book a flight to Tokyo",
        "current_instruction": "Confirm the payment on this page",
        "current_image": "media/E1/2.png",
        "history": [
            {
                "step_index": 0,
                "image_path": "media/E1/0.png",
                "action_json": "{}",
                "step_instruction": "Open the airline app",
            },
            {
                "step_index": 1,
                "image_path": "media/E1/1.png",
                "action_json": "{}",
                "step_instruction": "Select the cheapest flight",
            },
        ],
    }

    paths = extract_episode_image_paths_from_doc(doc, {})
    instrs = extract_turn_instructions_from_doc(doc, {})
    check("paths/instructions aligned", len(paths) == len(instrs) == 3,
          f"paths={len(paths)} instrs={len(instrs)}")
    check("turn0 uses its own step instruction", instrs[0] == "Open the airline app",
          repr(instrs[0]))
    check("turn1 uses its own step instruction",
          instrs[1] == "Select the cheapest flight", repr(instrs[1]))
    check("current turn uses current instruction",
          instrs[2] == "Confirm the payment on this page", repr(instrs[2]))

    # Episode-level-only doc: every turn falls back to the task goal (the same
    # text rich_turn renders when no per-step instruction exists).
    doc_plain = {
        "subset": "verify",
        "episode_id": "E2",
        "step_index": 1,
        "task_description": "Book a flight to Tokyo",
        "current_image": "media/E2/1.png",
        "history": [
            {"step_index": 0, "image_path": "media/E2/0.png", "action_json": "{}"},
        ],
    }
    plain_instrs = extract_turn_instructions_from_doc(doc_plain, {})
    check("episode-level-only doc falls back to task everywhere",
          plain_instrs == ["Book a flight to Tokyo"] * 2, repr(plain_instrs))

    banned = ("Analyze the screenshot", "Instruction:", "Step:", "<thought>", "tool_call")
    check("texts are pure (no template markers)",
          all(not any(b in t for b in banned) for t in instrs), repr(instrs))

    # Protocol consistency: the map text must be exactly what rich_turn renders
    # into the turn's Instruction line.
    rendered = _qwen_step_user_text(
        task_description=doc["task_description"],
        step_instruction="Open the airline app",
        step_index=0,
        current=False,
    )
    check("rich_turn renders the same text",
          f"Instruction: {instrs[0]}\n" in rendered, repr(rendered))

    inner = _FakeInner()
    set_geopact_episode_image_context(inner, doc, paths, {})
    turn_map = getattr(inner, "_geopact_turn_instr_by_path", {})
    check("model map has one entry per frame", len(turn_map) == 3, repr(turn_map))
    check("map keyed by normalized path",
          turn_map.get(_norm_geopact_path("media/E1/0.png")) == "Open the airline app",
          repr(sorted(turn_map)))

    q0, src0 = relevance_query_for_frame(inner, paths[0])
    check("frame 0 query from its turn", src0 == "turn" and q0 is not None, src0)
    q0_again, _ = relevance_query_for_frame(inner, paths[0])
    check("same text hits the cache", q0_again is q0)
    q2, src2 = relevance_query_for_frame(inner, paths[2])
    check("different instruction -> different query",
          src2 == "turn" and not torch.equal(q0, q2))
    q1, src1 = relevance_query_for_frame(inner, paths[1])
    check("middle turn resolves via map", src1 == "turn" and q1 is not None, src1)

    inner._geopact_task_instruction_text = doc["task_description"]
    q_unknown, src_unknown = relevance_query_for_frame(inner, "media/E1/other.png")
    check("unknown frame falls back to episode-task query",
          src_unknown == "global_fallback" and q_unknown is not None, src_unknown)

    # Suffix-style call (single-frame image_paths list must not corrupt the map).
    set_geopact_episode_image_context(inner, doc, paths, {})
    check("map rebuild is stable",
          getattr(inner, "_geopact_turn_instr_by_path", {}) == turn_map)

    if FAILURES:
        print(f"FAIL verify_turn_instruction_pruning: {len(FAILURES)} failed")
        return 1
    print("ALL TURN-INSTRUCTION PRUNING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
