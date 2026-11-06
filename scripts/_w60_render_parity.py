#!/usr/bin/env python
"""W60: prove the two assistant-text renderers agree, without a GPU.

The self-conditioned protocol only holds together if the assistant turn the
history loop renders at step t is byte-identical to what the invariant checker
expects for the action stored at step t-1. Both go through
``_qwen_assistant_text(step, _omni_action_to_qwen_tool_call(json))``, but "both
call the same function" is an assumption about the code as written, and the
invariant check would fire on every step of every episode if it were wrong.
This verifies it directly on the real renderers over a spread of action types.

Run with the eval environment so lmms_eval imports resolve:
  source scripts/eval_env_defaults.sh && "${EVAL_PYTHON}" scripts/_w60_render_parity.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmms_eval.tasks import self_conditioned as sc  # noqa: E402
from lmms_eval.tasks.agentcpm_gui.utils import (  # noqa: E402
    _omni_action_to_qwen_tool_call,
    _qwen_assistant_text,
    _selfcond_render_assistant,
)

ACTIONS = [
    {"action_type": 0, "coordinate": [123, 456]},
    {"action_type": 3, "coordinate": [10, 20], "start_coordinate": [10, 20],
     "end_coordinate": [10, 400]},
    {"action_type": 7, "text": "hello world"},
    {"action_type": 8},
    {"action_type": 9},
    {"action_type": 10},
    {"action_type": -1},
    dict(sc.UNPARSEABLE_ACTION),
]


def render_hist(step, action):
    """The history loop's renderer: dump -> tool_call -> assistant text."""
    return _qwen_assistant_text(
        step,
        _omni_action_to_qwen_tool_call(json.dumps(action, ensure_ascii=False)),
    )


def main():
    n_ok = 0
    for step in (0, 1, 4):
        for act in ACTIONS:
            clean = {k: v for k, v in act.items() if not str(k).startswith("_")}
            a = render_hist(step, clean)
            b = _selfcond_render_assistant(step, clean)
            if a != b:
                raise SystemExit(
                    f"renderer mismatch at step {step} action {clean}\n"
                    f"  history loop : {a!r}\n"
                    f"  invariant fn : {b!r}"
                )
            n_ok += 1
    print(f"ok  {n_ok} (step, action) pairs: history loop == invariant renderer")


if __name__ == "__main__":
    main()
