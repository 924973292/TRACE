#!/usr/bin/env python3
"""Dataset facts for history-window validation (max hist length vs window N)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    data_path = REPO / "data" / "merged_sorted.json"
    if not data_path.is_file():
        print(f"SKIP: missing {data_path}")
        return 0

    docs = json.loads(data_path.read_text(encoding="utf-8"))
    max_ah = max(len(d.get("action_history") or []) for d in docs)
    max_ep = 0
    for d in docs:
        ep = str(d.get("episode_id", ""))
        if ep:
            max_ep = max(max_ep, int(d.get("step_index", d.get("step_id", 0)) or 0) + 1)

    print(f"dataset_docs={len(docs)} max_action_history={max_ah} max_episode_steps~={max_ep}")
    if max_ah <= 50:
        print("OK: history_n=50 covers full dataset (equivalent to unlimited for accuracy)")
    else:
        print(f"WARN: max_action_history={max_ah} > 50 — history50 may truncate")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
