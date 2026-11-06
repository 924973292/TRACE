#!/usr/bin/env python3
"""Extract per-within-episode-step TTFT for Fig.A3 (dense vs GeoPACT).

Reads the OmniGUI lat2 samples.jsonl for the two arms that matter -- the
dense baseline (re-prefills all history) and GeoPACT (flat, KV-reuse) --
and emits mean TTFT per step index so the figure shows dense growing with
history length while GeoPACT stays flat. Zero extra compute: reuses the
already-landed lat2 probe.

Usage (remote): EVAL_PYTHON scripts/_w33_ttft_by_step.py <lat2_omni_root> > out.csv
"""
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/xds8b_lat2/omnigui"
ARMS = {"baseline_full": "dense", "ours_c50_h10": "ours"}


def load(arm_dir):
    fs = sorted(glob.glob(os.path.join(arm_dir, "results", "*",
                                       "*samples*.jsonl")))
    if not fs:
        return None
    # Within-episode step index comes from step_eval.step_id (authoritative;
    # episode_id groups steps). TTFT prefers the top-level ttft_ms, falling
    # back to step_eval.ttft_ms.
    by_step = defaultdict(lambda: [0.0, 0])
    with open(fs[-1], encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            se = r.get("step_eval") or {}
            if isinstance(se, str):
                try:
                    se = json.loads(se)
                except json.JSONDecodeError:
                    se = {}
            ttft = r.get("ttft_ms")
            if not isinstance(ttft, (int, float)):
                ttft = se.get("ttft_ms")
            step_id = se.get("step_id")
            if not isinstance(ttft, (int, float)) or step_id is None:
                continue
            try:
                s = int(step_id)
            except (TypeError, ValueError):
                continue
            by_step[s][0] += float(ttft)
            by_step[s][1] += 1
    return by_step


def main():
    series = {}
    for arm, tag in ARMS.items():
        bs = load(os.path.join(ROOT, arm))
        if bs:
            series[tag] = bs
    max_s = max((s for bs in series.values() for s in bs), default=-1)
    print("step," + ",".join(f"{t}_ttft_ms,{t}_n" for t in series))
    for s in range(max_s + 1):
        cells = []
        for t in series:
            acc = series[t].get(s)
            if acc and acc[1]:
                cells.append(f"{acc[0] / acc[1]:.1f},{acc[1]}")
            else:
                cells.append(",0")
        print(f"{s}," + ",".join(cells))


if __name__ == "__main__":
    main()
