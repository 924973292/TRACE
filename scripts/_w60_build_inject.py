#!/usr/bin/env python
"""W60 Tier iii: build the history-contamination file.

For every OmniGUI episode with at least two steps, force ONE wrong action into
the history at a chosen step and let the run continue self-conditioned. The
metric is whether the NEXT step recovers.

Design choices and why
----------------------
* One contaminated step per episode, at the FIRST step (index 0) by default.
  Contaminating early maximises the number of subsequent steps that must be
  produced from a polluted cache, which is exactly the channel a write-once
  pruned history is suspected of amplifying.

* The forced action is a `press_back` (action_type 8) unless the ground truth is
  already a back press, in which case it is a `home` (9). Both are valid,
  parseable, plausible GUI actions, so the model sees a coherent-but-wrong
  history rather than garbage. A malformed action would test the parser, not
  recoverability.

* The forced action is IDENTICAL across arms for the same (episode, step). The
  dense and pruned arms must be contaminated the same way, otherwise the
  recovery difference confounds the intervention with the arm.

* The frame sequence stays the ground-truth trajectory, as everywhere in this
  protocol. So "recovery" means: given a history that claims a back press
  happened and a screenshot showing it did not, does the model still choose the
  correct next action? Pruning is what could remove the visual evidence needed
  to notice the contradiction, which is precisely the quantity of interest.

Usage (on the remote pod, after sourcing scripts/eval_env_defaults.sh):
  "${EVAL_PYTHON}" scripts/_w60_build_inject.py \
      --out logs/w60_selfcond/inject.jsonl
"""
import argparse
import glob
import json
import os
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

BACK, HOME = 8, 9
DEFAULT_REF = "logs/w57b_unified/omnigui/baseline_full"


def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_episodes(ref_dir):
    """(episode -> {step: gt_action_type}) read from a finished OmniGUI run.

    A completed teacher-forced run is used as the episode/step manifest so the
    injection file covers exactly the units the metrics script will score, with
    no dataset loading and no GPU.
    """
    f = latest(os.path.join(ref_dir, "results", "*", "*samples*.jsonl"))
    if not f:
        raise SystemExit(f"no samples jsonl under {ref_dir}")
    eps = defaultdict(dict)
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        se = row.get("step_eval")
        if isinstance(se, str):
            try:
                se = json.loads(se)
            except json.JSONDecodeError:
                continue
        if not isinstance(se, dict):
            continue
        ep = str(se.get("episode_id", "?"))
        try:
            st = int(se.get("step_id", 0))
        except (TypeError, ValueError):
            continue
        eps[ep][st] = str(se.get("gt_action_type") or "")
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help="finished OmniGUI run used as the episode manifest")
    ap.add_argument("--out", default="logs/w60_selfcond/inject.jsonl")
    ap.add_argument("--at", type=int, default=0,
                    help="step index to contaminate (default 0, the earliest)")
    args = ap.parse_args()

    eps = load_episodes(args.ref)
    rows = []
    skipped = 0
    for ep, steps in sorted(eps.items()):
        order = sorted(steps)
        # An episode needs a step AFTER the contaminated one, otherwise there is
        # nothing to measure recovery on.
        if args.at not in steps or (args.at + 1) not in steps:
            skipped += 1
            continue
        gt = steps[args.at]
        forced = HOME if gt == "press" else BACK
        rows.append({
            "episode_id": ep,
            "step_index": args.at,
            "action": {"action_type": forced},
            "arm": "inject",
            "gt_action_type": gt,
            "n_steps": len(order),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_back = sum(1 for r in rows if r["action"]["action_type"] == BACK)
    print(f"episodes in manifest : {len(eps)}")
    print(f"skipped (no step {args.at}+1): {skipped}")
    print(f"contaminated episodes: {len(rows)}  (back={n_back}, home={len(rows) - n_back})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
