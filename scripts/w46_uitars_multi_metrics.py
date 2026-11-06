#!/usr/bin/env python3
"""UI-TARS multi-step metrics on the action steps where the token budget binds.

Why not plain Step SR. The stage-B gate showed that on OmniGUI,
UI-TARS-1.5-7B answers 35.7% of TASK_COMPLETE steps with no parseable action and
another 31 of them with a tap: it has no post-training habit for episode-level
termination. Two thirds of every failure on the dense run is a termination step.
A whole-episode Step SR on this backbone is therefore dominated by a deficit
that no token-selection policy can touch, and comparing pruning arms through it
would mostly measure noise around a constant.

What this reports instead, for every arm:
  * step_sr_all       -- the whole-episode number, unchanged, so nothing is hidden
  * step_sr_coord     -- exact match restricted to steps whose REFERENCE action
                         carries a coordinate (tap / long-press / swipe). This is
                         where visual evidence is the binding constraint.
  * type_acc_coord    -- action-type accuracy on the same subset, which separates
                         "looked in the wrong place" from "chose the wrong verb"
  * grounding_acc     -- exact match among coordinate steps whose type is already
                         right: the purest grounding read
  * parse_rate_coord  -- share of those steps that yielded any action at all
  * term_deficit      -- share of reference-termination steps answered with a
                         non-termination action, reported as a backbone property
The subset is defined by the REFERENCE action, never by the prediction, so no arm
can enlarge or shrink its own denominator.

Usage: python3 scripts/w46_uitars_multi_metrics.py <run_dir> [run_dir ...]
       python3 scripts/w46_uitars_multi_metrics.py --json out.json <dirs...>
"""
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

# Reference action types that carry a target location. Derived from
# lmms_eval/tasks/agentcpm_gui/utils.py: TAP=0, DOUBLE_TAP=1, LONG_PRESS=2,
# SWIPE_*=3..6. INPUT(7) is excluded: its correctness is text, not geometry.
COORD_TYPES = {0, 1, 2, 3, 4, 5, 6}
TERM_TYPES = {10, 11}


def step_records(run_dir):
    for f in sorted(glob.glob(os.path.join(run_dir, "**", "*samples*.jsonl"),
                              recursive=True)):
        for line in open(f, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ev = rec.get("step_eval") or {}
            if isinstance(ev, list) and ev:
                ev = ev[0]
            if isinstance(ev, dict) and ev:
                yield ev


def measure(run_dir):
    n = n_coord = 0
    em_all = em_coord = 0
    type_ok_coord = parsed_coord = 0
    right_type_coord = right_type_hit = 0
    n_term = term_missed = 0
    for ev in step_records(run_dir):
        gt = ev.get("gt_action_type")
        pd = ev.get("pd_action_type")
        em = bool(ev.get("exact_match"))
        n += 1
        em_all += em
        if gt in TERM_TYPES:
            n_term += 1
            if pd not in TERM_TYPES:
                term_missed += 1
        if gt in COORD_TYPES:
            n_coord += 1
            em_coord += em
            if pd not in (-1, None):
                parsed_coord += 1
            if gt == pd:
                type_ok_coord += 1
                right_type_coord += 1
                right_type_hit += em
    if not n:
        return None
    pct = lambda a, b: (round(100.0 * a / b, 2) if b else None)
    return {
        "run": run_dir,
        "steps": n,
        "coord_steps": n_coord,
        "step_sr_all": pct(em_all, n),
        "step_sr_coord": pct(em_coord, n_coord),
        "type_acc_coord": pct(type_ok_coord, n_coord),
        "grounding_acc": pct(right_type_hit, right_type_coord),
        "parse_rate_coord": pct(parsed_coord, n_coord),
        "term_steps": n_term,
        "term_deficit": pct(term_missed, n_term),
    }


def main(argv):
    out_json = None
    if argv and argv[0] == "--json":
        out_json, argv = argv[1], argv[2:]
    dirs = argv or sorted(glob.glob("logs/w46_gate/*"))
    rows = [r for r in (measure(d) for d in dirs if os.path.isdir(d)) if r]
    if not rows:
        print("no measurable run dirs")
        return 1
    hdr = (f"{'run':<46}{'n':>5}{'nC':>5}{'SR_all':>8}{'SR_coord':>10}"
           f"{'type':>7}{'ground':>8}{'parse':>7}{'termΔ':>7}")
    print(hdr)
    for r in rows:
        f = lambda k: ("  --  " if r[k] is None else f"{r[k]:.2f}")
        print(f"{os.path.basename(r['run']):<46}{r['steps']:>5}"
              f"{r['coord_steps']:>5}{f('step_sr_all'):>8}{f('step_sr_coord'):>10}"
              f"{f('type_acc_coord'):>7}{f('grounding_acc'):>8}"
              f"{f('parse_rate_coord'):>7}{f('term_deficit'):>7}")
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        json.dump({"rows": rows,
                   "coord_types": sorted(COORD_TYPES),
                   "rationale": ("Step SR on this backbone is dominated by a "
                                 "termination deficit unrelated to token "
                                 "budgets; the coordinate-step subset is where "
                                 "visual evidence binds. Subset defined by the "
                                 "reference action only.")},
                  open(out_json, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print("wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
