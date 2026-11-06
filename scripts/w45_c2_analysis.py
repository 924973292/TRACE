#!/usr/bin/env python
"""W43-C2 analysis: repaired-independent replay vs nested baseline.

The replay arms run the identical evaluation with GEOPACT_KEEP_OVERRIDE
pointing at the C1 repaired-independent history keeps (independent election
projected onto S_cur so the cache remains realizable). The nested baseline is
the w45_ret2 ours arm at the same budget. Every step whose episode has at
least one applied override is compared, paired by (episode_id, step_id).
CI: episode-cluster percentile bootstrap, 10k draws.

Decision rule (protocol C): the accuracy cost of nesting must come from this
C2 contrast, not from the C1 violation rates. Either direction is reportable:
"nesting costs <=X points" or "nesting is a serving constraint at no
measurable accuracy cost".

Writes docs/w45_c2_nested_vs_independent.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _w41_commit_conditional import find_samples, load_jsonl  # noqa: E402
from w45_causal_analysis import episode_bootstrap_diff, per_step_correct  # noqa: E402

ARMS = [
    ("omni_mild", "logs/w45_c2/omni_mild_indep_run",
     "logs/w45_c2/omni_mild_indep_repaired.jsonl",
     "logs/w45_ret2/omni_ours_c50_h10"),
    ("omni_tight", "logs/w45_c2/omni_tight_indep_run",
     "logs/w45_c2/omni_tight_indep_repaired.jsonl",
     "logs/w45_ret2/omni_ours_c25_h05"),
]


def main():
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    result = {"n_boot": 10000,
              "metric": ("per-step exact_match, replay(independent-repaired)"
                         " minus baseline(nested), paired by episode/step"),
              "arms": {}}
    for tag, run_dir, ov_path, base_dir in ARMS:
        overridden = {}
        for r in load_jsonl(ov_path):
            ep = str(r.get("episode_id") or r.get("episode_key"))
            # hook-side keys are "unknown-T0001"; step_eval uses bare "T0001"
            if ep.startswith("unknown-"):
                ep = ep[len("unknown-"):]
            st = int(r.get("frame_step", r.get("frame_index", -1)))
            overridden.setdefault(ep, set()).add(st)
        replay = per_step_correct(run_dir)
        base = per_step_correct(base_dir)
        pairs = {}
        for (ep, st), a in replay.items():
            if ep not in overridden:
                continue
            b = base.get((ep, st))
            if b is None:
                continue
            pairs[(ep, st)] = (a, b)
        if not pairs:
            result["arms"][tag] = {"steps": 0, "note": "no paired steps"}
            print(tag, "no paired steps")
            continue
        mean, lo, hi = episode_bootstrap_diff(pairs)
        sr_replay = 100.0 * sum(a for a, _ in pairs.values()) / len(pairs)
        sr_base = 100.0 * sum(b for _, b in pairs.values()) / len(pairs)
        entry = {
            "steps": len(pairs),
            "episodes": len({ep for ep, _ in pairs}),
            "override_rows": sum(len(v) for v in overridden.values()),
            "step_sr_independent_repaired": round(sr_replay, 2),
            "step_sr_nested_same_steps": round(sr_base, 2),
            "diff_independent_minus_nested": mean,
            "ci95_episode": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        result["arms"][tag] = entry
        print(tag, json.dumps(entry))
    os.makedirs("docs", exist_ok=True)
    with open("docs/w45_c2_nested_vs_independent.json", "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("W45_C2_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
