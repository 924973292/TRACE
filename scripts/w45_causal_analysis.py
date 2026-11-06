#!/usr/bin/env python
"""W43-B analysis: replayed arms vs same-step baseline, episode bootstrap.

For each (budget, arm) the affected steps are those whose in-window history
(H frames back) contains an intervened frame from the override file. Step SR
of the replay run is compared with the unmodified baseline run (the w45_ret2
ours arm at the same budget) restricted to exactly those steps, paired by
(episode_id, step_id). CI: episode-cluster percentile bootstrap, 10k draws.

Decision rule (protocol B): "commitment errors cause step failures" may
enter the paper only if the delete-arm interval excludes zero; restore-arm
effects are sufficiency evidence, reported separately.

Writes docs/w45_causal_replay.json.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _w41_commit_conditional import find_samples, load_jsonl  # noqa: E402

H_WINDOW = 5
N_BOOT = 10000
SEED = 20260730

ARMS = [
    ("omni_mild", "restore", "logs/w45_ret2/omni_ours_c50_h10"),
    ("omni_mild", "delete", "logs/w45_ret2/omni_ours_c50_h10"),
    ("omni_tight", "restore", "logs/w45_ret2/omni_ours_c25_h05"),
    ("omni_tight", "delete", "logs/w45_ret2/omni_ours_c25_h05"),
]


def per_step_correct(run_dir):
    out = {}
    for rec in load_jsonl(find_samples(run_dir)):
        se = rec.get("step_eval") or {}
        if isinstance(se, str):
            try:
                se = json.loads(se)
            except Exception:
                se = {}
        ep, st = str(se.get("episode_id")), int(se.get("step_id", -1))
        out[(ep, st)] = 1.0 if se.get("exact_match") else 0.0
    return out


def episode_bootstrap_diff(pairs, n_boot=N_BOOT, seed=SEED):
    """pairs: {(ep, st): (replay, base)}; returns (mean_diff, lo, hi)."""
    by_ep = {}
    for (ep, _st), (a, b) in pairs.items():
        by_ep.setdefault(ep, []).append(a - b)
    eps = sorted(by_ep)
    diffs = [d for v in by_ep.values() for d in v]
    mean = 100.0 * sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [d for _ in eps
                  for d in by_ep[eps[rng.randrange(len(eps))]]]
        boots.append(100.0 * sum(sample) / len(sample))
    boots.sort()
    return (round(mean, 2),
            round(boots[int(0.025 * n_boot)], 2),
            round(boots[int(0.975 * n_boot) - 1], 2))


def main():
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    result = {"h_window": H_WINDOW, "n_boot": N_BOOT,
              "metric": ("per-step exact_match (full-action EM, the same "
                         "field used by the W41 commitment-error audit)"),
              "arms": {}}
    for budget, arm, base_dir in ARMS:
        tag = f"{budget}_{arm}"
        ov_path = f"logs/w45_causal/{budget}_{arm}.jsonl"
        run_dir = f"logs/w45_causal/{budget}_{arm}_run"
        intervened = {}
        for r in load_jsonl(ov_path):
            intervened.setdefault(str(r["episode_id"]), set()).add(
                int(r["frame_step"]))
        replay = per_step_correct(run_dir)
        base = per_step_correct(base_dir)
        pairs = {}
        for (ep, st), a in replay.items():
            frames = intervened.get(ep)
            if not frames:
                continue
            if not any(st - H_WINDOW <= f < st for f in frames):
                continue
            b = base.get((ep, st))
            if b is None:
                continue
            pairs[(ep, st)] = (a, b)
        if not pairs:
            result["arms"][tag] = {"steps": 0, "note": "no affected steps"}
            print(tag, "no affected steps")
            continue
        mean, lo, hi = episode_bootstrap_diff(pairs)
        sr_replay = 100.0 * sum(a for a, _ in pairs.values()) / len(pairs)
        sr_base = 100.0 * sum(b for _, b in pairs.values()) / len(pairs)
        entry = {
            "steps": len(pairs),
            "episodes": len({ep for ep, _ in pairs}),
            "interventions": sum(len(v) for v in intervened.values()),
            "step_sr_replay": round(sr_replay, 2),
            "step_sr_baseline_same_steps": round(sr_base, 2),
            "diff": mean,
            "ci95_episode": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        result["arms"][tag] = entry
        print(tag, json.dumps(entry))
    os.makedirs("docs", exist_ok=True)
    with open("docs/w45_causal_replay.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("W45_CAUSAL_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
