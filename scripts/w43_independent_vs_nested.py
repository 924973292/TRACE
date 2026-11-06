#!/usr/bin/env python
"""W43-C: independent-vs-nested counterfactual (stats side; probe schema).

The nested order is justified in the paper as a legality condition of the
deletion-only cache, not by an accuracy claim. The missing counterfactual
has two halves:

  C1 (violation rate, mechanical): if the current keep S_cur at k_c and the
     history keep S_ind at k_h are each selected INDEPENDENTLY by the same
     election objective, how often does S_ind demand a token outside S_cur
     (S_ind - S_cur != empty)? Every such frame is unservable by a
     deletion-only cache without re-encoding.

  C2 (matched-budget repair, accuracy): the only realizable independent
     variant must repair each violation, e.g. replace out-of-cache tokens
     with their best in-cache substitutes. Teacher-forced comparison of
     nested vs repaired-independent at identical (k_c, k_h) then measures
     what the nesting constraint costs or saves in accuracy.

Independent selection needs token features, which the dumps do not store.
The harness must therefore run once with the probe flag

    GEOPACT_INDEPENDENT_PROBE=1

which makes the selector additionally run its election at k_h from scratch
(same mass, same instruction, same features, no prefix constraint) and dump
per-frame records:

    {"episode_key": str, "frame_index": int, "n_frame": int,
     "k_c": int, "k_h": int,
     "keep_cur": [int], "keep_hist_nested": [int],
     "keep_hist_independent": [int]}

This script consumes that dump and reports C1; it also emits the repaired
independent keeps needed for the C2 replay (via GEOPACT_KEEP_OVERRIDE).
No number from this protocol may enter the paper before the probe run.

Usage (remote):
    "${EVAL_PYTHON}" scripts/w43_independent_vs_nested.py \
        <probe_dump.jsonl> <tag> [--repair-out <overrides.jsonl>]

Writes/merges docs/w43_independent_nested.json:
    {tag: {"frames": int,
           "violation_rate": pct,        # frames with S_ind - S_cur != {}
           "violation_ci95_frame": [lo, hi],
           "mean_out_of_cache_tokens": float,
           "mean_jaccard_nested_vs_independent": float}}
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _w41_commit_conditional import load_jsonl  # noqa: E402

N_BOOT = 10000
SEED = 20260729


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    probe_dump, tag = sys.argv[1], sys.argv[2]
    repair_out = None
    if "--repair-out" in sys.argv:
        repair_out = sys.argv[sys.argv.index("--repair-out") + 1]
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    viol, n_out, jac = [], [], []
    repairs = []
    for rec in load_jsonl(probe_dump):
        if "keep_hist_independent" not in rec:
            continue  # non-probe record (history frame / reuse action)
        cur = set(rec["keep_cur"])
        nested = set(rec["keep_hist_nested"])
        indep = set(rec["keep_hist_independent"])
        out = indep - cur
        viol.append(1.0 if out else 0.0)
        n_out.append(len(out))
        u = nested | indep
        jac.append(len(nested & indep) / len(u) if u else 1.0)
        if repair_out is not None:
            # realizable repair: out-of-cache demands fall back to the
            # highest-ranked in-cache tokens not already selected
            in_cache = sorted(indep & cur)
            pool = [j for j in rec["keep_cur"] if j not in indep]
            repaired = in_cache + pool[:len(out)]
            repairs.append({"episode_key": rec["episode_key"],
                            "frame_index": rec["frame_index"],
                            "image_path": rec.get("image_path", ""),
                            "arm": "independent_repaired",
                            "k_h": rec["k_h"], "keep": sorted(repaired)})

    if not viol:
        raise SystemExit("probe dump is empty; run the harness with "
                         "GEOPACT_INDEPENDENT_PROBE=1 first")

    rng = random.Random(SEED)
    n = len(viol)
    rates = sorted(100.0 * sum(viol[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(N_BOOT))
    out = {
        "frames": n,
        "violation_rate": round(100.0 * sum(viol) / n, 2),
        "violation_ci95_frame": [round(rates[int(0.025 * N_BOOT)], 2),
                                 round(rates[int(0.975 * N_BOOT) - 1], 2)],
        "mean_out_of_cache_tokens": round(sum(n_out) / n, 2),
        "mean_jaccard_nested_vs_independent": round(sum(jac) / n, 3),
    }
    print(tag, json.dumps(out))

    if repair_out is not None:
        with open(repair_out, "w", encoding="utf-8") as f:
            for r in repairs:
                f.write(json.dumps(r) + "\n")
        print(f"repaired independent keeps -> {repair_out} "
              f"({len(repairs)} frames)")

    agg_path = "docs/w43_independent_nested.json"
    agg = {}
    if os.path.exists(agg_path):
        agg = json.load(open(agg_path, encoding="utf-8"))
    agg[tag] = out
    node = os.uname().nodename if hasattr(os, "uname") else "local"
    tmp = agg_path + f".tmp.{node}.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)
    os.replace(tmp, agg_path)
    print("W43_INDEP_NESTED_DONE")


if __name__ == "__main__":
    main()
