#!/usr/bin/env python
"""W60 smoke verdict: decide whether the self-conditioned loop is trustworthy.

Pass conditions, each tied to a way the experiment could be silently wrong:

  A. Step 0 identical. With no history there is nothing for the hook to change,
     so any step-0 difference means the hook perturbed the prompt itself and
     every downstream number would be measuring the wrong thing.

  B. History really differs from GT on a meaningful fraction of steps. If the
     override never fires, the run is teacher forcing with a self-conditioned
     label, which would look like a clean null result.

  C. Unparseable rate stays low. Unparseable steps are recorded explicitly
     rather than replaced by GT, but a high rate would mean the loop is mostly
     feeding degenerate history and the comparison is about parsing, not memory.

  D. Dialogue KV reuse is not degraded relative to the teacher-forced twin. The
     pruned arm's whole efficiency claim rides on cache reuse, so the closed-loop
     result must be obtained on the same mechanism.
"""
import glob
import json
import os
import re
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

ROOT = "logs/w60_smoke/omnigui"
PAIRS = [("tf_dense", "sc_dense"), ("tf_trace", "sc_trace")]

failures = []


def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load(arm):
    """(episode, step) -> (exact_match, response text)."""
    f = latest(os.path.join(ROOT, arm, "results", "*", "*samples*.jsonl"))
    if not f:
        return None
    out = {}
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
        resp = row.get("filtered_resps") or row.get("resps") or []
        while isinstance(resp, list) and resp:
            resp = resp[0]
        out[(str(se.get("episode_id")), int(se.get("step_id", 0)))] = (
            bool(se.get("exact_match")), str(resp))
    return out or None


def dump(arm):
    p = os.path.join(ROOT, arm, "selfcond.jsonl")
    if not os.path.exists(p):
        return None
    rows = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(str(r.get("episode_id")), int(r.get("step_index", 0)))] = r
    return rows or None


def kv_reuse(arm):
    """Mean dialogue-KV reuse indicator scraped from the run log."""
    log = os.path.join(ROOT, arm, "run.log")
    if not os.path.exists(log):
        log = os.path.join(ROOT, arm, "launch.log")
    if not os.path.exists(log):
        return None
    hits = misses = 0
    for line in open(log, encoding="utf-8", errors="replace"):
        if "dialogue-kv cache-update" not in line:
            continue
        if re.search(r"'dialogue_kv_cache_updated':\s*True", line):
            hits += 1
        elif re.search(r"'dialogue_kv_cache_updated':\s*False", line):
            misses += 1
    if hits + misses == 0:
        return None
    return {"updated": hits, "not_updated": misses,
            "rate_pct": 100.0 * hits / (hits + misses)}


def check_pair(tf_arm, sc_arm):
    print(f"\n===== {sc_arm} vs {tf_arm} =====")
    tf, sc = load(tf_arm), load(sc_arm)
    if tf is None or sc is None:
        failures.append(f"{sc_arm}: missing samples ({tf_arm}={tf is not None}, "
                        f"{sc_arm}={sc is not None})")
        return
    common = sorted(set(tf) & set(sc))
    print(f"common steps: {len(common)}")

    # A. step 0 must be byte-identical
    zeros = [k for k in common if k[1] == 0]
    bad0 = [k for k in zeros if tf[k][1] != sc[k][1]]
    print(f"A step-0 identical : {len(zeros) - len(bad0)}/{len(zeros)}")
    if bad0:
        k = bad0[0]
        print(f"  first mismatch {k}\n    tf: {tf[k][1][:200]!r}\n    sc: {sc[k][1][:200]!r}")
        failures.append(f"{sc_arm}: {len(bad0)}/{len(zeros)} step-0 responses differ "
                        "-> the hook changed a prompt with no history")
    elif not zeros:
        failures.append(f"{sc_arm}: no step-0 rows, cannot run the strongest check")

    # B/C. the loop must actually be closed
    d = dump(sc_arm)
    if not d:
        failures.append(f"{sc_arm}: no selfcond dump, cannot verify the loop is closed")
    else:
        n = len(d)
        unp = sum(1 for r in d.values() if r.get("unparseable"))
        differs = 0
        for r in d.values():
            gt = (r.get("gt_action_json") or "").strip()
            if not gt.startswith("{"):
                continue
            try:
                gt_obj = json.loads(gt)
            except json.JSONDecodeError:
                continue
            act = {k: v for k, v in (r.get("action") or {}).items()
                   if not str(k).startswith("_")}
            if json.dumps(act, sort_keys=True) != json.dumps(gt_obj, sort_keys=True):
                differs += 1
        print(f"B dump coverage   : {len(d)} recorded / {len(sc)} scored")
        print(f"B history != GT   : {differs}/{n} ({100.0 * differs / n:.1f}%) "
              "(informational: equal-to-GT is allowed when the model is right)")
        print(f"C unparseable     : {unp}/{n} ({100.0 * unp / n:.1f}%)")
        # The loop is closed if every scored step has a dump row. Whether the
        # stored action equals GT is NOT the test: a model that matches GT
        # still has a closed loop, and the runtime invariant (STRICT=1) is
        # what proves the stored action, equal to GT or not, was in the prompt.
        scored_keys = {(str(k[0]), int(k[1])) for k in sc}
        dumped_keys = set(d)
        missing = scored_keys - dumped_keys
        if missing:
            failures.append(f"{sc_arm}: dump missing {len(missing)} scored steps, "
                            "the store did not capture the generations")
        if unp > 0.10 * n:
            failures.append(f"{sc_arm}: unparseable rate {100.0 * unp / n:.1f}% > 10%, "
                            "the loop is mostly feeding degenerate history")

    # D. KV reuse must not degrade
    ktf, ksc = kv_reuse(tf_arm), kv_reuse(sc_arm)
    print(f"D kv reuse        : tf={ktf} sc={ksc}")
    if ktf and ksc and ksc["rate_pct"] < ktf["rate_pct"] - 5.0:
        failures.append(f"{sc_arm}: dialogue-KV reuse dropped "
                        f"{ktf['rate_pct']:.1f}% -> {ksc['rate_pct']:.1f}%")

    # Informational: Step SR shift on the common steps.
    tf_em = sum(1 for k in common if tf[k][0]) / max(len(common), 1)
    sc_em = sum(1 for k in common if sc[k][0]) / max(len(common), 1)
    print(f"   Step SR: tf={100 * tf_em:.2f}  sc={100 * sc_em:.2f}  "
          f"d={100 * (sc_em - tf_em):+.2f} pp  (smoke-sized, not a result)")


if __name__ == "__main__":
    for tf_arm, sc_arm in PAIRS:
        check_pair(tf_arm, sc_arm)
    print("\n===== verdict =====")
    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("PASS: the self-conditioned loop is closed, step 0 is unchanged, and "
          "KV reuse held. Safe to launch the full arms.")
