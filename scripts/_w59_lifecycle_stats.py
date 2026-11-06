#!/usr/bin/env python
"""W59: population lifecycle statistics over the whole W45 probe dump.

Lifecycle semantics recovered from the dump (all verified exactly, see the plan):
  step t admits frame f_t at the entry budget -> `keep_cur`, |S| = k_c
  when f_t becomes history it contracts ONCE  -> `keep_hist_nested`, |S| = k_h
  `keep_hist_nested == nap_order[:k_h]`, and `keep_hist_nested` is a subset of
  `keep_cur` on every record, so the write-once nesting contract holds.
  f_t leaves the cache entirely once t' - t > H (the history window).

The per-step ledger this produces is what makes an honest "admitted / survived /
retired" figure possible without any GPU rerun.
"""
import json
import os
import statistics
import sys

os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROBE = os.environ.get(
    "W59_PROBE", "logs/w45_probe/omni_ours_c50_h10_probe.jsonl")
OUT = os.environ.get("W59_OUT", "logs/w59_viz")
HIST_N = int(os.environ.get("W59_HIST_N", "5"))
os.makedirs(OUT, exist_ok=True)


def load():
    by_ep = {}
    with open(PROBE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_ep.setdefault(rec.get("episode_key", ""), []).append(rec)
    return by_ep


def main():
    by_ep = load()
    per_step, retire_frac, indep_div, cov, strata = [], [], [], [], []
    n_viol = n_readmit = 0
    for ep, recs in by_ep.items():
        for t, r in enumerate(recs):
            kc = set(r["keep_cur"])
            kh = set(r["keep_hist_nested"])
            ki = set(r.get("keep_hist_independent") or [])
            n = int(r["n_frame"])
            n_viol += len(kh - kc)
            retire_frac.append(len(kc - kh) / max(1, len(kc)))
            if ki:
                indep_div.append(len(ki - kh) / max(1, len(kh)))
            au = r.get("audit", {})
            if au.get("miu_coverage_ratio") is not None:
                cov.append(float(au["miu_coverage_ratio"]))
            # Survival as a function of position in the committed admission
            # order. Sampled on a fine grid (percentiles of rank) because the
            # contract predicts a single step at k_h/k_c and a decile histogram
            # is too coarse to show where the step lands.
            order = r.get("nap_order") or []
            if order:
                nb = 50
                b = [0] * nb
                tot = [0] * nb
                for pos, tok in enumerate(order):
                    d = min(nb - 1, int(nb * pos / len(order)))
                    tot[d] += 1
                    if tok in kh:
                        b[d] += 1
                strata.append([b[i] / max(1, tot[i]) for i in range(nb)])
            per_step.append({
                "episode": ep, "t": t, "n_frame": n,
                "k_c": len(kc), "k_h": len(kh),
                "admitted": len(kc), "survived": len(kh),
                "retired": len(kc - kh),
                "in_cache_frames": min(t + 1, HIST_N + 1),
            })

    def m(xs):
        return round(statistics.mean(xs), 4) if xs else None

    out = {
        "probe": PROBE, "n_episodes": len(by_ep), "n_records": len(per_step),
        "hist_window": HIST_N,
        "nesting_violations": n_viol, "readmissions": n_readmit,
        "mean_retired_frac_of_admitted": m(retire_frac),
        "mean_independent_divergence": m(indep_div),
        "mean_miu_coverage_ratio": m(cov),
        "rank_survival_bins": len(strata[0]) if strata else 0,
        "rank_survival_curve": [round(x, 4) for x in map(
            statistics.mean, zip(*strata))] if strata else [],
        "mean_kh_over_kc": m([len(r["keep_hist_nested"]) /
                              max(1, len(r["keep_cur"]))
                              for rs in by_ep.values() for r in rs]),
        "steps_hist": {},
        "per_step": per_step,
    }
    from collections import Counter
    out["steps_hist"] = {str(k): v for k, v in sorted(
        Counter(len(v) for v in by_ep.values()).items())}
    json.dump(out, open(f"{OUT}/lifecycle_stats.json", "w", encoding="utf-8"),
              indent=1)
    for k, v in out.items():
        if k != "per_step":
            print(k, "=", v if not isinstance(v, dict) else json.dumps(v))
    print("W59_LIFECYCLE_STATS_DONE")


if __name__ == "__main__":
    sys.exit(main())
