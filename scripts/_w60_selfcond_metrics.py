#!/usr/bin/env python
"""W60: metrics for the self-conditioned (student-forcing) closed-loop runs.

Reads the OmniGUI sample dumps of the W60 arms plus their teacher-forced twins
and emits docs/w60_selfcond.json + a stdout ledger.

What it reports, and why each piece is needed
---------------------------------------------
Tier i  SR / GP invariance. Success rate needs every step right, goal progress
        counts only the prefix before the first failure. On an all-correct
        prefix the self-conditioned history is semantically the ground-truth
        history, so SR and GP should barely move while Step SR is the quantity
        that can shift. That asymmetry is a falsifiable prediction, so it is
        measured before it is used as an argument.

Tier ii Step SR under self-conditioning, teacher-forced next to self-conditioned,
        and the paired dense-minus-TRACE difference. The paired difference is
        the headline: both arms replay the same ground-truth frames and so
        inherit the identical stale-frame artifact, which cancels in the
        difference but not in either absolute number.

Tier iii Recovery rate R = P(step t+1 correct | history carries a wrong action at
        step t) and dR = R_dense - R_TRACE. An interval on dR that contains zero
        is the positive result: pruning did not remove the evidence needed to
        recover.

Inference: bootstrap resampling clusters on the EPISODE, never the step. Steps
inside an episode are dependent by construction under self-conditioning, since
step t's history is built from step t-1's output, so a step-level bootstrap
would understate the variance.
"""
import glob
import json
import os
import random
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

N_BOOT = 10000
OUT_JSON = "artifacts/metrics/w60_selfcond.json"

SC = "logs/w60_selfcond/omnigui"
TF = {  # teacher-forced twins, same selector settings, from the main tables
    "dense": "logs/w57b_unified/omnigui/baseline_full",
    "trace_c50_h10": "logs/w57b_acc/omnigui/final_c50_h10",
    "trace_c25_h05": "logs/w57b_acc/omnigui/a2cap_c25_h05",
}
ARMS = {
    "dense": f"{SC}/sc_dense",
    "trace_c50_h10": f"{SC}/sc_trace_c50_h10",
    "trace_c25_h05": f"{SC}/sc_trace_c25_h05",
}
INJ = {
    "dense": f"{SC}/inj_dense",
    "trace_c50_h10": f"{SC}/inj_trace_c50_h10",
}


# ---------------------------------------------------------------- loaders
def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_steps(arm_dir):
    """(episode, step) -> per-step eval record for one arm."""
    f = latest(os.path.join(arm_dir, "results", "*", "*samples*.jsonl"))
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
        ep = f"{se.get('subset', '?')}-{se.get('episode_id', '?')}"
        try:
            st = int(se.get("step_id", 0))
        except (TypeError, ValueError):
            continue
        out[(ep, st)] = {
            "em": 1.0 if se.get("exact_match") else 0.0,
            "tm": 1.0 if se.get("type_match") else 0.0,
            "pd": se.get("pd_action_type"),
            "gt": se.get("gt_action_type"),
        }
    return out or None


def load_selfcond_dump(arm_dir):
    """(short_episode_id, step) -> audit row written by the self-conditioning store.

    Keyed on the SHORT episode id so the join with step_eval records does not
    depend on whether the dump's subset prefix was `omnigui` or `unknown`.
    """
    path = os.path.join(arm_dir, "selfcond.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = str(row.get("episode_id") or "")
        if not ep:
            ek = str(row.get("episode_key") or "")
            ep = ek.split("-", 1)[-1] if ek else ""
        if not ep:
            continue
        out[(ep, int(row.get("step_index", 0)))] = row
    return out or None


def short_ep(ep):
    """`subset-EPID` -> `EPID`. Episode ids themselves may contain hyphens."""
    s = str(ep or "")
    return s.split("-", 1)[-1] if "-" in s else s


def em_map(steps):
    return {k: v["em"] for k, v in steps.items()}


# ---------------------------------------------------------------- inference
def paired_clustered(a, b, seed=7):
    """Step-paired difference with an EPISODE-clustered bootstrap.

    Clustering is not optional here. Under self-conditioning step t's prompt is
    built from step t-1's output, so steps within an episode are dependent by
    construction and a step-level bootstrap would report intervals that are too
    narrow. Resampling whole episodes keeps that dependence intact.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    eps = defaultdict(list)
    for k in keys:
        eps[k[0]].append(a[k] - b[k])
    groups = list(eps.values())
    m, n = len(groups), len(keys)
    obs = sum(d for g in groups for d in g) / n
    rb, rs = random.Random(seed), random.Random(seed + 1)
    extreme = 0
    means = []
    for _ in range(N_BOOT):
        s = c = 0.0
        for _ in range(m):
            g = groups[rb.randrange(m)]
            s += sum(g)
            c += len(g)
        means.append(s / c if c else 0.0)
        ns = 0.0
        for g in groups:
            gs = sum(g)
            ns += gs if rs.getrandbits(1) else -gs
        if abs(ns / n) >= abs(obs):
            extreme += 1
    means.sort()
    return {
        "delta_pp": 100.0 * obs,
        "p": (extreme + 1.0) / (N_BOOT + 1.0),
        "n_steps": n,
        "n_episodes": m,
        "ci95_pp": [100.0 * means[int(0.025 * N_BOOT)],
                    100.0 * means[int(0.975 * N_BOOT) - 1]],
    }


def rate_clustered(units, seed=11):
    """Rate over units ``[(episode, 0/1)]`` with an episode-clustered CI."""
    if not units:
        return None
    eps = defaultdict(list)
    for ep, v in units:
        eps[ep].append(float(v))
    groups = list(eps.values())
    m = len(groups)
    n = sum(len(g) for g in groups)
    obs = sum(sum(g) for g in groups) / n
    rb = random.Random(seed)
    means = []
    for _ in range(N_BOOT):
        s = c = 0.0
        for _ in range(m):
            g = groups[rb.randrange(m)]
            s += sum(g)
            c += len(g)
        means.append(s / c if c else 0.0)
    means.sort()
    return {
        "rate_pct": 100.0 * obs,
        "n": n,
        "n_episodes": m,
        "ci95_pct": [100.0 * means[int(0.025 * N_BOOT)],
                     100.0 * means[int(0.975 * N_BOOT) - 1]],
    }


def diff_of_rates(units_a, units_b, seed=13):
    """R_a - R_b, paired on the episode and resampled on episodes.

    Pairing matters: the recovery denominator is whichever steps happened to be
    contaminated, so an unpaired difference would mix arm effects with which
    episodes each arm drew.
    """
    ga, gb = defaultdict(list), defaultdict(list)
    for ep, v in units_a or []:
        ga[ep].append(float(v))
    for ep, v in units_b or []:
        gb[ep].append(float(v))
    shared = sorted(set(ga) & set(gb))
    if not shared:
        return None
    ra = sum(sum(ga[e]) for e in shared) / sum(len(ga[e]) for e in shared)
    rbate = sum(sum(gb[e]) for e in shared) / sum(len(gb[e]) for e in shared)
    obs = ra - rbate
    rb = random.Random(seed)
    m = len(shared)
    diffs = []
    for _ in range(N_BOOT):
        sa = ca = sb = cb = 0.0
        for _ in range(m):
            e = shared[rb.randrange(m)]
            sa += sum(ga[e]); ca += len(ga[e])
            sb += sum(gb[e]); cb += len(gb[e])
        if ca and cb:
            diffs.append(sa / ca - sb / cb)
    diffs.sort()
    k = len(diffs)
    return {
        "delta_pp": 100.0 * obs,
        "r_a_pct": 100.0 * ra,
        "r_b_pct": 100.0 * rbate,
        "n_episodes": m,
        "ci95_pp": [100.0 * diffs[int(0.025 * k)], 100.0 * diffs[int(0.975 * k) - 1]],
    }


# ---------------------------------------------------------------- Tier i / ii
def by_episode(steps):
    out = defaultdict(dict)
    for (ep, st), v in steps.items():
        out[ep][st] = v
    return out


def episode_metrics(steps):
    """Step SR, episode SR and goal progress, matching the harness definitions.

    SR needs every step of the episode exact-matched; GP counts the fraction of
    the leading prefix that is correct. Recomputing them here rather than reading
    the harness aggregate keeps the teacher-forced and self-conditioned numbers
    on one identical episode set even when an arm is missing steps.
    """
    eps = by_episode(steps)
    step_sr = 100.0 * sum(v["em"] for v in steps.values()) / len(steps)
    srs, gps = [], []
    for ep, m in eps.items():
        order = sorted(m)
        ems = [m[s]["em"] for s in order]
        srs.append(1.0 if all(e > 0.5 for e in ems) else 0.0)
        pref = 0
        for e in ems:
            if e < 0.5:
                break
            pref += 1
        gps.append(pref / len(ems))
    return {
        "step_sr_pct": step_sr,
        "sr_pct": 100.0 * sum(srs) / len(srs),
        "gp_pct": 100.0 * sum(gps) / len(gps),
        "n_steps": len(steps),
        "n_episodes": len(eps),
    }


def divergence_onset(steps):
    """Per episode, the first step whose predicted action type differs from GT.

    Type-level rather than exact-match, because that is the step where the
    self-conditioned history genuinely departs from the teacher-forced one: a
    coordinate miss inside the right action type leaves the history's action
    semantics intact.
    """
    onsets = {}
    for ep, m in by_episode(steps).items():
        first = None
        for st in sorted(m):
            v = m[st]
            if v["pd"] is None or str(v["pd"]) != str(v["gt"]):
                first = st
                break
        onsets[ep] = first
    hist = defaultdict(int)
    for v in onsets.values():
        hist["none" if v is None else str(min(int(v), 5))] += 1
    n = len(onsets)
    return {
        "n_episodes": n,
        "never_diverges_pct": 100.0 * sum(1 for v in onsets.values() if v is None) / n,
        "onset_hist": dict(sorted(hist.items())),
        "onsets": onsets,
    }


def attribution(sc_steps, tf_steps):
    """Split self-conditioned errors into 'would have been wrong anyway' vs
    'wrong only after the history changed'.

    Step 0 is excluded: it has no history, so a disagreement there is not a
    closed-loop effect. The remainder is the quantity the protocol exists to
    measure, and reporting only the pooled Step-SR shift would hide whether the
    movement is a few recovered steps or a few newly-broken ones.
    """
    keys = sorted(set(sc_steps) & set(tf_steps))
    keys = [k for k in keys if k[1] > 0]
    n = len(keys)
    if n == 0:
        return None
    sc_wrong_tf_wrong = sc_wrong_tf_ok = sc_ok_tf_wrong = sc_ok_tf_ok = 0
    for k in keys:
        sc_ok = sc_steps[k]["em"] > 0.5
        tf_ok = tf_steps[k]["em"] > 0.5
        if sc_ok and tf_ok:
            sc_ok_tf_ok += 1
        elif (not sc_ok) and (not tf_ok):
            sc_wrong_tf_wrong += 1
        elif (not sc_ok) and tf_ok:
            sc_wrong_tf_ok += 1
        else:
            sc_ok_tf_wrong += 1
    return {
        "n_steps_with_history": n,
        "both_correct_pct": 100.0 * sc_ok_tf_ok / n,
        "would_be_wrong_anyway_pct": 100.0 * sc_wrong_tf_wrong / n,
        "broken_by_history_pct": 100.0 * sc_wrong_tf_ok / n,
        "recovered_under_selfcond_pct": 100.0 * sc_ok_tf_wrong / n,
        "n_broken_by_history": sc_wrong_tf_ok,
        "n_recovered": sc_ok_tf_wrong,
    }


# ---------------------------------------------------------------- Tier iii
def recovery_units(steps, dump, contaminated_only=True):
    """Units ``[(episode, recovered)]`` for step t+1 after a wrong action at t.

    "Contaminated at t" is read from the self-conditioning audit dump, not from
    the eval record, so the injected arms measure recovery from the action that
    was actually forced into the history rather than from whatever the model
    happened to say.
    """
    units = []
    eps = by_episode(steps)
    for ep, m in eps.items():
        for st in sorted(m):
            nxt = m.get(st + 1)
            if nxt is None:
                continue
            row = (dump or {}).get((short_ep(ep), st))
            if contaminated_only:
                if row is None or not row.get("injected"):
                    continue
            else:
                # Observational fallback when no injection file was used: the
                # history is contaminated wherever the model's own action type
                # already diverged from the ground truth.
                v = m[st]
                if v["pd"] is not None and str(v["pd"]) == str(v["gt"]):
                    continue
            units.append((ep, nxt["em"]))
    return units


def _fmt(d, keys=("delta_pp", "ci95_pp", "p")):
    if not d:
        return "n/a"
    lo, hi = d.get("ci95_pp", d.get("ci95_pct", [float("nan")] * 2))
    v = d.get("delta_pp", d.get("rate_pct"))
    s = f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    if "p" in d:
        s += f"  p={d['p']:.3f}"
    if "n_episodes" in d:
        s += f"  (eps={d['n_episodes']})"
    return s


def print_ledger(rep):
    print("\n===== W60 self-conditioned closed loop: OmniGUI =====")
    print("\n-- audit: is the loop actually closed --")
    for arm, a in (rep.get("audit") or {}).items():
        if a.get("dump") == "missing":
            print(f"  {arm:16s} dump MISSING (cannot verify self-conditioning)")
            continue
        print(f"  {arm:16s} steps={a['recorded_steps']:5d} "
              f"unparseable={a['unparseable_pct']:.2f}% "
              f"hist!=GT={a['history_differs_from_gt_pct']:.1f}% "
              f"injected={a['injected_steps']}")

    print("\n-- Tier i: SR/GP invariance (self-cond minus teacher-forced, pp) --")
    for arm, t in (rep.get("tier_i") or {}).items():
        print(f"  {arm:16s} Step SR {t['self_cond']['step_sr_pct']:.2f} vs "
              f"{t['teacher_forced']['step_sr_pct']:.2f}  "
              f"dStepSR={t['d_step_sr_pp']:+.2f}  dSR={t['d_sr_pp']:+.2f}  "
              f"dGP={t['d_gp_pp']:+.2f}")
        print(f"  {'':16s} paired dStepSR {_fmt(t['paired_step_sr'])}")
        a = t.get("attribution") or {}
        if a:
            print(f"  {'':16s} of steps with history: "
                  f"broken-by-hist={a['broken_by_history_pct']:.2f}% "
                  f"({a['n_broken_by_history']})  "
                  f"recovered={a['recovered_under_selfcond_pct']:.2f}% "
                  f"({a['n_recovered']})  "
                  f"wrong-anyway={a['would_be_wrong_anyway_pct']:.2f}%")

    print("\n-- Tier ii: Step SR per arm, and paired dense minus TRACE --")
    for arm, m in ((rep.get("tier_ii") or {}).get("arms") or {}).items():
        print(f"  {arm:16s} StepSR={m['step_sr_pct']:.2f}  SR={m['sr_pct']:.2f}  "
              f"GP={m['gp_pct']:.2f}  steps={m['n_steps']}  eps={m['n_episodes']}")
    for k, d in (rep.get("tier_ii") or {}).items():
        if not k.startswith("dense_minus_"):
            continue
        print(f"  {k}")
        print(f"    self-cond      {_fmt(d.get('self_cond'))}")
        print(f"    teacher-forced {_fmt(d.get('teacher_forced'))}")
    for arm, d in ((rep.get("tier_ii") or {}).get("divergence") or {}).items():
        print(f"  {arm:16s} never diverges {d['never_diverges_pct']:.1f}%  "
              f"onset hist {d['onset_hist']}")

    print("\n-- Tier iii: recovery after a contaminated history step --")
    for arm, d in ((rep.get("tier_iii") or {}).get("recovery") or {}).items():
        base = ((rep.get("tier_iii") or {}).get("clean_baseline") or {}).get(arm)
        print(f"  {arm:16s} R={_fmt(d)}"
              + (f"   clean baseline={_fmt(base)}" if base else ""))
    dR = (rep.get("tier_iii") or {}).get("delta_R_dense_minus_trace_c50_h10")
    if dR:
        print(f"  dR (dense - TRACE) = {dR['delta_pp']:+.2f} pp "
              f"[{dR['ci95_pp'][0]:+.2f}, {dR['ci95_pp'][1]:+.2f}]  "
              f"(R_dense={dR['r_a_pct']:.2f}, R_trace={dR['r_b_pct']:.2f})")
        if dR["ci95_pp"][0] <= 0.0 <= dR["ci95_pp"][1]:
            print("  -> interval contains 0: no detectable loss of recoverability "
                  "from pruning")


def main():
    report = {"tier_i": {}, "tier_ii": {}, "tier_iii": {}, "audit": {}}
    sc_steps, tf_steps, dumps = {}, {}, {}
    for arm, d in ARMS.items():
        s = load_steps(d)
        if s is None:
            print(f"MISSING self-cond arm: {d}")
            continue
        sc_steps[arm] = s
        dumps[arm] = load_selfcond_dump(d)
        t = load_steps(TF[arm])
        if t is None:
            print(f"MISSING teacher-forced twin: {TF[arm]}")
        else:
            tf_steps[arm] = t
    if not sc_steps:
        print("no self-conditioned arms found; run scripts/xds_jobs/w60_selfcond.tsv first")
        return

    # Audit: the run must actually be self-conditioned, and the fraction of
    # unparseable generations bounds how often the loop degraded.
    for arm, dmp in dumps.items():
        if not dmp:
            report["audit"][arm] = {"dump": "missing"}
            continue
        n = len(dmp)
        report["audit"][arm] = {
            "recorded_steps": n,
            "unparseable_pct": 100.0 * sum(1 for r in dmp.values() if r.get("unparseable")) / n,
            "injected_steps": sum(1 for r in dmp.values() if r.get("injected")),
            "history_differs_from_gt_pct": 100.0 * sum(
                1 for r in dmp.values()
                if r.get("gt_action_json") and json.dumps(
                    {k: v for k, v in (r.get("action") or {}).items()
                     if not str(k).startswith("_")},
                    sort_keys=True) != json.dumps(
                    json.loads(r["gt_action_json"]) if r["gt_action_json"].strip().startswith("{")
                    else {}, sort_keys=True)
            ) / n,
        }

    # Tier i: SR / GP should barely move, Step SR is the quantity that shifts.
    for arm in sc_steps:
        if arm not in tf_steps:
            continue
        # Common step set only, so the comparison is not confounded by coverage.
        common = sorted(set(sc_steps[arm]) & set(tf_steps[arm]))
        sc_c = {k: sc_steps[arm][k] for k in common}
        tf_c = {k: tf_steps[arm][k] for k in common}
        msc, mtf = episode_metrics(sc_c), episode_metrics(tf_c)
        report["tier_i"][arm] = {
            "self_cond": msc,
            "teacher_forced": mtf,
            "d_step_sr_pp": msc["step_sr_pct"] - mtf["step_sr_pct"],
            "d_sr_pp": msc["sr_pct"] - mtf["sr_pct"],
            "d_gp_pp": msc["gp_pct"] - mtf["gp_pct"],
            "paired_step_sr": paired_clustered(em_map(sc_c), em_map(tf_c)),
            "attribution": attribution(sc_c, tf_c),
        }

    # Tier ii: dense vs TRACE under self-conditioning, paired on the step.
    for arm in ("trace_c50_h10", "trace_c25_h05"):
        if arm not in sc_steps or "dense" not in sc_steps:
            continue
        report["tier_ii"][f"dense_minus_{arm}"] = {
            "self_cond": paired_clustered(em_map(sc_steps["dense"]), em_map(sc_steps[arm])),
            "teacher_forced": (
                paired_clustered(em_map(tf_steps["dense"]), em_map(tf_steps[arm]))
                if "dense" in tf_steps and arm in tf_steps else None
            ),
        }
    for arm, s in sc_steps.items():
        report["tier_ii"].setdefault("arms", {})[arm] = episode_metrics(s)
        report["tier_ii"].setdefault("divergence", {})[arm] = {
            k: v for k, v in divergence_onset(s).items() if k != "onsets"
        }

    # Tier iii: recovery after a contaminated history step.
    inj_steps, inj_dumps = {}, {}
    for arm, d in INJ.items():
        s = load_steps(d)
        if s is None:
            print(f"MISSING injected arm: {d}")
            continue
        inj_steps[arm] = s
        inj_dumps[arm] = load_selfcond_dump(d)
    rec = {}
    for arm, s in inj_steps.items():
        u = recovery_units(s, inj_dumps.get(arm), contaminated_only=True)
        rec[arm] = rate_clustered(u)
        report["tier_iii"].setdefault("recovery", {})[arm] = rec[arm]
        # Clean twin on the same episodes: the baseline probability that step
        # t+1 is right when the history was NOT contaminated. Without it a low
        # recovery rate cannot be separated from a generally hard step.
        if arm in sc_steps:
            dmp = inj_dumps.get(arm) or {}
            u_clean = [
                (ep, sc_steps[arm][(ep, st)]["em"])
                for (ep, st) in sorted(sc_steps[arm])
                if (dmp.get((short_ep(ep), st - 1)) or {}).get("injected")
            ]
            report["tier_iii"].setdefault("clean_baseline", {})[arm] = rate_clustered(u_clean)
    if "dense" in inj_steps and "trace_c50_h10" in inj_steps:
        report["tier_iii"]["delta_R_dense_minus_trace_c50_h10"] = diff_of_rates(
            recovery_units(inj_steps["dense"], inj_dumps.get("dense")),
            recovery_units(inj_steps["trace_c50_h10"], inj_dumps.get("trace_c50_h10")),
        )

    os.makedirs("docs", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print_ledger(report)
    print(f"\nwrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    main()
