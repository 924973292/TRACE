"""Aggregate UI-TARS-1.5-7B cross-backbone results (W45 Phase C).

Reads the C1 single-step and C2 multi-step run dirs on the remote pod,
computes per-arm scores and retention (percent of own dense), and writes
docs/w45_uitars_results.json. Retention is the pre-declared primary
comparison (docs/w45_uitars_config_ledger.json); absolute cross-family
scores are never compared directly.

Single-step reads `logs/w45_uitars_single_native` (the 2026-07-31 rerun
under the official bytedance grounding prompt). The earlier
`logs/w45_uitars_single` matrix used the GUI-Owl computer_use JSON
protocol, which is out-of-distribution for UI-TARS and drove a role-header
degeneration loop; it is retained on disk as a diagnostic only and is
reported under `single_step_ood_diagnostic` when present.

Run remotely: bash scripts/_w45_run_uitars_aggregate.sh
"""
import glob
import json
import os
import sys
import re

SINGLE_ROOT = "logs/w45_uitars_single_native"
SINGLE_OOD_ROOT = "logs/w45_uitars_single"
# W46: the multi-step matrix was re-run under UITARS_NATIVE_MULTISTEP=1 after the
# stage-B gate showed the GUI-Owl teacher-forced prompt drives the backbone to
# emit a deliberation and then stop with no action on 98% of steps, which scored a
# flat Step SR of 0 and produced meaningless retention ratios (up to 1000%). The
# old tree stays on disk as an OOD diagnostic and is NOT aggregated.
MULTI_ROOT = "logs/w46_uitars_multi"
MULTI_OOD_ROOT = "logs/w45_uitars_multi"
# W46: supplementary baseline arms live in their own tree so the original
# native matrix stays immutable; both are searched, first hit wins.
SINGLE_SUPP_ROOT = "logs/w46_uitars_supp"
# W46 corrective grid. The w45 `witness_*` arms were built without
# GEOPACT_BAM_MASS_ALPHA, GEOPACT_CRS_NESTED or GEOPACT_RECTIFY_PREFETCH, so the
# layout prior entered at the 1.0 default rather than the specified 2.0: those
# arms ran a partial method. `trace_*` here carries the complete configuration and
# is what the published column must report. `witness_*` is retained only as the
# configuration comparison in the appendix.
SINGLE_FIX_ROOT = "logs/w46_uitars_fix"
# W47: the two wide budgets (50%/25%), added because every arm lands in the single
# digits at 10%/5% on this carrier, which cannot separate methods. Same complete
# configuration as SINGLE_FIX_ROOT, so all four budgets share one method
# definition.
SINGLE_WIDE_ROOT = "logs/w47_uitars_wide"
# W47 calibration wave. The wide and fix trees both ran ONE transplanted 8B
# operating point (alpha 2.0, default coverage dose 0.10). The 8B protocol
# instead selects the prior cap and the coverage dose per benchmark-budget grid
# point from a declared public grid, so this tree sweeps that grid on the cells
# where the transplanted point is not the selected one. Adopted cells are named
# in SINGLE_DIR_ALIASES below and take precedence over the fixed-point trees.
SINGLE_CAL_ROOT = "logs/w47_uitars_cal"

# Search order for a single-step cell. First hit wins, so the more specific trees
# must precede the original matrix.
SINGLE_SEARCH_ROOTS = (SINGLE_CAL_ROOT, SINGLE_WIDE_ROOT,
                       SINGLE_FIX_ROOT, SINGLE_SUPP_ROOT)

# The corrective grid swept prior strength, so its directories carry an `_a<N>`
# suffix (trace_r10_a2, trace_r10_a4). Map the budget-only arm name onto the
# directory holding the operating point selected for that benchmark-budget cell.
# The wide grid needs no alias because it was launched at one point only.
#
# SINGLE_CAL_ROOT holds the per-cell grid sweep. Its entries are written by
# scripts/_w47_cal_select.py once a cell's sweep finishes, so a cell appears here
# only when the swept point beats the transplanted one. Cells absent from this
# map fall through to the fixed-point trees, which keeps every published number
# traceable to exactly one run dir.
SINGLE_DIR_ALIASES = {
    SINGLE_FIX_ROOT: {
        "trace_r10": "trace_r10_a2",
        "trace_r05": "trace_r05_a2",
    },
    SINGLE_CAL_ROOT: {},
}

# Per-benchmark selected calibration cells, keyed as (bench, arm) -> dirname.
# Loaded from docs/w47_uitars_cal_selected.json when present so the selection is
# recorded as data rather than edited into code by hand.
_CAL_SELECTION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs",
    "w47_uitars_cal_selected.json")


def _load_cal_selection() -> dict:
    """(bench -> {arm: dirname}) for the adopted calibration cells."""
    try:
        with open(_CAL_SELECTION_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    return {b: dict(v) for b, v in raw.items() if isinstance(v, dict)}


_CAL_SELECTION = _load_cal_selection()


def _resolve_dir(root: str, arm: str, bench: str = "") -> str:
    """Directory name for `arm` under `root`, honouring per-tree aliases.

    The calibration tree resolves per benchmark, since the operating point is
    selected per benchmark-budget grid point. An unselected cell returns a name
    that does not exist, so the caller falls through to the fixed-point trees.
    """
    if root == SINGLE_CAL_ROOT:
        return _CAL_SELECTION.get(bench, {}).get(arm, arm)
    return SINGLE_DIR_ALIASES.get(root, {}).get(arm, arm)

SINGLE_METRIC_KEYS = {
    "screenspot_v2": "screenspot_v2_center_acc,none",
    "screenspot_pro": "screenspot_pro_center_acc,none",
    "mmbench_l2": None,  # resolved from results.json keys at read time
}

# Sub-metrics the main-table layout needs (Text/Icon for the ScreenSpot suites,
# Basic/Advanced for MMBench-GUI L2). UI-TARS results.json carries only the
# pooled accuracy, so the two columns are recomputed from the per-sample dumps,
# where each metric field is a dict carrying `hit` plus the subgroup label.
#   bench -> (metric_field, subgroup_key, (label_col1, label_col2))
SINGLE_SUB_SPEC = {
    "screenspot_v2": ("screenspot_v2_center_acc", "data_type", ("text", "icon")),
    "screenspot_pro": ("screenspot_pro_center_acc", "ui_type", ("text", "icon")),
    "mmbench_l2": ("mmbench_gui_l2_acc", "grounding_type", ("basic", "advanced")),
}

SINGLE_ARMS = [
    "dense",
    # Published column: the complete-configuration method arm at all four
    # budgets. `witness_*` below is the partial-configuration predecessor, kept
    # for the appendix comparison only.
    "trace_r50", "trace_r25", "trace_r10", "trace_r05",
    "witness_r10", "witness_r05",
    "wolmp_r10", "wolmp_r05", "woawc_r10", "woawc_r05",
    # Baselines at all four budgets. The wide pair comes from W47, the tight pair
    # from the W45 native matrix and the W46 supplementary tree.
    "random_r50", "random_r25", "random_r10", "random_r05",
    "uniform_r50", "uniform_r25", "uniform_r10", "uniform_r05",
    "divprune_r50", "divprune_r25", "divprune_r10", "divprune_r05",
    "cdpruner_r50", "cdpruner_r25", "cdpruner_r10", "cdpruner_r05",
    "trim_r50", "trim_r25", "trim_r10", "trim_r05",
    "prunesid_r50", "prunesid_r25", "prunesid_r10", "prunesid_r05",
]
# W46 multi-step matrix arm names follow scripts/xds_jobs/w46_uitars_multi.tsv.
MULTI_ARMS = [
    "dense",
    "trace_c50_h10", "trace_c25_h05",
    "uniform_c50_h10", "uniform_c25_h05",
    "random_c50_h10", "random_c25_h05",
    "divprune_c50_h10", "divprune_c25_h05",
    "cdpruner_c50_h10", "cdpruner_c25_h05",
    "prunesid_c50_h10", "prunesid_c25_h05",
]


def _single_metrics(root: str, bench: str, arm: str):
    """Return the results.json metric dict for one cell, or None."""
    hits = sorted(glob.glob(f"{root}/{bench}/{arm}/results/*/*results.json"))
    if not hits:
        return None
    try:
        with open(hits[-1], encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    for _task, metrics in (d.get("results") or {}).items():
        if isinstance(metrics, dict):
            return metrics
    return None


def read_single(root: str, bench: str, arm: str):
    metrics = _single_metrics(root, bench, arm)
    if metrics is None and root == SINGLE_ROOT:
        for alt in SINGLE_SEARCH_ROOTS:
            metrics = _single_metrics(alt, bench, _resolve_dir(alt, arm, bench))
            if metrics is not None:
                break
    if not metrics:
        return None
    want = SINGLE_METRIC_KEYS.get(bench)
    if want and want in metrics:
        return float(metrics[want])
    for k, v in metrics.items():
        if k.endswith("acc,none") and isinstance(v, (int, float)):
            return float(v)
    return None


def read_single_subs(root: str, bench: str, arm: str):
    """(col1, col2) subgroup accuracies recomputed from the per-sample dump."""
    spec = SINGLE_SUB_SPEC.get(bench)
    if not spec:
        return (None, None)
    field, key, labels = spec
    hits = sorted(glob.glob(f"{root}/{bench}/{arm}/results/*/*samples*.jsonl"))
    if not hits and root == SINGLE_ROOT:
        for alt in SINGLE_SEARCH_ROOTS:
            adir = _resolve_dir(alt, arm, bench)
            hits = sorted(
                glob.glob(f"{alt}/{bench}/{adir}/results/*/*samples*.jsonl"))
            if hits:
                break
    if not hits:
        return (None, None)
    tally = {lab: [0, 0] for lab in labels}
    try:
        with open(hits[-1], encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                cell = rec.get(field)
                if not isinstance(cell, dict):
                    continue
                lab = str(cell.get(key, "")).strip().lower()
                if lab not in tally:
                    continue
                tally[lab][1] += 1
                tally[lab][0] += 1 if cell.get("hit") else 0
    except OSError:
        return (None, None)
    # Subgroup accuracies are recomputed from a dump the harness APPENDS to while
    # the run is in flight, so reading one mid-run yields a number that is legal,
    # plausible and wrong -- trim_r05/mmbench advanced was recorded as 0.00 from a
    # partial dump whose advanced bucket had not yet seen a hit, against a true
    # 1.66. Cross-check the row count against the pooled accuracy the finished
    # results.json reports: if the dump does not reproduce it, the dump is not the
    # same population and the subgroups are withheld rather than published stale.
    total_rows = sum(t[1] for t in tally.values())
    total_hits = sum(t[0] for t in tally.values())
    pooled = read_single(root, bench, arm)
    if pooled is not None and total_rows:
        pooled_pct = pooled * 100.0 if abs(pooled) <= 1.0 else pooled
        dump_pct = 100.0 * total_hits / total_rows
        if abs(dump_pct - pooled_pct) > 0.05:
            print(f"[subs] SKIP {bench}/{arm}: dump pools to {dump_pct:.2f}% but "
                  f"results.json reports {pooled_pct:.2f}% -- partial dump")
            return (None, None)
    out = []
    for lab in labels:
        got, tot = tally[lab]
        out.append(round(100.0 * got / tot, 2) if tot else None)
    return (out[0], out[1])


def read_multi_coord(bench: str, arm: str):
    """Coordinate-step metrics for one multi-step cell.

    UI-TARS never terminates an episode of its own accord, so whole-episode
    Step SR is governed by a stopping deficit that no token policy can move
    (stage-B gate). The reportable subset is the steps whose REFERENCE action
    carries a target
    location. Delegated to w46_uitars_multi_metrics so the definition lives in
    exactly one place.
    """
    d = f"{MULTI_ROOT}/{bench}/{arm}"
    if not os.path.isdir(d):
        return None
    # Put this file's directory on the path up front rather than only as an
    # ImportError fallback: when the aggregator is invoked from the repo root the
    # first import fails, and the retry used to be reported as "no data".
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        import w46_uitars_multi_metrics as m  # noqa: N813
    except ImportError as exc:
        print("[multi_coord IMPORT FAILED] %s" % exc, file=sys.stderr)
        return None
    try:
        return m.measure(d)
    except Exception as exc:
        # Never swallow this silently. A bare `return None` here is
        # indistinguishable from a missing run, and on 2026-08-03 it hid a
        # working computation behind an import-path problem: all 13 complete
        # cells reported null while measure() succeeded when called directly.
        print("[multi_coord FAILED] %s: %s: %s"
              % (d, type(exc).__name__, exc), file=sys.stderr)
        return None


def read_multi(bench: str, arm: str):
    """Parse TM/EM/SR/GP from the OmniGUI-style run.log tail."""
    p = f"{MULTI_ROOT}/{bench}/{arm}/run.log"
    if not os.path.exists(p):
        return None
    tm = None
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.match(
                r"GROUP ALL TM=([\d.]+)% EM=([\d.]+)% SR=([\d.]+)% GP=([\d.]+)%",
                ln.strip(),
            )
            if m:
                tm = {
                    "TM": float(m.group(1)),
                    "EM": float(m.group(2)),
                    "SR": float(m.group(3)),
                    "GP": float(m.group(4)),
                }
    return tm


def single_block(root: str, with_subs: bool = False):
    out = {}
    for bench in ("screenspot_v2", "screenspot_pro", "mmbench_l2"):
        row = {}
        for arm in SINGLE_ARMS:
            row[arm] = read_single(root, bench, arm)
            if with_subs and row[arm] is not None:
                a, b = read_single_subs(root, bench, arm)
                if a is not None or b is not None:
                    row[f"{arm}_subs"] = [a, b]
        dense = row.get("dense")
        if dense:
            for arm in SINGLE_ARMS:
                v = row.get(arm)
                if arm != "dense" and v is not None:
                    row[f"{arm}_retention"] = round(100.0 * v / dense, 2)
        out[bench] = row
    return out


def main():
    out = {
        "single_step": single_block(SINGLE_ROOT, with_subs=True),
        "single_step_root": SINGLE_ROOT,
        "single_step_search_roots": list(SINGLE_SEARCH_ROOTS),
        "multi_step": {},
    }

    for bench in ("omnigui", "mind2web_task"):
        row = {}
        for arm in MULTI_ARMS:
            row[arm] = read_multi(bench, arm)
            coord = read_multi_coord(bench, arm)
            if coord:
                row[f"{arm}_coord"] = coord
        # Retention on the reportable subset: this is the headline column, since
        # the whole-episode SR below is dominated by the termination deficit.
        dense_c = (row.get("dense_coord") or {}).get("step_sr_coord")
        if dense_c:
            for arm in MULTI_ARMS[1:]:
                v = (row.get(f"{arm}_coord") or {}).get("step_sr_coord")
                if v is not None:
                    row[f"{arm}_retention_coord"] = round(100.0 * v / dense_c, 2)
        dense = (row.get("dense") or {}).get("TM")
        if dense:
            for arm in MULTI_ARMS[1:]:
                v = (row.get(arm) or {}).get("TM")
                if v is not None:
                    row[f"{arm}_retention_TM"] = round(100.0 * v / dense, 2)
        dense_sr = (row.get("dense") or {}).get("SR")
        if dense_sr:
            for arm in MULTI_ARMS[1:]:
                v = (row.get(arm) or {}).get("SR")
                if v is not None:
                    row[f"{arm}_retention_SR"] = round(100.0 * v / dense_sr, 2)
        out["multi_step"][bench] = row

    if os.path.isdir(SINGLE_OOD_ROOT):
        out["single_step_ood_diagnostic"] = single_block(SINGLE_OOD_ROOT)

    os.makedirs("docs", exist_ok=True)
    tmp = f"docs/w45_uitars_results.json.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, "docs/w45_uitars_results.json")
    print(json.dumps({"single_step": out["single_step"],
                      "multi_step": out["multi_step"]}, indent=2))
    done = sum(
        1 for b in out["single_step"].values() for k, v in b.items()
        if not k.endswith("_retention") and v is not None
    ) + sum(
        1 for b in out["multi_step"].values() for k, v in b.items()
        if not k.endswith("_retention_TM") and not k.endswith("_retention_SR")
        and v is not None
    )
    print(f"CELLS_DONE={done}")


if __name__ == "__main__":
    main()
