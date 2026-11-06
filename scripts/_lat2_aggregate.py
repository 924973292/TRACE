#!/usr/bin/env python3
"""Aggregate the lat2 latency+memory anatomy waves into a paper-ready table.

Reads two kinds of logs:
  * multi-step (OmniGUI): per-step ``step_eval.generation_usage`` dicts in
    ``results/*/*samples*.jsonl``;
  * single-step (SSPro): ``SS_LATENCY {json}`` lines in ``run.log``.

For each arm it reports the per-step means of every latency stage, the token
budget (before/after pruning), decode, KV-cache footprint, peak GPU memory,
and a layer-aware prefill-FLOPs proxy.

Layer-aware prefill FLOPs
-------------------------
FLOPs ~= 2 * P_llm * T_eff, where T_eff is the per-layer-averaged prefill
token count. For every arm that reduces the *input* sequence (ours,
uniform/random, all input-level baselines, and the full baseline) every layer
processes the same post-prune length, so T_eff == mean input tokens. FastV
prunes *inside* the LLM at layer k, so shallow layers [0,k) still process the
full sequence N and deep layers [k,L) process the kept subset; we credit that
with T_eff = (k*N + (L-k)*keep*N) / L. This is the fair way to show that
FastV's compute saving is real but bounded to deep layers, while input-level
methods (incl. ours) save across all layers. Constants are the GUI-Owl-1.5-8B
LLM (L, P) and are printed with the table so the proxy is auditable.
"""
import glob
import json
import os
import sys
from collections import defaultdict

# GUI-Owl-1.5-8B language tower (Qwen3-8B backbone).
LLM_LAYERS = 36
LLM_PARAMS = 7.5e9  # LLM-only; ViT prefill FLOPs folded into vision stage
FASTV_DROP_LAYER = 2  # env GEOPACT_FASTV_LAYER default

FIELDS = [
    "e2e_latency_ms", "ttft_ms", "vision_prefill_ms", "llm_prefill_ms",
    "decode_latency_ms", "pruning_ms", "input_tokens", "tokens_before_pruning",
    "tokens_after_pruning", "output_tokens", "peak_gpu_memory_mb",
    "kv_cache_mb", "kv_frac_of_peak_mem",
]

# arm dir -> (pretty label, is_fastv, keep_ratio_for_fastv)
def arm_specs(root_ms, root_ss):
    ms = [
        ("baseline_full", "Baseline (full)", False, 1.0),
        ("ours_c50_h10", "GeoPACT (ours)", False, 1.0),
        ("fastv_c50_h10", "FastV", True, 0.50),
        ("divprune_c50_h10", "DivPrune", False, 1.0),
        ("dart_c50_h10", "DART", False, 1.0),
        ("cdpruner_c50_h10", "CDPruner", False, 1.0),
        ("visionzip_c50_h10", "VisionZip", False, 1.0),
        ("vispruner_c50_h10", "VisPruner", False, 1.0),
        # W38 / Minor 20: expand the anatomy to the full baseline set.
        ("zooprune_c50_h10", "ZOO-Prune", False, 1.0),
        ("trim_c50_h10", "TRIM", False, 1.0),
        ("prumergeplus_c50_h10", "PruMerge+", False, 1.0),
        ("visiontrim_c50_h10", "VisionTrim", False, 1.0),
        # W38 / E1: committable competitors ported onto the KV-rewrite path.
        ("pubsel_vispruner_c50_h10", "VisPruner->KVrw", False, 1.0),
        ("pubsel_zooprune_c50_h10", "ZOO-Prune->KVrw", False, 1.0),
        ("pubsel_divprune_c50_h10", "DivPrune->KVrw", False, 1.0),
    ]
    ss = [
        ("noprune_ref", "No-prune (ref)", False, 1.0),
        ("ours_r25", "GeoPACT (ours)", False, 1.0),
        ("fastv_r25", "FastV", True, 0.25),
        ("divprune_r25", "DivPrune", False, 1.0),
        ("dart_r25", "DART", False, 1.0),
        ("cdpruner_r25", "CDPruner", False, 1.0),
        ("visionzip_r25", "VisionZip", False, 1.0),
        ("vispruner_r25", "VisPruner", False, 1.0),
    ]
    return (
        [(os.path.join(root_ms, d), lbl, fv, kr) for d, lbl, fv, kr in ms],
        [(os.path.join(root_ss, d), lbl, fv, kr) for d, lbl, fv, kr in ss],
    )


def _mean(acc):
    return {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}


def collect_ms(arm_dir):
    fs = sorted(glob.glob(os.path.join(arm_dir, "results", "*", "*samples*.jsonl")))
    if not fs:
        return None
    acc = defaultdict(lambda: [0.0, 0])
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
                    continue
            gu = se.get("generation_usage") or {}
            for k in FIELDS:
                v = gu.get(k)
                if isinstance(v, (int, float)):
                    acc[k][0] += float(v)
                    acc[k][1] += 1
    return _mean(acc) if acc else None


def collect_ss(arm_dir):
    rl = os.path.join(arm_dir, "run.log")
    if not os.path.exists(rl):
        return None
    acc = defaultdict(lambda: [0.0, 0])
    with open(rl, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # tqdm progress bars share physical lines via bare \r — split so
            # the JSON payload is never contaminated by a trailing \r segment.
            for seg in line.split("\r"):
                i = seg.find("SS_LATENCY ")
                if i < 0:
                    continue
                payload = seg[i + len("SS_LATENCY "):].strip()
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    # payload may be followed by more log text on the same
                    # segment; retry with the outermost braces only.
                    j0, j1 = payload.find("{"), payload.rfind("}")
                    if j0 < 0 or j1 <= j0:
                        continue
                    try:
                        d = json.loads(payload[j0:j1 + 1])
                    except json.JSONDecodeError:
                        continue
                for k in FIELDS:
                    v = d.get(k)
                    if isinstance(v, (int, float)):
                        acc[k][0] += float(v)
                        acc[k][1] += 1
    return _mean(acc) if acc else None


def prefill_gflops(mean, is_fastv, keep):
    # N = full prefill length the LLM sees before any in-LLM prune.
    n = mean.get("tokens_before_pruning") or mean.get("input_tokens")
    if not n:
        return None
    if is_fastv:
        k = FASTV_DROP_LAYER
        t_eff = (k * n + (LLM_LAYERS - k) * keep * n) / LLM_LAYERS
    else:
        # input-level: LLM sees the reduced input on every layer.
        t_eff = mean.get("input_tokens") or n
    return 2.0 * LLM_PARAMS * t_eff / 1e9


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def render(title, specs, collector, baseline_label):
    print(f"\n### {title}\n")
    cols = ["Method", "In.Tok", "TokRed%", "Prefill GFLOPs", "Sel ms", "TTFT ms",
            "Vis ms", "LLMpre ms", "Decode ms", "E2E ms", "Peak GB", "KV MB", "KV%peak"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    base_tok = None
    rows = []
    for arm_dir, label, is_fastv, keep in specs:
        m = collector(arm_dir)
        if m is None:
            rows.append((label, None, is_fastv, keep))
            continue
        rows.append((label, m, is_fastv, keep))
        if label == baseline_label:
            base_tok = m.get("input_tokens") or m.get("tokens_before_pruning")
    for label, m, is_fastv, keep in rows:
        if m is None:
            print(f"| {label} | 在跑/缺 | | | | | | | | | | |")
            continue
        intok = m.get("input_tokens")
        tokred = None
        if base_tok and intok:
            tokred = 100.0 * (1 - intok / base_tok)
        gf = prefill_gflops(m, is_fastv, keep)
        kvpct = m.get("kv_frac_of_peak_mem")
        kvpct = kvpct * 100 if kvpct is not None else None
        peak_gb = (m.get("peak_gpu_memory_mb") or 0) / 1024.0 if m.get("peak_gpu_memory_mb") else None
        print("| " + " | ".join([
            label, fmt(intok, 0), fmt(tokred, 1), fmt(gf, 0),
            fmt(m.get("pruning_ms")), fmt(m.get("ttft_ms")),
            fmt(m.get("vision_prefill_ms")), fmt(m.get("llm_prefill_ms")),
            fmt(m.get("decode_latency_ms")), fmt(m.get("e2e_latency_ms")),
            fmt(peak_gb, 2), fmt(m.get("kv_cache_mb")), fmt(kvpct, 1),
        ]) + " |")


def main():
    root_ms = sys.argv[1] if len(sys.argv) > 1 else "logs/xds8b_lat2/omnigui"
    root_ss = sys.argv[2] if len(sys.argv) > 2 else "logs/xds8b_lat2/screenspot_pro"
    ms, ss = arm_specs(root_ms, root_ss)
    print(f"# lat2 efficiency anatomy  (LLM L={LLM_LAYERS}, P={LLM_PARAMS/1e9:.1f}B, "
          f"FastV drop@L{FASTV_DROP_LAYER}; FLOPs=2·P·T_eff, T_eff layer-averaged)")
    render("Multi-step — OmniGUI c50/h10 (per-step means)", ms, collect_ms, "Baseline (full)")
    render("Single-step — ScreenSpot-Pro r25 (per-doc means)", ss, collect_ss, "No-prune (ref)")


if __name__ == "__main__":
    main()
