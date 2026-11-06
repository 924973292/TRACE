# Reproduce TRACE paper results

Eval dumps are **not** in this repository. Two different operations exist:

- **Path A.** `python scripts/gen_tables.py` regenerates LaTeX from **frozen published numbers** (and UI-TARS / 2B JSON dumps under `artifacts/metrics/`). This is not a GPU rerun.
- **Path B.** Re-run the job TSVs on a CUDA GPU, then `python scripts/harvest_paper_metrics.py` and compare `artifacts/metrics/harvested_main.json` to the published column.

Adopted 8B TRACE cells: [configs/experiments/adopted_8b.json](../configs/experiments/adopted_8b.json), [w55_provenance_matrix.md](w55_provenance_matrix.md). Job index: [../configs/jobs/INDEX.md](../configs/jobs/INDEX.md).

Queue one TSV on GPU `i`:

```bash
source scripts/eval_env_defaults.sh
XDS_GPU=i bash scripts/_xds_gpu_queue.sh configs/jobs/main_8b.tsv
```

The queue skips an `OUT` that already has `results/*.json`. Set `LIMIT` for smokes. Do not launch the same `OUT` on two hosts at once.

Hardware: NVIDIA A800 80GB class (or equivalent). FlashAttention 2.5.8 is required on CUDA. Windows CPU cannot run Path B.

**Trust TSV EXTRA_ENV, not Python defaults.** See [CODEMAP.md](CODEMAP.md).

## Main tables (8B)

| Paper label | Jobs | How to score |
|-------------|------|----------------|
| `tab:main-single` | `configs/jobs/main_8b.tsv` (`logs/xds8b_tune/{screenspot_v2,screenspot_pro,mmbench_l2}/...`) + `baselines_pub.tsv` | `results/*.json` overall; Text/Icon via `python scripts/_w31_ss_ti.py`; MMBench Basic/Adv via `python scripts/_w31_mmb_subsplit.py` |
| `tab:supp-single` | same files, r=50%/25% arms | same |
| `tab:main-multi` | `main_8b.tsv` multi-step rows. OmniGUI **mild** EM=48.91 is `ablation.tsv` → `logs/xds8b_abl/omnigui/nested_rho05_c50_h10` | OmniGUI: `exact_match`. Mind2Web: pool task/website/domain by step counts 2008/1343/5429. AndroidControl: `python scripts/_ac_hist_metrics.py <run-parent>` (HistPrune protocol) |

`EXP_ARM=v3_nap` is TRACE. Published selectors use `EXP_ARM=nap_pubsel` plus `GEOPACT_NAP_PUBSEL_POLICY`.

After harvest, compare to the `published` field in `adopted_8b.json`. Editing literals in `scripts/gen_tables.py` is a manual step.

## Ablations

| Paper label | Jobs | Analysis |
|-------------|------|----------|
| `tab:ablation-main` | `ablation.tsv`, `raw/w53_*.tsv`, `raw/w55_*.tsv` | `python scripts/render_fig6_modules.py` |
| Detector parity | `p0_parity.tsv` | `python scripts/w46_parity_analysis.py` then `render_fig_parity.py` |
| Recent-N vs token order | `raw/w46_p0_*.tsv` | `python scripts/w46_p0_analysis.py` |

## Efficiency

| Paper label | Jobs | Scripts |
|-------------|------|---------|
| `tab:main-efficiency`, Fig. 4/7 | `efficiency.tsv`, `raw/w57b_unified.tsv`, `raw/w57_lat2_final.tsv` | `GEOPACT_TIMING_CUDA_SYNC=1`; `scripts/_w33_ttft_by_step.py`; `scripts/_lat2_aggregate.py` |

Measure TTFT, E2E, visual/LLM prefill, decode, KV MB, peak GPU memory from `generation_usage` in samples jsonl (multi-step) or `SS_LATENCY` lines in `run.log` (single-step).

## Transfer

| Paper label | Jobs | Notes |
|-------------|------|-------|
| `tab:uitars-single` | `uitars.tsv`, `raw/w45_uitars_single_native.tsv` | `LMMS_MODEL=ui_tars`; `python scripts/restore_q25vl_geopact_patches.py`; `w45_uitars_aggregate.py` |
| `tab:uitars-multi` | `raw/w52_uitars_multi.tsv` | `UITARS_NATIVE_MULTISTEP=1` |
| `tab:uitars-cal` | `raw/w47_uitars_cal.tsv` | ledger already in `artifacts/metrics/w47_uitars_cal_ledger.json` |
| `tab:supp-2b-*` | `scale_2b.tsv` | same EXTRA_ENV as 8B; `MODEL_DIR=models/GUI-Owl-1.5-2B-Instruct` |

## Mechanism / analysis

These consume keep dumps (`LIMIT=1000` in the source jobs). JSON snapshots used in the paper are under `artifacts/metrics/`. Re-running without dumps will not recreate keep-maps.

| Claim | Jobs | Scripts |
|-------|------|---------|
| Within-goal future-target retention | `raw/w45_ret2_*.tsv` | `w43_same_goal_retention.py` |
| Causal delete/restore | `raw/w45_causal_*.tsv` | `w43_causal_replay_prep.py`, `w45_causal_analysis.py` |
| Independent vs nested | `mechanism_w43_w45.tsv` | `w43_independent_vs_nested.py`, `w45_c2_analysis.py` |
| Episode-length buckets | retention dumps | `w46_length_buckets.py` |
| Self-conditioned (appendix) | `selfcond.tsv` | `_w60_selfcond_metrics.py`, `gen_w60_tables.py` |

## Tables and figures (Path A)

```bash
python scripts/gen_tables.py          # artifacts/tables/*.tex
python scripts/render_figs.py
python scripts/render_fig2_evidence.py
python scripts/render_fig6_modules.py
python scripts/render_fig_parity.py
python scripts/render_fig_lifecycle.py
python scripts/render_fig17_lifecycle.py
python scripts/render_fig1_motivation.py
python scripts/render_fig_selfcond.py
```

Keep maps (`render_fig9_keepmaps.py`) need qualitative dumps that are omitted from this extract.

Appendix hand tables that lived only in the manuscript (`tab:contract`, some module-internal `.tex`) have **no generator** here.

## Invariant tests

```bash
python scripts/verify_sliding_rewrite_invariants.py
python scripts/verify_kv_incremental_flow.py
python scripts/verify_dialogue_mrope_boundary.py
python scripts/verify_decode_mrope_contract.py
python scripts/verify_module_a_prefetch_identity.py
python scripts/_w60_selfcond_unit.py
python scripts/_w35_verify_tables.py
```
