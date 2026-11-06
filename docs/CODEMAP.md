# Paper ↔ code map

TRACE is the paper name. The implementation still uses GeoPACT identifiers.

## Modules

| Paper | Code name | Primary files |
|-------|-----------|----------------|
| **LIP** (layout-derived interaction prior) | Module A / MIU / mass prior | `lmms_eval/models/module_a_pipeline.py`, `gui_miu_detectors.py`, `geopact_module_a_*.py` |
| **NEO** (nested evidence ordering) | NAP | `lmms_eval/models/geopact_nap.py`, `geopact_qd_selector.py` |
| **NCR** (native-token coverage repair) | CRS / AWC | `lmms_eval/models/geopact_crs.py`, `geopact_c_absorb.py` |
| **MKC** (monotone KV contraction) | session KV / KV-rewrite | `lmms_eval/models/geopact_session_kv.py`, `geopact_vl_hook.py`, `geopact_decode_contract.py` |
| Published baselines | `nap_pubsel` / `pub_*` | `lmms_eval/models/geopact_pub_baselines.py` |
| GUI-Owl carrier | `gui_owl` | `lmms_eval/models/simple/qwen3_vl_gui.py` |
| UI-TARS carrier | `ui_tars` | `lmms_eval/models/simple/ui_tars.py` |

Orchestration: `geopact_vl_hook.py`. Transformers hooks live in `patches/qwen3_vl/modeling_qwen3_vl.py` (GUI-Owl) and `patches/qwen2_5_vl/modeling_qwen2_5_vl.py` (UI-TARS).

## Environment flags (job `EXTRA_ENV`)

| Flag | Meaning |
|------|---------|
| `GEOPACT_RATIO_CURRENT` | Current-frame keep ratio \(k_c/N\) (multi-step). Single-step uses `PRUNE_RATIO` / `RATIO` on the current frame. |
| `GEOPACT_EVAL_HISTORY_N` | History window (job column `HIST_N`, typically 5). |
| `GEOPACT_BAM_MASS_ALPHA` | LIP prior strength \(\bar\alpha\). |
| `GEOPACT_BAM_ALPHA_CAL` | Enable within-frame effective-support cap (Eq. of \(\alpha^\star\)). |
| `GEOPACT_NAP_ANTICIPATE` | Legacy NAP anticipate; adopted TRACE rows set `0`. |
| `GEOPACT_NAP_CONTRACT` | History contraction. **Code default is `strata`.** Adopted multi-step jobs usually set `prefix`. |
| `GEOPACT_CRS` | Enable NCR. |
| `GEOPACT_CRS_REP` | Representative geometry. **Code default is `medoid`.** Adopted cells use `stride`. |
| `GEOPACT_CRS_SKEL_FRAC` / `GEOPACT_CRS_NESTED_FRAC` | Coverage dose \(\rho\) on current vs nested history. |
| `GEOPACT_CRS_NESTED` | Nested retirement pass. |
| `GEOPACT_RECTIFY_PREFETCH` | Prefetch/rectify current-frame keep before decode. |
| `GEOPACT_SESSION_KV` | MKC on (`1`) vs dense re-prefill (`0`). |
| `GEOPACT_KV_HISTORY_REWRITE` | Deletion-only history rewrite (MKC). `0` is the no-rewrite control. |
| `GEOPACT_NAP_PUBSEL_POLICY` | Published selector: `divprune`, `vispruner`, `zooprune`, `prunesid`, … |
| `GEOPACT_PUB_HYBRID_ALPHA` | Detector-parity: graft the LIP field onto a baseline. |
| `GEOPACT_INDEPENDENT_PROBE` | Dump independent (non-nested) keeps for W43-C. |
| `GEOPACT_KEEP_OVERRIDE` | JSONL keep override for causal replay / nested projection. |
| `GEOPACT_SELFCOND` | Self-conditioned (student-forcing) eval. |
| `LMMS_MODEL=ui_tars` | Switch the carrier to UI-TARS. |
| `MODEL_DIR` | Checkpoint directory under `models/`. |

Track names in the TSV:

| `TRACK` | Serving path |
|---------|----------------|
| `baseline_full` | Dense re-prefill (`GEOPACT_SESSION_KV=0`) |
| `baseline_kv` | MKC, no prune |
| `abc_r*` | MKC + TRACE / NAP prune |

`EXP_ARM=v3_nap` is the TRACE selector. `nap_uniform` / `nap_random` are nested anchors. `nap_pubsel` is a published method on the MKC path.

## Tasks

| Benchmark | Task name |
|-----------|-----------|
| ScreenSpot-v2 | `task_screenspot_v2_guiowl` |
| ScreenSpot-Pro | `task_screenspot_pro_guiowl` |
| MMBench-GUI L2 (L1 in appendix) | `task_mmbench_gui_l2_guiowl` / `task_mmbench_gui_l1_guiowl` |
| OmniGUI | `task_v2_qwen3_omni_novideo` |
| Mind2Web task/website/domain | `task_mind2web_qwen3_omni_{task,website,domain}` |
| AndroidControl | `task_android_control_qwen3_omni` |

UI-TARS reuses the same task YAMLs with `LMMS_MODEL=ui_tars`.

The `lmms_eval/tasks/` tree still contains MiniCPM / TopoGravity / unrelated upstream YAML.
Those are unused by TRACE paper jobs. Ignore them.

## Code default vs adopted EXTRA_ENV

**Always trust the job TSV / `configs/experiments/adopted_8b.json`, not the Python default.**

| Knob | Code default | Typical adopted TRACE cell |
|------|----------------|----------------------------|
| `GEOPACT_RATIO_CURRENT` | 0.75 | 0.10 (single-step) / 0.50 or 0.25 (multi-step) |
| `GEOPACT_NAP_ANTICIPATE` | 1 (on) | **0** |
| `GEOPACT_NAP_CONTRACT` | `strata` | `prefix` on many multi-step rows; single-step often omitted |
| `GEOPACT_CRS_REP` | `medoid` | **`stride`** |
| `GEOPACT_CRS_SKEL_FRAC` | 0.10 | 0.50 (SSv2 r10) / 0.30 (several r5 cells) |
| `GEOPACT_CRS_NESTED_FRAC` | equals SKEL_FRAC | 0.05 (typical multi-step) |
| `GEOPACT_BAM_MASS_ALPHA` | 1.0 | 2.0 / 4.0 / 6.0 per cell |
| OmniParser weights | **must be set** | `OMNIPARSER_ICON_WEIGHTS=.../icon_detect/model.pt` |

Nestedness \(S_{hist} \subset S_{cur}\) is implemented by `contract_nap_order` plus `_rectify_nap_prefix_contract`, not by a per-step assert.
