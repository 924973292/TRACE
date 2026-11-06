# Qwen3 dialogue KV evaluation protocol (locked)

> Public extract: env flags and track semantics in this file still apply.
> Launch paper jobs from `configs/jobs/` as described in [REPRODUCE.md](REPRODUCE.md).
> Ignore historical cluster launcher names and log paths.

## Dual-track index

| Track | OmniGUI task | When to use |
|-------|--------------|-------------|
| **Mainline (this doc)** | `task_v2_qwen3_omni_novideo` | GeoPACT KV session, SSPC/ABC pruning, history6 rolling window |
| **Compare only** | `task_guiowl_omnigui_native_mt4_8b` | Optional prompt ablation; EM/GP not comparable to mainline |
| **Single-step benches** | `task_*_guiowl` | Official `mobile_use` prompt; KV env on but step=0 no-op |

Campaign phases: [`history6_8b_eval_phases.md`](history6_8b_eval_phases.md).

## Baseline (accuracy reference)

Use **task-once** novideo history — Task goal emitted only on the first user turn.

| Item | Value |
|------|--------|
| Task | `task_v2_qwen3_omni_novideo` |
| `qwen_repeat_task_in_history` | `false` (task-once) |
| `qwen_include_history` | `true` |
| Session KV | off (`GEOPACT_SESSION_KV=0`) |
| Vision embed cache | off (`GEOPACT_VISION_EMBED_CACHE=0`) |
| Locked full run | `logs/baseline_full_matrix_20260612_114748/task_once_vcache0` |

**Do not** use `task_v2_qwen3_omni_novideo_task_repeat` as the dialogue-KV baseline. Task-repeat hurts EM ~2pp and adds latency when vision cache is off.

## Dialogue KV (cache under test)

| Item | Value |
|------|--------|
| Task | same as baseline (`task_v2_qwen3_omni_novideo`) |
| `GEOPACT_SESSION_KV` | `1` |
| `GEOPACT_SESSION_KV_MODE` | `dialogue` |
| `GEOPACT_VISION_EMBED_CACHE` | `0` (current-frame ViT only; history in LLM KV) |
| `GEOPACT_FORCE_EPISODE_ORDER` | `1` |
| Compare against | `task_once_vcache0` metrics |

Launch:

```bash
bash scripts/run_qwen_dialogue_kv_cache_eval.sh          # full 2572
LIMIT=5 bash scripts/run_qwen_dialogue_kv_cache_eval.sh  # smoke
```

## Segment resets

Some episodes start a new sub-task after `TASK_COMPLETE` while keeping the same
`episode_id`. Those steps have `step_index > 0` but empty `history` /
`action_history`. Dialogue KV treats them like step 0 (full prefill, stale KV
cleared) instead of building an incremental suffix.

## Rolling history window + KV slide (default on)

When `len(action_history) > history_n`, the prompt keeps only the last `history_n` turns.
At each roll boundary the evicted turn is the oldest step in the previous window.

**Why not full prefill every roll?** RoPE attention depends on *relative* position
`(q_pos - k_pos)`. Dropping a common left prefix from KV and token ids preserves all
relative distances among remaining tokens; appending the incremental suffix is
equivalent to a fresh full prefill of the shorter window.

| Env | Meaning |
|-----|---------|
| `GEOPACT_EVAL_HISTORY_N` unset | Locked prod `history_n=6` |
| `GEOPACT_EVAL_HISTORY_N=0` | Unlimited history (no rolling on OmniGUI; max hist=24) |
| `GEOPACT_EVAL_HISTORY_N=50` | Window 50 (= unlimited on current dataset) |
| `GEOPACT_KV_SLIDE_ON_ROLL=1` | **Default:** left-trim evicted turn from KV + incremental suffix |
| `GEOPACT_KV_SLIDE_ON_ROLL=0` | Legacy: `clear_session_kv` + full prefill at roll |

Slide path verifies `cached[drop:] + suffix_ids == full_prompt_ids` before trimming.
On mismatch it falls back to full prefill.

**2-arm KV rewrite gate** (baseline_full / baseline_kv_rewrite, no ABC):

```bash
LIMIT=10 TAG=mytag bash scripts/run_kv_rewrite_gate_2arm.sh
```

Arm2 = incremental dialogue KV + hist9 slide; **no** `visual_prune_ratio`, **no** ABC rectify/merged rewrite (`kv_flow_mode=incremental`).

**3-arm ABC+KV gate** (baseline_full / baseline_kv / abc_r40_kv):

```bash
LIMIT=100 TAG=mytag bash scripts/run_kv_abc_gate_4b.sh
# decode-focused probe + profile diagnose:
LIMIT=10 bash scripts/run_abc_decode_probe.sh
```

## Production ABC + dialogue KV (hist9, 4B)

Frozen stack: **A** (YOLO + pentagonal + CDCT field) → **B** (ICKH @ r=0.40) → **C** (SSPC-ZAFD in-budget) → **dialogue KV**.

| Item | Value |
|------|--------|
| EXP_ARM | `18_prod_drop_scale_a02_sspc_zafd` |
| Policy | `oas_ickh_cdct_no_coord_yolo_v2_drop_pent_scale` |
| Retention | `visual_prune_ratio=0.40` |
| Scope | `image_sticky` |
| Module C | `GEOPACT_SSPC=1`, `GEOPACT_SSPC_ZAFD=1`, `GEOPACT_ZAFD_OMEGA_NORM=survivor` |
| Module C coord | `GEOPACT_COORD_METADATA_ONLY=1` (no token splice) |
| History | `GEOPACT_EVAL_HISTORY_N=9` (hist9 + current frame) |
| KV | `GEOPACT_SESSION_KV=1`, `GEOPACT_SESSION_KV_MODE=dialogue` |

Use **2-arm** gate for KV-only regression; **3-arm** gate when ABC is enabled. See gate commands in the rolling-window section above.


| Metric | Definition |
|--------|------------|
| **e2e_latency_ms** | Wall time `step_t0` → response ready (always full step). |
| **ttft_ms** | `preprocess_ms` + first-token time inside `generate()` (includes ViT prepare, MRoPE recompute, external batch suffix LLM prefill). |
| **vision_prefill_ms** | Processor + GeoPACT ViT/scatter (`geopact_prepare_ms` / `vision_encode_ms`) + optional `visual` hook. |
| **llm_prefill_ms** | `generate()` LM prefill hook (last prompt token) **+** `dialogue_kv_batch_prefill_ms` (712-tok external forward on step>0). |
| **decode_total_ms** | All LM forwards inside `generate()` after the first (autoregressive action tokens). |
| **dialogue_kv_batch_prefill_ms** | Step>0 only: one `language_model` forward for suffix[:-1] before `generate()`. |
| **dialogue_kv_mrope_ms** | Step>0 only: full-sequence MRoPE index for cached prefix + suffix. |

Baseline (no cache): all LLM prefill is inside `generate()` hooks. Cache step>0 splits LLM prefill into external batch + generate's final prompt token; both roll into `llm_prefill_ms`.

## Uniform pruning + dialogue KV (pre-LLM, cache-compatible)

Visual pruning runs in `prepare_geopact_projector_inputs` **before** `language_model` forward. Dialogue session KV stores **post-prefill** `past_key_values`; it does not need changes when pruning is enabled.

**MRoPE (origpos):** hook builds full-sequence `get_rope_index`, then `_subset_origpos_mrope(keep_mask)` — kept tokens keep their original spatial positions. Dialogue KV step>0 recomputes on **unpruned** virtual `cached_mrope_input_ids + suffix_mrope_source`, then subsets suffix `mrope_keep_mask` (not compact ids + full grid).

| Item | Value |
|------|--------|
| Policy | `uniform` (evenly spaced raster-order tokens; deterministic) |
| Retention | `visual_prune_ratio=0.40` (keep 40%) |
| Scope | `image_sticky` |
| Compact backend | `GEOPACT_COMPACT_BACKEND=source` (required) |
| Session KV | same dialogue settings as cache baseline |

Launch (smoke / full):

```bash
LIMIT=5 bash scripts/run_qwen_dialogue_kv_uniform_r40_eval.sh
bash scripts/run_qwen_dialogue_kv_uniform_r40_eval.sh
```

Paired no-cache prune baseline:

```bash
LIMIT=5 bash scripts/run_qwen_uniform_r40_baseline_eval.sh
```

## Token-level sticky prune (dialogue KV, no Module A)

Uses `prepare_dialogue_pruned_embed_inputs` → `compute_sticky_dialogue_prune_keep_set`
(`compact_backend=dialogue_prune_sticky`). Token policies do **not** use YOLO /
`oas_ickh_*`. OAS policies on dialogue sticky use the same prepare path with Module A
(see section below).

**Sticky semantics:** each screenshot is pruned once when it is the **current frame**;
the local keep mask is cached by `(episode_id, image_path)` and reused when that
screenshot appears in history (ViT re-encode still runs; token subset is fixed).

| Policy | Weights w_i | Greedy diversity term |
|--------|-------------|------------------------|
| `uniform` | N/A (even stride) | N/A |
| `token_ickh_diversity` | 1 | μ_i − avg_sim (pure feature dissimilarity) |
| `token_ickh_relevance` | r_i = max(0, cos(e_i, q)) | same (no InfoDensity) |
| `token_instr_topk` | r_i only | **none** — argtopk by relevance |
| `token_div_topk` | N/A | **none** — FPS on vision embeddings |

**Relevance query `q`:** on dialogue KV step 0 / segment-reset full prefill,
embed `doc.task_description` (or `instruction`) once and cache on the inner model.
Step>0 suffix prefills **reuse** the cached global `q` (suffix user text does not
re-emit the task goal under task-once protocol). Applies to `token_ickh_relevance`
and `token_instr_topk`.

Common settings: `visual_prune_ratio=0.40`, `visual_prune_scope=image_sticky`,
`GEOPACT_SESSION_KV=1`, `GEOPACT_VISION_EMBED_CACHE=0`.

Verify + LIMIT=100 compares:

```bash
python scripts/verify_token_ickh_sticky.py
python scripts/audit_relevance_query_source.py
TAG=token_ickh_r40_$(date +%Y%m%d) bash scripts/run_qwen_token_ickh_r40_limit100.sh
TAG=token_topk_ablation_$(date +%Y%m%d) bash scripts/run_qwen_token_topk_ablation_limit100.sh
```

Four-arm baseline: `04a_dialogue_kv_r40_random`, `04_dialogue_kv_r40` (uniform),
`04b_dialogue_kv_r40_tok_div`, `04c_dialogue_kv_r40_tok_rel`.

Top-K ablation (herding vs pure top-K): `04b`, `04c`, `04d_dialogue_kv_r40_tok_instr_topk`,
`04e_dialogue_kv_r40_tok_div_topk`.

## Dialogue sticky + Module A (oas, AB-only)

`oas_ickh_*` / `oas_random_*` policies now route through the same lightweight path as
token sticky prune when `GEOPACT_SESSION_KV=1` and `visual_prune_scope=image_sticky`:

`prepare_dialogue_pruned_embed_inputs` → `compute_sticky_dialogue_prune_keep_set` →
`_prune_single_frame` (Module A detect + MIU HES) → `_compact_sequence` with
`coord_text=None` by default (no Module C unless ephemeral C is enabled).

| Item | Value |
|------|--------|
| Example policy | `oas_ickh_coord_center_yolo` |
| Detector | `miu_detector=omniparser` (or policy suffix `yolo`) |
| Retention | `visual_prune_ratio=0.40` (typical) |
| Scope | `image_sticky` |
| Compact backend | `GEOPACT_COMPACT_BACKEND=source` |
| Module C | **off** by default (AB-only; coord text not injected) |

### Ephemeral Module C (dialogue-KV + CDCT)

When `GEOPACT_COORD_EPHEMERAL=1` and policy includes `coord_ephemeral` (legacy) or
`coord_sdb` (policy tag; format controlled by `GEOPACT_COORD_FORMAT`):

| Env | Value |
|-----|--------|
| `GEOPACT_COORD_EPHEMERAL` | `1` |
| `GEOPACT_COORD_METADATA_ONLY` | `0` (token splice active) |
| `GEOPACT_COORD_TOPK` | `3` (typical UDSI) or `5` (legacy 04o) |
| `GEOPACT_COORD_FORMAT` | **`spatial_plain`** (04r) or **`sidecar_*`** (04t–04w, GATS) or `spatial_ranked` (04s) |
| `GEOPACT_COORD_SLOT` | `before_gen_tail` (ephemeral default) |
| `GEOPACT_COORD_ANCHOR_POLICY` | `pick_order_topk` (default), `spatial_diverse_topk`, `keep_survivor_rank` |
| `GEOPACT_CDCT_ALPHA` | `0.1` (typical experiment) |
| YOLO | `OMNIPARSER_ICON_MAX_EDGE=1536` (typical) |

Deprecated formats: `sdb` (04q — contains `action_type` schema words), `legacy` (04o).

#### UDSI — Universal Decoupled Spatial Injection (Module C v3)

Spatial channel is **fully decoupled** from the Qwen eval output schema (`<thought>` +
`<tool_call>`). Appendix contains **reference coordinates only** — no `action_type`,
no `tool_call`, no binding contract.

Appendix examples (~15–30 tokens for topk=3):

- `spatial_plain`: `[Ref points xy@1000] (832,73) (500,670) (200,110)`
- `spatial_ranked`: `[Ref] P1=(832,73) P2=(500,670) P3=(200,110)`

Implementation package: [`lmms_eval/models/geopact_coord_injection/`](../lmms_eval/models/geopact_coord_injection/)

| Module | Role |
|--------|------|
| `pipeline.py` | Build anchors + render appendix |
| `record.py` | `CoordInjectionRecord` (insert_pos, token_count, ephemeral flag) |
| `lifecycle.py` | KV boundary strip via `insert_pos` (not tail guess) |
| `slot.py` | Declarative splice position (`before_gen_tail`) |
| `renderer.py` | Spatial-only text profiles |

Flow:

1. Module B greedy ICKH records **pick order**; `anchor_provider` maps pick order → top-k
   centers (normalized 0–1000).
2. Coord appendix is spliced **before chat-template gen tail** via `InjectionSlot.BEFORE_GEN_TAIL`.
3. `strip_for_kv_commit` removes coord block at `insert_pos` and gen tail before dialogue KV commit.
4. **Relevance `q` unchanged** — coord text must not appear in instruction embedding.

Launch (LIMIT=100):

```bash
TAG=udsi_v1 LIMIT=100 bash scripts/run_qwen_ab_cdct_a01_coord_ephemeral_limit100.sh
```

Arms: `04n` (A+B) vs `04r` (A+B+C `spatial_plain`). Ablations: `04s` (`spatial_ranked`),
GATS ladder `04t`–`04w`, legacy `04o` / deprecated `04q` (SDB). Preflight:
`python scripts/verify_dialogue_coord_ephemeral.py`

#### Production stack Module C (v2 drop_scale + α=0.2)

Aligned with factorial R3 champion `07_prod_drop_scale_a02` (GP 28.04% @ r=40).

| EXP_ARM | Module C | Notes |
|---------|----------|-------|
| `07_prod_drop_scale_a02` | OFF | A+B production baseline |
| `08_prod_drop_scale_a02_coord_plain` | UDSI `spatial_plain` topk=3 | primary A+B+C |
| `08_prod_drop_scale_a02_coord_ranked` | `spatial_ranked` | format ablation |
| `08_prod_drop_scale_a02_coord_sidecar_keep` | GATS `sidecar_keep` | format ablation |
| `08_prod_drop_scale_a02_coord_shuffle` | shuffle control | must lose vs plain |

Dev tree (fleet-safe): `$REPO`. Sync via `sync_to_module_c_dev.ps1` only.

```bash
# LIMIT=20 smoke
cd $REPO
CUDA_VISIBLE_DEVICES=0 LIMIT=20 bash scripts/run_module_c_dev_smoke.sh

# LIMIT=100 A+B vs A+B+C
LIMIT=100 bash scripts/run_module_c_prod_ab_vs_abc_limit100.sh

# Ablation ladder
VARIANTS="plain ranked shuffle" LIMIT=100 bash scripts/run_module_c_ablation_limit100.sh
```

After the module-C fleet completes: `merge_module_c_dev_to_omini.ps1`, then `bash scripts/run_module_c_prod_full_2572.sh`.

#### GATS — Geometry-Aware Textual Sidecar (Module C v4)

Fixed-schema **KEEP** rows for prune survivors (BTL-UI / MEGA-GUI inspired; no action
schema words). Coordinates are **xyxy @1000** as plain integers (Phi-Ground style),
not `<|box_start|>` tokens.

Header (one line):

```text
[Pruned view sidecar xy@1000; reference only]
```

Per-region line (pipe-separated):

```text
[KEEP] R1 | bbox=[128,214,346,501] | role=icon_button | text="支付" | conf=0.9
```

| Arm | `GEOPACT_COORD_FORMAT` | Fields |
|-----|-------------------------|--------|
| 04t | `sidecar_bbox` | bbox only (L0) |
| 04u | `sidecar_bbox_role` | + role (L1) |
| 04v | `sidecar_bbox_role_text` | + text (L2) |
| 04w | `sidecar_keep` | + conf (L3) |

Literature mapping: coarse ROI = Module B prune; structured list = GATS sidecar;
differs from BTL-UI (we list **kept** MIUs only). YOLO-only stacks often have
empty `text=` (`-`).

Launch (LIMIT=100, full L0–L3 ladder):

```bash
TAG=gats_v1 LIMIT=100 bash scripts/run_qwen_sidecar_ablation_limit100.sh
```

Post-metrics (`tool_call` / `parse_fail` rates):

```bash
python scripts/analyze_sidecar_eval_metrics.py --root logs/kv_prune_sidecar_ablation_maxedge1536_limit100_TAG
```

Success criteria (vs `04n`): `tool_call` ≤6%, `parse_fail` ≤2%, TM ≥60%, EM ≥45%.

**Sticky semantics:** identical to token arms — each screenshot is pruned once when it is
the **current frame**; local keep mask cached by `(episode_id, image_path)`.

**Relevance query `q`:** for `oas_ickh_*` (instruction relevance), step-0 full prefill
caches global task instruction embedding; suffix steps reuse cached `q` in MIU kernel
herding (same fix as `token_ickh_relevance`).

**PIL requirement:** `cache_geopact_image` runs on dialogue+oas steps (skipped only for
pure token baselines).

Verify + smoke:

```bash
python scripts/verify_token_ickh_sticky.py
LIMIT=5 TAG=dialogue_oas_smoke_$(date +%Y%m%d) bash scripts/run_qwen_dialogue_oas_r40_smoke.sh
```

Smoke arms: `04c_dialogue_kv_r40_tok_rel` (token, no YOLO) vs
`04f_dialogue_kv_r40_oas_ickh_yolo` (Module A + MIU HES).

## Token-B vs MIU-B vs CDCT (no_density / cdct) ablation

| Arm | EXP_ARM | Policy | A (detect) | B operates on |
|-----|---------|--------|------------|---------------|
| Pure B | `04c_dialogue_kv_r40_tok_rel` | `token_ickh_relevance` | none | vision tokens |
| MIU union (legacy) | `04g_dialogue_kv_r40_oas_ickh_nd_yolo` | `oas_ickh_no_density_no_coord_yolo` | YOLO | MIU box union + ICKH |
| **CDCT** | `04h_dialogue_kv_r40_oas_ickh_cdct_yolo` | `oas_ickh_cdct_no_coord_yolo` | YOLO | token ICKH, \(w_j=r_j(1+\alpha P_j)\) |

CDCT (Component-Density guided Token herding): Module A `info_density` is max-pooled onto
the vision token grid as field \(P_j\); weights become `r_j * (1 + GEOPACT_CDCT_ALPHA * P_j)`
(default α=1) before standard token ICKH (`μ = Kw/Σw`). Selection is always at token granularity
(`keep_fill_ratio=1`); `keep_outside_union_ratio` may be &gt;0 (soft spatial prior).

**0-MIU fallback (CDCT only):** when YOLO returns no MIUs, Module B uses **\(w_j=r_j\)** only
(spatial prior disabled) but **still prunes to `keep_k`** — equivalent to `token_ickh_relevance`.
Log field `cdct_fallback: no_spatial_prior` marks these steps. Legacy MIU-oas (04g) still
full-keeps on 0 MIU.

**Preflight audit** (same LIMIT=N doc set as eval):

```bash
python scripts/audit_yolo_zero_miu_eval_set.py --limit 100 --tag mytag --out-dir logs/...
```

Launch (04c vs 04h):

```bash
source scripts/geopact_yolo_eval_env.sh
TAG=ab_token_cdct_$(date +%Y%m%d) LIMIT=100 bash scripts/run_qwen_ab_token_vs_cdct_limit100.sh
```

Output: `logs/kv_prune_ab_token_vs_cdct_limit100_${TAG}/ab_token_vs_cdct_summary.md`

Legacy MIU ablation (04c vs 04g):

```bash
TAG=ab_token_miu_nd_$(date +%Y%m%d) LIMIT=100 bash scripts/run_qwen_ab_token_vs_miu_nd_limit100.sh
```

Output: `logs/kv_prune_ab_token_vs_miu_nd_limit100_${TAG}/ab_token_vs_miu_nd_summary.md`

## Pure B vs Module-A sideload (YOLO load, no spatial prior in B)

| Arm | EXP_ARM | Policy | A (detect) | B operates on |
|-----|---------|--------|------------|---------------|
| Pure B | `04c_dialogue_kv_r40_tok_rel` | `token_ickh_relevance` | none | vision tokens |
| **A sideload** | `04i_dialogue_kv_r40_tok_rel_yolo` | `token_ickh_relevance` | YOLO (full Module A) | vision tokens (same as 04c) |

04i sets `GEOPACT_MODULE_A_SIDELOAD=1`: Module A runs detect → OAS → pentagonal →
info_density using `oas_ickh_cdct_no_coord_yolo`, but B still calls `_hes_select(mius=[])`
with `token_ickh_relevance` — **no spatial prior**, identical keep set to 04c.
Log field `module_a_sideload: true` marks these steps.

```bash
TAG=ab_a_sidecar_$(date +%Y%m%d) LIMIT=100 bash scripts/run_qwen_ab_token_vs_a_sidecar_limit100.sh
```

Output: `logs/kv_prune_ab_a_sidecar_limit${LIMIT}_${TAG}/ab_a_sidecar_summary.md`

## Sideload-first A+B ladder (L0–L4)

Abandon GPU inline (`04h`) as the primary integration path. Build on **04i sideload parity** and add **CPU prefetch** before prune when spatial prior must affect the current screenshot keep set.

| Level | Arm | Module A timing | B policy | α | Parity gate |
|-------|-----|-----------------|----------|---|-------------|
| **L0** | `04c_dialogue_kv_r40_tok_rel` | none | `token_ickh_relevance` | — | baseline |
| **L1** | `04i_dialogue_kv_r40_tok_rel_yolo` | eval_end sideload | `token_ickh_relevance` | — | = L0 keep/resp/decode |
| **L1+** | `04j_dialogue_kv_r40_tok_rel_yolo_export` | eval_end sideload + weight JSONL | `token_ickh_relevance` | — | = L1 latency; export non-empty |
| **L2** | `04k_cpu_prefetch_cdct_a0_dialogue_kv_r40` | prepare CPU prefetch | `oas_ickh_cdct_no_coord_yolo` | 0 | = L0 keep/resp/OutTok (+≤250ms prepare) |
| **L3** | `04l_cpu_prefetch_cdct_a1_dialogue_kv_r40` | prepare CPU prefetch | `oas_ickh_cdct_no_coord_yolo` | 1 | decode healthy; keep may differ |
| **L4** (optional) | `04h_*` GPU inline | prepare inline GPU | CDCT | 1 | frozen对照 only |

**Env (L2 / 04k):**

```bash
GEOPACT_MODULE_A_SIDELOAD=0
GEOPACT_MODULE_A_PREFETCH=cpu
GEOPACT_CDCT_ALPHA=0
GEOPACT_MODULE_A_SUBPROCESS=1
GEOPACT_MODULE_A_SUBPROCESS_POOL=1
GEOPACT_MODULE_A_CACHE=0
MIU_YOLO_DEVICE=cpu
visual_prune_policy=oas_ickh_cdct_no_coord_yolo
```

At α=0, Module A still runs (pooled subprocess worker) and computes spatial field \(P_j\); CDCT uses \(w_j=r_j(1+0\cdot P_j)=r_j\) so **keep/resp must match pure B** (`verify_token_ickh_sticky.py`).

**Env (L3 / 04l — subprocess isolation):**

```bash
GEOPACT_MODULE_A_PREFETCH=cpu
GEOPACT_CDCT_ALPHA=1
GEOPACT_MODULE_A_SUBPROCESS=1
GEOPACT_MODULE_A_SUBPROCESS_POOL=1
GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${EVAL_PYTHON}"
GEOPACT_MODULE_A_CACHE=0
MIU_YOLO_DEVICE=cpu
```

`GEOPACT_MODULE_A_SUBPROCESS=1` runs Module A in an isolated child process. With `GEOPACT_MODULE_A_SUBPROCESS_POOL=1` (default), a **persistent worker** (`scripts/module_a_run_subprocess.py --serve`) loads YOLO once and serves newline-JSON RPC; set `POOL=0` to fall back to one-shot subprocess per step. Parent calls `torch.cuda.synchronize()` after each detect.

**Latency opts (P1–P4, default on for 04k/04l):**

```bash
GEOPACT_MODULE_A_PREFETCH_ASYNC=1      # overlap worker detect with ViT (prepare path)
GEOPACT_MODULE_A_WORKER_SHM=1          # RGB shared-memory IPC (PNG fallback)
GEOPACT_MODULE_A_SUBPROCESS_PREWARM=1    # warm worker in Qwen3VLGui.__init__ (eval process)
GEOPACT_MODULE_A_SUBPROCESS_PREWARM_ASYNC=1  # overlap worker warm with processor load
GEOPACT_MODULE_A_WORKER_YOLO_DEVICE=0  # worker subprocess only; parent MIU_YOLO_DEVICE=cpu
```

Diag fields: `module_a_prefetch_async`, `module_a_overlap_ms`, `module_a_join_ms`, `ipc_shm_ms` / `ipc_png_ms` in `timings_ms`.

```bash
TAG=sideload_ladder LIMIT=20 bash scripts/run_qwen_ab_sideload_ladder_limit20.sh
```

Output: `logs/kv_prune_ab_sideload_ladder_limit${LIMIT}_${TAG}/` with parity reports vs 04c before 04l runs.
