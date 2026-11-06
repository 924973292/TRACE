# Cross-benchmark eval protocol (GUI-Owl-1.5-8B-Instruct)

> Public extract: use [datasets.md](datasets.md) and [REPRODUCE.md](REPRODUCE.md)
> for downloads and job launch. The six frozen TRACE suites are OmniGUI,
> ScreenSpot-v2, ScreenSpot-Pro, MMBench-GUI, Mind2Web, and AndroidControl.
> Online AndroidWorld / vLLM launchers are not part of this extract.

Executable index for offline (`lmms_eval`) benchmarks.
Each dataset keeps **its native accuracy metric**; GeoPACT lightweight metrics are additive only.

## Benchmark matrix (latest)

| Benchmark | Type | Metric | Notes |
|-----------|------|--------|-------|
| OmniGUI | Offline | TM, EM, GP | 2572 steps, **image-only** current screenshot |
| ScreenSpot-Pro | Offline Grounding | center_acc | Desktop pro apps, high-res |
| ScreenSpot-v2 | Offline Grounding | center_acc (+ domain macro) | Mobile+Web+Desktop **1272** |
| AndroidControl | Offline Action | exact_match | **test** split only |
| MMBench-GUI | Offline L1/L2 | L1 QA acc; L2 grounding acc | `OpenGVLab/MMBench-GUI` |
| AndroidWorld | Online | Task SR (target **69.0%**) | MobileAgent-v3.5 + vLLM |

**Removed from matrix:** AITW, Mobile-World, ScreenSpot-v2 mobile-only subset.

## Model & inference

| Mode | Backend | Notes |
|------|---------|-------|
| Offline | `lmms_eval --model gui_owl` | `visual_prune_policy=none`, `GEOPACT_SESSION_KV=1` (default) |
| Online | `scripts/cross_benchmark/serve_guiowl_8b_vllm.sh` | `max-model-len=32768`, `limit-mm-per-prompt image=5` |

Weights: `python scripts/download_omini_models.py --models-dir models GUI-Owl-1.5-8B-Instruct`

Prompt / action:
- **OmniGUI mainline (GeoPACT / KV / pruning):** `task_v2_qwen3_omni_novideo` — see [`qwen_dialogue_kv_eval_protocol.md`](qwen_dialogue_kv_eval_protocol.md)
- **Single-step cross-benchmark:** Mobile-Agent-v3.5 `mobile_use` (`gui_owl_native_messages.py`)
- **Optional OmniGUI compare (not main table):** `task_guiowl_omnigui_native_mt4_8b`

## Phase 0 — OmniGUI (2572 steps, image)

| Item | Value |
|------|-------|
| Data | `data/merged_sorted.json` + `data/benchmark/` |
| **Mainline task** | `task_v2_qwen3_omni_novideo` (`qwen_history6_strict_v2`) |
| **Compare task (optional)** | `task_guiowl_omnigui_native_mt4_8b` |
| Metrics | TM, EM, GP, SR (+ latency) |
| Preflight | `python scripts/preflight_cross_benchmark.py --baseline-only --require-8b` |
| Launch (DSW mainline) | `bash scripts/run_history6_8b_limit100_phase1.sh` → smoke → full |
| Legacy native baseline | `bash scripts/run_baseline_native_mt4_8b_full.sh` |

## Phase 1 — Offline data download

```powershell
# Tier 1 ~22 GB
powershell -ExecutionPolicy Bypass -File scripts\download_to_e.ps1

# Tier 2 +46 GB AndroidControl screenshots
powershell -ExecutionPolicy Bypass -File scripts\download_to_e.ps1 -Tier 2

python scripts/verify_cross_benchmark_data.py --strict --repo-root $REPO
```

| Dataset | HF source | Local path |
|---------|-----------|------------|
| OmniGUI | `XIAOCHENLIN00zz/OmniGUI` | `data/merged_sorted.json` |
| ScreenSpot-Pro | `likaixin/ScreenSpot-Pro` | `data/screenspot_pro/` |
| ScreenSpot-v2 | `OS-Copilot/ScreenSpot-v2` | `data/screenspot_v2/` |
| MMBench-GUI | `OpenGVLab/MMBench-GUI` | `data/mmbench_gui/` |
| AndroidControl test | `endlesswinds/AndroidControl-JSON-Splits` | `data/android_control/` |

## Phase 2 — Grounding (ScreenSpot-Pro / ScreenSpot-v2)

| Task | Metric | Build |
|------|--------|-------|
| `task_screenspot_pro_guiowl` | center_acc | `build_screenspot_pro_manifest.py` |
| `task_screenspot_v2_guiowl` | center_acc, domain macro | `build_screenspot_v2_full_manifest.py` |

## Phase 3 — MMBench-GUI (L1/L2 offline)

| Task | Metric | Data |
|------|--------|------|
| `task_mmbench_gui_l1_guiowl` | mmbench_gui_l1_acc | `data/mmbench_gui/l1_eval_index.json` |
| `task_mmbench_gui_l2_guiowl` | mmbench_gui_l2_acc | `data/mmbench_gui/l2_eval_index.json` |

Build: `python scripts/build_mmbench_gui_manifest.py`

## Phase 4 — AndroidControl (exact match)

| Task | Metric | Data |
|------|--------|------|
| `task_android_control_high_guiowl` | exact_match | `data/android_control/eval_steps.json` + images |

## Phase 5 — AndroidWorld (online, DSW)

Metric: **Task SR** (target 69.0%). Script: `scripts/cross_benchmark/run_androidworld_guiowl15.sh`

## Pruning gate

Enable `visual_prune_policy` / SSPC arms only after OmniGUI + ScreenSpot-Pro + ScreenSpot-v2 baselines reproduce within tolerance.

## DSW 开箱流程（上传后）

合并 `data/` + `models/` 到单一 `REPO` 后：

```bash
export REPO=${REPO:-$(pwd)}   # 或你的 DSW 路径
bash scripts/post_upload_bootstrap.sh
bash scripts/run_cross_benchmark_offline_matrix_8b.sh
```

Windows 本地校验（上传前）：

```powershell
.\scripts\post_upload_bootstrap.ps1 -RepoRoot $REPO
```

| 步骤 | 脚本 | 作用 |
|------|------|------|
| Bootstrap | `post_upload_bootstrap.sh` | 构建 manifest、patch AC 路径、strict 校验、`--offline-matrix` preflight |
| 全矩阵 | `run_cross_benchmark_offline_matrix_8b.sh` | OmniGUI → SS-Pro → SS-v2 → MMBench L1/L2 → AndroidControl |
| 汇总 | `scripts/cross_benchmark/collect_cross_benchmark_results.py` | 更新 baseline 表 |
| 在线 | `scripts/cross_benchmark/run_androidworld_guiowl15.sh` | AndroidWorld Task SR |

单基准 launcher：

| Benchmark | Script | Output |
|-----------|--------|--------|
| OmniGUI | `run_baseline_native_mt4_8b_full.sh` | `outputs/guiowl_native_mt4_8b_full_baseline` |
| ScreenSpot-Pro | `run_screenspot_pro_guiowl_8b.sh` | `outputs/screenspot_pro_guiowl_8b` |
| ScreenSpot-v2 | `run_screenspot_v2_guiowl_8b.sh` | `outputs/screenspot_v2_guiowl_8b` |
| MMBench L1 | `run_mmbench_gui_l1_guiowl_8b.sh` | `outputs/mmbench_gui_l1_guiowl_8b` |
| MMBench L2 | `run_mmbench_gui_l2_guiowl_8b.sh` | `outputs/mmbench_gui_l2_guiowl_8b` |
| AndroidControl | `run_android_control_guiowl_8b.sh` | `outputs/android_control_guiowl_8b` |

冒烟（`LIMIT=5`）：

```bash
export LIMIT=5
bash scripts/run_android_control_guiowl_8b.sh
```

## Tier1 limit=100 冒烟（上传后链路验证）

Tier1 上传 + 8B 模型就绪后，对 5 个离线基准各跑 100 样本（AndroidControl 截图属 Tier2，本次跳过实测）：

```bash
export REPO=${REPO:-$(pwd)}
export EVAL_PYTHON=${EVAL_PYTHON:-python3}
export LIMIT=100

bash scripts/post_upload_bootstrap.sh
bash scripts/run_cross_benchmark_tier1_smoke_limit100.sh
python scripts/verify_tier1_smoke_outputs.py --root outputs/tier1_smoke_limit100
python scripts/test_android_control_eval.py   # scorer 单测 only
```

输出目录：`outputs/tier1_smoke_limit100/{omnigui,screenspot_v2,screenspot_pro,mmbench_l1,mmbench_l2}/`

Windows 远程启动（后台）：

```powershell
.\scripts\launch_tier1_smoke_limit100.ps1
.\scripts\launch_tier1_smoke_limit100.ps1 -BootstrapOnly
.\scripts\launch_tier1_smoke_limit100.ps1 -VerifyOnly
```

通过标准：5 个 job exit 0；`run.log` 无 traceback；指标非 NaN；`missing image` 接近 0。

## KV Session 与历史窗口（统一默认）

**默认环境**（[`geopact_yolo_eval_env.sh`](scripts/geopact_yolo_eval_env.sh)）：

```text
GEOPACT_SESSION_KV=1
GEOPACT_SESSION_KV_MODE=dialogue
GEOPACT_FORCE_EPISODE_ORDER=1
GEOPACT_VISION_EMBED_CACHE=0
```

### 主线 OmniGUI（剪枝 / Module C）

- Task：`task_v2_qwen3_omni_novideo`（`qwen_history6_strict_v2`）
- 历史：**最近 6 步** `user(截图)+assistant(GT)` + 当前 `user`；无 `Previous actions:` 文本
- 滚窗：step ≥ 7 时最老一步 **图+GT 完全消失**；触发 `rolling window reset` + full prefill
- 实现：[`history_window.py`](lmms_eval/tasks/history_window.py) + [`qwen_doc_to_messages`](lmms_eval/tasks/agentcpm_gui/utils.py)

### 跨基准矩阵（单步基准 + OmniGUI 双轨）

| 数据集 | 样例形态 | SESSION_KV | Prompt / Task |
|--------|----------|------------|---------------|
| **OmniGUI（主线）** | 多步 history≤6 | on（dialogue suffix） | `task_v2_qwen3_omni_novideo` |
| **OmniGUI（对照，可选）** | 多步 native_mt4 | on | `task_guiowl_omnigui_native_mt4_8b` — **不与主线混报** |
| **ScreenSpot / MMBench** | 1 step / sample | on（step=0 no-op） | 官方 `*_guiowl` grounding / QA |
| **AndroidControl** | 1 step / sample（单图单步） | on（no-op） | `task_android_control_high_guiowl` |

单步 task 在 yaml 中设 `supports_session_kv: false`；registry 保证 AC 不因 manifest `action_history` 误开 suffix。

审计：

```bash
python scripts/audit_history_window_strict.py
python scripts/test_gui_owl_dialogue_kv_suffix.py
```

可选 KV off 对照：`GEOPACT_SESSION_KV=0 bash scripts/run_cross_benchmark_kv_ablation_limit100.sh`

详见 [`docs/qwen_dialogue_kv_eval_protocol.md`](docs/qwen_dialogue_kv_eval_protocol.md)。
