# Datasets

TRACE uses six frozen offline benchmarks. Download them yourself; this repo does not ship images. Expected layout is under repo `data/`.

| Benchmark | Hugging Face / source | Local layout |
|-----------|----------------------|--------------|
| OmniGUI | `XIAOCHENLIN00zz/OmniGUI` | `data/merged_sorted.json`, `data/benchmark/` |
| ScreenSpot-v2 | `OS-Copilot/ScreenSpot-v2` | `data/screenspot_v2/` |
| ScreenSpot-Pro | `likaixin/ScreenSpot-Pro` | `data/screenspot_pro/` |
| MMBench-GUI L1+L2 | `OpenGVLab/MMBench-GUI` | `data/mmbench_gui/` |
| AndroidControl | `endlesswinds/AndroidControl-JSON-Splits` + screenshot archive | `data/android_control/` |
| Mind2Web | SeeClick test annots + screenshot dump | `data/mind2web/` |

## OmniGUI

```bash
huggingface-cli download XIAOCHENLIN00zz/OmniGUI --repo-type dataset --local-dir data
```

## ScreenSpot / MMBench / AndroidControl

```bash
python scripts/build_screenspot_v2_full_manifest.py
python scripts/build_screenspot_pro_manifest.py
python scripts/build_mmbench_gui_manifest.py
python scripts/build_android_control_manifest.py
python scripts/download_android_control_images.py --repo-root . --skip-download   # if zip already extracted
```

AndroidControl screenshots are tens of GB. JSON splits are enough only for dry runs.

## Mind2Web (SeeClick dump)

Paper eval uses a **filtered** conversion of the official SeeClick Mind2Web test splits, not the raw unfiltered JSON.

Place:

```text
data/mind2web/mind2web_annots/mind2web_data_test_{task,website,domain}.json
data/mind2web/ming2web_images/{annotation_id}-{action_uid}.jpg
```

(`mind2web_images/` is accepted as an alias; the historical dump directory is spelled `ming2web_images`.)

Then:

```bash
python scripts/build_mind2web_merged.py
```

The converter drops an entire episode if any step is missing its screenshot, has a degenerate bbox, or uses an op other than CLICK / TYPE / SELECT. On the dump used in the paper this yields about:

| Split | Kept episodes / steps |
|-------|------------------------|
| task | 242 / 2008 |
| website | 173 / 1343 |
| domain | 833 / 5429 |

This script is a reconstruction of a converter that was not committed in the research tree. If your SeeClick dump version differs, kept counts may drift; all arms still share the same filtered set.

Task YAML `task_mind2web_qwen3_omni_task` reads `data/mind2web/merged_sorted_task.json`.

## Verify

```bash
python scripts/verify_cross_benchmark_data.py --strict --repo-root .
python scripts/preflight_cross_benchmark.py --offline-matrix
```

## Models

See [../models/README.md](../models/README.md). LIP requires OmniParser v2 `icon_detect/model.pt` via `OMNIPARSER_ICON_WEIGHTS`.

Task names and flags: [CODEMAP.md](CODEMAP.md). Launch: [reproduction.md](reproduction.md). Historical KV notes: [qwen_dialogue_kv_eval_protocol.md](qwen_dialogue_kv_eval_protocol.md) (env flags still apply; ignore cluster launcher names).
