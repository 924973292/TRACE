# Paper job index

TSV format (same as `scripts/_xds_gpu_queue.sh`):

```
BENCHMARK|TASK|TRACK|EXP_ARM|RATIO|HIST_N|OUT|LIMIT|EXTRA_ENV
```

Rows were harvested from the research-repo campaign files and **deduplicated** on `(BENCHMARK, TASK, TRACK, EXP_ARM, OUT, EXTRA_ENV)`. GPU-port filenames were scheduling artifacts, not part of the protocol.

Unique rows in this extract: **1858** (831 source TSVs scanned, 175 duplicate rows dropped).

## Canonical files

| File | Rows | Use |
|------|------|-----|
| [main_8b.tsv](main_8b.tsv) | 304 | 8B tune / full / final grid (single- and multi-step TRACE + controls) |
| [baselines_pub.tsv](baselines_pub.tsv) | 209 | Published selectors (`nap_pubsel`, `pub_*`) |
| [ablation.tsv](ablation.tsv) | 278 | Module ladders, KV rewrite on/off, NCR geometry |
| [uitars.tsv](uitars.tsv) | 244 | UI-TARS-1.5-7B transfer |
| [scale_2b.tsv](scale_2b.tsv) | 692 | GUI-Owl-1.5-2B copy of the 8B grid |
| [efficiency.tsv](efficiency.tsv) | 51 | OmniGUI TTFT / anatomy |
| [mechanism_w43_w45.tsv](mechanism_w43_w45.tsv) | 28 | Retention / causal / nested probes |
| [p0_parity.tsv](p0_parity.tsv) | 32 | Recent-N and detector-parity |
| [selfcond.tsv](selfcond.tsv) | 5 | Self-conditioned OmniGUI |
| [other.tsv](other.tsv) | 15 | Gates / smokes (not adopted paper cells) |

Named originals with comments: [raw/](raw/). Prefer the canonical files for a third-party rerun; use `raw/` if you need the exact historical shard comments.

## Adopted 8B TRACE cells (w55 ledger)

Every published cell below has a matching job row.

| Published cell | Job file | `OUT` suffix |
|----------------|----------|----------------|
| SSv2 r10 = 73.90 | `main_8b.tsv` | `logs/xds8b_tune/screenspot_v2/covdose50_r10` |
| SSv2 r5 = 55.19 | `main_8b.tsv` | `logs/xds8b_tune/screenspot_v2/a2fix_r5` |
| SSPro r10 = 37.63 | `main_8b.tsv` | `logs/xds8b_tune/screenspot_pro/a4fix_r10` |
| SSPro r5 = 20.30 | `main_8b.tsv` | `logs/xds8b_tune/screenspot_pro/a4fix_r5` |
| MMBench r10 = 50.47 | `main_8b.tsv` | `logs/xds8b_tune/mmbench_l2/a4fix_r10` |
| MMBench r5 = 30.88 | `main_8b.tsv` | `logs/xds8b_tune/mmbench_l2/a4fix_r5` |
| OmniGUI mild EM = 48.91 | `ablation.tsv` | `logs/xds8b_abl/omnigui/nested_rho05_c50_h10` |
| OmniGUI tight EM = 43.08 | `main_8b.tsv` | `logs/xds8b_tune/omnigui/a2cap_c25_h05` |
| M2W mild (task) | `main_8b.tsv` | `logs/xds8b_tune/mind2web_task/a6fix_c50_h10` |
| M2W tight (task/web/dom) | `main_8b.tsv` | `logs/xds8b_tune/mind2web_*/a6nocov_c25_h05` |
| AC mild = 60.20 | `main_8b.tsv` | `logs/xds8b_tune/android_control/covdose_rho50_c50_h10` |
| AC tight = 57.06 | `main_8b.tsv` | `logs/xds8b_tune/android_control/rho10cov50_c25_h05` |

Flag commentary: [../../docs/w55_provenance_matrix.md](../../docs/w55_provenance_matrix.md).
Reproduction commands: [../../docs/reproduction.md](../../docs/reproduction.md).

Harvest report: [_harvest_report.json](_harvest_report.json).
