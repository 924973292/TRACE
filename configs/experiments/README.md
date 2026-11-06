# Experiment snapshots

Paper jobs live in [`../jobs/`](../jobs/). This folder stores the **adopted** GUI-Owl-1.5-8B TRACE operating points in one place so you do not have to grep a 300-row TSV.

| File | Contents |
|------|----------|
| [adopted_8b.json](adopted_8b.json) | 12 published 8B TRACE cells: OUT path, EXTRA_ENV, published metric |

Launch one cell from the matching TSV row (`OUT` column) with:

```bash
source scripts/eval_env_defaults.sh
XDS_GPU=0 bash scripts/_xds_gpu_queue.sh configs/jobs/main_8b.tsv
```

or export the `env` map from the JSON and call `scripts/run_cross_dataset_track_one.sh`.

Full grids (baselines, ablations, UI-TARS, 2B, efficiency) remain in `configs/jobs/`. See [docs/reproduction.md](../../docs/reproduction.md).
