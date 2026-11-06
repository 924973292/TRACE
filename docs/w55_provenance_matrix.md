# W55 provenance matrix (2026-08-10)

Published TRACE cell → source run → policy flags → runtime engagement.
This file is the standalone adopted-cell ledger. Matching job rows:
[configs/jobs/INDEX.md](../configs/jobs/INDEX.md).

Flags were read from the launching TSV row (`launcher.log` holds only infra
env). "ret" counts `crs_nested_*` audit keys in the source `run.log`; single-
step runs never print their `crs_*` audit (blind spot proven by the W55 ±CRS
hash A/B at LIMIT=400: 0/300 identical keep hashes and a +21.25 accuracy gap
on SSv2, 0/300 and +9.50 on SSPro).

| Published cell | Source run (`logs/...`) | CRS flags | Engagement |
|---|---|---|---|
| SSv2 r10 = 73.90 | `xds8b_tune/screenspot_v2/covdose50_r10` | CRS=1 stride ρ=0.50 αcal | live (dose ladder + A/B) |
| SSv2 r5 = 55.19 | `xds8b_tune/screenspot_v2/a2fix_r5` | CRS=1 stride ρ=0.30 α=2 | live |
| SSPro r10 = 37.63 | `xds8b_tune/screenspot_pro/a4fix_r10` | CRS=1 stride ρ=0.50 α=4 | live (A/B) |
| SSPro r5 = 20.30 | `xds8b_tune/screenspot_pro/a4fix_r5` | CRS=1 stride ρ=0.30 α=4 | live |
| MMBench r10 = 50.47 | `xds8b_tune/mmbench_l2/a4fix_r10` | CRS=1 stride ρ=0.30 α=4 | live |
| MMBench r5 = 30.88 | `xds8b_tune/mmbench_l2/a4fix_r5` | CRS=1 stride ρ=0.30 α=4 | live |
| OmniGUI mild EM = 48.91 | `xds8b_abl/omnigui/nested_rho05_c50_h10` | CRS=1 NESTED=1 ρ=0.05 | ret keys in log |
| OmniGUI tight EM = 43.08 | `xds8b_tune/omnigui/a2cap_c25_h05` | CRS=1 NESTED=1 frac=0.05 α=2cal | ret keys in log |
| M2W mild (task split) | `xds8b_tune/mind2web_task/a6fix_c50_h10` | CRS=1 NESTED=1 frac=0.10 ρ=0.50 α=6 | ret keys in log |
| M2W tight (task/web/dom) | `xds8b_tune/mind2web_*/a6nocov_c25_h05` | CRS=1 NESTED=1 frac=0.05 α=6 | ret keys in each log |
| AC mild = 60.20 | `xds8b_tune/android_control/covdose_rho50_c50_h10` | CRS=1 NESTED=1 ρ=0.50 | ret keys in log |
| AC tight = 57.06 | `xds8b_tune/android_control/rho10cov50_c25_h05` | CRS=1 NESTED=1 frac=0.10 ρ=0.50 | ret keys in log |

Correction (W56, 2026-08-10): the AC tight cell was first mis-attributed to
`covdose_rho70_c25_h05` via a value-grep that only hit the mild cell. A full
HistPrune-protocol rescore of the tune tree (`scripts/_w56_ac_rescore.sh`)
identifies the true anchor as `rho10cov50_c25_h05` — Type/Ground/StepSR
82.06/65.50/57.06 match the published row digit-for-digit, and its ρ_cov =
0.50 is inside the declared public ladder. The earlier "out-of-ladder 0.70"
concern is void: no adopted cell uses ρ_cov = 0.70 (rho70 scores 56.58 and was
not adopted, which is itself evidence of protocol discipline).

Verdict: **S1**. No main-table rerun. The `w53_abl4a_mass` and
`w53_transfer_m2w` leftovers completed and were backfilled (Table 4a mass×M2W
= −15.50; `transfer.tex` Mind2Web column now three-split pooled). The only
cells still measuring are the Table 4b retirement stride twins
(`scripts/xds_jobs/w55_nrep2.tsv`, correctly anchored, running on 13543).
