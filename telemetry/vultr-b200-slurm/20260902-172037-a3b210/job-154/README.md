# Job 154 — telemetry summary

**Telemetry gate: OK** · Slurm: FAILED (1:0) · 3,657,250 source records · 0 collector errors.

Observed window: 2026-09-03T01:37:27.105571Z to 2026-09-03T01:54:13.305656Z (1006.2 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 8.98 / 100 | 67.4 | 600 | 0.409 | 0.0992 | 10.3 |
| gpu-nodes-1 | trainer | 8/8 | 8.6 / 100 | 56.6 | 541 | 0.353 | 0.0549 | 11.1 |
| gpu-nodes-2 | rollout | 8/8 | 5.78 / 71 | 142 | 682 | 0.578 | 0.0391 | 0.0187 |
| gpu-nodes-3 | rollout | 8/8 | 5.88 / 70 | 142 | 745 | 0.541 | 1.87e-05 | 0.0186 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.

Invalid intervals among summarized counters: **630** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

| Node | Observed health-counter series | Always zero | Unchanged nonzero | Changed/reset |
|---|---:|---:|---:|---:|
| gpu-nodes-0 | 120 | 112 | 8 | 0 |
| gpu-nodes-1 | 120 | 112 | 8 | 0 |
| gpu-nodes-2 | 120 | 120 | 0 | 0 |
| gpu-nodes-3 | 120 | 112 | 8 | 0 |

Unchanged nonzero values predate the observation window; they are not new errors during this run. These are only the ECC/IB counters actually collected. They do not establish XID, throttle, row-remap, PCIe or DCGM coverage.

## Largest entity differences

Lowest/highest time-mean within each node; descriptive differences, not hardware-fault diagnoses.

| Node | Metric | Lowest entity : mean | Highest entity : mean | Across-entity CV |
|---|---|---|---|---:|
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 5.52 | 54f332eb: 9.97 | 0.149 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_9/1: 0.0596 | mlx5_1/1: 0.376 | 1.06 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 28ee82ef/link-8: 0.0219 | 042b468a/link-13: 0.024 | 0.0256 |
| gpu-nodes-1 | gpu_utilization (%) | 2ab0b43f: 8.09 | 15fdcc49: 9.35 | 0.041 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 0.0549 | mlx5_1/1: 0.055 | 0.000321 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | d1b19d48/link-14: 0.0192 | 0606304d/link-6: 0.0206 | 0.018 |
| gpu-nodes-2 | gpu_utilization (%) | 83015363: 5.61 | 7872c739: 5.97 | 0.0212 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_0/1: 5.12e-06 | mlx5_1/1: 0.156 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 983d90f7/link-7: 0.0314 | 7872c739/link-2: 0.0329 | 0.00981 |
| gpu-nodes-3 | gpu_utilization (%) | 1d061bc6: 5.74 | 70057fe0: 6.16 | 0.0289 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 4.99e-06 | mlx5_1/1: 5.98e-05 | 1.27 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 6a8efa14/link-5: 0.023 | 272aac0b/link-9: 0.0329 | 0.125 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
