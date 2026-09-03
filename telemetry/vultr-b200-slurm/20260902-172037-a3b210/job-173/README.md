# Job 173 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 1,141,504 source records · 0 collector errors.

Observed window: 2026-09-03T04:23:37.358091Z to 2026-09-03T04:28:55.079764Z (317.7 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 3.72 / 7 | 57.4 | 264 | 0.00018 | 2.29e-05 | 0.0222 |
| gpu-nodes-1 | trainer | 8/8 | 3.87 / 7.05 | 57.4 | 267 | 0.000181 | 2.29e-05 | 0.0221 |
| gpu-nodes-2 | trainer | 8/8 | 3.85 / 6 | 57.4 | 261 | 0.000179 | 2.27e-05 | 0.0275 |
| gpu-nodes-3 | trainer | 8/8 | 5.1 / 13 | 57.4 | 271 | 0.00018 | 2.29e-05 | 0.0382 |

## Failures and coverage

- Slurm allocation FAILED (1:0); the allocation is not qualified.
No collector-error records observed; this alone does not establish complete coverage.

Invalid intervals among summarized counters: **0** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 2.15 | 05b02219: 5.79 | 0.274 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 2.02e-05 | mlx5_1/1: 2.33e-05 | 0.045 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-10: 9.85e-06 | 2941c934/link-7: 1.01e-05 | 0.00809 |
| gpu-nodes-1 | gpu_utilization (%) | 15fdcc49: 2.69 | f2fbb0d1: 5.28 | 0.201 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 2.02e-05 | mlx5_1/1: 2.34e-05 | 0.0451 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | d1b19d48/link-10: 9.88e-06 | 1bc17818/link-8: 1.02e-05 | 0.00778 |
| gpu-nodes-2 | gpu_utilization (%) | 86218e11: 1.95 | 7872c739: 4.84 | 0.215 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_4/1: 2e-05 | mlx5_1/1: 2.32e-05 | 0.0448 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 983d90f7/link-10: 9.79e-06 | e00780e1/link-8: 1.01e-05 | 0.00819 |
| gpu-nodes-3 | gpu_utilization (%) | 138eab1f: 3.74 | 11d63c84: 6.23 | 0.176 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_4/1: 2.01e-05 | mlx5_1/1: 2.33e-05 | 0.0452 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 138eab1f/link-10: 9.85e-06 | 11d63c84/link-8: 1.02e-05 | 0.00835 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
