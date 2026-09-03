# Job 177 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 1,294,332 source records · 0 collector errors.

Observed window: 2026-09-03T04:56:41.919256Z to 2026-09-03T05:02:43.299131Z (361.4 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 9.2 / 100 | 62.9 | 594 | 0.0221 | 2.01e-05 | 0.0222 |
| gpu-nodes-1 | trainer | 8/8 | 7.25 / 89.5 | 57.4 | 584 | 0.0152 | 2.01e-05 | 0.0275 |
| gpu-nodes-2 | trainer | 8/8 | 10.9 / 100 | 63.1 | 620 | 0.0221 | 2.02e-05 | 0.0276 |
| gpu-nodes-3 | trainer | 8/8 | 10.3 / 100 | 57.4 | 626 | 0.0153 | 2.04e-05 | 0.0219 |

## Failures and coverage

- Slurm allocation FAILED (1:0); the allocation is not qualified.
No collector-error records observed; this alone does not establish complete coverage.

Invalid intervals among summarized counters: **945** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 6.1 | 54f332eb: 13 | 0.262 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 1.77e-05 | mlx5_1/1: 2.06e-05 | 0.0452 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-7: 0.0011 | 0f5cccee/link-6: 0.00172 | 0.0884 |
| gpu-nodes-1 | gpu_utilization (%) | 0606304d: 5.32 | 5952930a: 11.4 | 0.32 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.77e-05 | mlx5_1/1: 2.06e-05 | 0.045 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-11: 0.000738 | f2fbb0d1/link-14: 0.00096 | 0.0524 |
| gpu-nodes-2 | gpu_utilization (%) | 86218e11: 9.59 | e00780e1: 12 | 0.0684 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_4/1: 1.78e-05 | mlx5_1/1: 2.06e-05 | 0.0452 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 83015363/link-7: 0.00111 | 86218e11/link-8: 0.00171 | 0.089 |
| gpu-nodes-3 | gpu_utilization (%) | 1d061bc6: 5.11 | 6a8efa14: 13 | 0.214 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_4/1: 1.79e-05 | mlx5_1/1: 2.08e-05 | 0.0452 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 11d63c84/link-11: 0.000747 | 6a8efa14/link-14: 0.000937 | 0.0503 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
