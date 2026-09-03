# Job 175 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 2,100,864 source records · 0 collector errors.

Observed window: 2026-09-03T04:33:22.748942Z to 2026-09-03T04:42:56.775686Z (574.0 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 8.51 / 100 | 64.7 | 590 | 0.0723 | 0.0344 | 0.0275 |
| gpu-nodes-1 | trainer | 8/8 | 5.89 / 67 | 57.4 | 574 | 0.0599 | 0.0342 | 0.0274 |
| gpu-nodes-2 | trainer | 8/8 | 6.47 / 72 | 64.8 | 616 | 0.0722 | 0.0343 | 0.0275 |
| gpu-nodes-3 | trainer | 8/8 | 6.72 / 84 | 57.4 | 631 | 0.0602 | 0.0344 | 0.0167 |

## Failures and coverage

- Slurm allocation FAILED (1:0); the allocation is not qualified.
No collector-error records observed; this alone does not establish complete coverage.

Invalid intervals among summarized counters: **315** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

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
| gpu-nodes-0 | gpu_utilization (%) | 54f332eb: 6.7 | ec52f3ef: 9.87 | 0.0981 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_12/1: 0.0343 | mlx5_1/1: 0.0344 | 0.000336 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-12: 0.00376 | 042b468a/link-0: 0.00453 | 0.0442 |
| gpu-nodes-1 | gpu_utilization (%) | 2ab0b43f: 4.53 | 1bc17818: 7.51 | 0.162 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_9/1: 0.0342 | mlx5_0/1: 0.0343 | 0.000302 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-5: 0.00313 | f2fbb0d1/link-0: 0.00352 | 0.023 |
| gpu-nodes-2 | gpu_utilization (%) | e00780e1: 5.26 | 01f65707: 7.79 | 0.119 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_4/1: 0.0343 | mlx5_1/1: 0.0343 | 0.000187 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 83015363/link-12: 0.00376 | 86218e11/link-0: 0.00446 | 0.0438 |
| gpu-nodes-3 | gpu_utilization (%) | 7471b575: 5.64 | 70057fe0: 8.43 | 0.156 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_12/1: 0.0344 | mlx5_0/1: 0.0344 | 0.000129 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 11d63c84/link-12: 0.00316 | 1bd6c6e0/link-16: 0.0035 | 0.0236 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
