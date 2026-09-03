# Job 165 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 545,597 source records · 0 collector errors.

Observed window: 2026-09-03T03:26:18.553319Z to 2026-09-03T03:28:54.773819Z (156.2 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 0.446 / 0 | 5.19 | 260 | 0.000278 | 3.38e-05 | 0.00162 |
| gpu-nodes-1 | trainer | 8/8 | 0.0253 / 0 | 5.21 | 258 | 0.000278 | 3.48e-05 | 0.00162 |
| gpu-nodes-2 | rollout | 8/8 | 0.872 / 0 | 5.23 | 252 | 0.000276 | 3.46e-05 | 0.00194 |
| gpu-nodes-3 | rollout | 8/8 | 0.593 / 0 | 5.19 | 260 | 0.000278 | 3.38e-05 | 0.0016 |

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
| gpu-nodes-0 | gpu_utilization (%) | 28ee82ef: 0.014 | cf39917e: 0.706 | 0.75 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 3.38e-05 | mlx5_9/1: 3.39e-05 | 0.000861 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 0f5cccee/link-5: 1.45e-05 | 042b468a/link-0: 1.63e-05 | 0.0242 |
| gpu-nodes-1 | gpu_utilization (%) | 0606304d: 0.00699 | f2fbb0d1: 0.0769 | 0.861 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_0/1: 3.48e-05 | mlx5_4/1: 3.48e-05 | 0.000347 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 1bc17818/link-5: 1.46e-05 | 5952930a/link-1: 1.64e-05 | 0.0252 |
| gpu-nodes-2 | gpu_utilization (%) | 983d90f7: 0.694 | 01f65707: 1.38 | 0.29 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_9/1: 3.46e-05 | mlx5_0/1: 3.46e-05 | 0.000343 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | f453c594/link-5: 1.44e-05 | 86218e11/link-1: 1.62e-05 | 0.0256 |
| gpu-nodes-3 | gpu_utilization (%) | 7471b575: 0.217 | 70057fe0: 0.727 | 0.363 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_4/1: 3.38e-05 | mlx5_12/1: 3.38e-05 | 0.000566 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 1bd6c6e0/link-5: 1.46e-05 | 7471b575/link-1: 1.64e-05 | 0.0263 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
