# Job 171 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 256,407 source records · 0 collector errors.

Observed window: 2026-09-03T04:16:28.429013Z to 2026-09-03T04:18:00.447726Z (92.0 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 0 / 0 | 0.721 | 202 | 0 | 3.69e-08 | 0.00172 |
| gpu-nodes-1 | trainer | 8/8 | 0 / 0 | 0.721 | 202 | 0 | 2.14e-08 | 0.00162 |
| gpu-nodes-2 | trainer | 8/8 | 0 / 0 | 0.721 | 201 | 0 | 2.96e-08 | 0.00173 |
| gpu-nodes-3 | trainer | 8/8 | 0 / 0 | 0.721 | 204 | 0 | 2.55e-08 | 0.0016 |

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 0 | ec52f3ef: 0 | — |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_0/1: 3.69e-08 | mlx5_3/1: 3.69e-08 | 0.000499 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 042b468a/link-0: 0 | ec52f3ef/link-9: 0 | — |
| gpu-nodes-1 | gpu_utilization (%) | 0606304d: 0 | f2fbb0d1: 0 | — |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_12/1: 1.95e-08 | mlx5_13/1: 3.42e-08 | 0.227 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-0: 0 | f2fbb0d1/link-9: 0 | — |
| gpu-nodes-2 | gpu_utilization (%) | 01f65707: 0 | f453c594: 0 | — |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_0/1: 2.79e-08 | mlx5_13/1: 4.18e-08 | 0.156 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 01f65707/link-0: 0 | f453c594/link-9: 0 | — |
| gpu-nodes-3 | gpu_utilization (%) | 11d63c84: 0 | 7471b575: 0 | — |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_1/1: 2.54e-08 | mlx5_2/1: 2.66e-08 | 0.016 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 11d63c84/link-0: 0 | 7471b575/link-9: 0 | — |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
