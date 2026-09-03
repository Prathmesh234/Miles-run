# Job 167 — telemetry summary

**Telemetry gate: OK** · Slurm: COMPLETED (0:0) · 3,378,085 source records · 0 collector errors.

Observed window: 2026-09-03T03:34:31.183255Z to 2026-09-03T03:49:52.346062Z (921.2 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 7.39 / 98 | 67.6 | 583 | 0.314 | 0.0745 | 10.3 |
| gpu-nodes-1 | trainer | 8/8 | 9.27 / 100 | 56.4 | 567 | 0.289 | 0.039 | 10.1 |
| gpu-nodes-2 | rollout | 8/8 | 5.64 / 71 | 142 | 705 | 0.523 | 0.0319 | 0.0194 |
| gpu-nodes-3 | rollout | 8/8 | 4.03 / 28 | 142 | 754 | 0.484 | 1.66e-05 | 0.0186 |

## Failures and coverage

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 5.69 | cf39917e: 8.03 | 0.0935 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_9/1: 0.0426 | mlx5_1/1: 0.297 | 1.13 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-16: 0.0168 | 042b468a/link-2: 0.0183 | 0.0189 |
| gpu-nodes-1 | gpu_utilization (%) | 1bc17818: 8.41 | 15fdcc49: 10.2 | 0.0642 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_13/1: 0.039 | mlx5_1/1: 0.0391 | 0.000476 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-1: 0.0157 | 5952930a/link-7: 0.0165 | 0.00893 |
| gpu-nodes-2 | gpu_utilization (%) | 86218e11: 5.51 | f453c594: 5.83 | 0.0166 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_9/1: 5.52e-06 | mlx5_1/1: 0.128 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 983d90f7/link-7: 0.0285 | 7872c739/link-2: 0.0297 | 0.01 |
| gpu-nodes-3 | gpu_utilization (%) | 6a8efa14: 3.75 | 272aac0b: 4.45 | 0.0518 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_9/1: 5.38e-06 | mlx5_1/1: 5.03e-05 | 1.17 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 6a8efa14/link-5: 0.0211 | 272aac0b/link-9: 0.0293 | 0.114 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.

## Training context

Observed optimizer updates: **2**. These are log receipts, not proof of complete resume fidelity or held-out quality.

| Step | UTC | Train reward | Grad norm | Trainer time (s) | Weight update (s) |
|---:|---|---:|---:|---:|---:|
| 0 | 2026-09-03T03:46:39.868Z | 0.6875 | 0.645 | 145.4 | 5.419 |
| 1 | 2026-09-03T03:48:25.223Z | 0.75 | 1.386 | 22 | 3.151 |

Training reward is not held-out Terminal-Bench accuracy. MTP acceptance rate is omitted because this summary does not independently validate that field.
