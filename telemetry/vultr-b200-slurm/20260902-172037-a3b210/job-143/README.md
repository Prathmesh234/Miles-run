# Job 143 — telemetry summary

**Telemetry gate: FAIL** · Slurm: COMPLETED (0:0) · 3,323,471 source records · 12 collector errors.

Observed window: 2026-09-02T22:46:02.624843Z to 2026-09-02T23:03:12.245015Z (1029.6 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 6.57 / 86 | 64.9 | 616 | 0.256 | 0.0531 | 10.4 |
| gpu-nodes-1 | trainer | 8/8 | 7.19 / 96 | 56.6 | 549 | 0.182 | 0.0328 | 10.6 |
| gpu-nodes-2 | rollout | 8/8 | 6.09 / 75 | 142 | 685 | 0.567 | 0.0382 | 0.0178 |
| gpu-nodes-3 | rollout | 8/8 | 5.41 / 67 | 142 | 754 | 0.526 | 1.81e-05 | 0.0179 |

## Failures and coverage

- **gpu-nodes-0 / nvidia-smi**: 3 × timeout; UTC 23:02:31.784806, 23:02:41.126419, 23:02:54.319193.
- **gpu-nodes-0 / nvlink**: 4 × timeout; UTC 22:58:43.232631, 23:02:31.784806, 23:02:41.126419, 23:02:54.319193.
- **gpu-nodes-1 / nvidia-smi**: 2 × timeout; UTC 23:02:30.113331, 23:02:39.021293.
- **gpu-nodes-1 / nvlink**: 3 × timeout; UTC 22:58:42.348869, 23:02:30.113331, 23:02:39.021293.

Invalid intervals among summarized counters: **2520** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 4.38 | 05b02219: 7.83 | 0.149 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 0.0266 | mlx5_1/1: 0.238 | 1.32 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 042b468a/link-0: 0.0135 | 54f332eb/link-6: 0.0157 | 0.0249 |
| gpu-nodes-1 | gpu_utilization (%) | d1b19d48: 6.58 | f2fbb0d1: 7.77 | 0.0552 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 0.0328 | mlx5_13/1: 0.0329 | 0.00132 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-9: 0.00919 | 5952930a/link-6: 0.0117 | 0.086 |
| gpu-nodes-2 | gpu_utilization (%) | e00780e1: 5.79 | 86218e11: 6.33 | 0.0314 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_2/1: 4.96e-06 | mlx5_1/1: 0.153 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 983d90f7/link-7: 0.0308 | 7872c739/link-2: 0.0323 | 0.0102 |
| gpu-nodes-3 | gpu_utilization (%) | 1d061bc6: 5.2 | 138eab1f: 5.55 | 0.0211 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 4.82e-06 | mlx5_1/1: 5.8e-05 | 1.27 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 6a8efa14/link-5: 0.0223 | 272aac0b/link-9: 0.032 | 0.125 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.

## Training context

Observed optimizer updates: **3**. These are log receipts, not proof of complete resume fidelity or held-out quality.

| Step | UTC | Train reward | Grad norm | Trainer time (s) | Weight update (s) |
|---:|---|---:|---:|---:|---:|
| 0 | 2026-09-02T22:57:57.204Z | 0.5625 | 0.4187 | 175.5 | 8.833 |
| 1 | 2026-09-02T23:00:19.215Z | 0.625 | 0.9563 | 26.14 | 3.363 |
| 2 | 2026-09-02T23:01:44.946Z | 0.6875 | 0.8799 | 11.68 | 3.251 |

Training reward is not held-out Terminal-Bench accuracy. MTP acceptance rate is omitted because this summary does not independently validate that field.
