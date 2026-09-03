# Job 161 — telemetry summary

**Telemetry gate: FAIL** · Slurm: FAILED (1:0) · 3,226,229 source records · 4 collector errors.

Observed window: 2026-09-03T02:36:01.324988Z to 2026-09-03T02:50:52.503989Z (891.2 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 8.29 / 98 | 64.3 | 590 | 0.331 | 0.0772 | 10.1 |
| gpu-nodes-1 | trainer | 8/8 | 9.15 / 100 | 56.1 | 556 | 0.288 | 0.0389 | 10.4 |
| gpu-nodes-2 | rollout | 8/8 | 4.8 / 46 | 142 | 697 | 0.538 | 0.0332 | 0.0186 |
| gpu-nodes-3 | rollout | 8/8 | 4.88 / 49 | 142 | 752 | 0.501 | 1.72e-05 | 0.0182 |

## Failures and coverage

- 4 collector-error records.
- Slurm allocation FAILED (1:0); the allocation is not qualified.
- telemetry/sync-grpo-v12/gpu-nodes-0/cpu-memory-numa.jsonl: 14.8764s sampling gap exceeds the 12s heartbeat limit.
- telemetry/sync-grpo-v12/gpu-nodes-1/cpu-memory-numa.jsonl: 13.2964s sampling gap exceeds the 12s heartbeat limit.
- **gpu-nodes-0 / collector**: 2 × collection_failed; UTC 02:50:46.082108, 02:50:46.091616.
- **gpu-nodes-1 / collector**: 2 × collection_failed; UTC 02:50:42.984530, 02:50:42.994605.

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 5.59 | 05b02219: 9.61 | 0.134 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_12/1: 0.0432 | mlx5_1/1: 0.315 | 1.16 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-17: 0.0176 | 042b468a/link-17: 0.0194 | 0.0225 |
| gpu-nodes-1 | gpu_utilization (%) | d1b19d48: 8.59 | f2fbb0d1: 9.91 | 0.0531 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 0.0389 | mlx5_1/1: 0.0389 | 0.000312 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 15fdcc49/link-0: 0.0157 | 0606304d/link-6: 0.0165 | 0.0102 |
| gpu-nodes-2 | gpu_utilization (%) | 7872c739: 4.63 | 86218e11: 5.07 | 0.0265 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_9/1: 5.75e-06 | mlx5_1/1: 0.133 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | b159dd83/link-10: 0.0293 | 983d90f7/link-9: 0.0305 | 0.00973 |
| gpu-nodes-3 | gpu_utilization (%) | 11d63c84: 4.36 | 138eab1f: 5.34 | 0.0606 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 5.6e-06 | mlx5_1/1: 5.23e-05 | 1.17 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 6a8efa14/link-5: 0.0218 | 272aac0b/link-9: 0.0303 | 0.115 |

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
| 0 | 2026-09-03T02:47:46.404Z | 0.5625 | 0.5203 | 143.3 | 6.54 |
| 1 | 2026-09-03T02:49:31.495Z | 0.875 | 1.061 | 22.05 | 3.302 |

Training reward is not held-out Terminal-Bench accuracy. MTP acceptance rate is omitted because this summary does not independently validate that field.
