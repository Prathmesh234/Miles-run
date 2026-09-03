# Job 154 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 1,156,432 source records · 0 collector errors.

Observed window: 2026-09-03T01:37:27.105571Z to 2026-09-03T01:42:54.163925Z (327.1 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 0.26 / 0 | 5.19 | 257 | 0.000131 | 1.62e-05 | 0.0404 |
| gpu-nodes-1 | trainer | 8/8 | 0.00538 / 0 | 5.23 | 257 | 0.00013 | 1.64e-05 | 0.00197 |
| gpu-nodes-2 | rollout | 8/8 | 0.369 / 1 | 11.3 | 252 | 0.000129 | 1.63e-05 | 0.00159 |
| gpu-nodes-3 | rollout | 8/8 | 0.367 / 0 | 11.3 | 260 | 0.000129 | 1.57e-05 | 0.0016 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.
- telemetry/sync-grpo-v10/gpu-nodes-0/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-0/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-0/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-0/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-1/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-1/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-1/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-1/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-2/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-2/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-2/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-2/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-3/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-3/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-3/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-3/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v10/gpu-nodes-3/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v10/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v10/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v10/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v10/gpu-nodes-3/lustre.jsonl: **partial**.

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
| gpu-nodes-0 | gpu_utilization (%) | 28ee82ef: 0.00336 | 05b02219: 0.349 | 0.495 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_3/1: 1.61e-05 | mlx5_12/1: 1.62e-05 | 0.000817 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 0f5cccee/link-5: 6.86e-06 | 042b468a/link-0: 7.7e-06 | 0.0245 |
| gpu-nodes-1 | gpu_utilization (%) | 0606304d: 0.00331 | e1134404: 0.00993 | 0.428 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.64e-05 | mlx5_12/1: 1.65e-05 | 0.000704 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 1bc17818/link-5: 6.77e-06 | 5952930a/link-1: 7.61e-06 | 0.024 |
| gpu-nodes-2 | gpu_utilization (%) | 86218e11: 0.224 | 7872c739: 0.411 | 0.153 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_0/1: 1.63e-05 | mlx5_3/1: 1.63e-05 | 0.000288 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | f453c594/link-5: 6.73e-06 | 86218e11/link-0: 7.57e-06 | 0.0258 |
| gpu-nodes-3 | gpu_utilization (%) | 11d63c84: 0.22 | 1bd6c6e0: 0.451 | 0.166 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_1/1: 1.57e-05 | mlx5_13/1: 1.58e-05 | 0.000313 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 1bd6c6e0/link-5: 6.72e-06 | 7471b575/link-1: 7.57e-06 | 0.0255 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
