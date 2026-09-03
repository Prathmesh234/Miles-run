# Job 154 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 3,563,435 source records · 0 collector errors.

Observed window: 2026-09-03T01:37:27.105571Z to 2026-09-03T01:53:39.244866Z (972.1 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 8.03 / 98 | 67.4 | 600 | 0.355 | 0.0916 | 10.3 |
| gpu-nodes-1 | trainer | 8/8 | 7.93 / 100 | 56.6 | 541 | 0.295 | 0.056 | 11.1 |
| gpu-nodes-2 | rollout | 8/8 | 5.75 / 72 | 142 | 682 | 0.515 | 0.0302 | 0.0187 |
| gpu-nodes-3 | rollout | 8/8 | 5.83 / 70 | 142 | 745 | 0.527 | 1.56e-05 | 0.0186 |

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 5.38 | ec52f3ef: 8.89 | 0.131 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_9/1: 0.0612 | mlx5_1/1: 0.304 | 0.877 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 28ee82ef/link-8: 0.0189 | 042b468a/link-13: 0.0211 | 0.03 |
| gpu-nodes-1 | gpu_utilization (%) | 2ab0b43f: 7.43 | 15fdcc49: 8.71 | 0.0438 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 0.056 | mlx5_1/1: 0.0561 | 0.000278 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | d1b19d48/link-14: 0.0159 | 0606304d/link-6: 0.0173 | 0.0215 |
| gpu-nodes-2 | gpu_utilization (%) | 83015363: 5.58 | 7872c739: 5.94 | 0.0213 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_0/1: 5.28e-06 | mlx5_1/1: 0.121 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 983d90f7/link-7: 0.0279 | 7872c739/link-2: 0.0294 | 0.0111 |
| gpu-nodes-3 | gpu_utilization (%) | 1d061bc6: 5.67 | 6a8efa14: 6.12 | 0.0294 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 5.13e-06 | mlx5_1/1: 4.73e-05 | 1.16 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 6a8efa14/link-5: 0.0226 | 272aac0b/link-9: 0.032 | 0.12 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
