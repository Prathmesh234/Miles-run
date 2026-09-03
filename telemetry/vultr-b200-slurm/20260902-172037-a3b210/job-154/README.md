# Job 154 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 2,360,709 source records · 0 collector errors.

Observed window: 2026-09-03T01:37:27.105571Z to 2026-09-03T01:48:15.745929Z (648.6 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 2.1 / 5 | 53.7 | 585 | 0.108 | 0.0155 | 0.0404 |
| gpu-nodes-1 | trainer | 8/8 | 1.92 / 4 | 53.7 | 541 | 0.106 | 1.36e-05 | 0.00438 |
| gpu-nodes-2 | rollout | 8/8 | 3.67 / 17.1 | 142 | 682 | 0.451 | 0.0153 | 0.0187 |
| gpu-nodes-3 | rollout | 8/8 | 3.34 / 13 | 142 | 745 | 0.44 | 1.3e-05 | 0.0186 |

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 1.74 | ec52f3ef: 2.43 | 0.106 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 1.3e-05 | mlx5_1/1: 0.124 | 2.64 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 28ee82ef/link-8: 0.00584 | 042b468a/link-13: 0.00628 | 0.0154 |
| gpu-nodes-1 | gpu_utilization (%) | e1134404: 1.54 | 5952930a: 2.31 | 0.135 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.32e-05 | mlx5_1/1: 1.37e-05 | 0.0125 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-8: 0.00576 | 1bc17818/link-15: 0.00603 | 0.00985 |
| gpu-nodes-2 | gpu_utilization (%) | 83015363: 3.27 | 7872c739: 3.99 | 0.0543 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_0/1: 7.95e-06 | mlx5_1/1: 0.0611 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | b159dd83/link-10: 0.0244 | 7872c739/link-10: 0.0258 | 0.0134 |
| gpu-nodes-3 | gpu_utilization (%) | 138eab1f: 3.11 | 70057fe0: 3.71 | 0.0565 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 7.72e-06 | mlx5_1/1: 2.91e-05 | 0.707 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 1bd6c6e0/link-11: 0.0213 | 272aac0b/link-9: 0.026 | 0.0611 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
