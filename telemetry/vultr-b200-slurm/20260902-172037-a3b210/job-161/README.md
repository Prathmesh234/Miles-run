# Job 161 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 2,357,324 source records · 0 collector errors.

Observed window: 2026-09-03T02:36:01.324988Z to 2026-09-03T02:46:48.446810Z (647.1 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 3.73 / 11 | 61.9 | 590 | 0.128 | 0.0155 | 0.0404 |
| gpu-nodes-1 | trainer | 8/8 | 4.57 / 27.6 | 53.7 | 556 | 0.111 | 1.36e-05 | 0.00486 |
| gpu-nodes-2 | rollout | 8/8 | 3.26 / 6 | 142 | 697 | 0.453 | 0.0153 | 0.0186 |
| gpu-nodes-3 | rollout | 8/8 | 3.27 / 8 | 142 | 752 | 0.435 | 1.3e-05 | 0.0182 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.
- telemetry/sync-grpo-v12/gpu-nodes-0/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-0/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-0/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-0/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-1/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-1/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-1/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-1/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-2/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-2/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-2/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-2/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-3/nvidia-smi.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-3/nvlink.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-3/infiniband.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-3/cpu-memory-numa.jsonl: **partial**.
- telemetry/sync-grpo-v12/gpu-nodes-3/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-3/lustre.jsonl: **partial**.

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
| gpu-nodes-0 | gpu_utilization (%) | 0f5cccee: 3.15 | ec52f3ef: 4.28 | 0.0924 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 1.3e-05 | mlx5_1/1: 0.124 | 2.64 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-17: 0.00679 | 042b468a/link-17: 0.00768 | 0.0291 |
| gpu-nodes-1 | gpu_utilization (%) | 15fdcc49: 4 | f2fbb0d1: 5.19 | 0.0819 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.32e-05 | mlx5_1/1: 1.37e-05 | 0.0123 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-8: 0.00603 | 5952930a/link-7: 0.00639 | 0.0109 |
| gpu-nodes-2 | gpu_utilization (%) | 7872c739: 3.01 | e00780e1: 3.4 | 0.0401 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_9/1: 7.93e-06 | mlx5_1/1: 0.0612 | 1.73 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | b159dd83/link-10: 0.0245 | 7872c739/link-10: 0.0258 | 0.0137 |
| gpu-nodes-3 | gpu_utilization (%) | 11d63c84: 2.5 | 138eab1f: 3.8 | 0.125 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_0/1: 7.69e-06 | mlx5_1/1: 2.91e-05 | 0.709 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 1bd6c6e0/link-11: 0.021 | 272aac0b/link-9: 0.0256 | 0.062 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
