# Job 177 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 1,210,377 source records · 0 collector errors.

Observed window: 2026-09-03T04:56:41.919256Z to 2026-09-03T05:02:20.349929Z (338.4 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 7.71 / 100 | 58.7 | 594 | 0.0227 | 2.2e-05 | 0.0222 |
| gpu-nodes-1 | trainer | 8/8 | 6.34 / 68 | 57.4 | 584 | 0.0155 | 2.18e-05 | 0.0275 |
| gpu-nodes-2 | trainer | 8/8 | 10.1 / 100 | 63.1 | 620 | 0.0223 | 2.17e-05 | 0.0276 |
| gpu-nodes-3 | trainer | 8/8 | 10.4 / 100 | 57.4 | 626 | 0.016 | 2.14e-05 | 0.0219 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.
- telemetry/resume-replay-v4/gpu-nodes-0/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-0/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-0/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-0/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-1/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-1/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-1/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-1/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-2/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-2/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-2/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-2/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-3/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-3/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-3/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-3/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v4/gpu-nodes-3/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v4/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v4/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v4/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v4/gpu-nodes-3/lustre.jsonl: **partial**.

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
| gpu-nodes-0 | gpu_utilization (%) | 042b468a: 4.03 | 54f332eb: 11.8 | 0.355 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 1.94e-05 | mlx5_1/1: 2.25e-05 | 0.0453 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-7: 0.00114 | 0f5cccee/link-6: 0.00178 | 0.0882 |
| gpu-nodes-1 | gpu_utilization (%) | 0606304d: 4.1 | 5952930a: 10.7 | 0.396 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.92e-05 | mlx5_1/1: 2.23e-05 | 0.045 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-11: 0.000756 | f2fbb0d1/link-14: 0.000991 | 0.0523 |
| gpu-nodes-2 | gpu_utilization (%) | 86218e11: 8.5 | f453c594: 11.2 | 0.0863 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_4/1: 1.91e-05 | mlx5_1/1: 2.21e-05 | 0.0452 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 83015363/link-7: 0.00112 | 86218e11/link-8: 0.00173 | 0.0884 |
| gpu-nodes-3 | gpu_utilization (%) | 1d061bc6: 4.81 | 6a8efa14: 13.3 | 0.228 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_4/1: 1.88e-05 | mlx5_1/1: 2.19e-05 | 0.0452 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 11d63c84/link-11: 0.00078 | 6a8efa14/link-14: 0.000978 | 0.0503 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
