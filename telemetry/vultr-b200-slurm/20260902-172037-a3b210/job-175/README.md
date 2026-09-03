# Job 175 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 1,308,985 source records · 0 collector errors.

Observed window: 2026-09-03T04:33:22.748942Z to 2026-09-03T04:39:27.216463Z (364.5 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 8/8 | 10.5 / 100 | 63.2 | 590 | 0.0242 | 2.03e-05 | 0.0275 |
| gpu-nodes-1 | trainer | 8/8 | 6.3 / 68 | 57.4 | 574 | 0.0164 | 2.02e-05 | 0.0274 |
| gpu-nodes-2 | trainer | 8/8 | 6.78 / 70 | 63.3 | 616 | 0.0239 | 1.99e-05 | 0.0275 |
| gpu-nodes-3 | trainer | 8/8 | 7.18 / 86.3 | 57.4 | 631 | 0.0162 | 1.97e-05 | 0.0167 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.
- telemetry/resume-replay-v3/gpu-nodes-0/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-0/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-0/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-0/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-1/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-1/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-1/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-1/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-2/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-2/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-2/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-2/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-3/nvidia-smi.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-3/nvlink.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-3/infiniband.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-3/cpu-memory-numa.jsonl: **partial**.
- telemetry/resume-replay-v3/gpu-nodes-3/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v3/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v3/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v3/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/lustre-resume-replay-v3/gpu-nodes-3/lustre.jsonl: **partial**.

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
| gpu-nodes-0 | gpu_utilization (%) | 54f332eb: 9.09 | ec52f3ef: 12.2 | 0.0781 |
| gpu-nodes-0 | ib_rail_tx (GB/s) | mlx5_4/1: 1.79e-05 | mlx5_1/1: 2.07e-05 | 0.045 |
| gpu-nodes-0 | nvlink_link_tx (GB/s) | 2941c934/link-12: 0.00117 | 042b468a/link-0: 0.00184 | 0.0934 |
| gpu-nodes-1 | gpu_utilization (%) | 2ab0b43f: 3.74 | 1bc17818: 9.11 | 0.229 |
| gpu-nodes-1 | ib_rail_tx (GB/s) | mlx5_4/1: 1.78e-05 | mlx5_1/1: 2.06e-05 | 0.0449 |
| gpu-nodes-1 | nvlink_link_tx (GB/s) | 0606304d/link-12: 0.000792 | f2fbb0d1/link-16: 0.00102 | 0.052 |
| gpu-nodes-2 | gpu_utilization (%) | 83015363: 6.11 | 01f65707: 8.16 | 0.0936 |
| gpu-nodes-2 | ib_rail_tx (GB/s) | mlx5_4/1: 1.76e-05 | mlx5_1/1: 2.03e-05 | 0.0451 |
| gpu-nodes-2 | nvlink_link_tx (GB/s) | 83015363/link-12: 0.00116 | 86218e11/link-0: 0.00172 | 0.0942 |
| gpu-nodes-3 | gpu_utilization (%) | 11d63c84: 5.1 | 70057fe0: 10.4 | 0.248 |
| gpu-nodes-3 | ib_rail_tx (GB/s) | mlx5_4/1: 1.73e-05 | mlx5_1/1: 2e-05 | 0.045 |
| gpu-nodes-3 | nvlink_link_tx (GB/s) | 11d63c84/link-12: 0.000778 | 7471b575/link-0: 0.00102 | 0.0523 |

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
