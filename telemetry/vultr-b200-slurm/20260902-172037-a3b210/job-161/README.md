# Job 161 — telemetry summary

**Telemetry gate: PARTIAL** · Slurm: RUNNING (0:0) · 4,421 source records · 0 collector errors.

Observed window: 2026-09-03T02:36:01.324988Z to 2026-09-03T02:36:10.326114Z (9.0 s).

Exploratory synchronous qualification; includes startup, JIT, checkpoints and shutdown. No controlled async split comparison or held-out quality claim.

## Nodes and headline measurements

GPU columns pool observed GPU samples. NVLink sums 18 links per GPU tick; IB is **per rail**, not aggregate node bandwidth. Lustre is per client. All means are sample-weighted.

| Node | Role | GPUs/rails | GPU util mean / p95 (%) | HBM max (GiB/GPU) | Power max (W/GPU) | NVLink Tx mean (GB/s/GPU) | IB Tx mean (GB/s/rail) | Lustre write max (GB/s/client) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gpu-nodes-0 | trainer | 0/0 | — / — | — | — | — | — | 0.000416 |
| gpu-nodes-1 | trainer | 0/0 | — / — | — | — | — | — | 0.000416 |
| gpu-nodes-2 | rollout | 0/0 | — / — | — | — | — | — | 0.000392 |
| gpu-nodes-3 | rollout | 0/0 | — / — | — | — | — | — | 0.0004 |

## Failures and coverage

No collector-error records observed; this alone does not establish complete coverage.
- telemetry/sync-grpo-v12/gpu-nodes-0/nvidia-smi.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-0/nvlink.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-0/infiniband.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-0/cpu-memory-numa.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-0/lustre.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-1/nvidia-smi.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-1/nvlink.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-1/infiniband.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-1/cpu-memory-numa.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-1/lustre.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-2/nvidia-smi.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-2/nvlink.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-2/infiniband.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-2/cpu-memory-numa.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-2/lustre.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-3/nvidia-smi.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-3/nvlink.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-3/infiniband.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-3/cpu-memory-numa.jsonl: **missing**.
- telemetry/sync-grpo-v12/gpu-nodes-3/lustre.jsonl: **missing**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-0/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-1/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-2/lustre.jsonl: **partial**.
- telemetry/lustre-sync-grpo-v12/gpu-nodes-3/lustre.jsonl: **partial**.

Invalid intervals among summarized counters: **0** (per-series intervals, not independent outages). Missing/reset/>5 s intervals are excluded, never zero-filled.

| Node | Observed health-counter series | Always zero | Unchanged nonzero | Changed/reset |
|---|---:|---:|---:|---:|
| gpu-nodes-0 | 0 | 0 | 0 | 0 |
| gpu-nodes-1 | 0 | 0 | 0 | 0 |
| gpu-nodes-2 | 0 | 0 | 0 | 0 |
| gpu-nodes-3 | 0 | 0 | 0 | 0 |

Unchanged nonzero values predate the observation window; they are not new errors during this run. These are only the ECC/IB counters actually collected. They do not establish XID, throttle, row-remap, PCIe or DCGM coverage.

## Largest entity differences

Lowest/highest time-mean within each node; descriptive differences, not hardware-fault diagnoses.

| Node | Metric | Lowest entity : mean | Highest entity : mean | Across-entity CV |
|---|---|---|---|---:|

## What is retained

- The JSON contains node distributions (min/mean/median/p90/p95/p99/max/CV), compact GPU summaries, health exceptions, source hashes and gaps.
- [timeline.csv](timeline.csv) has one-minute min/mean/p95/max envelopes and sample counts. Missing minutes are absent, not zeros; short spikes survive as maxima.
- Static inventory values are recorded once. Repeated zero counters are counted once per series; their exceptions are retained. Repeated raw values, per-link tables and lifetime Lustre aggregates stay out of Git.

## Evidence and limits

Raw evidence root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`. All 24 source stream paths and available SHA-256 hashes are in the JSON. Raw source files were not deleted.

Formatting reference: ClusterMAX `fed871df5321d42706c98701522cc3ccd55898d5`, `bench/README.md` and `bench/result_summary.py`; private source and provider report were not copied.

Full host fabric/storage counters are not process-exclusive. Clock synchronization below the sampling interval is unproven. Percentiles describe the observed workload, not hardware capacity. The original ClusterMAX saturation results are not like-for-like comparisons.
