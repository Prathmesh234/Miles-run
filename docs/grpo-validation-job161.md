# Job 161: four-node GRPO validation

**FAIL** — Slurm FAILED (1:0). 2 optimizer updates and 2 save receipts verified.

## What passed

- 32 native samples reconciled through trainer tokens, logprobs, masks and GRPO advantages.
- All 48 controller episodes have durable native identities; 0 unjoined episodes and 0 unresolved dispositions.
- Environment audit found no grading-isolation or lifecycle imbalance.

| Population | Samples | Passed episodes | Environment-seconds |
|---|---:|---:|---:|
| selected_for_training | 32 | 23 | 356.62 |
| sync_unused_discarded | 16 | 11 | 143.60 |

Zero-variance GRPO groups: 2. These are clean training-task outcomes, not TB2.1 evaluation.

## Failure evidence

| Node | API | Duration (s) | UTC start |
|---|---|---:|---|
| gpu-nodes-0 | nvmlDeviceGetFieldValues | 14.425 | 2026-09-03T02:50:31.335542Z |
| gpu-nodes-0 | nvmlDeviceGetMemoryInfo | 2.256 | 2026-09-03T02:50:23.113882Z |
| gpu-nodes-0 | nvmlDeviceGetFieldValues | 1.341 | 2026-09-03T02:50:18.180937Z |
| gpu-nodes-1 | nvmlDeviceGetFieldValues | 9.016 | 2026-09-03T02:50:32.745449Z |
| gpu-nodes-1 | nvmlDeviceGetMemoryInfo | 3.057 | 2026-09-03T02:50:29.688306Z |
| gpu-nodes-1 | nvmlDeviceGetMemoryInfo | 1.378 | 2026-09-03T02:50:21.696242Z |

The overlong collector ticks occurred during teardown. Do not treat the completed training driver as a passing infrastructure gate.

## Timing observations

| Rollout | Environment rollout (s) | Trainer call (s) | Trainer wait (s) |
|---:|---:|---:|---:|
| 0 | 16.23 | 143.28 | 24.97 |
| 1 | 28.37 | 22.05 | 83.05 |

The first trainer call includes cold compilation. Waiting in this synchronous run includes checkpoint and rollout work; it does not classify an asynchronous role split.

## Attribution and limits

- **Infrastructure/driver:** blocking NVML calls observed; underlying cause unproven.
- **Miles:** unused sampled work is now fully accounted for. Default resume scheduler mismatch is separately reproduced in the resume report.
- **Model recipe:** finite gradients and native tensor checks passed; held-out quality and quantized execution remain unqualified.
- **Environment:** local task isolation and lifecycle audit passed for this scoped task subset.
- **Configuration:** tiny synchronous batches, cold compilation and checkpoint-every-step prevent a steady-state performance claim.

- Synchronous small-batch validation, not an asynchronous placement comparison.
- Training-task rewards are not held-out TB2.1 quality; no quality delta is established.
- Full checkpoint resume, actor placement capture, DCGM and all required telemetry remain unqualified.
- NVML call timing identifies the blocking API, not the underlying hardware or driver cause.
- Environment-seconds sum parallel episode lifetimes; they are not elapsed wall time.

## Reproducibility and raw evidence

Shared root: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210`.

Miles `1a6303b6204d853c5d812bb92be1319a28028d39`; site code `b472c3e359e839f75fe62188c31a113efbb2408d`.

- allocation: `tests/02-sync-grpo-result-audit-v12/audit.json`; SHA256 `1f9b304428d987cab6151b827f88727c092265eb23f5fbaa3aa015e986d45a58`.
- episodes: `tests/02-sync-grpo-environment-audit-v12-join-v1/episodes.json`; SHA256 `918b2398a5f9352db7f3026256dd931d97c824035ac6180a0199ba119cc9d9d4`.
- nvml: `tests/01-nvml-call-audit-job161-v1/result.json`; SHA256 `6c1fa22d2c6c5a953b8149045ea2cc708b6ded02ad76e14b95430bb840e7239e`.
- optimizer: `tests/02-sync-grpo-v12-optimizer-observation-a1/result.json`; SHA256 `fbfa0e4bde53f2b17fc3ca0587501f19d42f1e597efee7e7cd6d53c87699028e`.
- tensors: `tests/02-grpo-tensor-audit-v12-a1/result.json`; SHA256 `58788fa3d08c743ff6207f2b6d8382ebd32b92b7ce3781138e240c19104d945d`.
