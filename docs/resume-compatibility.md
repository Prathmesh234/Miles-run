# Checkpoint and resume qualification

**Full-state resume is not qualified.** Saving a checkpoint is not proof that restart preserves the next update.

| Component | Current evidence |
|---|---|
| Model, optimizer and LR scheduler | Job154 iteration1 contains common optimizer metadata and scheduler position32. Native replay proves default Miles initialization advances32 to48 without an update; explicit checkpoint-scheduler flag preserves32. Model/optimizer tensor restoration and next-update equality remain untested. |
| RNG | Job154 iteration1 contains a deserialized record with Python, NumPy, Torch CPU, CUDA and tracker state. Byte range and metadata hashed. No actual RNG restoration or next-update parity proof. |
| Consumed data position and sample identities | CPU save/load preserved sample_offset, epoch_id, sample_group_index, sample_index and metadata exactly. |
| Recycled prompt buffer | Stock inherited save/load drops the buffer. Opt-in CheckpointedRolloutDataSource passed7 native CPU roundtrip/negative tests (job155), also covered in job160; not enabled in training. |
| Fully asynchronous completed-group queue and accounting | No checkpoint state API on built-in DataBuffer; buffer and window counters initialize empty/zero. Not restart-qualified. |
| Broadcast policy version | Updater initializes at zero; no corresponding checkpoint restore was found in the inspected broadcast path. Must persist it with activation state. |
| In-flight environment and scheduler state | Live asyncio tasks and sandbox processes are not serialized. A qualified checkpoint boundary must drain or explicitly account for cancellation and re-submission. |

## Inspected checkpoint

`training/sync-grpo-v10/checkpoints/iter_0000001` — iteration 1, saved scheduler position 32.

Only DCP metadata and the small common/RNG byte ranges were read. Tensor payload checksums, actual model/optimizer loading and next-step equivalence remain untested.

The Megatron consumed-sample counters are zero; Miles dataset cursor state is separate. Do not interpret those fields as a resumed data position.

## Reproduced scheduler mismatch

| Configuration | Loaded position | After Miles initialization | LR |
|---|---:|---:|---:|
| use_checkpoint_opt_param_scheduler=False | 32 | 48 | 1e-06 |
| use_checkpoint_opt_param_scheduler=True | 32 | 32 | 1e-06 |

Pass --use-checkpoint-opt-param-scheduler for the controlled resume test, preserve the saved scheduler horizon and all other settings, then verify actual post-load state and the next update. Constant LR did not change in this fixture, but the scheduler counter did.

This reproduces the native scheduler plus the pinned Miles post-load branch, not a whole-model restart. The controlled GPU replay explicitly enables checkpoint-scheduler restoration.

## Frozen-input replay

Status: **gpu_replay_failed_before_model_initialization**. Native CPU job **170**; GPU job **171**, 32 GPUs, 2 independent trainer replicas.

Bitwise model/optimizer/scheduler/RNG reload and frozen next-update comparison. All32 ranks must pass reload before either replica steps. Original inputs mounted read-only; no new checkpoint payload.

| Limitation |
|---|
| CPU fixture writer is an explicit adapter, not the training writer. |
| Full asynchronous queue/accounting, policy version and data cursor remain outside this replay. |
| Megatron saves one RNG replica when data_parallel_random_init is false; this cannot prove original per-expert-rank RNG coverage. |
| Payload file identity and small-input checksums are frozen; no whole-file checksum claim for existing497GB checkpoints. |

| Preserved failure or source finding | Scoped change |
|---|---|
| Native fixture writer synchronized CUDA with no GPUs | Fixture-only CPU synchronous writer; no package or training writer modification. |
| Premature local audit while170 was still running | Original failed audit preserved; a2 required terminal COMPLETED0:0 before final export. |
| Native ckpt_step0 is treated as false in source | Isolated read-only load view with zero-step tracker; no original checkpoint metadata edited. |
| Checkpoint file stat differed inside container; external stat matches all four worker pods. | Exact-mount CPU check matches all32 payload identities; cause unknown. Retain expected/observed stat values on any future mismatch; no comparison relaxation or checkpoint changes. |

CPU proof: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-resume-native-cpu-audit-v3-a2/result.json`. Submission: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-resume-replay-submission-v1/submission.json`.

## Remaining requirements

- Verify model, optimizer, scheduler and RNG state in a completed checkpoint, including load/next-update comparison on identical frozen samples.
- Preserve and restore recycled prompts, completed groups, version counters and lifetime accounting, or explicitly quiesce and prove none remain.
- Atomically publish a complete state manifest only after all checkpoint components are durable and hashed.
- Demonstrate exact submitted/completed/trained/retried/stale/failed accounting across a restart; never silently reset policy version or the data cursor.

## Evidence

- Dataset cursor probe: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-resume-surface-probe-v1/result.json`.
- Cursor/buffer candidate: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-data-source-resume-test-v1/result.json`.
- Cursor/buffer candidate: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-data-source-resume-test-v1/results.xml`.
- Saved-state inspection: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-checkpoint-resume-metadata-v10-step1-a2/result.json`; SHA256 `a3ad4e32339542e60b032c68bfc1c2815079d587576f88193313413e36206ac0`.
- Saved-state inspection: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-checkpoint-resume-metadata-v10-step1-a4/result.json`; SHA256 `9645c0020b0f77c499b3f6548f236c3a7668e618811d91281199d2ec4e3bbca2`.
- Scheduler replay: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-checkpoint-scheduler-replay-v10-step1-a2/result.json`; SHA256 `96d1c901c4979130a15ff330b68b4d8f824fa81d4defc85f79d1786db2d57066`.
