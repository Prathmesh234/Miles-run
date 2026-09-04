# Run 3 — asynchronous Miles PPO + truncated importance sampling

**Slurm job 197 is queued with `afterok:196`.** It will start only if synchronous PPO job 196 succeeds. It requests the same four assigned nodes / 32 B200 GPUs for at most one hour. It starts from the original base-model weights, not the trained preceding checkpoint.

**Startup scripts are essential:** [scripts/README.md](scripts/README.md). The exact submission and source hashes are in [config/submission.json](config/submission.json); the controller's queued state is in `snapshots/job-197-queued/slurm-status.txt`. The job was renamed to `miles-async-ppo-terminal-lego` after submission; the original submission line retains the earlier display name.

## Asynchrony and correction

Uses the pinned native `train_async.py` **one-batch-ahead** driver. Batch 1 is generated while batch 0 trains. Weight publication waits for that in-flight batch to finish, preventing mid-episode weight changes. With two updates, behavior versions should be `[1,1]` and trainer versions `[1,2]`: lag zero, then one update. This is not the separate `--fully-async` persistent-worker implementation, which is incompatible with the custom pinned-task rollout hook without a larger rewrite.

PPO uses its ordinary clipped new-policy / trainer-before-update ratio. A separate detached TIS weight is `clamp(exp(trainer_before_update_logprob - recorded_behavior_logprob), 0, 2)`. `--use-rollout-logprobs` is removed so these two ratios do not double-count the behavior correction. Existing action masks exclude tool/observation tokens. Model, task, sampling, batch, learning rates and actor/critic configuration otherwise follow synchronous PPO.

The actor CPU-backup and resident-broadcast fixes are retained. A pre-training policy-validity gate rejects nonfinite or uniform-vocabulary behavior logprobs.

## Metrics enabled

- Driver wall time for each weight publication, with actor onload/offload timed separately.
- Per-rank model sleep/wake/update durations, CPU RSS and CUDA allocated/reserved/free memory.
- Optimizer step wall time, CPU/GPU referenced optimizer storage sizes before/after each step, optimizer classes and instrumentation overhead.
- Native TIS ratio, absolute mismatch and clip fraction; added effective weight, squared weight and upper clip fraction.
- Behavior policy versions, expected lag and rollout/actor/critic timestamps to check actual overlap.
- Existing per-node GPU/host telemetry, IB port counters, NCCL logs, SGLang Prometheus and health snapshots.

CPU Adam-state residency is distinct from model parameter/gradient offloading. Referenced tensor storage bytes are **not** physical resident pages or bytes transferred over PCIe. Host wall times and whole-node IB counters are **not** direct CUDA-copy timing or isolated weight-transfer bandwidth. No optimizer/RNG checkpoint is saved, matching the preceding weights-only experiment.

## Evidence and checks

CPU-only validation passed before submission: the actual generated async loop overlaps mocked rollout/training with bounded lag; TIS values/gradients, clipping, mask behavior and on-policy identity pass; adding optimizer instrumentation leaves a CPU Adam update unchanged. These are unit/integration gates, not a claim of successful distributed training.

`logs/submission/` records preflight and CPU validation; `snapshots/job-197-queued/` contains remote queued evidence and frozen submitted scripts. New helper scripts `collect_evidence.py` and `plot_run.py` are post-submission reporting tools. `charts/plot-manifest.json` honestly reports no runtime chart data while the job is pending.

Remote campaign: `/shared/clustermax-campaigns/miles-async-ppo-terminal-lego-20260904T010040Z`; future run directory: `runs/job-197`. Original baseline and PPO outputs were not removed or modified.
