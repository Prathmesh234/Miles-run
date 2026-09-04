# Run 3 — asynchronous Miles PPO + truncated importance sampling

**Job 197 completed both PPO updates: Slurm COMPLETED, exit 0:0, in 19m45s.** It ran 2026-09-04 01:13:58–01:33:43 UTC, after successful job 196, on the same four nodes / 32 B200 GPUs. It started from original base-model weights, not the trained preceding checkpoint.

**[Final infrastructure report and comparison](../comparison-infrastructure/README.md)** · **[Final logs](logs/job-197-final/)** · **[Per-run charts](charts/final/)**. Runtime checks confirmed 54.49s of rollout/training overlap, behavior weight versions `[1,1]`, policy lag `[0,1]`, live TIS correction and 64 successful optimizer rank-step records. Both actor and critic checkpoint structure and sampled CPU tensors passed validation; sampled parameters changed from base. No full optimizer/RNG resume test was performed.

**Startup scripts are essential:** [scripts/README.md](scripts/README.md). The exact submission and source hashes are in [config/submission.json](config/submission.json); the controller's queued state is in `snapshots/job-197-queued/slurm-status.txt`. The job was renamed to `miles-async-ppo-terminal-lego` after submission; the original submission line retains the earlier display name.

## Asynchrony and correction

Uses the pinned native `train_async.py` **one-batch-ahead** driver. Batch 1 is generated while batch 0 trains. Weight publication waits for that in-flight batch to finish, preventing mid-episode weight changes. Both rollouts were observed serving weight version 1; the instrumented expected trainer versions were `[1,2]`: lag zero, then one update, corroborated by nontrivial TIS metrics. This is not the separate `--fully-async` persistent-worker implementation, which is incompatible with the custom pinned-task rollout hook without a larger rewrite.

PPO uses its ordinary clipped new-policy / trainer-before-update ratio. A separate detached TIS weight is `clamp(exp(trainer_before_update_logprob - recorded_behavior_logprob), 0, 2)`. `--use-rollout-logprobs` is removed so these two ratios do not double-count the behavior correction. Existing action masks exclude tool/observation tokens. Model, task, sampling, batch, learning rates and actor/critic configuration otherwise follow synchronous PPO.

The actor CPU-backup and resident-broadcast fixes are retained. A pre-training policy-validity gate rejects nonfinite or uniform-vocabulary behavior logprobs.

## Metrics enabled

**[RL chart gallery](charts/rl/README.md):** six multi-panel PNG/SVG figures covering rewards, task outcomes, policy optimization, rollout behavior, critic learning, policy lag and TIS. The correction view distinguishes behavior-policy mismatch from PPO's pre-update reference metrics.

- Driver wall time for each weight publication, with actor onload/offload timed separately.
- Attempted per-rank lifecycle wrappers produced no records; native structured logs retain a partial, rounded rank-labelled view. Complete driver onload/offload spans and optimizer-step CPU RSS/CUDA memory records are available.
- Optimizer step wall time, CPU/GPU referenced optimizer storage sizes before/after each step, optimizer classes and instrumentation overhead.
- Native TIS ratio, absolute mismatch and clip fraction; added effective weight, squared weight and upper clip fraction.
- Behavior policy versions, expected lag and rollout/actor/critic timestamps to check actual overlap.
- Existing per-node GPU/host telemetry, IB port counters, NCCL logs, SGLang Prometheus and health snapshots.

CPU Adam-state residency is distinct from model parameter/gradient offloading. Referenced tensor storage bytes are **not** physical resident pages or bytes transferred over PCIe. Host wall times and whole-node IB counters are **not** direct CUDA-copy timing or isolated weight-transfer bandwidth. No optimizer/RNG checkpoint is saved, matching the preceding weights-only experiment.

## Evidence and checks

CPU-only validation passed before submission: the actual generated async loop overlaps mocked rollout/training with bounded lag; TIS values/gradients, clipping, mask behavior and on-policy identity pass; adding optimizer instrumentation leaves a CPU Adam update unchanged. These are unit/integration gates; successful distributed training is established separately by final step metrics, scheduler exit and checkpoint checks.

`logs/submission/` and `snapshots/job-197-queued/` preserve historical preflight/queued evidence. The original top-level chart manifest describes that earlier queued state. **Use `logs/job-197-final/` and `charts/final/` for completed results.** Reporting helpers are distinct from the frozen startup scripts. Additional read-only IB/NVLink sampling, its failed startup attempt and corrected coverage, are documented in the comparison report; training sources were not changed.

Remote campaign: `/shared/clustermax-campaigns/miles-async-ppo-terminal-lego-20260904T010040Z`; completed run directory: `runs/job-197`. Original baseline and PPO outputs were not removed or modified.
