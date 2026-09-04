# Run 2 — synchronous PPO

**Job 196 completed both updates: Slurm COMPLETED, exit 0:0.** Both actor and critic had finite nonzero gradients and saved weights after update 1 (the second, zero-based update).

**Startup scripts are essential:** begin with [scripts/README.md](scripts/README.md). They preserve the exact job-196 configuration and actor-residency/CPU-backup fixes.

## Final evidence

- [RL chart gallery](charts/rl/README.md): rewards, task outcomes, policy loss/entropy/gradients, PPO clipping/KL/ESS, rollout behavior and critic learning, in PNG/SVG with source metrics and hashes.
- [Final training/infrastructure logs](logs/job-196-final/) and [SHA-256 snapshot manifest](logs/job-196-final/snapshot-manifest.json).
- [Actor checkpoint checks](logs/job-196-final/checkpoint-verification.json) and [critic checkpoint checks](logs/job-196-final/critic-checkpoint-verification.json): all referenced byte ranges exist, six small tensors loaded on CPU per role, sampled parameters changed from base. Not a full distributed resume test.
- [Final per-run GPU charts](charts/final/) and [three-run infrastructure comparison](../comparison-infrastructure/README.md): rollout time, execution overlap, weight publication, InfiniBand, NVLink, TIS and optimizer/offload observations.

The prior `logs/job-196-live/` and original top-level charts are retained as **earlier snapshots**, not the final result. Job 195's incomplete cancelled attempt remains separately in `attempts/job-195-partial-snapshot/`; it is not a valid learning baseline.

Job 195 lost actor parameters because CPU parameter-buffer backup was disabled. Job 196 retains backup, restores actor residency before broadcasts and rejects invalid/uniform behavior logprobs. SGLang/Gloo peer-reset errors occurred during final teardown after checkpointing and publication; the training process and Slurm allocation exited successfully. See the final logs rather than treating the run as warning-free.

Same base model, task, 32 GPUs, two updates, batch 16 and group size 8 as the comparison workload. Generated trajectories and work differ; no model-quality or controlled speedup claim.

Remote corrected run: `/shared/clustermax-campaigns/miles-ppo-terminal-lego-20260904T005533Z/runs/job-196`.
Remote cancelled run: `/shared/clustermax-campaigns/miles-ppo-terminal-lego-20260904T004324Z/runs/job-195`.

Full raw/binary outputs remain on the cluster. Weights-only checkpoints omit optimizer/RNG state. Nothing from those runs was deleted to publish this report.
