# Run 2 — synchronous PPO

Corrected PPO job **196** was running when this snapshot was collected. It uses the same base model, task code, 32-GPU allocation, two updates, batch 16 and group size 8. Do not interpret this snapshot as completed training or a final comparison.

**Startup scripts are essential:** [scripts/README.md](scripts/README.md).

- `scripts/`: frozen job-196 startup code and its two actor-residency/backup patches.
- `logs/job-196-live/`: timestamped, SHA-256-verified training/infrastructure snapshot and submitted sources.
- `charts/`: measured GPU utilization, memory and power from that snapshot; PNG and SVG.
- `attempts/job-195-partial-snapshot/`: earlier incomplete snapshot, retained separately. Job 195 was cancelled after a uniform-policy/zero-gradient actor step; it is **not** a valid PPO learning result.

Job 195's disjoint actor lost parameters because CPU parameter-buffer backup was disabled. Job 196 retains that backup and restores actor residency before broadcasts, with a uniform-policy rejection gate. Whether the corrected distributed run succeeds must be established from its final logs, not just passing CPU tests.

Raw corrected run: `/shared/clustermax-campaigns/miles-ppo-terminal-lego-20260904T005533Z/runs/job-196`.
Raw cancelled run: `/shared/clustermax-campaigns/miles-ppo-terminal-lego-20260904T004324Z/runs/job-195`.
Full raw/binary data and checkpoints remain on the cluster. `collect_evidence.py` can create a new snapshot without overwriting this one.
