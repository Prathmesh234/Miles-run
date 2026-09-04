# Run 1 — completed Miles GRPO-style credit / IPO

Job **190**, two optimizer updates, Slurm `COMPLETED` / exit `0:0`.
The folder name is convenient shorthand: this run used **group-relative credit with the IPO objective**, not standard clipped GRPO.

**Startup scripts matter:** see [scripts/README.md](scripts/README.md). Those are the frozen run sources, including `run.sbatch`, `coordinator.py`, `training_entry.py` and the pinned-harness adapter.

- `logs/`: training, harness, infrastructure and event logs.
- `rollouts/`: recorded task episodes and accepted batches.
- `metrics/`: extracted scalars, checkpoint verification and baseline comparison.
- `charts/`: GPU, CPU, memory, IB, inference and phase charts from this completed run.
- `config/` and `reports/`: arguments, pinned inputs and original comparison reports.

Raw run: `/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/runs/job-190`.
Model/checkpoint shards remain there. Original reports are archival copies and can retain their original relative references; the repository root also retains the original evidence layout.
