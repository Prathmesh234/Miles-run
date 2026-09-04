# Run 5 — longer asynchronous PPO + TIS (preparation)

**Not submitted.** Model selection and checkpoint cadence are pending as
described in [run 4](../run4-ppo-long/README.md). Both runs start from the same
new base checkpoint; this run must not resume run 4's trained weights.

Target: 10 actor and 10 critic optimizer updates, retaining job 197's native
one-batch-ahead schedule and detached TIS clipped to `[0, 2]`. This is not the
persistent fully-async worker implementation. Retain the same workload,
sampling, batch sizes and CPU-offload settings.

## Important: startup scripts

Use [scripts/](scripts/README.md) for startup and validation instructions.
The configuration is in [config/run.json](config/run.json). The common
duration/configuration code lives in run 4's scripts directory so it is
versioned once. A launch command will be recorded after model compatibility
and storage decisions are resolved; no scheduler job currently exists.

Logs and charts will stay in this run's `logs/` and `charts/` subdirectories.
Publish consolidated evidence, not duplicated source snapshots or weights.

After this run produces evidence, use the shared [RL renderer](../comparison-infrastructure/rl_charts.py) with `--folder run5-async-ppo-long --log-root <repository-relative-log-directory> --rollout-root <repository-relative-rollout-directory> --job-id <actual-job-id> --label 'Long async PPO + TIS'`. All five arguments are required. The gallery will include reward/task-outcome, policy, rollout, critic and off-policy/TIS charts, with PNG/SVG exports and source hashes. No curves are fabricated for this unsubmitted run. This recipe assumes the original binary solved rewards, eight-sample task groups, one update per rollout, and engine version 1 at the start of the native async schedule.
