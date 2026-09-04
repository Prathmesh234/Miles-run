# Important: asynchronous startup scripts

**Job 197 is already queued. Do not run this command again just to check status.**

For a deliberately new experiment, from this directory:

```sh
python3 launch_async_ppo.py --submit --after-job <successful-predecessor-job-id>
```

This creates a fresh campaign, preserves old outputs, runs CPU validation in a bounded GPU-free container, and submits all four nodes with an `afterok` dependency. `run.sbatch` invokes `coordinator.py`; `training_entry.py` selects async PPO + TIS. `prepare_async_driver.py` verifies the pinned upstream SHA-256 and generates `train_async_ppo.py` locally to the new run. `async_runtime.py` preserves actor backups and measures driver phases; `async_metrics.py` instruments rank-level offload/optimizer work and delegates TIS math to pinned Miles. `test_async.py` is a required gate.

`prepare_ppo_driver.py` and `ppo-resident-broadcast.patch` are retained for inherited PPO lifecycle validation. `launch_ppo.py` is an inherited synchronous reference, **not the run-3 entry point**.

To collect evidence, choose a destination that does not already exist:

```sh
python3 collect_evidence.py /shared/clustermax-campaigns/miles-async-ppo-terminal-lego-20260904T010040Z 197 ../snapshots/job-197-new
python3 plot_run.py ../snapshots/job-197-new ../charts/job-197-new --title 'Async PPO job 197'
```

The plotting helper requires Matplotlib. Missing metrics produce no fabricated charts. Collector snapshots retain complete JSONL lines and a SHA-256 manifest; binary checkpoints and rollout dumps remain on the cluster. Neither reporting helper was part of the frozen training submission.
