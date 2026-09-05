# Important: PPO startup scripts

Canonical entry point: `python3 launch_ppo.py --submit` after checking cluster availability. This allocates all four assigned nodes for up to one hour; job 196 is a completed historical run; do not submit merely to check its status.

`launch_ppo.py` freezes named sources into a new campaign; `run.sbatch` starts `coordinator.py`. The coordinator installs the pinned precision fix, applies `ppo-resident-broadcast.patch` through `prepare_ppo_driver.py`, validates arguments and runs the native/transport tests before `training_entry.py` starts PPO. Keep the patch with the startup scripts: the disjoint actor requires both residency restoration and CPU parameter backup.

The original independently collected source snapshot is recoverable through `repository-cleanup.json`; byte-identical extra copies were removed. `comparison-spec.json` here is the original matched-workload input needed by the launcher, not a claim that the async experiment has the same optimization algorithm.
