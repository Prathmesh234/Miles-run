# Important: startup scripts

`run.sbatch` allocates four 8-GPU nodes. `coordinator.py` establishes the isolated containers, telemetry, Ray and pinned task harness. `training_entry.py` records and executes the exact model/training arguments. `harness_bridge.py` and `rollout_adapter.py` transport the original task's sampled IDs, logprobs and masks.

These are **historical, frozen job-190 sources**. Do not submit them against the existing campaign paths: that would target preserved evidence. A repeat must use a fresh campaign/run directory while retaining the pinned model, task and image. `source-file-sha256.json` in the parent folder describes copied historical inputs. The added `collect_evidence.py` and `plot_run.py` are publication helpers, not part of the original training launch.
