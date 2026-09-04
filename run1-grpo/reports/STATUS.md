# Miles versus Prime-RL, Terminal-Lego

**Complete.** Miles job 190 finished two optimizer updates, saved the final checkpoint and exited `COMPLETED 0:0`. See [COMPARISON.md](COMPARISON.md) and [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

## Baseline

Slurm job 181, campaign `prime-rl-terminal-lego-b29c37e00`, run `20260903-150011`.
The trainer completed two updates and saved `step_2` at 15:30:03 UTC. The allocation remained open until 19:00:11 UTC because telemetry did not exit. That idle tail is not training time.

The immutable model revision, dataset revision, task hashes, resolved settings and observed metrics are recorded in `comparison-spec.json`. Original files have not been deliberately modified or deleted. The pre-existing job-181 `sstat` monitor continues appending errors to its own log; 188 other inventoried artifact metadata entries and all 49 hashed source/config files passed preservation checks after setup.

## New experiment

Remote campaign: `/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030`.

Miles revision: `70b89e11770fc9bac984e22cfff89c51cca44203`.

Image: `radixark/miles@sha256:4ee6da9f16e06f8ad24991b18a950482572c458a357aae0bfc396feaf3fe0a6d`.

The user approved Megatron + SGLang. The four tasks, base model, two updates, batch 16, group 8, 8192-token training context, 2048 tokens per turn, eight turns, temperature, optimizer hyperparameters and IPO objective are retained. The original pinned Verifiers + renderer + task code run through an HTTP token-transport adapter.

The IPO objective passed 32 CPU value/gradient fixtures against the actual baseline function, with exact equality. Actual TrainClient stop/length propagation, all 32 accepted baseline trace conversions, and native errored-group admission/credit tests passed.

### Attempts

| Attempt | Outcome |
| --- | --- |
| Original Docker image staging | Failed because `vfs` copied cumulative image layers and exhausted worker disks. Pulls stopped; Docker reclaimed temporary layers. No baseline images or run artifacts were pruned. |
| Isolated Docker staging | Succeeded on all four workers with `fuse-overlayfs`, separate sockets and data directories. No changes to original Docker daemon configuration. |
| Job 185 | GPU preflight rejected an unsupported seed CLI flag. Corrected via SGLang's supported YAML override. No training performed. |
| Job 186 | GPU argument validation and original-model checkpoint conversion succeeded. Ray startup rejected overlapping worker/head ports. No training performed. |
| Job 187 | Training placement initialized; SGLang failed during breakable prefill CUDA graph capture with a reduce-scatter shape mismatch. Stopped before rollouts, preserving logs. |
| Job 188 | SGLang prefill workaround succeeded; decode graphs and Megatron initialization passed. Initial expert-weight transfer failed with tensor dimensions 64 vs 2048. No task rollouts or updates. |
| Job 189 | Weight synchronization succeeded; 16 task episodes ran. Adapter aborted on an errored over-context episode. Corrected to native survivor filtering; no optimizer updates. |
| Job 190 | Completed both optimizer updates, saved checkpoint, driver exit 0 and Slurm COMPLETED 0:0. |

## Comparison limits

Megatron uses FP32 gradient accumulation/reduction, unlike the baseline BF16 reduction. Optimizer state precision, kernels, checkpoint format, inference implementation and scheduling also differ. Matching workload settings does not imply identical trajectories or bit-identical updates.

The baseline generated 120 episodes and accepted 32 traces after filtering. Its accepted groups involved the KnockoutJS and Docker/systemd tasks. Both runs must report generated versus accepted work, truncation and rewards, not only elapsed time. Two updates are a smoke test, not a model-quality study.

## Infrastructure evidence

Four nodes, 32 B200 GPUs, dual AMD EPYC 9575F sockets per node, NVIDIA driver 595.71.05, active 400 Gb/s InfiniBand, NVLink and Lustre shared storage. `kubernetes-infra.json` and `infra-preflight/` contain placement, resources, topology and device details.

Per-job remote evidence includes before/after snapshots, two-second GPU/CPU/memory/disk/network telemetry, NCCL transport logs, Ray logs, package versions, command timelines and Slurm accounting. Initial sysfs IB counters were unavailable; job 190 adds ten-second local PMA/RDMA counters, with documented port-coverage gaps. Model conversion and one-time staging must be reported separately from inference startup, rollout, optimizer work, weight synchronization and checkpoint writes.

Corrected adapter tests passed: actual TrainClient stop/length propagation, all 32 accepted baseline traces, errored-group native admission/FP32 credit, and 32 IPO value/gradient fixtures. Job 190 enabled FP32 LM-head output, FP32 router-logit GEMM and round-robin request routing.

## Final outcome

Job 190 ran 21:57:12–22:11:46 UTC on 3 September 2026. It executed 56 episodes (all without harness errors), accepted 32 traces and completed two optimizer updates. Trainer paths took 129.23 and 19.01 seconds; the first includes compilation. The ready-to-checkpoint window was 459.65 seconds versus baseline 341 seconds, with workload/scheduling/precision differences described in the comparison.

The final checkpoint is `runs/job-190/checkpoints/iter_0000001` (zero-based ID, after update two). Metadata/range verification and three actual CPU tensor reads passed; no full distributed resume test was run. Final SGLang disposal emitted connection-reset warnings after save/sync, but the driver and allocation exited successfully.

The final preservation audit confirms all 49 source/config hashes unchanged and 188/189 artifact metadata entries unchanged. The sole changing original artifact remains its own pre-existing sstat monitor log.

All four nodes are Ready without DiskPressure. Final runtime checks found no queued jobs, GPU compute processes, running Miles containers or campaign collectors. Images, stopped containers, failed attempts and checkpoints remain retained. Local evidence excludes the approximately 69 GB checkpoint, which remains on shared storage.
