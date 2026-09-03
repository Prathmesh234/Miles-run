# Miles / Megatron Terminal-Lego run — job 190

Completed **3 September 2026** on **32 NVIDIA B200 GPUs**. This repository now contains the completed Miles experiment, its comparison with Prime-RL job 181, reproducibility scripts, and infrastructure evidence. It replaces the previous job-177 working tree; older work remains in Git history.

## Results

| Item | Result |
| --- | --- |
| Slurm job | **190 — COMPLETED, exit 0:0** |
| Training | **Two optimizer updates**, 16 accepted traces each |
| Model | Qwen3.6-35B-A3B, original base weights |
| Backends | Miles + Megatron trainer + SGLang inference |
| Allocation | Four nodes, 14m34s, 7.77 allocated GPU-hours |
| Final checkpoint | `runs/job-190/checkpoints/iter_0000001` — zero-based ID, after update two |
| Second trainer update | 19.01s; baseline approximately 19.40s with different timer boundaries |
| Inference-ready → checkpoint | Miles 7m39.65s; baseline 5m41s |
| Original run | Preserved; final source/config hash audit passed |

This is a **two-update functionality/performance smoke test**, not a model-quality result or a controlled backend speedup benchmark. Precision, kernels, rollout scheduling and generated trajectories differ. Failed setup attempts and final shutdown warnings are documented, not omitted.

## Start here

- **[Comparison report](COMPARISON.md):** workload, timing, rewards, memory, failed attempts, caveats and checkpoint checks.
- **[Infrastructure report](INFRASTRUCTURE.md):** GPU/CPU/NUMA topology, IB/RDMA, storage, resource constraints, software versions and telemetry coverage.
- **[Final status](STATUS.md)** and **[machine-readable results](comparison-results.json)**.
- **[Configuration and deviations](comparison-spec.json)**, including exact model, dataset and source revisions.
- **[Checkpoint verification](evidence-job-190/job-190/checkpoint-verification.json)** and **[baseline preservation audit](baseline-preservation-after-job-190.json)**.

## Repository contents

| Location | Contents |
| --- | --- |
| Root `*.py`, `run.sbatch`, `sglang.yaml` | Campaign launch, adapters, precision patch, telemetry, analysis and validation scripts |
| `evidence-job-190/job-190/source/` | Frozen launch-time source, distinct from later analysis additions |
| `evidence-job-190/job-190/analysis-source/` | Frozen post-run analysis scripts |
| `evidence-job-190/job-190/` | Training/harness logs, exact argv, resolved arguments, timelines, accepted traces and all 56 episode records |
| `evidence-job-190/job-190/infra/` | NCCL/Ray logs, GPU/host time series, local PMA/RDMA counters, before/after snapshots, runtime constraints and precision-patch provenance |
| `infra-preflight/` | Hardware snapshots before the experiment |
| `baseline-*.json`, `baseline-metrics.jsonl` | Baseline metrics, task schedule and preservation evidence |
| `publication-inputs/` | Pinned image digest, base-model conversion provenance and recorded adapter-test result |
| `publication-manifest.json` | SHA-256 and size of every copied file, exclusions and previous repository commit |

Published evidence is approximately **167 MB of uncompressed text**. Checkpoint/model weights, Docker images, archives, caches, binary rollout dumps and TensorBoard event files are not committed. The complete local/cluster archive retains those run artifacts. The reports were written against that complete archive, so references to TensorBoard and binary debug dumps describe retained evidence, not files included here.

Infrastructure files intentionally include observed private network addresses, node names, device identifiers and filesystem paths. They do **not** include a kubeconfig or access credentials. This is a public evidence repository, not an access mechanism.

## Reproduction scope

These scripts preserve the actual campaign and its environment-specific paths. They are **not a portable one-command installer**. A fresh environment must supply the pinned base-model files, task images, original harness and the four-node Slurm/Docker infrastructure. Large inputs and the original baseline source are referenced by revision/path, not vendored into this repository.

The recorded campaign root is:

```text
/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030
```

The execution sequence was:

1. Inventory and preserve the baseline; pin the model, dataset, harness, renderer and Miles revisions in `comparison-spec.json`.
2. Stage the pinned Miles image in separate `fuse-overlayfs` Docker daemons; keep the original task-image daemon unchanged. See `isolated_docker.py` and `preflight_image.py`.
3. Supply the pinned Miles checkout at the campaign's `miles/`, scripts at `code/`, and the pinned image digest at `image-digest.json`. The original model is mounted read-only. `remote.py` requires your own local kubeconfig configuration.
4. Run the adapter fidelity checks in the original pinned harness environment. They require original accepted-episode fixtures and the job-189 errored-group fixture, which are retained on cluster but not all bundled here. The successful recorded result is in [adapter-test-result.json](publication-inputs/preflight/harness-test-v2/adapter-test-result.json).
5. Submit `run.sbatch` to the four assigned nodes. `coordinator.py` creates a **new job-ID directory**, snapshots its source, applies and validates the job-local precision patch, validates arguments, converts or reuses original base weights, starts the harness and Ray, runs two updates, and stops only job-scoped processes.
6. Supplement the built-in two-second collector with `capture_rdma.py`, `capture_metrics.py` and `runtime_constraints.py`, as recorded in the evidence. These additional collectors were started separately; `run.sbatch` alone does not reproduce their collection start times.
7. Extract metrics, verify checkpoint storage ranges and sample tensors, audit baseline preservation, and compare the measured work using the included analysis scripts.

**Submitting this workload reserves 32 GPUs.** No new training job was launched to publish this repository. Do not rerun against an existing job directory or use the trained baseline checkpoint as the initial model.

## Validation and limitations

The completed run passed 32 exact IPO value/gradient fixtures, real TrainClient stop/length mapping, all 32 baseline trace conversions, native errored-group admission/credit tests, five GPU precision shapes and CUDA-graph replay. The final checkpoint passed metadata/range checks and three actual CPU tensor reads. **A full distributed checkpoint-resume test was not performed.**

Publication checks verify copied-file hashes, frozen source manifests, Python syntax, JSON/JSONL parsing and README links. These checks do not rerun GPU training or replace the recorded fidelity tests. Credential-pattern screening found no credentials after reviewing the SGLang `IP:port@DP-rank` URL notation; pattern-based screening is not an exhaustive security guarantee.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the retained SGLang/vLLM source snapshots.
