# PostTrainingX on Vultr B200

This workspace implements the gated Miles GRPO campaign requested by Pratt Bhatt.
The complete objective uses four nodes and 32 B200 GPUs. Infrastructure evidence,
exact accounting, training correctness, and held-out quality are independent gates.
Completion of a process does not establish benchmark success.

The clean training corpus must be a pinned Terminal-Lego or TMAX subset. Terminal
Bench 2.1 is evaluation-only. No oracle, solution, grader, credential, or hidden
test may enter the policy context or training corpus. The online runtime uses
Miles OpenEnv with task containers on Vultr, following the operator's local-runtime
steering. The async flow still derives from the GLM-5.2 OpenEnv reference.
A separate Python 3.12 environment owns offline
Verifiers/Harbor evaluation. Strict Verifiers online compatibility remains blocked
until its token, logprob, grouping, renderer, reward, trace, and gradient tests pass.

## Safety

All initial cluster inspection is read-only. Cluster configuration, firmware,
devices, and shared data are not changed. Allocations specify `gpu-nodes` and own
whole 8-GPU nodes. Persistent campaign evidence, task data, and checkpoints belong
to the unique directory under `/shared/posttrainingx/runs/vultr-b200-slurm/<run-id>/`.
Run-owned diagnostic containers also use private writable layers; normal image
caches live in the container runtime's storage. Shared user data is not cleaned up.
A failed gate is preserved and stops dependent execution. Any fix needs an
explicit intervention entry and, when the user requires it, operator approval.

The environment controller is trusted and may use a Docker socket. Policy task
containers must receive no Docker socket, host namespaces, GPU devices, cluster
credentials, shared checkpoint tree, or hidden evaluation assets. The controller
and grader remain outside the policy container. Test isolation and cleanup after
a crash must pass before model trajectories or optimizer steps. The selected
mode is `TB2_MODE=docker`. OpenEnv's `TB2_MODE=local`, which directly executes
in the server process, is not the selected architecture.

## Evidence

`scripts/evidence.py` writes atomic JSON, reports generated from that JSON, raw
logs, command records, a sweep summary, and a SHA-256 inventory. It does not
fabricate metrics for stages that have not executed. Raw artifacts remain outside
Git in `runs/`; the campaign must preserve the complete bundle on shared storage.

## Planned stages

1. Freeze source, model, package, image, environment, and hardware provenance.
2. Reconcile GPU UUIDs, Kubernetes resources, Slurm GRES, storage, and telemetry.
3. Validate environment isolation, offline evaluation, serving/MTP, and synchronous GRPO with resume.
4. Measure CollectiveX and inference-only concurrency before freezing settings.
5. Prove fully asynchronous overlap and accounting with the 2T/2R reference.
6. Sweep 1T/3R, 2T/2R, and 3T/1R with physical role rotation and fixed settings.
7. Run the selected placement longer and evaluate fixed checkpoints on untouched TB2.1.

The final benchmark requires one warmup and three measured repetitions per layout.
Any shorter execution is labeled exploratory. The final report must attribute
failures to infrastructure, Miles, model recipe, environment, or configuration.

The proposed longer run uses 400 optimizer steps, separate from the 2–5-step
correctness test. Its evaluation protocol and unresolved resource budget are
recorded in [docs/quality-protocol.md](docs/quality-protocol.md).

## Implementation and current evidence

The committed Miles implementation and exact base/patched revisions are recorded
in [patches/manifest.json](patches/manifest.json). The three commits replay onto
the pinned upstream base with an identical final source tree. The launcher
supports all three whole-node layouts and requires an external, explicitly mapped
Ray cluster. CPU tests do not prove placement or training correctness on GPUs.

[The compatibility report](docs/compatibility.md) separates the blocked strict
online combination from OpenEnv online training and separately locked offline
evaluation. [The current status](docs/current-status.md) lists the failed and
unvalidated gates. No model training or Terminal-Bench quality result exists yet.

To resolve the documented dependency combinations without installing them:

```sh
python3 scripts/check_compatibility.py --run-dir /absolute/path/to/a/fresh/run
```

To render the retained native telemetry into statistics and a plot:

```sh
.venv-analysis/bin/python scripts/summarize_native.py --run-dir /absolute/path/to/run
```

The latter requires the finalized native telemetry to be present locally. It
refuses to replace an existing report phase; diagnostic reruns must retain their
own phase identity. Dependency freezes for CPU checks and analysis are separate
from the unvalidated GPU image and the offline evaluator lock.

## Native four-node preflight entrypoint

Commit the code, then run the following command from this workspace:

```sh
python3 scripts/submit_native_preflight.py \
  --run-dir /absolute/path/to/the/current/run \
  --kubeconfig /absolute/path/to/vultr-kubeconfig
```

The submitter requires an empty queue, rechecks Kubernetes GPU capacity, creates
a new run directory on the workers' Lustre mount, and refuses to overwrite an
existing staged directory. It submits one exclusive 15-minute allocation in
`gpu-nodes` with all four nodes and eight GPUs per node. It does not use Pyxis,
change GPU visibility, or modify cluster configuration.

Inside the allocation, the controller records start and end inventory, runs
one-second native collectors, tests all-reduce on each node and across all four,
and writes and verifies a 256 MiB fio file per node. Every file belongs to the
current run. Failed gates stop remaining load phases and retain raw evidence.
The native all-reduce uses one MPI process with eight GPUs per node. These
measurements do not replace the full collective suite, model-checkpoint tests,
CollectiveX, or the role-layout benchmark.

Submission is not completion. Inspect the returned Slurm job ID and its phase
results before proceeding. An ambiguous submission must be reconciled with
`squeue` and `sacct`, not repeated. The local environment-isolation gate remains
failed until a versioned fix passes the required tests.

Run the local parser and evidence checks with:

```sh
python3 -m unittest discover -s tests -v
```
