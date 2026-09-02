# Miles-run

PostTrainingX infrastructure and Miles GRPO qualification on Vultr B200.
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
fabricate metrics for stages that have not executed. The complete raw bundle
remains outside Git in `runs/` and on shared storage. Reviewed, normalized
infrastructure telemetry is also published under `telemetry/` as described below.

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
in [patches/manifest.json](patches/manifest.json). The cumulative patch includes
all 16 local Miles commits and replays onto the pinned upstream base with an
identical final source tree. The launcher
supports all three whole-node layouts and requires an external, explicitly mapped
Ray cluster. CPU tests do not prove placement or training correctness on GPUs.

[The compatibility report](docs/compatibility.md) separates the blocked strict
online combination from OpenEnv online training and separately locked offline
evaluation. [The current status](docs/current-status.md) lists the failed and
unvalidated gates. The four-node synchronous qualification completed three real
optimizer updates with 48 audited training samples. Its telemetry has collection
failures, and it is not a completed asynchronous benchmark. No held-out
Terminal-Bench quality result or full resume equivalence has been established.

## Repository contents and source setup

This repository contains the project source, tests, documentation, version locks,
and patches. Private ClusterMAX sources, kubeconfigs and credentials, virtual
environments, model/checkpoint binaries, raw task data, and unrestricted run evidence
are not published. Normalized telemetry snapshots are the explicit exception.
Raw evidence remains in the run directory on shared storage. Older
dummy-run checkpoint payloads were pruned with explicit operator authorization;
the newest full checkpoint and the older metadata/logs were retained.

Reconstruct the exact public Miles source tree from the pinned base and patch:

```sh
git clone https://github.com/radixark/miles.git vendor/miles
git -C vendor/miles checkout --detach 0709889b2848f293b5575d50aa3340fa4de5a20d
git -C vendor/miles apply --index ../../patches/miles-posttrainingx.patch
```

The reconstructed tree hash must match `source_tree_sha1` in the patch manifest.
The local launcher is then available at
`vendor/miles/scripts/run_qwen3_6_35b_a3b_posttrainingx.py`. The OpenEnv patch and
its exact revisions are recorded separately in `locks/openenv-patches.json`.
Third-party source retains its upstream license. Do not force upgrades into the
training environment to resolve the separate offline evaluator's dependencies.

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
from the pinned GPU image and the offline evaluator lock.

## Periodic telemetry publication

`scripts/stage_grpo.py` starts a run-owned local publisher after a successful
training submission. It snapshots, commits, and pushes new normalized metric
records approximately every five minutes, plus a final snapshot when Slurm reports
the job terminal. The watcher is bounded to 100 minutes; extend `--max-seconds`
explicitly for longer jobs or queue waits. The workstation must stay awake with
cluster and GitHub access. This is not a cluster-side scheduled service.

Publication includes GPU, NVLink, InfiniBand, CPU/memory/NUMA, and Lustre JSONL
streams for all four nodes. Missing streams and collector errors remain explicit;
no measurements are replaced with zero. Other collectors, RL traces, and raw
process logs are not automatically exported by this allowlisted publisher.
Credentials, task transcripts, hidden tests, and checkpoints are excluded.

Snapshots live at `telemetry/vultr-b200-slurm/<run-id>/job-<id>/` as four small,
human-readable files: a generated `README.md`, a ClusterMAX-style
`telemetry.values.json` (or `.failed.json`, `.partial.json`, `.skipped.json`),
`timeline.csv`, and `checksums.sha256`. **No raw JSONL, chunk trees, or giant
compressed archives are committed.**

The summary keeps node-level distributions, concise per-GPU measurements,
GPU/rail/link extremes, every collector error time, health-counter exceptions,
and source paths/checksums. Constant inventory values are written once; repeated
zero ECC/IB counters collapse to observed-series counts. The minute-level CSV
retains sample counts and min/mean/p95/max envelopes, not millions of repeated
samples. Means are sample-weighted; invalid counter intervals and missing samples
are not replaced with zero. IB rates stay per rail instead of summing clocks that
were sampled at different times. Full-resolution raw data, lifetime Lustre
statistics, and detailed per-link distributions remain in the run evidence.

The presentation follows ClusterMAX's compact topology-aware tables and explicit
failure/caveat conventions, without copying private source or internal provider
reports into this public repository. Each periodic update replaces the same small
summary files. Download/resume caches remain under the ignored `runs/` directory.
A successful publication does not imply that the telemetry or training passed.

To backfill a job, or restart its watcher with `--watch`:

```sh
python3 scripts/publish_telemetry.py \
  --run-dir /absolute/path/to/run \
  --kubeconfig /absolute/path/to/vultr-kubeconfig \
  --stream-label sync-grpo-v9 --job-id 143 --push
```

Only the job's telemetry directory is staged. Pre-staged unrelated changes,
changed source boundaries, unreviewed schemas, and credential-like contents cause
an explicit failure. Watcher logs and its startup receipt remain under
`runs/.../provenance/telemetry-publisher-job-<id>/`. Inspect those logs after a
publication failure; do not infer that a spawned watcher has successfully pushed.

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
`squeue` and `sacct`, not repeated. The latest local environment gate passed ten
CPU-only checks in job 146; that result does not qualify the full task corpus or
replace model-driven and asynchronous accounting tests.

Run the local parser and evidence checks with:

```sh
python3.12 -m unittest discover -s tests -v
```
