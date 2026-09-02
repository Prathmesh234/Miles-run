# Local task environments on Vultr

## Decision

The operator asked whether task environments can run on the cluster itself.
The campaign will validate self-hosted Docker environments through OpenEnv.
Daytona credentials are not required for this architecture. No environment or
training workload has yet been launched as part of this decision.

## Current validation status

Docker API access was verified on all four Slurm workers. This establishes local
runtime availability, not task isolation or end-to-end correctness.

A source review of OpenEnv revision
`b9d8c1f953e0c3e0bbee2f3f6f6c73d8eae61f5f` found three launch blockers in
`envs/tbench2_env/server/tbench2_env_environment.py`:

1. `_start_container` selects host networking. Policy commands must not inherit
   access to the controller through the host network namespace.
2. The same Docker call provides no CPU, memory, or process-count limits.
3. `_evaluate_docker` copies hidden tests into the live policy container.
   Excluding tests at reset does not prove isolation from agent-created
   background processes during grading. This is a source-level exposure risk;
   no leakage experiment or policy episode has been executed.

Policy execution is stopped at this gate. The smallest proposed change is a
version-controlled local-runtime adapter with explicit resource limits, isolated
networking, run-owned cleanup, and a grading boundary that never returns hidden
assets or verifier output to a live policy session. A separate grading container
from a stopped task's filesystem is a candidate, but its task semantics and
resistance to agent-planted processes must be tested before adoption. Changing
Docker options alone does not establish hidden-test isolation.

The required evidence includes tests for background-process access, symlinks,
controller reachability, missing verdicts, disconnect cleanup, and task-semantic
equivalence. The original failed gate must remain in the run bundle after any
patch; a new validation phase records the outcome of the proposed fix.

## Source evidence

Miles is pinned to `0709889b2848f293b5575d50aa3340fa4de5a20d`.
`examples/experimental/openenv/README.md`, under "Alternative: one shared env
server", documents `TB2_MODE=docker` and a container for each task. The launcher
`examples/experimental/openenv/run-openenv-tbench2.py` selects the shared-server
adapter through `openenv_env_url`. It does not require a sandbox provider.

That example's GLM model and colocated GPU topology are not adopted. The Qwen
recipe and four-node, disaggregated GPU role layouts remain required.

## Proposed placement

Each physical GPU node receives a bounded share of CPU-only task environments.
The environment capacity, CPU and memory limits, disk budget, routing policy,
and physical host assignment stay fixed across 1T/3R, 2T/2R, and 3T/1R.
Changing the number of rollout nodes must not silently change environment
capacity. Each GPU node remains an indivisible trainer or rollout role unit.
This CPU task placement does not enable Miles `--colocate`.

Separate collectors attribute CPU, memory, disk, metadata traffic, process
startup, container creation, episode latency, and cleanup costs to environments.
The reports must identify contention between environments and GPU services.

## Required validation

1. Confirm Docker and exact runtime versions on all four workers.
2. Pin OpenEnv, all Python dependencies, task images, and the taskset revision.
3. Keep the controller and grader outside policy containers. Do not expose
   controller source, hidden tests, solutions, socket, credentials, or shared data.
4. Check network isolation from cluster administration and other task containers.
5. Confirm CPU, RAM, process count, file-descriptor, disk, and timeout limits.
6. Prove test withholding before and after grading, including background processes.
7. Prove transcript and reward capture, and distinguish task failure from grader error.
8. Simulate a session disconnect and check that only this run's labeled containers
   are removed. Do not use a global Docker prune or a broad container sweep.
9. Run a known-good clean training task, followed by two policy trajectories
   with no optimizer step, before enabling training.

The upstream shared-server documentation warns about orphaned containers and
file-descriptor leakage after unclean disconnects. These are validation targets,
not behaviors that the campaign can assume are already handled.

## Scientific constraints

The clean training data remains Terminal-Lego or TMAX. TB2.1 stays evaluation-only.
Offline Verifiers 0.3.1/Harbor uses its separate Python 3.12 environment. Local
containers do not resolve the strict online Miles/Verifiers dependency conflict.
