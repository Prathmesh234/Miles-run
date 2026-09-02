# Dependency and runtime compatibility

Generated from `docs/compatibility.json`.

| Path | Dependencies | Status | Evidence |
|---|---|---|---|
| Miles documented online Verifiers adapter | verifiers>=0.2.0,<0.2.1; openai-agents<0.5; documented SGLang OpenAI==2.6.1 | Strict 0.3.1 combination blocked | Three unsatisfiable resolver results; no installation |
| Miles OpenEnv online, local Docker | GPU image OpenAI==2.6.1 unchanged. Hash-pinned Python 3.12.11 OpenEnv server in a separate controller image; thin WebSocket client uses the native Miles sampler and TITO tracing. | Local isolated-container runtime qualified for four clean training task IDs; model-driven training validation in progress | Slurm job 134: ten live CPU runtime checks passed, including five reference solutions, isolation, real failure versus harness error, timeout, WebSocket roundtrip and cleanup. Model trajectories and GRPO remain separate qualification gates. |
| Offline Verifiers/Harbor evaluation | Python 3.12.11; verifiers==0.3.1; harbor==0.21.0; openai==2.54.0; openai-agents==0.20.0; renderers==0.1.11 | Pinned package environment passed; runtime gates remain: missing Docker CLI, image preparation, sealed grading, and missing-verdict error classification. | Job142 built/imported the hash-locked image. Installed-source inspection and an actual HarborTask synthetic-runtime probe are retained; no TB2.1 tasks executed. |
| Strict Verifiers split-process bridge / upgrade | Not implemented | Blocked before optimizer steps | No equivalence tests have passed |

## Findings

- Miles adapter requirements are recorded from the pinned source; OpenAI==2.6.1 has now also been verified inside the pinned GPU image.
- Published Verifiers 0.3.1 metadata directly requires OpenAI>=2.9.0 and Python>=3.11,<3.14. Its Harbor extra activates harbor==0.21.0 only on Python>=3.12; Harbor itself requires Python>=3.12.
- The offline lock resolves OpenAI 2.54.0. That resolved version must not be confused with Verifiers direct minimum requirement. The frozen openai-agents 0.20.0 requires OpenAI>=2.45.0,<3.
- Resolution used uv 0.11.19, Python target 3.12.11, x86_64-unknown-linux-gnu and package cutoff 2026-09-02T00:00:00Z. It is not proof that task loading, tools, rewards, cleanup, or inference work.
- Terminal-Bench 2.1 metadata is pinned to 7131e4375048a0e408a8fb404b5f499d726b695b with 89 ordered task IDs. Environment-package revision, task-image pins and runtime evaluation remain unvalidated.
- Offline Verifiers evaluation does not generate the OpenEnv online training trajectories. No forced package upgrades were attempted.
- OpenEnv server dependencies conflict with GPU image OpenAI==2.6.1. The server remains isolated. The local thin WebSocket transport passed a live maintained-OpenEnv protocol roundtrip; it does not re-tokenize or replace Miles sampling. This is not a Verifiers compatibility bridge.
- The live runtime qualification covers task_00000 (runtime only) and task_06652, task_14118, task_10753, task_09467 (the first four pre-registered training IDs). It does not qualify all 512 training tasks, model trajectories, mid-command WebSocket disconnection, GRPO gradients, or offline TB2.1 evaluation.
- Installed Verifiers v1 includes a local Docker runtime; no external sandbox provider is intrinsically required. Its default unrestricted runtime uses host networking, and its restricted helper refers to alpine:3.22. Explicit isolated network policy and digest-pinned helper images are required before evaluation.
- The installed Harbor taskset rejects Dockerfile-only environments unless ignore_dockerfile is enabled. Do not ignore task Dockerfiles: prepare the exact task image and record the derived image binding instead.
- The installed default shared Harbor scorer stages tests in the still-existing agent runtime. Separate-verifier mode is supported only when task artifacts and verifier environment are declared. Policy background-process isolation must be proved, not inferred from the agent loop ending.
- Actual pinned HarborTask._graded probe: missing verifier files after exit7 -> reward0.0, legitimate negative ->0.0, positive ->1.0. An explicit missing-verdict error guard is required to avoid reporting runtime failures as task failures.
- Full clean source audit verified all641 task sources (512 train,128 development,1 runtime). It found313 absent public task_file directories,231 harness-setup reviews,68 additional base-image reviews,and2 policy-context asset-name reviews across all roles. These are structural review requirements, not a reward-based exclusion or a claim of leakage.

## Strict online acceptance tests

All must pass before enabling optimizer steps:

- Exact sampled token IDs
- Per-token sampled logprobs
- Renderer and chat-template identity
- n-sample group identity and boundaries
- Rewards and error categories
- Tool traces and masks
- GRPO advantages, loss and gradients on identical data

## Reproducer

```sh
python3 scripts/check_compatibility.py --run-dir /absolute/path/to/a/fresh/run
```

Dependency resolution only; expected conflicts never trigger forced installation.

## Locks

- `locks/offline-verifiers-py312-linux.lock`: SHA256 `e0581971c6c19df4ac6aed6fa3d8cd34665b824137738e87c1e3e9e17df08be0`.
- `locks/openenv-server-py312-macos.lock`: SHA256 `be25b5fdfac79d651578949c1b768ed3d43ec1fd981d4649e5126c287552e461`.
- `locks/openenv-server-py312-linux.lock`: SHA256 `763e72aee51b4d3a3febdc1d14801d89b8215c4b8d49a99f6e0951e42cbae311`.

## Sources

- [harbor==0.21.0](https://pypi.org/pypi/harbor/0.21.0/json)
- [openai-agents==0.20.0](https://pypi.org/pypi/openai-agents/0.20.0/json)
- [openai-agents==0.8.2](https://pypi.org/pypi/openai-agents/0.8.2/json)
- [verifiers==0.3.1](https://pypi.org/pypi/verifiers/0.3.1/json)

Evidence run: `20260902-172037-a3b210`.
