# Dependency and runtime compatibility

Generated from `docs/compatibility.json`.

| Path | Dependencies | Status | Evidence |
|---|---|---|---|
| Miles documented online Verifiers adapter | verifiers>=0.2.0,<0.2.1; openai-agents<0.5; documented SGLang OpenAI==2.6.1 | Strict 0.3.1 combination blocked | Three unsatisfiable resolver results; no installation |
| Miles OpenEnv online, local Docker | Pinned Miles and OpenEnv sources; GPU image and task-runtime package locks pending | Recommended architecture; environment-isolation gate failed | No policy trajectories or optimizer steps yet |
| Offline Verifiers/Harbor evaluation | Python 3.12.11; verifiers==0.3.1; harbor==0.21.0; openai==2.54.0; openai-agents==0.20.0; renderers==0.1.11 | Dependency lock resolved; runtime and TB2.1 environment not validated | Hash-pinned Linux lock; no install or evaluation execution |
| Strict Verifiers split-process bridge / upgrade | Not implemented | Blocked before optimizer steps | No equivalence tests have passed |

## Findings

- The exact Miles adapter requirements come from examples/experimental/verifiers/requirements.txt at the recorded source revision. The 2.6.1 SGLang pin is documented there, not yet verified from the selected GPU image.
- Published Verifiers 0.3.1 metadata directly requires OpenAI>=2.9.0 and Python>=3.11,<3.14. Its Harbor extra activates harbor==0.21.0 only on Python>=3.12; Harbor itself requires Python>=3.12.
- The offline lock resolves OpenAI 2.54.0. That resolved version must not be confused with Verifiers direct minimum requirement. The frozen openai-agents 0.20.0 requires OpenAI>=2.45.0,<3.
- Resolution used uv 0.11.19, Python target 3.12.11, x86_64-unknown-linux-gnu and package cutoff 2026-09-02T00:00:00Z. It is not proof that task loading, tools, rewards, cleanup, or inference work.
- The Prime terminal-bench-2 environment package revision and untouched TB2.1 taskset still need exact pins. This lock covers the evaluator framework, not a fully validated TB2.1 task runtime.
- Offline Verifiers evaluation does not generate the OpenEnv online training trajectories. No forced package upgrades were attempted.

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

Uses dependency resolution only. Expected strict combinations emit failed phase JSON; they do not install packages.

Offline lock: `locks/offline-verifiers-py312-linux.lock` (SHA-256 `e0581971c6c19df4ac6aed6fa3d8cd34665b824137738e87c1e3e9e17df08be0`).

## Sources

- [verifiers==0.3.1](https://pypi.org/pypi/verifiers/0.3.1/json)
- [harbor==0.21.0](https://pypi.org/pypi/harbor/0.21.0/json)
- [openai-agents==0.8.2](https://pypi.org/pypi/openai-agents/0.8.2/json)
- [openai-agents==0.20.0](https://pypi.org/pypi/openai-agents/0.20.0/json)

Evidence run: `20260902-172037-a3b210`.
