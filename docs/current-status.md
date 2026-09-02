# Campaign status

Generated from `docs/current-status.json`.

Run: `20260902-172037-a3b210`. **No training or held-out quality measurement has run.**

## Completed work

- Native job 109: four nodes, 32 reconciled B200 GPUs, all-reduce correctness on each node and all four nodes, 256 MiB verified fio per node.
- Committed B200 Miles launcher, whole-node host mapping and safe external-Ray option. Three patches replay to the exact same Git tree.
- Patched Linux CPU tests: 595 passed, 12 failures, 4 errors. Unpatched baseline: 564 passed, same 12 failures and 4 errors; no new failing test IDs.
- Separate hashed offline Verifiers 0.3.1 / Harbor 0.21.0 dependency lock; three strict-online dependency conflicts reproduced without installation.
- Native statistics and plots derived from retained JSONL. Read-only perfquery probes returned counters for all 32 active 400G rails without errors.

## Remaining gates

| Gate | State | Smallest next step |
|---|---|---|
| Hidden-test isolation | failed source review | Versioned local OpenEnv isolation/grading adapter and adversarial tests. No policy execution until it passes. |
| Full infrastructure telemetry | failed | Validate explicit perfquery backend under load, expose a read-only Lustre client statistics source, and obtain complete per-GPU monitoring. Do not repair cluster-wide settings. |
| Broad launcher suite | failed upstream baseline and diagnostic image | Preserve identical failures. Missing envsubst and existing launcher/snapshot errors need separate fixes before claiming a green full suite. |
| GPU runtime and provenance | not executed | Pull pinned GPU image/model, verify package and model hashes, validate B200 MTP serving. |
| GRPO, async and resume fidelity | not executed | Prove grouping/logprobs/gradient updates and full restart state. Miles async buffer/policy accounting restore is not implemented yet. |
| 3T/1R batch compatibility | unvalidated | Check pinned Megatron dense-DP microbatch divisibility. Do not silently round or drop the fixed global batch. |
| Task corpus and offline TB2.1 runtime | not executed | Pin ordered clean training subset and untouched evaluation suite; validate environment and evaluator before examining baseline outcomes. |
| Placement sweep and quality hill climb | not executed | Pass preceding gates, freeze budget and settings, run warmup plus three rotated repetitions, then the longer selected layout and paired checkpoint evaluation. |

## Quality budget

400 steps is a proposal, not a guarantee of improvement. The operator wall-clock/GPU-hour cap has not been specified. No positive quality delta is claimed.
