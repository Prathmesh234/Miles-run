# Campaign status

Generated from `docs/current-status.json`.

Run: `20260902-172037-a3b210`. Training started: **False**. Held-out quality measured: **False**.

## Completed work

- Native job 109: four nodes, 32 reconciled B200 GPUs, all-reduce correctness on each node and all four nodes, 256 MiB verified fio per node.
- Committed B200 Miles launcher, whole-node host mapping and safe external-Ray option. Three patches replay to the exact same Git tree.
- Patched Linux CPU tests: 595 passed, 12 failures, 4 errors. Unpatched baseline: 564 passed, same 12 failures and 4 errors; no new failing test IDs.
- Separate hashed offline Verifiers 0.3.1 / Harbor 0.21.0 dependency lock; three strict-online dependency conflicts reproduced without installation.
- Native statistics and plots derived from retained JSONL. Read-only perfquery probes returned counters for all 32 active 400G rails without errors.
- Qwen3.6-35B-A3B revision 995ad96eacd98c81ed38be0c5b274b04031597b0: all 40 files, 71,926,865,825 bytes downloaded and individually hash-verified. Deterministic EP8 serving passed in job 115; an MTP-preserving candidate checkpoint was subsequently created in job 117.
- Pinned Miles image imported as a 32,608,673,792-byte SquashFS, SHA256 2262e639192be83c00b285a79bcda07c6a23a06295cb4828aecc879e0f2d2698. Private tmpfs overlay workaround avoids Lustre overlay incompatibility; Docker VFS amplification and failed imports are retained.
- CPU image provenance captured: Python 3.12.3, Torch 2.13.0+cu130, CUDA toolkit 13.0.88, NCCL 2.29.7, Ray 2.58.0, SGLang 0.5.19.dev52+g98bb145, OpenAI 2.6.1. The bundled Miles SHA differs from the requested pinned source and must not be used accidentally.
- Native job 110: four-node all-reduce correctness and read-only perfquery load validation passed on all 32 active 400G rails; no counter resets or collector errors.
- Native job 111: verified 2 GiB fio per node and 180 one-second host Lustre samples per node; all four collectors passed load coverage without errors. Host statistics are read-only mounts in run-owned pods; cluster settings unchanged.
- Native job 112: all eight B200 GPUs in the node-0 Enroot container reconcile to the frozen UUIDs; deterministic BF16 matmul passed on every GPU. Run-scoped NVIDIA_VISIBLE_DEVICES=all; no model or optimizer step.
- Root preparation/evidence suite: 22 tests passed. The full Miles suite remains at its previously documented baseline failures.
- Terminal-Bench 2.1 evaluation metadata frozen: 89 ordered task IDs at git 7131e4375048a0e408a8fb404b5f499d726b695b. No task instructions, hidden tests, solutions or oracle files fetched by this metadata stage; Harbor 0.21 compatibility remains untested.
- Job 115 completed in 4m14s: one node-local Qwen TP8/EP8 serving engine, MTP off then on, both deterministic prompts correct, exact output-token/logprob counts valid, positive MTP acceptance. This is not RL or controlled inference benchmarking.
- Serving telemetry audit: six finalized native/host-Lustre streams, no collector-error rows, eight GPU UUIDs, 88 per-GPU metric distributions. Forward-timer gauge remained unavailable during the very short requests; full workload telemetry remains unqualified.
- Isolated OpenEnv Python 3.12.11 development stack resolved with hashes. Baseline tests 26 passed/1 skipped; archive-staging hardening 32 passed/1 skipped, clean Ruff check and exact patch replay verified. Hidden-test grading isolation is still not implemented.
- Pinned full Megatron image git SHA 8c1e05747eb612b382df2632783df5c83a853646. Campaign Miles b61dbe83ee815412b72c84ed367ffd329d7922d4 runtime sources materialized on shared storage from their verified archive; one non-runtime agent-skill symlink omitted explicitly.
- Job 117 completed in 4m36s: verified campaign Miles/Megatron imports, a two-rank loopback rendezvous check, and eight-rank cookbook checkpoint conversion. All 11 payload files (71,108,393,235 bytes) hashed; 539 MTP-related tensor keys present. Conversion command 192.06s; hashing 32.65s. This is a candidate checkpoint, not proven tensor parity or a training checkpoint.
- Conversion telemetry audit: six non-empty native/host-Lustre streams, eight GPU UUIDs, no collector errors. Megatron argument-consistency, router-dtype, deprecated-save, unrelated ASR docstring and empty-stack log messages are retained for review.
- Exact pinned Megatron microbatch calculator exercised at dense DP 8/16/24. Batch64 rejected at DP24; batch96 passed all three without rounding. Stage4 stays at 64; a common Stage5 batch96 is a prospective proposal, still requiring full-trainer/gradient validation.
- Separate hash-pinned OpenEnv Linux x86_64/Python3.12.11 server lock resolved without changing the GPU stack. Runtime/client compatibility and live policy/grader isolation remain open.

## Remaining gates

| Gate | State | Smallest next step |
|---|---|---|
| Hidden-test isolation | failed source review | Archive-staging hardening passed unit tests. Still implement and adversarially test local policy/grader separation, resource/network limits, and a dependency-compatible client before policy execution. |
| Full infrastructure telemetry | partially recovered; full gate still open | Use validated perfquery and host Lustre collectors in workload launchers. DCGM still misses node 3; full required GPU/fabric/CPU/Ray/SGLang/Miles coverage has not been proved during RL. |
| Broad launcher suite | failed upstream baseline and diagnostic image | Preserve identical failures. Missing envsubst and existing launcher/snapshot errors need separate fixes before claiming a green full suite. |
| GPU runtime and provenance | MTP off/on serving passed; candidate MTP checkpoint converted and hashed | Prove all-tensor HF/Megatron round-trip parity using the pinned conversion path, review conversion warnings, and validate trainer forward/backward before optimizer execution. |
| GRPO, async and resume fidelity | not executed | Prove grouping/logprobs/gradient updates and full restart state. Miles async buffer/policy accounting restore is not implemented yet. |
| 3T/1R batch compatibility | Batch64 rejected by exact pinned calculator at dense DP24 | Prospectively validate global96/rollout_batch12/group8 for every Stage5 layout. Keep Stage4 global64 unchanged; no implicit rounding or dropped trajectories. |
| Task corpus and offline TB2.1 runtime | TB2.1 evaluation metadata pinned; clean training subset and evaluator not validated | Freeze a diverse clean training subset; fetch evaluation files only into an isolated evaluator workspace, pin task images, and run the separate Verifiers/Harbor smoke before baseline outcomes. |
| Placement sweep and quality hill climb | not executed | Pass preceding gates, freeze budget and settings, run warmup plus three rotated repetitions, then the longer selected layout and paired checkpoint evaluation. |

## Quality budget

400 steps is a proposal, not a guarantee of improvement. The operator wall-clock/GPU-hour cap has not been specified. No positive quality delta is claimed.
