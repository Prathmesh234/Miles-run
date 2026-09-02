# Campaign status

Generated from `docs/current-status.json`.

Run: `20260902-172037-a3b210`. Training started: **False**. Held-out quality measured: **False**.

## Completed work

- Native job 109: four nodes, 32 reconciled B200 GPUs, all-reduce correctness on each node and all four nodes, 256 MiB verified fio per node.
- Committed B200 Miles launcher, whole-node host mapping and safe external-Ray option. Four patches replay to the exact same Git tree.
- Patched Linux CPU tests: 595 passed, 12 failures, 4 errors. Unpatched baseline: 564 passed, same 12 failures and 4 errors; no new failing test IDs.
- Separate hashed offline Verifiers 0.3.1 / Harbor 0.21.0 dependency lock; three strict-online dependency conflicts reproduced without installation.
- Native statistics and plots derived from retained JSONL. Read-only perfquery probes returned counters for all 32 active 400G rails without errors.
- Qwen3.6-35B-A3B revision 995ad96eacd98c81ed38be0c5b274b04031597b0: all 40 files, 71,926,865,825 bytes downloaded and individually hash-verified. Deterministic EP8 serving passed in job 115; an MTP-preserving candidate checkpoint was subsequently created in job 117.
- Pinned Miles image imported as a 32,608,673,792-byte SquashFS, SHA256 2262e639192be83c00b285a79bcda07c6a23a06295cb4828aecc879e0f2d2698. Private tmpfs overlay workaround avoids Lustre overlay incompatibility; Docker VFS amplification and failed imports are retained.
- CPU image provenance captured: Python 3.12.3, Torch 2.13.0+cu130, CUDA toolkit 13.0.88, NCCL 2.29.7, Ray 2.58.0, SGLang 0.5.19.dev52+g98bb145, OpenAI 2.6.1. The bundled Miles SHA differs from the requested pinned source and must not be used accidentally.
- Native job 110: four-node all-reduce correctness and read-only perfquery load validation passed on all 32 active 400G rails; no counter resets or collector errors.
- Native job 111: verified 2 GiB fio per node and 180 one-second host Lustre samples per node; all four collectors passed load coverage without errors. Host statistics are read-only mounts in run-owned pods; cluster settings unchanged.
- Native job 112: all eight B200 GPUs in the node-0 Enroot container reconcile to the frozen UUIDs; deterministic BF16 matmul passed on every GPU. Run-scoped NVIDIA_VISIBLE_DEVICES=all; no model or optimizer step.
- Root preparation/evidence suite: 27 tests passed. The full Miles suite remains at its previously documented baseline failures.
- Terminal-Bench 2.1 evaluation metadata frozen: 89 ordered task IDs at git 7131e4375048a0e408a8fb404b5f499d726b695b. No task instructions, hidden tests, solutions or oracle files fetched by this metadata stage; Harbor 0.21 compatibility remains untested.
- Job 115 completed in 4m14s: one node-local Qwen TP8/EP8 serving engine, MTP off then on, both deterministic prompts correct, exact output-token/logprob counts valid, positive MTP acceptance. This is not RL or controlled inference benchmarking.
- Serving telemetry audit: six finalized native/host-Lustre streams, no collector-error rows, eight GPU UUIDs, 88 per-GPU metric distributions. Forward-timer gauge remained unavailable during the very short requests; full workload telemetry remains unqualified.
- Isolated OpenEnv Python 3.12.11 development stack resolved with hashes. Baseline tests 26 passed/1 skipped; archive-staging hardening 32 passed/1 skipped, clean Ruff check and exact patch replay verified. Hidden-test grading isolation is still not implemented.
- Pinned full Megatron image git SHA 8c1e05747eb612b382df2632783df5c83a853646. Campaign Miles b61dbe83ee815412b72c84ed367ffd329d7922d4 runtime sources materialized on shared storage from their verified archive; one non-runtime agent-skill symlink omitted explicitly.
- Job 117 completed in 4m36s: verified campaign Miles/Megatron imports, a two-rank loopback rendezvous check, and eight-rank cookbook checkpoint conversion. All 11 payload files (71,108,393,235 bytes) hashed; 539 MTP-related metadata entries present (includes extra_state; later audit identifies 16 weight tensors). Conversion command 192.06s; hashing 32.65s. This is a candidate checkpoint, not proven tensor parity or a training checkpoint.
- Conversion telemetry audit: six non-empty native/host-Lustre streams, eight GPU UUIDs, no collector errors. Megatron argument-consistency, router-dtype, deprecated-save, unrelated ASR docstring and empty-stack log messages are retained for review.
- Exact pinned Megatron microbatch calculator exercised at dense DP 8/16/24. Batch64 rejected at DP24; batch96 passed all three without rounding. Stage4 stays at 64; a common Stage5 batch96 is a prospective proposal, still requiring full-trainer/gradient validation.
- Separate hash-pinned OpenEnv Linux x86_64/Python3.12.11 server lock resolved without changing the GPU stack. Runtime/client compatibility and live policy/grader isolation remain open.
- Full CPU parity check executed in Slurm job118: 32,118 comparisons, including 785 MTP components, covering 712 HF text/MTP tensors. 32,088 were dtype- and byte-exact; the 30 A_log dtype mismatches intentionally failed the gate. All six telemetry streams finalized without collector errors.
- Independent CPU-only diagnosis verified all 960 A_log scalars are exact, reversible BF16-to-FP32 widenings and match the prior comparison hashes. Pinned upstream source explicitly performs this widening. A sub-BF16 perturbation negative control passed; original failure and checkpoint remain unchanged.
- Job 119 completed with exit0 in 4m12s: 32,118 qualified text/MTP comparisons, 32,088 byte/dtype-equal and 30 exact reversible A_log widenings. All 785 MTP components pass. The original strict-dtype failure in job118 remains unchanged. Six telemetry streams, zero collector errors; native sample gaps reached 3.30s.
- New Miles commit 346946ae870be97e9cb6f4e8b7214c7fcf66c041 adds explicit pinned checkpoint paths, rejects incompatible dense-DP batches and removes obsolete SGLANG_ENABLE_SPEC_V2. Targeted tests:14 passed. Linux:597 passed/12 failures/4 errors; macOS:461 passed/30 failures/122 errors. Failing IDs match each prior patched suite.
- Four Miles patches replay to exact tree c48ad99923d5f0815a1424ba15f58e1d3186320b; patch SHA256 b5cd7e36465a3de2fb7d2abcc533ee86b0645df41d12c34af99b0bb0c6b9b435. Whitespace-error replay failed on original snapshot formatting; warn-and-preserve replay succeeded without editing bytes.
- New Miles runtime source 346946ae8 materialized at provenance/training-source-v2/miles: 1,830 regular files, manifest SHA256 8155596d8f3f65a5a2f47378d17db5e0af702d334c31d66472916841ade8f356. Prior b61 source used for conversion/parity remains intact.
- Terminal-Lego revision 9c197f1c2e87b64cc316b1a5bfcef57b584929f0: all 15,049 catalog IDs pinned before outcomes; deterministic 512 training/128 development tasks plus reserved runtime task_00000. Independently rehashed 641 task sources, 5,097 files and 10,748,004 bytes. Controller directory0700; images/oracles/tasks not executed. Two failed acquisition attempts retained; third resolved three LFS assets against their pinned SHA256.
- CollectiveX source pinned at InferenceX 9ddcc1fe59f4f2cf621667e24e90587288907956, DeepEP V2 pin 01dc3aaac82068020353dce2c302e38153c0bfaa. Archived 31 source files. Static image inspection identifies incompatible legacy DeepEP API; no collective performance claim.

## Remaining gates

| Gate | State | Smallest next step |
|---|---|---|
| Hidden-test isolation | failed source review | Archive-staging hardening passed unit tests. Still implement and adversarially test local policy/grader separation, resource/network limits, and a dependency-compatible client before policy execution. |
| Full infrastructure telemetry | partially recovered; full gate still open | Use validated perfquery and host Lustre collectors in workload launchers. DCGM still misses node 3; full required GPU/fabric/CPU/Ray/SGLang/Miles coverage has not been proved during RL. |
| Broad launcher suite | New launcher597passed, same12failures/4errors in pinned Linux image; no new failing IDs | Preserve baseline failures; do not call the full suite green. Continue the targeted runtime qualification of the committed launcher. |
| GPU runtime and provenance | Job119 qualified text/MTP tensor parity under contractv2; trainer execution still unvalidated | Run TP1/EP8/PP1 trainer reshard, forward/logit and gradient checks using pinned source346946ae8. Do not treat CPU tensor equality as optimizer or resume validation. |
| GRPO, async and resume fidelity | not executed | Prove grouping/logprobs/gradient updates and full restart state. Miles async buffer/policy accounting restore is not implemented yet. |
| 3T/1R batch compatibility | Exact pinned calculator rejects64 at denseDP24; launcher now rejects that configuration early | Validate global96/rollout_batch12/group8 in full trainer across all three layouts before campaign freeze. Stage4 reference stays64. |
| Task corpus and offline TB2.1 runtime | Prospective clean split pinned; 641 controller-only task sources and all5097 files verified. Runtime and evaluator remain unvalidated. | Pin task image digests and validate local sandbox/grader isolation, task correctness and separate offline Verifiers/Harbor runtime. Never replace failed tasks implicitly or train on TB2.1. |
| Placement sweep and quality hill climb | not executed | Pass preceding gates, freeze budget and settings, run warmup plus three rotated repetitions, then the longer selected layout and paired checkpoint evaluation. |
| Local sandbox disk quota | Hard quota probe failed before task container start | Docker VFS backing filesystem does not support the requested per-container quota. Qualify a bounded alternative with free-space monitoring; no daemon/cluster changes or implicit resource guarantees. |
| CollectiveX normal EP8 runtime | Pinned Miles image exports legacy Buffer, not required ElasticBuffer | Qualify a separate pinned collective runtime for DeepEP V2. Upstream setup overrides NCCL to2.30.4; do not apply it to Miles. No BF16/FP8 CollectiveX GPU case has run. |

## Quality budget

Initial plan: 400 optimizer steps, with TB2.1 evaluation at steps 0/50/100/200/400 and approximately one-second infrastructure collectors throughout. Prospective common batch96 implies 38,400 eligible training trajectories; Stage4 reference remains64. Budget cap and remaining correctness/isolation gates still need resolution. This is not a guarantee of improvement; training and held-out quality measurement have not started.
