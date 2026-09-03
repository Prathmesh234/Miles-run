# Campaign status

Generated from `docs/current-status.json`.

Run: `20260902-172037-a3b210`. Optimizer steps verified: **Job154: 3; job143: 3 historical validation updates. Neither qualifies the measured benchmark.**. Held-out quality measured: **False**.

Latest training allocation: Slurm **154**, 32 GPUs, 2t2r, 3 requested steps. Four-node BF16 synchronous validation completed three updates. Tensor audit passed for all 48 trained samples. Native telemetry stalled for 15.22/15.48s on trainer nodes during teardown; 32 of 80 controller episodes lack native sample joins. Not async, resume, or held-out quality qualification.

**Allocation status: FAILED, Slurm exit 1:0; three optimizer updates and three saves verified**

Status snapshot: `2026-09-03T02:13:30.568423Z`; inspect Slurm for live state.

## Historical milestones (later gates supersede earlier limits)

- Committed B200 launcher with whole-node role mapping and separate trainer/rollout placement. The current 17-commit Miles patch series replays to its recorded tree; runtime source remains pinned independently of the image-bundled checkout.
- Four-node preflight: jobs 109/110 reconciled32 B200 GPUs and exercised node-local/four-node all-reduce correctness plus all32 active400G IB rails. Job 111 passed 2 GiB fio per node and180 one-second Lustre samples per node. These are bounded checks, not saturation/burn-in results.
- Pinned runtime: Python 3.12.3, Torch 2.13.0+cu130, CUDA 13.0.88, NCCL 2.29.7, Ray 2.58.0, SGLang 0.5.19.dev52+g98bb145, OpenAI 2.6.1; Megatron 8c1e05747eb612b382df2632783df5c83a853646. Run-local Enroot settings and NVIDIA_VISIBLE_DEVICES=all leave cluster-wide configuration unchanged.
- Qwen model revision 995ad96eacd98c81ed38be0c5b274b04031597b0:40 files /71,926,865,825 bytes individually hash-verified. Pinned32,608,673,792-byte Miles image SHA256 2262e639192be83c00b285a79bcda07c6a23a06295cb4828aecc879e0f2d2698.
- Job 115 passed deterministic node-local EP8 serving with MTP off/on, exact output-token/logprob counts and positive acceptance. Six finalized telemetry streams; short-request forward-timer gauge unavailable. Not a concurrency benchmark.
- Conversion/parity: job 117 generated an MTP-preserving checkpoint; job 119 verified 32,118 comparisons including 785 MTP components, with 32,088 byte/dtype-equal comparisons and30 documented reversible A_log widenings. Original job 118 strict-dtype failure remains preserved.
- Job 120 loaded the cookbook TP1/EP8 trainer on eight GPUs:3,191 unchanged parameter hashes per rank, finite nonzero main/MTP gradients and consistent diagnostic logprobs. No optimizer constructed; not full resume or GRPO validation.
- Training corpus: Terminal-Lego 9c197f1c2e87b64cc316b1a5bfcef57b584929f0;512 train/128 development/1 runtime tasks,5,097 files rehashed. All18 task base tags plus uv helper have frozen digest bindings. Full-corpus image/reference qualification remains pending.
- Evaluation:89 ordered TB2.1 task IDs frozen at 7131e4375048a0e408a8fb404b5f499d726b695b, excluded from training. Job 142 built/imported separate locked Verifiers 0.3.1/Harbor 0.21.0 image; no TB task evaluation executed. Three strict-online dependency conflicts reproduced without forced installation.
- Local environments: jobs 134/146 passed reference solutions, sealed policy/grader isolation, timeouts, protocol/episode identity and cleanup checks for five qualified images. OpenEnv regression101 passed/3 skipped. This is not qualification of the full corpus.
- Actual training-container fabric: job 141 verified 32 ranks using IB and all eight400G HCAs per node, with correct all-reduce and node-local EP8 all-to-all. No Socket fallback; communication throughput still unmeasured.
- Broadcast/model protocol: initial target/MTP weight equality passed on both rollout engines; Qwen placeholder and unfinished-final-message comparisons were independently tested without changing sampled token IDs. Miles 977bdee2f fixes canonical speculative counters;23 codec tests passed. Historical job 143 zero MTP acceptance was a field-name bug, not a valid measurement.
- Synchronous GRPO: jobs 143 and154 each executed three updates and three checkpoint saves on32 GPUs; each48-sample native tensor audit passed. Job 143 had 12 collector timeouts. Job 154 failed on15.22/15.48 s trainer-node sampling gaps despite zero collector exception rows. Neither qualifies the measured benchmark.
- Job 154 environment accounting:80 graded episodes,55 pass/25 fail, balanced isolated cleanup.48 episodes join explicitly to trainer samples;32 remain unresolved. These are training-task outcomes, not held-out quality.
- Resume component: job 155 passed seven native CPU cursor/recycled-buffer checkpoint tests. Candidate1411d18 remains opt-in and disabled in training; optimizer, scheduler, RNG, asynchronous queue and policy-version resume remain unverified.
- Telemetry controls: job 153 passed node-local all-reduce collection (max gap1.523 s). Jobs156/157 passed 64 GiB-per-GPU context teardown without/with EP8 NCCL (max gaps2.142/2.004 s); neither reproduced job 154. Separate owned DCGM probe144 returned real node3 identities, but continuous training-load DCGM remains unqualified.
- Optional quantization: jobs 147/148 passed expert MXFP8 byte checks and full conversion (71.904 GB to39.502 GB tensor payload). Metadata-only repackaging preserved weights. Serving149/150/151 failed before readiness; no quantized optimizer step or speedup/quality claim.
- Batch compatibility: pinned Megatron calculator rejects global64 at denseDP24 and accepts96 across DP8/16/24. Stage4 stays64; Stage5 common96 requires actual trainer validation. CollectiveX/DeepEP V2 sources are pinned, but the Miles image has the incompatible legacy API; no CollectiveX performance result.
- Evidence/retention: authorized pruning removed 1,989,034,824,640 bytes of older dummy checkpoint payloads while retaining latest checkpoint and metadata. No S3 archive configured. Public telemetry keeps four dense files per job, not raw chunks; job 154 summarizes3,657,250 records in 156,009 bytes and explicitly fails its coverage gate.
- Tests: root evidence/launcher-preparation64 passed; targeted launcher28 passed; OpenEnv 101 passed/3 skipped. Latest tested full Linux Miles suite had 597 passes/12 failures/4 errors matching its baseline failures; this is not a whole-tree green claim.

## Remaining gates

| Gate | State | Smallest next step |
|---|---|---|
| Hidden-test isolation | Job 146 passed all ten local-runtime checks, including controller episode identity reconciliation; no leaked containers. Scope remains the five qualified task images. | Extend runtime qualification to the frozen full subset. Mid-command WebSocket disconnect remains untested; no policy/grader co-residency. |
| Full infrastructure telemetry | Job154 failed: trainer-node native sample gaps15.22/15.48s, before first NVLink field of the tick; parallel PMA calls <0.31s and separate host Lustre sampling remained live. Individual GPU getter timing was absent; hardware causality unproven. | Instrument the actual trainer/Ray allocation lifecycle; simple memory-context and EP8 NCCL-context controls156/157 both passed and did not reproduce154. Preserve12s deadline and failed coverage state. |
| Broad launcher suite | New launcher597passed, same12failures/4errors in pinned Linux image; no new failing IDs | Preserve baseline failures; do not call the full suite green. Continue the targeted runtime qualification of the committed launcher. |
| GPU runtime and provenance | Job120 passed native TP1/EP8/PP1 checkpoint load and packed forward/backward; all parameter hashes unchanged, finite main/MTP gradients on eight GPUs. | Preserve checkpoint parity and native tensor audit evidence. Qualify actual full-state resume and complete runtime telemetry before the async benchmark. |
| GRPO, async and resume fidelity | Job15448 tensor samples passed;32 of80 controller episodes have no durable native sample join. Job1557 opt-in cursor/buffer tests passed; full restart and async state remain unqualified. | Add durable lifecycle receipts for every sampled group, including discarded/cancelled work; test model/optimizer/scheduler/RNG/queue/policy resume on frozen samples. |
| 3T/1R batch compatibility | Exact pinned calculator rejects64 at denseDP24; launcher now rejects that configuration early | Validate global96/rollout_batch12/group8 in full trainer across all three layouts before campaign freeze. Stage4 reference stays64. |
| Task corpus and offline TB2.1 runtime | Full source hashes and base digest bindings frozen. Full corpus runtime unqualified. Offline package image passed; local Docker CLI, sealed grading, task image preparation, and missing-verdict error handling remain open. | Qualify remaining training runtime/images and the separate offline evaluator. Never replace failed task IDs implicitly or train on TB2.1. |
| Placement sweep and quality hill climb | not executed | Pass preceding gates, freeze budget and settings, run warmup plus three rotated repetitions, then the longer selected layout and paired checkpoint evaluation. |
| Local sandbox disk quota | Docker VFS hard quota unsupported; file-only runtime uses read-only root and size-limited run-owned tmpfs volumes. Live isolation/timeout/cleanup tests passed. | Retain bounded resources and free-space guards; do not generalize this runtime to unrestricted service tasks. |
| CollectiveX normal EP8 runtime | Pinned Miles image exports legacy Buffer, not required ElasticBuffer | Qualify a separate pinned collective runtime for DeepEP V2. Upstream setup overrides NCCL to2.30.4; do not apply it to Miles. No BF16/FP8 CollectiveX GPU case has run. |
| Optional Qwen MXFP8 runtime | Kernel/export and serialized conversion passed; EP8 serving failed before readiness in149/150/151. No quantized optimizer steps. | Qualify matched BF16 exceptions for affected TP8 dense projections, or routed-experts-only MXFP8. Preserve BF16 baseline and do not bypass original infrastructure/async/resume gates. |

## Quality budget

Initial plan: 400 optimizer steps with untouched TB2.1 evaluations at 0/50/100/200/400 after correctness and infrastructure gates. The three-step dummy validation is not the quality budget. Prospective common batch 96 implies 38,400 eligible trajectories; Stage 4 reference remains 64. No held-out quality delta or complete resume has been verified; improvement is not guaranteed. Retention of superseded dummy checkpoints is operator-authorized.
