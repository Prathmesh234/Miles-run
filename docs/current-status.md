# Campaign status

Generated from `docs/current-status.json`.

Run: `20260902-172037-a3b210`. Optimizer steps verified: **Job167:2; job161:2; jobs154/143:3 each. These are validation updates, not a completed measured benchmark.**. Held-out quality measured: **False**.

Latest training allocation: Slurm **167**, 32 GPUs, 2T/2R, 2 requested steps. 2 optimizer updates,2 saves,32 tensor-audited samples,48 accounted episodes,16 host-release receipts; native telemetry and periodic actor-placement audits passed. Not full resume, asynchronous qualification or held-out quality.

**Allocation status: COMPLETED 0:0; synchronous component validation passed**

Status snapshot: `2026-09-03T04:22:37.834478Z`; inspect Slurm for live state.

## Historical milestones (later gates supersede earlier limits)

- Pinned B200 launcher implements whole-node 1T/3R,2T/2R,3T/1R layouts. All22 Miles patch commits reconstruct the recorded source tree; see patches/manifest.json.
- Model995ad96:40 files/71,926,865,825 bytes individually verified. Native image32,608,673,792 bytes verified; exact image/package/model provenance remains in the run inventory.
- Bounded preflight109–111 reconciled32 B200 GPUs, all32 active400G IB rails, node-local/four-node collective correctness, 2GiB fio per node and180 one-second Lustre samples per node. Not burn-in or saturation characterization.
- Serving115 passed EP8 MTP-off/on deterministic request checks. Conversion119 passed32,118 comparisons with30 documented reversible A_log widenings; trainer120 had finite main/MTP gradients. See docs/checkpoint-parity.md.
- Clean corpus: Terminal-Lego9c197f has512 train/128 development tasks;5,097 source files rehashed. Jobs134/146 qualified only five task images, with sealed grading, timeout and cleanup checks. Full corpus runtime remains open.
- Untouched evaluation:89 TB2.1 task IDs pinned at7131e4 and excluded from training. Separate Python3.12/Verifiers0.3.1/Harbor0.21 image142 built; no TB evaluation executed. Strict online Miles/Verifiers dependency conflicts remain blocked.
- Synchronous reference167 completed two updates/two saves on32 GPUs. All32 trainer samples and48 episodes reconcile:16 unused discards,zero unresolved. See docs/grpo-validation-job167.md; no held-out quality claim.
- Job167 native telemetry:24 finalized streams/3,378,085 records,zero collector errors,maxGPU sample gap3.338s. All16 trainers released423,352,733,696 pinned bytes;166 Ray snapshots verified whole-node actor placement. Full metric coverage and async qualification remain open.
- Current tests:76 root audit/report tests;101 pinned native Miles journal/parser/cleanup tests in166;32 targeted launcher checks. Historical broad suites had platform/dependency failures and are not reported green.
- Checkpoint probe170 passed native CPU loader/comparison and corruption controls with exact replay imports. Fixture writer uses native MCore formatting plus Torch synchronous CPU I/O; no full-model or optimizer restore claim yet.
- Optional MXFP8 conversion148 reduced tensor payload71.904GB to39.502GB, but serving149–151 failed before readiness. No quantized optimizer step or speedup claim; BF16 remains the baseline.
- Retention removed only authorized superseded dummy checkpoint payloads and preserved metadata. Six full training checkpoints remain after167; no S3 archive configured. Public telemetry uses four dense files per job; full-resolution evidence stays outside Git.
- Job171 failure bundle exported307 files; all eight pre/post-allocation inventory checks passed. Final dense telemetry records256,407 samples,zero collector errors, and correctly labels the job failed; no optimizer execution.

## Remaining gates

| Gate | State | Smallest next step |
|---|---|---|
| Hidden-test isolation | Job 146 passed all ten local-runtime checks, including controller episode identity reconciliation; no leaked containers. Scope remains the five qualified task images. | Extend runtime qualification to the frozen full subset. Mid-command WebSocket disconnect remains untested; no policy/grader co-residency. |
| Full infrastructure telemetry | Job167 passed the native synchronous telemetry/cleanup gate without relaxed thresholds:0 errors,maxGPU gap3.338s versus earlier roughly15s. Full DCGM/XID/throttle,complete RL pipeline/SGLang coverage and asynchronous qualification remain open. | Preserve job167 as the passing synchronous reference; qualify remaining metric families and fully asynchronous overlap/accounting before the measured sweep. |
| Broad launcher suite | 76 root audit/report tests and32 targeted launcher tests passed; pinned cleanup/parser components101 passed. Broad macOS/native suites retain documented failures; no full-suite pass. | Preserve baseline failures; do not call the full suite green. Continue the targeted runtime qualification of the committed launcher. |
| GPU runtime and provenance | Job120 passed native TP1/EP8/PP1 checkpoint load and packed forward/backward; all parameter hashes unchanged, finite main/MTP gradients on eight GPUs. | Preserve checkpoint parity and native tensor audit evidence. Qualify actual full-state resume and complete runtime telemetry before the async benchmark. |
| GRPO, async and resume fidelity | Native CPU replay fixture170 passed. GPU replay171 stopped before model initialization due to a frozen-input identity mismatch; zero optimizer updates. Original failure retained; full resume remains unqualified. | Audit the frozen-input replay without relaxing equality. Then qualify policy version, data source and quiesced async accounting at checkpoint boundaries. |
| 3T/1R batch compatibility | Exact pinned calculator rejects64 at denseDP24; launcher now rejects that configuration early | Validate global96/rollout_batch12/group8 in full trainer across all three layouts before campaign freeze. Stage4 reference stays64. |
| Task corpus and offline TB2.1 runtime | Full source hashes and base digest bindings frozen. Full corpus runtime unqualified. Offline package image passed; local Docker CLI, sealed grading, task image preparation, and missing-verdict error handling remain open. | Qualify remaining training runtime/images and the separate offline evaluator. Never replace failed task IDs implicitly or train on TB2.1. |
| Placement sweep and quality hill climb | not executed | Pass preceding gates, freeze budget and settings, run warmup plus three rotated repetitions, then the longer selected layout and paired checkpoint evaluation. |
| Local sandbox disk quota | Docker VFS hard quota unsupported; file-only runtime uses read-only root and size-limited run-owned tmpfs volumes. Live isolation/timeout/cleanup tests passed. | Retain bounded resources and free-space guards; do not generalize this runtime to unrestricted service tasks. |
| CollectiveX normal EP8 runtime | Pinned Miles image exports legacy Buffer, not required ElasticBuffer | Qualify a separate pinned collective runtime for DeepEP V2. Upstream setup overrides NCCL to2.30.4; do not apply it to Miles. No BF16/FP8 CollectiveX GPU case has run. |
| Optional Qwen MXFP8 runtime | Kernel/export and serialized conversion passed; EP8 serving failed before readiness in149/150/151. No quantized optimizer steps. | Qualify matched BF16 exceptions for affected TP8 dense projections, or routed-experts-only MXFP8. Preserve BF16 baseline and do not bypass original infrastructure/async/resume gates. |

## Quality budget

Initial plan: 400 optimizer steps with untouched TB2.1 evaluations at 0/50/100/200/400 after correctness and infrastructure gates. The three-step dummy validation is not the quality budget. Prospective common batch 96 implies 38,400 eligible trajectories; Stage 4 reference remains 64. No held-out quality delta or complete resume has been verified; improvement is not guaranteed. Retention of superseded dummy checkpoints is operator-authorized.
