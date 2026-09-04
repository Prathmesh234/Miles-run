# Terminal-Lego: Prime-RL versus Miles

## Result

**Miles job 190 completed two optimizer updates and saved its checkpoint.** Slurm reports `COMPLETED`, exit `0:0`, from **21:57:12 to 22:11:46 UTC on 3 September 2026**. This reproduces the baseline model, task suite, two-update workload and objective using the approved Megatron trainer, but is **not an identical-backend or identical-trajectory experiment**.

The original job 181 run and checkpoint were not deleted, overwritten or resumed. The final audit found all **49 hashed source/config files unchanged**, and unchanged size/mtime for **188 of 189 inventoried artifacts**. The exception is the original, pre-existing `sstat` monitor log, which keeps appending errors. That monitor was left untouched. Large baseline artifacts were checked by metadata, not fully content-hashed.

Final checks found all four Kubernetes nodes Ready without DiskPressure; no Slurm jobs, GPU compute processes, running Miles containers or campaign collectors remained. The isolated image daemons, images, stopped containers, conversion and run artifacts are retained.

## What was held constant

Both runs start from **Qwen3.6-35B-A3B revision `995ad96eacd98c81ed38be0c5b274b04031597b0`**, not from the baseline's trained checkpoint. The dataset revision is `9c197f1c2e87b64cc316b1a5bfcef57b584929f0`; the same pinned Verifiers, renderer, task code and Docker task images were reused. Thirty task-file hashes were checked.

The source task cycle is `task_06652` (XPath/XML), `task_14118` (KnockoutJS), `task_10753` (Docker/systemd/GitLab CI), and `task_09467` (Python multi-key sorting). Both use two training nodes plus two inference nodes, eight B200 GPUs each; two updates; batch 16; task groups of eight; at most 16 concurrent episodes; 8,192-token context; 2,048 tokens per generation; eight turns; and zero off-policy lag.

Sampling settings remain temperature 1, top-p 1, top-k disabled, min-p 0, inference seed 0. AdamW uses learning rate 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay zero, gradient clip 1 and a constant schedule. The custom IPO objective retains epsilon 0.1, KL tau 0.001, reward-minus-group-mean advantages without standard-deviation normalization, and global active-token reduction. Reference KL, MTP, speculative decoding and evaluation are disabled.

The adapter passed **32 exact IPO value/gradient fixtures**, actual pinned TrainClient stop/length tests, conversion of all 32 accepted baseline traces, and native admission/credit tests for an errored group. The first KnockoutJS request's 1,211 prompt token IDs matched the baseline exactly. These checks establish important interface equivalence, not identical end-to-end numerical updates.

## Training and workload comparison

| Measurement | Prime-RL 181: update 1 / 2 | Miles 190: update 1 / 2 |
| --- | ---: | ---: |
| Accepted traces | 16 / 16 | 16 / 16 |
| Unpadded sequence tokens | 81,076 / 80,012 | 84,096 / 85,761 |
| Active training tokens | 47,732 / 48,371 | 51,030 / 52,362 |
| Accepted raw reward mean | 0.375 / 0.4375 | 0.500 / 0.5625 |
| Truncated traces | 13 / 13 | 8 / 13 |
| Loss | -0.00463 / -0.00410 | 0.000502 / 0.00807 |
| Gradient norm | 0.6565 / 0.4350 | 0.3893 / 0.4661 |
| Approximate active trainer path, seconds¹ | 97.61 / 19.40 | 129.23 / 19.01 |
| Baseline forward/backward timer, seconds | 91.30 / 19.28 | Not the same timer |
| Baseline batch wait / Miles harness episode time, seconds² | 94.16 / 93.32 | 143.21 / 123.55 |

¹ Baseline active path is `time/step − wait_for_batch − broadcast_weights − load_data`; Miles uses `perf/train_time`. This removes major non-training stages but does not make instrumentation boundaries identical. Miles's first update includes substantial first-use TileLang and FlashAttention/CuTe compilation. The second update is approximately 19 seconds in both implementations; **one warmed update is insufficient to establish a throughput advantage**. Miles processed 7.6% more active training tokens overall.

² These are intentionally labeled different timers: asynchronous prefetch makes baseline batch wait different from the full generation wall time. Miles's reported first `train_wait` is 242.93 seconds because it also includes initialization readiness; it must not be presented as pure rollout time.

The baseline recorded **120 episodes**; Miles recorded **56**, all with `ok=true` and no episode errors, comprising 16 XML, 16 KnockoutJS, 16 Docker and eight Python episodes. Miles made 342 generation calls. Both accepted 32 traces after native trainability checks and zero-advantage filtering; accepted tasks were KnockoutJS and Docker in both updates. Zero-advantage rejection and scheduling mean equal accepted batch sizes do not imply equal generated work. `ok=true` means the episode executed without a harness error, not that its task was solved.

Miles's raw reward is recorded as `rollout/raw_reward`; its `rollout/rewards` field contains centered training credit and averages zero. Comparing that zero to baseline raw rewards would be misleading. Truncation uses the native trace flag, including turn-limit truncation, not just the 2,048-token generation limit. The observed higher reward is **not evidence of improved model quality**: this is a two-update smoke test with different sampled trajectories and no evaluation.

## Timing and allocation

From inference-ready to saved checkpoint, baseline took **5m41s** (15:24:22–15:30:03 UTC), versus Miles **7m39.65s** (22:02:14.229–22:09:53.881 UTC). This is a useful operational observation, not an isolated backend-speed result: generated work, scheduling, compilation and precision differ.

Miles's initial weight transfer took 22.52 seconds, the transfer between updates 2.98 seconds, and final checkpoint write approximately 17.11 seconds. A final weight sync completed before the driver exited zero at 22:10:04; remaining allocation time captured evidence and stopped job-scoped processes. Job 190 occupied 32 GPUs for **14m34s = 7.77 allocated GPU-hours**. It reused the converted base model and image/filesystem caches, so it is not a cold-start benchmark.

| Earlier attempt | Failure before optimizer updates | Allocation seconds |
| --- | --- | ---: |
| 185 | Unsupported SGLang seed CLI argument | 175 |
| 186 | Ray head/worker port collision; model conversion succeeded | 227 |
| 187 | SGLang breakable-prefill graph shape error | 598 |
| 188 | Initial expert-weight transfer shape error | 372 |
| 189 | Adapter aborted instead of filtering an errored over-context episode | 486 |

Those attempts consumed **16.52 allocated GPU-hours**; including job 190, the six jobs used **24.28 allocated GPU-hours**, excluding non-Slurm image staging. These are reserved-capacity figures, not measured GPU compute time or monetary charges. All attempts are retained. Baseline job 181's four-hour allocation includes roughly 3.5 hours after the trainer finished; that tail is not training time.

## Infrastructure observations

Full topology, constraints, versions and collection details are in [INFRASTRUCTURE.md](INFRASTRUCTURE.md). Both runs used the same four assigned nodes: 32 B200 GPUs, dual EPYC 9575F CPUs per node, NVLink and InfiniBand. No device clocks, power limits or network configuration were changed.

The following driver-sampled statistics cover each run's **inference-ready → checkpoint** window. Means combine samples across each node's eight GPUs; peaks are the largest single-GPU sample. These are not allocator-reserved memory metrics. Different window lengths and sampling cadence limit causal interpretation.

| Node / role | GPU utilization mean %, baseline → Miles | Peak GPU memory GiB, baseline → Miles | Mean GPU power W, baseline → Miles |
| --- | ---: | ---: | ---: |
| 0 / train | 15.6 → 7.8 | 65.7 → 100.9 | 280.3 → 266.5 |
| 1 / train | 15.7 → 9.2 | 52.2 → 123.2 | 275.5 → 264.4 |
| 2 / inference | 84.6 → 40.8 | 158.8 → 155.9 | 292.3 → 271.3 |
| 3 / inference | 83.8 → 42.1 | 158.8 → 156.0 | 296.8 → 276.1 |

Miles used more training GPU memory. Its synchronous schedule leaves inference idle during updates: inference utilization sampled **0% throughout both trainer windows**, whereas second-update training-node means were approximately 62% and 72%. These measurements identify idle phases, but GPU utilization is not FLOP efficiency. Both runs' sampled SM clocks remained 1,965 MHz.

Job 190 NCCL logs confirm IB transport and GPU Direct RDMA initialization. Eight high-speed local IB ports per node returned PMA counters; four 100-Gb/s ports per node timed out, so those ports are a coverage gap, not proven failed links. Across successfully monitored ports, no increases were recorded in the captured link/error/discard counters or RDMA hardware error counters. Transmit-wait counters did increase; without a calibrated interpretation they are not a congestion diagnosis.

Node-wide captured IB TX/RX totals were **297/297, 522/298, 576/801 and 801/801 GB** for nodes 0–3, respectively. These span the longer PMA collection window, beginning during initialization and ending around cleanup; they include non-training traffic and are **not** application bandwidth measurements or additive end-to-end payload totals.

## Backend differences and remaining limits

Prime-RL uses its FSDP trainer and vLLM; Miles uses Megatron and SGLang. Trainer EP=8 is retained. vLLM TP8/DP2/global EP16 maps to SGLang global TP16, attention DP2/TP8 and EP16. BF16 parameters are retained, but **Megatron accumulates/reduces gradients in FP32 rather than baseline BF16**. Optimizer state precision, recomputation, packing and kernels differ.

Miles disables only SGLang prefill CUDA graphs to avoid the observed failure, retains decode graphs, and selects `flashinfer_cutlass` for compatible expert-weight transfer. A job-local FP32 router-logit GEMM patch plus FP32 LM-head output matches the requested inference precision; five GPU shape checks and CUDA-graph replay passed. Patch source and hashes are retained, without modifying baseline software or the shared Miles checkout. Request routing is round-robin; SGLang automatic CPU NUMA pinning was unavailable in its inner container.

SGLang emitted Gloo connection-reset traces during final disposal, **after checkpoint save and final weight synchronization**. The driver and Slurm both exited successfully. These shutdown warnings are retained; this report does not claim a fully clean teardown or a diagnosed fix for those warnings.

## Checkpoint and evidence

Remote campaign root:

```
/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030
```

Final checkpoint: `runs/job-190/checkpoints/iter_0000001`. **The suffix is the zero-based rollout ID: this is after update two.** Approximately 69.42 GB across 16 distributed shard files plus metadata/debug material remains on shared storage. All 51,955 referenced byte ranges were checked, tensor chunk volumes matched metadata, and three small tensors were loaded on CPU, confirmed finite and hashed. This is **not a full distributed resume test**; optimizer/RNG state saving was disabled.

Local `evidence-job-190/job-190/` contains the downloaded non-checkpoint evidence: exact argv and resolved arguments, frozen launch/analysis source and hashes, training/harness/Ray/NCCL logs, all episodes and accepted token traces, TensorBoard, timelines, before/after snapshots, two-second host/GPU telemetry, ten-second PMA/RDMA counters and checkpoint verification. The large checkpoint was intentionally not downloaded.

Machine-readable results are in `comparison-results.json`; inputs and deviations in `comparison-spec.json`; final baseline preservation in `baseline-preservation-after-job-190.json`; and idle-process checks in `final-runtime-check.json`. The remote run's frozen prelaunch specification is retained as-is, alongside the final reports.
