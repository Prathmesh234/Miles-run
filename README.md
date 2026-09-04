# Miles: PPO versus IPO on Qwen3.6-35B-A3B

This repository contains reproducible launch code and compact evidence for the four-node Vultr B200 experiment. The PPO repeat uses the same base model, four Terminal-Lego tasks, batch settings, and **two actor updates** as Miles IPO job 190. Prime-RL IPO job 181 is the historical comparison.

**This is a functionality/performance smoke test, not evidence of improved held-out model quality.** PPO adds a learned critic and retains zero-variance reward groups, so accepted task mixtures and generated work differ. The matched recipe is synchronous, without MTP, and uses the prior cross-node SGLang EP16 engine. It is not the earlier proposed asynchronous EP8-per-node benchmark.

## Results and charts

- [Generated comparison report](results/REPORT.md), [training charts](results/training-comparison.svg), [PPO diagnostics](results/ppo-diagnostics.svg), and [infrastructure timelines](results/infrastructure-timeline.svg).
- [Consolidated metrics](results/comparison.json): all training scalars, task-level outcomes, timing phases, GPU/node/link distributions, and collector errors.
- [Sampled timelines](results/timeseries.csv), [provenance and archive index](results/provenance.json), and [failure/intervention log](results/interventions.json).
- [Original job-190 report](COMPARISON.md), [infrastructure inventory](INFRASTRUCTURE.md), [pinned baseline specification](comparison-spec.json), and [consolidated historical evidence](results/legacy-evidence.json).

Raw transcripts, TensorBoard files, NCCL/Ray logs, per-node collector output, and checkpoints remain in the checksummed local and `/shared` archives. They are not copied into hundreds of Git files. Previous published raw evidence remains in Git history at `0fa4636863b1f61b444a830f74980cb06d59c10e`; this change does not rewrite that history.

## Reproduce on this cluster

The cluster must already have the pinned model/conversion, baseline harness, task images, Miles source, and isolated Docker image cache described in [comparison-spec.json](comparison-spec.json). The launcher is cluster-specific, not a general installer. Supply your own kubeconfig. Submitting reserves all **32 GPUs**.

```bash
python3 launch_ppo.py --submit --kubeconfig ~/.kube/vultr-vke.yaml
```

Without `--submit`, this runs only read-only inventory checks. Each submission gets a unique campaign and job directory. The launcher checks GPU inventory, Slurm/Kubernetes counts, idle allocation, image availability, fabric ports, and free space. It runs precision, configuration, PPO math, token/mask transport, and lifecycle gates before optimizer work. It never prunes shared images, alters cluster configuration, or retries automatically.

[ppo-resident-broadcast.patch](ppo-resident-broadcast.patch) records two narrowly scoped Miles commits: retain the non-colocated PPO actor's parameter backup, restore it before broadcasting live tensors, and offload afterward. The original source and base weights stay read-only. The patched driver is generated inside each run after checking the base source SHA-256. Every attempted fix and failed launch is retained in the intervention log.

After a run has finished:

On the worker, use the pinned harness Python to run `verify_checkpoint.py RUN --base BASE_CONVERSION/release`, repeat with `--critic`, then run `finalize_evidence.py RUN`. The latter hashes all base/model/tokenizer and actor/critic checkpoint files **after** the measured window, verifies baseline source/config preservation, and records final scheduler state. Both scripts write only into the completed run directory and refuse to replace existing evidence.

```bash
python3 archive_run.py /shared/clustermax-campaigns/<campaign>/runs/job-<id> /path/to/archive/job-<id>
python3 extract_metrics.py /path/to/archive/job-<id>
python3 analyze_ppo.py \
  --ipo /path/to/job-190 \
  --ppo /path/to/archive/job-<id> \
  --baseline /path/to/original-comparison-archive \
  --baseline-gpu-csv /path/to/baseline-gpu-active.csv \
  --out results
python3 render_comparison.py results
```

`archive_run.py` verifies each downloaded file against a SHA-256 manifest and excludes large checkpoint shards. `extract_metrics.py` requires TensorBoard. Plotting uses Matplotlib 3.9.4 and NumPy 2.0.2; the full analysis package lock is recorded in provenance. The Markdown and charts are generated from the JSON/CSV, not manually entered measurements.

## Local checks

```bash
python3 -m unittest test_analysis -v
```

`test_ppo.py` runs against the pinned Miles/harness environments inside the allocation. It checks native policy/value loss gradients, masked GAE, raw rewards, zero-reward admission, sampled IDs/logprobs/tool masks, and the patched lifecycle. Checkpoints match the original weights-only policy; a full optimizer/RNG resume is not claimed.

## Environment compatibility

The online environment is the original pinned Verifiers **v1 API** Terminal-Lego harness in a separate process, connected through exact token-ID/logprob/mask transport to Miles. It is not Miles' stock Verifiers plugin, nor a forced Verifiers package upgrade. The baseline specification records the Verifiers and renderer commits; provenance records both environment package inventories. All 56 archived baseline trajectory fixtures exercise the transport contract, and the PPO tests cover native policy/value gradients and masked GAE.

Strict in-process Miles plus Verifiers 0.3.1 remains **blocked/unvalidated**: the documented Miles 0.2.x/OpenAI 2.6.1 dependency set conflicts with the newer Verifiers/Harbor requirements. This run does not claim to resolve that dependency conflict. No TB2.1 evaluation or Terminal-Bench training data is used in this matched Terminal-Lego smoke test.

No credentials, kubeconfigs, model weights, private ClusterMAX source, or Terminal-Bench hidden tests are published. [Third-party notices](THIRD_PARTY_NOTICES.md) describe retained source provenance.
