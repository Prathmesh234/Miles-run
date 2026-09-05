# Miles experiments and CollectiveX on Vultr B200

**[Open the CollectiveX HTML dashboard](collectivex/index.html)** · [CollectiveX results and methods](collectivex/README.md) · [Repository map and cleanup](REPOSITORY.md)

The CollectiveX dashboard is a self-contained HTML file: download it and open
it locally. It provides measured dispatch/combine/round-trip latency, payload
rates, node comparisons, infrastructure traces, and CSV/JSON/SVG exports.
No browser extension, server, GPU allocation or internet connection is required.

## CollectiveX: completed EP8 characterization

Jobs **221–224**, four nodes / 32 B200 GPUs, **112/112 measured points passed**.
At 8,192 tokens/GPU, FP8 dispatch reduced chained pair-period latency by about
22.8% versus BF16; combine remains BF16. This tests four separate NVLink
islands, not EP32/inter-node RDMA or model quality. Full methodology, pins,
telemetry caveats and evidence-archive checksum are in [collectivex/](collectivex/README.md).

## Historical Miles runs

These are **two-update functionality/performance experiments**, not a held-out
quality hill climb or a controlled algorithm speedup claim. All use
Qwen3.6-35B-A3B; generated trajectories and some runtime choices differ.

- [Run 1: GRPO-style credit / custom IPO](run1-grpo/README.md), job 190.
- [Run 2: synchronous PPO](run2-ppo/README.md), job 196.
- [Run 3: native one-batch-ahead async PPO + TIS](run3-async-ppo/README.md), job 197.
- [Run guide and RL galleries](RUNS.md): rewards, optimizer diagnostics, critic
  learning, rollout behavior, task outcomes and TIS.
- [Three-run infrastructure comparison](comparison-infrastructure/README.md):
  overlap, weight publication, GPU/IB/NVLink utilization and CPU offload.
- Original job-190 reports: [comparison](COMPARISON.md),
  [infrastructure](INFRASTRUCTURE.md), [final status](STATUS.md).

Run 4/5 folders contain **unsubmitted preparation**, not completed long runs.
Cancelled job 195 is retained separately and excluded from valid comparisons.
Checkpoints were weights-only; a full optimizer/RNG resume test was not performed.

## Keep this repository compact

The cleanup removed 2,170 redundant files, including 222 duplicate script
copies. Canonical per-run startup modules and every distinct historical
content hash were preserved during deduplication. See [REPOSITORY.md](REPOSITORY.md)
and [the machine-readable deletion ledger](repository-cleanup.json).
Original files remain in Git history; historical manifests are not rewritten.

Do not commit duplicated source snapshots, empty process logs, checkpoint/model
weights, Docker images, kubeconfigs, credentials or private Slack transcripts.
Keep full archives separately; publish compact measured data and readable reports.

**Startup scripts are environment-specific historical sources, not a portable
one-command installer.** Inspect each run's scripts README before reusing them.
Submitting the old workload can reserve 32 GPUs; this repository cleanup and
CollectiveX dashboard publication did not launch another training job.

[Third-party notices](THIRD_PARTY_NOTICES.md)
