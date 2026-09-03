# GPU telemetry qualification

**Training telemetry remains unqualified; individual control results are below.**

Collector qualification around node-local all-reduce only. Actual training/checkpoint load, continuous DCGM and full required telemetry remain unqualified.

| Job | Workload | Maximum GPU sample gap (s) | Result |
|---|---|---:|---|
| 153 | Node-local all-reduce | 1.522 | passed |
| 156 | context-teardown | 2.142 | passed |
| 157 | nccl-context-teardown | 2.004 | passed |
| 158 | fragmented-nccl-teardown: 4096x16MiB allocations per GPU, 64GiB/GPU plus EP8 NCCL, normal process exit; no training workload | 2.830 | passed |
| 162 | pinned host:64GiB/GPU plus24GiB pinned host/rank and EP8 NCCL; ordinary exit | 10.382 | failed |
| 163 | same pinned-host control; explicit24GiB/rank host-cache release before ordinary exit | 2.101 | passed |

Job 154 had trainer-node gaps of 15.224 s / 15.479 s.

A control result does not repair the failed training gate or, by itself, prove a hardware cause.

## Contract and next test

Read-only NVML; no resets. Parent rejects collector errors, missing/stale host-local heartbeat (>12s), wrong node/job identity. Unique node-owned stop markers; final CLI parity and Lustre finalization required.

Smallest next change: an opt-in normal-exit Miles cleanup that releases actor/reference pinned backups after all training, saves and broadcasts finish. Keep CUDA/NCCL teardown and telemetry collection intact; validate in the full trainer before enabling the long run.

## Pinned-host release control

One matched four-node control per condition. Job162 failed the3s sampling criterion (all nodes7.48-10.38s);163 passed (maximum2.10s). All32 ranks proved24GiB of pinned-host active and allocated bytes released before exit. This supports the resource-lifecycle hypothesis, but does not establish a unique hardware cause or qualify full Miles training.

| Node | Ordinary-exit gap (s) | Explicit-release gap (s) | Host release min / max (s) | Verified ranks |
|---|---:|---:|---:|---:|
| gpu-nodes-0 | 9.053 | 1.951 | 3.252 / 3.561 | 8 |
| gpu-nodes-1 | 9.049 | 1.996 | 3.126 / 3.510 | 8 |
| gpu-nodes-2 | 10.382 | 2.101 | 3.154 / 3.838 | 8 |
| gpu-nodes-3 | 7.481 | 1.818 | 3.151 / 4.719 | 8 |

Smallest next change: an opt-in normal-exit Miles cleanup that releases actor/reference pinned backups after all training, saves and broadcasts finish. Keep CUDA/NCCL teardown and telemetry collection intact; validate in the full trainer before enabling the long run.

## Evidence

- Initial control: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v2/result.json`.
- Job 156: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v3/result.json`; SHA256 `5a1e0635cf662787bd209b69507cfdda9dead9377d14f998f6a6b42ae32849f0`.
- Job 157: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v4/result.json`; SHA256 `9c63d2ab0e58985092e1b8e0898e625f74702d0b02f3825483ecb2e34a558009`.
- Job 158: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v5/result.json`; SHA256 `331944125eacd7b0867454332bd226744c6935a8ad2b0285ac735449d312330a`.
- Job 162: `/Users/prathmeshbhatt/Desktop/PostTrainingX/runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v6/result.json`; SHA256 `ada770b1c6dd7cab083e644e9214814addc2eb909380acec73bd68e2ef6e88bf`.
- Job 163: `/Users/prathmeshbhatt/Desktop/PostTrainingX/runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v7/result.json`; SHA256 `35eb443c06c6f6263383218580d26ad31b183bf06c2f310a5d031b08054a6bd9`.

Full per-node results, source pins and prior failures remain in `telemetry-qualification.json` and the linked raw bundles.
