# GPU telemetry qualification

**Training telemetry remains unqualified; individual control results are below.**

Collector qualification around node-local all-reduce only. Actual training/checkpoint load, continuous DCGM and full required telemetry remain unqualified.

| Job | Workload | Maximum GPU sample gap (s) | Result |
|---|---|---:|---|
| 153 | Node-local all-reduce | 1.522 | passed |
| 156 | context-teardown | 2.142 | passed |
| 157 | nccl-context-teardown | 2.004 | passed |
| 158 | fragmented-nccl-teardown: 4096x16MiB allocations per GPU, 64GiB/GPU plus EP8 NCCL, normal process exit; no training workload | 2.830 | passed |

Job 154 had trainer-node gaps of 15.224 s / 15.479 s.

A control result does not repair the failed training gate or, by itself, prove a hardware cause.

## Contract and next test

Read-only NVML; no resets. Parent rejects collector errors, missing/stale host-local heartbeat (>12s), wrong node/job identity. Unique node-owned stop markers; final CLI parity and Lustre finalization required.

Exercise the actual trainer allocation/communicator lifecycle with per-API timing; do not relax the 12s deadline or infer hardware fault.

**planned; no GPU outcome yet — pinned-host-nccl-teardown**

Job161 enables --use-kl-loss even with coefficient0; Miles allocates pinned actor/reference backups. Test whether pinned host-buffer release can reproduce NVML blocking absent from prior GPU-only controls. This is a hypothesis, not an attribution.

Four whole nodes, eight ranks per node. Per rank:64GiB GPU memory in4096x16MiB chunks plus24GiB pinned host memory in3072x8MiB chunks and node-local EP8 NCCL. Require96GiB free HBM and320GiB host/cgroup-available memory. Five-minute allocation; telemetry includes ordinary exit plus15s afterward. No model/checkpoint changes or optimizer steps.

Preserve any failed gate. A pass only rules out this bounded pattern; a stall motivates a separate explicit cleanup control. Keep the12s deadline and all teardown samples. Full trainer/Ray lifecycle remains a separate gate.

## Evidence

- Initial control: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v2/result.json`.
- Job 156: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v3/result.json`; SHA256 `5a1e0635cf662787bd209b69507cfdda9dead9377d14f998f6a6b42ae32849f0`.
- Job 157: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v4/result.json`; SHA256 `9c63d2ab0e58985092e1b8e0898e625f74702d0b02f3825483ecb2e34a558009`.
- Job 158: `runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v5/result.json`; SHA256 `331944125eacd7b0867454332bd226744c6935a8ad2b0285ac735449d312330a`.

Full per-node results, source pins and prior failures remain in `telemetry-qualification.json` and the linked raw bundles.
