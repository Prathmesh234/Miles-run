# persistent-nvml-qualification

Status: ok


## Metadata

```json
{
  "counter_documentation": "https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueEnums.html",
  "evidence": "runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v2/result.json",
  "failed_attempt": {
    "classification": "PostTrainingX implementation",
    "failure": "Pinned module loader omitted sys.modules registration; stopped before load or optimizer.",
    "fix_commit": "e921c7d661261706d8de3479a2200e3759af5ffb",
    "job_id": 152
  },
  "failed_training_validation": {
    "gaps_s": [
      15.223752644029446,
      15.479490234982222
    ],
    "job_id": 154,
    "next_step": "Exercise the actual trainer allocation/communicator lifecycle with per-API timing; do not relax the 12s deadline or infer hardware fault.",
    "scope": "Neither memory-only nor EP8 NCCL context teardown control reproduces this training-runtime stall. Original failed training gate remains failed."
  },
  "job_id": 153,
  "local_tests": "64 passed; scoped health, API timing and publication-gate regressions included",
  "nodes": [
    {
      "cli_bracket_passed": true,
      "collector_errors": 0,
      "hostname": "gpu-nodes-0",
      "max_gap_s": 1.2116344660753384,
      "nvlink_counter_identities": 288,
      "ticks": 23
    },
    {
      "cli_bracket_passed": true,
      "collector_errors": 0,
      "hostname": "gpu-nodes-1",
      "max_gap_s": 1.522265745094046,
      "nvlink_counter_identities": 288,
      "ticks": 26
    },
    {
      "cli_bracket_passed": true,
      "collector_errors": 0,
      "hostname": "gpu-nodes-2",
      "max_gap_s": 1.1708550699986517,
      "nvlink_counter_identities": 288,
      "ticks": 23
    },
    {
      "cli_bracket_passed": true,
      "collector_errors": 0,
      "hostname": "gpu-nodes-3",
      "max_gap_s": 1.2730063030030578,
      "nvlink_counter_identities": 288,
      "ticks": 26
    }
  ],
  "raw_evidence_root": "/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210",
  "runtime_contract": "Read-only NVML; no resets. Parent rejects collector errors, missing/stale host-local heartbeat (>12s), wrong node/job identity. Unique node-owned stop markers; final CLI parity and Lustre finalization required.",
  "scope": "Collector qualification around node-local all-reduce only. Actual training/checkpoint load, continuous DCGM and full required telemetry remain unqualified.",
  "source_sha": "e921c7d661261706d8de3479a2200e3759af5ffb",
  "subsequent_controls": [
    {
      "evidence": "runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v3/result.json",
      "findings": [],
      "job_id": 156,
      "nodes": [
        {
          "hostname": "gpu-nodes-0",
          "max_gpu_sample_gap_s": 2.1190556900110096,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-1",
          "max_gpu_sample_gap_s": 2.141821142984554,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-2",
          "max_gpu_sample_gap_s": 1.9388169770099921,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-3",
          "max_gpu_sample_gap_s": 1.9495135570177808,
          "status": "ok"
        }
      ],
      "scope": "context-teardown",
      "sha256": "5a1e0635cf662787bd209b69507cfdda9dead9377d14f998f6a6b42ae32849f0"
    },
    {
      "evidence": "runs/vultr-b200-slurm/20260902-172037-a3b210/tests/01-nvml-result-audit-v4/result.json",
      "findings": [],
      "job_id": 157,
      "nodes": [
        {
          "hostname": "gpu-nodes-0",
          "max_gpu_sample_gap_s": 1.646085308981128,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-1",
          "max_gpu_sample_gap_s": 2.0035865120589733,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-2",
          "max_gpu_sample_gap_s": 1.9001114400016377,
          "status": "ok"
        },
        {
          "hostname": "gpu-nodes-3",
          "max_gpu_sample_gap_s": 1.8425092799589038,
          "status": "ok"
        }
      ],
      "scope": "nccl-context-teardown",
      "sha256": "9c63d2ab0e58985092e1b8e0898e625f74702d0b02f3825483ecb2e34a558009"
    }
  ]
}
```
