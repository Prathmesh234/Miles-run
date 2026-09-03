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
  "job_id": 153,
  "local_tests": "60 passed; includes exact pinned binding import on CPU",
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
  "source_sha": "e921c7d661261706d8de3479a2200e3759af5ffb"
}
```
