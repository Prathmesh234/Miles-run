# resume-compatibility

Status: partial


## Metadata

```json
{
  "components": [
    {
      "component": "Model, optimizer and LR scheduler",
      "source": "miles/backends/megatron_utils/model.py:save",
      "state": "Native save_checkpoint receives all three objects; no-save-optim defaults false. Runtime fidelity unvalidated."
    },
    {
      "component": "RNG",
      "source": "miles/backends/megatron_utils/model.py",
      "state": "Megatron owns saving/loading. Must verify the actual training checkpoint and next-step equivalence; conversion-checkpoint topology warnings are not a successful resume test."
    },
    {
      "component": "Consumed data position and sample identities",
      "source": "miles/rollout/data_source.py:RolloutDataSource.save/load",
      "state": "CPU save/load preserved sample_offset, epoch_id, sample_group_index, sample_index and metadata exactly."
    },
    {
      "component": "Recycled prompt buffer",
      "source": "miles/rollout/data_source.py:RolloutDataSourceWithBuffer",
      "state": "Not persisted: CPU fixture had one buffered group before save and zero after load."
    },
    {
      "component": "Fully asynchronous completed-group queue and accounting",
      "source": "miles/rollout/fully_async_data_buffer.py:DefaultDataBuffer",
      "state": "No checkpoint state API on built-in DataBuffer; buffer and window counters initialize empty/zero. Not restart-qualified."
    },
    {
      "component": "Broadcast policy version",
      "source": "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py:UpdateWeightFromDistributed.__init__",
      "state": "Updater initializes at zero; no corresponding checkpoint restore was found in the inspected broadcast path. Must persist it with activation state."
    },
    {
      "component": "In-flight environment and scheduler state",
      "source": "miles/rollout/fully_async_rollout.py:FullyAsyncRolloutFn",
      "state": "Live asyncio tasks and sandbox processes are not serialized. A qualified checkpoint boundary must drain or explicitly account for cancellation and re-submission."
    }
  ],
  "cpu_probe": {
    "path": "runs/vultr-b200-slurm/20260902-172037-a3b210/tests/02-resume-surface-probe-v1/result.json",
    "result": {
      "buffer_groups_after": 0,
      "buffer_groups_before": 1,
      "checkpoint_path": "/run-artifacts/provenance/resume-surface-probe-v1/rollout/global_dataset_state_dict_2.pt",
      "conclusion": "Buffered prompts are not persisted by inherited data-source checkpoint methods. Full async resume remains unqualified.",
      "cuda_device_count": 0,
      "fields_restored": [
        "sample_offset",
        "epoch_id",
        "sample_group_index",
        "sample_index",
        "metadata"
      ],
      "scope": "Actual pinned Miles data-source save/load methods on a CPU fixture, not model/optimizer/resume validation."
    },
    "sha256": "8ca700ccfb094d2bd85aaff1a570cf3f8af0664243e1e76cf2e4f6edd65a7771"
  },
  "miles_revision": "1181014c3c78ed290d4e53fbc33ef8d4cdb1949a",
  "required_before_resume_claim": [
    "Verify model, optimizer, scheduler and RNG state in a completed checkpoint, including load/next-update comparison on identical frozen samples.",
    "Preserve and restore recycled prompts, completed groups, version counters and lifetime accounting, or explicitly quiesce and prove none remain.",
    "Atomically publish a complete state manifest only after all checkpoint components are durable and hashed.",
    "Demonstrate exact submitted/completed/trained/retried/stale/failed accounting across a restart; never silently reset policy version or the data cursor."
  ],
  "scope": "Source inspection and a CPU data-source save/load test. No end-to-end model/optimizer resume is claimed."
}
```
