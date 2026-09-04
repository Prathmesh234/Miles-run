# Run 4 — longer synchronous PPO (preparation)

**Not submitted.** The requested “ADB” model needs clarification; 80B is the
proposed interpretation, not a confirmed selection. Checkpoint cadence also
needs confirmation before launch. No model weights have been downloaded.

The target is 10 actor optimizer updates and 10 critic optimizer updates:
10 rollout batches × 1 update per role per batch, with global batch size 16.
Start from the larger model's original weights, not a trained checkpoint.
Use job 196 as the synchronous reference and preserve its workload settings.

## Important: startup scripts

After this run produces evidence, use the shared [RL renderer](../comparison-infrastructure/rl_charts.py) with `--folder run4-ppo-long --log-root <repository-relative-log-directory> --rollout-root <repository-relative-rollout-directory> --job-id <actual-job-id> --label 'Long synchronous PPO'`. All five arguments are required. It will write reward, task-outcome, policy, rollout and critic charts under `charts/rl/`, plus compact metrics and source hashes. No RL curves exist yet because this job has not run. The original workload uses binary solved rewards and eight-sample groups; different workloads require adapting and validating the recipe.

The `scripts/` directory is the authoritative place for startup and validation
code. `long_run_config.py` provides the shared contract and argument extension.
`build_bundle.py` connects that contract to the trainer, rollout bridge, and
coordinator. It verifies prior source hashes and produces a frozen runtime
bundle plus a provenance manifest. It is **not a submission script** and
rejects unresolved model and save options. Run 5 shares this code rather than
duplicating it. Generated source copies belong under ignored `.work/` locally
and on the cluster, not in Git.

Run the preparation checks from the repository root:

```sh
python3 -m unittest discover -s run4-ppo-long/scripts -p test_long_run.py -v
```

The ten tests cover the ten-update boundary, immutable model revisions,
preservation of unrelated training arguments, CPU offload requirements, the
native async driver's ten-step schedule with fake actors, and the generated
runtime scripts for both modes. The coordinator rejects a converted checkpoint
whose model ID/revision differs from the requested model before creating a run.
These tests do not establish GPU/model compatibility or prove ten actual
optimizer updates.

After resolving `config/run.json`, build a new source bundle from the repository
root (choose a fresh destination; existing bundles are never overwritten):

```sh
python3 run4-ppo-long/scripts/build_bundle.py \
  run4-ppo-long/config/run.json run4-ppo-long/.work/source
```

Before submission, stage and verify the selected model and its distinct
Megatron conversion. The conversion's `conversion-complete.json` must identify
`model_id` and immutable `model_revision`, and the `release/` directory must
exist. Run model/tokenizer/renderer and precision checks in the pinned image.
The old 35B conversion is not a fallback. Keep both model inputs read-only in
the training containers.

## CPU offloading

The prior runs already enabled CPU Adam. Retain
`--optimizer-cpu-offload`, `--overlap-cpu-optimizer-d2h-h2d`, and
`--use-precision-aware-optimizer`, and make the phase-boundary target explicit
with `--offload-train --offload-train-target cpu`.

Miles' [pinned Qwen3-Next 80B recipe](https://github.com/radixark/miles/blob/70b89e11770fc9bac984e22cfff89c51cca44203/docs/models/qwen/qwen3-next.md)
documents CPU Adam for that model. Its stock GSPO, colocation, parallelism and
speculative-decoding settings must **not** silently replace this experiment's
PPO task configuration. The model's tokenizer, renderer, architecture recipe,
weight conversion and router precision need validation after model selection.

Miles distinguishes [paused-model offload from optimizer state placement](https://github.com/radixark/miles/blob/70b89e11770fc9bac984e22cfff89c51cca44203/docs/advanced/disk-offload.md).
CPU optimizer referenced bytes, process RSS, HBM usage and offload wall time
will be reported separately; none alone measures PCIe bandwidth. CPU offload
is requested here, not NVMe streaming or disk-backed model offload.

## Storage and publication

At preparation time all four nodes were idle and `/shared` had approximately
2,643.5 GiB free. Two 80B runs saving actor and critic weights at updates
2/4/6/8/10 would require approximately 3.2 TB of BF16 checkpoints alone,
before model staging/conversion and safety headroom. All old runs must remain.
The proposed alternative is final actor/critic checkpoints at update 10 only,
with logs and metrics for every update. This is not yet approved.

Keep the existing per-run `scripts/`, `config/`, `logs/`, and `charts/` layout.
Publish only reviewed scripts/configuration, consolidated metrics, essential
logs, charts and verification receipts. Retain full archives and weights on
the cluster; do not add duplicated source trees or per-process debug dumps to
Git. Final reports must distinguish preparation tests from executed results.
