# EP8 trainer execution gate

This probe loads the qualified PP8 conversion into the actual TP1/EP8/PP1,
CP1/ETP1 model topology on one eight-GPU node. It uses pinned campaign Miles
`346946ae870be97e9cb6f4e8b7214c7fcf66c041` and the existing digest-pinned GPU
image. No package installation or runtime monkey-patch is performed.

The worker uses Miles' model provider and production packed batching, Megatron
DDP and its forward/backward schedule. It retains the recipe's flex dispatcher,
full uniform recomputation, BF16 model, FP32 gradient accumulation, zero dropout,
one MTP layer, 0.2 MTP loss scale and detached MTP shared heads. The resolved
topology and model configuration must match before checkpoint loading.

Each rank consumes 128 fixed diagnostic text tokens. The main loss is ordinary
next-token cross-entropy, with the native MTP auxiliary loss attached by the
model. This is **not GRPO**, a task trajectory, or quality evidence. The probe
does not validate sampled logprobs against the serving engine. It prepares that
next comparison by retaining token IDs and teacher-forced logprobs.

The probe constructs **no optimizer or scheduler**, performs no optimizer step,
and loads no optimizer/RNG state from the release checkpoint. Every parameter is
hashed before and after execution; any change fails the gate. Gradient buffers
must be finite and include nonzero main-model and MTP gradients. Per-rank buffer
norms are diagnostic only, not a deduplicated global training gradient norm.

Inputs are rehashed and mounted read-only at their canonical input paths. Fresh
physical/CUDA UUID and Slurm inventory checks bracket the allocation. Native
GPU/NVLink/IB/CPU and host Lustre collectors run throughout. Raw per-rank records,
torchrun stdout/stderr and NCCL logs stay under the run directory.

After committing the code, run from the project root:

```sh
.venv-launch-tests/bin/python scripts/stage_trainer_probe.py \
  --run-dir runs/vultr-b200-slurm/20260902-172037-a3b210 \
  --kubeconfig /Users/prathmeshbhatt/.kube/vke-config --attempt 1
```

The explicit Slurm partition is `gpu-nodes`; one whole node is reserved for at
most 25 minutes. The worker execution cap is 900 seconds and its enclosing
container cap is 1050 seconds. A free-space reserve of 128 GiB is enforced.
`control/trainer-probe-v1.stop` requests owned-process shutdown. New attempts
must use new versioned paths, preserving any failure and its artifacts.

A passing result does not authorize GRPO training by itself: local task/grader
isolation, true n-sample grouping and policy loss, serving-logprob agreement,
weight activation, full telemetry and checkpoint/resume remain separate gates.

## Executed result

Job **120** completed with exit zero in **5m51s**. All eight ranks passed the
native load/forward/backward contract. Each retained 3,191 identical before/after
parameter hashes and finite main/MTP gradient buffers. Megatron resolved
sequence parallelism to false at TP1, while preserving TP1/EP8/PP1/CP1/ETP1.
No optimizer or scheduler was constructed; zero optimizer steps occurred.

An independent finalized-record audit checked parameter/gradient inventories,
counts, MTP classification, packed token IDs and diagnostic-loss/logprob
consistency. This does **not** independently establish that each loaded EP8
shard contains the correct checkpoint values; that numerical comparison remains
open, along with serving-logprob equivalence.

All six telemetry streams finalized without collector errors. Native sampling
had a maximum 2.58-second gap. The generated report has 554 metric distributions
and 13,838 plotted-source records. The first forward/backward includes startup
and compilation effects; it is not a steady-state throughput measurement.

Reproducible audit and reporting entrypoints (each refuses to reuse its phase):

```sh
.venv-launch-tests/bin/python scripts/audit_trainer_probe.py \
  --run-dir runs/vultr-b200-slurm/20260902-172037-a3b210 \
  --kubeconfig /Users/prathmeshbhatt/.kube/vke-config --attempt 1 --job-id 120
.venv-analysis/bin/python scripts/report_trainer_probe.py \
  --run-dir runs/vultr-b200-slurm/20260902-172037-a3b210 \
  --kubeconfig /Users/prathmeshbhatt/.kube/vke-config
```

The report uses the separately pinned Python 3.12.11 / Matplotlib 3.10.6 analysis
environment. An earlier plotting import in the launcher-test environment failed
and remains recorded; no packages were added to the training image.

Outputs: `reports/trainer-probe-v1.json`, `.md` and `.png` inside the run bundle.
Raw per-rank artifacts and torchrun logs remain in
`tests/02-trainer-probe-child-v1/` on shared storage.
