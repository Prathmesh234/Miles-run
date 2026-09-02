# Converted checkpoint qualification

The pinned HF weights and the converted MTP checkpoint are inputs, never output
destinations. The parity stage rehashes both sets of input files and compares all
text and MTP weights exactly, including dtype, shape and byte hashes. Vision
weights are explicitly excluded because this campaign is text-only.

The pinned Megatron revision stores common state inside DCP. The older Miles
standalone reverse-conversion script expects `common.pt`. This checker uses
Megatron's own current common-state loader, Torch DCP's flat tensor loader, and
the **unmodified Miles parameter-expansion and weight-conversion functions**.
It does not patch the model or force a package upgrade. Per-expert outputs are
compared against the matching slices of Qwen's fused expert reference tensors;
coverage must include every expert exactly once, including MTP experts.

From the repository root, after committing code and locks:

```sh
.venv-launch-tests/bin/python scripts/stage_checkpoint_parity.py \
  --run-dir runs/vultr-b200-slurm/20260902-172037-a3b210 \
  --kubeconfig /Users/prathmeshbhatt/.kube/vke-config \
  --attempt 1
```

This reserves one whole node in `gpu-nodes` for at most 30 minutes. The tensor
comparison is CPU-only inside the pinned GPU image; GPU UUID inventory is still
reconciled at both ends. Native GPU, NVLink, IB, CPU and filesystem collectors,
plus the run-owned host Lustre collector, remain active. The child is capped at
25 minutes with a 128 GiB free-space reserve and owned-process cleanup. It may
be stopped using the run-owned `control/checkpoint-parity-v1.stop` marker.

Each attempt uses new phase, code and runtime directories. A failed attempt
must be retained, diagnosed and followed by a separately committed correction
before a new numbered attempt. Submission is not successful execution.

Even a fully passing result does **not** prove EP8 trainer resharding, forward
logits, gradients, optimizer state, checkpoint/resume fidelity, environment
isolation, or a quality improvement. Those remain separate gates.
