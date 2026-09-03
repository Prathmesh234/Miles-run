# Third-party source retained as experiment evidence

The `qwen2_moe.before.py` and `qwen2_moe.after.py` files in the four `evidence-job-190/job-190/infra/precision-patch-gpu-nodes-*/` directories are SGLang source snapshots from revision `c16b821ef3177a688a073c173b44c0ce48b5bf3e`, adapted upstream from vLLM. Their original SPDX and copyright notices are preserved. These files are licensed under Apache License 2.0; a copy is included at [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The `.before.py` files are the unmodified sources. The `.after.py` files were modified for job 190 by replacing the Qwen MoE router's `ReplicatedLinear` construction with the experiment's `FP32RouterLinear` subclass. This keeps BF16 parameter storage while computing FP32 router-logit outputs. The exact patch and validation are in `install_precision_patch.py` and `sglang_precision.py`; per-worker manifests retain the before/after hashes. The snapshot bytes were not changed during publication.

Model weights, full upstream Miles/SGLang repositories and the baseline source tree are not redistributed here. Their provenance is recorded in `comparison-spec.json` and the software inventories. Task records are retained as experiment inputs/outputs; this repository does not claim ownership of third-party task content or change its original licensing.
