# Qwen3.6 low-precision preflight

Exact stock selection logic replayed on real headers. No weights read, quantized, replaced, or trained.

Packed expert tensors: **82**, **91.84%** of checkpoint tensor bytes.

| Stock mode | Selected tensors | Packed experts selected | GDN / vision selected | State |
|---|---:|---:|---:|---|
| fp8_blockwise | 499 | 0 | 180 / 110 | blocked_stock_recipe |
| mxfp8 | 442 | 0 | 150 / 83 | blocked_stock_recipe |

## Findings

- fp8_blockwise: The block converter unpacks exactly two dimensions but selects non-2D tensors.
- fp8_blockwise: Packed MoE expert names do not end in .weight and are left unquantized by the stock selector.
- mxfp8: The MXFP8 converter requires top-level num_hidden_layers; this model nests it under text_config.
- mxfp8: Packed MoE expert names do not end in .weight and are left unquantized by the stock selector.

## Next candidate

Qwen-aware expert unpacking, nested text configuration, explicit BF16 GDN/vision exceptions, and matching export/loader precision; not enabled until conversion, hot reload, logprob and gradient checks pass.

This failure applies to the stock conversion recipe, not to B200 low-precision capability. The BF16 run remains unchanged. Higher-precision master weights and optimizer state remain in the documented recipes; quantization does not imply proportional full-checkpoint shrinkage.

## Qualification

- Miles documents MXFP8 on B200 as beta, including BF16-trainer/MXFP8-rollout and matching MXFP8 forward recipes. Its tested models do not list Qwen3.6.
- Qwen-aware candidate selects 249 tensors / 66,892,857,344 input bytes, including all 82 packed expert tensors. GDN, vision, routers, norms, embeddings and MTP glue remain BF16. Full conversion and serialized audit passed in job148; no optimizer change.
- Installed SGLang source captured directly from the pinned SquashFS: Qwen loader supports individual expert projections and FP8 scales. Live loader/activation remains untested.
- Bounded B200 probe pre-registers maximum relative L2 reconstruction error 0.06 and byte-exact agreement between packed conversion and Miles per-expert live export. Tests real main/MTP slices and zero control; not a gradient, speed or quality claim.
- Prior source inspection failures retained: v1 timed out pulling an uncached Docker image; v2 assumed a nonexistent standalone mxfp8.py module. v3 discovered and captured installed sources without running Docker.
- Training quantization stays disabled until full load, MTP, broadcast activation, fixed-token logprob/gradient and end-to-end telemetry gates pass. NVFP4 is not enabled.
- The older Miles Qwen3 MXFP8 launcher comment disallows EP for its cutlass path. The newer pinned SGLang flashinfer_trtllm_routed implementation accepts local expert offset/count for MXFP8. Preserve EP8 and qualify that backend rather than silently changing topology.

## Reproduce the bounded GPU probe

```sh
.venv-launch-tests/bin/python scripts/stage_model_conversion.py \
  --run-dir runs/vultr-b200-slurm/20260902-172037-a3b210 \
  --kubeconfig /path/to/kubeconfig --format mxfp8-probe --attempt 1
```

## Provenance

- header_evidence: `tests/02-quantization-model-header-preflight-v1/model-header-inventory.json`
- image_digest: `sha256:59a11219eae0defc6594ec678fafe4e897c16904263223f79968cd3e0209a502`
- installed_loader_sources: `tests/02-qwen-quantization-sglang-source-v3/manifest.json`
- kernel_sha256: `47b13f3dc1d5e144090c45ce253a50de443f47b9a972cc97580a21975a07b8fd`
- miles_base_sha: `0709889b2848f293b5575d50aa3340fa4de5a20d`
- model_revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- run_id: `20260902-172037-a3b210`
- stock_audit: `tests/02-quantization-recipe-audit-v1/result.json`

[Pinned Miles low-precision documentation](https://github.com/radixark/miles/blob/0709889b2848f293b5575d50aa3340fa4de5a20d/docs/advanced/low-precision.md)

## Executed B200 kernel probe

Slurm 147: COMPLETED, exit 0; kernel/export and allocation audit passed. 24 exported weight/scale tensors matched byte-for-byte; maximum relative L2 error 2.6639% (limit 6%).

Telemetry: 6 finalized streams, 0 collector errors; maximum sample gap 2.163s.

Representative real expert slices plus zero control on GPU0 of an 8-GPU allocation. Not full model conversion, SGLang activation, gradient equivalence, throughput or quality validation.

Raw evidence: `tests/02-qwen-mxfp8-probe-result-audit-v1/result.json`.

## Full converted candidate

Slurm 148: COMPLETED, exit 0; serialized audit passed, runtime unqualified. Conversion took 105.07s.

Tensor payload: **39.50 GB**, from 71.90 GB. All 64,106 serialized tensors passed names/shapes/dtypes/coverage checks. 796 higher-precision tensors and tokenizer metadata are byte-exact.

37 file checksums verified; maximum quantization relative L2 2.6771%. 6 finalized infrastructure streams, 0 collector errors.

This reduces inference-weight storage, not the full optimizer checkpoint. No training speed or held-out quality claim.

Checkpoint: `/shared/posttrainingx/runs/vultr-b200-slurm/20260902-172037-a3b210/models/qwen3.6-35b-a3b-mxfp8-v1`. Audit: `tests/02-mxfp8-serialized-checkpoint-audit-v1/result.json`.
