"""Run the two-step Miles comparison inside its isolated container."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from miles.utils.external_utils.model_args_utils import load_model_args

ROOT = Path("/campaign")
MODEL = "/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/model-fetch/models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0"


def model_args():
    args = shlex.split(load_model_args("qwen3.6-35B-A3B"))
    idx = args.index("--mtp-num-layers")
    del args[idx:idx+2]
    return args


def training_args(run):
    result = model_args() + [
        "--train-backend", "megatron", "--hf-checkpoint", MODEL,
        "--ref-load", str(ROOT/"converted-model"), "--save", str(run/"checkpoints"), "--save-interval", "2",
        "--no-save-optim", "--no-save-rng", "--no-load-optim", "--no-load-rng",
        "--actor-num-nodes", "2", "--actor-num-gpus-per-node", "8", "--num-gpus-per-node", "8",
        "--rollout-num-gpus", "16", "--rollout-num-gpus-per-engine", "16",
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--expert-model-parallel-size", "8", "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "selective", "--seq-length", "8192", "--max-position-embeddings", "8192",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", "8192",
        "--disable-rollout-global-dataset", "--rollout-function-path", "rollout_adapter.Rollout",
        "--num-rollout", "2", "--rollout-batch-size", "2", "--n-samples-per-prompt", "8",
        "--global-batch-size", "16", "--micro-batch-size", "1", "--num-steps-per-rollout", "1",
        "--rollout-max-response-len", "2048", "--rollout-temperature", "1", "--rollout-top-p", "1", "--rollout-top-k", "-1",
        "--advantage-estimator", "grpo", "--disable-grpo-std-normalization",
        "--custom-reward-post-process-path", "rollout_adapter.rewards_with_baseline_credit",
        "--loss-type", "custom_loss", "--custom-loss-function-path", "ipo_loss.loss", "--calculate-per-token-loss",
        "--use-rollout-logprobs", "--entropy-coef", "0", "--kl-coef", "0",
        "--optimizer", "adam", "--lr", "1e-6", "--min-lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0", "--adam-beta1", "0.9", "--adam-beta2", "0.999", "--adam-eps", "1e-8", "--clip-grad", "1",
        "--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d", "--use-precision-aware-optimizer",
        "--bf16", "--attention-dropout", "0", "--hidden-dropout", "0", "--attention-backend", "flash",
        "--sglang-dp-size", "2", "--sglang-enable-dp-attention", "--sglang-ep-size", "16",
        "--sglang-config", str(Path(__file__).parent/"sglang.yaml"),
        # This pinned SGLang build's breakable prefill graph fails its
        # reduce-scatter shape check with TP16 / attention DP2 / EP16.
        "--sglang-cuda-graph-backend-prefill", "disabled",
        # Miles' Blackwell Qwen3.5/3.6 recipe selects CUTLASS to support
        # Megatron-to-SGLang expert-weight updates in canonical BF16 layout.
        "--sglang-moe-runner-backend", "flashinfer_cutlass",
        "--sglang-enable-fp32-lm-head", "--sglang-router-policy", "round_robin",
        "--sglang-mem-fraction-static", "0.85", "--sglang-context-length", "8192", "--sglang-dtype", "bfloat16",
        "--sglang-max-running-requests", "16", "--sglang-server-concurrency", "16", "--sglang-enable-metrics",
        "--use-tensorboard", "--tb-project-name", str(run/"tensorboard"), "--tb-experiment-name", "miles",
        "--save-debug-rollout-data", str(run/"rollout-data-{rollout_id}.pt"),
        "--save-debug-event-data", str(run/"events"),
        "--log-interval", "1",
    ]
    if os.environ.get("MILES_ALGORITHM", "ipo") == "ppo":
        # Keep the job-190 workload. PPO replaces both IPO's objective and
        # group-centered credit, and trains a colocated value model.
        for flag in ["--custom-loss-function-path", "--custom-reward-post-process-path"]:
            index = result.index(flag)
            del result[index:index+2]
        result.remove("--disable-grpo-std-normalization")
        result[result.index("--advantage-estimator")+1] = "ppo"
        result[result.index("--loss-type")+1] = "policy_loss"
        result += ["--disable-rewards-normalization", "--offload-train",
                   "--critic-lr", "1e-5", "--num-critic-only-steps", "0",
                   "--eps-clip", "0.2", "--eps-clip-high", "0.2",
                   "--value-clip", "0.2", "--gamma", "1", "--lambd", "1",
                   "--normalize-advantages", "--observe-training-entropy",
                   "--update-weight-transfer-mode", "broadcast"]
    assert os.environ.get("MILES_ALGORITHM") == "ppo", "This launcher is async PPO only"
    result.remove("--use-rollout-logprobs")
    result += ["--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0",
               "--custom-tis-function-path", "async_metrics.tis",
               "--custom-megatron-init-path", "async_metrics.install",
               "--custom-megatron-before-train-step-hook-path", "async_metrics.before_train_step",
               "--update-weights-interval", "1"]
    return result


def main():
    mode = sys.argv[1]
    run = Path(os.environ.get("MILES_RUN_DIR", "/campaign/preflight"))
    if mode == "convert":
        argv = ["torchrun", "--standalone", "--nproc-per-node", "8", str(ROOT/"miles/tools/convert_hf_to_torch_dist.py"),
                *model_args(), "--hf-checkpoint", MODEL, "--save", str(ROOT/"converted-model"), "--bf16"]
    else:
        driver = Path(__file__).parent/"train_async_ppo.py" if os.environ.get("MILES_ALGORITHM") == "ppo" else ROOT/"miles/train.py"
        argv = [sys.executable, str(driver), *training_args(run)]
    (run/(mode+"-argv.json")).write_text(json.dumps(argv, indent=2)+"\n")
    if mode == "args":
        print(shlex.join(argv))
        return
    if mode == "validate":
        sys.argv = argv[1:]
        from miles.utils.arguments import parse_args
        args = parse_args()
        from miles.utils.arguments import validate_async_off_policy_correction
        validate_async_off_policy_correction(args)
        # The pinned validator disables this backup; disjoint PPO requires it.
        args.disable_param_buffers_cpu_backup = False
        assert args.use_tis and not args.use_rollout_logprobs and not args.skip_actor_forward_only
        (run/"validated-arguments.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print("Miles argument validation passed", flush=True)
        return
    print(shlex.join(argv), flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
