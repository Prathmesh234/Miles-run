"""Shared configuration contract for the longer matched PPO runs.

Imported by both run folders. This module neither submits nor downloads anything.
Unresolved model/checkpoint decisions fail closed, before resource allocation.
"""
import copy
import json
from pathlib import Path


def load(path):
    config = json.loads(Path(path).read_text())
    validate(config)
    return config


def validate(config):
    if config.get("mode") not in ("ppo", "async_ppo_tis"):
        raise ValueError("Select synchronous PPO or one-batch-ahead async PPO + TIS")
    if (config.get("num_rollout"), config.get("num_steps_per_rollout"),
            config.get("optimizer_updates_per_role")) != (10, 1, 10):
        raise ValueError("Exactly 10 rollouts, one optimizer update per rollout and role")
    if config.get("global_batch_size") != 16:
        raise ValueError("The matched global batch size must remain 16")
    required = ("model_id", "model_revision", "model_path", "converted_model_path",
                "model_args_recipe", "renderer", "save_interval")
    unresolved = [key for key in required if not config.get(key)]
    if unresolved:
        raise ValueError("Unresolved run configuration: " + ", ".join(unresolved))
    revision = config["model_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin model_revision to an immutable 40-character commit SHA")
    for key in ("model_path", "converted_model_path"):
        if not Path(config[key]).is_absolute() or ".." in Path(config[key]).parts:
            raise ValueError(key + " must be an absolute normalized path")
    if config["model_path"] == config["converted_model_path"]:
        raise ValueError("HF input and Megatron conversion must be separate directories")
    interval = config["save_interval"]
    if type(interval) is not int or interval not in (2, 10):
        raise ValueError("Checkpoint cadence must be explicitly chosen: 2 or 10")
    offload = config.get("cpu_offload", {})
    for key in ("optimizer_cpu_offload", "overlap_cpu_optimizer_d2h_h2d",
                "use_precision_aware_optimizer", "offload_train"):
        if offload.get(key) is not True:
            raise ValueError("Required CPU offload option missing: " + key)
    if offload.get("offload_train_target") != "cpu":
        raise ValueError("CPU offload is requested, not disk offload")
    if config["mode"] == "async_ppo_tis" and config.get("tis") != {
            "enabled": True, "clip_low": 0.0, "clip_high": 2.0}:
        raise ValueError("Preserve the reference run's detached TIS clipped to [0, 2]")


def load_runtime():
    """The same frozen config is beside this module in every run process."""
    return load(Path(__file__).with_name("run-config.json"))


def extend_training_args(base_args, config):
    """Change only duration, model input and explicitly selected CPU/save options.

Architecture-specific model args and renderer must be resolved separately before
launch. Preserve every unrelated training flag, including the reference's TIS.
"""
    validate(config)
    result = copy.copy(base_args)
    values = {"--num-rollout": "10", "--num-steps-per-rollout": "1",
              "--hf-checkpoint": config["model_path"],
              "--save-interval": str(config["save_interval"])}
    for flag, value in values.items():
        if result.count(flag) != 1:
            raise ValueError("Expected exactly one base option: " + flag)
        result[result.index(flag) + 1] = value
    if "--offload-train-target" in result:
        result[result.index("--offload-train-target") + 1] = "cpu"
    else:
        result.extend(("--offload-train-target", "cpu"))
    for flag in ("--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d",
                 "--use-precision-aware-optimizer", "--offload-train"):
        if result.count(flag) != 1:
            raise ValueError("Reference run must already enable " + flag)
    return result


def validate_rollout_id(rollout_id, config):
    """Replace the old bridge's [0, 1] guard; no modulo/repeated batch IDs."""
    if type(rollout_id) is not int or not 0 <= rollout_id < config["num_rollout"]:
        raise ValueError("Rollout ID outside the configured run")
    return rollout_id
