"""Build frozen longer-run sources from hash-checked prior-run scripts.

No downloads, Kubernetes writes or Slurm submission. Generated copies belong in
an ignored staging directory and later on the cluster, not in Git. Publish this
builder, the config, and its source-hash manifest instead of vendoring snapshots.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from long_run_config import load, validate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "run3-async-ppo/scripts"
SOURCES = (
    "coordinator.py", "training_entry.py", "harness_bridge.py", "rollout_adapter.py",
    "ipo_loss.py", "test_adapters.py", "test_ppo.py", "sglang.yaml",
    "prepare_ppo_driver.py", "ppo-resident-broadcast.patch", "install_precision_patch.py",
    "sglang_precision.py", "capture_infra.py", "capture_rdma.py", "capture_metrics.py",
    "capture_health.py", "container_evidence.py", "extract_metrics.py", "verify_checkpoint.py",
    "run.sbatch", "prepare_async_driver.py", "async_runtime.py", "async_metrics.py", "test_async.py",
)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Source contract drift: " + before[:100])
    return source.replace(before, after, 1)


def transform_entry(source):
    source = replace_once(source, 'ROOT = Path("/campaign")',
        'from long_run_config import load_runtime, extend_training_args\n'
        'CONFIG = load_runtime()\nROOT = Path("/campaign")')
    start = source.index('MODEL = ')
    end = source.index('\n', start)
    source = source[:start] + 'MODEL = CONFIG["model_path"]' + source[end:]
    source = replace_once(source, 'load_model_args("qwen3.6-35B-A3B")',
                          'load_model_args(CONFIG["model_args_recipe"])')
    source = replace_once(source, '    idx = args.index("--mtp-num-layers")\n    del args[idx:idx+2]',
        '    if "--mtp-num-layers" in args:\n'
        '        idx = args.index("--mtp-num-layers")\n        del args[idx:idx+2]')
    begin = source.index('    assert os.environ.get("MILES_ALGORITHM") == "ppo"')
    end = source.index('\n\ndef main():', begin)
    source = source[:begin] + '''    assert os.environ.get("MILES_ALGORITHM") == "ppo", "PPO is required"
    result += ["--custom-megatron-init-path", "async_metrics.install",
               "--custom-megatron-before-train-step-hook-path", "async_metrics.before_train_step"]
    if CONFIG["mode"] == "async_ppo_tis":
        result.remove("--use-rollout-logprobs")
        result += ["--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0",
                   "--custom-tis-function-path", "async_metrics.tis",
                   "--update-weights-interval", "1"]
    return extend_training_args(result, CONFIG)

''' + source[end:]
    source = replace_once(source,
        '        driver = Path(__file__).parent/"train_async_ppo.py" if os.environ.get("MILES_ALGORITHM") == "ppo" else ROOT/"miles/train.py"',
        '        driver = Path(__file__).parent / ("train_async_ppo.py" if CONFIG["mode"] == "async_ppo_tis" else "train_ppo.py")')
    source = replace_once(source, '        validate_async_off_policy_correction(args)',
        '        if CONFIG["mode"] == "async_ppo_tis":\n            validate_async_off_policy_correction(args)')
    source = replace_once(source,
        '        assert args.use_tis and not args.use_rollout_logprobs and not args.skip_actor_forward_only',
        '        if CONFIG["mode"] == "async_ppo_tis":\n'
        '            assert args.use_tis and not args.use_rollout_logprobs and not args.skip_actor_forward_only\n'
        '        assert args.num_rollout == 10 and args.num_steps_per_rollout == 1\n'
        '        assert args.optimizer_cpu_offload and args.offload_train_target == "cpu"')
    return source.replace('"""Run the two-step Miles comparison inside its isolated container."""',
                          '"""Run the configured ten-update PPO experiment in its isolated container."""')


def transform_bridge(source):
    source = replace_once(source, 'STATE = {}',
        'from long_run_config import load_runtime, validate_rollout_id\n'
        'RUN_CONFIG = load_runtime()\nMODEL = RUN_CONFIG["model_path"]\nSTATE = {}')
    source = replace_once(source,
        '        if rollout_id not in [0, 1]:\n            raise ValueError("Only the two baseline training steps are authorized")',
        '        validate_rollout_id(rollout_id, RUN_CONFIG)')
    source = replace_once(source,
        'renderer=json.loads(ORCHESTRATOR_CONFIG.read_text())["renderer"], renderer_model_name=MODEL',
        'renderer=RUN_CONFIG["renderer"], renderer_model_name=MODEL')
    return source


def transform_coordinator(source):
    source = replace_once(source, 'JID = os.environ["SLURM_JOB_ID"]',
        'from long_run_config import load_runtime\nCONFIG = load_runtime()\n'
        'MODEL = Path(CONFIG["model_path"])\nCONVERTED = Path(CONFIG["converted_model_path"])\n'
        'JID = os.environ["SLURM_JOB_ID"]')
    source = replace_once(source, '    RUN.mkdir(exist_ok=False)', '''    # A different model must never inherit job 190's 35B conversion silently.
    receipt = json.loads((CONVERTED / "conversion-complete.json").read_text())
    if (receipt.get("model_id"), receipt.get("model_revision")) != (CONFIG["model_id"], CONFIG["model_revision"]):
        raise ValueError("Converted checkpoint identity differs from the requested model")
    if not (CONVERTED / "release").is_dir():
        raise ValueError("Converted model release is missing")
    RUN.mkdir(exist_ok=False)''')
    source = replace_once(source, 'str(REUSE/"converted-model")+":/campaign/converted-model:ro"',
                          'str(CONVERTED)+":/campaign/converted-model:ro"')
    begin = source.index('        if not (REUSE/"converted-model/conversion-complete.json").exists():')
    end = source.index('        if os.environ.get("MILES_ALGORITHM") == "ppo":', begin)
    source = source[:begin] + '        event("checkpoint_conversion_reused", path=str(CONVERTED), model_revision=CONFIG["model_revision"])\n' + source[end:]
    # The original renderer/transport gate consumes the bridge's same new model.
    return source


def build(config, destination):
    validate(config)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a source bundle: " + str(destination))
    expected = json.loads((ROOT / "run3-async-ppo/config/publication-script-sha256.json").read_text())
    output, inputs = {}, {}
    transforms = {"training_entry.py": transform_entry, "harness_bridge.py": transform_bridge,
                  "coordinator.py": transform_coordinator}
    for name in SOURCES:
        data = (BASE / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected["scripts/" + name]:
            raise ValueError("Reference source hash changed: " + name)
        inputs[str((BASE / name).relative_to(ROOT))] = digest
        output[name] = transforms[name](data.decode()).encode() if name in transforms else data
    output["long_run_config.py"] = (HERE / "long_run_config.py").read_bytes()
    output["run-config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    for name, data in output.items():
        if name.endswith(".py"):
            ast.parse(data.decode(), filename=name)
    manifest = {"stage": "source_bundle_only_not_submitted", "input_sha256": inputs,
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_sha256": {n: hashlib.sha256(d).hexdigest() for n, d in output.items()},
                "changed_reference_sources": sorted(transforms),
                "remaining_gates": ["model/tokenizer/renderer compatibility", "router precision compatibility",
                    "immutable model download and conversion verification", "CPU preflight in pinned image",
                    "storage budget for both runs", "Slurm submission and ten actual updates per role"]}
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in output.items():
        (destination / name).write_bytes(data)
    (destination / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = build(load(args.config), args.destination)
    print(json.dumps({"status": manifest["stage"], "files": len(manifest["output_sha256"]),
                      "destination": str(args.destination)}))
