"""Duration/configuration gates; runtime model and CUDA validation are separate."""
import asyncio
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from long_run_config import extend_training_args, validate, validate_rollout_id
from build_bundle import build

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    import json
    config = json.loads((ROOT / "run4-ppo-long/config/run.json").read_text())
    config.update(model_id="test/model", model_revision="a" * 40,
                  model_path="/test/model", converted_model_path="/test/converted-model",
                  model_args_recipe="fixture", renderer="fixture",
                  save_interval=10)
    return config


class ConfigTests(unittest.TestCase):
    def test_pending_choices_block_launch(self):
        import json
        for folder in ("run4-ppo-long", "run5-async-ppo-long"):
            config = json.loads((ROOT / folder / "config/run.json").read_text())
            unresolved = copy.deepcopy(config)
            unresolved["model_revision"] = None
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                validate(unresolved)

    def test_exact_ten_update_boundary(self):
        cfg = fixture()
        self.assertEqual([validate_rollout_id(i, cfg) for i in range(10)], list(range(10)))
        for bad in (-1, 10, 11, True, 1.5):
            with self.assertRaises(ValueError):
                validate_rollout_id(bad, cfg)
        cfg["num_steps_per_rollout"] = 10
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_only_authorized_argument_changes(self):
        base = ["--num-rollout", "2", "--num-steps-per-rollout", "1", "--hf-checkpoint", "/old",
                "--save-interval", "2", "--optimizer-cpu-offload", "--overlap-cpu-optimizer-d2h-h2d",
                "--use-precision-aware-optimizer", "--offload-train", "--lr", "1e-6",
                "--global-batch-size", "16", "--use-tis", "--tis-clip", "2.0"]
        original = base[:]
        actual = extend_training_args(base, fixture())
        expected = base[:]
        for flag, value in {"--num-rollout": "10", "--hf-checkpoint": "/test/model",
                            "--save-interval": "10"}.items():
            expected[expected.index(flag) + 1] = value
        expected.extend(("--offload-train-target", "cpu"))
        self.assertEqual(actual, expected)
        self.assertEqual(base, original)

    def test_cpu_offload_required(self):
        cfg = fixture()
        cfg["cpu_offload"]["optimizer_cpu_offload"] = False
        with self.assertRaisesRegex(ValueError, "optimizer_cpu_offload"):
            validate(cfg)

    def test_immutable_model_revision_required(self):
        cfg = fixture()
        cfg["model_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate(cfg)

    def test_distinct_conversion_required(self):
        cfg = fixture()
        cfg["converted_model_path"] = cfg["model_path"]
        with self.assertRaisesRegex(ValueError, "separate"):
            validate(cfg)

    def test_async_tis_preserved(self):
        cfg = fixture()
        cfg["mode"] = "async_ppo_tis"
        cfg["tis"] = {"enabled": True, "clip_low": 0.0, "clip_high": 2.0}
        validate(cfg)
        cfg["tis"]["clip_high"] = 5.0
        with self.assertRaisesRegex(ValueError, "TIS"):
            validate(cfg)

    def test_generated_scripts_use_new_config_in_both_modes(self):
        """Execute generated argument functions, not a duplicate expected recipe."""
        import hashlib
        import json
        import os
        import shlex
        for mode, reference in (("ppo", "run2-ppo"), ("async_ppo_tis", "run3-async-ppo")):
            cfg = fixture()
            cfg["mode"] = mode
            if mode == "async_ppo_tis":
                cfg["tis"] = {"enabled": True, "clip_low": 0.0, "clip_high": 2.0}
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "source"
                manifest = build(cfg, dest)
                for name, digest in manifest["output_sha256"].items():
                    self.assertEqual(hashlib.sha256((dest / name).read_bytes()).hexdigest(), digest)
                with self.assertRaises(FileExistsError):
                    build(cfg, dest)

                def arguments(source, config=None):
                    tree = ast.parse(source)
                    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                    scope = dict(os=os, Path=Path, shlex=shlex, ROOT=Path("/campaign"),
                                 MODEL="/test/model", CONFIG=config, __file__=str(dest / "training_entry.py"),
                                 load_model_args=lambda recipe: "--test-model-recipe " + recipe + " --mtp-num-layers 1",
                                 extend_training_args=extend_training_args)
                    exec(compile(tree, "generated_entry.py", "exec"), scope)
                    with patch.dict(os.environ, {"MILES_ALGORITHM": "ppo"}):
                        return scope["training_args"](Path("/campaign/runs/test"))

                actual = arguments((dest / "training_entry.py").read_text(), cfg)
                expected = arguments((ROOT / reference / "scripts/training_entry.py").read_text())
                for flag, value in {"--test-model-recipe": "fixture", "--num-rollout": "10",
                                    "--save-interval": "10"}.items():
                    expected[expected.index(flag) + 1] = value
                if mode == "ppo":
                    expected += ["--custom-megatron-init-path", "async_metrics.install",
                                 "--custom-megatron-before-train-step-hook-path", "async_metrics.before_train_step"]
                else:
                    # Moving the two instrumentation flags does not alter their values.
                    for flag in ("--custom-megatron-init-path", "--custom-megatron-before-train-step-hook-path"):
                        i = expected.index(flag)
                        pair = expected[i:i+2]
                        del expected[i:i+2]
                        position = expected.index("--use-tis")
                        expected[position:position] = pair
                expected += ["--offload-train-target", "cpu"]
                self.assertEqual(actual, expected)
                self.assertNotIn("--mtp-num-layers", actual)

                bridge_tree = ast.parse((dest / "harness_bridge.py").read_text())
                guard = [n for n in ast.walk(bridge_tree) if isinstance(n, ast.Expr)
                         and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                         and n.value.func.id == "validate_rollout_id"]
                self.assertEqual(len(guard), 1)
                code = compile(ast.fix_missing_locations(ast.Module(body=guard, type_ignores=[])),
                               "generated_bridge_guard", "exec")
                for i in range(10):
                    exec(code, dict(validate_rollout_id=validate_rollout_id, rollout_id=i, RUN_CONFIG=cfg))
                with self.assertRaises(ValueError):
                    exec(code, dict(validate_rollout_id=validate_rollout_id, rollout_id=10, RUN_CONFIG=cfg))

    def test_conversion_identity_checked_before_start(self):
        import json
        from build_bundle import transform_coordinator
        source = transform_coordinator((ROOT / "run3-async-ppo/scripts/coordinator.py").read_text())
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        # Execute the real generated identity guard, stopping before run creation.
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "receipt" for t in n.targets))
        end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "RUN.mkdir")
        code = compile(ast.fix_missing_locations(ast.Module(body=main.body[start:end], type_ignores=[])),
                       "conversion_identity_guard", "exec")
        cfg = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            converted = Path(tmp)
            (converted / "release").mkdir()
            receipt = converted / "conversion-complete.json"
            receipt.write_text(json.dumps({"model_id": cfg["model_id"], "model_revision": cfg["model_revision"]}))
            scope = dict(json=json, CONVERTED=converted, CONFIG=cfg)
            exec(code, scope)
            receipt.write_text(json.dumps({"model_id": "old/35b", "model_revision": "b" * 40}))
            with self.assertRaisesRegex(ValueError, "identity"):
                exec(code, scope)

    def test_ten_step_native_async_schedule(self):
        """Extend the previous native-driver fixture to exercise all ten steps.

        This proves scheduling, not GPU execution or model compatibility.
        """
        import sys
        base = ROOT / "run3-async-ppo/scripts"
        sys.path.insert(0, str(base))
        try:
            source = (base / "test_async.py").read_text()
            source = source.replace("num_rollout=2,", "num_rollout=10,")
            source = source.replace("assert versions == [1, 1], versions",
                                    "assert versions == [1, 1, *range(2, 10)], versions")
            source = source.replace("assert version == 3", "assert version == 11\n    "
                "assert all(events.count(f'{role}{i}_end') == 1 for role in ('actor', 'critic') for i in range(10))")
            tree = ast.parse(source)
            tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
            scope = {"__name__": "long_schedule_fixture"}
            exec(compile(tree, "extended_test_async.py", "exec"), scope)
            driver = ROOT / "run3-async-ppo/config/local-driver-check/train_async_ppo.py"
            import os
            with tempfile.TemporaryDirectory() as tmp:
                old = os.environ.get("MILES_RUN_DIR")
                os.environ["MILES_RUN_DIR"] = tmp
                try:
                    result = asyncio.run(scope["scheduling"](driver))
                    self.assertEqual(result["behavior_versions"], [1, 1, *range(2, 10)])
                finally:
                    if old is None:
                        os.environ.pop("MILES_RUN_DIR", None)
                    else:
                        os.environ["MILES_RUN_DIR"] = old
        finally:
            sys.path.remove(str(base))


if __name__ == "__main__":
    unittest.main()
