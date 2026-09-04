"""Duration/configuration gates; runtime model and CUDA validation are separate."""
import asyncio
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from long_run_config import extend_training_args, validate, validate_rollout_id

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    import json
    config = json.loads((ROOT / "run4-ppo-long/config/run.json").read_text())
    config.update(model_id="test/model", model_revision="a" * 40,
                  model_path="/test/model", model_args_recipe="fixture", renderer="fixture",
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
