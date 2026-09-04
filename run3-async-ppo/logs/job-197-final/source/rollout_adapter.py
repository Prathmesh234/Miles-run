"""Miles rollout hook that uses the unchanged, pinned Terminal-Lego harness."""
import json
import math
import os
from pathlib import Path
import time
import httpx
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.utils.types import Sample


def rewards_with_baseline_credit(args, samples):
    """Credit was computed over native surviving groups before batch cutting."""
    return [s.reward for s in samples], [s.metadata["advantage"] for s in samples]


def validate_policy(payload, vocab_size, output=None):
    """Fail before optimizer work if the actor produces the job-195 zero-weight signature."""
    active = [lp for group in payload["groups"] for row in group
              for lp, mask in zip(row["logprobs"], row["loss_mask"]) if mask]
    valid = bool(active) and all(math.isfinite(x) for x in active)
    uniform = bool(active) and all(abs(x + math.log(vocab_size)) < .01 for x in active)
    result = {"status": "passed" if valid and not uniform else "failed",
              "active_tokens": len(active), "uniform_distribution_signature": uniform,
              "finite_logprobs": valid, "vocab_size": vocab_size}
    if output:
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.replace(output)
    if not valid or uniform:
        raise RuntimeError("Policy validity gate failed; preserve artifacts and inspect actor weights")
    return result


class Rollout:
    def __init__(self, input):
        self.args = input.args

    async def __call__(self, input):
        if input.evaluation:
            raise ValueError("The baseline has no evaluation phase")
        args = self.args
        routers = getattr(args, "sglang_model_routers", None)
        ip, port = routers["default"] if routers and "default" in routers else (args.sglang_router_ip, args.sglang_router_port)
        from async_runtime import record
        version = input.weight_version
        if version is None or int(version) != max(1, input.rollout_id):
            raise ValueError(f"Unexpected one-batch-ahead policy version: {version}, rollout {input.rollout_id}")
        record("rollout", "start", rollout_id=input.rollout_id, behavior_version=int(version),
               expected_trainer_version=input.rollout_id+1, policy_lag=input.rollout_id+1-int(version))
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=1500, trust_env=False) as client:
            response = await client.post(os.environ["MILES_HARNESS_URL"] + "/rollout",
                                         json={"rollout_id": input.rollout_id, "weight_version": int(version), "router": f"http://{ip}:{port}"})
            response.raise_for_status()
            payload = response.json()
        if os.environ.get("MILES_ALGORITHM") == "ppo":
            model_config = json.loads((Path(args.hf_checkpoint)/"config.json").read_text())
            vocab_size = model_config.get('text_config', model_config)['vocab_size']
            validate_policy(payload, vocab_size,
                            Path(os.environ["MILES_RUN_DIR"])/f"policy-validity-{input.rollout_id}.json")
        groups = []
        sample_index = input.rollout_id * 16
        for group_index, group in enumerate(payload["groups"]):
            samples = []
            for row in group:
                sample = Sample(group_index=input.rollout_id * 20 + group_index,
                                index=sample_index)
                sample_index += 1
                sample.tokens = row["tokens"]
                sample.response_length = row["response_length"]
                sample.loss_mask = row["loss_mask"]
                sample.rollout_log_probs = row["logprobs"]
                sample.reward = row["reward"]
                sample.response = row["response"]
                sample.status = Sample.Status.TRUNCATED if row["truncated"] else Sample.Status.COMPLETED
                sample.metadata = row["metadata"]
                assert int(sample.metadata["policy_version"]) == int(version)
                sample.weight_versions = [str(version)]
                samples.append(sample)
            groups.append(samples)
        record("rollout", "end", rollout_id=input.rollout_id, behavior_version=int(version),
               elapsed_seconds=time.monotonic()-start)
        return RolloutFnTrainOutput(samples=groups, metrics={"harness_seconds": time.monotonic() - start, **payload["metrics"]})
