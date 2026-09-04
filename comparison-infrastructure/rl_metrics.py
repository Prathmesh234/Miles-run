"""Compact RL evidence from existing logs; no model execution or remote writes."""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalars(path):
    """Use rank-0 reduced metrics; merge identical duplicate log emissions only."""
    result = defaultdict(lambda: defaultdict(dict))
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        identity = re.search(r"\[(20\d\d-[\d: .-]+) (actor|critic)_cell0_rank0\]", line)
        metric = re.search(r"(?:rollout|(?:critic-)?step) (\d+): (\{.*\})", line)
        if not identity or not metric:
            continue
        values = ast.literal_eval(metric[2])
        row = result[int(metric[1])][identity[2]]
        for key, value in values.items():
            if not key.startswith(("train/", "rollout/")):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Non-finite/non-numeric metric {path}:{number}: {key}")
            if key in row and not math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Conflicting duplicate metric {path}:{number}: {key}")
            row[key] = value
    return {step: dict(roles) for step, roles in sorted(result.items())}


def sample_summary(sample):
    """Keep loss-mask action tokens distinct from the response span (includes tools)."""
    meta = sample["metadata"]
    mask, logprobs = sample["loss_mask"], sample["logprobs"]
    if len(mask) != sample["response_length"] or len(logprobs) != len(mask):
        raise ValueError("Response span, mask and behavior log probabilities are misaligned")
    selected = [float(p) for p, active in zip(logprobs, mask) if active]
    if not selected or not all(math.isfinite(p) for p in selected):
        raise ValueError("Accepted sample has no finite active action log probabilities")
    reward = float(sample["reward"])
    if not math.isfinite(reward):
        raise ValueError("Non-finite reward")
    return {"trace_id": meta["trace_id"], "task": meta["task"], "reward": reward,
            "turns": meta["turns"], "total_tokens": len(sample["tokens"]),
            "response_span_tokens": sample["response_length"], "active_action_tokens": len(selected),
            "behavior_nll_nats": -statistics.mean(selected), "truncated": sample["truncated"],
            "trace_truncated": meta.get("trace_truncated"), "packing_truncated": meta.get("packing_truncated"),
            "stop_condition": meta.get("stop_condition"),
            "behavior_weight_version": meta.get("policy_version"),
            # Deliberately exclude metadata.advantage: PPO's 1.0 is transport-only.
            "credit_kind": meta.get("credit_kind", "group_centered")}


def episode_summary(episode):
    traces = episode["traces"]
    if len(traces) > 1:
        raise ValueError("Branching episodes require an explicit denominator; expected one trace")
    trace = traces[0] if traces else {}
    score = trace.get("rewards", {}).get("solved", {}).get("score")
    if score is not None and not math.isfinite(score):
        raise ValueError("Non-finite task solved score")
    return {"episode_id": episode["id"], "trace_id": trace.get("id"),
            "task": episode["task"]["data"]["name"], "solved_score": score,
            "ok": bool(episode["ok"] and trace.get("ok", False)),
            "stop_condition": trace.get("stop_condition"),
            "error_count": len(episode.get("errors", [])) + len(trace.get("errors", []))}


def extract(repo, folder, log_root, rollout_root, job_id, label):
    repo, folder, log_root, rollout_root = map(Path, (repo, folder, log_root, rollout_root))
    trainlog = log_root / "training.log"
    metrics = scalars(trainlog)
    paths = [trainlog]
    updates = []
    sample_paths = sorted((p for p in rollout_root.glob("step*-samples.json")
                           if re.fullmatch(r"step\d+-samples.json", p.name)),
                          key=lambda p: int(re.fullmatch(r"step(\d+)-samples.json", p.name)[1]))
    if not sample_paths:
        raise ValueError("No accepted-batch evidence; do not generate empty charts")
    for path in sample_paths:
        index = int(re.fullmatch(r"step(\d+)-samples.json", path.name)[1])
        data = json.loads(path.read_text())
        samples = [sample_summary(s) for group in data["groups"] for s in group]
        if not samples:
            raise ValueError("Empty accepted batch")
        paths.append(path)
        episodes = []
        for ep in sorted(rollout_root.glob(f"step{index}-group*-*.json")):
            episodes.append(episode_summary(json.loads(ep.read_text())))
            paths.append(ep)
        expected_attempts = data["metrics"]["attempted_groups"] * 8
        if len(episodes) != expected_attempts:
            raise ValueError(f"Incomplete attempted-episode evidence for update {index}: {len(episodes)} != {expected_attempts}")
        accepted_ids = {s["trace_id"] for s in samples}
        episode_ids = {e["trace_id"] for e in episodes}
        if len(accepted_ids) != len(samples) or not accepted_ids <= episode_ids:
            raise ValueError("Accepted traces must be unique and present in attempted episodes")
        traced_episodes = [e for e in episodes if e["trace_id"] is not None]
        by_trace = {e["trace_id"]: e for e in traced_episodes}
        if len(by_trace) != len(traced_episodes):
            raise ValueError("Attempted episode trace IDs must be unique")
        for sample in samples:
            episode = by_trace[sample["trace_id"]]
            if (sample["task"] != episode["task"] or episode["solved_score"] is None
                    or not math.isclose(sample["reward"], episode["solved_score"], abs_tol=1e-9)):
                raise ValueError("Accepted task/reward disagrees with its attempted episode")
        reward_mean = statistics.mean(s["reward"] for s in samples)
        if not math.isclose(reward_mean, data["metrics"]["reward_mean"], abs_tol=1e-9):
            raise ValueError("Raw reward mean disagrees with bridge")
        actor = metrics.get(index - 1, {}).get("actor", {})
        critic = metrics.get(index - 1, {}).get("critic", {})
        if "train/loss" not in actor:
            raise ValueError(f"Missing actor training evidence for update {index}")
        if "rollout/raw_reward" in actor and not math.isclose(actor["rollout/raw_reward"], reward_mean, abs_tol=1e-6):
            raise ValueError("Raw reward mean disagrees with trainer")
        updates.append({"update": index, "source_step": index - 1, "samples": samples,
                        "attempted_episodes": episodes, "raw_reward_mean": reward_mean,
                        "rollout_seconds": data["metrics"]["episode_seconds"],
                        "actor": actor, "critic": critic})
    is_async = any("train/tis" in u["actor"] for u in updates)
    if not is_async:
        # Job 196's metadata.policy_version was a rollout ID, not engine version.
        for u in updates:
            for s in u["samples"]:
                s["behavior_weight_version"] = None
    if is_async:
        for u in updates:
            for s in u["samples"]:
                s["policy_lag_updates"] = u["update"] - int(s["behavior_weight_version"])
    missing = ["held-out evaluation", "critic explained variance / per-token value distribution",
               "full importance-weight histogram", "reward components beyond the recorded solved score"]
    if not any(u["critic"] for u in updates):
        missing.append("critic metrics: not applicable to this IPO run")
    if not is_async:
        missing.append("TIS: not enabled in this run")
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id, "label": label, "folder": str(folder.relative_to(repo)),
            "updates": updates, "is_async_tis": is_async,
            "source_sha256": {str(p.relative_to(repo)): sha(p) for p in paths},
            "missing_or_not_applicable": missing,
            "semantics": {"update": "1-based display; raw log step/rollout IDs are zero-based",
                "rewards": "Raw accepted-trace reward; training batch, not held-out evaluation",
                "attempts": "All serialized attempted episodes, including errors/unshipped traces; missing scores are excluded explicitly",
                "length": "Active action tokens from loss_mask; response span also contains tool/observation tokens",
                "entropy": "Trainer vocabulary entropy on selected positions, not sampled-token surprisal",
                "loss": "Metrics evaluated during each update, not post-update evaluation; IPO and PPO objectives differ",
                "ppo_kl": "Logged signed old-minus-current sampled log probability; not a guaranteed nonnegative KL divergence",
                "ppo_zero": "Async recomputes its pre-update policy: one step can report PPO KL=0 and ESS=1 despite stale behavior/TIS",
                "sync_zero": "Sync train_rollout mismatch=0 reuses behavior logprobs as its reference; not an independent mismatch measurement",
                "advantage": "Use rollout/advantages logged by trainer; never PPO metadata.advantage (transport-only 1.0)",
                "uncertainty": "Two updates and tiny, correlated task groups do not establish convergence or model-quality ranking"}}
