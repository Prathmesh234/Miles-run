"""Run the baseline Verifiers and renderer unchanged, translating token transport to SGLang.

This process runs in the original pinned Python environment with bytecode writes
disabled. It writes only into the new run directory. Miles communicates over HTTP.
"""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import time
import uuid

import httpx
import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import verifiers.v1 as vf
from verifiers.v1.configs.client import TrainClientConfig
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.trainer.batch import prepare_sample
from renderers import RendererConfig

BASE = Path("/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00")
CONFIG = BASE / "runs/20260903-150011/prime-rl/configs/attempt_1/resolved/envs/train/clustermax-terminal-lego.json"
ORCHESTRATOR_CONFIG = BASE / "runs/20260903-150011/prime-rl/configs/attempt_1/resolved/orchestrator.json"
MODEL = str(BASE / "model-fetch/models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0")
OUT = Path(os.environ["MILES_RUN_DIR"])
PORT = int(os.environ.get("MILES_HARNESS_PORT", "18981"))
STATE = {}
LOCK = asyncio.Lock()


@contextlib.asynccontextmanager
async def lifespan(app):
    env = vf.load_environment(vf.resolve_env_config(json.loads(CONFIG.read_text())["env"]))
    STATE["env"] = env
    STATE["tasks"] = list(env.taskset.load())
    STATE["cursor"] = 0
    STATE["http"] = httpx.AsyncClient(timeout=1250, trust_env=False, limits=httpx.Limits(max_connections=40))
    (OUT / "rollouts").mkdir(parents=True, exist_ok=True)
    async with env.serving():
        yield
    await STATE["http"].aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "tasks": [t.data.name for t in STATE["tasks"]]}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": 8192}]}


@app.post("/inference/v1/generate")
async def generate(request: Request):
    body = await request.json()
    sp = body["sampling_params"]
    token_ids = body["token_ids"]
    if len(token_ids) >= 8192:
        return JSONResponse({"error": {"message": "Prompt exceeds model context", "type": "invalid_request_error"}}, status_code=400)
    sampling = {"temperature": sp.get("temperature", 1.0), "top_p": sp.get("top_p", 1.0),
                "top_k": sp.get("top_k", -1), "min_p": sp.get("min_p", 0.0),
                "max_new_tokens": min(sp.get("max_tokens", 2048), 8192 - len(token_ids)),
                "stop_token_ids": sp["stop_token_ids"], "skip_special_tokens": False}
    payload = {"input_ids": token_ids, "sampling_params": sampling, "return_logprob": True,
               "logprob_start_len": -1, "return_text_in_logprobs": False}
    start = time.monotonic()
    response = await STATE["http"].post(STATE["router"] + "/generate", json=payload)
    if response.status_code >= 400:
        return JSONResponse({"error": {"message": response.text[:4000], "type": "invalid_request_error"}}, status_code=response.status_code)
    data = response.json()
    meta = data["meta_info"]
    evidence = meta["output_token_logprobs"]
    ids = [int(x[1]) for x in evidence]
    assert len(ids) == meta["completion_tokens"], "SGLang must return complete token evidence"
    finish = meta["finish_reason"]["type"]
    result = {"request_id": meta.get("id", uuid.uuid4().hex), "prompt_token_ids": token_ids,
              "choices": [{"token_ids": ids, "finish_reason": "length" if finish == "length" else "stop",
                           "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": float(lp)} for lp, t, *_ in evidence]}}]}
    with (OUT / "generation-timing.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), "seconds": time.monotonic()-start, "prompt_tokens":len(token_ids),
                            "completion_tokens":len(ids), "finish": finish, "meta": meta}) + "\n")
    return result


def convert_group(episodes, task_name):
    if len(episodes) != 8:
        raise ValueError("Every generated baseline task group must have eight episodes")
    # Use the original admission rule: exclude errored/untrainable traces,
    # then compute FP32 group credit over the surviving cohort.
    traces = [trace for _, trace in iter_trainable_traces(episodes)]
    if not traces:
        return []
    rewards = torch.tensor([t.reward for t in traces], dtype=torch.float32)
    advantages = (rewards - rewards.mean()).tolist()
    if not any(advantages):
        # Prime-RL prunes groups with no nonzero advantages before filling its
        # constant 16-trace training batch. Advance the same cyclic task source.
        return []
    result = []
    for trace, advantage in zip(traces, advantages, strict=True):
        if advantage == 0.0:
            continue
        assign_advantages(trace, advantage)
        samples = trace_to_samples(trace, env_name="clustermax-terminal-lego")
        if len(samples) != 1:
            raise ValueError("Unexpected branching would change the group-size comparison")
        sample = samples[0]
        sample.temperatures = [1.0] * len(sample.token_ids)
        prepared = prepare_sample(sample, 8192)
        mask = [int(x and advantage != 0.0) for x in prepared.loss_mask]
        if not any(mask):
            raise ValueError("Zero-advantage group requires explicit baseline admission handling")
        first = mask.index(1)
        response_tokens = prepared.input_ids[first:]
        result.append({"tokens": prepared.input_ids, "response_length": len(response_tokens),
                       "loss_mask": mask[first:], "logprobs": prepared.inference_logprobs[first:],
                       "reward": float(trace.reward), "response": "", "truncated": trace.is_truncated or len(sample.token_ids) > 8192,
                       "metadata": {"task": task_name, "trace_id": str(trace.id), "turns": trace.num_turns,
                                    "full_tokens": len(sample.token_ids), "advantage": advantage,
                                    "trace_truncated": trace.is_truncated, "stop_condition": trace.stop_condition,
                                    "packing_truncated": len(sample.token_ids) > 8192}})
    return result


@app.post("/rollout")
async def rollout(request: Request):
    body = await request.json()
    async with LOCK:
        rollout_id = int(body["rollout_id"])
        if rollout_id not in [0, 1]:
            raise ValueError("Only the two baseline training steps are authorized")
        STATE["router"] = body["router"]
        ctx = vf.ModelContext(model=MODEL,
            client=TrainClientConfig(base_url=f"http://127.0.0.1:{PORT}/v1", api_key_var="MILES_UNUSED_API_KEY",
                                    renderer=json.loads(ORCHESTRATOR_CONFIG.read_text())["renderer"], renderer_model_name=MODEL),
            sampling=vf.SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=2048,
                                       extra_body={"top_k":-1,"min_p":0.0,"return_token_ids":True}))
        start = time.monotonic()
        async def run_group(task):
            return await asyncio.gather(*(STATE["env"].run_episode(task, ctx) for _ in range(8)))
        groups = []
        attempted = 0
        while sum(map(len, groups)) < 16:
            count = min(2, (16 - sum(map(len, groups)) + 7) // 8)
            tasks = [STATE["tasks"][(STATE["cursor"] + k) % 4] for k in range(count)]
            STATE["cursor"] += count
            episodes = await asyncio.gather(*(run_group(task) for task in tasks))
            for task, group in zip(tasks, episodes):
                attempted += 1
                for index, episode in enumerate(group):
                    (OUT/"rollouts"/f"step{rollout_id+1}-group{attempted}-{task.data.name}-{index}.json").write_text(episode.model_dump_json())
                converted = convert_group(group, task.data.name)
                if converted:
                    groups.append(converted)
            if attempted >= 20 and sum(map(len, groups)) < 16:
                raise ValueError("No complete nonzero-advantage batch after 160 episodes")
        # A partially surviving group may cross the fixed 16-trace boundary.
        # Keep its original cohort advantages, never re-center the cut group.
        # Excess samples are retained as evidence but cannot cross a weight
        # update because the baseline permits zero off-policy steps.
        overflow = []
        accepted = []
        remaining = 16
        for group in groups:
            if remaining:
                accepted.append(group[:remaining])
            overflow.extend(group[remaining:])
            remaining = max(0, remaining - len(group))
        groups = accepted
        (OUT/"rollouts"/f"step{rollout_id+1}-unshipped-samples.json").write_text(json.dumps(overflow))
        result = {"groups": groups, "metrics": {"episode_seconds":time.monotonic()-start,
                  "attempted_groups":attempted,
                  "reward_mean":sum(s["reward"] for g in groups for s in g)/16}}
        (OUT/"rollouts"/f"step{rollout_id+1}-samples.json").write_text(json.dumps(result))
        return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
