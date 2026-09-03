"""CPU fidelity checks against the pinned baseline and native token transport."""
import ast
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from ipo_loss import token_loss

BASE = Path('/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00')


def test_ipo():
    source = BASE / 'prime-rl/src/prime_rl/trainer/rl/loss.py'
    names = {'_safe_mean', 'compute_importance_ratio_and_mismatch_kl', 'ipo_loss_fn'}
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'torch':torch, 'Tensor':torch.Tensor, 'LossInputs':SimpleNamespace,
                 'IPOLossConfig':SimpleNamespace, 'LossOutputs':SimpleNamespace}
    exec(compile(tree, str(source), 'exec'), namespace)
    config = SimpleNamespace(eps=.1, adv_tau=1., kl_tau=.001)
    for seed in range(32):
        torch.manual_seed(seed)
        current = (-torch.rand(137)*8).requires_grad_()
        old = -torch.rand(137)*8
        advantage = torch.randn(137)
        mask = torch.rand(137) > .4
        inputs = SimpleNamespace(trainer_logprobs=current, inference_logprobs=old,
                                 advantages=advantage, loss_mask=mask, loss_weights=None)
        expected = namespace['ipo_loss_fn'](inputs, config).loss
        actual = token_loss(current, old, advantage, mask).sum()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad_a = torch.autograd.grad(actual, current, retain_graph=True)[0]
        grad_b = torch.autograd.grad(expected, current)[0]
        torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)


async def test_transport():
    import harness_bridge as bridge
    class Request:
        async def json(self):
            return {'token_ids':[12, 15, 17], 'sampling_params':{'max_tokens':2048, 'stop_token_ids':[99]}}
    class Client:
        async def post(self, url, json):
            assert url == 'http://test-router/generate'
            assert json['input_ids'] == [12, 15, 17]
            assert json['return_logprob'] is True
            assert json['sampling_params']['max_new_tokens'] == 2048
            return SimpleNamespace(status_code=200, json=lambda:{'meta_info':{
                'output_token_logprobs':[[-.7, 34, None],[-.9, 99, None]],
                'completion_tokens':2, 'finish_reason':{'type':'stop'}}})
    bridge.STATE.update(http=Client(), router='http://test-router')
    result = await bridge.generate(Request())
    choice = result['choices'][0]
    assert choice['token_ids'] == [34, 99]
    assert choice['finish_reason'] == 'stop'
    assert choice['logprobs']['content'] == [{'token':'token_id:34','logprob':-.7},{'token':'token_id:99','logprob':-.9}]


async def test_real_client_and_admission():
    import httpx
    from openai import AsyncOpenAI
    import harness_bridge as bridge
    import verifiers.v1 as vf
    from verifiers.v1.clients.train import TrainClient
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.dialects import ChatDialect
    from prime_rl.configs.algorithm import GRPOAlgoConfig
    from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
    from prime_rl.orchestrator.algo.base import iter_trainable_traces
    from prime_rl.orchestrator.algo.routing import scalar_advantage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bridge.MODEL, local_files_only=True)
    ids = tokenizer.encode('Done.', add_special_tokens=False)
    finish = 'stop'
    class Engine:
        async def post(self, url, json):
            return SimpleNamespace(status_code=200, json=lambda:{'meta_info':{
                'output_token_logprobs':[[-.25, t, None] for t in ids],
                'completion_tokens':len(ids), 'finish_reason':{'type':finish}}})
    bridge.STATE.update(http=Engine(), router='http://test-router')
    async def transport(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json=await bridge.models())
        assert request.url.path == '/inference/v1/generate'
        class Request:
            async def json(self):
                return json.loads(request.content)
        return httpx.Response(200, json=await bridge.generate(Request()))
    config = TrainClientConfig(base_url='http://adapter-test/v1', api_key_var='MILES_UNUSED_API_KEY',
        renderer=json.loads(bridge.ORCHESTRATOR_CONFIG.read_text())['renderer'], renderer_model_name=bridge.MODEL)
    client = TrainClient(config)
    await client.client.close()
    client.client = AsyncOpenAI(base_url='http://adapter-test/v1', api_key='test',
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    for finish in ['stop', 'length']:
        response = await client.get_response(ChatDialect(),
            {'model':bridge.MODEL,'messages':[{'role':'user','content':'Say done.'}]},
            vf.SamplingConfig(temperature=1.,top_p=1.,max_tokens=2048))
        assert response.finish_reason == finish
    await client.close()

    groups = json.loads((bridge.OUT.parent/'baseline-accepted-episodes.json').read_text())
    for group in groups:
        episodes = [vf.Episode.model_validate(e) for e in group['episodes']]
        rows = bridge.convert_group(episodes, group['task'])
        assert len(rows) == 8
        for row in rows:
            assert len(row['loss_mask']) == len(row['logprobs']) == row['response_length']
    # Real failed-attempt fixtures cover the failure that the original test missed.
    paths = sorted((bridge.OUT.parents[1]/'runs/job-189/rollouts').glob('*task_14118-*.json'))
    episodes = [vf.Episode.model_validate_json(p.read_text()) for p in paths]
    assert len(episodes) == 8 and any(not e.ok for e in episodes)
    algorithm = GRPOAlgorithm(GRPOAlgoConfig(), clients=None)
    await algorithm.score_group(episodes)
    expected = {t.id:scalar_advantage(t) for _,t in iter_trainable_traces(episodes)}
    expected = {k:v for k,v in expected.items() if v != 0}
    rows = bridge.convert_group(episodes, 'task_14118')
    assert {r['metadata']['trace_id']:r['metadata']['advantage'] for r in rows} == expected


if __name__ == '__main__':
    test_ipo()
    asyncio.run(test_transport())
    asyncio.run(test_real_client_and_admission())
    result = {'ipo_value_and_gradient_fixtures':32, 'transport_mapping':'passed',
              'actual_TrainClient_stop_and_length':'passed', 'all_32_baseline_traces':'passed',
              'native_errored_group_admission_and_credit':'passed', 'status':'passed'}
    Path(os.environ['MILES_RUN_DIR'], 'adapter-test-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
