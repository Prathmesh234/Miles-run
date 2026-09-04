"""Fail-closed PPO mathematics and pinned trajectory-transport gates.

The math gate executes unmodified functions from the pinned Miles source using
CP1 and a tensor-only value-head fixture. Distributed model execution is tested
by the subsequent two-update run, not by these CPU unit tests.
"""
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

def native():
    import torch
    root = Path(os.environ.get('MILES_SOURCE_ROOT', '/campaign/miles'))/'miles/backends/training_utils/loss_hub'
    ns = {'torch': torch, 'F': torch.nn.functional,
          'get_parallel_state': lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
          '_LOG_RATIO_EXP_CLAMP': 20.0}
    names = {'_safe_clamp_log_ratio', '_safe_exp_neg_ppo_kl', 'compute_policy_loss',
             'vanilla_gae', 'chunked_gae', 'get_advantages_and_returns_batch', 'value_loss_function'}
    hashes = {}
    for name in ['math_utils.py', 'losses.py']:
        path = root/name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_text())
        tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    for seed in range(16):
        torch.manual_seed(seed)
        lengths = [9, 7, 3]
        values = [torch.randn(n, requires_grad=True) for n in lengths]
        masks = [torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 0]),
                 torch.tensor([1, 1, 0, 1, 0, 1, 0]), torch.zeros(3)]
        raw_rewards = [1., 0., 1.]
        for gamma, lambd in [(1., 1.), (.99, .95)]:
            for chunked in [False, True]:
                adv, returns = ns['get_advantages_and_returns_batch'](
                    lengths, lengths, values, [torch.zeros(n) for n in lengths],
                    raw_rewards, 'thd', None, masks, gamma, lambd, chunked=chunked)
                for v, mask, reward, a, ret in zip(values, masks, raw_rewards, adv, returns):
                    indices = mask.nonzero().flatten().tolist()
                    expected = torch.zeros_like(v)
                    carry = 0.
                    following = 0.
                    for j in reversed(indices):
                        r = reward if j == indices[-1] else 0.
                        carry = r + gamma * following - v[j].detach() + gamma*lambd*carry
                        expected[j] = carry
                        following = v[j].detach()
                    torch.testing.assert_close(a, expected, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(ret, (expected+v.detach())*mask, atol=2e-5, rtol=2e-5)
                    assert not a.requires_grad and not ret.requires_grad
        current = torch.randn(19, requires_grad=True)
        old = torch.randn(19)
        advantage = torch.randn(19)
        mask = (torch.arange(19) % 3 != 0).float()
        actual, _ = ns['compute_policy_loss'](old-current, advantage, .2, .2)
        ratio = torch.exp(current-old)
        expected = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad((actual*mask).sum(), current, retain_graph=True)[0]
        gb = torch.autograd.grad((expected*mask).sum(), current)[0]
        torch.testing.assert_close(ga, gb)
        assert (ga[mask == 0] == 0).all()
        v = torch.randn(19, requires_grad=True)
        previous, targets = torch.randn(19), torch.randn(19)
        ns['get_values'] = lambda logits, **kw: {'values': [logits]}
        batch = {'values':[previous], 'returns':[targets], 'unconcat_tokens':[],
                 'total_lengths':[], 'response_lengths':[]}
        actual, _ = ns['value_loss_function'](SimpleNamespace(value_clip=.2), batch, v,
                                             lambda x: (x*mask).sum()/mask.sum())
        clipped = previous + (v-previous).clamp(-.2, .2)
        expected = (torch.maximum((v-targets).square(), (clipped-targets).square())*mask).sum()/mask.sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, v, retain_graph=True)[0],
                                   torch.autograd.grad(expected, v)[0])
    import training_entry
    args = training_entry.training_args(Path('/campaign/test'))
    assert args[args.index('--advantage-estimator')+1] == 'ppo'
    assert '--custom-reward-post-process-path' not in args
    assert '--disable-rewards-normalization' in args
    from rollout_adapter import validate_policy
    good = {'groups':[[{'logprobs':[-.1,-2.], 'loss_mask':[1,1]}]]}
    assert validate_policy(good, 248320)['status'] == 'passed'
    import math
    bad = {'groups':[[{'logprobs':[-math.log(248320)]*8, 'loss_mask':[1]*8}]]}
    try:
        validate_policy(bad, 248320)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Uniform policy was accepted')
    return {'fixtures':16, 'source_sha256':hashes, 'lifecycle':asyncio.run(lifecycle()), 'mask_aware_gae':'passed',
            'policy_and_value_gradients':'passed', 'raw_reward_configuration':'passed'}


async def transport():
    import harness_bridge as bridge
    import verifiers.v1 as vf
    from prime_rl.orchestrator.algo.base import iter_trainable_traces
    from test_adapters import test_transport
    original_out = bridge.OUT
    bridge.OUT = original_out/'validation'
    bridge.OUT.mkdir(exist_ok=True)
    try:
        await test_transport()
    finally:
        bridge.OUT = original_out
    fixture = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/runs/job-190/rollouts')
    cohorts = {}
    hashes = {}
    for path in sorted(fixture.glob('*.json')):
        match = re.fullmatch(r'(step\d+-group\d+)-(task_\d+)-\d+\.json', path.name)
        if match:
            cohorts.setdefault((match[1], match[2]), []).append(vf.Episode.model_validate_json(path.read_text()))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert cohorts, 'No pinned job-190 trajectory fixtures found'
    tested = 0
    for (_, task), episodes in cohorts.items():
        rows = bridge.convert_group(episodes, task)
        traces = [t for _, t in iter_trainable_traces(episodes)]
        assert len(rows) == len(traces) == 8
        for row, trace in zip(rows, traces, strict=True):
            assert row['reward'] == trace.reward
            assert len(row['loss_mask']) == len(row['logprobs']) == row['response_length']
            assert row['metadata']['credit_kind'] == 'transport_only'
        # Changing every reward to zero must not discard the group or alter its
        # sampled IDs, behavior logprobs, tool masks, lengths, or truncation.
        zero_payloads = [episode.model_dump() for episode in episodes]
        for payload in zero_payloads:
            for trace in payload['traces']:
                for component in trace['rewards'].values():
                    component['score'] = 0.
        zero_rows = bridge.convert_group([vf.Episode.model_validate(p) for p in zero_payloads], task)
        assert len(zero_rows) == 8
        for before, after in zip(rows, zero_rows, strict=True):
            assert after['reward'] == 0.
            for field in ['tokens', 'logprobs', 'loss_mask', 'response_length', 'truncated']:
                assert before[field] == after[field], field
        tested += len(rows)
    return {'fixture_directory':str(fixture), 'fixture_sha256':hashes,
            'traces':tested, 'sampled_ids_logprobs_masks':'passed',
            'zero_reward_group_admission':'passed', 'raw_reward_transport':'passed'}


async def lifecycle():
    path=Path(os.environ.get('MILES_DRIVER_PATH',str(Path(__file__).parent/'train_ppo.py')))
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.AsyncFunctionDef,ast.FunctionDef)) and n.name in ['update_actor_weights','preserve_actor_backup']]
    assert len(tree.body)==2
    namespace={};exec(compile(tree,str(path),'exec'),namespace)
    cases=0
    for critic,offload,colocate in [(True,True,False),(False,True,False),(True,False,False),(True,True,True)]:
        config=SimpleNamespace(use_critic=critic,offload_train=offload,colocate=colocate,disable_param_buffers_cpu_backup=True)
        namespace['preserve_actor_backup'](config)
        assert config.disable_param_buffers_cpu_backup == (not (critic and offload and not colocate))
        for fail in [False,True]:
            events=[]
            class Actor:
                async def onload(self):events.append('onload')
                async def update_weights(self,**kw):
                    assert kw=={'rollout_id':1};events.append('broadcast')
                    if fail:raise ValueError('fixture')
                async def offload(self):events.append('offload')
            raised=False
            try:
                await namespace['update_actor_weights'](SimpleNamespace(use_critic=critic,offload_train=offload,colocate=colocate),Actor(),rollout_id=1)
            except ValueError:raised=True
            assert raised==fail
            expected=['onload','broadcast','offload'] if critic and offload and not colocate else ['broadcast']
            assert events==expected,(events,expected)
            cases+=1
    return {'cases':cases,'ordering_and_exception_cleanup':'passed','patched_driver_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode in ['native','transport','lifecycle']
    result = native() if mode == 'native' else asyncio.run(lifecycle() if mode=='lifecycle' else transport())
    result.update(status='passed', mode=mode)
    out = Path(os.environ['MILES_RUN_DIR'])/f'ppo-{mode}-tests.json'
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(out)
    print(json.dumps(result), flush=True)
