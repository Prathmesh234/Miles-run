"""CPU gates for actual pinned async scheduling, TIS gradients and metrics."""
import ast
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from async_runtime import measured, preserve_actor_backup, update_actor_weights


async def lifecycle():
    for fail in (False, True):
        events = []
        class Actor:
            async def onload(self): events.append('onload')
            async def offload(self): events.append('offload')
            async def update_weights(self, **kw):
                events.append('broadcast')
                if fail: raise ValueError('fixture')
        args = SimpleNamespace(use_critic=True, offload_train=True, colocate=False,
                               disable_param_buffers_cpu_backup=True)
        preserve_actor_backup(args)
        assert args.disable_param_buffers_cpu_backup is False
        try:
            await update_actor_weights(args, Actor(), rollout_id=0)
            assert not fail
        except ValueError:
            assert fail
        assert events == ['onload', 'broadcast', 'offload']


async def scheduling(path):
    """Execute the generated native train() with fake async model/rollout actors."""
    events, versions = [], []
    training_started, generation_started = asyncio.Event(), asyncio.Event()
    version = 0
    class Model:
        def __init__(self, role): self.role = role
        async def onload(self): pass
        async def offload(self): pass
        async def save_model(self, *a, **kw): pass
        async def update_weights(self, **kw):
            nonlocal version
            version += 1
            events.append('publish'+str(version))
        async def train(self, rollout, data, **kw):
            events.append(self.role+str(rollout)+'_start')
            if self.role == 'critic' and rollout == 0:
                training_started.set()
                await generation_started.wait()
            await asyncio.sleep(0)
            events.append(self.role+str(rollout)+'_end')
    async def generate(rollout):
        versions.append(version)
        events.append('rollout'+str(rollout)+'_start')
        if rollout == 1:
            generation_started.set()
            await training_started.wait()
        await asyncio.sleep(0)
        events.append('rollout'+str(rollout)+'_end')
        return rollout
    async def no_op(*a, **kw): pass
    manager = SimpleNamespace(generate=SimpleNamespace(remote=lambda r: asyncio.create_task(generate(r))),
                              save=SimpleNamespace(remote=no_op), dispose=SimpleNamespace(remote=no_op))
    async def create_models(*a): return Model('actor'), Model('critic')
    class Eval:
        def __init__(self, *a): pass
        async def drain(self): pass
    args = SimpleNamespace(colocate=False, use_critic=True, offload_train=True,
        disable_param_buffers_cpu_backup=True, control_server_port=None,
        check_weight_update_equal=False, eval_interval=None, start_rollout_id=0,
        num_rollout=2, num_critic_only_steps=0, save_trigger_sentinel=None,
        save_interval=None, update_weights_interval=1, debug_exit_after_rollout=None)
    noop = lambda *a, **k: None
    ns = dict(measured=measured, preserve_actor_backup=preserve_actor_backup,
              update_actor_weights=update_actor_weights, os=os,
              validate_async_off_policy_correction=noop, configure_logger=noop,
              MainProcessIdentity=lambda: None, maybe_start_periodic_pyspy_dump=noop,
              create_placement_groups=lambda a: {'rollout': None},
              object_store=SimpleNamespace(init_instance=noop), init_tracking=noop,
              create_rollout_manager=lambda *a: (manager, 2), create_training_models=create_models,
              maybe_start_mini_ft_controller=noop, EvalDispatcher=Eval,
              remove_rollout_data_refs=noop, should_run_periodic_action=lambda *a: False)
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'train']
    exec(compile(tree, str(path), 'exec'), ns)
    await asyncio.wait_for(ns['train'](args), 5)
    assert versions == [1, 1], versions
    assert events.index('rollout1_start') < events.index('critic0_end')
    assert events.index('rollout1_end') < events.index('publish2') < events.index('actor1_start')
    assert version == 3
    return {'events': events, 'behavior_versions': versions, 'trainer_versions': [1, 2],
            'bounded_lag_updates': [0, 1]}


def mathematics(root):
    import torch
    from async_metrics import tis, optimizer_inventory, before_train_step
    path = root/'miles/backends/training_utils/loss_hub/corrections.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '971ccb0bf00b43b0582839c5b8dc05e91162c878ab7ec0ca878e3b1e668f5318'
    name = 'miles.backends.training_utils.loss_hub.corrections'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    args = SimpleNamespace(tis_clip=2., tis_clip_low=0.)
    old = torch.log(torch.tensor([.1, .2, .3, .4]))
    behavior = old-torch.log(torch.tensor([.5, 1., 3., 4.]))
    current = (old+torch.tensor([.05, -.3, .4, -.1])).requires_grad_()
    advantages = torch.tensor([1., -2., 3., -4.])
    mask = torch.tensor([1., 1., 1., 0.])
    policy_ratio = (current-old).exp()
    pg = -torch.minimum(policy_ratio*advantages, policy_ratio.clamp(.8, 1.2)*advantages)
    actual, masks, metrics = tis(args, pg_loss=pg, train_log_probs=[old],
                                rollout_log_probs=[behavior], loss_masks=[mask])
    expected = pg*torch.tensor([.5, 1., 2., 2.])
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad((actual*mask).sum(), current, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected*mask).sum(), current)[0]
    torch.testing.assert_close(grad, expected_grad)
    assert grad[-1] == 0 and grad.abs().sum() > 0
    torch.testing.assert_close(metrics['tis_upper_clipfrac'], torch.tensor([0., 0., 1., 1.]))
    assert masks[0] is mask and not metrics['tis_weight'].requires_grad
    # On-policy TIS is identity; no double importance correction.
    identity, _, _ = tis(args, pg_loss=pg.detach(), train_log_probs=[old],
                         rollout_log_probs=[old], loss_masks=[mask])
    torch.testing.assert_close(identity, pg.detach())
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    q = torch.nn.Parameter(p.detach().clone())
    instrumented = torch.optim.Adam([p], lr=.01)
    control = torch.optim.Adam([q], lr=.01)
    before_train_step(args, 0, 0, None, instrumented, None)
    for param, opt in [(p, instrumented), (q, control)]:
        (param.square().sum()).backward(); opt.step()
    torch.testing.assert_close(p, q, atol=0, rtol=0)
    inventory = optimizer_inventory(instrumented)
    assert inventory['unique_storage_bytes']['state:cpu'] > 0
    return {'tis_value_gradient_mask_identity': 'passed', 'optimizer_hook_update_equality': 'passed',
            'cpu_optimizer_inventory': inventory}


if __name__ == '__main__':
    root = Path(os.environ.get('MILES_SOURCE_ROOT', '/campaign/miles'))
    driver = Path(os.environ.get('MILES_PATCH_OUTPUT_DIR', str(Path(__file__).parent)))/'train_async_ppo.py'
    # Keep test-only timeline rows separate from the measured training run.
    run = Path(os.environ['MILES_RUN_DIR'])
    os.environ['MILES_RUN_DIR'] = str(run/'validation')
    asyncio.run(lifecycle())
    result = {'status': 'passed', 'schedule': asyncio.run(scheduling(driver)), 'math': mathematics(root)}
    (run/'async-tests.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
