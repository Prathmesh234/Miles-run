"""Patch the pinned native one-batch-ahead driver into this run only."""
import difflib
import hashlib
import json
import os
from pathlib import Path

BASE_SHA = 'b0550f83b912e6ad133cbcf39949b48d2eb1cbf0f2bfd9e54884f4cc8dc25dca'


def prepare(upstream, destination):
    original = (upstream/'train_async.py').read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == BASE_SHA
    source = original.replace('import asyncio\n', 'import asyncio\nfrom async_runtime import measured, preserve_actor_backup, update_actor_weights\n', 1)
    replacements = {
        '    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)':
        '    preserve_actor_backup(args)\n    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)',
        'await actor_model.update_weights()': 'await update_actor_weights(args, actor_model)',
        'await actor_model.update_weights(rollout_id=rollout_id)': 'await update_actor_weights(args, actor_model, rollout_id=rollout_id)',
        'await critic_model.train(rollout_id, rollout_data_curr_ref)':
        'await measured("critic_train", critic_model.train(rollout_id, rollout_data_curr_ref), rollout_id=rollout_id)',
        'await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)':
        'await measured("actor_train", actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values), rollout_id=rollout_id)',
    }
    for before, after in replacements.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    compile(source, str(destination), 'exec')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as f:
        f.write(source)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                       fromfile='train_async.py', tofile='train_async_ppo.py'))
    destination.with_suffix('.patch').write_text(patch)
    manifest = {'base_miles_sha': '70b89e11770fc9bac984e22cfff89c51cca44203',
                'base_driver_sha256': BASE_SHA, 'driver_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'scheduling': 'native one-batch-ahead, not fully-async persistent-worker mode',
                'max_policy_lag_updates': 1, 'initial_weight_version': 1,
                'lifecycle_fixes': ['retain actor CPU parameter backup', 'resident actor broadcast'],
                'upstream_unchanged': True}
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    prepare(Path(os.environ.get('MILES_SOURCE_ROOT', '/campaign/miles')),
            Path(os.environ.get('MILES_PATCH_OUTPUT_DIR', str(Path(__file__).parent)))/'train_async_ppo.py')
