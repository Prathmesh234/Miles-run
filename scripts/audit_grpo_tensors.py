"""Read-only native-sample -> trainer tensor audit in the pinned CPU runtime.

This checks data and GRPO advantages, not optimizer execution or held-out quality.
It intentionally supports the qualified CP1, one-step, unwhitened GRPO recipe.
"""
import argparse
import inspect
import json
from pathlib import Path

from evidence import Run, atomic


def audit_tensors(root, expected_ranks):
    import collections
    import hashlib
    import json
    from pathlib import Path
    import torch

    root = Path(root)
    findings, artifacts, batches = [], [], []
    native = {}

    def record(path):
        artifacts.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    for path in sorted((root / 'qualification-groups').glob('*.json')):
        record(path)
        for sample in json.loads(path.read_text())['samples']:
            if sample['index'] in native:
                raise ValueError('Ambiguous duplicate native sample index.')
            native[sample['index']] = sample

    for path in sorted((root / 'rollout_data').glob('*.pt')):
        record(path)
        rollout = torch.load(path, map_location='cpu', weights_only=False)
        rid = rollout['rollout_id']
        samples = {s['index']: s for s in rollout['samples']}
        if len(samples) != len(rollout['samples']):
            raise ValueError('Duplicate sample in rollout dump.')
        groups = collections.defaultdict(list)
        expected = {}
        zero_variance = 0
        for sample in samples.values():
            original = native[sample['index']]
            for key in ('tokens', 'response_length', 'rollout_log_probs', 'loss_mask', 'reward', 'group_index'):
                if sample[key] != original[key]:
                    raise ValueError(f'Native rollout field changed: {sample["index"]}/{key}')
            groups[sample['group_index']].append(sample)
        for group in groups.values():
            if len(group) != 8:
                raise ValueError('Expected complete eight-sample GRPO groups.')
            raw = torch.tensor([s['reward'] for s in group], dtype=torch.float32)
            centered = raw - raw.mean()
            std = raw.std()
            if std > 0:
                centered = centered / (std + 1e-6)
            else:
                zero_variance += 1
            expected.update({s['index']: reward.item() for s, reward in zip(group, centered, strict=True)})

        ranks, seen, rows = [], [], []
        for train_path in sorted((root / 'train_data').glob(str(rid) + '_*.pt')):
            record(train_path)
            dump = torch.load(train_path, map_location='cpu', weights_only=False)
            if dump['rollout_id'] != rid or dump['cp_size'] != 1 or dump['cp_rank'] != 0:
                raise ValueError('Unsupported rollout or context-parallel identity.')
            ranks.append(dump['rank'])
            data = dump['rollout_data']
            # raw_reward is a global, unsharded list; do not zip it to local indices.
            if data['raw_reward'] != [s['reward'] for s in rollout['samples']]:
                raise ValueError('Global raw reward ordering differs from rollout dump.')
            for position, index in enumerate(data['sample_indices']):
                seen.append(index)
                sample = samples[index]
                for target, source, dtype in (
                    ('tokens', 'tokens', torch.int64),
                    ('loss_masks', 'loss_mask', torch.int32),
                    ('rollout_log_probs', 'rollout_log_probs', torch.float32),
                ):
                    actual = data[target][position]
                    reference = torch.tensor(sample[source], dtype=dtype)
                    if actual.dtype != dtype or not torch.equal(actual, reference):
                        raise ValueError(f'Native-to-trainer tensor mismatch: {index}/{target}')
                if data['rewards'][position] != expected[index]:
                    raise ValueError('Independent GRPO reward normalization differs.')
                response_length = sample['response_length']
                for key in ('advantages', 'returns', 'log_probs', 'ref_log_probs'):
                    tensor = data[key][position]
                    if tensor.shape != (response_length,) or not torch.isfinite(tensor).all():
                        raise ValueError(f'Nonfinite or misaligned trainer tensor: {index}/{key}')
                    if key in ('advantages', 'returns') and not torch.equal(
                        tensor, torch.full_like(tensor, expected[index])
                    ):
                        raise ValueError(f'Unwhitened GRPO advantage mismatch: {index}/{key}')
                if data['weight_versions'][position] != sample['weight_versions']:
                    raise ValueError('Sample policy versions changed before training.')
                rows.append({'sample_index': index, 'group_index': sample['group_index'],
                             'rank': dump['rank'], 'raw_reward': sample['reward'],
                             'normalized_reward': expected[index], 'response_length': response_length,
                             'sampled_loss_tokens': sum(sample['loss_mask'])})
        if sorted(ranks) != list(range(expected_ranks)):
            raise ValueError('Missing, duplicate, or unexpected trainer rank dumps.')
        if sorted(seen) != sorted(samples):
            raise ValueError('Trainer coverage has missing or duplicate samples.')
        batches.append({'rollout_id': rid, 'samples': rows, 'groups': len(groups),
                        'zero_variance_groups': zero_variance, 'trainer_ranks': sorted(ranks)})
    if not batches:
        findings.append('No finalized trainer batches available.')
    if torch.cuda.device_count() != 0:
        findings.append('Audit must not allocate GPUs.')
    return {'schema_version': 1, 'findings': findings, 'batches': batches, 'artifacts': artifacts,
            'native_samples': len(native), 'trained_input_samples': sum(len(b['samples']) for b in batches),
            'scope': 'Exact sampled IDs/masks and float32 logprobs through trainer inputs; independent n8 GRPO normalization and finite response-aligned tensors. Dumps are pre-optimizer, not proof of updates, gradients, complete episode accounting, or quality improvement.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--training-attempt', type=int, required=True)
    parser.add_argument('--audit-attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    label = f'grpo-tensor-audit-v{args.training_attempt}-a{args.audit_attempt}'
    phase = run.phase('02-' + label)
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    inner = inspect.getsource(audit_tensors) + '\nimport json\nprint(json.dumps(audit_tensors("/dumps",16)))\n'
    outer = '''import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); attempt=sys.argv[4]
code=root/('provenance/sync-grpo-code-v'+attempt)
sys.path.insert(0,str(code));from enroot_run_config import prepare
runtime=root/('images/'+sys.argv[3]);runtime.mkdir(exist_ok=False)
env=prepare(runtime);env['NVIDIA_VISIBLE_DEVICES']='void'
cmd=['enroot','start','--pid','--ipc','--rw','--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/miles-source','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1']
for source,target in [(root/('provenance/sync-grpo-source-v'+attempt+'/miles'),'/miles-source'),(root/('training/sync-grpo-v'+attempt+'/dump_details'),'/dumps')]:cmd+=['--mount',str(source)+':'+target+':none:bind,ro,x-create=dir']
cmd += [str(root/'images/enroot-import-v2/miles-amd64.sqsh'),'python3','-c',sys.argv[2]]
raise SystemExit(subprocess.call(cmd,env=env))
'''
    atomic(phase.path / 'audit-source.py', inner)
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', outer, remote, inner, label,
        str(args.training_attempt)], timeout=180)
    data = json.loads(out.splitlines()[-1]) if not rc else {'findings': ['Pinned CPU tensor audit failed; inspect retained stderr.']}
    atomic(phase.path / 'result.json', data)
    phase.finish('fail' if data['findings'] else 'ok', metadata=data,
                 failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({k: v for k, v in data.items() if k not in ('artifacts', 'batches')}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
