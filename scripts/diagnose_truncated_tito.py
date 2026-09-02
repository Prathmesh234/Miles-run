"""Replay saved Qwen session comparisons on CPU; never mutate training tokens."""
import argparse
import inspect
import json

from evidence import Run, atomic


def validate_candidate(root, source):
    import copy
    import hashlib
    import json
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    import torch
    from miles.utils.types import Sample

    module = ModuleType('posttrainingx_tito_candidate')
    exec(compile(source, 'posttrainingx_local_agent.py', 'exec'), module.__dict__)
    args = SimpleNamespace(n_samples_per_prompt=8, tito_model='qwen36', hf_checkpoint='/model',
        chat_template_path='/miles-source/miles/utils/chat_template_utils/templates/qwen3.6_fixed.jinja',
        apply_chat_template_kwargs={'preserve_thinking': True})
    samples, corrections = [], []
    for path in sorted(Path(root).glob('*.json')):
        group = [Sample.from_dict(s) for s in json.loads(path.read_text())['samples']]
        originals = [copy.deepcopy(s.to_dict()) for s in group]
        assert module._validate_group(args, group).keep
        for sample, original in zip(group, originals, strict=True):
            result = sample.to_dict()
            correction = result['metadata'].pop('posttrainingx_tito_comparison', None)
            assert result == original, 'Candidate changed native trajectory fields.'
            if correction:
                corrections.append({'index': sample.index, 'comparison': correction})
        samples.extend(group)
    base = next(s for s in samples if s.index == 11)
    tito = module._qwen_comparator(args.hf_checkpoint, args.chat_template_path,
                                   json.dumps(args.apply_chat_template_kwargs, sort_keys=True))
    expected = tito.apply_chat_template(base.metadata['messages'], add_generation_prompt=False, tokenize=True)
    end_id = tito.tokenizer.convert_tokens_to_ids('<|im_end|>')
    cases = []
    for fault in ('missing_internal_end', 'changed_system', 'extra_end', 'not_length', 'completed', 'diverged_native_ids', 'changed_verification'):
        sample = copy.deepcopy(base)
        if fault == 'missing_internal_end':
            sample.tokens.pop(sample.tokens.index(end_id))
        elif fault == 'changed_system':
            sample.tokens[1] = tito.tokenizer.encode('corrupted', add_special_tokens=False)[0]
        elif fault == 'extra_end':
            sample.tokens.insert(2, end_id)
        elif fault == 'not_length':
            sample.metadata['agent_metrics']['end_reason'] = 'max_turns'
        elif fault == 'completed':
            sample.status = Sample.Status.COMPLETED
        if fault in ('missing_internal_end', 'changed_system', 'extra_end'):
            sample.metadata['accumulated_token_ids'] = list(sample.tokens)
            sample.metadata['tito_session_mismatch'] = [m.to_dict() for m in tito.create_comparator().compare_sequences(expected, sample.tokens)]
        if fault == 'diverged_native_ids':
            sample.tokens[0] += 1
        if fault == 'changed_verification':
            sample.metadata['tito_session_mismatch'][0]['detail'] = 'altered'
        try:
            module._validate_truncated_tito(args, sample)
        except RuntimeError as exc:
            cases.append({'fault': fault, 'rejected': True, 'error': str(exc)})
        else:
            raise AssertionError('Candidate silently accepted ' + fault)
    assert len(samples) == 32 and len(corrections) == 3 and torch.cuda.device_count() == 0
    return {'schema_version': 1, 'candidate_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'samples_unchanged': len(samples), 'corrections': corrections, 'negative_controls': cases,
            'cuda_device_count': 0, 'findings': [],
            'scope': 'Pinned CPU tokenizer and actual candidate gate on saved native groups. All sample fields unchanged except new audit metadata. No GPU optimizer or resume qualification.'}


def replay(root):
    import copy
    import hashlib
    import json
    from pathlib import Path
    import torch
    from miles.utils.processing_utils import load_tokenizer
    from miles.utils.chat_template_utils.tito_tokenizer import Qwen36TITOTokenizer

    tokenizer = load_tokenizer('/model', trust_remote_code=True,
        chat_template_path='/miles-source/miles/utils/chat_template_utils/templates/qwen3.6_fixed.jinja')
    tito = Qwen36TITOTokenizer(tokenizer)
    comparator = tito.create_comparator()
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    findings, rows = [], []
    for path in sorted(Path(root).glob('*.json')):
        for sample in json.loads(path.read_text())['samples']:
            original = copy.deepcopy(sample)
            expected = tito.apply_chat_template(sample['metadata']['messages'], add_generation_prompt=False, tokenize=True)
            actual = sample['metadata']['accumulated_token_ids']
            mismatch = [m.to_dict() for m in comparator.compare_sequences(expected, actual)]
            stored = sample['metadata']['tito_session_mismatch']
            row = {'index': sample['index'], 'status': sample['status'], 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'stored_mismatch_types': [m['type'] for m in stored],
                   'replayed_mismatch': mismatch, 'stored_reproduced': mismatch == stored,
                   'sample_equals_accumulated_ids': sample['tokens'] == actual}
            if mismatch != stored:
                findings.append('Stored mismatch not exactly reproduced: ' + str(sample['index']))
            if stored:
                while expected and expected[-1] in tito.trailing_token_ids:
                    expected.pop()
                if not expected or expected[-1] != eos or actual[-1] == eos:
                    findings.append('Not solely an unclosed final message candidate: ' + str(sample['index']))
                else:
                    # Diagnostic comparison only. Do not alter sample IDs, mask, logprobs or rewards.
                    after = comparator.compare_sequences(expected[:-1], actual)
                    row['comparison_without_expected_closer'] = [m.to_dict() for m in after]
                    row['remaining_strict_types'] = [m.type.value for m in after if m.type.value != 'assistant_text']
                    row['literal_prefix_equal_after_closer_removal'] = expected[:-1] == actual
                    if row['remaining_strict_types']:
                        findings.append('Strict mismatch remains after removing expected closer: ' + str(sample['index']))
            if sample != original:
                raise AssertionError('Diagnostic mutated native evidence.')
            rows.append(row)
    if torch.cuda.device_count() != 0 or len(rows) != 32:
        findings.append('Expected CPU-only replay of32 saved samples.')
    return {'schema_version': 1, 'cuda_device_count': torch.cuda.device_count(), 'samples': rows, 'findings': findings,
            'scope': 'Read-only exact pinned tokenizer/comparator replay. No optimizer, no acceptance relaxation, no resume or quality claim.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    ap.add_argument('--candidate', action='store_true')
    args = ap.parse_args()
    run = Run(args.run_dir)
    label = ('tito-candidate-validation' if args.candidate else 'truncated-tito-replay') + f'-v{args.attempt}'
    phase = run.phase('02-' + label)
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    inner = inspect.getsource(replay) + '\nimport json\nprint(json.dumps(replay("/samples")))\n'
    if args.candidate:
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'vendor/miles/examples/experimental/openenv/posttrainingx_local_agent.py').read_text()
        atomic(phase.path / 'candidate.py', source)
        inner = inspect.getsource(validate_candidate) + '\nimport json\nprint(json.dumps(validate_candidate("/samples", ' + repr(source) + ')))\n'
    outer = '''import pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);code=root/'provenance/sync-grpo-code-v6'
sys.path.insert(0,str(code));from enroot_run_config import prepare
runtime=root/('images/'+sys.argv[3]);runtime.mkdir(exist_ok=False)
env=prepare(runtime);env['NVIDIA_VISIBLE_DEVICES']='void'
cmd=['enroot','start','--pid','--ipc','--rw','--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/miles-source','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1']
for source,target in [(root/'provenance/sync-grpo-source-v6/miles','/miles-source'),(root/'models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0','/model'),(root/'training/sync-grpo-v6/dump_details/qualification-groups','/samples')]:cmd+=['--mount',str(source)+':'+target+':none:bind,ro,x-create=dir']
cmd += [str(root/'images/enroot-import-v2/miles-amd64.sqsh'),'python3','-c',sys.argv[2]]
raise SystemExit(subprocess.call(cmd,env=env))
'''
    rc, out, _ = phase.command(['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec',
        'slurm-worker-gpu-nodes-0', '--', 'python3', '-c', outer, remote, inner, label], timeout=120)
    data = json.loads(out.splitlines()[-1]) if not rc else {'findings': ['Pinned CPU replay failed.']}
    atomic(phase.path / 'result.json', data)
    phase.finish('fail' if data['findings'] else 'ok', metadata=data,
                 failure_summary='; '.join(data['findings']) or None, refresh=False)
    print(json.dumps({'findings': data['findings'], 'samples': len(data.get('samples', []))}))
    return int(bool(data['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
