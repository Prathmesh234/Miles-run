"""Pinned HF-to-Megatron conversion inside an isolated, eight-GPU container."""
import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

from evidence import Run, atomic, sha256, utcnow


MILES_SHA = 'b61dbe83ee815412b72c84ed367ffd329d7922d4'
MEGATRON_SHA = '8c1e05747eb612b382df2632783df5c83a853646'


def torchrun_prefix(ranks):
    return [sys.executable, '-m', 'torch.distributed.run', '--rdzv-backend=static',
            '--master-addr=127.0.0.1', '--master-port=31873', '--nnodes=1', '--node-rank=0',
            '--nproc-per-node=' + str(ranks)]


def conversion_command(source, model, destination, model_args):
    args = shlex.split(model_args)
    if args.count('--mtp-num-layers') != 1 or args[args.index('--mtp-num-layers') + 1] != '1':
        raise ValueError('The pinned conversion recipe must preserve its one MTP layer.')
    return [*torchrun_prefix(8), str(source / 'tools/convert_hf_to_torch_dist.py'), *args,
            '--hf-checkpoint', str(model), '--save', str(destination)]


def checkpoint_files(root):
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Non-regular file in converted checkpoint.')
        if path.is_file():
            files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                          'sha256': sha256(path)})
    if not files or not any(f['bytes'] > 1024**2 for f in files):
        raise ValueError('No substantial converted checkpoint payload.')
    return files


def validate_imports(run, source, model):
    import torch
    import miles
    import megatron.core
    if Path(miles.__file__).resolve() != source / 'miles/__init__.py':
        raise ValueError('Imported bundled image Miles instead of the pinned campaign source.')
    if torch.cuda.device_count() != 8:
        raise ValueError('Conversion requires exactly eight allocated CUDA devices.')
    physical = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                       '--format=csv,noheader,nounits'], text=True)
    actual = {r[1] for r in csv.reader(physical.splitlines(), skipinitialspace=True)}
    expected = {g['uuid'] for g in json.loads((run.root / 'inventory/gpu.values.json').read_text())['gpus']
                if g['hostname'] == socket.gethostname()}
    if len(expected) != 8 or actual != expected:
        raise ValueError('Container physical UUIDs do not reconcile to the frozen allocation.')
    for config in ('config.json', 'tokenizer_config.json'):
        if json.loads((model / config).read_text()).get('auto_map'):
            raise ValueError('Dynamic model/tokenizer code requires a separate provenance review.')
    mega = subprocess.check_output(['git', '-C', '/root/Megatron-LM', 'rev-parse', 'HEAD'], text=True).strip()
    if mega != MEGATRON_SHA:
        raise ValueError('Megatron image revision changed.')
    runpy.run_path(str(source / 'tools/convert_hf_to_torch_dist.py'), run_name='posttrainingx_import_probe')
    import mbridge
    from importlib.metadata import version
    return {'miles_git_sha': MILES_SHA, 'miles_import': miles.__file__, 'megatron_git_sha': mega,
            'megatron_import': megatron.core.__file__, 'mbridge_import': mbridge.__file__,
            'mbridge_version': version('mbridge'), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu_uuids': sorted(actual)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-model-conversion-child-v{args.attempt}')
    source, model = Path('/miles-source'), Path('/model')
    parent = run.root / 'models'
    final = parent / f'qwen3.6-35b-a3b-torch-dist-v{args.attempt}'
    partial = final.with_name(final.name + '.partial')
    process, results, errors, handles = None, {}, [], []
    started_at, timed_out = None, False
    try:
        if partial.exists() or final.exists():
            raise ValueError('Checkpoint destination already exists; no overwrite or implicit resume.')
        if shutil.disk_usage(parent).free < 512 * 1024**3:
            raise ValueError('Conversion requires at least 512 GiB free before starting.')
        results['imports'] = validate_imports(run, source, model)
        atomic(phase.path / 'import-probe.json', results['imports'])
        rc, _, _ = phase.command(torchrun_prefix(2) + [str(Path(__file__).with_name('rendezvous_probe.py'))], timeout=120)
        if rc:
            raise RuntimeError('Static loopback rendezvous probe failed before conversion.')
        from miles.utils.external_utils.model_args_utils import load_model_args
        command = conversion_command(source, model, partial, load_model_args('qwen3.6-35B-A3B'))
        results['command'] = command
        results['conversion_topology'] = 'Cookbook conversion auto-selects PP8 on one 8-GPU node; training remains TP1/EP8/PP1.'
        atomic(phase.path / 'conversion-command.json', results)
        partial.mkdir(exist_ok=False)
        started = time.monotonic()
        started_at = utcnow()
        print(json.dumps({'command': command, 'started_at': utcnow()}), flush=True)
        handles = [(phase.path / ('logs/conversion.' + suffix)).open('x') for suffix in ('out', 'err')]
        process = subprocess.Popen(command, stdout=handles[0], stderr=handles[1], start_new_session=True)
        with (phase.path / 'space-guard.jsonl').open('x') as guard:
            while process.poll() is None:
                free = shutil.disk_usage(parent).free
                guard.write(json.dumps({'time': utcnow(), 'monotonic_s': time.monotonic(), 'free_bytes': free}) + '\n')
                guard.flush()
                if free < 128 * 1024**3:
                    raise RuntimeError('Conversion free-space guard reached 128 GiB reserve.')
                if time.monotonic() - started > 1200:
                    raise TimeoutError('Conversion exceeded its 20-minute execution budget.')
                if (run.root / f'control/model-conversion-v{args.attempt}.stop').exists():
                    raise RuntimeError('Operator/controller stop marker received; preserve partial conversion.')
                time.sleep(2)
        results['conversion_duration_s'] = time.monotonic() - started
        results['conversion_ended_at'] = utcnow()
        results['conversion_exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError('Pinned conversion exited with ' + str(process.returncode))
        if (partial / 'latest_checkpointed_iteration.txt').read_text().strip() != 'release':
            raise ValueError('Conversion release tracker is missing or incomplete.')
        metadata_paths = list(partial.rglob('.metadata'))
        if len(metadata_paths) != 1:
            raise ValueError('Expected exactly one distributed-checkpoint metadata file.')
        from torch.distributed.checkpoint import FileSystemReader
        metadata = FileSystemReader(metadata_paths[0].parent).read_metadata()
        keys = sorted(metadata.state_dict_metadata)
        mtp_keys = [key for key in keys if 'mtp' in key.lower()]
        if not mtp_keys:
            raise ValueError('Converted checkpoint has no MTP tensors.')
        atomic(phase.path / 'checkpoint-key-inventory.json', {'keys': keys, 'mtp_keys': mtp_keys})
        hashing = time.monotonic()
        files = checkpoint_files(partial)
        results.update(checkpoint_bytes=sum(f['bytes'] for f in files), checkpoint_files=len(files),
                       hash_duration_s=time.monotonic() - hashing, mtp_tensor_keys=len(mtp_keys))
        atomic(partial / 'conversion.manifest.json', dict(results, files=files,
               model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0'))
        atomic(partial / 'checksums.sha256', ''.join(f"{f['sha256']}  {f['path']}\n" for f in files))
        if final.exists():
            raise ValueError('Final destination appeared during conversion; no overwrite.')
        os.rename(partial, final)
        results['checkpoint_relpath'] = str(final.relative_to(run.root))
        results['manifest_sha256'] = sha256(final / 'conversion.manifest.json')
    except Exception as exc:
        errors.append(str(exc))
        timed_out = isinstance(exc, TimeoutError)
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            results['cleanup_exit_code'] = process.returncode
        for handle in handles:
            handle.close()
        if process is not None:
            phase.commands.append({'argv': command, 'started_at': started_at,
                'ended_at': results.get('conversion_ended_at', utcnow()),
                'duration_s': results.get('conversion_duration_s', time.monotonic() - started),
                'exit_code': 124 if timed_out else process.returncode,
                'timeout': timed_out, 'stdout': str((phase.path / 'logs/conversion.out').relative_to(run.root)),
                'stderr': str((phase.path / 'logs/conversion.err').relative_to(run.root))})
            atomic(phase.path / 'logs/commands.json', phase.commands)
    results['findings'] = errors
    results['scope'] = 'MTP-preserving weight conversion only. Tensor parity, trainer forward/backward and resumable training remain unvalidated.'
    atomic(phase.path / 'conversion-result.json', results)
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None, metadata=results, refresh=False)
    print(json.dumps({'findings': errors, 'checkpoint_bytes': results.get('checkpoint_bytes')}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
