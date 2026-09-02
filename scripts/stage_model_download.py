"""Stage a committed downloader and run a bounded, non-GPU model pull."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from evidence import Run, sha256
from submit_native_preflight import BOOTSTRAP, batches, entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    repo = Path(__file__).resolve().parents[1]
    phase = run.phase('00-qwen-model-download-dispatch')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        phase.finish('fail', failure_summary='Commit the downloader before staging it.')
        return 1
    prefix = 'provenance/model-pull-code/'
    files = {prefix+name: entry((repo/'scripts'/name).read_bytes())
             for name in ['pull_pinned_model.py', 'evidence.py']}
    files['provenance/qwen-file-manifest.json'] = entry((run.root/'provenance/qwen-file-manifest.json').read_bytes())
    files[prefix+'source-revision.txt'] = entry((revision+'\n').encode())
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=0', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    common = {'root': remote, 'create': False, 'manifest_sha256': sha256(run.root/'run.json')}
    for data in batches(common, files, limit=128*1024):
        code, _, _ = phase.command(worker+['python3', '-c', BOOTSTRAP], stdin=data, timeout=45)
        if code:
            phase.finish('fail', failure_summary='Model downloader staging failed. Inspect staged files before retry.')
            return 1
    code, _, _ = phase.command(worker+[
        'timeout', '1800', 'python3', remote+'/'+prefix+'pull_pinned_model.py',
        '--run-dir', remote, '--manifest', remote+'/provenance/qwen-file-manifest.json', '--workers', '4',
    ], timeout=1830)
    phase.finish('fail' if code else 'ok',
        failure_summary='Pinned model pull failed or observation timed out. Inspect the same process and preserved partial files before any retry.' if code else None,
        metadata={'source_revision': revision, 'shared_run_dir': remote,
                  'remote_phase': '00-qwen-model-download', 'timeout_s': 1800, 'workers': 4,
                  'scope': 'Model provenance preparation only, no GPU or optimizer execution.'})
    print(json.dumps({'remote_phase': '00-qwen-model-download', 'exit_code': code}))
    return code


if __name__ == '__main__':
    sys.exit(main())
