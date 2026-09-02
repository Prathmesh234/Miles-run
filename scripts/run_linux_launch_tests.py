"""Execute a pinned CPU-only test container without changing cluster settings."""
import argparse
import json
from pathlib import Path
import sys

from evidence import Run, atomic
from stage_linux_launch_tests import IMAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--kubeconfig', required=True)
    ap.add_argument('--phase-name', default='00-miles-linux-launch-suite')
    ap.add_argument('--input-dir-name', default='linux-launch-tests-v1')
    ap.add_argument('--source-dir-name', default='linux-launch-tests-v1')
    args = ap.parse_args()
    run = Run(args.run_dir)
    if any(Path(n).name != n or n in ('', '.', '..') for n in (args.input_dir_name, args.source_dir_name)):
        raise ValueError('Input names must be single directory components.')
    phase = run.phase(args.phase_name)
    remote = '/shared/posttrainingx/runs/vultr-b200-slurm/' + run.root.name
    inputs = remote + '/provenance/' + args.input_dir_name
    source = remote + '/provenance/' + args.source_dir_name
    outputs = remote + '/tests/' + phase.name + '/logs/container'
    container = 'ptx-' + phase.name + '-' + run.root.name
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '--request-timeout=0', '-n', 'slurm',
              'exec', '-i', 'slurm-worker-gpu-nodes-0', '--']
    guard = """from pathlib import Path
import json,shutil,subprocess,sys
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2])
if not out.is_relative_to(root) or not (root/'run.json').is_file():raise ValueError('Invalid run output path.')
docker_root=subprocess.check_output(['docker','info','--format','{{.DockerRootDir}}'],text=True).strip()
if shutil.disk_usage(docker_root).free < 8*1024**3:raise ValueError('Docker cache has less than 8 GiB free.')
if shutil.disk_usage(root).free < 10*1024**3:raise ValueError('Shared storage has less than 10 GiB free.')
out.mkdir(parents=True,exist_ok=False)
print(json.dumps({'docker_root':docker_root,'docker_free_bytes':shutil.disk_usage(docker_root).free,'shared_free_bytes':shutil.disk_usage(root).free}))
"""
    code, _, _ = phase.command(worker + ['python3', '-c', guard, remote, outputs], timeout=45)
    if code:
        phase.finish('fail', failure_summary='The CPU container free-space or output guard failed.')
        return 1
    code, _, _ = phase.command(worker + ['docker', 'pull', '--platform', 'linux/amd64', IMAGE], timeout=180)
    if code:
        phase.finish('fail', failure_summary='The pinned CPU test image could not be pulled.')
        return 1
    argv = worker + ['docker', 'run', '--name', container,
        '--label', 'posttrainingx.run_id=' + run.root.name, '--label', 'posttrainingx.role=launcher-cpu-tests',
        '--runtime=runc', '--network=bridge', '--cpus=4', '--memory=8g', '--pids-limit=512',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
        '--mount', f'type=bind,source={inputs},target=/input,readonly',
        '--mount', f'type=bind,source={source},target=/source,readonly',
        '--mount', f'type=bind,source={outputs},target=/artifacts',
        IMAGE, 'timeout', '600', 'python', '/input/runner.py']
    code, _, _ = phase.command(argv, timeout=660)
    inspect_code, inspect_text, _ = phase.command(worker + ['docker', 'inspect', container], timeout=30)
    state = json.loads(inspect_text)[0]['State'] if not inspect_code else {}
    # Do not restart or remove a container whose completion is ambiguous.
    if state.get('Running'):
        phase.finish('fail', failure_summary='Client observation ended while the named CPU test container is still running. Inspect this container before proceeding.',
                     metadata={'container': container, 'state': state, 'image': IMAGE})
        return 1
    fetch = """import json
from pathlib import Path
p=Path(__import__('sys').argv[1]);print(json.dumps({f.name:f.read_text() for f in p.iterdir() if f.is_file()}))
"""
    fetch_code, text, _ = phase.command(worker + ['python3', '-c', fetch, outputs], timeout=30)
    if not fetch_code:
        for name, data in json.loads(text).items():
            atomic(phase.path / 'logs/container' / name, data)
    cleanup_code = None
    if not code and not inspect_code and state.get('ExitCode') == 0 and not fetch_code:
        cleanup_code, _, _ = phase.command(worker + ['docker', 'rm', container], timeout=30)
    errors = code or inspect_code or fetch_code or cleanup_code or state.get('ExitCode') or int(not state)
    phase.finish('fail' if errors else 'ok',
                 failure_summary='The CPU launcher test or evidence/cleanup gate failed; the named container is retained on test failure.' if errors else None,
                 metadata={'image': IMAGE, 'container': container, 'state': state,
                    'scope': 'Full launch-script directories plus command and host-map tests; global GPU fixtures excluded with --noconftest.',
                    'cleanup_exit_code': cleanup_code, 'retained_for_diagnosis': bool(errors),
                    'artifacts': [str((phase.path / 'logs/container').relative_to(run.root))]})
    print(json.dumps({'phase': phase.name, 'status': 'fail' if errors else 'ok', 'container': container}))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
