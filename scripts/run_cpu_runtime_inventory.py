"""CPU-only inspection using an explicit run-scoped Enroot hook fix."""
import argparse
import json
import os
from pathlib import Path
import sys

from enroot_run_config import prepare
from evidence import Run, atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase('00-pinned-image-cpu-runtime-v2')
    root = run.root / 'images/runtime-cpu-inventory-v2'
    try:
        root.mkdir(exist_ok=False)
        env = prepare(root)
    except (OSError, ValueError) as exc:
        phase.finish('fail', failure_summary='Run-scoped Enroot configuration preparation failed: ' + str(exc), refresh=False)
        return 1
    os.environ.update({k: v for k, v in env.items() if k.startswith('ENROOT_')})
    image = run.root / 'images/enroot-import-v2/miles-amd64.sqsh'
    code = Path(__file__).resolve().parent
    rc, out, _ = phase.command(['enroot', 'start', '--net', '--pid', '--ipc', '--env',
        'NVIDIA_VISIBLE_DEVICES=void', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--mount',
        str(code) + ':/ptx:none:bind,ro', str(image), 'python3', '/ptx/runtime_inventory.py'], timeout=180)
    errors = []
    if rc:
        errors.append('Enroot startup/package inspection failed with exit code ' + str(rc))
    else:
        try:
            data = json.loads(out)
            atomic(phase.path / 'packages.json', data)
        except ValueError as exc:
            errors.append('Runtime did not emit a single valid inventory JSON: ' + str(exc))
    phase.finish('fail' if errors else 'ok', failure_summary='; '.join(errors) or None,
        metadata={'scope': 'CPU-only pinned image inspection; no GPU/model/policy/optimizer step.',
                  'artifacts': [str(root.relative_to(run.root)), str(phase.path.relative_to(run.root)) + '/packages.json']}, refresh=False)
    print(json.dumps({'status': 'fail' if errors else 'ok', 'findings': errors}), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
