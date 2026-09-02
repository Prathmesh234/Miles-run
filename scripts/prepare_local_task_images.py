"""Build pinned local policy/grader images; never copy tests into policy images.

The initial file-only runtime subset is the first four pre-registered training
IDs, plus the separate runtime-validation task. No reward is used for selection.
The grader pre-installs its exact Python/pytest toolchain. Its derived test.sh
removes only network-based dependency installation, preserving the pytest and
verdict block verbatim. Original and derived harness hashes are retained.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import socket
import traceback

from evidence import Run, atomic, metric, sha256


TASK_IDS = ['task_00000', 'task_06652', 'task_14118', 'task_10753', 'task_09467']
MANIFEST_SHA = 'a61bd1bfa37d60325df6bb4b448c2961cf0113750aa8144688f8cbd6837195eb'


def offline_harness(original):
    """Known setup-only adaptation, not a fallback after failed grading."""
    lines = original.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == '# Run pytest tests']
    if len(matches) != 1:
        raise ValueError('Unrecognized harness boundary; no automatic rewrite.')
    setup, scoring = ''.join(lines[:matches[0]]), ''.join(lines[matches[0]:])
    if ('https://astral.sh/uv/0.9.5/install.sh' not in setup
            or 'pytest==8.4.1' not in scoring or 'pytest-json-ctrf==0.3.5' not in scoring
            or '/tests/test_outputs.py -rA' not in scoring or '/logs/verifier/reward.txt' not in scoring):
        raise ValueError('Unrecognized pinned toolchain or scoring block.')
    # Refuse to drop executable statements beyond the reviewed installer/cwd guard.
    allowed = re.compile(r'^(?:#.*|\s*|#!/bin/bash|apt-get update|apt-get install -y curl(?: libxml2-utils)?|'
                         r'curl -LsSf https://astral\.sh/uv/0\.9\.5/install\.sh \| sh|'
                         r'source \$HOME/\.local/bin/env|if \[ "\$PWD" = "/" \]; then|'
                         r'\s*echo "Error:.*"|\s*exit 1|fi)$')
    if any(not allowed.fullmatch(line) for line in setup.splitlines()):
        raise ValueError('Unknown setup operation; preserve it and require review.')
    header = ('#!/bin/bash\n# PostTrainingX offline dependency packaging v1.\n'
              '# Original scoring block below is byte-preserved.\n'
              'export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never UV_PYTHON_PREFERENCE=only-managed\n'
              'if [ "$PWD" = "/" ]; then exit 1; fi\n')
    return header + scoring


def pin_dockerfile(text, bases):
    found = re.findall(r'^FROM ([^\s]+)\s*$', text, re.M)
    if len(found) != 1 or found[0] not in bases:
        raise ValueError('Expected one pre-resolved base image.')
    return re.sub(r'^FROM [^\s]+\s*$', 'FROM ' + bases[found[0]], text, count=1, flags=re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'02-local-task-images-v{args.attempt}')
    parent = run.root / f'environments/local-file-runtime-v{args.attempt}'
    result = {'scope': __doc__, 'findings': [], 'task_ids': TASK_IDS, 'images': [], 'bases': {}}

    def command(argv, timeout=300):
        if shutil.disk_usage(run.root).free < 128 * 1024**3 or shutil.disk_usage('/var/lib/docker').free < 128 * 1024**3:
            raise ValueError('Shared/local free-space reserve below 128 GiB.')
        rc, out, _ = phase.command(argv, timeout=timeout)
        if rc:
            raise RuntimeError('Image preparation command failed; inspect captured command and stderr.')
        return out

    try:
        parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        source = run.root / 'environments/terminal-lego-controller-v3'
        manifest_path = source / 'files.manifest.json'
        if sha256(manifest_path) != MANIFEST_SHA:
            raise ValueError('Pinned task materialization manifest changed.')
        split = json.loads((Path(__file__).parent / 'terminal-lego-subset.json').read_text())
        if TASK_IDS != split['runtime_validation_task_ids'] + split['training_task_ids'][:4]:
            raise ValueError('Initial task IDs differ from the pre-registered prefix.')
        files = json.loads(manifest_path.read_text())['files']
        for row in files:
            if Path(row['path']).parts[0] not in TASK_IDS:
                continue
            path = source / 'source' / row['path']
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError('Pinned task source changed: ' + row['path'])
        for tag in ['ubuntu:22.04', 'node:20-slim', 'python:3.13-slim-bookworm', 'ghcr.io/astral-sh/uv:0.9.5']:
            command(['docker', 'pull', '--platform=linux/amd64', tag], timeout=300)
            info = json.loads(command(['docker', 'image', 'inspect', tag]))[0]
            digests = info.get('RepoDigests', [])
            if len(digests) != 1 or info['Architecture'] != 'amd64' or '@sha256:' not in digests[0]:
                raise ValueError('Base image resolution is ambiguous or not amd64.')
            result['bases'][tag] = digests[0]
            atomic(parent / 'base-images.json', result['bases'])
        for task in TASK_IDS:
            task_source = source / 'source' / task
            context = parent / 'builds' / task
            shutil.copytree(task_source / 'environment', context)
            original = (context / 'Dockerfile').read_text()
            atomic(context / 'Dockerfile.original', original)
            atomic(context / 'Dockerfile', pin_dockerfile(original, result['bases']))
            # Build context consists of environment/ only; no tests or solutions.
            if any(p.parts[-1] in ('tests', 'solution') for p in context.rglob('*')):
                raise ValueError('Forbidden asset in policy build context.')
            tag = 'posttrainingx-local/' + run.root.name + '-' + task + f'-v{args.attempt}'
            command(['docker', 'build', '--pull=false', '--network=default', '--label', 'posttrainingx.run=' + run.root.name,
                     '--tag', tag, str(context)], timeout=300)
            policy = json.loads(command(['docker', 'image', 'inspect', tag]))[0]
            grader_context = parent / 'grader-builds' / task
            grader_context.mkdir(parents=True)
            # No task test files enter this image. Runtime grading stages them
            # only after the policy container has been permanently sealed.
            dockerfile = ('FROM ' + tag + '\n'
                'COPY --from=' + result['bases']['ghcr.io/astral-sh/uv:0.9.5'] + ' /uv /uvx /usr/local/bin/\n'
                'RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*\n'
                'ENV UV_PYTHON_PREFERENCE=only-managed\n'
                'RUN uv python install 3.13.7 && uvx --python 3.13.7 --with pytest==8.4.1 --with pytest-json-ctrf==0.3.5 pytest --version\n'
                'ENV UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never\n')
            atomic(grader_context / 'Dockerfile', dockerfile)
            grader_tag = tag + '-grader'
            command(['docker', 'build', '--pull=false', '--network=default', '--label', 'posttrainingx.run=' + run.root.name,
                     '--tag', grader_tag, str(grader_context)], timeout=360)
            grader = json.loads(command(['docker', 'image', 'inspect', grader_tag]))[0]
            original_harness = task_source / 'tests/test.sh'
            runtime_harness = parent / 'harness' / task / 'test.sh'
            atomic(runtime_harness, offline_harness(original_harness.read_text()))
            row = {'task_id': task, 'policy_image_id': policy['Id'], 'grader_image_id': grader['Id'],
                   'policy_tag': tag, 'grader_tag': grader_tag, 'hostname': socket.gethostname(),
                   'original_harness_sha256': sha256(original_harness), 'offline_harness_sha256': sha256(runtime_harness),
                   'runtime_harness_relpath': str(runtime_harness.relative_to(run.root)),
                   'source_relpath': str(task_source.relative_to(run.root)),
                   'profile': 'file-only-v1: read-only root; bounded tmpfs task tree; no network; separate sealed grader',
                   'reference_and_policy_validation': 'not_started'}
            result['images'].append(row)
            atomic(parent / 'images.partial.json', result)
            print(json.dumps({'event': 'task_images_built', 'task_id': task, 'policy_image_id': policy['Id']}), flush=True)
        atomic(parent / 'images.json', result)
    except Exception as exc:
        result['findings'].append(str(exc))
        atomic(phase.path / 'exception.txt', traceback.format_exc())
    atomic(phase.path / 'result.json', result)
    phase.finish('fail' if result['findings'] else 'ok', failure_summary='; '.join(result['findings']) or None,
                 metadata=result, results=[metric('task_image_pairs_built', len(result['images']), 'count', socket.gethostname())], refresh=False)
    print(json.dumps({'findings': result['findings'], 'image_pairs': len(result['images'])}), flush=True)
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
