"""Resolve every additional clean-corpus base to an explicit Linux/amd64 digest.

Registry metadata only: no image builds, task execution, or task selection.
Existing qualified base digests are preserved rather than resolved again.
"""
import argparse
import json
from pathlib import Path
import re

from evidence import Run, atomic, sha256
from prepare_local_task_images import BASE_IMAGES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run = Run(args.run_dir)
    phase = run.phase(f'00-full-clean-base-image-pins-v{args.attempt}')
    audit_path = run.root / 'tests/02-clean-corpus-structural-audit-v1/result.json'
    audit = json.loads(audit_path.read_text())
    tags = sorted({tag for task in audit['tasks'] for tag in task['base_tags']})
    result = {'schema_version': 1, 'source_audit_sha256': sha256(audit_path),
              'bases': dict(BASE_IMAGES), 'new_resolutions': [], 'findings': [],
              'scope': __doc__}
    worker = ['kubectl', '--kubeconfig', args.kubeconfig, '-n', 'slurm', 'exec', 'slurm-worker-gpu-nodes-0', '--']
    try:
        for tag in tags:
            if tag in result['bases']:
                continue
            rc, out, _ = phase.command(worker + ['docker', 'buildx', 'imagetools', 'inspect', tag,
                                               '--format', '{{json .Manifest}}'], timeout=45)
            if rc:
                raise RuntimeError('Registry resolution failed: ' + tag)
            manifest = json.loads(out)
            repository = tag.rsplit(':', 1)[0]
            children = [m for m in manifest.get('manifests', [])
                        if m.get('platform', {}).get('os') == 'linux'
                        and m.get('platform', {}).get('architecture') == 'amd64']
            if len(children) == 1:
                digest = children[0]['digest']
            elif 'manifests' not in manifest:
                rc, config, _ = phase.command(worker + ['docker', 'buildx', 'imagetools', 'inspect',
                    repository + '@' + manifest['digest'], '--format', '{{json .Image}}'], timeout=45)
                image = json.loads(config) if not rc else {}
                if image.get('os') != 'linux' or image.get('architecture') != 'amd64':
                    raise ValueError('Single-manifest platform is not proven Linux/amd64: ' + tag)
                digest = manifest['digest']
            else:
                raise ValueError('Ambiguous Linux/amd64 platform: ' + tag)
            if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
                raise ValueError('Malformed registry digest: ' + tag)
            pinned = repository + '@' + digest
            result['bases'][tag] = pinned
            result['new_resolutions'].append({'original_tag': tag, 'pinned_linux_amd64_image': pinned,
                                              'registry_descriptor_digest': manifest['digest']})
            atomic(phase.path / 'pins.partial.json', result)
    except Exception as exc:
        result['findings'].append(str(exc))
    atomic(phase.path / 'result.json', result)
    if not result['findings']:
        lock = Path(__file__).resolve().parents[1] / 'locks/clean-corpus-base-images.json'
        if lock.exists():
            raise ValueError('Existing full-corpus base lock may not be silently repinned.')
        atomic(lock, result)
    phase.finish('fail' if result['findings'] else 'ok', metadata=result,
                 failure_summary='; '.join(result['findings']) or None, refresh=False)
    print(json.dumps({'findings': result['findings'], 'pinned_bases': len(result['bases'])}))
    return int(bool(result['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
