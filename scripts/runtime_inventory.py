"""Capture package/source provenance inside the pinned image without a GPU step."""
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


def safe_direct_url(text):
    data = json.loads(text)
    parts = urlsplit(data['url'])
    data['url'] = urlunsplit((parts.scheme, parts.hostname or '' if parts.netloc else '', parts.path, '', ''))
    return data


def main():
    packages = []
    for dist in metadata.distributions():
        item = {'name': dist.metadata['Name'], 'version': dist.version}
        direct = dist.read_text('direct_url.json')
        if direct:
            item['direct_url'] = safe_direct_url(direct)
        packages.append(item)
    modules, revisions = {}, {}
    for name in ('miles', 'sglang', 'megatron', 'ray', 'torch', 'verifiers'):
        spec = importlib.util.find_spec(name)
        modules[name] = {'origin': spec.origin, 'search_locations': list(spec.submodule_search_locations or [])} if spec else None
        if spec:
            for path in list(spec.submodule_search_locations or []):
                result = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True, text=True)
                if result.returncode == 0:
                    revisions[path] = result.stdout.strip()
    import torch
    output = {'python': sys.version, 'platform': platform.platform(), 'packages': sorted(packages, key=lambda p: p['name'].lower()),
              'modules': modules, 'git_revisions': revisions, 'torch': {'version': torch.__version__,
              'cuda_build': torch.version.cuda, 'nccl_version': torch.cuda.nccl.version()},
              'scope': 'CPU-only package and source inspection. No model, GPU workload, policy or optimizer execution.'}
    for name, command in {'cuda_toolkit': ['nvcc', '--version'], 'os_release': ['cat', '/etc/os-release']}.items():
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=15)
            output[name] = {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        except OSError as exc:
            output[name] = {'exit_code': 127, 'error': str(exc)}
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
