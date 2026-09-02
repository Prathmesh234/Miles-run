"""Write immutable phase evidence without a dependency on the GPU software stack."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 128 * 1024**2:
        raise RuntimeError('Evidence free-space guard requires 128 MiB.')
    payload = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric(name, value, unit, node=None, **labels):
    item = {'metric': name, 'value': value, 'unit': unit, 'labels': labels}
    if node is not None:
        item['node'] = node
    return item


def markdown(data):
    lines = [f"# {data['runner']}", '', f"Status: {data['status']}", '']
    for key in ('started_at', 'ended_at', 'duration_s', 'failure_summary', 'reason', 'log_relpath', 'log_sha256'):
        if key in data:
            lines += [f"{key}: `{data[key]}`", '']
    if data.get('results'):
        lines += ['| Metric | Value | Unit | Node | Labels |', '|---|---:|---|---|---|']
        for row in data['results']:
            cells = [row['metric'], row['value'], row['unit'], row.get('node', ''), json.dumps(row.get('labels', {}), sort_keys=True)]
            lines.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in cells) + ' |')
    lines += ['', '## Metadata', '', '```json', json.dumps(data.get('metadata', {}), indent=2, sort_keys=True), '```', '']
    return '\n'.join(lines)


class Run:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / 'run.json').is_file():
            raise ValueError('A run.json is required.')

    @classmethod
    def create(cls, root, metadata):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        for name in ('inventory', 'tests', 'telemetry', 'rl', 'reports', 'provenance'):
            (root / name).mkdir()
        atomic(root / 'run.json', {'schema_version': 1, 'run_id': root.name, 'cluster_slug': 'vultr-b200-slurm', 'status': 'in_progress', 'started_at': utcnow(), 'metadata': metadata})
        run = cls(root)
        run.refresh()
        return run

    def phase(self, name):
        return Phase(self, name)

    def refresh(self):
        phases = []
        for path in sorted((self.root / 'tests').glob('*/*.json')):
            if not re.search(r'\.(values|failed|skipped)\.json$', path.name):
                continue
            d = json.loads(path.read_text())
            phases.append({'phase': d['runner'], 'status': d['status'], 'duration_s': d['duration_s'],
                           'result_file': str(path.relative_to(self.root)), 'log': d.get('log_relpath'),
                           'exit_code': d.get('exit_code', 0 if d['status'] == 'ok' else None),
                           'artifacts': d.get('metadata', {}).get('artifacts', [])})
        counts = {s: sum(p['status'] == s for p in phases) for s in ('ok', 'fail', 'skip')}
        summary = {'schema_version': 1, 'run_id': self.root.name, 'phases': phases, 'counts': counts}
        atomic(self.root / 'sweep.summary.json', summary)
        atomic(self.root / 'run.md', '# ' + self.root.name + '\n\n' +
               'This campaign is incomplete until all scientific and infrastructure gates are verified.\n\n' +
               '| Phase | Status | Duration (s) | Result |\n|---|---|---:|---|\n' +
               '\n'.join(f"| {p['phase']} | {p['status']} | {p['duration_s']:.3f} | [{p['result_file']}]({p['result_file']}) |" for p in phases) + '\n')
        hashes = []
        for p in sorted(self.root.rglob('*')):
            if p.is_symlink():
                raise ValueError(f'Symlink is not permitted in the evidence bundle: {p}')
            if p.is_file() and p.name != 'checksums.sha256' and not p.name.startswith('.'):
                hashes.append(f'{sha256(p)}  {p.relative_to(self.root)}')
        atomic(self.root / 'checksums.sha256', '\n'.join(hashes) + '\n')


class Phase:
    def __init__(self, run, name):
        if not re.fullmatch('[a-z0-9][a-z0-9-]*', name):
            raise ValueError('Invalid phase name.')
        self.run = run
        self.name = name
        self.path = run.root / 'tests' / name
        self.path.mkdir(exist_ok=False)
        (self.path / 'logs').mkdir()
        self.started = utcnow()
        self.monotonic = time.monotonic()
        self.commands = []
        self.finished = False

    def command(self, argv, timeout=60, stdin=None):
        """argv and stdin must contain no credential values. Stdin is hashed, not logged."""
        index = len(self.commands)
        start = utcnow()
        t0 = time.monotonic()
        expired = False
        try:
            p = subprocess.run(argv, input=stdin, text=True, capture_output=True, timeout=timeout)
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            def decode(x):
                return x.decode(errors='replace') if isinstance(x, bytes) else x or ''
            code, out, err, expired = 124, decode(e.stdout), decode(e.stderr), True
        except OSError as e:
            code, out, err = 127, '', str(e)
        prefix = self.path / 'logs' / f'{index:03d}'
        atomic(str(prefix) + '.out', out)
        atomic(str(prefix) + '.err', err)
        record = {'argv': argv, 'started_at': start, 'ended_at': utcnow(), 'duration_s': time.monotonic() - t0,
                  'exit_code': code, 'timeout': expired, 'stdout': str(prefix.relative_to(self.run.root)) + '.out',
                  'stderr': str(prefix.relative_to(self.run.root)) + '.err'}
        if stdin is not None:
            record['stdin_sha256'] = hashlib.sha256(stdin.encode()).hexdigest()
        self.commands.append(record)
        atomic(self.path / 'logs' / 'commands.json', self.commands)
        return code, out, err

    def finish(self, status, results=None, metadata=None, failure_summary=None, reason=None, exit_code=None, refresh=True):
        if self.finished or status not in ('ok', 'fail', 'skip'):
            raise ValueError('Invalid or repeated phase completion.')
        if status == 'ok' and any(c['exit_code'] != 0 for c in self.commands):
            raise ValueError('A failed command cannot silently become a successful phase.')
        if status == 'fail' and not failure_summary:
            raise ValueError('Failure must have a summary.')
        if status == 'skip' and not reason:
            raise ValueError('Skip must have a machine-readable reason.')
        out, err, raw = [], [], []
        for c in self.commands:
            o = (self.run.root / c['stdout']).read_text()
            e = (self.run.root / c['stderr']).read_text()
            out.append(o)
            err.append(e)
            raw.extend([json.dumps(c, sort_keys=True), '\nSTDOUT\n', o, '\nSTDERR\n', e, '\n'])
        logbase = self.path / 'logs' / self.name
        atomic(str(logbase) + '.out', ''.join(out))
        atomic(str(logbase) + '.err', ''.join(err))
        rawpath = Path(str(logbase) + '.raw.out')
        atomic(rawpath, ''.join(raw))
        d = {'schema_version': 1, 'runner': self.name, 'status': status, 'started_at': self.started,
             'ended_at': utcnow(), 'duration_s': time.monotonic() - self.monotonic, 'metadata': metadata or {},
             'results': results or [], 'log_relpath': str(rawpath.relative_to(self.run.root)), 'log_sha256': sha256(rawpath)}
        if status == 'fail':
            d.update(exit_code=exit_code if exit_code is not None else next((c['exit_code'] for c in self.commands if c['exit_code']), 1),
                     timeout=any(c['timeout'] for c in self.commands), failure_summary=failure_summary)
        if status == 'skip':
            d['reason'] = reason
        suffix = {'ok': 'values', 'fail': 'failed', 'skip': 'skipped'}[status]
        atomic(self.path / f'{self.name}.{suffix}.json', d)
        atomic(self.path / f'{self.name}.md', markdown(d))
        self.finished = True
        if refresh:
            self.run.refresh()
        return d
