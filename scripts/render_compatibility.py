"""Generate the dependency report exclusively from its JSON source."""
import json
from pathlib import Path

from evidence import atomic


def render(data):
    lines = ['# Dependency and runtime compatibility', '', 'Generated from `docs/compatibility.json`.', '',
             '| Path | Dependencies | Status | Evidence |', '|---|---|---|---|']
    for row in data['matrix']:
        lines.append('| ' + ' | '.join(row[key].replace('|', '\\|') for key in
                                     ('path', 'dependencies', 'status', 'evidence')) + ' |')
    lines += ['', '## Findings', ''] + ['- ' + note for note in data['notes']]
    lines += ['', '## Strict online acceptance tests', '', 'All must pass before enabling optimizer steps:', '']
    lines += ['- ' + item for item in data['strict_online_required_tests']]
    lines += ['', '## Reproducer', '', '```sh', data['reproducer'], '```', '',
              'Dependency resolution only; expected conflicts never trigger forced installation.', '',
              '## Locks', '']
    for name in ('offline_lock', 'openenv_development_lock', 'openenv_server_linux_lock'):
        lock = data.get(name)
        if lock:
            lines += [f"- `{lock['path']}`: SHA256 `{lock['sha256']}`."]
    lines += ['', '## Sources', '']
    lines += [f"- [{name}]({item['source']})" for name, item in data['locked_metadata'].items()]
    lines += ['', f"Evidence run: `{data['run_id']}`.", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    source = Path('docs/compatibility.json')
    atomic(source.with_suffix('.md'), render(json.loads(source.read_text())))
