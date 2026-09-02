"""Pin the complete clean task catalog and prospectively select train/dev IDs.

Only root-directory metadata is fetched here. Instructions, solutions, hidden
tests, task images and model outcomes are neither read nor selected on.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

from evidence import Run, atomic, sha256


REPOSITORY = 'Lego-X/Terminal-Lego-15k'
REVISION = '9c197f1c2e87b64cc316b1a5bfcef57b584929f0'
CATALOG_URL = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}'


def select_tasks(task_ids, train_count=512, dev_count=128):
    ids = set(task_ids)
    if len(ids) != len(task_ids) or any(not re.fullmatch(r'task_[0-9]{5}', x) for x in ids):
        raise ValueError('Task catalog contains duplicate or malformed IDs.')
    # A task whose configuration was inspected is reserved solely for runtime validation.
    ids.discard('task_00000')
    if min(train_count, dev_count) < 1 or len(ids) < train_count + dev_count:
        raise ValueError('Not enough tasks for the prospectively fixed split.')
    ordered = sorted(ids, key=lambda task: (hashlib.sha256(('posttrainingx-clean-v1:1234\0' + task).encode()).hexdigest(), task))
    return ordered[:train_count], ordered[train_count:train_count + dev_count]


def next_page(header):
    for part in (header or '').split(','):
        match = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if match:
            url = match.group(1)
            parsed = urllib.parse.urlsplit(url)
            expected = urllib.parse.urlsplit(CATALOG_URL)
            if (parsed.scheme, parsed.netloc, parsed.path) != (expected.scheme, expected.netloc, expected.path):
                raise ValueError('Pagination escaped the pinned public dataset catalog.')
            return url
    return None


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    url, seen_urls, tasks, pages = CATALOG_URL + '?recursive=false&expand=false', set(), [], []
    while url:
        if url in seen_urls or len(seen_urls) >= 40:
            raise ValueError('Unexpected catalog pagination loop or size.')
        seen_urls.add(url)
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(4 * 1024**2 + 1)
            if len(payload) > 4 * 1024**2:
                raise ValueError('Catalog page exceeds 4 MiB guard.')
            following = next_page(response.headers.get('Link'))
        values = json.loads(payload)
        page_file = destination / f'page-{len(pages):03d}.json'
        atomic(page_file, values)
        pages.append({'url': url, 'path': page_file.name, 'sha256': sha256(page_file), 'entries': len(values)})
        for item in values:
            if item['type'] == 'directory' and re.fullmatch(r'task_[0-9]{5}', item['path']):
                tasks.append(item['path'])
        url = following
    if len(set(tasks)) != len(tasks):
        raise ValueError('Catalog pagination returned duplicate task directories.')
    result = {'schema_version': 1, 'repository': REPOSITORY, 'revision': REVISION,
              'ordered_task_ids': sorted(tasks), 'task_count': len(tasks), 'pages': pages}
    atomic(destination / 'catalog.json', result)
    print(json.dumps({'task_count': len(tasks), 'pages': len(pages), 'catalog_sha256': sha256(destination / 'catalog.json')}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--fetch-into')
    args = ap.parse_args()
    if args.fetch_into:
        fetch(args.fetch_into)
        return 0
    run = Run(args.run_dir)
    phase = run.phase('00-terminal-lego-catalog-and-prospective-split')
    destination = run.root / 'provenance/terminal-lego-catalog-v1'
    rc, out, _ = phase.command([sys.executable, str(Path(__file__).resolve()), '--fetch-into', str(destination)], timeout=900)
    if rc:
        phase.finish('fail', failure_summary='Pinned clean catalog acquisition failed; do not replace or silently retry incomplete pages.')
        return rc
    catalog = json.loads((destination / 'catalog.json').read_text())
    train, dev = select_tasks(catalog['ordered_task_ids'])
    data = {
        'schema_version': 1, 'status': 'prospective_ids_pinned_runtime_eligibility_unvalidated',
        'repository': REPOSITORY, 'revision': REVISION,
        'selection': 'Ascending SHA256 of UTF-8 posttrainingx-clean-v1:1234 + NUL + task ID; tie-break by task ID.',
        'catalog_count': catalog['task_count'], 'catalog_sha256': sha256(destination / 'catalog.json'),
        'training_task_ids': train, 'development_task_ids': dev, 'runtime_validation_task_ids': ['task_00000'],
        'training_order_sha256': hashlib.sha256(('\n'.join(train) + '\n').encode()).hexdigest(),
        'development_order_sha256': hashlib.sha256(('\n'.join(dev) + '\n').encode()).hexdigest(),
        'terminal_bench_training_tasks': [],
        'rules': ['No task chosen using policy or evaluation outcomes.',
                  'No implicit replacement of a task whose image, oracle or runtime gate fails.',
                  'Any eligibility amendment requires a new manifest before baseline evaluation.',
                  'Instruction leakage checks, per-file hashes, image pins and known-good verification remain required.',
                  'TB2.1 remains evaluation-only; development tasks are from the clean training corpus.'],
    }
    atomic(phase.path / 'prospective-split.json', data)
    atomic(Path(__file__).resolve().parents[1] / 'locks/terminal-lego-subset.json', data)
    phase.finish('ok', metadata=dict(data, scope='Prospective ID selection, not a validated dataset or training result.'))
    print(out)
    print(json.dumps({'train_tasks': len(train), 'development_tasks': len(dev), 'status': data['status']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
