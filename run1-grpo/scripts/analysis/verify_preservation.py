"""Read-only original-run audit; writes a new comparison evidence file."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030')
baseline = Path('/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00/runs/20260903-150011')
reference = json.loads((root/'baseline-preservation-reference.json').read_text())
result = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'hashes':[], 'metadata':[], 'limitations':'Only the listed source/config files are content-hashed. Large run artifacts and checkpoints are checked by size and mtime, not full content hash.'}
for path, expected in reference['hashes'].items():
    p = Path(path)
    actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    result['hashes'].append({'path':path,'expected':expected,'actual':actual,'unchanged':expected==actual})
for before in reference['run_inventory']:
    p = baseline/before['path']
    after = {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} if p.exists() else None
    expected = {k:before[k] for k in ['size','mtime_ns']}
    result['metadata'].append({'path':before['path'],'before':expected,'after':after,'unchanged':expected==after})
result['summary'] = {'hashed_files':len(result['hashes']),
    'changed_hashes':[x['path'] for x in result['hashes'] if not x['unchanged']],
    'inventoried_artifacts':len(result['metadata']),
    'changed_metadata':[x['path'] for x in result['metadata'] if not x['unchanged']]}
out = root/sys.argv[1]
assert out.is_relative_to(root) and not out.exists()
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result['summary'],indent=2))
