"""Post-run CPU validation in the pinned image; no GPU or network access needed.

Uses the existing isolated task Docker daemon. Preserves checkpoints and writes
new validation JSON/log files only after training and Slurm complete.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

CAMPAIGN=Path('/shared/clustermax-campaigns/miles-async-ppo-terminal-lego-20260904T010040Z')
RUN=CAMPAIGN/'runs/job-197'
IMAGE='radixark/miles@sha256:4ee6da9f16e06f8ad24991b18a950482572c458a357aae0bfc396feaf3fe0a6d'
BASE=Path('/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030/converted-model/release')


def main():
    assert (RUN/'exit-code.txt').read_text().strip()=='0'
    state=subprocess.run(['sacct','-j','197','--allocations','-n','-P','--format=State,ExitCode'],capture_output=True,text=True,check=True).stdout.strip()
    assert state=='COMPLETED|0:0',state
    manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'image':IMAGE,
              'verification_source_sha256':hashlib.sha256((CAMPAIGN/'analysis-source/verify_final_checkpoints.py').read_bytes()).hexdigest(),'commands':[],
              'scope':'CPU structural/sampled checks only, not a full distributed resume; no GPU/network; checkpoint files preserved'}
    for role in ['actor','critic']:
        argv=['docker','--host=unix:///tmp/miles-terminal-lego-20260903-2030/docker.sock','run','--rm',
              '--network=none','--cpus=2','--memory=8g','-e','OMP_NUM_THREADS=1',
              '--mount',f'type=bind,src={RUN},dst=/verify/run',
              '--mount',f'type=bind,src={BASE},dst=/verify/base,readonly',
              '--mount',f'type=bind,src={CAMPAIGN}/analysis-source,dst=/verify/source,readonly',
              '--entrypoint','python',IMAGE,'/verify/source/verify_final_checkpoints.py','/verify/run','--base','/verify/base']
        if role=='critic':argv.append('--critic')
        name='critic-checkpoint-verification.json' if role=='critic' else 'checkpoint-verification.json'
        assert not (RUN/name).exists()
        with (RUN/f'checkpoint-validation-{role}.log').open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        manifest['commands'].append({'role':role,'argv':argv,'exit_code':result.returncode})
        print(json.dumps(manifest['commands'][-1]),flush=True)
        if result.returncode:
            (RUN/'post-run-checkpoint-validation-failed.json').write_text(json.dumps(manifest,indent=2)+'\n')
            raise RuntimeError(f'{role} validation failed; inspect retained log')
    manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (RUN/'post-run-checkpoint-validation.json').open('x') as f:json.dump(manifest,f,indent=2)

if __name__=='__main__':main()
