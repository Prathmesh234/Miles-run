"""Read-only preflight, freeze sources, and submit one matched PPO experiment.

Usage: python3 launch_async_ppo.py --submit
No Docker pull, baseline write, cluster reconfiguration, or automatic retry.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
OLD = '/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030'
IMAGE = 'radixark/miles@sha256:4ee6da9f16e06f8ad24991b18a950482572c458a357aae0bfc396feaf3fe0a6d'
SOCKET = 'unix:///tmp/miles-terminal-lego-20260903-2030/docker.sock'
SOURCE = ['coordinator.py', 'training_entry.py', 'harness_bridge.py', 'rollout_adapter.py',
          'ipo_loss.py', 'test_adapters.py', 'test_ppo.py', 'sglang.yaml',
          'prepare_ppo_driver.py', 'ppo-resident-broadcast.patch',
          'install_precision_patch.py', 'sglang_precision.py',
          'capture_infra.py', 'capture_rdma.py', 'capture_metrics.py', 'capture_health.py', 'container_evidence.py',
          'extract_metrics.py', 'verify_checkpoint.py', 'run.sbatch', 'launch_async_ppo.py', 'prepare_async_driver.py', 'async_runtime.py', 'async_metrics.py', 'test_async.py']


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubeconfig', default=str(Path.home()/'.kube/vultr-vke.yaml'))
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--after-job', type=int, default=196)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    campaign = '/shared/clustermax-campaigns/miles-async-ppo-terminal-lego-'+stamp
    local = ROOT/'.async-work'/stamp
    local.mkdir(parents=True, exist_ok=False)
    kube = ['kubectl', '--kubeconfig', args.kubeconfig]

    def run(command, label, data=None):
        p = subprocess.run(command, input=data, capture_output=True, timeout=180)
        (local/(label+'.out')).write_bytes(p.stdout)
        (local/(label+'.err')).write_bytes(p.stderr)
        atomic(local/(label+'.command.json'), {'argv':command, 'exit_code':p.returncode})
        if p.returncode:
            raise RuntimeError(f'{label} failed with {p.returncode}; inspect {local}')
        return p.stdout

    def remote(node, command, label, data=None):
        return run(kube+['exec', '-i', '-n', 'slurm', f'slurm-worker-gpu-nodes-{node}',
                        '-c', 'slurmd', '--', *command], label, data)

    script = '''import json,subprocess,socket,shutil,pathlib
def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {"argv":args,"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
print(json.dumps({"host":socket.gethostname(),"free_bytes":shutil.disk_usage("/shared").free,
 "gpus":run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used","--format=csv,noheader"]),
 "processes":run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"]),
 "slurm":run(["scontrol","show","node",socket.gethostname(),"--oneliner"]),
 "ib":run(["ibstat"]),"mount":run(["findmnt","-J","/shared"]),
 "memory":pathlib.Path("/proc/meminfo").read_text(),
 "image":run(["docker","--host=SOCKET","image","inspect","--format={{.Id}}","IMAGE"]),
 "driver":run(["docker","--host=SOCKET","info","--format={{.Driver}}"]),
 "queue":run(["squeue","-h","-o","%i|%T|%N"])}))
'''.replace('SOCKET', SOCKET).replace('IMAGE', IMAGE)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(remote, n, ['python3', '-c', script], f'node-{n}') for n in range(4)]
            nodes = [json.loads(f.result()) for f in futures]
        kubenodes = json.loads(run(kube+['get', 'nodes', '-o', 'json'], 'kubernetes-nodes'))
        kubegpu = sum(int(n['status']['allocatable'].get('nvidia.com/gpu', 0)) for n in kubenodes['items'])
        assert kubegpu == 32, f'Kubernetes allocatable GPUs={kubegpu}'
        uuids = []
        for node in nodes:
            rows = node['gpus']['stdout'].strip().splitlines()
            assert node['gpus']['exit_code'] == 0 and len(rows) == 8, node['host']
            uuids += [row.split(',')[0] for row in rows]
            assert all('NVIDIA B200' in row for row in rows)
            assert node['processes']['exit_code'] == 0
            assert 'gres/gpu=8' in node['slurm']['stdout']
            queued = [x.split('|')[0] for x in node['queue']['stdout'].strip().splitlines()]
            assert all(x == str(args.after_job) for x in queued), 'Unexpected queued work; inspect before submitting'
            if not queued:
                assert not node['processes']['stdout'].strip(), 'GPU processes remain after predecessor'
                assert 'State=IDLE' in node['slurm']['stdout']
            assert node['image']['exit_code'] == 0 and node['image']['stdout'].startswith('sha256:')
            assert node['driver']['stdout'].strip() == 'fuse-overlayfs'
            assert node['free_bytes'] > 512*1024**3
            assert node['ib']['exit_code'] == 0 and node['ib']['stdout'].count('Rate: 400') == 8
        assert len(set(uuids)) == 32
        spec = json.loads((ROOT/'comparison-spec.json').read_text())
        manifest = {'schema_version':1, 'status':'preflight_passed', 'created_at':stamp,
            'campaign':campaign, 'baseline_jobs':[181,190], 'reused_campaign_read_only':OLD,
            'algorithm':'async_ppo_tis', 'predecessor_job':args.after_job,
            'dependency':'afterok:'+str(args.after_job),
            'scheduling':'native train_async.py one-batch-ahead; not fully-async persistent worker',
            'importance_sampling':{'ppo_denominator':'recomputed trainer-before-update logprob',
                'tis_ratio':'exp(trainer_before_update_logprob - recorded_behavior_logprob)',
                'clip_low':0.,'clip_high':2.,'detached':True,'use_rollout_logprobs':False},
            'instrumentation':['driver transfer/onload/offload seconds', 'per-rank memory and offload wall time',
                'CPU/GPU optimizer referenced storage bytes', 'optimizer step host wall time',
                'TIS ratios/weights/clipping', 'policy versions and overlap timeline',
                'GPU telemetry, IB counter deltas, NCCL logs, SGLang Prometheus'], 'image':IMAGE, 'miles_sha':spec['miles']['commit'],
            'clustermax_sha':'fed871df5321d42706c98701522cc3ccd55898d5',
            'publication_parent_sha':'0fa4636863b1f61b444a830f74980cb06d59c10e',
            'matched_workload':spec['required_invariants'],
            'inherited_backend_deviations':spec['explicit_backend_deviations'],
            'ppo_changes':{'critic_lr':1e-5, 'critic_warmup_steps':0, 'policy_clip':.2,
                'value_clip':.2, 'gamma':1., 'gae_lambda':1., 'normalize_advantages':True,
                'rewards':'raw, no group centering', 'zero_variance_groups':'retained',
                'critic':'colocated on actor GPUs with train offload',
                'truncation':'native GAE zero bootstrap; terminal observed reward retained',
                'entropy_coef':0., 'reference_kl_coef':0.},
            'comparison_limitations':['two updates; no held-out quality evidence',
                'task source/order/batch settings match; sampled outcomes and accepted task mix do not',
                'same base weights; warmed image and conversion caches',
                'weights-only actor/critic checkpoints; no optimizer/RNG resume test',
                'native PPO adds critic work and keeps zero-variance groups'],
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE},
            'inventory':nodes, 'kubernetes_allocatable_gpus':kubegpu}
        atomic(local/'run.json', manifest)
        if not args.submit:
            print(json.dumps({'status':'preflight_passed_not_submitted','evidence':str(local)}))
            return
        # The archive has only named source and manifest files. No task secrets,
        # model weights, baseline outputs, or local credentials are staged.
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w') as tar:
            for name in SOURCE:
                tar.add(ROOT/name, arcname='code/'+name, recursive=False)
            for name, payload in {'comparison-spec.json':manifest,
                                  'image-digest.json':[IMAGE]}.items():
                content = json.dumps(payload, indent=2).encode()
                entry = tarfile.TarInfo(name); entry.size = len(content)
                tar.addfile(entry, io.BytesIO(content))
            for path in local.iterdir():
                tar.add(path, arcname='preflight/'+path.name, recursive=False)
        stage = '''import sys,tarfile,pathlib,shutil
p=pathlib.Path(sys.argv[1]); assert p.parent==pathlib.Path('/shared/clustermax-campaigns')
assert shutil.disk_usage(p.parent).free>512*1024**3
p.mkdir(exist_ok=False); (p/'runs').mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
 for m in t:
  assert m.isfile() and not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts
  target=p/m.name; target.parent.mkdir(parents=True,exist_ok=True)
  tmp=target.with_suffix(target.suffix+'.tmp'); tmp.write_bytes(t.extractfile(m).read()); tmp.replace(target)
print(p)
'''
        remote(0, ['python3','-c',stage,campaign], 'stage', archive.getvalue())
        # CPU-only validation container: no GPU devices, no network, bounded CPU/RAM.
        # Keep its stopped container and all validation outputs as evidence.
        validation = ['docker','--host='+SOCKET,'run','--name','miles-async-preflight-'+stamp,
            '--network','none','--cpus','2','--memory','4g',
            '-v',campaign+':/campaign','-v',OLD+'/miles:/campaign/miles:ro',
            '-w','/campaign/code','-e','PYTHONDONTWRITEBYTECODE=1','-e','MILES_ALGORITHM=ppo',
            '-e','MILES_RUN_DIR=/campaign/preflight','-e','MILES_PATCH_OUTPUT_DIR=/campaign/preflight/driver',
            '-e','PYTHONPATH=/campaign/code',IMAGE,'python','-c',
            "import runpy; runpy.run_path('prepare_async_driver.py',run_name='__main__'); runpy.run_path('test_async.py',run_name='__main__')"]
        remote(0, validation, 'cpu-validation')
        export = f'ALL,MILES_ALGORITHM=ppo,MILES_CAMPAIGN_ROOT={campaign},MILES_REUSE_CAMPAIGN={OLD}'
        job = remote(0, ['sbatch','--parsable','--job-name=miles-ppo-terminal-lego',
                        '--time=01:00:00','--dependency=afterok:'+str(args.after_job),'--export='+export,
                        '--output='+campaign+'/preflight/slurm-%j.out',campaign+'/code/run.sbatch'], 'submit').decode().strip()
        manifest.update(status='submitted', job_id=int(job.split(';')[0]))
        atomic(local/'run.json', manifest)
        print(json.dumps({'status':'submitted','job_id':manifest['job_id'],'campaign':campaign,'local_evidence':str(local)}))
    except Exception as error:
        atomic(local/'failed.json', {'status':'failed','error':repr(error),'campaign':campaign,
                                    'automatic_retry':False})
        raise


if __name__ == '__main__':
    main()
