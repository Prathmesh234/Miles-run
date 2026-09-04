"""Coordinate one allocation; every process and container is scoped to its job ID."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time
import urllib.request

C = Path(os.environ.get("MILES_CAMPAIGN_ROOT", "/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030"))
REUSE = Path(os.environ.get("MILES_REUSE_CAMPAIGN", str(C)))
BASE = Path("/shared/clustermax-campaigns/prime-rl-terminal-lego-b29c37e00")
MODEL = BASE / "model-fetch/models/qwen3.6-35b-a3b-995ad96eacd98c81ed38be0c5b274b04031597b0"
PYTHON = BASE / "prime-rl/.venv/bin/python"
TASK_CODE = BASE / "code-git/scripts/2-performance/training/prime-rl-terminal-lego/workload"
JID = os.environ["SLURM_JOB_ID"]
RUN = C / "runs" / f"job-{JID}"
CONTAINER_RUN = "/campaign/runs/" + RUN.name
CODE = RUN / "source"
CONTAINER_CODE = CONTAINER_RUN + "/source"
NAME = "miles-" + os.environ.get("MILES_ALGORITHM", "ipo") + "-" + JID
HEAD_PORT = 16379
DASHBOARD_PORT = 18265
BRIDGE_PORT = 18981
DOCKER_SOCKET = "unix:///tmp/miles-terminal-lego-20260903-2030/docker.sock"


def event(name, **data):
    row = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "event": name, **data}
    with (RUN / "timeline.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def execute(argv, name, timeout=1800, check=True):
    event(name+"_start", argv=argv)
    with (RUN / (name+".log")).open("w") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    event(name+"_end", exit_code=p.returncode)
    if check and p.returncode:
        raise RuntimeError(f"{name} exited with {p.returncode}")
    return p.returncode


def cleanup(argv, name, timeout):
    try:
        execute(argv, name, timeout, check=False)
    except Exception as error:
        event(name + "_cleanup_error", error=repr(error))


def on(node, argv):
    if argv[0] == "docker":
        argv = ["docker", "--host="+DOCKER_SOCKET, *argv[1:]]
    return ["srun", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--nodelist="+node, *argv]


def docker_env():
    return {"PYTHONPATH":"/campaign/miles:/root/Megatron-LM:"+CONTAINER_CODE, "PYTHONDONTWRITEBYTECODE":"1",
            "MILES_RUN_DIR":CONTAINER_RUN, "NCCL_DEBUG":"INFO", "NCCL_DEBUG_SUBSYS":"INIT,NET,GRAPH",
            "NCCL_DEBUG_FILE":CONTAINER_RUN+"/infra/nccl-%h-%p.log", "WANDB_MODE":"disabled",
            "HF_HUB_OFFLINE":"1", "TOKENIZERS_PARALLELISM":"false", "OMP_NUM_THREADS":"8",
            "MILES_HARNESS_URL":f"http://{socket.gethostbyname('gpu-nodes-3')}:{BRIDGE_PORT}",
            "RAY_ADDRESS":f"{socket.gethostbyname('gpu-nodes-0')}:{HEAD_PORT}",
            "MILES_ALGORITHM":os.environ.get("MILES_ALGORITHM", "ipo"),
            "MILES_CAMPAIGN_ROOT":str(C)}


def main():
    if not C.is_relative_to(Path("/shared/clustermax-campaigns")):
        raise ValueError("Run root must be under /shared/clustermax-campaigns")
    if shutil.disk_usage(C).free < 512 * 1024**3:
        raise RuntimeError("Free-space guard: less than 512 GiB remains")
    RUN.mkdir(exist_ok=False)
    (RUN/"infra").mkdir()
    (RUN/"tensorboard").mkdir()
    shutil.copytree(C/"code", RUN/"source")
    (RUN/"source-sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/"source").iterdir() if p.is_file()}, indent=2))
    shutil.copy2(C/"comparison-spec.json", RUN/"comparison-spec.json")
    hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    assert sorted(hosts) == [f"gpu-nodes-{i}" for i in range(4)], hosts
    hosts = sorted(hosts)
    (RUN/"hosts.json").write_text(json.dumps({h:socket.gethostbyname(h) for h in hosts}, indent=2))
    image = json.loads((C/"image-digest.json").read_text())[0]
    (RUN/"image.txt").write_text(image+"\n")
    started_containers = []
    monitor = bridge = rdma = prometheus = health_monitor = None
    code = 1
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        idle_check = "import subprocess; p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True); assert not p.stdout.strip(), 'GPU processes remain from preceding workload'"
        for host in hosts:
            execute(on(host,["python3","-c",idle_check]), "allocation-idle-"+host, 30)
        execute(["srun", "--overlap", "--ntasks-per-node=1", "python3", str(CODE/"capture_infra.py"), str(RUN/"infra"), "before"], "infra-before", 180)
        monitor_log = (RUN/"monitor.log").open("w")
        monitor = subprocess.Popen(["srun", "--overlap", "--ntasks-per-node=1", "python3", str(CODE/"capture_infra.py"), str(RUN/"infra"), "monitor"], stdout=monitor_log, stderr=subprocess.STDOUT)
        health_monitor = subprocess.Popen(["srun", "--overlap", "--ntasks-per-node=1", "python3", str(CODE/"capture_health.py"), str(RUN)],
                                          stdout=(RUN/"health-collector.log").open("w"), stderr=subprocess.STDOUT)
        for host in hosts:
            args = ["docker", "run", "-d", "--name", NAME, "--gpus", "all", "--network", "host", "--ipc", "host",
                    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--device", "/dev/infiniband",
                    "-v", str(C)+":/campaign", "-v", str(MODEL)+":"+str(MODEL)+":ro", "-w", CONTAINER_RUN]
            if REUSE != C:
                args += ["-v", str(REUSE/"miles")+":/campaign/miles:ro",
                         "-v", str(REUSE/"converted-model")+":/campaign/converted-model:ro"]
            for k,v in docker_env().items():
                args += ["-e", k+"="+v]
            args += [image, "sleep", "14400"]
            execute(on(host,args), "container-start-"+host, 120)
            started_containers.append(host)
            execute(on(host,["docker","exec",NAME,"python",CONTAINER_CODE+"/install_precision_patch.py"]), "precision-patch-"+host, 30)
        execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/sglang_precision.py"]), "precision-gpu-validation", 120)
        if os.environ.get("MILES_ALGORITHM") == "ppo":
            execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/prepare_ppo_driver.py"]), "ppo-driver-patch", 30)
            execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/prepare_async_driver.py"]), "async-driver-patch", 30)
            execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/test_async.py"]), "async-tests", 120)
        execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/training_entry.py","validate"]), "argument-validation", 180)
        if not (REUSE/"converted-model/conversion-complete.json").exists():
            if REUSE != C:
                raise RuntimeError("Pinned reusable base conversion is missing; do not repair it in place")
            execute(on("gpu-nodes-0", ["docker","exec",NAME,"python",CONTAINER_CODE+"/training_entry.py","convert"]), "checkpoint-conversion", 2400)
            assert (C/"converted-model/release").is_dir()
            (C/"converted-model/conversion-complete.json").write_text(json.dumps({"job_id":JID,"source":str(MODEL),"image":image}))
        else:
            event("checkpoint_conversion_reused", path=str(REUSE/"converted-model"))
        if os.environ.get("MILES_ALGORITHM") == "ppo":
            execute(on("gpu-nodes-0", ["docker", "exec", NAME, "python", CONTAINER_CODE+"/test_ppo.py", "native"]), "ppo-native-tests", 300)
            test_env = ["env", "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=", "HF_HUB_OFFLINE=1",
                        "MILES_ALGORITHM=ppo", "MILES_RUN_DIR="+str(RUN),
                        "PYTHONPATH="+str(CODE)+":"+str(TASK_CODE), str(PYTHON), str(CODE/"test_ppo.py"), "transport"]
            execute(on("gpu-nodes-3", test_env), "ppo-transport-tests", 300)
        bridge_log = (RUN/"harness.log").open("w")
        bridge_env = ["env", "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=", "HF_HUB_OFFLINE=1",
                      "PYTHONPATH="+str(CODE)+":"+str(TASK_CODE), "MILES_RUN_DIR="+str(RUN),
                      "MILES_HARNESS_PORT="+str(BRIDGE_PORT), "XDG_CACHE_HOME="+str(RUN/"harness-cache"),
                      "MILES_ALGORITHM="+os.environ.get("MILES_ALGORITHM", "ipo"),
                      str(PYTHON), str(CODE/"harness_bridge.py")]
        bridge = subprocess.Popen(on("gpu-nodes-3",bridge_env), stdout=bridge_log, stderr=subprocess.STDOUT)
        health = docker_env()["MILES_HARNESS_URL"]+"/health"
        for _ in range(90):
            try:
                with urllib.request.urlopen(health, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                if bridge.poll() is not None:
                    raise RuntimeError("Harness process exited")
                time.sleep(1)
        else:
            raise RuntimeError("Harness did not become ready")
        event("harness_ready")
        rdma = subprocess.Popen(["srun", "--overlap", "--ntasks-per-node=1", "python3", str(CODE/"capture_rdma.py"), str(RUN)],
                                stdout=(RUN/"rdma-collector.log").open("w"), stderr=subprocess.STDOUT)
        prometheus = subprocess.Popen(["python3", str(CODE/"capture_metrics.py"), str(RUN),
                                       f"http://{socket.gethostbyname('gpu-nodes-3')}:15000/metrics"],
                                      stdout=(RUN/"prometheus-collector.log").open("w"), stderr=subprocess.STDOUT)
        for host in hosts:
            ip = socket.gethostbyname(host)
            args = ["docker","exec",NAME,"ray","start","--node-ip-address="+ip,"--num-gpus=8","--num-cpus=64",
                    "--object-store-memory=4294967296", "--disable-usage-stats",
                    "--min-worker-port=20000", "--max-worker-port=29999"]
            if host == "gpu-nodes-0":
                args += ["--head", "--port="+str(HEAD_PORT), "--dashboard-host=0.0.0.0", "--dashboard-port="+str(DASHBOARD_PORT)]
            else:
                args += ["--address="+docker_env()["RAY_ADDRESS"]]
            execute(on(host,args), "ray-start-"+host, 180)
        execute(on("gpu-nodes-0",["docker","exec",NAME,"ray","status"]), "ray-status", 60)
        event("training_start")
        code = execute(on("gpu-nodes-0",["docker","exec",NAME,"python",CONTAINER_CODE+"/training_entry.py","train"]), "training", 9000, check=False)
        event("training_end", exit_code=code)
    except Exception as e:
        event("failure", error=repr(e))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for host in started_containers:
            # Copy logs out before stopping this job's container; retain the stopped
            # containers and all artifacts for postmortem inspection.
            cleanup(on(host,["docker","exec",NAME,"python",CONTAINER_CODE+"/container_evidence.py",CONTAINER_RUN]), "container-evidence-"+host, 120)
            cleanup(on(host,["docker","stop","--time","5",NAME]), "container-stop-"+host, 60)
        for p in [bridge, monitor, rdma, prometheus, health_monitor]:
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    p.kill()
        cleanup(["srun","--overlap","--ntasks-per-node=1","python3",str(CODE/"capture_infra.py"),str(RUN/"infra"),"after"], "infra-after", 180)
        cleanup(["sacct","-j",JID,"-P","--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,AllocTRES,MaxRSS,TotalCPU"], "slurm-accounting", 30)
        (RUN/"exit-code.txt").write_text(str(code)+"\n")
        event("coordinator_exit", exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
