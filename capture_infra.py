"""Capture infrastructure evidence without changing device or cluster settings."""
import argparse
import datetime
import json
import pathlib
import shutil
import signal
import socket
import subprocess
import time

STOP = False


def command(argv, timeout=25):
    started = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "exit_code": p.returncode, "seconds": time.time() - started,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": argv, "error": str(e), "seconds": time.time() - started}


def read(path):
    try:
        return pathlib.Path(path).read_text()
    except OSError as e:
        return {"error": str(e)}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def snapshot(out, phase):
    commands = [
        ["uname", "-a"], ["lscpu", "-J"], ["numactl", "--hardware"],
        ["nvidia-smi", "-q", "-x"], ["nvidia-smi", "topo", "-m"],
        ["nvidia-smi", "nvlink", "-s"], ["nvidia-smi", "nvlink", "-e"],
        ["lspci", "-nn"], ["ibstat"], ["ibv_devinfo", "-v"],
        ["rdma", "link", "show"], ["ip", "-j", "-s", "link", "show"],
        ["ip", "-j", "address", "show"], ["ip", "-j", "route", "show"],
        ["df", "-hT"], ["findmnt", "-J"], ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL"],
        ["docker", "version"], ["docker", "info", "--format", "{{.ServerVersion}} {{.Driver}} {{.CgroupVersion}} {{.DockerRootDir}}"],
        ["sinfo", "-N", "-l"], ["scontrol", "show", "nodes"],
        ["scontrol", "show", "partition", "gpu-nodes"],
        ["ulimit"], ["lsmod"],
    ]
    results = [command(c) for c in commands if shutil.which(c[0])]
    for dev in pathlib.Path("/sys/class/net").iterdir():
        if shutil.which("ethtool") and dev.name != "lo":
            for flags in [[], ["-i"], ["-k"], ["-S"]]:
                results.append(command(["ethtool", *flags, dev.name]))
    files = {p: read(p) for p in ["/etc/os-release", "/proc/meminfo", "/proc/cpuinfo", "/proc/self/limits",
        "/proc/self/cgroup", "/proc/driver/nvidia/version", "/proc/mounts", "/proc/sys/kernel/numa_balancing",
        "/proc/sys/vm/swappiness", "/proc/sys/net/ipv4/tcp_congestion_control"]}
    for pattern in ["/sys/class/infiniband/*/ports/*/counters/*", "/sys/class/infiniband/*/ports/*/hw_counters/*",
                    "/sys/class/infiniband/*/fw_ver", "/sys/class/net/*/mtu", "/sys/class/net/*/speed"]:
        import glob
        files.update({p: read(p) for p in glob.glob(pattern)})
    path = out / f"{socket.gethostname()}-{phase}.json"
    path.write_text(json.dumps({"timestamp": stamp(), "hostname": socket.gethostname(), "commands": results, "files": files}, indent=2))
    print(path, flush=True)


def monitor(out):
    global STOP
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    fields = "index,uuid,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate,pcie.link.gen.current,pcie.link.width.current"
    import glob
    counter_paths = []
    for name in ["port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets", "port_xmit_wait", "port_rcv_errors", "port_xmit_discards", "link_downed"]:
        counter_paths.extend(glob.glob("/sys/class/infiniband/*/ports/*/counters/" + name))
    with (out / f"{socket.gethostname()}-timeseries.jsonl").open("a", buffering=1) as f:
        while not STOP:
            start = time.monotonic()
            row = {"timestamp": stamp(), "monotonic": start,
                   "gpu": command(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], 8),
                   "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], 8),
                   "infiniband_counters": {p:read(p) for p in counter_paths},
                   "system": {p: read(p) for p in ["/proc/stat", "/proc/meminfo", "/proc/loadavg", "/proc/net/dev", "/proc/diskstats", "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io"]}}
            f.write(json.dumps(row) + "\n")
            time.sleep(max(0, 2 - (time.monotonic() - start)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("mode", choices=["before", "after", "preflight", "monitor"])
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(pathlib.Path("/shared/clustermax-campaigns")):
        raise ValueError("Evidence must go in the new campaign")
    args.out.mkdir(parents=True, exist_ok=True)
    monitor(args.out) if args.mode == "monitor" else snapshot(args.out, args.mode)
