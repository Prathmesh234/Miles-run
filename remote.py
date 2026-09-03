"""Run a command in a Slurm worker, or transfer files into this campaign only."""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030"
KUBE = "/Users/prathmeshbhatt/.kube/vultr-vke.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--put", type=pathlib.Path)
    parser.add_argument("--dest")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    prefix = ["kubectl", f"--kubeconfig={KUBE}", "--request-timeout=20s", "exec", "-i", "-n", "slurm", f"slurm-worker-gpu-nodes-{args.node}", "-c", "slurmd", "--"]
    if args.put:
        dest = pathlib.PurePosixPath(CAMPAIGN) / (args.dest or args.put.name)
        if ".." in dest.parts or not str(dest).startswith(CAMPAIGN + "/"):
            raise ValueError("Destination must be inside the new campaign")
        script = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(sys.stdin.buffer.read())"
        result = subprocess.run(prefix + ["python3", "-c", script, str(dest)], input=args.put.read_bytes())
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = subprocess.run(prefix + command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
