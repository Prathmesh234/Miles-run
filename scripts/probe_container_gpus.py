"""Check every allocated GPU and a deterministic BF16 operation; no model steps."""
import argparse
import csv
import json
import socket
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expected-uuids', required=True)
    args = ap.parse_args()
    expected = set(json.loads(args.expected_uuids))
    import torch
    if torch.cuda.device_count() != 8 or len(expected) != 8:
        raise ValueError('Expected exactly eight allocated CUDA devices.')
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                    '--format=csv,noheader,nounits'], text=True)
    inventory = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if {r[1] for r in inventory} != expected:
        raise ValueError('Container GPU UUIDs do not reconcile to the frozen physical inventory.')
    devices = []
    for index in range(8):
        t0 = time.monotonic()
        with torch.cuda.device(index):
            props = torch.cuda.get_device_properties(index)
            if 'B200' not in props.name or (props.major, props.minor) != (10, 0):
                raise ValueError('Unexpected B200 architecture: ' + str(props))
            a = torch.ones((128, 128), dtype=torch.bfloat16, device='cuda')
            product = a @ a
            torch.cuda.synchronize()
            if not bool(torch.all(product == 128).item()):
                raise ValueError('Deterministic BF16 matmul did not match the expected result.')
            devices.append({'index': index, 'name': props.name, 'compute_capability': [props.major, props.minor],
                            'total_memory_bytes': props.total_memory, 'elapsed_s': time.monotonic()-t0})
            del a, product
            torch.cuda.empty_cache()
    print('PTX_GPU_PROBE=' + json.dumps({'hostname': socket.gethostname(), 'gpus': devices,
          'nvidia_smi_csv': text, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
          'nccl': torch.cuda.nccl.version(), 'scope': 'Device visibility and BF16 sanity only; no Qwen serving or optimizer.'}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
