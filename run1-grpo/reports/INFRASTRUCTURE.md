# Terminal-Lego infrastructure evidence

Captured on 3 September 2026 from the four assigned GPU workers. Job 190 completed two optimizer updates and saved its checkpoint. This report documents inventory and collection methods; measured results and comparison limits are in [COMPARISON.md](COMPARISON.md).

## Placement

| Slurm worker | Kubernetes node | Pod address | Host address | Role |
| --- | --- | --- | --- | --- |
| gpu-nodes-0 | b200-nodepool-ac23753e6cfa | 10.244.18.139 | 10.6.96.7 | Training |
| gpu-nodes-1 | b200-nodepool-b401f8abb9f1 | 10.244.13.18 | 10.6.96.10 | Training |
| gpu-nodes-2 | b200-nodepool-b615b018060d | 10.244.218.62 | 10.6.96.8 | Inference |
| gpu-nodes-3 | b200-nodepool-83ee4175afdb | 10.244.180.242 | 10.6.96.9 | Inference and task harness |

These are the same physical training/inference node groups used by baseline job 181. Ray may order ranks within each pair differently. Actual rank placement is saved in each attempt's `training.log`.

## Hardware and operating system

| Item | Observed per worker |
| --- | --- |
| GPUs | 8 NVIDIA B200; 32 across the allocation |
| GPU memory reported by driver | 183,359 MiB per GPU |
| GPU power limit | 1,000 W per GPU |
| GPU fabric | Every GPU pair reports NV18 in the NVIDIA topology matrix |
| CPUs | 2 AMD EPYC 9575F 64-core processors; 128 physical cores, 256 logical CPUs |
| Host memory | Approximately 3,023 GiB visible to Linux |
| NUMA | 2 nodes; distance matrix 10 local / 32 remote |
| GPU locality | GPUs 0–3 near NUMA 0; GPUs 4–7 near NUMA 1 |
| GPU driver | 595.71.05 |
| Kernel | 6.8.0-138-generic |
| Worker userspace | Ubuntu 26.04 LTS |
| Slurm | 26.05.2 |

Raw GPU UUIDs, PCI bus IDs, clocks, firmware, ECC/error information, CPU flags, and device topology are retained in `hardware-inventory.json` and `infra-preflight/`. No power limits, clocks, device settings or host networking were changed for this comparison.

## Networking

`ibstat` reports 14 RDMA devices per worker: eight active 400 Gb/s InfiniBand ports, four active 100 Gb/s InfiniBand ports and two active 200 Gb/s Ethernet/RoCE ports. This inventory does **not** establish that all ports are independent physical bandwidth or that the application saturates them.

The 400 Gb/s adapters report MT4129 / firmware 28.43.2026. The Ethernet adapters report MT41692 / firmware 32.42.1000. Full output, GUIDs and the GPU-to-NIC matrix are preserved.

NCCL logs from completed job 190 report version 2.29.7+cuda13.2, `Using network IB`, and GPU Direct RDMA enabled. The plugin enumerates both IB and RoCE devices; bootstrap/control traffic uses pod `eth0`. The baseline also reports the IB network backend. This confirms RDMA initialization, not a measured collective-bandwidth result.

The initial sysfs counter collector found no counter files in the worker namespaces, so its InfiniBand field is empty in the early attempts. Job 190 supplements it with local-port `perfquery -x` and `rdma -j statistic show` every ten seconds, starting during model initialization. These read only the worker's own ports; no remote destination, reset flag or fabric scan is used. Data/packet/error/discard/link counters and RDMA request/retry/error counters are retained. They are node-wide and may include non-training traffic. Counter decreases are treated as resets, not negative traffic. `PortXmitData` and `PortRcvData` deltas are multiplied by four to obtain bytes, as documented in the [perfquery manual](https://man7.org/linux/man-pages/man8/perfquery.8.html).

## Storage and isolation

The original worker Docker daemon uses the `vfs` storage driver. Staging the large Miles image through it caused cumulative layer copies to fill the approximately 1.7 TiB worker disks. Only the new pull processes were stopped. Docker reclaimed its temporary layers; no original run files or baseline images were pruned.

The successful staging uses separate Docker daemons with `fuse-overlayfs`, job-specific container names and isolated sockets at `/tmp/miles-terminal-lego-20260903-2030/docker.sock`. Image storage is on each worker's existing approximately 1.5 TiB `/tmp` tmpfs, consuming about 40 GB per worker. This uses host RAM and is a setup difference to account for. Original Docker settings are unchanged; the task harness continues using its original daemon and task images.

`/shared` is a 4 TiB Lustre filesystem mounted through TCP endpoints. New checkpoints, converted base weights, logs, traces and telemetry go under:

```
/shared/clustermax-campaigns/miles-terminal-lego-20260903-2030
```

The original base-model directory is mounted read-only in Miles containers. The baseline Python environment and task code are reused read-only, with bytecode writes disabled and new cache/output directories. The original trained checkpoint is not used as the new initialization.

## Effective resource constraints

The Kubernetes Slurm pods are privileged and request eight GPUs plus one RDMA resource. They do not use host PID, IPC or network namespaces. Inner Miles containers share their worker pod's network and IPC namespaces, so their `--network host` is not the Kubernetes node network namespace.

Container constraint snapshots, including job 190 initialization and first-update snapshots, report CPUs 0–255 and both NUMA nodes available, with no container-level CPU quota or memory cap. Ray advertises 64 scheduling CPUs per worker, which is not an OS CPU quota. Slurm reserves all four workers exclusively. `/dev/shm` is 16 GiB; each Ray object store is configured for 4 GiB.

Megatron reports successful per-GPU CPU affinity setup. SGLang reports that its automatic NUMA affinity could not be applied in the inner container. This is retained as a possible CPU-locality performance difference. It is not evidence that GPU placement or the RDMA transport failed.

## Software provenance

| Component | Pinned or observed value |
| --- | --- |
| Active Miles source | `70b89e11770fc9bac984e22cfff89c51cca44203` |
| Container image | `radixark/miles@sha256:4ee6da9f16e06f8ad24991b18a950482572c458a357aae0bfc396feaf3fe0a6d` |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0.3 |
| Megatron Core | 0.19.0+8c1e05747 |
| Megatron-LM source | `8c1e05747eb612b382df2632783df5c83a853646` |
| SGLang | 0.5.19.dev54+gc16b821 |
| SGLang source | `c16b821ef3177a688a073c173b44c0ce48b5bf3e` |
| Transformer Engine | 2.17.0 |
| Ray | 2.58.0 |
| Transformers | 5.12.1 |
| FlashInfer | 0.6.17 |
| Triton | 3.7.1 |

The active Miles checkout is mounted separately and takes precedence over the older Miles checkout bundled into the image. Per-container package inventories and software revision evidence are retained with the run.

## What is measured

Before/after snapshots, two-second GPU utilization/memory/power/temperature/clocks, GPU processes, CPU and memory pressure, network and disk counters, IB counters, NCCL logs, Ray logs, harness episodes and token traces, TensorBoard scalars, exact argv, source hashes, command timelines and Slurm accounting are retained.

The comparison must separate one-time image staging, model conversion, inference initialization, rollouts, optimizer work, weight synchronization, checkpoint writes and allocated idle time. Slurm task RSS is not treated as complete application memory accounting because the nested Docker daemon launches the GPU processes; GPU telemetry and container/process snapshots are the primary memory evidence. No intrusive network/storage benchmark is run alongside the workload. Baseline job 181's approximately 3.5-hour telemetry-only tail is not counted as training time.

## Preservation

The original run is not deleted or resumed. After completed job 190, all 49 hashed source/config files and 188 of 189 inventoried artifact metadata entries were unchanged. The one growing file is the baseline's pre-existing `sstat` collector log, which continues appending errors after its allocation ended. That process is left untouched.

Each new attempt has its own directory and source snapshot. Failed attempts, their logs and stopped containers are retained rather than overwritten. The final audit is saved as `baseline-preservation-after-job-190.json`; large artifacts were checked by size and mtime, not full content hash.

## Compatibility failures retained

Job 187 failed in SGLang breakable prefill CUDA-graph capture. Disabling that prefill graph path allowed job 188 to initialize both training and inference, including decode graphs. Job 188 then failed on the initial expert-weight update, reporting tensor dimensions 64 versus 2048. The pinned Miles Blackwell Qwen3.5 documentation describes the same error and selects `flashinfer_cutlass`; jobs 189 and 190 use that MoE kernel. These changes affect performance and are not hidden as identical backend settings. Neither failed attempt completed a task rollout or optimizer update.

Job 190 additionally records a narrowly scoped, job-local SGLang router patch: BF16-input GEMM with FP32 output, preserving BF16 parameter storage and weight-update layout. Five GPU shape checks and CUDA-graph replay passed. SGLang FP32 LM-head output is enabled. All before/after patch source and hashes are saved; neither the baseline environment nor the shared Miles source is patched.

## Final collection coverage and health

Job 190 retained 32 NCCL log files, PMA/RDMA samples, GPU/CPU/memory/disk/network time series, Prometheus snapshots, per-container runtime constraints, package inventories and before/after device snapshots. `infrastructure-runtime-summary.json` contains per-node and trainer-window GPU statistics and per-port counter deltas with timestamps.

The eight 400-Gb/s IB ports per node responded to local PMA queries; the four 100-Gb/s ports timed out. No fabric scan or counter reset was attempted. This is a coverage limitation, not evidence that the nonresponding ports are unusable. Captured link/error/discard counters and RDMA hardware error counters did not increase on the monitored devices. Transmit-wait counters did increase; no congestion diagnosis is inferred from those raw values. PMA deltas are node-wide and cover a longer window than the ready-to-checkpoint GPU summaries.

The final Kubernetes snapshot reports all four assigned nodes Ready and no DiskPressure, MemoryPressure or PIDPressure. Host node OS is Ubuntu 24.04.4 LTS; the Ubuntu 26.04 value above describes the worker container userspace, not the host. At 22:15:53 UTC, read-only checks found no Slurm jobs, GPU compute processes, running Miles containers or campaign collectors. Isolated image daemons and stored images remain; no retained evidence or original monitor was removed.

SGLang connection-reset traces occurred during final disposal after the checkpoint and final weight sync. They are retained in the evidence; successful Slurm exit does not mean the logs contain no warnings.
