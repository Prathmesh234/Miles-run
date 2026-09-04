# Infrastructure charts — job 190

These original charts are preserved unchanged from commit `f7ec463e3c28429578118a0aba9ddff3e13a9684`. The raw `evidence-job-190/job-190/` files now live in the retained archive and earlier Git history, not the current compact checkout. Regenerate with `python3 plot_infra_charts.py --evidence /path/to/job-190 --output /path/to/charts` (requires matplotlib and numpy). See the [new PPO comparison](../results/REPORT.md) for matched-run charts. The original charts below use their original allocation-wide windows; the new comparison states its distinct ready-to-checkpoint window explicitly.

Shaded bands on time-series charts are the two trainer update windows from `infrastructure-runtime-summary.json`; dotted lines are `training_start` / `training_end` from `timeline.jsonl`. All times are UTC, 3 September 2026.

| Chart | Source files | What it shows |
| --- | --- | --- |
| [run-phase-timeline.png](run-phase-timeline.png) | `timeline.jsonl`, `infrastructure-runtime-summary.json`, `infra/sglang-prometheus.jsonl` | Wall-clock breakdown of the 14.6 min coordinator run: setup, Megatron/SGLang init, two rollouts, two updates, checkpoint, teardown |
| [gpu-utilization-by-node.png](gpu-utilization-by-node.png) | `infra/gpu-nodes-*-timeseries.jsonl` | Node-mean SM utilization with min–max band across the 8 GPUs per node |
| [gpu-utilization-heatmap.png](gpu-utilization-heatmap.png) | `infra/gpu-nodes-*-timeseries.jsonl` | Per-GPU SM utilization for all 32 B200s |
| [gpu-memory-by-node.png](gpu-memory-by-node.png) | `infra/gpu-nodes-*-timeseries.jsonl` | GPU memory used per node against the 179 GiB driver-reported capacity |
| [gpu-power-temperature-by-node.png](gpu-power-temperature-by-node.png) | `infra/gpu-nodes-*-timeseries.jsonl` | Node GPU power (sum of 8) against the 8 kW limit, and hottest GPU per node |
| [host-cpu-memory-by-node.png](host-cpu-memory-by-node.png) | `infra/gpu-nodes-*-timeseries.jsonl` (`/proc/stat`, `/proc/meminfo`) | Host CPU busy % and host memory used per worker |
| [infiniband-throughput-by-node.png](infiniband-throughput-by-node.png) | `infra/gpu-nodes-*-rdma-counters.jsonl` | TX/RX Gb/s summed over the 8 responding 400 Gb/s IB ports per node (10 s PMA deltas × 4 bytes) |
| [infiniband-per-port-totals.png](infiniband-per-port-totals.png) | `infrastructure-runtime-summary.json` | Total TX bytes and `PortXmitWait` delta per local IB device |
| [sglang-inference-metrics.png](sglang-inference-metrics.png) | `infra/sglang-prometheus.jsonl` | Cumulative prompt/generated tokens, decode throughput, running and queued requests |
| [sglang-kv-cache.png](sglang-kv-cache.png) | `infra/sglang-prometheus.jsonl` | KV tokens in use against pool capacity, and prefix-cache hits versus prompt tokens |

Caveats carried over from [INFRASTRUCTURE.md](../INFRASTRUCTURE.md): PMA counters are node-wide and may include non-training traffic; the four 100 Gb/s ports timed out and are excluded; SGLang metrics were scraped from the single router endpoint on gpu-nodes-3 and gauges are read from `tp_rank` 0; token counters are engine totals, not accepted-trace counts. This is a two-update smoke test, not a controlled benchmark.
