#!/usr/bin/env python3
"""Render infrastructure telemetry charts for completed Slurm job 190.

Reads only files committed in this repository:
  evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl    (2 s nvidia-smi + /proc samples)
  evidence-job-190/job-190/infra/gpu-nodes-*-rdma-counters.jsonl (10 s perfquery PMA samples)
  evidence-job-190/job-190/infra/sglang-prometheus.jsonl         (10 s SGLang /metrics scrapes)
  evidence-job-190/job-190/infrastructure-runtime-summary.json   (trainer update windows)
  evidence-job-190/job-190/timeline.jsonl                        (coordinator phase markers)

Writes PNG files to charts/. Usage: python3 plot_infra_charts.py
"""
from __future__ import annotations

import json
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
EVID = ROOT / "evidence-job-190" / "job-190"
INFRA = EVID / "infra"
OUT = ROOT / "charts"
OUT.mkdir(exist_ok=True)

NODES = ["gpu-nodes-0", "gpu-nodes-1", "gpu-nodes-2", "gpu-nodes-3"]
ROLE = {
    "gpu-nodes-0": "training",
    "gpu-nodes-1": "training",
    "gpu-nodes-2": "inference",
    "gpu-nodes-3": "inference + harness",
}
COLORS = {
    "gpu-nodes-0": "#20808D",
    "gpu-nodes-1": "#1B474D",
    "gpu-nodes-2": "#A84B2F",
    "gpu-nodes-3": "#FFC553",
}
TEXT = "#28251D"
MUTED = "#7A7974"
GRID = "#D4D1CA"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "legend.frameon": False,
    }
)


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- phases
summary = json.loads((EVID / "infrastructure-runtime-summary.json").read_text())
UPDATE_WINDOWS = [(utc(a), utc(b)) for a, b in summary["training_windows_epoch"]]

timeline = [json.loads(l) for l in (EVID / "timeline.jsonl").read_text().splitlines() if l.strip()]
EVENTS = {}
for ev in timeline:
    EVENTS.setdefault(ev["event"], parse_iso(ev["time"]))
T_TRAIN_START = EVENTS["training_start"]
T_TRAIN_END = EVENTS["training_end"]
T_RUN_START = EVENTS["infra-before_start"]
T_RUN_END = EVENTS["coordinator_exit"]


def shade_phases(ax, label=True):
    """Shade the two optimizer-update windows and mark training start/end."""
    for i, (a, b) in enumerate(UPDATE_WINDOWS):
        ax.axvspan(a, b, color="#20808D", alpha=0.12, lw=0)
        if label:
            ax.text(
                a + (b - a) / 2, 1.01, f"update {i + 1}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#0C4E54", clip_on=False,
            )
    for t, name, ha in ((T_TRAIN_START, "train start ", "right"), (T_TRAIN_END, " train exit", "left")):
        ax.axvline(t, color=MUTED, ls=":", lw=0.9)
        if label:
            ax.text(t, 1.01, name, transform=ax.get_xaxis_transform(), ha=ha, va="bottom",
                    fontsize=8, color=MUTED, clip_on=False)


def fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    ax.set_xlabel("UTC, 3 September 2026")


def footer(fig, note):
    fig.text(0.01, -0.02, textwrap.fill(note, 150), ha="left", va="top", fontsize=7.5, color=MUTED)


# --------------------------------------------------------------------------- GPU + host time series
def load_timeseries(node: str):
    rows = []
    prev_cpu = None
    prev_net = None
    for line in (INFRA / f"{node}-timeseries.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = parse_iso(r["timestamp"])
        gpu = r["gpu"]
        if gpu.get("exit_code") != 0:
            continue
        per_gpu = []
        for gl in gpu["stdout"].strip().splitlines():
            f = [x.strip() for x in gl.split(",")]
            per_gpu.append(
                dict(idx=int(f[0]), util=float(f[3]), mem_util=float(f[4]), mem_used=float(f[5]),
                     mem_total=float(f[6]), power=float(f[7]), temp=float(f[9]), sm_clk=float(f[10]))
            )
        sysd = r.get("system", {})
        # CPU busy fraction from /proc/stat aggregate line
        cpu_line = sysd.get("/proc/stat", "").splitlines()[0].split()
        cpu_vals = list(map(int, cpu_line[1:])) if cpu_line and cpu_line[0] == "cpu" else None
        cpu_busy = None
        if cpu_vals and prev_cpu:
            d = [a - b for a, b in zip(cpu_vals, prev_cpu)]
            idle = d[3] + d[4]
            total = sum(d[:8])
            cpu_busy = 100.0 * (total - idle) / total if total > 0 else None
        if cpu_vals:
            prev_cpu = cpu_vals
        mem_used_gib = None
        mi = {}
        for ml in sysd.get("/proc/meminfo", "").splitlines():
            k, v = ml.split(":", 1)
            mi[k] = int(v.split()[0])
        if "MemTotal" in mi and "MemAvailable" in mi:
            mem_used_gib = (mi["MemTotal"] - mi["MemAvailable"]) / 1024 / 1024
        load1 = None
        la = sysd.get("/proc/loadavg", "").split()
        if la:
            load1 = float(la[0])
        # Ethernet (pod eth0) bytes for bootstrap/control traffic
        net = None
        for nl in sysd.get("/proc/net/dev", "").splitlines():
            if nl.strip().startswith("eth0:"):
                f = nl.split(":", 1)[1].split()
                net = (int(f[0]), int(f[8]), t)
        net_rx_gbps = net_tx_gbps = None
        if net and prev_net:
            dt = (net[2] - prev_net[2]).total_seconds()
            if dt > 0:
                net_rx_gbps = (net[0] - prev_net[0]) * 8 / dt / 1e9
                net_tx_gbps = (net[1] - prev_net[1]) * 8 / dt / 1e9
        if net:
            prev_net = net
        rows.append(dict(t=t, gpus=per_gpu, cpu_busy=cpu_busy, mem_used_gib=mem_used_gib, load1=load1,
                         net_rx_gbps=net_rx_gbps, net_tx_gbps=net_tx_gbps))
    return rows


TS = {n: load_timeseries(n) for n in NODES}


def series(node, key, agg=np.mean):
    t = [r["t"] for r in TS[node]]
    v = [agg([g[key] for g in r["gpus"]]) for r in TS[node]]
    return t, np.array(v)


# ---- Chart 1: GPU utilization per node
fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, layout="constrained")
for ax, n in zip(axes, NODES):
    t, mean = series(n, "util")
    _, mx = series(n, "util", np.max)
    _, mn = series(n, "util", np.min)
    ax.fill_between(t, mn, mx, color=COLORS[n], alpha=0.18, lw=0, label="min–max across 8 GPUs")
    ax.plot(t, mean, color=COLORS[n], lw=1.4, label="node mean")
    ax.set_ylim(0, 105)
    ax.set_ylabel("SM util %")
    ax.set_title(f"{n}  ({ROLE[n]})", fontsize=10.5, pad=14)
    shade_phases(ax, label=(n == NODES[0]))
axes[0].legend(loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0.0, 1.45))
fmt_time_axis(axes[-1])
fig.suptitle("Trainer GPUs are busy only in short bursts inside the update windows; inference GPUs saturate during rollouts",
             fontsize=13, fontweight="bold", x=0.01, ha="left")
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl (nvidia-smi utilization.gpu, 2 s sampling). "
            "Shaded bands: trainer update windows from infrastructure-runtime-summary.json. Job 190, 32× B200, Qwen3.6-35B-A3B.")
fig.savefig(OUT / "gpu-utilization-by-node.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 2: per-GPU utilization heatmap (32 rows)
all_t = sorted({r["t"] for n in NODES for r in TS[n]})
t0 = all_t[0]
grid_t = np.arange(0, (all_t[-1] - t0).total_seconds() + 2, 2.0)
heat = np.full((32, len(grid_t)), np.nan)
for ni, n in enumerate(NODES):
    for r in TS[n]:
        col = int(round((r["t"] - t0).total_seconds() / 2.0))
        if 0 <= col < len(grid_t):
            for g in r["gpus"]:
                heat[ni * 8 + g["idx"], col] = g["util"]
fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
x_dates = [t0 + np.timedelta64(int(s), "s").astype("timedelta64[s]").item() for s in grid_t]
x_num = mdates.date2num(x_dates)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#E9E7E1")
im = ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
               extent=[x_num[0], x_num[-1], 31.5, -0.5])
ax.xaxis_date(tz=timezone.utc)
ax.set_yticks([3.5, 11.5, 19.5, 27.5])
ax.set_yticklabels([f"{n}\n({ROLE[n]})" for n in NODES], fontsize=9)
for y in (7.5, 15.5, 23.5):
    ax.axhline(y, color="white", lw=1.2)
ax.grid(False)
for a, b in UPDATE_WINDOWS:
    ax.axvline(a, color="white", ls="--", lw=0.9)
    ax.axvline(b, color="white", ls="--", lw=0.9)
ax.axvline(T_TRAIN_START, color="white", ls=":", lw=0.9)
ax.axvline(T_TRAIN_END, color="white", ls=":", lw=0.9)
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
cb.set_label("SM utilization %")
fmt_time_axis(ax)
ax.set_title("Per-GPU utilization, all 32 B200s (rows = GPU index 0–7 within each node)\n"
             "Dashed: optimizer update windows · dotted: training_entry start/exit · light grey: no sample")
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl (nvidia-smi utilization.gpu, 2 s sampling). "
            "Training nodes 0–1 run Megatron; inference nodes 2–3 run SGLang. Job 190.")
fig.savefig(OUT / "gpu-utilization-heatmap.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 3: GPU memory per node
fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")
for n in NODES:
    t, mean = series(n, "mem_used")
    _, mx = series(n, "mem_used", np.max)
    _, mn = series(n, "mem_used", np.min)
    ax.fill_between(t, mn / 1024, mx / 1024, color=COLORS[n], alpha=0.15, lw=0)
    ax.plot(t, mean / 1024, color=COLORS[n], lw=1.5, label=f"{n} ({ROLE[n]})")
ax.axhline(183359 / 1024, color=MUTED, ls="--", lw=0.9)
ax.text(T_RUN_START, 183359 / 1024 + 2, "driver-reported capacity 179 GiB", fontsize=8, color=MUTED, va="bottom")
ax.set_ylim(0, 195)
ax.set_ylabel("GPU memory used (GiB, node mean; band = min–max)")
shade_phases(ax)
ax.legend(loc="center left", fontsize=8.5)
fmt_time_axis(ax)
ax.set_title("SGLang inference GPUs hold ~155 GiB steadily; Megatron trainer GPUs step from ~40 to 85–115 GiB across the two updates", pad=16)
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl (nvidia-smi memory.used, 2 s sampling). "
            "Job 190, Qwen3.6-35B-A3B, Miles + Megatron trainer + SGLang inference.")
fig.savefig(OUT / "gpu-memory-by-node.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 4: GPU power + temperature
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, layout="constrained")
for n in NODES:
    t, p = series(n, "power", np.sum)
    axes[0].plot(t, p / 1000, color=COLORS[n], lw=1.5, label=f"{n} ({ROLE[n]})")
    t, tmp = series(n, "temp", np.max)
    axes[1].plot(t, tmp, color=COLORS[n], lw=1.5)
axes[0].axhline(8.0, color=MUTED, ls="--", lw=0.9)
axes[0].text(T_RUN_START, 8.05, "8 GPUs × 1,000 W power limit", fontsize=8, color=MUTED, va="bottom")
axes[0].set_ylim(0, 8.6)
axes[0].set_ylabel("Node GPU power (kW, sum of 8 GPUs)")
axes[0].set_title("Node GPU power peaked around 5 kW of an 8 kW limit; hottest GPU never exceeded 45 °C", pad=16)
axes[0].legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.92))
axes[1].set_ylabel("Hottest GPU on node (°C)")
axes[1].set_ylim(25, 50)
for ax in axes:
    shade_phases(ax, label=(ax is axes[0]))
fmt_time_axis(axes[-1])
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl (nvidia-smi power.draw, temperature.gpu, 2 s sampling). "
            "Power limit and clocks unchanged for the comparison (see INFRASTRUCTURE.md). Job 190.")
fig.savefig(OUT / "gpu-power-temperature-by-node.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 5: host CPU / memory
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, layout="constrained")
for n in NODES:
    t = [r["t"] for r in TS[n] if r["cpu_busy"] is not None]
    v = [r["cpu_busy"] for r in TS[n] if r["cpu_busy"] is not None]
    axes[0].plot(t, v, color=COLORS[n], lw=1.3, label=f"{n} ({ROLE[n]})")
    t = [r["t"] for r in TS[n] if r["mem_used_gib"] is not None]
    v = [r["mem_used_gib"] for r in TS[n] if r["mem_used_gib"] is not None]
    axes[1].plot(t, v, color=COLORS[n], lw=1.3)
axes[0].set_ylabel("Host CPU busy % (256 logical CPUs)")
axes[0].set_ylim(0, max(30, axes[0].get_ylim()[1]))
axes[0].set_title("Host CPU and memory stayed far below capacity on all four workers", pad=16)
axes[0].legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.92))
axes[1].set_ylabel("Host memory used (GiB, MemTotal − MemAvailable)")
axes[1].set_ylim(0, max(400, axes[1].get_ylim()[1]))
for ax in axes:
    shade_phases(ax, label=(ax is axes[0]))
fmt_time_axis(axes[-1])
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-timeseries.jsonl (/proc/stat and /proc/meminfo deltas, 2 s sampling). "
            "Host memory includes the ~40 GB per-worker Miles image staged on tmpfs. Approximately 3,023 GiB visible per worker. Job 190.")
fig.savefig(OUT / "host-cpu-memory-by-node.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- InfiniBand PMA counters
PQ_RE = re.compile(r"^(\w+):\.*\s*(\S+)$", re.M)


def load_rdma(node):
    per_port = defaultdict(list)  # port -> [(t, xmit_bytes, rcv_bytes, xmit_wait)]
    for line in (INFRA / f"{node}-rdma-counters.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = parse_iso(r["utc"])
        for p in r.get("ports", []):
            if p.get("status") != 0:
                continue
            dev = p["argv"][p["argv"].index("-C") + 1]
            vals = {}
            for m in PQ_RE.finditer(p["stdout"]):
                try:
                    vals[m.group(1)] = int(m.group(2), 0)
                except ValueError:
                    pass
            if "PortXmitData" in vals:
                per_port[dev].append((t, vals["PortXmitData"] * 4, vals["PortRcvData"] * 4, vals.get("PortXmitWait", 0)))
    return per_port


RDMA = {n: load_rdma(n) for n in NODES}


def node_ib_rate(node):
    """Aggregate TX/RX Gb/s across all responding local IB ports, per 10 s sample."""
    rate = defaultdict(lambda: [0.0, 0.0])
    for dev, samples in RDMA[node].items():
        for (t1, x1, r1, _), (t2, x2, r2, _) in zip(samples, samples[1:]):
            dt = (t2 - t1).total_seconds()
            dx, dr = x2 - x1, r2 - r1
            if dt <= 0 or dx < 0 or dr < 0:  # counter reset -> skip
                continue
            rate[t2][0] += dx * 8 / dt / 1e9
            rate[t2][1] += dr * 8 / dt / 1e9
    ts = sorted(rate)
    return ts, np.array([rate[t][0] for t in ts]), np.array([rate[t][1] for t in ts])


fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
for ax, n in zip(axes, NODES):
    ts, tx, rx = node_ib_rate(n)
    nports = len(RDMA[n])
    ax.plot(ts, tx, color=COLORS[n], lw=1.5, label="TX (PortXmitData × 4 B)")
    ax.plot(ts, rx, color=COLORS[n], lw=1.2, ls="--", label="RX (PortRcvData × 4 B)")
    ax.set_ylabel("Gb/s (sum of ports)")
    ax.set_title(f"{n}  ({ROLE[n]}) — {nports} responding 400 Gb/s IB ports", fontsize=10.5, pad=14)
    shade_phases(ax, label=(n == NODES[0]))
    ax.set_ylim(bottom=0)
axes[0].legend(loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0.0, 1.45))
fmt_time_axis(axes[-1])
fig.suptitle("InfiniBand traffic: trainer nodes burst to ~60–115 Gb/s at update boundaries; inference nodes see ~20–30 Gb/s during rollouts",
             fontsize=13, fontweight="bold", x=0.01, ha="left")
footer(fig, "Source: evidence-job-190/job-190/infra/gpu-nodes-*-rdma-counters.jsonl (perfquery -x, PortXmitData/PortRcvData deltas × 4 bytes). "
            "Counters are node-wide and may include non-training traffic; four 100 Gb/s ports timed out and are excluded. "
            "Aggregate of ~3.2 Tb/s theoretical across 8 ports. Job 190.")
fig.savefig(OUT / "infiniband-throughput-by-node.png", bbox_inches="tight")
plt.close(fig)

# ---- Chart 7: per-port total bytes + XmitWait (bar)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
port_order = None
width = 0.2
for i, n in enumerate(NODES):
    ports = summary["nodes"][n]["rdma_ports"]
    names = sorted(ports, key=lambda d: int(d.split("_")[1]))
    if port_order is None:
        port_order = names
    x = np.arange(len(names))
    tx_gb = [ports[p]["tx_bytes"] / 1e9 for p in names]
    wait = [ports[p]["counter_deltas"].get("PortXmitWait", 0) / 1e3 for p in names]
    axes[0].bar(x + (i - 1.5) * width, tx_gb, width, color=COLORS[n], label=f"{n} ({ROLE[n]})")
    axes[1].bar(x + (i - 1.5) * width, wait, width, color=COLORS[n])
for ax in axes:
    ax.set_xticks(np.arange(len(port_order)))
    ax.set_xticklabels(port_order, rotation=0, fontsize=9)
    ax.set_xlabel("Local IB device")
axes[0].set_ylabel("Bytes transmitted over collection window (GB)")
axes[0].set_title("NCCL spreads ~37–52 GB per port; extra TX volume lands on mlx5_1 (nodes 1–3) and mlx5_3 (node 3)", fontsize=11)
axes[0].legend(fontsize=8.5)
axes[1].set_ylabel("PortXmitWait delta (thousands of ticks)")
axes[1].set_title("Transmit-wait ticks accrued on every port (raw; no congestion diagnosis inferred)", fontsize=11)
footer(fig, "Source: evidence-job-190/job-190/infrastructure-runtime-summary.json (rdma_ports.tx_bytes and counter_deltas.PortXmitWait). "
            "Window ≈ 22:00:19–22:11:37 UTC, node-wide local ports. Link, error, discard and RDMA HW-error counters did not increase. Job 190.")
fig.savefig(OUT / "infiniband-per-port-totals.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- SGLang Prometheus
PROM_RE = re.compile(r'^(sglang:\w+)(\{[^}]*\})?\s+(\S+)$')


def parse_prom(text):
    out = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = PROM_RE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2) or ""))
        out[m.group(1)].append((labels, v))
    return out


prom = []
for line in (INFRA / "sglang-prometheus.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("status") != 200:
        continue
    prom.append((parse_iso(r["utc"]), parse_prom(r["text"])))

pt = [t for t, _ in prom]


def gauge_sum(name, label_filter=None):
    vals = []
    for _, d in prom:
        s = 0.0
        for labels, v in d.get(name, []):
            if label_filter and not label_filter(labels):
                continue
            s += v
        vals.append(s)
    return np.array(vals)


def gauge_rank0(name):
    return gauge_sum(name, lambda l: l.get("tp_rank") == "0" and l.get("dp_rank", "0") == "0")


gen_tok = gauge_sum("sglang:generation_tokens_total")
prompt_tok = gauge_sum("sglang:prompt_tokens_total")
running = gauge_rank0("sglang:num_running_reqs")
queued = gauge_rank0("sglang:num_queue_reqs")
kv_used = gauge_rank0("sglang:kv_used_tokens")
kv_cap = gauge_rank0("sglang:max_total_num_tokens")
cached_tok = gauge_sum("sglang:cached_tokens_total")
gen_thr = gauge_rank0("sglang:gen_throughput")

fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
axes[0].plot(pt, prompt_tok / 1e3, color="#1B474D", lw=1.6, label="prompt tokens (cumulative)")
axes[0].plot(pt, gen_tok / 1e3, color="#A84B2F", lw=1.6, label="generated tokens (cumulative)")
axes[0].set_ylabel("Tokens (thousands)")
axes[0].legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.92))
axes[0].set_title(f"SGLang served {prompt_tok[-1]/1e3:,.0f}K prompt tokens and {gen_tok[-1]/1e3:,.0f}K generated tokens across two rollout phases", pad=16)
axes[1].plot(pt, gen_thr, color="#20808D", lw=1.6, label="decode throughput (tok/s, scheduler gauge)")
axes[1].set_ylabel("Generation throughput (tok/s)")
axes[1].set_ylim(bottom=0)
axes[1].legend(loc="upper left", fontsize=8.5)
axes[2].step(pt, running, where="post", color="#20808D", lw=1.6, label="running requests")
axes[2].step(pt, queued, where="post", color="#944454", lw=1.3, label="queued requests")
ax2b = axes[2]
ax2b.set_ylabel("Requests")
ax2b.set_ylim(bottom=0)
ax2b.legend(loc="upper left", fontsize=8.5)
for ax in axes:
    shade_phases(ax, label=(ax is axes[0]))
fmt_time_axis(axes[-1])
footer(fig, "Source: evidence-job-190/job-190/infra/sglang-prometheus.jsonl (SGLang /metrics on 10.244.180.242:15000, 10 s scrapes; "
            "scrapes before 22:02 UTC returned connection refused while the engine initialized). Gauges read from tp_rank 0. "
            "Token totals are engine counters, not accepted-trace counts. Job 190.")
fig.savefig(OUT / "sglang-inference-metrics.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout="constrained")
cap = float(np.nanmax(kv_cap))
peak = float(np.nanmax(kv_used))
axes[0].plot(pt, kv_used / 1e3, color="#20808D", lw=1.6)
axes[0].set_ylabel("KV tokens in use (thousands)")
axes[0].set_ylim(0, peak / 1e3 * 1.3)
axes[0].set_title(f"KV cache peaked at {peak/1e3:,.1f}K tokens of a {cap/1e6:,.2f}M-token pool ({peak/cap*100:.2f}%) — "
                  "rollouts of 16 traces barely touch B200 memory", pad=16)
axes[0].annotate(f"peak {peak/1e3:,.1f}K tokens", xy=(pt[int(np.nanargmax(kv_used))], peak / 1e3),
                 xytext=(10, 6), textcoords="offset points", fontsize=8.5, color="#0C4E54")
axes[1].plot(pt, prompt_tok / 1e3, color="#1B474D", lw=1.6, label="prompt tokens (cumulative)")
axes[1].plot(pt, cached_tok / 1e3, color="#A84B2F", lw=1.6, label="prefix-cache hits (cumulative cached_tokens_total)")
share = cached_tok[-1] / prompt_tok[-1] * 100 if prompt_tok[-1] else 0
axes[1].set_ylabel("Tokens (thousands)")
axes[1].set_title(f"{share:.0f}% of prompt tokens were served from the prefix cache (multi-turn agent contexts re-use earlier turns)", fontsize=11)
axes[1].legend(loc="upper left", fontsize=8.5)
for ax in axes:
    shade_phases(ax, label=(ax is axes[0]))
fmt_time_axis(axes[-1])
footer(fig, "Source: evidence-job-190/job-190/infra/sglang-prometheus.jsonl (sglang:kv_used_tokens, sglang:max_total_num_tokens on tp_rank 0; "
            "sglang:prompt_tokens_total and sglang:cached_tokens_total engine counters; 10 s scrapes of 10.244.180.242:15000). Job 190.")
fig.savefig(OUT / "sglang-kv-cache.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- Run phase timeline
phases = []
def add_phase(label, a, b, color):
    phases.append((label, a, b, color))

add_phase("Container start + precision patch (4 nodes)", EVENTS["container-start-gpu-nodes-0_start"], EVENTS["precision-patch-gpu-nodes-3_end"], "#BCE2E7")
add_phase("Precision GPU validation", EVENTS["precision-gpu-validation_start"], EVENTS["precision-gpu-validation_end"], "#BCE2E7")
add_phase("Argument validation", EVENTS["argument-validation_start"], EVENTS["argument-validation_end"], "#BCE2E7")
add_phase("Harness ready → Ray start (4 nodes)", EVENTS["harness_ready"], EVENTS["ray-status_end"], "#BCE2E7")
first_prom_ok = pt[0]
add_phase("training_entry: Megatron + SGLang init", T_TRAIN_START, first_prom_ok, "#848456")
add_phase("Rollout 1 (16 traces, 3 groups)", first_prom_ok, UPDATE_WINDOWS[0][0], "#A84B2F")
add_phase("Optimizer update 1 (incl. first-step warmup)", *UPDATE_WINDOWS[0], "#20808D")
add_phase("Weight sync → Rollout 2 (16 traces)", UPDATE_WINDOWS[0][1], UPDATE_WINDOWS[1][0], "#A84B2F")
add_phase("Optimizer update 2 (19.0 s)", *UPDATE_WINDOWS[1], "#20808D")
add_phase("Checkpoint save + final weight sync + shutdown", UPDATE_WINDOWS[1][1], T_TRAIN_END, "#1B474D")
add_phase("Container evidence + stop (4 nodes)", EVENTS["container-evidence-gpu-nodes-0_start"], EVENTS["container-stop-gpu-nodes-3_end"], "#BCE2E7")
add_phase("Infra-after snapshot + Slurm accounting", EVENTS["infra-after_start"], EVENTS["coordinator_exit"], "#BCE2E7")

fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
for i, (label, a, b, color) in enumerate(phases):
    ax.barh(i, mdates.date2num(b) - mdates.date2num(a), left=mdates.date2num(a), height=0.6, color=color)
    dur = (b - a).total_seconds()
    ax.text(mdates.date2num(b), i, f"  {dur/60:.1f} min" if dur >= 60 else f"  {dur:.0f} s", va="center", fontsize=8.5, color=MUTED)
ax.set_yticks(range(len(phases)))
ax.set_yticklabels([p[0] for p in phases], fontsize=9)
ax.invert_yaxis()
ax.xaxis_date(tz=timezone.utc)
fmt_time_axis(ax)
ax.grid(axis="y", visible=False)
total = (T_RUN_END - T_RUN_START).total_seconds() / 60
ax.set_title(f"Job 190 wall-clock breakdown: {total:.1f} min coordinator run, 14m34s Slurm allocation, two optimizer updates")
footer(fig, "Source: evidence-job-190/job-190/timeline.jsonl (coordinator events), infrastructure-runtime-summary.json (trainer windows), "
            "sglang-prometheus.jsonl (first successful /metrics scrape marks inference-ready). Rollout/update boundaries are approximate; "
            "exact per-step timings are in COMPARISON.md. Job 190.")
fig.savefig(OUT / "run-phase-timeline.png", bbox_inches="tight")
plt.close(fig)

print("Wrote:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.relative_to(ROOT))
