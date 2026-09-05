# CollectiveX — Vultr B200

**[Open the offline HTML dashboard](index.html).** Download the HTML and open it
in any modern browser; it embeds its data and makes no network requests.

Four independent EP8 allocations, jobs **221–224**, completed on 5 September
2026 UTC. All **112 measured points** and **64 execution phases** passed;
32 distinct GPUs were reconciled and released. Zero collector errors were
recorded; maximum GPU sampling interval was 1.415 seconds.

At 8,192 tokens/GPU, chained pair-period p50 was **2.265–2.275 ms BF16** versus
**1.749–1.756 ms FP8**, a 22.75–22.84% reduction on the same node. FP8 was
6.64–16.18% slower at one token/GPU. Combine remains BF16 in both cases.

## Contents

- `index.html`: self-contained interactive plots, tables, CSV/JSON/SVG export.
- `metrics.json`: allowlisted numeric measurements, sampling traces and pins.
- `render.py`: one standard-library importer/renderer; no cluster mutation.
- `node-comparison.svg`, `infrastructure.svg`: standalone exportable figures.

The controls follow the [CollectiveX dashboard](https://inferencex.com/collectivex):
source tokens per rank, dispatch/combine/measured round trip, p50/p90/p95/p99,
and payload/activation rates. The additional chained-pair view stays distinct
from drained round trip. The discussion in Slack was a presentation reference,
not measurement data; private messages are not published here.

## Interpretation

This is a synthetic DeepSeek-V3 communication shape (hidden 7,168, 256 experts,
top-k 8, uniform routing, seed 67), **not Qwen inference or an RL quality run**.
One timing-budget repetition per node cannot establish repeat-run stability.
The largest-size cross-node p50 CV is 0.18% BF16 and 0.15% FP8.

Logical payload GB/s/GPU is routed bytes divided by selected latency and eight
GPUs. Rate at p99 latency is not p99 bandwidth. Native correctness uses its
numerical tolerance, not bit-exact equality. GPU utilization is busy time, not
useful model FLOPs. Missing time bins stay null and break plot lines.

NVLink counter-rate peaks are observational and unvalidated as physical link
bandwidth; an isolated 7,993.7 GB/s node-total estimate is retained explicitly.
IB counters do **not** qualify inter-node performance: all EP8 payload is local
to an NVLink island. No KV-transfer or EP32 case was run.

CollectiveX was **not registered in the inspected CMAX CLI** at
`b308f8ab424b866c6704df5fc7fe21bf7327b1f0`; its DeepEP comm-lib-usability smoke
suite is different. These runs used the standalone InferenceX harness.

## Rebuild and verify

```sh
python3 collectivex/render.py
python3 collectivex/render.py --check
python3 -m unittest discover -s collectivex -p 'test_*.py' -v
```

To reimport the audited campaign bundle, run `render.py --import-campaign PATH`.
The importer requires the fleet audit, publication receipt, native case JSON,
and compressed telemetry. It does not import secrets, internal addresses,
GPU UUIDs or Slack transcripts.

The separate **27.1 MB** complete evidence archive is
`collectivex-vultr-fleet-221-224.tar.gz`, SHA-256
`45dc34c0b29a49db09d33f05e411d5ed6a148c71db560dc1c830c3db08496cd5`.
All 521 bundled files were checksum-verified after upload to the owned cluster
run directory. Raw logs and detailed per-GPU/port/link summaries remain there;
this repository intentionally publishes the compact dashboard instead of
another copied campaign tree.
