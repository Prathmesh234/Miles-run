# Miles experiments

**Start with the `scripts/` README in each run folder. Startup scripts are essential reproducibility artifacts, not optional utilities.**

- [Run 1 — GRPO-style/IPO](run1-grpo/README.md): completed job 190; original logs, rollout evidence, configuration and charts.
- [Run 2 — synchronous PPO](run2-ppo/README.md): completed job 196; final verified text evidence and checkpoint checks. Cancelled job 195 is explicitly separate.
- [Run 3 — asynchronous PPO + TIS](run3-async-ppo/README.md): job 197; asynchronous one-batch-ahead PPO with measured TIS, transfer and optimizer/offload instrumentation.

**[Final infrastructure comparison and charts](comparison-infrastructure/README.md)** covers jobs 190, 196 and 197, with explicit measurement gaps and scaling limitations.

## RL charts, beyond GPU telemetry

- [Run 1 RL gallery](run1-grpo/charts/rl/README.md): four figures for rewards/admission, custom IPO optimization, rollout behavior and task outcomes.
- [Run 2 RL gallery](run2-ppo/charts/rl/README.md): five figures, including PPO loss/entropy/KL/clipping/ESS and critic learning.
- [Run 3 RL gallery](run3-async-ppo/charts/rl/README.md): six figures, including critic learning, behavior-policy lag and TIS ratios/weights/clipping.

Every figure has PNG and SVG exports. The shared [RL parser](comparison-infrastructure/rl_metrics.py) validates accepted rewards against the exact attempted episodes and trainer scalars. The [renderer](comparison-infrastructure/rl_charts.py) regenerates all three galleries from preserved local evidence:

```sh
python3 comparison-infrastructure/rl_charts.py
```

Rendering requires Matplotlib and NumPy. Each run's `metrics/rl-metrics.json` contains compact chart data and input hashes; `charts/rl/manifest.json` hashes the parser, renderer and exports. Held-out evaluation, explained variance and token-level importance-weight histograms were not recorded and are not fabricated. PPO transport credit is not treated as an advantage measurement. Rejected/unshipped attempts remain distinct from accepted training traces.

[RL verification receipt](comparison-infrastructure/rl-verification.json): 13 analysis tests, 129 source-file hashes, 30 image exports, gallery links, raw-reward checks and preservation checks passed.

These directories preserve the originals; they do not delete or replace the baseline campaign. Live snapshots are timestamped evidence, not promises of final results. No weights/checkpoint shards or credentials are committed. Full binary run artifacts remain on the cluster paths recorded in the manifests.
