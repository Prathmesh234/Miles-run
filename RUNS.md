# Miles experiments

**Start with the `scripts/` README in each run folder. Startup scripts are essential reproducibility artifacts, not optional utilities.**

- [Run 1 — GRPO-style/IPO](run1-grpo/README.md): completed job 190; original logs, rollout evidence, configuration and charts.
- [Run 2 — synchronous PPO](run2-ppo/README.md): completed job 196; final verified text evidence and checkpoint checks. Cancelled job 195 is explicitly separate.
- [Run 3 — asynchronous PPO + TIS](run3-async-ppo/README.md): job 197; asynchronous one-batch-ahead PPO with measured TIS, transfer and optimizer/offload instrumentation.

**[Final infrastructure comparison and charts](comparison-infrastructure/README.md)** covers jobs 190, 196 and 197, with explicit measurement gaps and scaling limitations.

These directories preserve the originals; they do not delete or replace the baseline campaign. Live snapshots are timestamped evidence, not promises of final results. No weights/checkpoint shards or credentials are committed. Full binary run artifacts remain on the cluster paths recorded in the manifests.
