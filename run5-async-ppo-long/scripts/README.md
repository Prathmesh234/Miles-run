# Important: asynchronous startup scripts

No new job has been submitted. Do not rerun job 197's launcher: it retains the
old model and two-update limit.

The shared preparation implementation is
[`../../run4-ppo-long/scripts/long_run_config.py`](../../run4-ppo-long/scripts/long_run_config.py).
Run the six checks from the repository root:

```sh
python3 -m unittest discover -s run4-ppo-long/scripts -p test_long_run.py -v
```

The native-driver fixture exercises ten iterations, verifies one actor and one
critic train call per iteration, and checks behavior versions
`[1, 1, 2, 3, 4, 5, 6, 7, 8, 9]`. This is a CPU scheduling test, not a training
result. Actual launch, model/renderer validation and per-rank optimizer-update
verification are still required.
