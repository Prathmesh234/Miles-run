# Repository layout and deduplication

The September 2026 cleanup removed **2,170 files**, including **222 script
copies**, from a 4,394-file / 357-script tree. Before adding the CollectiveX
dashboard, 2,224 files and 135 scripts remained. Canonical startup directories
were left intact; there were 70 distinct script contents before cleanup.

Removed material was limited to:

- 756 byte-identical files in `run1-grpo/logs/`; the canonical job-190 evidence
  remains in `evidence-job-190/job-190/`.
- 1,200 zero-byte Ray log files.
- 214 byte-identical frozen source snapshots, including repeated per-node
  precision-patch source copies.

Every distinct pre-cleanup content hash remained in the working tree immediately
after deduplication. Nonempty per-process logs were not globally deduplicated:
identical messages can have different node/process attribution. Unique failed
attempts, runtime fixes, training trajectories, metrics and charts were retained.

## Where to work

- `collectivex/`: offline CollectiveX dashboard, compact measured data and one renderer.
- `run1-grpo/scripts/`, `run2-ppo/scripts/`, `run3-async-ppo/scripts/`: canonical
  historical launchers and their required local modules. These are not interchangeable.
- `run4-ppo-long/scripts/`: shared preparation builder for the unsubmitted longer runs.
- `comparison-infrastructure/`: shared analysis and chart generation.
- `evidence-job-190/`, the final run-2/run-3 log directories: historical evidence.

Historical configuration still refers to the environment that produced each
run. Nothing here authorizes launching new GPU jobs or overwriting old runs.

## Recover an old path

`repository-cleanup.json` records every removed path, SHA-256, reason and retained
canonical copy when applicable. No Git history was rewritten. Exact original
files remain at commit `acc563f24807eff03677de181a920eea480277b8`:

```sh
git show acc563f24807eff03677de181a920eea480277b8:PATH/TO/FILE
```

The historical snapshot verifier reads an explicitly retired missing path from
that immutable commit and checks its recorded hash. Unlisted missing inputs
still fail. Original publication/snapshot manifests remain historical records,
not claims about the reduced working tree. The RL renderer now uses the
canonical job-190 evidence directory directly.

Git history still stores old blobs, so this cleanup reduces the checked-out
tree and review clutter; it does not claim to shrink all historical clone data.
