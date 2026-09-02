# Quality and infrastructure protocol

The operator requires a meaningful Terminal-Bench hill climb as well as complete
infrastructure measurements. Two to five optimizer steps only establish training
correctness. They cannot establish a successful quality run.

## Proposed training budget

After the correctness gates and controlled placement sweep, the selected layout
will receive an initial budget of 400 optimizer steps. Checkpoints at steps 0,
50, 100, 200, and 400 are the proposed evaluation points. This schedule must be
frozen with the task IDs, seeds, evaluation budgets, and image revisions before
baseline outcomes are examined. The operator's wall-clock or GPU-hour limit is
not yet specified. This document does not authorize an unbounded allocation.

At global batch 64 and one optimizer step per rollout, 400 steps consume 25,600
eligible training trajectories. Retries, filtered trajectories, stale drops,
warmup, environment validation, and evaluation are counted separately. No
trajectory count is inferred solely from optimizer steps in the final report.

Four hundred steps is a planning budget, not evidence that this model or taskset
will improve. Any extension requires a prospective budget and a new documented
analysis plan. A plateau, regressions, or failed gates must be reported.

The pinned Megatron calculator has now been executed with dense DP 8, 16 and 24:
global batch 64 is rejected at DP24, while 96 passes all three without rounding.
The Stage 4 reference remains 64. The proposed common Stage 5 batch is therefore
96 (12 prompts × 8 samples), pending full-trainer/gradient validation and the
prospective campaign freeze. At that batch, a subsequent 400-step quality run
would consume 38,400 eligible trajectories, not 25,600. No measured run or
baseline outcome has been used to choose this arithmetic compatibility change.

## Evaluation isolation

Training uses only the pinned Terminal-Lego or TMAX subset. Terminal-Bench 2.1
remains evaluation-only. Held-out tests, solutions, verifier output, and oracle
traces cannot enter policy inputs or training data.

The prospective Terminal-Lego split is now recorded in
`locks/terminal-lego-subset.json`: 512 training IDs and 128 development IDs from
revision `9c197f1c2e87b64cc316b1a5bfcef57b584929f0`, with `task_00000` reserved
only for runtime validation. Selection used deterministic ID hashes before any
baseline outcomes. Sources are materialized and hash-verified, but images,
reference-solution correctness and runtime eligibility are not yet validated.
Failed tasks must remain in the accounting; any split amendment must be explicit
and prospective, never a silent replacement selected after model outcomes.

Use a separate development subset from the training corpus for curriculum and
stopping decisions. Run the untouched TB2.1 suite at the pre-registered base and
checkpoint points. Do not tune task selection, hyperparameters, checkpoint
selection, or the stopping step in response to TB2.1 outcomes. The primary
comparison is the pre-registered final checkpoint versus the base model, not the
best checkpoint chosen after examining test outcomes. Intermediate comparisons
are exploratory and must be labeled accordingly.

Keep the same task-level sampling and evaluation seed schedule across compared
checkpoints. Report paired pass@1 deltas, task-bootstrap 95% confidence intervals,
newly solved and regressed tasks, and all runtime errors and timeouts. Statistical
uncertainty over tasks does not establish training-seed reproducibility. A
positive delta must be independently replicated before the benchmark is called
successful. No positive delta is guaranteed by a step budget.

## Infrastructure coverage

The same approximately one-second collectors run during the placement sweep,
the longer training run, weight broadcasts, checkpoint writes, restart tests,
and evaluation. Label phases and role assignments explicitly. Attribute CPU and
storage contention from self-hosted task containers as well as GPU and fabric
behavior. Preserve missing samples as collector errors, never zeroes.

Report eligible trajectories per hour and held-out progress per wall-clock hour
alongside utilization, queues, staleness, weight-transfer phases, storage, and
fabric counters. Keep hyperparameters fixed during the placement sweep. More
optimizer steps must not reduce the required metric coverage.
