"""Export per-run RL charts, using one shared parser and no duplicated raw logs.

Default: all three completed runs. For future runs pass --folder, --log-root,
--rollout-root, --job-id and --label explicitly. Paths are repository-relative.
Requires Matplotlib and NumPy; extraction uses only the standard library.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

from rl_metrics import extract, sha

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#203044", "#64748b", "#e6ebef"
BLUE, GREEN, AMBER = "#4263a0", "#138b87", "#bb7a30"
COLORS = [BLUE, GREEN, AMBER]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"})


def clean(ax):
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5)


def save(fig, data, out, name, title, note):
    count = len(data["updates"])
    fig.suptitle(title, x=.075, y=.98, ha="left", fontsize=20, fontweight="bold")
    fig.text(.075, .925, f'{data["label"]}  ·  job {data["job_id"]}  ·  {count} optimizer updates  ·  '
             f'{sum(len(u["samples"]) for u in data["updates"])} accepted training traces', fontsize=10.5, color=MUTED)
    fig.text(.075, .025, note + "\nSource: recorded training.log + rollout JSON; compact data and input hashes in ../../metrics/rl-metrics.json.\n"
             "Training-batch diagnostics only. These short runs do not establish convergence or held-out quality.",
             fontsize=8.4, color=MUTED, linespacing=1.45, va="bottom")
    fig.subplots_adjust(left=.09, right=.97, top=.835, bottom=.18, wspace=.40, hspace=.55)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=.18)
        if ext == "svg":
            # Matplotlib emits trailing spaces in multiline SVG path attributes.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    return name


def series(ax, xs, ys, title, unit, label=None, color=BLUE, percent=False, precision=4, annotation_y=8):
    factor = 100 if percent else 1
    ys = [y * factor if y is not None else np.nan for y in ys]
    ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.4, color=color, label=label)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.set_xlabel("Optimizer update (1-based)")
    ax.set_ylabel(unit)
    ax.set_xticks(xs)
    ax.margins(x=.20, y=.25)
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            ax.annotate(f"{y:.{precision}g}", (x, y), xytext=(0, annotation_y), textcoords="offset points", ha="center", fontsize=8)
    formatter = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(formatter)
    clean(ax)


def values(data, key, role="actor"):
    return [u[role].get(key) for u in data["updates"]]


def rewards(data, out):
    updates = data["updates"]; xs = [u["update"] for u in updates]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    series(axes[0, 0], xs, [u["raw_reward_mean"] for u in updates], "Accepted training-batch reward", "Mean raw reward")
    axes[0, 0].set_ylim(-.04, 1.15)
    scored = [[e["solved_score"] for e in u["attempted_episodes"] if e["solved_score"] is not None] for u in updates]
    series(axes[0, 1], xs, [statistics.mean(s) if s else None for s in scored],
           "All scored attempts · before admission", "Mean task solved score", color=GREEN)
    axes[0, 1].set_ylim(-.04, 1.15)
    ax = axes[1, 0]
    x = np.arange(len(xs))
    for offset, field, title, color in [(-.18, "attempted_episodes", "Attempted episodes", "#b7c3d3"),
                                       (.18, "samples", "Accepted traces", BLUE)]:
        bars = ax.bar(x + offset, [len(u[field]) for u in updates], width=.34, label=title, color=color)
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, xs); ax.set_xlabel("Optimizer update (1-based)"); ax.set_ylabel("Count")
    ax.set_title("Batch admission changes the population", loc="left", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.margins(y=.35); clean(ax)
    ax = axes[1, 1]
    width = .72 / len(updates)
    for i, u in enumerate(updates):
        counts = Counter(s["reward"] for s in u["samples"])
        bars = ax.bar([r + (i - (len(xs)-1)/2) * width for r in (0, 1)],
                     [counts.get(r, 0) for r in (0, 1)], width=width*.9, color=COLORS[i % 3], label=f'Update {u["update"]}')
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xticks([0, 1], ["0 · not solved", "1 · solved"])
    ax.set_xlabel("Raw accepted-trace reward"); ax.set_ylabel("Accepted trace count")
    ax.set_title("Training reward distribution", loc="left", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8); ax.margins(y=.35); clean(ax)
    unscored = [len(u["attempted_episodes"]) - len(s) for u, s in zip(updates, scored)]
    return save(fig, data, out, "01-rewards-and-admission", "Task rewards and the traces that reached training",
        f"Unscored attempts excluded from the scored-attempt mean, by update: {unscored}. Accepted rewards are never group-centered here.\n"
        "The IPO run filters zero-credit groups; PPO retains them. Admission differences prevent an apples-to-apples quality ranking.")


def policy(data, out):
    is_ppo = any(u["critic"] for u in data["updates"])
    specs = [("train/loss", "Policy loss" if is_ppo else "Custom IPO objective", "Logged objective", False),
             ("train/entropy_loss" if is_ppo else "train/entropy", "Policy vocabulary entropy", "Nats", False),
             ("train/grad_norm", "Actor gradient norm", "Reported L2 norm", False)]
    specs += [("train/ppo_kl", "PPO sampled log-probability change", "Mean old − current log p (nats)", False),
              ("train/pg_clipfrac", "PPO policy clipping", "Logged clipped fraction (%)", True),
              ("train/ess_ratio", "PPO effective sample-size ratio", "Logged ESS ratio", False)] if is_ppo else [
              ("train/mismatch_kl", "Train / behavior mismatch · k3", "Logged KL estimate (nats)", False),
              ("train/ipo_masked_fraction", "IPO probability-difference mask", "Masked action positions (%)", True),
              ("train/lr-pg_0", "Actor learning rate", "Learning rate", False)]
    specs = [s for s in specs if any(v is not None for v in values(data, s[0]))]
    fig, axes = plt.subplots((len(specs)+2)//3, 3, figsize=(15, 9), squeeze=False)
    xs = [u["update"] for u in data["updates"]]
    for ax, (key, title, unit, pct) in zip(axes.flat, specs):
        series(ax, xs, values(data, key), title, unit, percent=pct)
    for ax in list(axes.flat)[len(specs):]: ax.remove()
    note = "Metrics are recorded during each optimizer update, not a separate evaluation after training. Entropy coefficient is zero."
    if data["is_async_tis"]:
        note += "\nPPO KL=0 / ESS=1 reflect the recomputed pre-update policy with one step; behavior-policy lag is measured separately by TIS."
    elif is_ppo:
        note += "\nPPO's signed sampled log-probability difference is not a guaranteed nonnegative KL divergence."
    else:
        note += "\nIPO's custom objective, k3 mismatch estimator and probability mask differ from PPO loss / KL / clipping."
    return save(fig, data, out, "02-policy-optimization", "What the policy optimizer actually saw", note)


def distribution(ax, data, key, title, unit):
    batches = [[s[key] for s in u["samples"]] for u in data["updates"]]
    positions = [u["update"] for u in data["updates"]]
    boxes = ax.boxplot(batches, positions=positions, widths=.45, patch_artist=True,
                       showfliers=False, medianprops={"color": INK, "linewidth": 1.7})
    for i, (pos, vals, box) in enumerate(zip(positions, batches, boxes["boxes"])):
        box.set_facecolor(COLORS[i % 3]); box.set_alpha(.15)
        ax.scatter(pos + np.linspace(-.13, .13, len(vals)), vals, s=18, color=COLORS[i % 3], alpha=.7)
    ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel("Optimizer update (1-based)")
    ax.set_xticks(positions); ax.set_ylabel(unit); clean(ax)


def rollout_behavior(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    distribution(axes[0, 0], data, "active_action_tokens", "Tokens contributing to the policy loss", "Active action tokens / trace")
    distribution(axes[0, 1], data, "total_tokens", "Packed conversation length", "Total tokens / trace")
    distribution(axes[0, 2], data, "turns", "Agent interaction length", "Turns / trace")
    ax = axes[1, 0]; x = np.arange(len(data["updates"]))
    for offset, key, label, color in [(-.17, "trace_truncated", "Environment truncation", BLUE),
                                      (.17, "packing_truncated", "Packing truncation", AMBER)]:
        ys = [100 * statistics.mean(s[key] for s in u["samples"]) for u in data["updates"]]
        bars = ax.bar(x+offset, ys, width=.31, color=color, label=label); ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x, [u["update"] for u in data["updates"]]); ax.set_xlabel("Optimizer update (1-based)")
    ax.set_ylabel("Accepted traces (%)"); ax.set_title("Why traces were truncated", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.set_ylim(0, 105); clean(ax)
    ax = axes[1, 1]
    counts = Counter(s["stop_condition"] or "unrecorded" for u in data["updates"] for s in u["samples"])
    labels, counts = zip(*sorted(counts.items())); y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=GREEN); ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_yticks(y, [s.replace("_", " ") for s in labels]); ax.set_xlabel("Accepted traces (all updates)")
    ax.set_ylabel("Recorded stop condition"); ax.set_title("Episode termination", loc="left", fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=.3); clean(ax)
    ax = axes[1, 2]
    for i, u in enumerate(data["updates"]):
        ax.scatter([s["behavior_nll_nats"] for s in u["samples"]], [s["reward"] for s in u["samples"]],
                   s=28, alpha=.65, color=COLORS[i % 3], label=f'Update {u["update"]}')
    ax.set_xlabel("Mean sampled-action surprisal (nats)"); ax.set_ylabel("Raw trace reward")
    ax.set_title("Behavior confidence and reward", loc="left", fontsize=11)
    ax.set_yticks([0, 1]); ax.set_ylim(-.1, 1.25); ax.legend(frameon=False, fontsize=8, loc="upper right"); clean(ax)
    return save(fig, data, out, "03-rollout-behavior", "How the agent spent its token and turn budget",
        "Dots show every accepted trace; boxes show median/IQR and 1.5×IQR whiskers. Truncation categories can overlap.\n"
        "Sampled-action surprisal is −mean(behavior log p) on the loss mask, not vocabulary entropy. Task mix confounds reward correlations.")


def task_outcomes(data, out):
    tasks = sorted({e["task"] for u in data["updates"] for e in u["attempted_episodes"]})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8))
    for ax, field, key, title in [(axes[0], "samples", "reward", "Accepted training traces"),
                                 (axes[1], "attempted_episodes", "solved_score", "All scored attempted episodes")]:
        matrix = np.full((len(tasks), len(data["updates"])), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        for col, u in enumerate(data["updates"]):
            for row, task in enumerate(tasks):
                vals = [s[key] for s in u[field] if s["task"] == task and s[key] is not None]
                if vals: matrix[row, col] = statistics.mean(vals); counts[row, col] = len(vals)
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#edf0f3")
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (row, col), v in np.ndenumerate(matrix):
            ax.text(col, row, "—" if np.isnan(v) else f"{v:.1%}\nn={counts[row, col]}",
                    ha="center", va="center", color="white" if np.isfinite(v) and v > .6 else INK, fontsize=11)
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_xticks(range(len(data["updates"])), [u["update"] for u in data["updates"]])
        ax.set_yticks(range(len(tasks)), tasks); ax.set_xlabel("Optimizer update (1-based)"); ax.set_ylabel("Task ID")
        fig.colorbar(im, ax=ax, shrink=.8, label="Mean binary solved score")
    return save(fig, data, out, "04-task-outcomes", "Separate task difficulty from aggregate reward",
        "Each cell is the mean recorded binary score, with its exact sample count. Gray / — means no scored observations, not zero reward.\n"
        "Attempts include rejected and unshipped traces; training cells include accepted traces only. Eight-sample task groups are correlated.")


def critic(data, out):
    if not any(u["critic"] for u in data["updates"]): return None
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9)); xs = [u["update"] for u in data["updates"]]
    series(axes[0, 0], xs, values(data, "train/critic-value_loss", "critic"), "Critic value loss", "Logged value objective", color=GREEN)
    series(axes[0, 1], xs, values(data, "train/critic-grad_norm", "critic"), "Critic gradient norm", "Reported L2 norm", color=GREEN)
    ax = axes[1, 0]
    series(ax, xs, values(data, "rollout/values"), "Value scale versus return scale", "Logged mean scalar", label="Value prediction", color=BLUE)
    series(ax, xs, values(data, "rollout/returns"), "Value scale versus return scale", "Logged mean scalar", label="Return", color=GREEN, annotation_y=-16)
    ax.legend(frameon=False, fontsize=8)
    series(axes[1, 1], xs, values(data, "rollout/advantages"), "Trainer-reported advantages", "Logged reduced advantage")
    axes[1, 1].axhline(0, color=MUTED, linewidth=.8, linestyle=":")
    return save(fig, data, out, "05-critic-learning", "Value-function learning and advantage scale",
        "Value/return/advantage panels use trainer-reduced rollout scalars, not token-level distributions or calibration estimates.\n"
        "The adapter's constant 1.0 transport credit is excluded. Explained variance cannot be reconstructed from these scalar means.")


def offpolicy(data, out):
    if not data["is_async_tis"]: return None
    fig, axes = plt.subplots(2, 3, figsize=(15, 9)); xs = [u["update"] for u in data["updates"]]
    series(axes[0, 0], xs, [max(s["policy_lag_updates"] for s in u["samples"]) for u in data["updates"]],
           "Behavior-policy age", "Trainer − behavior version (updates)")
    specs = [("train/train_rollout_logprob_abs_diff", "Train / behavior log-p mismatch", "Mean absolute difference (nats)", False),
             ("train/train_rollout_kl", "Behavior → trainer mismatch · k3", "Logged KL estimate (nats)", False),
             ("train/tis", "Unclipped importance ratio", "Mean trainer / behavior ratio", False),
             ("train/tis_weight", "Detached TIS correction weight", "Mean weight after clamp [0, 2]", False),
             ("train/tis_upper_clipfrac", "TIS upper-bound clipping", "Active positions clipped (%)", True)]
    for ax, (key, title, unit, pct) in zip(list(axes.flat)[1:], specs):
        series(ax, xs, values(data, key), title, unit, color=GREEN, percent=pct,
               precision=7 if key in ("train/tis", "train/tis_weight") else 4)
        if key in ("train/tis", "train/tis_weight"): ax.axhline(1, color=MUTED, lw=.8, ls=":")
    return save(fig, data, out, "06-off-policy-tis", "Asynchronous sampling and the correction it required",
        "Policy age uses recorded engine weight versions and the native one-batch-ahead schedule. TIS = clamp(exp(log p_train − log p_behavior), 0, 2).\n"
        "TIS means near 1 do not imply every token weight is near 1; a weight histogram was not recorded. PPO clipping and TIS clipping are distinct.")


def render_run(folder, log_root, rollout_root, job_id, label):
    data = extract(ROOT, ROOT/folder, ROOT/log_root, ROOT/rollout_root, job_id, label)
    # These task charts explicitly depict binary solved scores, never silently bin continuous rewards.
    for u in data["updates"]:
        if any(s["reward"] not in (0., 1.) for s in u["samples"]) or any(e["solved_score"] not in (None, 0., 1.) for e in u["attempted_episodes"]):
            raise ValueError("This chart recipe requires binary solved rewards; continuous rewards need different axes")
    metrics = ROOT/folder/"metrics/rl-metrics.json"; metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    out = ROOT/folder/"charts/rl"
    names = [name for fn in (rewards, policy, rollout_behavior, task_outcomes, critic, offpolicy)
             if (name := fn(data, out))]
    manifest = {"job_id": job_id, "updates": len(data["updates"]), "figures": names,
                "metrics_sha256": sha(metrics), "renderer_sha256": sha(Path(__file__)),
                "parser_sha256": sha(Path(__file__).with_name("rl_metrics.py")),
                "outputs_sha256": {p.name: sha(p) for n in names for p in (out/f"{n}.png", out/f"{n}.svg")},
                "missing_or_not_applicable": data["missing_or_not_applicable"]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paragraphs = [f'# RL charts — {label}, job {job_id}',
        f'{len(data["updates"])} recorded updates. Charts use **training batches**, not held-out evaluation. Displayed update 1 corresponds to raw log step 0.',
        'Regenerate from the repository root with `python3 comparison-infrastructure/rl_charts.py` (Matplotlib and NumPy required). '
        'The shared renderer accepts explicit paths for future runs; no raw logs or source trees are duplicated.',
        'Compact inputs and source hashes: [rl-metrics.json](../../metrics/rl-metrics.json). Output hashes: [manifest.json](manifest.json).']
    for name in names:
        paragraphs += [f'## {name[3:].replace("-", " ").title()}', f'![{name} ]({name}.png)', f'[Download SVG]({name}.svg)']
    paragraphs += ['## Interpretation and missing metrics',
                   *['- ' + v for v in data["semantics"].values()],
                   '', *['- Not available / not applicable: ' + v for v in data["missing_or_not_applicable"]]]
    (out/"README.md").write_text("\n\n".join(paragraphs) + "\n")
    print(json.dumps({"folder": folder, "job_id": job_id, "figures": len(names), "updates": len(data["updates"])}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("folder", "log-root", "rollout-root", "label"): ap.add_argument("--" + arg)
    ap.add_argument("--job-id", type=int)
    args = ap.parse_args()
    custom = (args.folder, args.log_root, args.rollout_root, args.job_id, args.label)
    if any(v is not None for v in custom):
        if any(v is None for v in custom): ap.error("Supply all five options for a custom run")
        render_run(*custom)
    else:
        render_run("run1-grpo", "evidence-job-190/job-190", "run1-grpo/rollouts", 190, "GRPO-style credit / custom IPO")
        render_run("run2-ppo", "run2-ppo/logs/job-196-final", "run2-ppo/logs/job-196-final/rollouts", 196, "Synchronous PPO")
        render_run("run3-async-ppo", "run3-async-ppo/logs/job-197-final", "run3-async-ppo/logs/job-197-final/rollouts", 197, "Async PPO + TIS")
