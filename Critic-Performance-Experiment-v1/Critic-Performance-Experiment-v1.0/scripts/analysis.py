"""Ties both halves of the research question together.

Reads:
  - <history_dir>/<backbone>_seed<seed>_history.json   (from
    pipeline.run_offline_sweep() - offline_history has plasticity metrics
    at every checkpoint)
  - <branches_dir>/<backbone>_seed<seed>_branches.json (from
    pipeline.run_online_sweep() - online improvement achieved when
    branching from each of those same checkpoints)

Two standalone comparison plots, usable with either live data (passed
straight from pipeline.py's run_offline_sweep()/run_online_sweep() in the
same call) or data reloaded from disk via the loaders below (for a later,
separate session):
  - plot_offline_sweep(): offline eval return + plasticity probes vs.
    offline step, one line per backbone.
  - plot_online_sweep(): online eval return (or any plasticity metric) vs.
    online step, one line per (backbone, checkpoint) - compares branches
    by algorithm and by how much offline pretraining they had.

Plus the main correlation figure (this module's `main()`), with three rows:
  A. each plasticity metric vs. offline step (mean +/- std across seeds)
  B. online improvement (delta_return, auc) vs. offline step
  C. online improvement vs. each plasticity metric, scattered across every
     (checkpoint, seed) pair, with Pearson/Spearman correlation - the
     actual test of "does low plasticity limit online improvement"
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

PLASTICITY_METRICS = {
    "probe1_ratio": lambda e: e["critic_plasticity"]["probe1"]["ratio"],
    "probe2_ratio": lambda e: e["critic_plasticity"]["probe2"]["ratio"],
    "q1_fau": lambda e: e["critic_plasticity"]["q1_activation"]["mean"]["fau"],
    "q1_dormant_ratio": lambda e: e["critic_plasticity"]["q1_activation"]["mean"]["dormant_ratio"],
    "q1_dead_fraction": lambda e: e["critic_plasticity"]["q1_activation"]["mean"]["dead_fraction"],
    "q1_linearized_fraction": lambda e: e["critic_plasticity"]["q1_activation"]["mean"]["linearized_fraction"],
    "q1_srank": lambda e: e["critic_plasticity"]["q1_srank"],
    "q1_weight_norm": lambda e: e["critic_plasticity"]["q1_weight_norm"],
}


def load_history_files(history_dir: str, backbone: str) -> dict:
    """{seed: offline_history list}."""
    out = {}
    for path in glob.glob(os.path.join(history_dir, f"{backbone}_seed*_history.json")):
        with open(path) as f:
            data = json.load(f)
        out[data["seed"]] = data["offline_history"]
    return out


def load_branch_files(branches_dir: str, backbone: str) -> dict:
    """{seed: {checkpoint_step: branch_result}}."""
    out = {}
    for path in glob.glob(os.path.join(branches_dir, f"{backbone}_seed*_branches.json")):
        with open(path) as f:
            data = json.load(f)
        out[data["seed"]] = {b["checkpoint_step"]: b for b in data["branches"]}
    return out


def seed_results_from_history_files(history_dir: str, backbone: str) -> list[dict]:
    """Reshapes load_history_files()'s output into the same
    [{"backbone", "seed", "offline_history"}, ...] shape
    pipeline.run_offline_sweep() returns live, so plot_offline_sweep() can
    be fed either one the same way."""
    histories = load_history_files(history_dir, backbone)
    return [{"backbone": backbone, "seed": seed, "offline_history": hist}
            for seed, hist in sorted(histories.items())]


def seed_branches_from_branch_files(branches_dir: str, backbone: str) -> list[dict]:
    """Reshapes load_branch_files()'s output into the same
    [{"backbone", "seed", "branches"}, ...] shape
    pipeline.run_online_sweep() returns live, so plot_online_sweep() can be
    fed either one the same way."""
    branch_sets = load_branch_files(branches_dir, backbone)
    return [{"backbone": backbone, "seed": seed, "branches": sorted(by_checkpoint.values(), key=lambda b: b["checkpoint_step"])}
            for seed, by_checkpoint in sorted(branch_sets.items())]


def improvement_metrics(branch: dict) -> dict:
    final_return = branch["online_history"][-1]["eval_return"]
    delta_return = final_return - branch["pre_branch_return"]
    auc = float(np.mean([e["eval_return"] for e in branch["online_history"]]))
    return {"delta_return": delta_return, "auc": auc}


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def plot_offline_sweep(results_by_backbone: dict, out_path: str) -> None:
    """Mean +/- std across seeds, one line per backbone, of offline eval
    return and plasticity probe ratios vs. offline step.
    `results_by_backbone`: {backbone: [{"backbone","seed","offline_history"}, ...]}
    - from run_offline_sweep()'s return value, or from
    seed_results_from_history_files() for data reloaded off disk."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"cal_ql": "tab:blue", "iql": "tab:orange"}
    for backbone, seed_results in results_by_backbone.items():
        if not seed_results:
            continue
        color = colors.get(backbone, None)

        steps = [e["step"] for e in seed_results[0]["offline_history"]]
        returns = np.array([[e["eval_return"] for e in r["offline_history"]] for r in seed_results])
        probe1 = np.array([[e["critic_plasticity"]["probe1"]["ratio"] for e in r["offline_history"]] for r in seed_results])
        probe2 = np.array([[e["critic_plasticity"]["probe2"]["ratio"] for e in r["offline_history"]] for r in seed_results])

        for data, ax, label in [(returns, axes[0], backbone), (probe1, axes[1], f"{backbone} probe1"), (probe2, axes[2], f"{backbone} probe2")]:
            mean, std = data.mean(axis=0), data.std(axis=0)
            ax.plot(steps, mean, label=label, color=color)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    n_seeds = len(next(iter(results_by_backbone.values()), []))
    axes[0].set_title(f"Offline pretraining\n(n={n_seeds} seeds, shaded = +/-1 std)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("eval return"); axes[0].legend()
    axes[1].set_title("Probe 1 (random-target) plasticity ratio\nfresh_error / trained_error")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("ratio"); axes[1].legend()
    axes[2].set_title("Probe 2 (TD-to-other-policy) plasticity ratio\nfresh_error / trained_error")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("ratio"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved offline sweep plot to {out_path}")


def plot_online_sweep(branches_by_backbone: dict, out_path: str, metric: str = "eval_return") -> None:
    """One line per (backbone, checkpoint_step), mean +/- std across seeds -
    lets online fine-tuning curves be compared by algorithm (color) and by
    how much offline pretraining that branch's checkpoint had (shade).
    `branches_by_backbone`: {backbone: [{"backbone","seed","branches"}, ...]}
    - from run_online_sweep()'s return value, or from
    seed_branches_from_branch_files() for data reloaded off disk."""
    cmaps = {"cal_ql": plt.cm.Blues, "iql": plt.cm.Oranges}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fn = (lambda e: e["eval_return"]) if metric == "eval_return" else PLASTICITY_METRICS[metric]

    any_plotted = False
    for backbone, seed_results in branches_by_backbone.items():
        if not seed_results:
            continue
        by_checkpoint: dict = {}
        for seed_result in seed_results:
            for branch in seed_result["branches"]:
                by_checkpoint.setdefault(branch["checkpoint_step"], []).append(branch["online_history"])

        checkpoint_steps = sorted(by_checkpoint)
        cmap = cmaps.get(backbone, plt.cm.Greys)
        for i, checkpoint_step in enumerate(checkpoint_steps):
            histories = by_checkpoint[checkpoint_step]
            shade = cmap(0.4 + 0.5 * i / max(len(checkpoint_steps) - 1, 1))
            steps = [e["step"] for e in histories[0]]
            values = np.array([[fn(e) for e in hist] for hist in histories])
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(steps, mean, label=f"{backbone} @ offline step {checkpoint_step}", color=shade)
            ax.fill_between(steps, mean - std, mean + std, color=shade, alpha=0.15)
            any_plotted = True

    if not any_plotted:
        print("no branch data given, skipping online sweep plot")
        return

    ax.set_xlabel("online step (within this branch)")
    ax.set_ylabel(metric)
    ax.set_title("Online fine-tuning by backbone and offline checkpoint\n(shaded = +/-1 std across seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved online sweep plot to {out_path}")


def main():
    backbone = "iql"
    history_dir = "results"
    # optional - if None (or no branch files found yet), only row A
    # (plasticity vs. offline step) is produced, so half 1's trend can be
    # checked before committing to half 2's branching sweep
    branches_dir = None
    out_path = "results/plasticity_analysis.png"

    histories = load_history_files(history_dir, backbone)
    if not histories:
        raise SystemExit(f"no history files found for backbone={backbone} in {history_dir}")

    branch_sets = load_branch_files(branches_dir, backbone) if branches_dir else {}
    have_branches = bool(branch_sets)

    if have_branches:
        seeds = sorted(set(histories) & set(branch_sets))
        if not seeds:
            raise SystemExit("no overlapping seeds between history and branch files")
        print(f"Found {len(seeds)} seeds with both history and branch data: {seeds}")
        checkpoint_steps = sorted(branch_sets[seeds[0]].keys())
    else:
        seeds = sorted(histories)
        print(f"Found {len(seeds)} seeds with history data, no branch data "
              f"({'branches_dir not set' if not branches_dir else 'no branch files found'}) "
              f"- only plotting plasticity vs. offline step.")
        checkpoint_steps = sorted({e["step"] for seed in seeds for e in histories[seed]})

    metric_by_step = {name: [] for name in PLASTICITY_METRICS}
    delta_by_step, auc_by_step = [], []
    pooled_metric = {name: [] for name in PLASTICITY_METRICS}
    pooled_delta, pooled_auc = [], []

    for step in checkpoint_steps:
        for name, fn in PLASTICITY_METRICS.items():
            vals = []
            for seed in seeds:
                entry = next((e for e in histories[seed] if e["step"] == step), None)
                if entry is None:
                    continue
                v = fn(entry)
                vals.append(v)
                pooled_metric[name].append(v)
            metric_by_step[name].append(vals)

        if have_branches:
            deltas, aucs = [], []
            for seed in seeds:
                branch = branch_sets[seed].get(step)
                if branch is None:
                    continue
                m = improvement_metrics(branch)
                deltas.append(m["delta_return"])
                aucs.append(m["auc"])
                pooled_delta.append(m["delta_return"])
                pooled_auc.append(m["auc"])
            delta_by_step.append(deltas)
            auc_by_step.append(aucs)

    n_metrics = len(PLASTICITY_METRICS)
    n_rows = 3 if have_branches else 1
    fig, axes = plt.subplots(n_rows, n_metrics, figsize=(3.2 * n_metrics, 3.3 * n_rows), squeeze=False)

    for col, (name, values) in enumerate(metric_by_step.items()):
        mean = [np.mean(v) if v else np.nan for v in values]
        std = [np.std(v) if v else np.nan for v in values]
        ax = axes[0][col]
        ax.errorbar(checkpoint_steps, mean, yerr=std, marker="o", color="tab:blue")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("offline step")

    if have_branches:
        for col in range(n_metrics):
            ax = axes[1][col]
            if col == 0:
                mean = [np.mean(v) if v else np.nan for v in delta_by_step]
                std = [np.std(v) if v else np.nan for v in delta_by_step]
                ax.errorbar(checkpoint_steps, mean, yerr=std, marker="o", color="tab:red")
                ax.set_title("delta_return vs offline step", fontsize=9)
            elif col == 1:
                mean = [np.mean(v) if v else np.nan for v in auc_by_step]
                std = [np.std(v) if v else np.nan for v in auc_by_step]
                ax.errorbar(checkpoint_steps, mean, yerr=std, marker="o", color="tab:orange")
                ax.set_title("auc vs offline step", fontsize=9)
            else:
                ax.axis("off")
            if col < 2:
                ax.set_xlabel("offline step")

        for col, (name, values) in enumerate(pooled_metric.items()):
            ax = axes[2][col]
            ax.scatter(values, pooled_delta, s=15)
            r = pearson(values, pooled_delta)
            rho = spearman(values, pooled_delta)
            ax.set_title(f"pearson={r:.2f} spearman={rho:.2f}", fontsize=9)
            ax.set_xlabel(name)
            if col == 0:
                ax.set_ylabel("delta_return")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved analysis plot to {out_path}")

    if have_branches:
        print("\nCorrelations with delta_return (pooled across checkpoints x seeds):")
        for name, values in pooled_metric.items():
            r = pearson(values, pooled_delta)
            rho = spearman(values, pooled_delta)
            print(f"  {name}: pearson={r:.3f}  spearman={rho:.3f}")

    sweep_backbones = ["cal_ql", "iql"]
    results_by_backbone = {b: r for b in sweep_backbones if (r := seed_results_from_history_files(history_dir, b))}
    plot_offline_sweep(results_by_backbone, "results/offline_sweep.png")

    branches_by_backbone = {b: r for b in sweep_backbones if (r := seed_branches_from_branch_files(branches_dir or "results/branches", b))}
    plot_online_sweep(branches_by_backbone, "results/online_sweep.png")


if __name__ == "__main__":
    main()
