"""Ties both halves of the research question together.

Reads:
  - <history_dir>/<backbone>_seed<seed>_history.json   (from pipeline.py's
    long offline run - offline_history has plasticity metrics at every
    checkpoint)
  - <branches_dir>/<backbone>_seed<seed>_branches.json (from
    pipeline.run_branch_sweep() - online improvement achieved when branching from
    each of those same checkpoints)

Produces one figure with three rows:
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


if __name__ == "__main__":
    main()
