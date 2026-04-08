"""
Visualize Differential Evolution (DE) benchmark results.

Creates:
  - convergence curves (mean/min/max envelope across runs)
  - final improvement bar chart vs initial population best
  - cost summary per difficulty
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _align_histories(histories: List[List[float]]) -> np.ndarray:
    """Return array shape (runs, max_len) by padding with final value."""
    if not histories:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(h) for h in histories)
    aligned = []
    for h in histories:
        if not h:
            aligned.append([np.nan] * max_len)
            continue
        padded = list(h)
        padded.extend([padded[-1]] * (max_len - len(padded)))
        aligned.append(padded)
    return np.asarray(aligned, dtype=np.float64)


def _plot_convergence(results: Dict, out_dir: Path) -> None:
    benchmarks = results["benchmarks"]

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(16, 5))
    if n == 1:
        axes = [axes]

    # Consistent colors for strategies across subplots.
    # We pick a few default matplotlib palettes.
    palette = ["#2E86AB", "#C73E1D", "#A23B72", "#F18F01", "#06A77D"]

    for ax_idx, bench in enumerate(benchmarks):
        ax = axes[ax_idx]
        network_file = bench["network_file"]
        difficulty = bench["difficulty"]

        strategies = bench["strategies"]
        for s_idx, strat in enumerate(strategies):
            runs = strat["runs"]
            hist_best_runs = [r["best_fitness_history"] for r in runs]
            arr = _align_histories(hist_best_runs)
            mean_curve = np.nanmean(arr, axis=0)
            min_curve = np.nanmin(arr, axis=0)
            max_curve = np.nanmax(arr, axis=0)

            color = palette[s_idx % len(palette)]
            gens = np.arange(mean_curve.shape[0])
            ax.plot(gens, mean_curve, color=color, linewidth=2, label=strat["strategy_name"])
            ax.fill_between(gens, min_curve, max_curve, color=color, alpha=0.15)

        ax.set_title(f"{difficulty}: {network_file}\n(pipes={bench['network_stats']['num_pipes']})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best fitness (cost + penalty)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = out_dir / "01_de_convergence_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_final_improvement(results: Dict, out_dir: Path) -> None:
    benchmarks = results["benchmarks"]
    difficulties = [b["difficulty"] for b in benchmarks]

    # Assume all benchmarks have the same strategies order.
    strategies = benchmarks[0]["strategies"]
    strategy_names = [s["strategy_name"] for s in strategies]

    x = np.arange(len(difficulties))
    width = 0.8 / max(1, len(strategy_names))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    palette = ["#2E86AB", "#C73E1D", "#A23B72", "#F18F01", "#06A77D"]

    for s_idx, strat_name in enumerate(strategy_names):
        vals = []
        stds = []
        for b in benchmarks:
            s_entry = next(s for s in b["strategies"] if s["strategy_name"] == strat_name)
            vals.append(s_entry["aggregates"]["summary"]["mean_improvement_percent"])
            stds.append(s_entry["aggregates"]["summary"]["std_improvement_percent"])

        ax.bar(
            x + (s_idx - (len(strategy_names) - 1) / 2) * width,
            vals,
            width=width * 0.95,
            yerr=stds,
            capsize=4,
            alpha=0.9,
            color=palette[s_idx % len(palette)],
            label=strat_name,
        )

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Improvement vs initial best (%)")
    ax.set_xlabel("Difficulty")
    ax.set_title("DE improvement across benchmark difficulties")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = out_dir / "02_de_final_improvement.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_cost_summary(results: Dict, out_dir: Path) -> None:
    benchmarks = results["benchmarks"]
    difficulties = [b["difficulty"] for b in benchmarks]

    strategies = benchmarks[0]["strategies"]
    strategy_names = [s["strategy_name"] for s in strategies]

    x = np.arange(len(difficulties))
    width = 0.8 / max(1, len(strategy_names))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    palette = ["#2E86AB", "#C73E1D", "#A23B72", "#F18F01", "#06A77D"]

    for s_idx, strat_name in enumerate(strategy_names):
        vals = []
        stds = []
        for b in benchmarks:
            s_entry = next(s for s in b["strategies"] if s["strategy_name"] == strat_name)
            vals.append(s_entry["aggregates"]["summary"]["mean_best_cost"])
            stds.append(s_entry["aggregates"]["summary"]["std_best_cost"])

        ax.bar(
            x + (s_idx - (len(strategy_names) - 1) / 2) * width,
            vals,
            width=width * 0.95,
            yerr=stds,
            capsize=4,
            alpha=0.9,
            color=palette[s_idx % len(palette)],
            label=strat_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Mean best cost")
    ax.set_xlabel("Difficulty")
    ax.set_title("Final solution quality (cost) by difficulty")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = out_dir / "03_de_cost_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_file = base_dir / "results" / "benchmark_results.json"
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    _plot_convergence(results, out_dir)
    _plot_final_improvement(results, out_dir)
    _plot_cost_summary(results, out_dir)

    print(f"✓ Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

