from __future__ import annotations

import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_run_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "generation": float(row["generation"]),
                "best_fitness": float(row["best_fitness"]),
                "mean_fitness": float(row["mean_fitness"]),
                "std_fitness": float(row["std_fitness"]),
            }
            for row in reader
        ]


def _read_run_summaries(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "run_id": float(row["run_id"]),
                "seed": float(row["seed"]),
                "best_fitness": float(row["best_fitness"]),
            }
            for row in reader
        ]


def _latest_results_dir(results_root: Path) -> Path:
    candidates = [path for path in results_root.iterdir() if path.is_dir()]
    if not candidates:
        raise ValueError(f"No experiment folders found in {results_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _plot_convergence(result_dir: Path, histories: list[list[dict[str, float]]]) -> None:
    generations = np.array([row["generation"] for row in histories[0]], dtype=float)
    best_matrix = np.array(
        [[row["best_fitness"] for row in history] for history in histories],
        dtype=float,
    )

    best_mean = np.mean(best_matrix, axis=0)
    best_std = np.std(best_matrix, axis=0)
    best_min = np.min(best_matrix, axis=0)
    best_max = np.max(best_matrix, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_mean, label="mean best fitness", linewidth=2)
    plt.fill_between(
        generations,
        best_mean - best_std,
        best_mean + best_std,
        alpha=0.2,
        label="mean +/- 1 std",
    )
    plt.plot(generations, best_min, linestyle="--", linewidth=1.5, label="min best fitness")
    plt.plot(generations, best_max, linestyle="--", linewidth=1.5, label="max best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Differential Evolution Convergence Across Runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "convergence_over_runs.png", dpi=200)
    plt.close()


def _plot_best_fitness_distribution(
    result_dir: Path, run_summaries: list[dict[str, float]]
) -> None:
    values = [row["best_fitness"] for row in run_summaries]

    plt.figure(figsize=(8, 6))
    plt.boxplot(values, vert=True, tick_labels=["Best fitness"])
    plt.scatter(np.ones(len(values)), values, alpha=0.7, s=20, color="tab:blue")
    plt.ylabel("Fitness")
    plt.title("Best Fitness Distribution Across Runs")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(result_dir / "best_fitness_distribution.png", dpi=200)
    plt.close()


def _write_plot_summary(result_dir: Path, run_summaries: list[dict[str, float]]) -> None:
    values = [row["best_fitness"] for row in run_summaries]
    text = (
        f"runs: {len(values)}\n"
        f"min: {min(values):.8e}\n"
        f"median: {statistics.median(values):.8e}\n"
        f"mean: {statistics.mean(values):.8e}\n"
        f"std: {statistics.stdev(values) if len(values) > 1 else 0.0:.8e}\n"
        f"max: {max(values):.8e}\n"
    )
    (result_dir / "plot_summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_root = base_dir / "results"
    result_dir = _latest_results_dir(results_root)

    history_files = sorted(result_dir.glob("run_*_history.csv"))
    if not history_files:
        raise ValueError(f"No run history files found in {result_dir}")

    histories = [_read_run_history(path) for path in history_files]
    run_summaries_path = result_dir / "run_summaries.csv"
    run_summaries = _read_run_summaries(run_summaries_path)

    _plot_convergence(result_dir, histories)
    _plot_best_fitness_distribution(result_dir, run_summaries)
    _write_plot_summary(result_dir, run_summaries)

    print(f"Plots saved in: {result_dir}")
    print("- convergence_over_runs.png")
    print("- best_fitness_distribution.png")
    print("- plot_summary.txt")


if __name__ == "__main__":
    main()
