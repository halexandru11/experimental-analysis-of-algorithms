from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_run_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        best_key = "best_cost" if "best_cost" in (reader.fieldnames or []) else "best_fitness"
        return [
            {
                "generation": float(row["generation"]),
                "best_cost": float(row[best_key]),
            }
            for row in reader
        ]


def _read_run_summaries(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        uses_cost_columns = "best_cost" in fieldnames
        return [
            {
                "run_id": float(row["run_id"]),
                "seed": float(row["seed"]),
                "best_cost": float(row["best_cost"] if uses_cost_columns else row["best_fitness"]),
                "published_best_cost": float(
                    row["published_best_cost"] if "published_best_cost" in fieldnames else 0.0
                ),
                "cost_effectiveness_pct": float(
                    row["cost_effectiveness_pct"]
                    if "cost_effectiveness_pct" in fieldnames
                    else 0.0
                ),
            }
            for row in reader
        ]


def _latest_results_dir(results_root: Path) -> Path:
    candidates = [path for path in results_root.iterdir() if path.is_dir()]
    if not candidates:
        raise ValueError(f"No experiment folders found in {results_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _read_aggregate_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_convergence(result_dir: Path, histories: list[list[dict[str, float]]]) -> None:
    generations = np.array([row["generation"] for row in histories[0]], dtype=float)
    best_matrix = np.array(
        [[row["best_cost"] for row in history] for history in histories],
        dtype=float,
    )

    best_mean = np.mean(best_matrix, axis=0)
    best_std = np.std(best_matrix, axis=0)
    best_min = np.min(best_matrix, axis=0)
    best_max = np.max(best_matrix, axis=0)
    aggregate = _read_aggregate_summary(result_dir / "aggregate_summary.json")
    published_best_cost = float(aggregate.get("published_best_cost", 0.0))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_mean, label="mean best cost", linewidth=2)
    plt.fill_between(
        generations,
        best_mean - best_std,
        best_mean + best_std,
        alpha=0.2,
        label="mean +/- 1 std",
    )
    plt.plot(generations, best_min, linestyle="--", linewidth=1.5, label="min best cost")
    plt.plot(generations, best_max, linestyle="--", linewidth=1.5, label="max best cost")
    if published_best_cost > 0.0:
        plt.axhline(
            published_best_cost,
            color="tab:red",
            linestyle=":",
            linewidth=2.0,
            label="published best cost",
        )
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("Differential Evolution Cost Convergence Across Runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "convergence_over_runs.png", dpi=200)
    plt.close()


def _plot_convergence_log_scale(result_dir: Path, histories: list[list[dict[str, float]]]) -> None:
    generations = np.array([row["generation"] for row in histories[0]], dtype=float)
    best_matrix = np.array(
        [[row["best_cost"] for row in history] for history in histories],
        dtype=float,
    )

    best_mean = np.mean(best_matrix, axis=0)
    best_min = np.min(best_matrix, axis=0)
    best_max = np.max(best_matrix, axis=0)
    aggregate = _read_aggregate_summary(result_dir / "aggregate_summary.json")
    published_best_cost = float(aggregate.get("published_best_cost", 0.0))

    eps = 1e-9
    # On log-y plots, a multiplicative spread is more faithful than linear +/- std.
    # We compute spread in log-space, then map back around the arithmetic mean.
    log_matrix = np.log(np.maximum(best_matrix, eps))
    log_std = np.std(log_matrix, axis=0)
    scale = np.exp(log_std)
    lower_band = np.maximum(best_mean / scale, eps)
    upper_band = np.maximum(best_mean * scale, eps)
    best_mean = np.maximum(best_mean, eps)
    best_min = np.maximum(best_min, eps)
    best_max = np.maximum(best_max, eps)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_mean, label="mean best cost", linewidth=2)
    plt.fill_between(
        generations,
        lower_band,
        upper_band,
        alpha=0.2,
        label="mean x/÷ exp(std(log(cost)))",
    )
    plt.plot(generations, best_min, linestyle="--", linewidth=1.5, label="min best cost")
    plt.plot(generations, best_max, linestyle="--", linewidth=1.5, label="max best cost")
    if published_best_cost > 0.0:
        plt.axhline(
            published_best_cost,
            color="tab:red",
            linestyle=":",
            linewidth=2.0,
            label="published best cost",
        )
    plt.xlabel("Generation")
    plt.ylabel("Cost (log scale)")
    plt.yscale("log")
    plt.title("Differential Evolution Cost Convergence (Log Scale)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "convergence_over_runs_log.png", dpi=200)
    plt.close()


def _plot_best_cost_distribution(
    result_dir: Path, run_summaries: list[dict[str, float]]
) -> None:
    values = [row["best_cost"] for row in run_summaries]
    published_best_cost = run_summaries[0]["published_best_cost"] if run_summaries else 0.0

    plt.figure(figsize=(8, 6))
    plt.boxplot(values, vert=True, tick_labels=["Best cost"])
    plt.scatter(np.ones(len(values)), values, alpha=0.7, s=20, color="tab:blue")
    if published_best_cost > 0.0:
        plt.axhline(
            published_best_cost,
            color="tab:red",
            linestyle=":",
            linewidth=1.5,
            label="published best cost",
        )
    plt.ylabel("Cost")
    plt.title("Best Cost Distribution Across Runs")
    plt.grid(True, axis="y", alpha=0.3)
    if published_best_cost > 0.0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "best_cost_distribution.png", dpi=200)
    plt.close()


def _write_plot_summary(result_dir: Path, run_summaries: list[dict[str, float]]) -> None:
    values = [row["best_cost"] for row in run_summaries]
    published_best_cost = run_summaries[0]["published_best_cost"] if run_summaries else 0.0
    effectiveness = [
        (published_best_cost / value) * 100.0 for value in values
    ] if published_best_cost > 0.0 else []
    text = (
        f"runs: {len(values)}\n"
        f"min_cost: {min(values):.8e}\n"
        f"median_cost: {statistics.median(values):.8e}\n"
        f"mean_cost: {statistics.mean(values):.8e}\n"
        f"std_cost: {statistics.stdev(values) if len(values) > 1 else 0.0:.8e}\n"
        f"max_cost: {max(values):.8e}\n"
        f"published_best_cost: {published_best_cost:.8e}\n"
        f"best_run_effectiveness_pct: {max(effectiveness) if effectiveness else 0.0:.8f}\n"
        f"median_run_effectiveness_pct: {statistics.median(effectiveness) if effectiveness else 0.0:.8f}\n"
    )
    (result_dir / "plot_summary.txt").write_text(text, encoding="utf-8")


def generate_plots(result_dir: Path) -> None:
    history_files = sorted(result_dir.glob("run_*_history.csv"))
    if not history_files:
        raise ValueError(f"No run history files found in {result_dir}")

    histories = [_read_run_history(path) for path in history_files]
    run_summaries_path = result_dir / "run_summaries.csv"
    run_summaries = _read_run_summaries(run_summaries_path)

    _plot_convergence(result_dir, histories)
    _plot_convergence_log_scale(result_dir, histories)
    _plot_best_cost_distribution(result_dir, run_summaries)
    _write_plot_summary(result_dir, run_summaries)

    print(f"Plots saved in: {result_dir}")
    print("- convergence_over_runs.png")
    print("- convergence_over_runs_log.png")
    print("- best_cost_distribution.png")
    print("- plot_summary.txt")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_root = base_dir / "results"
    result_dir = _latest_results_dir(results_root)
    generate_plots(result_dir)


if __name__ == "__main__":
    main()
