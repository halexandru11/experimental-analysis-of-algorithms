"""
Run Differential Evolution (DE) on selected WDN benchmark instances.

This script:
  1) selects easy/medium/hard networks from `data/`
  2) runs multiple DE strategy configurations
  3) saves raw results (including per-generation fitness histories) to JSON
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from de import DEConfig, DifferentialEvolutionOptimizer
from network_parser import parse_inp_file
from select_benchmarks import select_easy_medium_hard


def _pop_size_for_dim(dim: int) -> int:
    # Keep runtime manageable for large graphs by reducing population.
    if dim <= 50:
        return 25
    if dim <= 200:
        return 20
    if dim <= 600:
        return 16
    return 14


def _max_gens_for_dim(dim: int) -> int:
    # Early stopping will likely kick in; still cap for large instances.
    if dim <= 50:
        return 80
    if dim <= 200:
        return 70
    if dim <= 600:
        return 60
    return 50


def _strategy_list(dim_hint: int) -> List[Dict[str, Any]]:
    pop_size = _pop_size_for_dim(dim_hint)
    max_gens = _max_gens_for_dim(dim_hint)

    return [
        {
            "name": "DE/rand/1/bin (F=0.9, CR=0.9)",
            "config": DEConfig(
                mutation="rand/1",
                crossover="bin",
                pop_size=pop_size,
                max_generations=max_gens,
                stagnation_patience=12,
                F=0.9,
                CR=0.9,
                adaptive=False,
                seed=None,
            ),
        },
        {
            "name": "DE/best/1/bin (jDE adaptive)",
            "config": DEConfig(
                mutation="best/1",
                crossover="bin",
                pop_size=pop_size,
                max_generations=max_gens,
                stagnation_patience=12,
                F=0.8,
                CR=0.9,
                adaptive=True,
                tau1=0.15,
                tau2=0.15,
                seed=None,
            ),
        },
    ]


def _aggregate_histories(histories: List[List[float]]) -> Dict[str, List[float]]:
    """
    Align curves by padding each run with its final value so we can compute
    per-generation mean across runs.
    """
    if not histories:
        return {"mean": [], "min": [], "max": []}

    max_len = max(len(h) for h in histories)
    aligned = []
    for h in histories:
        if not h:
            aligned.append([float("inf")] * max_len)
            continue
        padded = list(h)
        padded.extend([padded[-1]] * (max_len - len(padded)))
        aligned.append(padded)

    arr = np.asarray(aligned, dtype=np.float64)
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    easy, medium, hard = select_easy_medium_hard(str(data_dir))
    benchmarks = [easy, medium, hard]

    # Strategy configs depend on dimension; we build them per benchmark.
    num_runs = 3
    base_seed = 42

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": {
            "easy": easy,
            "medium": medium,
            "hard": hard,
        },
        "benchmarks": [],
    }

    for difficulty, b in zip(["easy", "medium", "hard"], benchmarks):
        network_file = b["filename"]
        filepath = data_dir / network_file
        net = parse_inp_file(str(filepath))

        dim = net.get_pipe_count()
        strategies = _strategy_list(dim)

        benchmark_entry: Dict[str, Any] = {
            "difficulty": difficulty,
            "network_file": network_file,
            "network_stats": net.get_network_stats(),
            "strategies": [],
        }

        print("\n" + "=" * 80)
        print(f"Running benchmark: {network_file} ({difficulty})")
        print(f"  pipes={dim}, junctions={net.get_network_stats()['num_junctions']}")

        for strat in strategies:
            strat_entry: Dict[str, Any] = {
                "strategy_name": strat["name"],
                "config": {
                    **strat["config"].__dict__,
                    # seed is set per run
                },
                "runs": [],
            }

            histories_best: List[List[float]] = []
            histories_avg: List[List[float]] = []

            for run_id in range(num_runs):
                seed = base_seed + run_id * 100 + (0 if "rand/1" in strat["name"] else 1)
                cfg = strat["config"]
                cfg = DEConfig(**{**cfg.__dict__, "seed": seed})  # type: ignore[arg-type]

                optimizer = DifferentialEvolutionOptimizer(net, cfg)
                start = time.perf_counter()
                best_result, hist_best, hist_avg = optimizer.run()
                runtime = time.perf_counter() - start

                # Convert to JSON friendly
                run_entry = {
                    "run_id": run_id + 1,
                    "runtime_sec": runtime,
                    "initial_best_fitness": best_result["initial_best_fitness"],
                    "best_cost": best_result["best_cost"],
                    "best_fitness_history": hist_best,
                    "avg_fitness_history": hist_avg,
                }

                initial = float(best_result["initial_best_fitness"])
                final = float(best_result["best_cost"])
                improvement_pct = (
                    (initial - final) / initial * 100.0 if initial > 0 else 0.0
                )
                run_entry["improvement_percent_vs_initial"] = improvement_pct

                strat_entry["runs"].append(run_entry)
                histories_best.append(hist_best)
                histories_avg.append(hist_avg)

                print(
                    f"  [{strat['name']}] run {run_id+1}/{num_runs}: "
                    f"init={initial:.2e}, best={final:.2e}, imp={improvement_pct:.2f}% "
                    f"in {runtime:.1f}s"
                )

            # Add aggregate curves for plotting.
            strat_entry["aggregates"] = {
                "best_curve": _aggregate_histories(histories_best),
                "avg_curve": _aggregate_histories(histories_avg),
                "summary": {
                    "mean_best_cost": float(np.mean([r["best_cost"] for r in strat_entry["runs"]])),
                    "std_best_cost": float(np.std([r["best_cost"] for r in strat_entry["runs"]])),
                    "mean_improvement_percent": float(
                        np.mean([r["improvement_percent_vs_initial"] for r in strat_entry["runs"]])
                    ),
                    "std_improvement_percent": float(
                        np.std([r["improvement_percent_vs_initial"] for r in strat_entry["runs"]])
                    ),
                    "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in strat_entry["runs"]])),
                },
            }

            benchmark_entry["strategies"].append(strat_entry)

        all_results["benchmarks"].append(benchmark_entry)

    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n✓ Saved benchmark results to:", str(results_file))


if __name__ == "__main__":
    main()

