from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import statistics

import numpy as np

from de_algorithm import DifferentialEvolutionConfig, run_differential_evolution
from inp_parser import InpFileParser


def _smallest_instance_by_pipe_count(data_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for inp_path in sorted(data_dir.glob("*.inp")):
        parsed = InpFileParser(inp_path).parse()
        pipe_count = len(parsed.pipes.entries) if parsed.pipes else 0
        if pipe_count > 0:
            candidates.append((pipe_count, inp_path))

    if not candidates:
        raise ValueError("No usable INP file found with at least one pipe.")

    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1]


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    instance_path = _smallest_instance_by_pipe_count(data_dir)
    parsed = InpFileParser(instance_path).parse()
    if parsed.pipes is None or not parsed.pipes.entries:
        raise ValueError(f"Instance {instance_path.name} has no pipes.")

    original_diameters = np.array([pipe.diameter for pipe in parsed.pipes.entries], dtype=float)
    scale = np.where(original_diameters == 0.0, 1.0, original_diameters)

    bounds = np.column_stack((0.5 * scale, 1.5 * scale))

    def objective(candidate: np.ndarray) -> float:
        # Simple baseline objective for fast iteration:
        # recover original diameters (known optimum close to zero loss).
        diff = (candidate - original_diameters) / scale
        return float(np.mean(diff * diff))

    config = DifferentialEvolutionConfig(
        population_size=20,
        generations=60,
        mutation_factor=0.8,
        crossover_rate=0.9,
    )
    runs = 10

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(__file__).resolve().parent / "results" / f"{instance_path.stem}-{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_meta = {
        "instance": instance_path.name,
        "runs": runs,
        "dimensions": int(bounds.shape[0]),
        "bounds_factor": [0.5, 1.5],
        "de_config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "mutation_factor": config.mutation_factor,
            "crossover_rate": config.crossover_rate,
        },
    }
    (results_dir / "experiment_config.json").write_text(
        json.dumps(experiment_meta, indent=2), encoding="utf-8"
    )

    best_fitnesses: list[float] = []
    run_summaries: list[dict[str, float]] = []

    for run_id in range(1, runs + 1):
        seed = 10_000 + run_id
        rng = np.random.default_rng(seed)
        result = run_differential_evolution(
            objective=objective,
            bounds=bounds,
            rng=rng,
            config=config,
        )

        best_fitnesses.append(result.best_fitness)
        run_summary = {
            "run_id": float(run_id),
            "seed": float(seed),
            "best_fitness": result.best_fitness,
        }
        run_summaries.append(run_summary)

        _write_csv(results_dir / f"run_{run_id:02d}_history.csv", result.history)
        (results_dir / f"run_{run_id:02d}_best_vector.csv").write_text(
            ",".join(str(value) for value in result.best_vector) + "\n",
            encoding="utf-8",
        )

    aggregate = {
        "instance": instance_path.name,
        "runs": runs,
        "best_fitness_min": min(best_fitnesses),
        "best_fitness_max": max(best_fitnesses),
        "best_fitness_mean": statistics.mean(best_fitnesses),
        "best_fitness_median": statistics.median(best_fitnesses),
        "best_fitness_stdev": statistics.stdev(best_fitnesses) if len(best_fitnesses) > 1 else 0.0,
    }

    _write_csv(results_dir / "run_summaries.csv", run_summaries)
    (results_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    print(f"Instance selected: {instance_path.name}")
    print(f"Dimension (pipes): {bounds.shape[0]}")
    print(f"Runs completed: {runs}")
    print(f"Results saved to: {results_dir}")
    print(
        "Fitness stats:",
        f"min={aggregate['best_fitness_min']:.6e},",
        f"median={aggregate['best_fitness_median']:.6e},",
        f"mean={aggregate['best_fitness_mean']:.6e},",
        f"std={aggregate['best_fitness_stdev']:.6e}",
    )


if __name__ == "__main__":
    main()
