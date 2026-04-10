from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
from pathlib import Path
import os
import re
import statistics
import sys
import tempfile
import threading
import time

import numpy as np
import wntr

from de_algorithm import DifferentialEvolutionConfig, run_differential_evolution
from inp_parser import InpFileParser
from plot_results import generate_plots


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


def _load_reference_entry(reference_path: Path, instance_name: str) -> dict:
    all_scores = json.loads(reference_path.read_text(encoding="utf-8"))
    if instance_name not in all_scores:
        raise ValueError(
            f"No published reference entry found for {instance_name} in {reference_path.name}"
        )
    return all_scores[instance_name]


def _snap_to_allowed_diameters(
    candidate: np.ndarray, allowed_diameters: np.ndarray
) -> np.ndarray:
    distances = np.abs(candidate[:, np.newaxis] - allowed_diameters[np.newaxis, :])
    nearest_idx = np.argmin(distances, axis=1)
    return allowed_diameters[nearest_idx]


def _load_wntr_model_robust(inp_path: Path) -> tuple[wntr.network.WaterNetworkModel, Path | None]:
    try:
        return wntr.network.WaterNetworkModel(str(inp_path)), None
    except UnicodeDecodeError:
        raw = inp_path.read_bytes()
        text = raw.decode("latin-1")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".inp", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.flush()
        tmp_path = Path(tmp.name)
        tmp.close()
        return wntr.network.WaterNetworkModel(str(tmp_path)), tmp_path


def _min_head_requirement(reference_entry: dict) -> float:
    constraints = str(reference_entry.get("constraints", ""))
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m", constraints, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 20.0


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--"
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_progress_line(
    run_id: int,
    done_generations: int,
    total_generations: int,
    elapsed_seconds: float,
    best_cost: float | None,
    finished: bool,
    published_best_cost: float,
) -> str:
    bar_width = 24
    progress = done_generations / max(1, total_generations)
    filled = min(bar_width, int(round(progress * bar_width)))
    bar = f"{'#' * filled}{'-' * (bar_width - filled)}"
    pct = progress * 100.0

    if finished:
        eta_fragment = f"done in {_format_duration(elapsed_seconds)}"
    elif done_generations <= 0:
        eta_fragment = "eta --:--"
    else:
        remaining = max(0, total_generations - done_generations)
        eta_seconds = (elapsed_seconds / done_generations) * remaining
        eta_fragment = f"eta {_format_duration(eta_seconds)}"

    if best_cost is not None and published_best_cost > 0:
        distance_pct = max(0.0, ((best_cost / published_best_cost) - 1.0) * 100.0)
        best_fragment = f"{distance_pct:.2f}%"
    else:
        best_fragment = "--"
    return (
        f"run {run_id:02d} [{bar}] {pct:6.2f}% "
        f"{done_generations:3d}/{total_generations:3d} {eta_fragment} best={best_fragment}"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"

    instance_name = os.environ.get("WDN_INSTANCE", "BIN.inp")
    preferred_instance_path = data_dir / instance_name
    if preferred_instance_path.exists():
        instance_path = preferred_instance_path
    else:
        instance_path = _smallest_instance_by_pipe_count(data_dir)
    parsed = InpFileParser(instance_path).parse()
    if parsed.pipes is None or not parsed.pipes.entries:
        raise ValueError(f"Instance {instance_path.name} has no pipes.")

    reference_path = Path(__file__).resolve().parent / "results" / "published_reference_scores.json"
    reference_entry = _load_reference_entry(reference_path, instance_path.name)
    published_best_cost = float(reference_entry["published_best_universal_score"])

    diameter_set = reference_entry["diameter_set"]
    allowed_diameters = np.array(
        [float(item["diameter_m"]) for item in diameter_set], dtype=float
    )
    unit_costs = np.array([float(item["unit_cost_per_m"]) for item in diameter_set], dtype=float)

    pipe_lengths = np.array([pipe.length for pipe in parsed.pipes.entries], dtype=float)
    if pipe_lengths.size != len(parsed.pipes.entries):
        raise ValueError("Pipe length and decision vector dimensions are inconsistent.")
    min_head = _min_head_requirement(reference_entry)

    junction_names = [node.id for node in parsed.junctions.entries] if parsed.junctions else []

    bounds = np.column_stack(
        (
            np.full(pipe_lengths.shape, np.min(allowed_diameters), dtype=float),
            np.full(pipe_lengths.shape, np.max(allowed_diameters), dtype=float),
        )
    )

    def _run_single_experiment(run_id: int) -> dict[str, object]:
        # Isolate WNTR state per run to avoid cross-thread mutations.
        _mark_progress_started(run_id)
        wn, temp_inp = _load_wntr_model_robust(instance_path)
        objective_cache: dict[tuple[float, ...], float] = {}

        def objective(candidate: np.ndarray) -> float:
            snapped = _snap_to_allowed_diameters(candidate, allowed_diameters)
            key = tuple(np.round(snapped, 8).tolist())
            cached = objective_cache.get(key)
            if cached is not None:
                return cached

            # Cost definition from benchmark metadata:
            # sum(unit_cost(d_i) * pipe_length_i), with d_i snapped to allowed set.
            snapped_indices = np.argmin(
                np.abs(snapped[:, np.newaxis] - allowed_diameters[np.newaxis, :]), axis=1
            )
            snapped_unit_costs = unit_costs[snapped_indices]
            cost = float(np.sum(snapped_unit_costs * pipe_lengths))

            for i, pipe in enumerate(parsed.pipes.entries):
                if pipe.id in wn.pipe_name_list:
                    wn.get_link(pipe.id).diameter = float(snapped[i])

            try:
                # EPANET toolkit calls used by WNTR are not reliably thread-safe.
                # Keep threaded run orchestration, but serialize toolkit simulations.
                with simulator_lock:
                    sim = wntr.sim.EpanetSimulator(wn)
                    results = sim.run_sim()
                pressure_ts = results.node["pressure"]
                final_pressure = pressure_ts.iloc[-1]
                available_junctions = [j for j in junction_names if j in final_pressure.index]
                if not available_junctions:
                    value = cost + 1e12
                else:
                    junction_pressures = final_pressure.loc[available_junctions]
                    shortfalls = np.maximum(0.0, min_head - junction_pressures.values)
                    violation = float(np.sum(shortfalls))
                    if violation <= 1e-9:
                        value = cost
                    else:
                        # Keep a finite penalty so DE can rank infeasible candidates.
                        value = cost + (2e5 * violation) + 1e7
            except Exception:
                value = cost + 1e12

            objective_cache[key] = value
            return value

        try:
            seed = 10_000 + run_id
            rng = np.random.default_rng(seed)
            result = run_differential_evolution(
                objective=objective,
                bounds=bounds,
                rng=rng,
                config=config,
                progress_callback=lambda generation, best: _update_progress(
                    run_id, generation, best
                ),
            )

            best_cost = result.best_fitness
            # 0% is the target (matches/beats published best). Positive values show distance above target.
            distance_from_published_best_pct = (
                max(0.0, ((best_cost / published_best_cost) - 1.0) * 100.0)
                if published_best_cost > 0
                else 0.0
            )
            run_summary = {
                "run_id": float(run_id),
                "seed": float(seed),
                "best_cost": best_cost,
                "published_best_cost": published_best_cost,
                "distance_from_published_best_pct": distance_from_published_best_pct,
            }
            return {
                "run_id": run_id,
                "best_cost": best_cost,
                "run_summary": run_summary,
                "history": result.history,
                "best_vector": result.best_vector,
            }
        finally:
            if temp_inp and temp_inp.exists():
                try:
                    temp_inp.unlink()
                except OSError:
                    pass

    config = DifferentialEvolutionConfig(
        population_size=40,
        generations=2000,
        mutation_factor=0.5,
        crossover_rate=0.8,
    )
    runs = 1

    instance_run_id = f"{config.generations}-{config.population_size}-{int(config.mutation_factor*100)}-{int(config.crossover_rate*100)}"
    results_dir = Path(__file__).resolve().parent / "results" / f"{instance_path.stem}-{instance_run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_meta = {
        "instance": instance_path.name,
        "runs": runs,
        "dimensions": int(bounds.shape[0]),
        "bounds_diameter_m": [float(np.min(allowed_diameters)), float(np.max(allowed_diameters))],
        "objective_type": reference_entry.get("objective_type", "minimize total pipe cost"),
        "cost_definition": reference_entry.get("cost_definition", ""),
        "published_best_cost": published_best_cost,
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

    best_costs: list[float] = []
    run_durations_seconds: list[float] = []
    run_summaries: list[dict[str, float]] = []
    simulator_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress_state: dict[int, dict[str, float | int | bool | None]] = {
        run_id: {
            "done_generations": 0,
            "start_time": None,
            "finished": False,
            "best_cost": None,
            "finished_elapsed_seconds": None,
        }
        for run_id in range(1, runs + 1)
    }

    progress_enabled = sys.stdout.isatty()
    stop_progress_render = threading.Event()
    rendered_line_count = 0

    def _update_progress(run_id: int, done_generations: int, best_cost: float) -> None:
        with progress_lock:
            state = progress_state[run_id]
            state["done_generations"] = done_generations
            state["best_cost"] = best_cost

    def _mark_progress_started(run_id: int) -> None:
        with progress_lock:
            state = progress_state[run_id]
            state["start_time"] = time.monotonic()
            state["finished"] = False
            state["finished_elapsed_seconds"] = None

    def _mark_progress_done(run_id: int, final_best_cost: float) -> None:
        with progress_lock:
            state = progress_state[run_id]
            start_time = state["start_time"]
            if start_time is None:
                start_time = time.monotonic()
                state["start_time"] = start_time
            finished_elapsed_seconds = time.monotonic() - float(start_time)
            state["done_generations"] = config.generations
            state["best_cost"] = final_best_cost
            state["finished"] = True
            state["finished_elapsed_seconds"] = finished_elapsed_seconds

    def _render_progress() -> None:
        nonlocal rendered_line_count
        if not progress_enabled:
            return
        now = time.monotonic()
        with progress_lock:
            snapshot = {
                run_id: {
                    "done_generations": int(state["done_generations"]),
                    "start_time": (
                        float(state["start_time"]) if state["start_time"] is not None else None
                    ),
                    "finished": bool(state["finished"]),
                    "best_cost": (
                        float(state["best_cost"]) if state["best_cost"] is not None else None
                    ),
                    "finished_elapsed_seconds": (
                        float(state["finished_elapsed_seconds"])
                        if state["finished_elapsed_seconds"] is not None
                        else None
                    ),
                }
                for run_id, state in progress_state.items()
            }

        lines = ["Per-run progress:"]
        for run_id in sorted(snapshot):
            state = snapshot[run_id]
            start_time = state["start_time"]
            elapsed = (
                state["finished_elapsed_seconds"]
                if state["finished"] and state["finished_elapsed_seconds"] is not None
                else (now - start_time if start_time is not None else 0.0)
            )
            lines.append(
                _build_progress_line(
                    run_id=run_id,
                    done_generations=state["done_generations"],
                    total_generations=config.generations,
                    elapsed_seconds=elapsed,
                    best_cost=state["best_cost"],
                    finished=state["finished"],
                    published_best_cost=published_best_cost,
                )
            )

        if rendered_line_count > 0:
            sys.stdout.write(f"\x1b[{rendered_line_count}F")
        for line in lines:
            sys.stdout.write(f"\x1b[K{line}\n")
        sys.stdout.flush()
        rendered_line_count = len(lines)

    def _progress_renderer_worker() -> None:
        while not stop_progress_render.is_set():
            _render_progress()
            if stop_progress_render.wait(timeout=0.5):
                break
        _render_progress()

    max_workers = min(runs, max(1, (os.cpu_count() or 1) - 1))
    progress_thread = (
        threading.Thread(target=_progress_renderer_worker, daemon=True)
        if progress_enabled
        else None
    )
    if progress_thread is not None:
        progress_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_experiment, run_id): run_id
                for run_id in range(1, runs + 1)
            }
            for future in as_completed(futures):
                run_payload = future.result()
                run_id = int(run_payload["run_id"])
                best_cost = float(run_payload["best_cost"])
                run_summary = run_payload["run_summary"]
                history = run_payload["history"]
                best_vector = run_payload["best_vector"]

                best_costs.append(best_cost)
                run_summaries.append(run_summary)  # type: ignore[arg-type]
                _mark_progress_done(run_id, best_cost)
                with progress_lock:
                    finished_elapsed_seconds = progress_state[run_id]["finished_elapsed_seconds"]
                if finished_elapsed_seconds is not None:
                    run_durations_seconds.append(float(finished_elapsed_seconds))

                _write_csv(results_dir / f"run_{run_id:02d}_history.csv", history)  # type: ignore[arg-type]
                (results_dir / f"run_{run_id:02d}_best_vector.csv").write_text(
                    ",".join(str(value) for value in best_vector) + "\n",  # type: ignore[arg-type]
                    encoding="utf-8",
                )
    finally:
        stop_progress_render.set()
        if progress_thread is not None:
            progress_thread.join()
            if rendered_line_count > 0:
                print()

    aggregate = {
        "instance": instance_path.name,
        "experiment_config": experiment_meta["de_config"],
        "runs": runs,
        "published_best_cost": published_best_cost,
        "best_cost_min": min(best_costs),
        "best_cost_max": max(best_costs),
        "best_cost_mean": statistics.mean(best_costs),
        "best_cost_median": statistics.median(best_costs),
        "best_cost_stdev": statistics.stdev(best_costs) if len(best_costs) > 1 else 0.0,
        "run_time_seconds_min": min(run_durations_seconds),
        "run_time_seconds_max": max(run_durations_seconds),
        "run_time_seconds_mean": statistics.mean(run_durations_seconds),
        "run_time_seconds_median": statistics.median(run_durations_seconds),
        "run_time_seconds_stdev": (
            statistics.stdev(run_durations_seconds) if len(run_durations_seconds) > 1 else 0.0
        ),
        "best_run_distance_from_published_best_pct": max(
            0.0, ((min(best_costs) / published_best_cost) - 1.0) * 100.0
        ),
        "median_run_distance_from_published_best_pct": max(
            0.0, ((statistics.median(best_costs) / published_best_cost) - 1.0) * 100.0
        ),
    }

    _write_csv(results_dir / "run_summaries.csv", run_summaries)
    (results_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    generate_plots(results_dir)

    print(f"Instance selected: {instance_path.name}")
    print(f"Dimension (pipes): {bounds.shape[0]}")
    print(f"Runs completed: {runs}")
    print(f"Results saved to: {results_dir}")
    print(
        "Run time stats:",
        f"min={_format_duration(aggregate['run_time_seconds_min'])},",
        f"median={_format_duration(aggregate['run_time_seconds_median'])},",
        f"mean={_format_duration(aggregate['run_time_seconds_mean'])},",
        f"max={_format_duration(aggregate['run_time_seconds_max'])},",
        f"std={_format_duration(aggregate['run_time_seconds_stdev'])}",
    )
    print(
        "Cost stats:",
        f"min={aggregate['best_cost_min']:.2f},",
        f"median={aggregate['best_cost_median']:.2f},",
        f"mean={aggregate['best_cost_mean']:.2f},",
        f"max={aggregate['best_cost_max']:.2f},",
        f"std={aggregate['best_cost_stdev']:.2f}",
    )
    print(
        "Distance from published best (target=0%):",
        f"best-run={aggregate['best_run_distance_from_published_best_pct']:.2f}%,",
        f"median-run={aggregate['median_run_distance_from_published_best_pct']:.2f}%",
    )
if __name__ == "__main__":
    main()
