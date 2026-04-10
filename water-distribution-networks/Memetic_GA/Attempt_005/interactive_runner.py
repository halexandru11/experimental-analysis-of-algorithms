import threading
import time
import uuid
import warnings
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force non-interactive matplotlib backend to avoid Tk backend calls from worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")

from memetic_ga import Individual, MemeticGA
from network_parser import parse_inp_file
from test_benchmarks import BenchmarkRunner

from history_visualizer import HistoryVisualizer
from persistence import RunPersistence


@dataclass
class RunConfig:
    network_file: str
    algorithm: str  # "memetic" or "standard"
    population_size: int
    max_generations: int
    local_search_intensity: float
    seed: int
    strict_objective_for_optimization: bool
    strict_check_full_population_each_gen: bool
    repair_mode: str  # "none", "first_generation", "every_generation"


class _ActiveRun:
    def __init__(self, run_id: str, config: RunConfig):
        self.run_id = run_id
        self.config = config
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.status = "running"
        self.current_generation = 0
        self.best_training_fitness: Optional[float] = None
        self.best_paper_score: Optional[float] = None
        self.best_gap_to_published_pct: Optional[float] = None
        self.started_at = time.time()
        self.ended_at: Optional[float] = None


class InteractiveRunManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.attempt4_results_dir = self.base_dir / "Memetic_GA" / "Attempt_004" / "results"
        self.attempt5_dir = self.base_dir / "Memetic_GA" / "Attempt_005"
        self.attempt5_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.attempt5_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.persistence = RunPersistence(self.attempt5_dir / "run_history.sqlite")
        self._runner = BenchmarkRunner(str(self.data_dir), str(self.attempt4_results_dir))

        self._active: Dict[str, _ActiveRun] = {}
        self._manager_lock = threading.Lock()

    def generate_history_visualizations(self) -> List[Path]:
        output_dir = self.attempt5_dir / "results"
        visualizer = HistoryVisualizer(
            self.persistence,
            output_dir=output_dir,
            reference_scores_path=self.attempt4_results_dir / "published_reference_scores.json",
        )
        return visualizer.generate_all()

    def delete_runs(self, run_ids: List[str]) -> int:
        if not run_ids:
            return 0

        active_ids = set(self.get_active_run_ids())
        blocked = [rid for rid in run_ids if rid in active_ids]
        if blocked:
            raise ValueError(f"Cannot delete active running runs: {', '.join(blocked)}")

        for run_id in run_ids:
            checkpoint_path = self._checkpoint_path(run_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)

        return self.persistence.delete_runs(run_ids)

    def force_delete_runs_from_db(self, run_ids: List[str]) -> int:
        """Force-delete runs from DB without checking if they're active.
        Use this to nuke stuck runs that won't stop."""
        if not run_ids:
            return 0
        
        # Kill any stuck threads for these runs
        with self._manager_lock:
            for run_id in run_ids:
                if run_id in self._active:
                    self._active.pop(run_id, None)

        for run_id in run_ids:
            checkpoint_path = self._checkpoint_path(run_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)
        
        # Delete from DB directly
        return self.persistence.delete_runs(run_ids)

    def _normalize_config_dict(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(cfg or {})
        if "repair_mode" not in normalized:
            legacy_repair = bool(normalized.get("repair_best_each_generation", True))
            normalized["repair_mode"] = "every_generation" if legacy_repair else "none"
        normalized.pop("repair_best_each_generation", None)
        return normalized

    def _config_from_dict(self, cfg: Dict[str, Any]) -> RunConfig:
        normalized = self._normalize_config_dict(cfg)
        return RunConfig(
            network_file=str(normalized.get("network_file", "BIN.inp")),
            algorithm=str(normalized.get("algorithm", "memetic")),
            population_size=int(normalized.get("population_size", 40)),
            max_generations=int(normalized.get("max_generations", 250)),
            local_search_intensity=float(normalized.get("local_search_intensity", 0.8)),
            seed=int(normalized.get("seed", 42)),
            strict_objective_for_optimization=bool(normalized.get("strict_objective_for_optimization", True)),
            strict_check_full_population_each_gen=bool(normalized.get("strict_check_full_population_each_gen", True)),
            repair_mode=str(normalized.get("repair_mode", "every_generation")),
        )

    def _checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoints_dir / f"{run_id}.json"

    def _save_checkpoint(
        self,
        run_id: str,
        cfg: RunConfig,
        ga: MemeticGA,
        best_paper_score_seen: float,
        best_gap_seen: float,
    ) -> None:
        payload = {
            "run_id": run_id,
            "saved_at": time.time(),
            "generation": int(ga.generation),
            "config": self._normalize_config_dict(asdict(cfg)),
            "population": [[int(gene) for gene in ind.chromosome] for ind in ga.population],
            "best_paper_score_seen": None if best_paper_score_seen == float("inf") else float(best_paper_score_seen),
            "best_gap_seen": None if best_gap_seen == float("inf") else float(best_gap_seen),
        }
        self._checkpoint_path(run_id).write_text(json.dumps(payload), encoding="utf-8")

    def _load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._checkpoint_path(run_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def resume_run(self, run_id: str, force: bool = False) -> str:
        with self._manager_lock:
            active = self._active.get(run_id)
            if active is not None and active.thread is not None and active.thread.is_alive():
                raise ValueError(f"Run {run_id} is already active")

        row = self.persistence.load_run(run_id)
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        if row.get("status") == "completed":
            raise ValueError("Cannot resume a completed run")
        if row.get("status") == "running" and not force:
            raise ValueError("Run is marked as running. Use force resume if app previously crashed.")

        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            raise ValueError("No checkpoint found for this run; cannot resume")

        cfg = self._config_from_dict(checkpoint.get("config") or row.get("config") or {})
        checkpoint_gen = int(checkpoint.get("generation", 0))
        if checkpoint_gen >= cfg.max_generations:
            raise ValueError("Run already reached max generations; nothing to resume")

        normalized_cfg = self._normalize_config_dict(asdict(cfg))
        self.persistence.mark_run_running(run_id, normalized_cfg)
        self.persistence.append_log(
            run_id,
            "info",
            f"Resuming from checkpoint at generation {checkpoint_gen}{' (forced)' if force else ''}",
        )

        active = _ActiveRun(run_id, cfg)
        t = threading.Thread(target=self._run_loop, args=(active, checkpoint), daemon=True)
        active.thread = t

        with self._manager_lock:
            self._active[run_id] = active

        t.start()
        return run_id

    def supported_benchmarks(self) -> List[str]:
        return ["TLN.inp", "hanoi.inp", "BIN.inp"]

    def start_run(self, config: RunConfig) -> str:
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        active = _ActiveRun(run_id, config)
        config_payload = self._normalize_config_dict(asdict(config))

        self.persistence.create_run(
            run_id,
            config.algorithm,
            config.network_file,
            config_payload,
        )

        t = threading.Thread(target=self._run_loop, args=(active,), daemon=True)
        active.thread = t

        with self._manager_lock:
            self._active[run_id] = active

        t.start()
        return run_id

    def stop_run(self, run_id: str) -> None:
        run = self._active.get(run_id)
        if run is None:
            return
        run.stop_event.set()
        self.persistence.append_log(run_id, "info", "Stop requested by user")

    def get_active_run_ids(self) -> List[str]:
        with self._manager_lock:
            return list(self._active.keys())

    def get_live_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self._active.get(run_id)
        if run is None:
            return None

        with run.lock:
            return {
                "run_id": run.run_id,
                "status": run.status,
                "generation": run.current_generation,
                "best_training_fitness": run.best_training_fitness,
                "best_paper_score": run.best_paper_score,
                "best_gap_to_published_pct": run.best_gap_to_published_pct,
                "started_at": run.started_at,
                "ended_at": run.ended_at,
                "config": asdict(run.config),
            }

    def _strict_objective_fn(
        self,
        network_file: str,
        inp_filepath: str,
        network,
    ):
        cache: Dict[tuple, float] = {}

        def objective(diameters: List[float]) -> float:
            key = tuple(round(float(d), 8) for d in diameters)
            cached = cache.get(key)
            if cached is not None:
                return cached

            paper_eval = self._runner._evaluate_paper_score(network_file, inp_filepath, network, diameters)
            if paper_eval.get("paper_eval_ok", 0.0) <= 0.5:
                value = 1e15
            elif paper_eval.get("paper_feasible", 0.0) > 0.5:
                value = float(paper_eval.get("paper_cost", float("inf")))
            else:
                base_cost = float(paper_eval.get("paper_cost", float("inf")))
                violation = float(paper_eval.get("paper_violation", float("inf")))
                if base_cost == float("inf"):
                    base_cost = 1e12
                if violation == float("inf"):
                    violation = 1e6
                value = base_cost + (2e5 * violation) + 1e7

            cache[key] = float(value)
            return float(value)

        return objective

    def _log(self, run_id: str, message: str, level: str = "info") -> None:
        self.persistence.append_log(run_id, level, message)

    def _run_loop(self, active: _ActiveRun, resume_state: Optional[Dict[str, Any]] = None) -> None:
        run_id = active.run_id
        cfg = active.config

        try:
            if cfg.repair_mode not in {"none", "first_generation", "every_generation"}:
                cfg.repair_mode = "every_generation"
                self._log(run_id, "Invalid repair_mode in config; defaulting to every_generation", level="warning")

            # Suppress non-critical WNTR warnings about headloss formula and convergence
            warnings.filterwarnings("ignore", message=".*Changing the headloss formula.*")
            warnings.filterwarnings("ignore", message=".*Simulation did not converge.*")

            network_path = self.data_dir / cfg.network_file
            network = parse_inp_file(str(network_path))

            diameter_options, unit_cost_lookup = self._runner._get_benchmark_cost_spec(cfg.network_file)
            published_best = self._runner.reference_scores.get(cfg.network_file, {}).get("published_best_universal_score")

            fitness_score_fn = None
            if cfg.strict_objective_for_optimization:
                fitness_score_fn = self._strict_objective_fn(cfg.network_file, str(network_path), network)

            ga = MemeticGA(
                network=network,
                population_size=cfg.population_size,
                max_generations=cfg.max_generations,
                crossover_rate=0.8,
                mutation_rate=0.1,
                local_search_intensity=cfg.local_search_intensity if cfg.algorithm == "memetic" else 0.0,
                diameter_options=diameter_options,
                unit_cost_lookup=unit_cost_lookup if unit_cost_lookup else None,
                fitness_score_fn=fitness_score_fn,
                benchmark_eval_interval=1,
                enable_early_stopping=False,
                seed=cfg.seed,
            )

            best_paper_score_seen = float("inf")
            best_gap_seen = float("inf")

            if resume_state:
                checkpoint_gen = int(resume_state.get("generation", 0))
                population_state = resume_state.get("population") or []
                ga.population = [
                    Individual(list(map(int, chrom)), ga.fitness_evaluator, ga.fitness_score_fn)
                    for chrom in population_state
                ]
                ga.generation = checkpoint_gen
                if not ga.population:
                    raise RuntimeError("Checkpoint is missing population state")

                seen_score = resume_state.get("best_paper_score_seen")
                if seen_score is not None:
                    best_paper_score_seen = float(seen_score)
                seen_gap = resume_state.get("best_gap_seen")
                if seen_gap is not None:
                    best_gap_seen = float(seen_gap)

                with active.lock:
                    active.current_generation = checkpoint_gen
                self._log(run_id, f"Resumed GA state at generation {checkpoint_gen}")
            else:
                ga.initialize_population()
                init_mean = sum(ind.fitness for ind in ga.population) / max(1, len(ga.population))
                self._log(run_id, f"Initial population fitness mean={init_mean:.2e}")
                self._save_checkpoint(run_id, cfg, ga, best_paper_score_seen, best_gap_seen)
        except Exception as exc:
            with active.lock:
                active.status = "failed"
                active.ended_at = time.time()
            self.persistence.append_log(run_id, "error", f"Run failed during startup: {exc}")
            self.persistence.finalize_run(run_id, "failed", {"error": str(exc)})
            with self._manager_lock:
                self._active.pop(run_id, None)
            return

        try:
            remaining_generations = max(0, cfg.max_generations - int(ga.generation))
            for _ in range(remaining_generations):
                if active.stop_event.is_set():
                    with active.lock:
                        active.status = "stopped"
                    self._log(run_id, "Run stopped by user")
                    break

                ga.evolve_one_generation()
                generation = ga.generation

                best_ind = min(ga.population, key=lambda i: i.fitness)
                diam_best = ga.fitness_evaluator.indices_to_diameters(best_ind.chromosome)
                best_training_fitness = float(best_ind.fitness)
                best_training_cost = float(ga.fitness_evaluator.calculate_total_cost(diam_best))

                strict_best = self._runner._evaluate_paper_score(
                    cfg.network_file,
                    str(network_path),
                    network,
                    diam_best,
                )

                should_repair = False
                if strict_best.get("paper_feasible", 0.0) <= 0.5:
                    if cfg.repair_mode == "every_generation":
                        should_repair = True
                    elif cfg.repair_mode == "first_generation" and generation == 1:
                        should_repair = True

                if should_repair:
                    repaired = self._runner._repair_to_paper_feasible(
                        cfg.network_file,
                        str(network_path),
                        network,
                        diam_best,
                    )
                    repaired_diams = repaired.get("repaired_diameters")
                    if repaired.get("paper_feasible", 0.0) > 0.5 and repaired_diams:
                        repaired_chromosome = [ga.fitness_evaluator.diameter_to_index(d) for d in repaired_diams]
                        repaired_ind = Individual(repaired_chromosome, ga.fitness_evaluator, ga.fitness_score_fn)
                        worst_idx = max(range(len(ga.population)), key=lambda j: ga.population[j].fitness)
                        if repaired_ind.fitness < ga.population[worst_idx].fitness:
                            ga.population[worst_idx] = repaired_ind
                        strict_best = repaired
                        self._log(
                            run_id,
                            f"Gen {generation}: repaired best candidate to feasible (paper_cost={strict_best.get('paper_cost', float('inf')):.2e})",
                        )
                    
                    # Check for stop after repair operation
                    if active.stop_event.is_set():
                        with active.lock:
                            active.status = "stopped"
                        self._log(run_id, "Run stopped by user")
                        break

                feasible_count = 1 if strict_best.get("paper_feasible", 0.0) > 0.5 else 0
                if cfg.strict_check_full_population_each_gen:
                    feasible_count = 0
                    pop_best_feasible_cost = float("inf")
                    for ind in ga.population:
                        # Check for stop signal between each individual evaluation
                        if active.stop_event.is_set():
                            with active.lock:
                                active.status = "stopped"
                            self._log(run_id, "Run stopped by user during full-population check")
                            break
                        
                        d = ga.fitness_evaluator.indices_to_diameters(ind.chromosome)
                        ev = self._runner._evaluate_paper_score(cfg.network_file, str(network_path), network, d)
                        if ev.get("paper_feasible", 0.0) > 0.5:
                            feasible_count += 1
                            pop_best_feasible_cost = min(pop_best_feasible_cost, float(ev.get("paper_cost", float("inf"))))
                    if pop_best_feasible_cost < float("inf"):
                        strict_best = {
                            **strict_best,
                            "paper_score": pop_best_feasible_cost,
                            "paper_cost": pop_best_feasible_cost,
                            "paper_feasible": 1.0,
                        }
                    
                    # Re-check stop signal after population feasibility check
                    if active.stop_event.is_set():
                        with active.lock:
                            active.status = "stopped"
                        self._log(run_id, "Run stopped by user")
                        break

                paper_score = float(strict_best.get("paper_score", float("inf")))
                paper_cost = float(strict_best.get("paper_cost", float("inf")))
                paper_feasible = float(strict_best.get("paper_feasible", 0.0))

                gap_pct = None
                if published_best and paper_score < float("inf") and published_best > 0:
                    gap_pct = 100.0 * (paper_score - float(published_best)) / float(published_best)

                if paper_score < best_paper_score_seen:
                    best_paper_score_seen = paper_score
                if gap_pct is not None and gap_pct < best_gap_seen:
                    best_gap_seen = gap_pct

                self.persistence.append_generation(
                    run_id,
                    {
                        "generation": generation,
                        "best_training_fitness": best_training_fitness,
                        "best_training_cost": best_training_cost,
                        "best_paper_score": paper_score,
                        "best_paper_cost": paper_cost,
                        "best_paper_feasible": paper_feasible,
                        "feasible_count": feasible_count,
                        "gap_to_published_pct": gap_pct,
                    },
                )

                log_line = (
                    f"Gen {generation}: train_fit={best_training_fitness:.2e}, "
                    f"paper_score={'inf' if paper_score == float('inf') else f'{paper_score:.2e}'}, "
                    f"feasible={'yes' if paper_feasible > 0.5 else 'no'}, "
                    f"feasible_in_pop={feasible_count}/{len(ga.population)}"
                )
                if gap_pct is not None:
                    log_line += f", gap_to_sota={gap_pct:+.2f}%"
                self._log(run_id, log_line)

                with active.lock:
                    active.current_generation = generation
                    active.best_training_fitness = best_training_fitness
                    active.best_paper_score = None if paper_score == float("inf") else paper_score
                    active.best_gap_to_published_pct = gap_pct

                self._save_checkpoint(run_id, cfg, ga, best_paper_score_seen, best_gap_seen)

            with active.lock:
                if active.status == "running":
                    active.status = "completed"
                active.ended_at = time.time()

            summary = {
                "current_generation": active.current_generation,
                "best_training_fitness": active.best_training_fitness,
                "best_paper_score": None if best_paper_score_seen == float("inf") else best_paper_score_seen,
                "best_gap_to_published_pct": None if best_gap_seen == float("inf") else best_gap_seen,
            }
            self.persistence.finalize_run(run_id, active.status, summary)
            self._log(run_id, f"Run finished with status={active.status}")

            if active.status == "completed":
                checkpoint_path = self._checkpoint_path(run_id)
                if checkpoint_path.exists():
                    checkpoint_path.unlink(missing_ok=True)

        except Exception as exc:
            with active.lock:
                active.status = "failed"
                active.ended_at = time.time()
            self.persistence.append_log(run_id, "error", f"Run failed: {exc}")
            self.persistence.finalize_run(run_id, "failed", {"error": str(exc)})

        finally:
            with self._manager_lock:
                # Keep finished run available via SQLite history; remove from active set.
                self._active.pop(run_id, None)
