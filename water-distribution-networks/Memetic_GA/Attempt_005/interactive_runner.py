import threading
import multiprocessing as mp
import time
import uuid
import warnings
import json
import os
import numpy as np
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


def process_run_worker_entry(
    base_dir_str: str,
    run_id: str,
    config_payload: Dict[str, Any],
    resume_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Top-level process worker entry (must be module-level for Windows spawn)."""
    manager = InteractiveRunManager(Path(base_dir_str))
    cfg = manager._config_from_dict(config_payload)
    active = _ActiveRun(run_id, cfg)
    manager._run_loop(active, resume_state=resume_state)


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
    strict_population_scan_mode: str  # "full", "best_first", "hybrid"
    strict_population_scan_top_k: int
    strict_population_full_scan_interval: int


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
        self.results_dir = self.attempt5_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.attempt5_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.persistence = RunPersistence(self.attempt5_dir / "run_history.sqlite")
        self._runner = BenchmarkRunner(str(self.data_dir), str(self.attempt4_results_dir))

        self._active: Dict[str, _ActiveRun] = {}
        self._process_runs: Dict[str, mp.Process] = {}
        self._manager_lock = threading.Lock()

    def _prune_process_runs(self) -> None:
        with self._manager_lock:
            dead = [rid for rid, proc in self._process_runs.items() if not proc.is_alive()]
            for rid in dead:
                self._process_runs.pop(rid, None)

    def generate_history_visualizations(self) -> List[Path]:
        output_dir = self.attempt5_dir / "results"
        visualizer = HistoryVisualizer(
            self.persistence,
            output_dir=output_dir,
            reference_scores_path=self.attempt4_results_dir / "published_reference_scores.json",
        )
        return visualizer.generate_all()

    def generate_group_statistics_visualization(
        self,
        network_file: str,
        algorithm: str,
        latest_only: bool,
        latest_limit: int = 12,
    ) -> Optional[Path]:
        output_dir = self.attempt5_dir / "results"
        visualizer = HistoryVisualizer(
            self.persistence,
            output_dir=output_dir,
            reference_scores_path=self.attempt4_results_dir / "published_reference_scores.json",
        )
        return visualizer.plot_group_run_statistics(
            network_file=network_file,
            algorithm=algorithm,
            latest_only=latest_only,
            latest_limit=latest_limit,
        )

    def generate_all_existing_group_statistics(self, latest_only: bool = False, latest_limit: int = 12) -> List[Path]:
        output_dir = self.attempt5_dir / "results"
        visualizer = HistoryVisualizer(
            self.persistence,
            output_dir=output_dir,
            reference_scores_path=self.attempt4_results_dir / "published_reference_scores.json",
        )
        return visualizer.generate_all_existing_group_statistics(latest_only=latest_only, latest_limit=latest_limit)

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

        # Backward-compatible feasibility scan behavior for old checkpoints.
        if "strict_population_scan_mode" not in normalized:
            legacy_full = bool(normalized.get("strict_check_full_population_each_gen", True))
            normalized["strict_population_scan_mode"] = "hybrid" if legacy_full else "best_first"

        mode = str(normalized.get("strict_population_scan_mode", "hybrid")).strip().lower()
        if mode not in {"full", "best_first", "hybrid"}:
            mode = "hybrid"
        normalized["strict_population_scan_mode"] = mode

        try:
            top_k = int(normalized.get("strict_population_scan_top_k", 8))
        except Exception:
            top_k = 8
        normalized["strict_population_scan_top_k"] = max(1, min(1000, top_k))

        try:
            interval = int(normalized.get("strict_population_full_scan_interval", 5))
        except Exception:
            interval = 5
        normalized["strict_population_full_scan_interval"] = max(1, min(10000, interval))
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
            repair_mode=str(normalized.get("repair_mode", "first_generation")),
            strict_population_scan_mode=str(normalized.get("strict_population_scan_mode", "hybrid")),
            strict_population_scan_top_k=int(normalized.get("strict_population_scan_top_k", 8)),
            strict_population_full_scan_interval=int(normalized.get("strict_population_full_scan_interval", 5)),
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

    def _build_synthetic_checkpoint_from_history(self, run_id: str, cfg: RunConfig) -> Optional[Dict[str, Any]]:
        """
        Best-effort resume state when the checkpoint file is missing.

        This is approximate (population cannot be perfectly reconstructed), but it
        allows extending older completed runs that only have persisted generation rows.
        """
        last = self.persistence.load_last_generation(run_id)
        if not last:
            return None

        chromosome = last.get("best_chromosome")
        if chromosome is None:
            raw = last.get("best_chromosome_json")
            if isinstance(raw, str) and raw.strip():
                try:
                    chromosome = json.loads(raw)
                except Exception:
                    chromosome = None

        if not isinstance(chromosome, list) or not chromosome:
            return None

        try:
            chromosome = [int(g) for g in chromosome]
        except Exception:
            return None

        rows = self.persistence.load_generations(run_id)
        best_paper_score_seen = None
        best_gap_seen = None
        for r in rows:
            score = r.get("best_paper_score")
            if score is not None:
                try:
                    score_val = float(score)
                    if np.isfinite(score_val):
                        best_paper_score_seen = score_val if best_paper_score_seen is None else min(best_paper_score_seen, score_val)
                except Exception:
                    pass

            gap = r.get("gap_to_published_pct")
            if gap is not None:
                try:
                    gap_val = float(gap)
                    if np.isfinite(gap_val):
                        best_gap_seen = gap_val if best_gap_seen is None else min(best_gap_seen, gap_val)
                except Exception:
                    pass

        generation = int(last.get("generation") or 0)
        pop_size = max(1, int(cfg.population_size))
        population = [chromosome[:] for _ in range(pop_size)]

        return {
            "run_id": run_id,
            "saved_at": time.time(),
            "generation": generation,
            "config": self._normalize_config_dict(asdict(cfg)),
            "population": population,
            "best_paper_score_seen": best_paper_score_seen,
            "best_gap_seen": best_gap_seen,
            "synthetic": True,
        }

    def resume_run(self, run_id: str, force: bool = False, max_generations_override: Optional[int] = None) -> str:
        with self._manager_lock:
            active = self._active.get(run_id)
            if active is not None and active.thread is not None and active.thread.is_alive():
                raise ValueError(f"Run {run_id} is already active")

        row = self.persistence.load_run(run_id)
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        if row.get("status") == "running" and not force:
            raise ValueError("Run is marked as running. Use force resume if app previously crashed.")

        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            cfg_for_synthetic = self._config_from_dict(row.get("config") or {})
            checkpoint = self._build_synthetic_checkpoint_from_history(run_id, cfg_for_synthetic)
            if checkpoint is None:
                raise ValueError("No checkpoint found for this run; cannot resume")
            self.persistence.append_log(
                run_id,
                "warning",
                "Checkpoint missing. Using synthetic resume state from persisted best chromosome (approximate resume).",
            )

        cfg = self._config_from_dict(checkpoint.get("config") or row.get("config") or {})
        if max_generations_override is not None:
            override = int(max_generations_override)
            if override <= 0:
                raise ValueError("max_generations override must be > 0")
            cfg.max_generations = override
        checkpoint_gen = int(checkpoint.get("generation", 0))
        if checkpoint_gen >= cfg.max_generations:
            raise ValueError(
                f"Run already at generation {checkpoint_gen}; set Max Gens higher than this to resume"
            )

        if row.get("status") == "completed" and max_generations_override is None:
            raise ValueError(
                f"Run is completed at generation {checkpoint_gen}. Increase Max Gens above this value to extend and resume."
            )

        normalized_cfg = self._normalize_config_dict(asdict(cfg))
        self.persistence.mark_run_running(run_id, normalized_cfg)
        self.persistence.append_log(
            run_id,
            "info",
            f"Resuming from checkpoint at generation {checkpoint_gen}{' (forced)' if force else ''}",
        )
        if max_generations_override is not None:
            self.persistence.append_log(
                run_id,
                "info",
                f"Resume override applied: max_generations={cfg.max_generations}",
            )

        active = _ActiveRun(run_id, cfg)
        t = threading.Thread(target=self._run_loop, args=(active, checkpoint), daemon=False)
        active.thread = t

        with self._manager_lock:
            self._active[run_id] = active

        t.start()
        return run_id

    def resume_run_multiprocess(self, run_id: str, force: bool = False, max_generations_override: Optional[int] = None) -> str:
        self._prune_process_runs()
        with self._manager_lock:
            active = self._active.get(run_id)
            proc = self._process_runs.get(run_id)
            if active is not None and active.thread is not None and active.thread.is_alive():
                raise ValueError(f"Run {run_id} is already active (thread)")
            if proc is not None and proc.is_alive():
                raise ValueError(f"Run {run_id} is already active (process)")

        row = self.persistence.load_run(run_id)
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        if row.get("status") == "running" and not force:
            raise ValueError("Run is marked as running. Use force resume if app previously crashed.")

        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            cfg_for_synthetic = self._config_from_dict(row.get("config") or {})
            checkpoint = self._build_synthetic_checkpoint_from_history(run_id, cfg_for_synthetic)
            if checkpoint is None:
                raise ValueError("No checkpoint found for this run; cannot resume")
            self.persistence.append_log(
                run_id,
                "warning",
                "Checkpoint missing. Using synthetic resume state from persisted best chromosome (approximate resume).",
            )

        cfg = self._config_from_dict(checkpoint.get("config") or row.get("config") or {})
        if max_generations_override is not None:
            override = int(max_generations_override)
            if override <= 0:
                raise ValueError("max_generations override must be > 0")
            cfg.max_generations = override
        checkpoint_gen = int(checkpoint.get("generation", 0))
        if checkpoint_gen >= cfg.max_generations:
            raise ValueError(
                f"Run already at generation {checkpoint_gen}; set Max Gens higher than this to resume"
            )

        if row.get("status") == "completed" and max_generations_override is None:
            raise ValueError(
                f"Run is completed at generation {checkpoint_gen}. Increase Max Gens above this value to extend and resume."
            )

        normalized_cfg = self._normalize_config_dict(asdict(cfg))
        self.persistence.mark_run_running(run_id, normalized_cfg)
        self.persistence.append_log(
            run_id,
            "info",
            f"Resuming in multiprocessing mode from checkpoint at generation {checkpoint_gen}{' (forced)' if force else ''}",
        )
        if max_generations_override is not None:
            self.persistence.append_log(
                run_id,
                "info",
                f"Resume override applied: max_generations={cfg.max_generations}",
            )

        proc = mp.Process(
            target=process_run_worker_entry,
            args=(str(self.base_dir), run_id, normalized_cfg, checkpoint),
            daemon=False,
        )
        proc.start()

        with self._manager_lock:
            self._process_runs[run_id] = proc

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

        t = threading.Thread(target=self._run_loop, args=(active,), daemon=False)
        active.thread = t

        with self._manager_lock:
            self._active[run_id] = active

        t.start()
        return run_id

    def start_run_multiprocess(self, config: RunConfig) -> str:
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        config_payload = self._normalize_config_dict(asdict(config))

        self.persistence.create_run(
            run_id,
            config.algorithm,
            config.network_file,
            config_payload,
        )

        proc = mp.Process(
            target=process_run_worker_entry,
            args=(str(self.base_dir), run_id, config_payload),
            daemon=False,
        )
        proc.start()

        with self._manager_lock:
            self._process_runs[run_id] = proc

        return run_id

    def stop_run(self, run_id: str) -> None:
        self._prune_process_runs()
        run = self._active.get(run_id)
        if run is not None:
            run.stop_event.set()
            self.persistence.append_log(run_id, "info", "Stop requested by user")
            return

        with self._manager_lock:
            proc = self._process_runs.get(run_id)

        if proc is None:
            return

        if proc.is_alive():
            try:
                proc.terminate()
                proc.join(timeout=1.0)
            except Exception:
                pass

        with self._manager_lock:
            self._process_runs.pop(run_id, None)

        self.persistence.append_log(run_id, "warning", "Run process terminated by user")
        last = self.persistence.load_last_generation(run_id)
        generations = self.persistence.load_generations(run_id)

        best_training_fitness = None
        best_paper_score = None
        best_gap_to_published_pct = None

        if generations:
            train_vals = [
                float(g["best_training_fitness"])
                for g in generations
                if g.get("best_training_fitness") is not None and np.isfinite(float(g.get("best_training_fitness")))
            ]
            if train_vals:
                best_training_fitness = float(min(train_vals))

            paper_vals = [
                float(g["best_paper_score"])
                for g in generations
                if g.get("best_paper_score") is not None and np.isfinite(float(g.get("best_paper_score")))
            ]
            if paper_vals:
                best_paper_score = float(min(paper_vals))

            gap_vals = [
                float(g["gap_to_published_pct"])
                for g in generations
                if g.get("gap_to_published_pct") is not None and np.isfinite(float(g.get("gap_to_published_pct")))
            ]
            if gap_vals:
                best_gap_to_published_pct = float(min(gap_vals))

        summary = {
            "current_generation": int(last.get("generation", 0)) if last else 0,
            "best_training_fitness": best_training_fitness,
            "best_paper_score": best_paper_score,
            "best_gap_to_published_pct": best_gap_to_published_pct,
        }
        self.persistence.finalize_run(run_id, "stopped", summary)

    def get_active_run_ids(self) -> List[str]:
        self._prune_process_runs()
        with self._manager_lock:
            return list(self._active.keys()) + list(self._process_runs.keys())

    def wait_for_active_threads(self, timeout_seconds: float = 0.25) -> bool:
        deadline = time.time() + max(0.01, float(timeout_seconds))
        while time.time() < deadline:
            with self._manager_lock:
                threads = [a.thread for a in self._active.values() if a.thread is not None and a.thread.is_alive()]
                procs = [p for p in self._process_runs.values() if p.is_alive()]

            if not threads and not procs:
                return True

            for t in threads:
                t.join(timeout=0.03)
            for p in procs:
                p.join(timeout=0.03)

        with self._manager_lock:
            remaining = [a.thread for a in self._active.values() if a.thread is not None and a.thread.is_alive()]
            remaining_procs = [p for p in self._process_runs.values() if p.is_alive()]
        return not remaining and not remaining_procs

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
        strict_eval_fn=None,
    ):
        cache: Dict[tuple, float] = {}

        def objective(diameters: List[float]) -> float:
            key = tuple(round(float(d), 8) for d in diameters)
            cached = cache.get(key)
            if cached is not None:
                return cached

            if strict_eval_fn is not None:
                paper_eval = strict_eval_fn(diameters)
            else:
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
                # Strict feasibility-first scalarization:
                # any infeasible point is dominated by any feasible point.
                value = 1e12 + base_cost + (1e9 * violation)

            # Avoid poisoning the cache with solver-failure/extreme-penalty values.
            # Re-evaluating those later can recover if hydraulics become numerically stable.
            if np.isfinite(value) and value < 1e14:
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

            strict_eval_cache: Dict[tuple, Dict[str, Any]] = {}

            def _stable_strict_eval_uncached(diameters: List[float]) -> Dict[str, Any]:
                """Conservative strict evaluation: require repeatable feasibility."""
                first = self._runner._evaluate_paper_score(
                    cfg.network_file,
                    str(network_path),
                    network,
                    diameters,
                )
                if float(first.get("paper_feasible", 0.0)) <= 0.5:
                    return first

                second = self._runner._evaluate_paper_score(
                    cfg.network_file,
                    str(network_path),
                    network,
                    diameters,
                )
                if float(second.get("paper_feasible", 0.0)) <= 0.5:
                    return {
                        **first,
                        "paper_score": float("inf"),
                        "paper_cost": float("inf"),
                        "paper_feasible": 0.0,
                        "paper_violation": float("inf"),
                        "paper_eval_note": "unstable_feasibility_on_repeat_check",
                    }

                # If both are feasible, keep the more conservative (higher) cost.
                first_cost = float(first.get("paper_cost", float("inf")))
                second_cost = float(second.get("paper_cost", float("inf")))
                chosen = first if first_cost >= second_cost else second
                chosen = {**chosen}
                chosen["paper_score"] = float(chosen.get("paper_cost", float("inf")))
                chosen["paper_feasible"] = 1.0
                return chosen

            def stable_strict_eval(diameters: List[float]) -> Dict[str, Any]:
                """Shared cached strict evaluator used by both optimization and reporting."""
                key = tuple(round(float(d), 8) for d in diameters)
                cached = strict_eval_cache.get(key)
                if cached is not None:
                    return {**cached}

                evaluated = _stable_strict_eval_uncached(diameters)
                score = float(evaluated.get("paper_score", float("inf")))

                # Cache stable finite results to keep train_fit/paper logs aligned.
                # Avoid caching failure-path extreme penalties to allow later recovery.
                if np.isfinite(score) and score < 1e14:
                    strict_eval_cache[key] = {**evaluated}

                return evaluated

            diameter_options, unit_cost_lookup = self._runner._get_benchmark_cost_spec(cfg.network_file)
            reference_entry = self._runner.reference_scores.get(cfg.network_file, {})
            published_best = reference_entry.get("published_best_universal_score")
            reference_reliability = reference_entry.get("source_reliability") or reference_entry.get("confidence")
            if published_best and float(published_best) > 0:
                self._log(
                    run_id,
                    (
                        f"Published reference comparator: {float(published_best):.2f}"
                        + (f" (source reliability: {reference_reliability})" if reference_reliability else "")
                        + ". Negative delta means below this reference under current evaluator, not a certified new SOTA."
                    ),
                    level="info",
                )

            fitness_score_fn = None
            if cfg.strict_objective_for_optimization:
                # Keep optimization objective fully aligned with logged strict paper score
                # by reusing the same stable two-pass evaluator.
                fitness_score_fn = self._strict_objective_fn(
                    cfg.network_file,
                    str(network_path),
                    network,
                    strict_eval_fn=stable_strict_eval,
                )
                self._log(run_id, "Optimization objective: strict paper feasibility-first (infeasible dominated)")
            else:
                self._log(run_id, "Optimization objective: fast proxy fitness (non-strict)", level="warning")

            # Set up periodic feasibility checkpoints for large networks
            # (prevents population from drifting infeasible)
            feasibility_checkpoint_interval = None
            feasibility_check_fn = None
            repair_fn = None
            
            if network.get_pipe_count() > 100:
                # MORE AGGRESSIVE: check every 5-10 generations for quick recovery
                feasibility_checkpoint_interval = max(5, cfg.max_generations // 30)
                if feasibility_checkpoint_interval > 15:
                    feasibility_checkpoint_interval = 15
                
                def check_feasibility(diameters: list) -> bool:
                    """Check if solution is hydraulically feasible under paper constraints."""
                    result = self._runner._evaluate_paper_score(
                        cfg.network_file, str(network_path), network, diameters
                    )
                    return result.get('paper_feasible', 0.0) > 0.5
                
                def repair_solution(diameters: list) -> list:
                    """Repair infeasible solution using targeted greedy upsizing."""
                    repair_result = self._runner._repair_to_paper_feasible(
                        cfg.network_file, str(network_path), network, diameters
                    )
                    return repair_result.get('repaired_diameters', diameters)
                
                feasibility_check_fn = check_feasibility
                repair_fn = repair_solution
                self._log(run_id, f"Feasibility checkpoints enabled: every {feasibility_checkpoint_interval} generations (aggressive mode)")

            ga = MemeticGA(
                network=network,
                population_size=cfg.population_size,
                max_generations=cfg.max_generations,
                crossover_rate=0.8,
                mutation_rate=0.1,
                local_search_intensity=(
                    min(cfg.local_search_intensity, 0.35)
                    if (cfg.algorithm == "memetic" and cfg.strict_objective_for_optimization and network.get_pipe_count() > 100)
                    else (cfg.local_search_intensity if cfg.algorithm == "memetic" else 0.0)
                ),
                diameter_options=diameter_options,
                unit_cost_lookup=unit_cost_lookup if unit_cost_lookup else None,
                fitness_score_fn=fitness_score_fn,
                benchmark_eval_interval=1,
                enable_early_stopping=False,
                feasibility_checkpoint_interval=feasibility_checkpoint_interval,
                feasibility_check_fn=feasibility_check_fn,
                repair_fn=repair_fn,
                seed=cfg.seed,
            )

            if cfg.algorithm == "memetic" and cfg.strict_objective_for_optimization and network.get_pipe_count() > 100 and cfg.local_search_intensity > 0.35:
                self._log(run_id, "Capped memetic local_search_intensity to 0.35 for large strict benchmark stability")

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

                # Strict-mode feasibility anchor: keep one deterministic high-diameter
                # candidate in population to avoid all-infeasible deadlocks.
                if cfg.strict_objective_for_optimization and ga.population:
                    anchor_diams = [ga.fitness_evaluator.diameter_values[-1]] * ga.num_pipes
                    anchor_eval = self._runner._evaluate_paper_score(
                        cfg.network_file,
                        str(network_path),
                        network,
                        anchor_diams,
                    )
                    if anchor_eval.get("paper_feasible", 0.0) > 0.5:
                        anchor_chromosome = [
                            ga.fitness_evaluator.diameter_to_index(d)
                            for d in anchor_diams
                        ]
                        anchor_ind = Individual(anchor_chromosome, ga.fitness_evaluator, ga.fitness_score_fn)
                        worst_idx = max(range(len(ga.population)), key=lambda j: ga.population[j].fitness)
                        ga.population[worst_idx] = anchor_ind
                        self._log(run_id, "Injected strict feasible anchor individual into initial population")

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
            no_feasible_streak = 0
            feasibility_scan_mode = str(cfg.strict_population_scan_mode or "hybrid").strip().lower()
            if feasibility_scan_mode not in {"full", "best_first", "hybrid"}:
                feasibility_scan_mode = "hybrid"
            top_k_scan = max(1, int(cfg.strict_population_scan_top_k or 8))
            full_scan_interval = max(1, int(cfg.strict_population_full_scan_interval or 5))
            for _ in range(remaining_generations):
                if active.stop_event.is_set():
                    with active.lock:
                        active.status = "stopped"
                    self._log(run_id, "Run stopped by user")
                    break

                ga.evolve_one_generation()
                generation = ga.generation

                # Run periodic feasibility checkpoint when configured.
                if (
                    ga.feasibility_checkpoint_interval
                    and generation % int(ga.feasibility_checkpoint_interval) == 0
                ):
                    ga._check_feasibility_at_checkpoint()

                best_ind = min(ga.population, key=lambda i: i.fitness)
                diam_best = ga.fitness_evaluator.indices_to_diameters(best_ind.chromosome)
                best_snapshot_chromosome = [int(gene) for gene in best_ind.chromosome]
                best_training_fitness = float(best_ind.fitness)
                best_training_cost = float(ga.fitness_evaluator.calculate_total_cost(diam_best))

                strict_best = stable_strict_eval(diam_best)

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
                    repaired_verified = None
                    if repaired_diams:
                        repaired_verified = stable_strict_eval(repaired_diams)
                    if repaired_verified is not None and repaired_verified.get("paper_feasible", 0.0) > 0.5:
                        repaired_chromosome = [int(ga.fitness_evaluator.diameter_to_index(d)) for d in repaired_diams]
                        repaired_ind = Individual(repaired_chromosome, ga.fitness_evaluator, ga.fitness_score_fn)
                        worst_idx = max(range(len(ga.population)), key=lambda j: ga.population[j].fitness)
                        if repaired_ind.fitness < ga.population[worst_idx].fitness:
                            ga.population[worst_idx] = repaired_ind
                        strict_best = repaired_verified
                        best_snapshot_chromosome = repaired_chromosome
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

                def _eval_population_subset(population_subset: List[Individual]) -> tuple[int, float, Optional[List[int]]]:
                    feasible_local = 0
                    best_cost_local = float("inf")
                    best_chr_local: Optional[List[int]] = None
                    for ind in population_subset:
                        if active.stop_event.is_set():
                            with active.lock:
                                active.status = "stopped"
                            self._log(run_id, "Run stopped by user during feasibility scan")
                            break
                        d = ga.fitness_evaluator.indices_to_diameters(ind.chromosome)
                        ev = stable_strict_eval(d)
                        if ev.get("paper_feasible", 0.0) > 0.5:
                            feasible_local += 1
                            cost_local = float(ev.get("paper_cost", float("inf")))
                            if cost_local < best_cost_local:
                                best_cost_local = cost_local
                                best_chr_local = list(ind.chromosome)
                    return feasible_local, best_cost_local, best_chr_local

                feasible_count = 1 if strict_best.get("paper_feasible", 0.0) > 0.5 else 0
                scan_note = "single-best"

                # Explicit override from UI checkbox: force full-pop scan each generation.
                if cfg.strict_check_full_population_each_gen:
                    scan_mode_effective = "full"
                elif feasibility_scan_mode == "full":
                    scan_mode_effective = "full"
                elif feasibility_scan_mode == "best_first":
                    scan_mode_effective = "best_first"
                else:
                    scan_mode_effective = "hybrid"

                pop_best_feasible_cost = float("inf")
                pop_best_feasible_chromosome: Optional[List[int]] = None

                if scan_mode_effective == "full":
                    feasible_count, pop_best_feasible_cost, pop_best_feasible_chromosome = _eval_population_subset(list(ga.population))
                    scan_note = f"full/{len(ga.population)}"
                elif scan_mode_effective == "best_first":
                    # Scan candidates by training fitness ranking until first feasible.
                    ranked = sorted(ga.population, key=lambda ind: ind.fitness)
                    checked = 0
                    feasible_count = 0
                    for ind in ranked:
                        checked += 1
                        f_cnt, best_cost_local, best_chr_local = _eval_population_subset([ind])
                        if f_cnt > 0:
                            feasible_count = 1
                            pop_best_feasible_cost = best_cost_local
                            pop_best_feasible_chromosome = best_chr_local
                            break
                    scan_note = f"best_first/{checked}"
                else:
                    # Hybrid: top-k each generation for speed, full scan periodically for exact feasible_in_pop.
                    ranked = sorted(ga.population, key=lambda ind: ind.fitness)
                    k = min(len(ranked), top_k_scan)
                    feasible_k, best_cost_k, best_chr_k = _eval_population_subset(ranked[:k])
                    feasible_count = feasible_k
                    pop_best_feasible_cost = best_cost_k
                    pop_best_feasible_chromosome = best_chr_k
                    scan_note = f"topk/{k}"

                    if generation % full_scan_interval == 0:
                        feasible_full, best_cost_full, best_chr_full = _eval_population_subset(ranked)
                        feasible_count = feasible_full
                        scan_note = f"full/{len(ranked)}"
                        if best_cost_full < pop_best_feasible_cost:
                            pop_best_feasible_cost = best_cost_full
                            pop_best_feasible_chromosome = best_chr_full

                if pop_best_feasible_cost < float("inf"):
                    strict_best = {
                        **strict_best,
                        "paper_score": pop_best_feasible_cost,
                        "paper_cost": pop_best_feasible_cost,
                        "paper_feasible": 1.0,
                    }
                    if pop_best_feasible_chromosome is not None:
                        best_snapshot_chromosome = [int(gene) for gene in pop_best_feasible_chromosome]

                # Re-check stop signal after scan
                if active.stop_event.is_set():
                    with active.lock:
                        active.status = "stopped"
                    self._log(run_id, "Run stopped by user")
                    break

                # Guardrail for first-generation-only repair mode:
                # if strict-mode population collapses to all-infeasible for several
                # generations, run an occasional rescue repair to reintroduce feasibility.
                if cfg.strict_objective_for_optimization:
                    if feasible_count <= 0:
                        no_feasible_streak += 1
                    else:
                        no_feasible_streak = 0

                    if (
                        cfg.repair_mode == "first_generation"
                        and no_feasible_streak >= 3
                        and generation % 5 == 0
                    ):
                        rescued = self._runner._repair_to_paper_feasible(
                            cfg.network_file,
                            str(network_path),
                            network,
                            diam_best,
                        )
                        rescued_diams = rescued.get("repaired_diameters")
                        rescued_verified = None
                        if rescued_diams:
                            rescued_verified = stable_strict_eval(rescued_diams)
                        if rescued_verified is not None and rescued_verified.get("paper_feasible", 0.0) > 0.5 and rescued_diams:
                            rescued_chr = [int(ga.fitness_evaluator.diameter_to_index(d)) for d in rescued_diams]
                            rescued_ind = Individual(rescued_chr, ga.fitness_evaluator, ga.fitness_score_fn)
                            worst_idx = max(range(len(ga.population)), key=lambda j: ga.population[j].fitness)
                            ga.population[worst_idx] = rescued_ind
                            strict_best = rescued_verified
                            feasible_count = 1
                            no_feasible_streak = 0
                            best_snapshot_chromosome = rescued_chr
                            self._log(
                                run_id,
                                (
                                    "Applied feasibility rescue in first_generation mode "
                                    f"(paper_cost={float(rescued.get('paper_cost', float('inf'))):.2e})"
                                ),
                                level="warning",
                            )

                paper_score = float(strict_best.get("paper_score", float("inf")))
                paper_cost = float(strict_best.get("paper_cost", float("inf")))
                paper_feasible = float(strict_best.get("paper_feasible", 0.0))

                # Metrics are already based on stable repeat-checked strict evaluation.

                if paper_score < best_paper_score_seen:
                    best_paper_score_seen = paper_score

                current_gap_pct = None
                best_gap_pct = None
                if published_best and float(published_best) > 0:
                    published_best_value = float(published_best)
                    if paper_score < float("inf"):
                        current_gap_pct = 100.0 * (paper_score - published_best_value) / published_best_value
                    if best_paper_score_seen < float("inf"):
                        best_gap_pct = 100.0 * (best_paper_score_seen - published_best_value) / published_best_value

                if best_gap_pct is not None and best_gap_pct < best_gap_seen:
                    best_gap_seen = best_gap_pct

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
                        "gap_to_published_pct": best_gap_pct,
                        "best_chromosome_json": best_snapshot_chromosome,
                    },
                )

                log_line = (
                    f"Gen {generation}: train_fit={best_training_fitness:.2e}, "
                    f"paper_score_now={'inf' if paper_score == float('inf') else f'{paper_score:.2e}'}, "
                    f"paper_score_best={'inf' if best_paper_score_seen == float('inf') else f'{best_paper_score_seen:.2e}'}, "
                    f"feasible={'yes' if paper_feasible > 0.5 else 'no'}, "
                    f"feasible_in_pop={feasible_count}/{len(ga.population)}"
                )
                log_line += f", feas_scan={scan_note}"
                if paper_feasible <= 0.5 and best_paper_score_seen < float("inf"):
                    log_line += ", best_from_previous_generations=yes"
                if best_gap_pct is not None:
                    log_line += f", delta_to_ref_best={best_gap_pct:+.2f}%"
                self._log(run_id, log_line)

                with active.lock:
                    active.current_generation = generation
                    active.best_training_fitness = best_training_fitness
                    active.best_paper_score = None if paper_score == float("inf") else paper_score
                    active.best_gap_to_published_pct = best_gap_pct

                self._save_checkpoint(run_id, cfg, ga, best_paper_score_seen, best_gap_seen)
                
                # Export live results CSV every 5 generations (for graph generation without stopping)
                if generation % 5 == 0:
                    try:
                        results_csv = self.attempt5_dir / "live_results.csv"
                        self.persistence.export_live_results_csv(results_csv)
                    except Exception as e:
                        self._log(run_id, f"Warning: Failed to export live results: {e}", level="warning")

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
