"""
Test the Memetic GA on water distribution network benchmarks.

Runs the algorithm on easy/medium/hard networks and compares:
- Memetic GA (with local search)
- Standard GA (without local search)

Generates comprehensive results and statistics.
"""

import os
import sys
import json
import tempfile
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import time
import wntr

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from network_parser import parse_inp_file, WaterNetwork
from fitness_evaluator import FitnessEvaluator, AVAILABLE_DIAMETERS
from memetic_ga import MemeticGA, Individual
from visualize_results import ResultsVisualizer


class BenchmarkRunner:
    """Runs and evaluates algorithms on benchmark instances."""
    
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.reference_scores = self._load_reference_scores()

    def _build_cached_strict_paper_objective(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork
    ):
        """Return a cached strict-paper objective for small networks (e.g., TLN)."""
        cache: Dict[Tuple[float, ...], float] = {}

        def objective(diameters: List[float]) -> float:
            key = tuple(round(float(d), 8) for d in diameters)
            cached = cache.get(key)
            if cached is not None:
                return cached

            paper_eval = self._evaluate_paper_score(network_file, inp_filepath, network, diameters)
            if paper_eval.get('paper_eval_ok', 0.0) <= 0.5:
                value = 1e15
            elif paper_eval.get('paper_feasible', 0.0) > 0.5:
                value = float(paper_eval.get('paper_cost', float('inf')))
            else:
                # Finite penalty keeps selection pressure among infeasible candidates.
                base_cost = float(paper_eval.get('paper_cost', float('inf')))
                violation = float(paper_eval.get('paper_violation', float('inf')))
                if not np.isfinite(base_cost):
                    base_cost = 1e12
                if not np.isfinite(violation):
                    violation = 1e6
                value = base_cost + (2e5 * violation) + 1e7

            cache[key] = float(value)
            return float(value)

        return objective

    def _load_reference_scores(self) -> Dict:
        """Load benchmark specs and published references from JSON if present."""
        ref_path = os.path.join(self.results_dir, 'published_reference_scores.json')
        if not os.path.exists(ref_path):
            return {}
        with open(ref_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _get_benchmark_cost_spec(self, network_file: str) -> Tuple[List[float], Dict[float, float]]:
        """Get benchmark diameter catalog and unit costs for a network."""
        spec = self.reference_scores.get(network_file, {})
        diameter_rows = spec.get('diameter_set', [])

        if not diameter_rows:
            return AVAILABLE_DIAMETERS.copy(), {}

        diameters = [float(row['diameter_m']) for row in diameter_rows]
        costs = {float(row['diameter_m']): float(row['unit_cost_per_m']) for row in diameter_rows}
        return diameters, costs

    @staticmethod
    def _get_min_head_requirement(network_file: str) -> float:
        """Benchmark minimum pressure head requirement in meters."""
        defaults = {
            'TLN.inp': 30.0,
            'hanoi.inp': 30.0,
            'BIN.inp': 20.0
        }
        return defaults.get(network_file, 20.0)

    @staticmethod
    def _load_wntr_model_robust(inp_filepath: str):
        """
        Load INP file with fallback for non-UTF8 encoded benchmark files.
        """
        try:
            return wntr.network.WaterNetworkModel(inp_filepath), None
        except UnicodeDecodeError:
            with open(inp_filepath, 'rb') as f:
                raw = f.read()

            # Latin-1 keeps byte values lossless and is robust for legacy INP files.
            text = raw.decode('latin-1')
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False, encoding='utf-8')
            tmp.write(text)
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()
            return wntr.network.WaterNetworkModel(tmp_path), tmp_path

    def _evaluate_paper_score(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        diameters: List[float],
        return_diagnostics: bool = False
    ) -> Dict:
        """
        Evaluate final solution with paper-style benchmark metric.

        Score = total benchmark cost if hydraulically feasible, else +inf.
        """
        diameter_options, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)

        # Cost from benchmark table (strictly tabulated per diameter)
        cost = 0.0
        try:
            for i, d in enumerate(diameters):
                pipe = network.pipes_list[i]
                cost += unit_cost_lookup[float(d)] * pipe.length
        except KeyError:
            result = {
                'paper_score': float('inf'),
                'paper_cost': float('inf'),
                'paper_feasible': 0.0,
                'paper_violation': float('inf'),
                'paper_min_pressure': float('-inf'),
                'paper_eval_ok': 0.0,
                'paper_eval_note': 'diameter_not_in_benchmark_catalog'
            }
            if return_diagnostics:
                result['paper_diagnostics'] = {}
            return result

        # Hydraulic feasibility via EPANET simulator through WNTR
        temp_inp = None
        try:
            wn, temp_inp = self._load_wntr_model_robust(inp_filepath)

            for i, pipe in enumerate(network.pipes_list):
                if pipe.id not in wn.pipe_name_list:
                    continue
                wn.get_link(pipe.id).diameter = float(diameters[i])

            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            min_head = self._get_min_head_requirement(network_file)
            pressure_ts = results.node['pressure']
            final_pressure = pressure_ts.iloc[-1]

            junction_names = [j for j in network.junctions.keys() if j in final_pressure.index]
            junction_pressures = final_pressure.loc[junction_names]

            shortfalls = np.maximum(0.0, min_head - junction_pressures.values)
            total_violation = float(np.sum(shortfalls))
            min_pressure = float(np.min(junction_pressures.values)) if len(junction_pressures.values) > 0 else float('-inf')
            feasible = 1.0 if total_violation <= 1e-9 else 0.0
            score = cost if feasible > 0 else float('inf')

            result = {
                'paper_score': float(score),
                'paper_cost': float(cost),
                'paper_feasible': feasible,
                'paper_violation': total_violation,
                'paper_min_pressure': min_pressure,
                'paper_eval_ok': 1.0,
                'paper_eval_note': ''
            }

            if return_diagnostics:
                deficits = {
                    node: float(max(0.0, min_head - p))
                    for node, p in junction_pressures.items()
                }
                result['paper_diagnostics'] = {
                    'junction_pressures': {node: float(p) for node, p in junction_pressures.items()},
                    'junction_deficits': deficits,
                    'link_headloss_abs': {
                        link: float(abs(v))
                        for link, v in results.link['headloss'].iloc[-1].items()
                    }
                }

            return result
        except Exception as e:
            result = {
                'paper_score': float('inf'),
                'paper_cost': float(cost),
                'paper_feasible': 0.0,
                'paper_violation': float('inf'),
                'paper_min_pressure': float('-inf'),
                'paper_eval_ok': 0.0,
                'paper_eval_note': f'wntr_eval_failed: {e}'
            }
            if return_diagnostics:
                result['paper_diagnostics'] = {}
            return result
        finally:
            if temp_inp and os.path.exists(temp_inp):
                try:
                    os.remove(temp_inp)
                except OSError:
                    pass

    def _repair_to_paper_feasible(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        diameters: List[float]
    ) -> Dict:
        """
        Greedy deterministic repair: upsize hydraulically critical pipes until
        paper feasibility is achieved or upgrade budget is exhausted.
        """
        diameter_options, _ = self._get_benchmark_cost_spec(network_file)
        if not diameter_options:
            return {
                'repaired': 0.0,
                'repair_steps': 0.0,
                'repair_note': 'no_diameter_catalog',
                'paper_score': float('inf'),
                'paper_cost': float('inf'),
                'paper_feasible': 0.0,
                'paper_violation': float('inf'),
                'paper_min_pressure': float('-inf'),
                'paper_eval_ok': 0.0,
                'paper_eval_note': 'repair_not_possible'
            }

        options = sorted(float(d) for d in diameter_options)
        _, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)
        current = [float(d) for d in diameters]

        def to_index(d: float) -> int:
            return int(np.argmin([abs(v - d) for v in options]))

        idx = [to_index(d) for d in current]

        current_eval = self._evaluate_paper_score(
            network_file,
            inp_filepath,
            network,
            current,
            return_diagnostics=True
        )
        if current_eval['paper_feasible'] > 0.5:
            current_eval.update({'repaired': 0.0, 'repair_steps': 0.0, 'repair_note': 'already_feasible'})
            return current_eval

        pipe_count = network.get_pipe_count()
        if pipe_count > 300:
            max_steps = 120
        elif pipe_count > 100:
            max_steps = 220
        else:
            max_steps = max(120, min(800, pipe_count * 7))

        for step in range(1, max_steps + 1):
            upgradable = [
                i for i in range(len(idx))
                if idx[i] < (len(options) - 1)
            ]
            if not upgradable:
                break

            diagnostics = current_eval.get('paper_diagnostics', {})
            deficits = diagnostics.get('junction_deficits', {})
            headloss_abs = diagnostics.get('link_headloss_abs', {})

            # Pressure-deficit-aware scoring with cost-efficiency.
            # Pipes feeding high-deficit nodes get priority, adjusted by headloss and upgrade cost.
            node_deficits = {
                node: d for node, d in deficits.items() if d > 1e-9
            }

            if node_deficits:
                pipe_weight = {str(network.pipes_list[i].id): 0.0 for i in upgradable}
                for pipe in network.pipes_list:
                    a = str(pipe.node1)
                    b = str(pipe.node2)
                    w = node_deficits.get(a, 0.0) + node_deficits.get(b, 0.0)
                    if w > 0:
                        pid = str(pipe.id)
                        if pid in pipe_weight:
                            pipe_weight[pid] += float(w)
            else:
                pipe_weight = {str(network.pipes_list[i].id): 0.0 for i in upgradable}

            best_i = None
            best_score = -1.0

            for i in upgradable:
                pipe = network.pipes_list[i]
                pid = str(pipe.id)

                next_d = options[idx[i] + 1]
                cur_d = options[idx[i]]
                delta_cost = (unit_cost_lookup.get(next_d, 0.0) - unit_cost_lookup.get(cur_d, 0.0)) * pipe.length
                if delta_cost <= 0:
                    delta_cost = 1e-9

                demand_weight = pipe_weight.get(pid, 0.0)
                hydraulic_weight = headloss_abs.get(pid, 1.0)

                # If no deficits are directly adjacent, fallback to a classic headloss sensitivity proxy.
                if demand_weight <= 0:
                    demand_weight = pipe.length / max(current[i] ** 4.871, 1e-12)

                score = (demand_weight * max(hydraulic_weight, 1e-9)) / delta_cost
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i is None:
                break

            idx[best_i] += 1
            current[best_i] = options[idx[best_i]]

            current_eval = self._evaluate_paper_score(
                network_file,
                inp_filepath,
                network,
                current,
                return_diagnostics=True
            )
            if current_eval['paper_feasible'] > 0.5:
                current_eval.update({
                    'repaired': 1.0,
                    'repair_steps': float(step),
                    'repair_note': 'feasible_after_targeted_cost_aware_upsizing'
                })
                break

        if current_eval.get('paper_feasible', 0.0) <= 0.5:
            current_eval.update({
                'repaired': 1.0,
                'repair_steps': float(max_steps),
                'repair_note': 'repair_budget_exhausted_infeasible'
            })

        # Safety fallback: run legacy global headloss-sensitivity repair and pick cheaper feasible result.
        fallback_eval = self._repair_to_paper_feasible_global(
            network_file,
            inp_filepath,
            network,
            diameters,
            options
        )

        if current_eval.get('paper_feasible', 0.0) > 0.5 and fallback_eval.get('paper_feasible', 0.0) > 0.5:
            if fallback_eval.get('paper_cost', float('inf')) < current_eval.get('paper_cost', float('inf')):
                fallback_eval['repair_note'] = f"{fallback_eval.get('repair_note', '')}|selected_over_targeted"
                return fallback_eval
            current_eval['repair_note'] = f"{current_eval.get('repair_note', '')}|selected_over_global"
            return current_eval

        if fallback_eval.get('paper_feasible', 0.0) > current_eval.get('paper_feasible', 0.0):
            fallback_eval['repair_note'] = f"{fallback_eval.get('repair_note', '')}|selected_over_targeted"
            return fallback_eval

        return current_eval

    def _repair_to_paper_feasible_global(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        diameters: List[float],
        options: List[float]
    ) -> Dict:
        """
        Legacy global repair heuristic: repeatedly upsize pipes with highest
        headloss sensitivity proxy until feasibility or budget cap.
        """
        current = [float(d) for d in diameters]

        def to_index(d: float) -> int:
            return int(np.argmin([abs(v - d) for v in options]))

        idx = [to_index(d) for d in current]
        current_eval = self._evaluate_paper_score(network_file, inp_filepath, network, current)
        if current_eval['paper_feasible'] > 0.5:
            current_eval.update({'repaired': 0.0, 'repair_steps': 0.0, 'repair_note': 'already_feasible_global'})
            return current_eval

        pipe_count = network.get_pipe_count()
        if pipe_count > 300:
            max_steps = 80
            check_interval = 4
            batch_size = 10
        elif pipe_count > 80:
            max_steps = 180
            check_interval = 3
            batch_size = 8
        else:
            max_steps = max(80, min(800, pipe_count * 8))
            check_interval = 1
            batch_size = 1

        for step in range(1, max_steps + 1):
            upgradable = [i for i in range(len(idx)) if idx[i] < (len(options) - 1)]
            if not upgradable:
                break

            ranked = sorted(
                upgradable,
                key=lambda i: network.pipes_list[i].length / max(current[i] ** 4.871, 1e-12),
                reverse=True
            )
            to_upgrade = ranked[:min(batch_size, len(ranked))]
            for critical_i in to_upgrade:
                idx[critical_i] += 1
                current[critical_i] = options[idx[critical_i]]

            if (step % check_interval == 0) or (step == max_steps):
                current_eval = self._evaluate_paper_score(network_file, inp_filepath, network, current)
                if current_eval['paper_feasible'] > 0.5:
                    current_eval.update({
                        'repaired': 1.0,
                        'repair_steps': float(step),
                        'repair_note': 'feasible_after_global_greedy_upsizing'
                    })
                    return current_eval

        current_eval.update({
            'repaired': 1.0,
            'repair_steps': float(max_steps),
            'repair_note': 'global_repair_budget_exhausted_infeasible'
        })
        return current_eval

    def _polish_feasible_paper_cost(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        diameters: List[float],
        base_eval: Dict
    ) -> Dict:
        """
        Bounded post-process: attempt a few one-step downsizes that keep
        hydraulic feasibility and lower paper cost.
        """
        if base_eval.get('paper_feasible', 0.0) <= 0.5:
            return {
                **base_eval,
                'paper_polished': 0.0,
                'paper_polish_attempts': 0.0,
                'paper_polish_improvements': 0.0,
                'paper_polish_note': 'not_feasible'
            }

        diameter_options, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)
        if not diameter_options or not unit_cost_lookup:
            return {
                **base_eval,
                'paper_polished': 0.0,
                'paper_polish_attempts': 0.0,
                'paper_polish_improvements': 0.0,
                'paper_polish_note': 'no_catalog'
            }

        options = sorted(float(d) for d in diameter_options)
        current = [float(d) for d in diameters]

        def to_index(d: float) -> int:
            return int(np.argmin([abs(v - d) for v in options]))

        idx = [to_index(d) for d in current]
        current_eval = dict(base_eval)

        pipe_count = network.get_pipe_count()
        if pipe_count <= 12:
            max_attempts = 4
        elif pipe_count <= 80:
            max_attempts = 6
        else:
            max_attempts = 8

        # Try highest immediate savings opportunities first.
        candidates = []
        for i, pipe in enumerate(network.pipes_list):
            if idx[i] <= 0:
                continue
            cur_d = options[idx[i]]
            lower_d = options[idx[i] - 1]
            delta = (unit_cost_lookup.get(cur_d, 0.0) - unit_cost_lookup.get(lower_d, 0.0)) * pipe.length
            if delta > 0:
                candidates.append((delta, i))
        candidates.sort(reverse=True)

        attempts = 0
        improvements = 0

        for _, i in candidates:
            if attempts >= max_attempts:
                break

            old_idx = idx[i]
            idx[i] = old_idx - 1
            current[i] = options[idx[i]]
            attempts += 1

            trial_eval = self._evaluate_paper_score(network_file, inp_filepath, network, current)
            if (
                trial_eval.get('paper_feasible', 0.0) > 0.5 and
                trial_eval.get('paper_cost', float('inf')) < current_eval.get('paper_cost', float('inf'))
            ):
                current_eval = trial_eval
                improvements += 1
            else:
                idx[i] = old_idx
                current[i] = options[old_idx]

        current_eval.update({
            'paper_polished': 1.0 if improvements > 0 else 0.0,
            'paper_polish_attempts': float(attempts),
            'paper_polish_improvements': float(improvements),
            'paper_polish_note': 'bounded_one_step_downsize'
        })
        return current_eval
        
    def run_memetic_ga(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        population_size: int = 30,
        max_generations: int = 30,
        local_search_intensity: float = 0.5,
        use_strict_paper_objective: bool = False,
        enable_early_stopping: bool = True,
        seed: int = 42
    ) -> Tuple[Individual, List[float], List[float], Dict, float]:
        """
        Run Memetic GA on network.
        
        Returns:
            (best_individual, best_fitness_history, avg_fitness_history,
             benchmark_history, runtime)
        """
        start_time = time.time()
        diameter_options, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)
        fitness_score_fn = None
        if use_strict_paper_objective:
            fitness_score_fn = self._build_cached_strict_paper_objective(
                network_file,
                inp_filepath,
                network
            )
        
        ga = MemeticGA(
            network,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            local_search_intensity=local_search_intensity,
            diameter_options=diameter_options,
            unit_cost_lookup=unit_cost_lookup if unit_cost_lookup else None,
            fitness_score_fn=fitness_score_fn,
            benchmark_eval_interval=5,
            enable_early_stopping=enable_early_stopping,
            seed=seed
        )
        
        best_individual, best_hist, avg_hist, benchmark_hist = ga.run()
        runtime = time.time() - start_time
        
        return best_individual, best_hist, avg_hist, benchmark_hist, runtime
    
    def run_standard_ga(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        population_size: int = 30,
        max_generations: int = 30,
        use_strict_paper_objective: bool = False,
        enable_early_stopping: bool = True,
        seed: int = 42
    ) -> Tuple[Individual, List[float], List[float], Dict, float]:
        """
        Run GA without local search (standard GA).
        
        Returns:
            (best_individual, best_fitness_history, avg_fitness_history,
             benchmark_history, runtime)
        """
        start_time = time.time()
        diameter_options, unit_cost_lookup = self._get_benchmark_cost_spec(network_file)
        fitness_score_fn = None
        if use_strict_paper_objective:
            fitness_score_fn = self._build_cached_strict_paper_objective(
                network_file,
                inp_filepath,
                network
            )
        
        ga = MemeticGA(
            network,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            local_search_intensity=0.0,  # Disable local search
            diameter_options=diameter_options,
            unit_cost_lookup=unit_cost_lookup if unit_cost_lookup else None,
            fitness_score_fn=fitness_score_fn,
            benchmark_eval_interval=5,
            enable_early_stopping=enable_early_stopping,
            seed=seed
        )
        
        best_individual, best_hist, avg_hist, benchmark_hist = ga.run()
        runtime = time.time() - start_time
        
        return best_individual, best_hist, avg_hist, benchmark_hist, runtime
    
    def evaluate_solution(
        self,
        network_file: str,
        inp_filepath: str,
        individual: Individual,
        network: WaterNetwork,
        apply_paper_polish: bool = True
    ) -> Dict:
        """Evaluate solution and return detailed metrics."""
        
        diameters = individual.evaluator.indices_to_diameters(individual.chromosome)
        cost = individual.evaluator.calculate_total_cost(diameters)
        universal = individual.evaluator.evaluate_universal_score(diameters)
        raw_paper_eval = self._evaluate_paper_score(network_file, inp_filepath, network, diameters)
        if raw_paper_eval['paper_feasible'] > 0.5:
            paper_eval = raw_paper_eval
            repair_meta = {'paper_repaired': 0.0, 'paper_repair_steps': 0.0, 'paper_repair_note': 'not_needed'}
        else:
            repaired_eval = self._repair_to_paper_feasible(network_file, inp_filepath, network, diameters)
            paper_eval = repaired_eval if repaired_eval.get('paper_feasible', 0.0) > 0.5 else raw_paper_eval
            repair_meta = {
                'paper_repaired': repaired_eval.get('repaired', 0.0),
                'paper_repair_steps': repaired_eval.get('repair_steps', 0.0),
                'paper_repair_note': repaired_eval.get('repair_note', '')
            }

        if apply_paper_polish:
            polished_eval = self._polish_feasible_paper_cost(
                network_file,
                inp_filepath,
                network,
                diameters,
                paper_eval
            )
            paper_eval = polished_eval
            polish_meta = {
                'paper_polished': polished_eval.get('paper_polished', 0.0),
                'paper_polish_attempts': polished_eval.get('paper_polish_attempts', 0.0),
                'paper_polish_improvements': polished_eval.get('paper_polish_improvements', 0.0),
                'paper_polish_note': polished_eval.get('paper_polish_note', '')
            }
        else:
            polish_meta = {
                'paper_polished': 0.0,
                'paper_polish_attempts': 0.0,
                'paper_polish_improvements': 0.0,
                'paper_polish_note': 'disabled'
            }
        
        # Calculate some pipe statistics
        diameter_counts = {}
        for idx in individual.chromosome:
            d = individual.evaluator.index_to_diameter(idx)
            diameter_counts[d] = diameter_counts.get(d, 0) + 1
        
        return {
            'fitness': individual.fitness,
            'cost': cost,
            'universal_score': universal['score'],
            'universal_violation': universal['violation'],
            'universal_feasible': universal['is_feasible'],
            'paper_score': paper_eval['paper_score'],
            'paper_cost': paper_eval['paper_cost'],
            'paper_feasible': paper_eval['paper_feasible'],
            'paper_violation': paper_eval['paper_violation'],
            'paper_min_pressure': paper_eval['paper_min_pressure'],
            'paper_eval_ok': paper_eval['paper_eval_ok'],
            'paper_eval_note': paper_eval['paper_eval_note'],
            'paper_raw_score': raw_paper_eval['paper_score'],
            'paper_raw_feasible': raw_paper_eval['paper_feasible'],
            'paper_raw_violation': raw_paper_eval['paper_violation'],
            'paper_raw_min_pressure': raw_paper_eval['paper_min_pressure'],
            'paper_raw_eval_note': raw_paper_eval['paper_eval_note'],
            **repair_meta,
            **polish_meta,
            'avg_diameter': np.mean(diameters),
            'max_diameter': max(diameters),
            'min_diameter': min(diameters),
            'diameter_distribution': diameter_counts
        }
    
    def run_benchmark(
        self,
        network_file: str,
        difficulty: str,
        num_runs: int = 3
    ) -> Dict:
        """
        Run complete benchmark on network file.
        
        Tests both Memetic GA and Standard GA multiple times.
        """
        
        print(f"\n{'='*80}")
        print(f"Running benchmark: {network_file} ({difficulty})")
        print(f"{'='*80}")

        # Per-benchmark configuration tuned for stronger search on larger networks.
        benchmark_config = {
            'TLN.inp': {
                'population_size': 60,
                'max_generations': 120,
                'local_search_intensity': 0.7
            },
            'hanoi.inp': {
                'population_size': 80,
                'max_generations': 100,
                'local_search_intensity': 0.8
            },
            'BIN.inp': {
                'population_size': 70,
                'max_generations': 150,
                'local_search_intensity': 0.8
            }
        }
        cfg = benchmark_config.get(network_file, {
            'population_size': 50,
            'max_generations': 60,
            'local_search_intensity': 0.7
        })
        
        # Parse network
        filepath = os.path.join(self.data_dir, network_file)
        network = parse_inp_file(filepath)
        stats = network.get_network_stats()
        
        results = {
            'network': network_file,
            'difficulty': difficulty,
            'network_stats': stats,
            'memetic_ga_runs': [],
            'standard_ga_runs': []
        }
        
        # Run multiple instances
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            use_strict_tln = (network_file == 'TLN.inp')
            
            # Memetic GA
            print("Running Memetic GA (with local search)...")
            best_meme, best_hist_meme, avg_hist_meme, benchmark_hist_meme, time_meme = self.run_memetic_ga(
                network_file,
                filepath,
                network,
                population_size=cfg['population_size'],
                max_generations=cfg['max_generations'],
                local_search_intensity=cfg['local_search_intensity'],
                use_strict_paper_objective=use_strict_tln,
                enable_early_stopping=(network_file != 'BIN.inp'),
                seed=42 + run
            )
            
            eval_meme = self.evaluate_solution(network_file, filepath, best_meme, network, apply_paper_polish=True)
            results['memetic_ga_runs'].append({
                'run': run + 1,
                'best_fitness_history': best_hist_meme,
                'avg_fitness_history': avg_hist_meme,
                'universal_generations': benchmark_hist_meme['generations'],
                'universal_best_history': benchmark_hist_meme['best_history'],
                'universal_avg_history': benchmark_hist_meme['avg_history'],
                'runtime': time_meme,
                'evaluation': eval_meme
            })
            
            print(f"  Memetic GA Best Cost: {eval_meme['cost']:.2e}")
            
            # Standard GA
            print("Running Standard GA (no local search)...")
            best_std, best_hist_std, avg_hist_std, benchmark_hist_std, time_std = self.run_standard_ga(
                network_file,
                filepath,
                network,
                population_size=cfg['population_size'],
                max_generations=cfg['max_generations'],
                use_strict_paper_objective=use_strict_tln,
                enable_early_stopping=(network_file != 'BIN.inp'),
                seed=42 + run
            )
            
            eval_std = self.evaluate_solution(network_file, filepath, best_std, network, apply_paper_polish=False)
            results['standard_ga_runs'].append({
                'run': run + 1,
                'best_fitness_history': best_hist_std,
                'avg_fitness_history': avg_hist_std,
                'universal_generations': benchmark_hist_std['generations'],
                'universal_best_history': benchmark_hist_std['best_history'],
                'universal_avg_history': benchmark_hist_std['avg_history'],
                'runtime': time_std,
                'evaluation': eval_std
            })
            
            print(f"  Standard GA Best Cost: {eval_std['cost']:.2e}")
            
            # Calculate improvement
            improvement = (
                (eval_std['fitness'] - eval_meme['fitness']) / eval_std['fitness'] * 100
            ) if eval_std['fitness'] > 0 else 0
            print(f"  Memetic GA Improvement: {improvement:.2f}%")
        
        # Calculate aggregate statistics
        meme_costs = [r['evaluation']['cost'] for r in results['memetic_ga_runs']]
        std_costs = [r['evaluation']['cost'] for r in results['standard_ga_runs']]
        meme_universal_scores = [r['evaluation']['universal_score'] for r in results['memetic_ga_runs']]
        std_universal_scores = [r['evaluation']['universal_score'] for r in results['standard_ga_runs']]
        meme_paper_scores = [r['evaluation']['paper_score'] for r in results['memetic_ga_runs']]
        std_paper_scores = [r['evaluation']['paper_score'] for r in results['standard_ga_runs']]
        
        results['summary'] = {
            'memetic_ga': {
                'mean_cost': np.mean(meme_costs),
                'mean_universal_score': np.mean(meme_universal_scores),
                'mean_paper_score': np.mean(meme_paper_scores),
                'std_cost': np.std(meme_costs),
                'min_cost': min(meme_costs),
                'max_cost': max(meme_costs),
                'mean_runtime': np.mean([r['runtime'] for r in results['memetic_ga_runs']])
            },
            'standard_ga': {
                'mean_cost': np.mean(std_costs),
                'mean_universal_score': np.mean(std_universal_scores),
                'mean_paper_score': np.mean(std_paper_scores),
                'std_cost': np.std(std_costs),
                'min_cost': min(std_costs),
                'max_cost': max(std_costs),
                'mean_runtime': np.mean([r['runtime'] for r in results['standard_ga_runs']])
            }
        }
        
        avg_improvement = (
            (results['summary']['standard_ga']['mean_cost'] - 
             results['summary']['memetic_ga']['mean_cost']) /
            results['summary']['standard_ga']['mean_cost'] * 100
        )
        results['summary']['avg_improvement_percent'] = avg_improvement

        avg_universal_improvement = (
            (results['summary']['standard_ga']['mean_universal_score'] -
             results['summary']['memetic_ga']['mean_universal_score']) /
            results['summary']['standard_ga']['mean_universal_score'] * 100
        ) if results['summary']['standard_ga']['mean_universal_score'] > 0 else 0.0
        results['summary']['avg_universal_improvement_percent'] = avg_universal_improvement
        
        print(f"\n--- Summary ---")
        print(f"Memetic GA (avg): {results['summary']['memetic_ga']['mean_cost']:.2e}")
        print(f"Standard GA (avg): {results['summary']['standard_ga']['mean_cost']:.2e}")
        print(f"Average Improvement: {avg_improvement:.2f}%")
        print(f"Average Universal Improvement: {avg_universal_improvement:.2f}%")
        
        return results
    
    def run_all_benchmarks(self, num_runs: int = 3):
        """Run complete benchmark suite."""

        print("\nUsing strict benchmark set: TLN, Hanoi, Balerma (BIN)...")
        benchmarks = [
            ('TLN.inp', 'TLN'),
            ('hanoi.inp', 'Hanoi'),
            ('BIN.inp', 'Balerma')
        ]
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': []
        }
        
        for network_file, difficulty in benchmarks:
            results = self.run_benchmark(network_file, difficulty, num_runs)
            all_results['benchmarks'].append(results)
        
        # Save results
        results_file = os.path.join(self.results_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json.dump(
                all_results,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
            )
        
        print(f"\n✓ Results saved to {results_file}")
        
        return all_results


def main():
    """Main execution."""
    
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    results_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_004\results"
    
    runner = BenchmarkRunner(data_dir, results_dir)
    results = runner.run_all_benchmarks(num_runs=3)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for benchmark in results['benchmarks']:
        print(f"\n{benchmark['network']} ({benchmark['difficulty']}):")
        print(
            f"  Network: {benchmark['network_stats']['num_pipes']} pipes, "
            f"{benchmark['network_stats']['num_junctions']} junctions"
        )
        summary = benchmark['summary']
        print(
            f"  Memetic GA:  {summary['memetic_ga']['mean_cost']:.2e} +/- "
            f"{summary['memetic_ga']['std_cost']:.2e}"
        )
        print(
            f"  Standard GA: {summary['standard_ga']['mean_cost']:.2e} +/- "
            f"{summary['standard_ga']['std_cost']:.2e}"
        )
        print(f"  Improvement: {summary['avg_improvement_percent']:.2f}%")


if __name__ == "__main__":
    import sys
    
    # Allow passing number of runs as command line argument
    num_runs = 1  # Default: 1 run for faster development testing
    generate_plots = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '--plots':
                generate_plots = True
                continue
            try:
                num_runs = int(arg)
            except ValueError:
                pass
    
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    results_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_004\results"
    
    runner = BenchmarkRunner(data_dir, results_dir)
    results = runner.run_all_benchmarks(num_runs=num_runs)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for benchmark in results['benchmarks']:
        print(f"\n{benchmark['network']} ({benchmark['difficulty']}):")
        print(
            f"  Network: {benchmark['network_stats']['num_pipes']} pipes, "
            f"{benchmark['network_stats']['num_junctions']} junctions"
        )
        summary = benchmark['summary']
        print(
            f"  Memetic GA:  {summary['memetic_ga']['mean_cost']:.2e} +/- "
            f"{summary['memetic_ga']['std_cost']:.2e}"
        )
        print(
            f"  Standard GA: {summary['standard_ga']['mean_cost']:.2e} +/- "
            f"{summary['standard_ga']['std_cost']:.2e}"
        )
        print(f"  Improvement: {summary['avg_improvement_percent']:.2f}%")
    
    if generate_plots:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        results_file = os.path.join(results_dir, "benchmark_results.json")
        visualizer = ResultsVisualizer(results_file, results_dir)
        visualizer.plot_all()
        print("\n✓ All visualizations generated successfully!")
    else:
        print("\nSkipping visualization generation (use --plots to enable).")
