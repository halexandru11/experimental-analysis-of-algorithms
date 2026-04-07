"""
Strict benchmark-first optimizer for Attempt 003.

The core idea is different from Attempt 002:
- optimize the published benchmark score directly
- start from feasible high-diameter solutions
- greedily reduce diameters only when feasibility is preserved
- use hydraulic diagnostics to prioritize repairs and local moves
"""

from __future__ import annotations

import os
import random
import tempfile
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import wntr

from network_parser import WaterNetwork


@dataclass
class EvaluationResult:
    score: float
    cost: float
    feasible: bool
    violation: float
    min_pressure: float
    eval_ok: bool
    note: str = ''
    diagnostics: Optional[Dict] = None


class StrictBenchmarkOptimizer:
    def __init__(
        self,
        network_file: str,
        inp_filepath: str,
        network: WaterNetwork,
        reference_scores: Dict,
        seed: int = 42,
        max_restarts: int = 3,
        max_passes: int = 8,
        max_candidate_pipes: int = 12,
    ):
        self.network_file = network_file
        self.inp_filepath = inp_filepath
        self.network = network
        self.reference_scores = reference_scores
        self.seed = seed
        self.max_restarts = max_restarts
        self.max_passes = max_passes
        self.max_candidate_pipes = max_candidate_pipes

        random.seed(seed)
        np.random.seed(seed)

        self.spec = self.reference_scores.get(network_file, {})
        self.diameter_options = sorted(float(row['diameter_m']) for row in self.spec.get('diameter_set', []))
        self.unit_cost_lookup = {
            float(row['diameter_m']): float(row['unit_cost_per_m'])
            for row in self.spec.get('diameter_set', [])
        }
        self.min_head = self._get_min_head_requirement(network_file)

    @staticmethod
    def _get_min_head_requirement(network_file: str) -> float:
        return {
            'TLN.inp': 30.0,
            'hanoi.inp': 30.0,
            'BIN.inp': 20.0,
        }.get(network_file, 20.0)

    @staticmethod
    def _load_wntr_model_robust(inp_filepath: str):
        try:
            return wntr.network.WaterNetworkModel(inp_filepath), None
        except UnicodeDecodeError:
            with open(inp_filepath, 'rb') as handle:
                raw = handle.read()
            text = raw.decode('latin-1')
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False, encoding='utf-8')
            tmp.write(text)
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()
            return wntr.network.WaterNetworkModel(tmp_path), tmp_path

    def _evaluate(self, diameters: List[float], return_diagnostics: bool = False) -> EvaluationResult:
        cost = 0.0
        try:
            for pipe, diameter in zip(self.network.pipes_list, diameters):
                cost += self.unit_cost_lookup[float(diameter)] * pipe.length
        except KeyError:
            return EvaluationResult(
                score=float('inf'),
                cost=float('inf'),
                feasible=False,
                violation=float('inf'),
                min_pressure=float('-inf'),
                eval_ok=False,
                note='diameter_not_in_catalog',
            )

        temp_inp = None
        try:
            wn, temp_inp = self._load_wntr_model_robust(self.inp_filepath)
            for pipe, diameter in zip(self.network.pipes_list, diameters):
                if pipe.id in wn.pipe_name_list:
                    wn.get_link(pipe.id).diameter = float(diameter)

            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            pressure_ts = results.node['pressure']
            final_pressure = pressure_ts.iloc[-1]
            junction_names = [junction_id for junction_id in self.network.junctions.keys() if junction_id in final_pressure.index]
            junction_pressures = final_pressure.loc[junction_names]

            shortfalls = np.maximum(0.0, self.min_head - junction_pressures.values)
            violation = float(np.sum(shortfalls))
            min_pressure = float(np.min(junction_pressures.values)) if len(junction_pressures.values) else float('-inf')
            feasible = violation <= 1e-9
            score = cost if feasible else float('inf')

            diagnostics = None
            if return_diagnostics:
                diagnostics = {
                    'junction_pressures': {node: float(value) for node, value in junction_pressures.items()},
                    'junction_deficits': {node: float(max(0.0, self.min_head - value)) for node, value in junction_pressures.items()},
                    'link_headloss_abs': {link: float(abs(value)) for link, value in results.link['headloss'].iloc[-1].items()},
                }

            return EvaluationResult(
                score=float(score),
                cost=float(cost),
                feasible=feasible,
                violation=violation,
                min_pressure=min_pressure,
                eval_ok=True,
                note='',
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return EvaluationResult(
                score=float('inf'),
                cost=float(cost),
                feasible=False,
                violation=float('inf'),
                min_pressure=float('-inf'),
                eval_ok=False,
                note=f'wntr_eval_failed: {exc}',
            )
        finally:
            if temp_inp and os.path.exists(temp_inp):
                try:
                    os.remove(temp_inp)
                except OSError:
                    pass

    def _pipe_priority(self, evaluation: EvaluationResult) -> List[int]:
        if evaluation.diagnostics is None:
            return list(range(self.network.get_pipe_count()))

        deficits = evaluation.diagnostics.get('junction_deficits', {})
        headloss_abs = evaluation.diagnostics.get('link_headloss_abs', {})

        priorities: List[Tuple[float, int]] = []
        for idx, pipe in enumerate(self.network.pipes_list):
            endpoint_deficit = 0.0
            if pipe.node1 in deficits:
                endpoint_deficit += deficits[pipe.node1]
            if pipe.node2 in deficits:
                endpoint_deficit += deficits[pipe.node2]

            headloss = 0.0
            if pipe.id in headloss_abs:
                headloss = headloss_abs[pipe.id]

            priorities.append((endpoint_deficit + 0.05 * headloss + 1e-6 * pipe.length, idx))

        priorities.sort(reverse=True)
        return [idx for _, idx in priorities]

    def _repair_to_feasible(self, indices: List[int]) -> Tuple[List[int], EvaluationResult]:
        current = indices.copy()
        current_eval = self._evaluate(self._indices_to_diameters(current), return_diagnostics=True)
        if current_eval.feasible:
            return current, current_eval

        for _ in range(max(1, self.network.get_pipe_count() * 3)):
            order = self._pipe_priority(current_eval)[:self.max_candidate_pipes]
            best_trial = None
            best_eval = None

            max_batch = min(6, len(order))
            for batch_size in range(1, max_batch + 1):
                trial = current.copy()
                for pipe_idx in order[:batch_size]:
                    if trial[pipe_idx] < len(self.diameter_options) - 1:
                        trial[pipe_idx] += 1
                trial_eval = self._evaluate(self._indices_to_diameters(trial), return_diagnostics=True)

                if best_eval is None or trial_eval.violation < best_eval.violation:
                    best_trial = trial
                    best_eval = trial_eval

                if trial_eval.feasible:
                    best_trial = trial
                    best_eval = trial_eval
                    break

            if best_trial is not None and best_eval is not None and best_eval.violation < current_eval.violation:
                current = best_trial
                current_eval = best_eval
            else:
                break

        return current, current_eval

    def _indices_to_diameters(self, indices: List[int]) -> List[float]:
        return [self.diameter_options[idx] for idx in indices]

    def _diameters_to_indices(self, diameters: List[float]) -> List[int]:
        indices: List[int] = []
        for diameter in diameters:
            distances = [abs(option - float(diameter)) for option in self.diameter_options]
            indices.append(int(np.argmin(distances)))
        return indices

    def _random_perturbation(self, indices: List[int], strength: float) -> List[int]:
        perturbed = indices.copy()
        pipe_count = len(perturbed)
        moves = max(1, int(pipe_count * strength))
        for pipe_idx in random.sample(range(pipe_count), min(pipe_count, moves)):
            if random.random() < 0.7 and perturbed[pipe_idx] > 0:
                perturbed[pipe_idx] -= 1
            elif perturbed[pipe_idx] < len(self.diameter_options) - 1:
                perturbed[pipe_idx] += 1
        return perturbed

    def _greedy_descent(self, indices: List[int]) -> Tuple[List[int], EvaluationResult]:
        current = indices.copy()
        current_eval = self._evaluate(self._indices_to_diameters(current), return_diagnostics=True)

        if not current_eval.feasible:
            current, current_eval = self._repair_to_feasible(current)

        if not current_eval.feasible:
            return current, current_eval

        if self.network.get_pipe_count() <= 12:
            max_subset_size = 3
            candidate_limit = 8
        elif self.network.get_pipe_count() <= 80:
            max_subset_size = 2
            candidate_limit = 8
        else:
            max_subset_size = 2
            candidate_limit = 6

        for _ in range(self.max_passes):
            improved = False
            order = self._pipe_priority(current_eval)[:candidate_limit]

            best_trial = None
            best_eval = None

            for subset_size in range(1, min(max_subset_size, len(order)) + 1):
                for subset in combinations(order, subset_size):
                    trial = current.copy()
                    valid = True
                    for pipe_idx in subset:
                        if trial[pipe_idx] <= 0:
                            valid = False
                            break
                        trial[pipe_idx] -= 1
                    if not valid:
                        continue

                    trial_eval = self._evaluate(self._indices_to_diameters(trial), return_diagnostics=True)
                    if trial_eval.feasible and trial_eval.score < current_eval.score:
                        if best_eval is None or trial_eval.score < best_eval.score:
                            best_trial = trial
                            best_eval = trial_eval

            if best_trial is not None and best_eval is not None:
                current = best_trial
                current_eval = best_eval
                improved = True
            if not improved:
                break

        return current, current_eval

    def optimize(self) -> Dict:
        pipe_count = self.network.get_pipe_count()
        max_seed = [len(self.diameter_options) - 1] * pipe_count
        file_seed = self._diameters_to_indices([pipe.diameter for pipe in self.network.pipes_list])
        min_seed = [0] * pipe_count

        seed_pool = [
            max_seed,
            self._random_perturbation(max_seed, 0.08),
            self._random_perturbation(max_seed, 0.12),
        ]

        best_indices, best_eval = self._repair_to_feasible(max_seed)
        if not best_eval.feasible:
            best_indices = max_seed
            best_eval = self._evaluate(self._indices_to_diameters(best_indices), return_diagnostics=True)

        for restart in range(self.max_restarts):
            if restart < len(seed_pool):
                seed_indices = seed_pool[restart]
            else:
                source = max_seed if restart % 2 == 0 else file_seed
                seed_indices = self._random_perturbation(source, 0.05 + 0.02 * restart)

            seed_indices, seed_eval = self._repair_to_feasible(seed_indices)
            if seed_eval.feasible and seed_eval.score < best_eval.score:
                best_indices = seed_indices
                best_eval = seed_eval

            candidate_indices, candidate_eval = self._greedy_descent(seed_indices)

            if candidate_eval.feasible and candidate_eval.score < best_eval.score:
                best_indices = candidate_indices
                best_eval = candidate_eval

        return {
            'network_file': self.network_file,
            'diameters': self._indices_to_diameters(best_indices),
            'indices': best_indices,
            'score': best_eval.score,
            'cost': best_eval.cost,
            'feasible': best_eval.feasible,
            'violation': best_eval.violation,
            'min_pressure': best_eval.min_pressure,
            'eval_ok': best_eval.eval_ok,
            'note': best_eval.note,
        }