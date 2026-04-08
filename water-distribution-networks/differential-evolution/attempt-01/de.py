"""
Differential Evolution (DE) for WDN pipe diameter optimization.

Adaptation note (discrete variables):
  - The decision variables are discrete diameter options.
  - DE is naturally defined over continuous vectors, so we:
      1) run mutation/crossover in "index space" as floats
      2) discretize trials via rounding to nearest integer option index
      3) clip to valid [0, K-1] range
  - Fitness is then evaluated on the discrete indices.

The fitness model is the same simplified surrogate used in the existing GA
baseline, which makes DE results comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from fitness_evaluator import FastFitnessEvaluator
from network_parser import WaterNetwork


MutationScheme = Literal["rand/1", "best/1"]


@dataclass(frozen=True)
class DEConfig:
    mutation: MutationScheme
    crossover: Literal["bin"]

    pop_size: int = 25
    max_generations: int = 60
    stagnation_patience: int = 12

    # Fixed parameters (used when adaptive=False)
    F: float = 0.8
    CR: float = 0.9

    # jDE-style adaptation (used when adaptive=True)
    adaptive: bool = False
    tau1: float = 0.1  # prob. to update F_i
    tau2: float = 0.1  # prob. to update CR_i
    F_l: float = 0.1
    F_u: float = 1.0

    seed: Optional[int] = None


class DifferentialEvolutionOptimizer:
    def __init__(self, network: WaterNetwork, config: DEConfig):
        self.network = network
        self.config = config
        self.evaluator = FastFitnessEvaluator(network)

        self.dim = self.evaluator.num_pipes
        self.num_options = self.evaluator.num_options

        if self.config.pop_size < 4 and self.config.mutation == "rand/1":
            raise ValueError("pop_size too small for rand/1 mutation")
        if self.config.pop_size < 3 and self.config.mutation == "best/1":
            raise ValueError("pop_size too small for best/1 mutation")

        self.rng = np.random.default_rng(self.config.seed)

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Round and clip to valid integer indices."""
        return np.clip(np.rint(x), 0, self.num_options - 1).astype(np.int64)

    def _init_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (population_float, fitness)."""
        pop = self.config.pop_size
        x_int = self.rng.integers(0, self.num_options, size=(pop, self.dim), endpoint=False, dtype=np.int64)
        x = x_int.astype(np.float64)
        fitness = self.evaluator.evaluate_indices(x_int)
        return x, fitness

    def _select_distinct(self, exclude: int, k: int) -> np.ndarray:
        """Sample k distinct indices from [0, pop_size) excluding `exclude`."""
        candidates = np.delete(np.arange(self.config.pop_size), exclude)
        return self.rng.choice(candidates, size=k, replace=False)

    def _make_trial(
        self,
        x: np.ndarray,
        x_best: np.ndarray,
        i: int,
        F_i: float,
        CR_i: float,
    ) -> np.ndarray:
        """
        Create a discretized trial vector for individual i.

        Returns integer indices with shape (dim,).
        """
        # Mutation
        if self.config.mutation == "rand/1":
            a, b, c = self._select_distinct(i, 3)
            v = x[a] + F_i * (x[b] - x[c])  # float trial (still "continuous")
        elif self.config.mutation == "best/1":
            # v = best + F*(r1 - r2)
            b, c = self._select_distinct(i, 2)
            v = x_best + F_i * (x[b] - x[c])
        else:
            raise ValueError(f"Unknown mutation: {self.config.mutation}")

        # Binomial crossover
        if self.config.crossover != "bin":
            raise ValueError("Only binomial crossover is supported in this attempt.")

        j_rand = int(self.rng.integers(0, self.dim))
        u = x[i].copy()
        mask = self.rng.random(self.dim) < CR_i
        mask[j_rand] = True
        u[mask] = v[mask]

        return self._discretize(u)

    def run(self) -> Tuple[Dict, List[float], List[float]]:
        """
        Returns
          - best_result dict
          - best_fitness_history (per generation)
          - avg_fitness_history (per generation)
        """
        x, fitness = self._init_population()

        initial_best_fitness = float(np.min(fitness))

        # Track histories
        best_fitness_history: List[float] = []
        avg_fitness_history: List[float] = []

        best_idx = int(np.argmin(fitness))
        best_fitness = float(fitness[best_idx])
        x_best = x[best_idx].copy()

        # Initialize adaptive parameters (jDE)
        if self.config.adaptive:
            pop = self.config.pop_size
            F_i = np.full(pop, self.config.F, dtype=np.float64)
            CR_i = np.full(pop, self.config.CR, dtype=np.float64)
        else:
            F_i = np.full(self.config.pop_size, self.config.F, dtype=np.float64)
            CR_i = np.full(self.config.pop_size, self.config.CR, dtype=np.float64)

        stagnation = 0
        prev_best = best_fitness

        for gen in range(self.config.max_generations):
            # Optional parameter adaptation for jDE
            if self.config.adaptive:
                for i in range(self.config.pop_size):
                    if self.rng.random() < self.config.tau1:
                        F_i[i] = self.config.F_l + self.rng.random() * (self.config.F_u - self.config.F_l)
                    if self.rng.random() < self.config.tau2:
                        CR_i[i] = self.rng.random()

            # Create all trials (one-by-one because selection depends on i)
            trial_indices = np.empty((self.config.pop_size, self.dim), dtype=np.int64)

            for i in range(self.config.pop_size):
                trial_indices[i] = self._make_trial(
                    x=x,
                    x_best=x_best,
                    i=i,
                    F_i=float(F_i[i]),
                    CR_i=float(CR_i[i]),
                )

            # Evaluate trials in batch
            trial_fitness = self.evaluator.evaluate_indices(trial_indices)

            # Selection: greedy replacement
            improved = trial_fitness < fitness
            if np.any(improved):
                x[improved] = trial_indices[improved].astype(np.float64)
                fitness[improved] = trial_fitness[improved]

            best_idx = int(np.argmin(fitness))
            best_fitness = float(fitness[best_idx])
            x_best = x[best_idx].copy()

            avg_fitness = float(np.mean(fitness))
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            # Early stopping
            if best_fitness < prev_best - 1e-12:
                prev_best = best_fitness
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= self.config.stagnation_patience:
                    break

        # Convert best to integer indices for reporting.
        best_indices = self._discretize(x_best)
        best_cost = float(self.evaluator.evaluate_indices(best_indices))

        best_result = {
            "best_indices": best_indices.tolist(),
            "best_cost": best_cost,
            "initial_best_fitness": initial_best_fitness,
        }
        return best_result, best_fitness_history, avg_fitness_history


def describe_config(config: DEConfig) -> str:
    if config.adaptive:
        adaptive_str = ", adaptive(F,CR)=jDE"
    else:
        adaptive_str = ""
    return f"{config.mutation}/bin: F={config.F:.2f}, CR={config.CR:.2f}{adaptive_str}".strip(", ")

