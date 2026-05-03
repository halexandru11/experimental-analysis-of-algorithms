from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class DifferentialEvolutionConfig:
    population_size: int = 20
    generations: int = 150
    mutation_factor: float = 0.8
    crossover_rate: float = 0.9


@dataclass(slots=True)
class DifferentialEvolutionResult:
    best_vectors: list[np.ndarray]
    best_fitness: float
    history: list[dict[str, float]]


def run_differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    rng: np.random.Generator,
    config: DifferentialEvolutionConfig,
    checkpoint_interval: int,
    checkpoint_callback: Callable[[int, list[dict[str, float]], list[np.ndarray], float], None],
    progress_callback: Callable[[int, float], None] | None = None,
    resume_state: dict | None = None,
) -> DifferentialEvolutionResult:
    dimension = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]

    # Resume from checkpoint if provided
    if resume_state is not None:
        population = resume_state.get("population", None)
        fitness = resume_state.get("fitness", None)
        best_vector = resume_state.get("best_vector", None)
        best_fitness = resume_state.get("best_fitness", float("inf"))
        start_generation = resume_state.get("start_generation", 0)
        history: list[dict[str, float]] = resume_state.get("history", [])
        best_vectors: list[np.ndarray] = resume_state.get("best_vectors", [])
        
        # If population/fitness are missing (not saved to disk), reinitialize them
        # but preserve the generation offset and best solution found so far
        if population is None or fitness is None:
            population = rng.uniform(
                low=low, high=high, size=(config.population_size, dimension)
            )
            fitness = np.array(
                [objective(individual) for individual in population], dtype=float
            )
            # If we have a best_vector from checkpoint, seed it into the population
            if best_vector is not None:
                population[0] = best_vector.copy()
                fitness[0] = objective(best_vector)
        
        # Ensure best_vector is valid
        if best_vector is None:
            best_idx = int(np.argmin(fitness))
            best_vector = population[best_idx].copy()
            best_fitness = float(fitness[best_idx])
    else:
        population = rng.uniform(
            low=low, high=high, size=(config.population_size, dimension)
        )
        fitness = np.array(
            [objective(individual) for individual in population], dtype=float
        )

        best_idx = int(np.argmin(fitness))
        best_vector = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        start_generation = 0
        history: list[dict[str, float]] = []
        best_vectors: list[np.ndarray] = []

    for generation in range(start_generation, config.generations):
        for i in range(config.population_size):
            choices = [idx for idx in range(config.population_size) if idx != i]
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)

            mutant = population[r1] + config.mutation_factor * (
                population[r2] - population[r3]
            )
            mutant = np.clip(mutant, low, high)

            crossover_mask = rng.random(dimension) < config.crossover_rate
            forced_idx = int(rng.integers(0, dimension))
            crossover_mask[forced_idx] = True
            trial = np.where(crossover_mask, mutant, population[i])

            trial_fitness = objective(trial)
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = float(trial_fitness)
                    best_vector = trial.copy()

        history.append(
            {
                "generation": float(generation),
                "best_fitness": best_fitness,
                "mean_fitness": float(np.mean(fitness)),
                "std_fitness": float(np.std(fitness)),
            }
        )
        best_vectors.append(best_vector.copy())
        if progress_callback is not None:
            progress_callback(generation + 1, best_fitness)
        # Checkpointing: call checkpoint callback every checkpoint_interval generations
        if checkpoint_interval and (generation + 1) % checkpoint_interval == 0:
            # pass copies so the callback can safely serialize
            checkpoint_callback(generation + 1, list(history), [v.copy() for v in best_vectors], best_fitness)
            # clear in-memory history and best_vectors to free memory
            history.clear()
            best_vectors.clear()

    # Final checkpoint at end of run (write remaining entries)
    if checkpoint_interval:
        checkpoint_callback(config.generations, list(history), [v.copy() for v in best_vectors], best_fitness)
    return DifferentialEvolutionResult(
        best_vectors=best_vectors, best_fitness=best_fitness, history=history
    )
