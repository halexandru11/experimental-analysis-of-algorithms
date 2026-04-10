"""
Memetic Genetic Algorithm for water distribution network optimization.

Design Rationale:
================

1. REPRESENTATION (Integer Encoding)
   - Each gene represents a pipe diameter as an index to AVAILABLE_DIAMETERS
   - Chromosome length = number of pipes in network
   - Discrete representation matches real commercial pipe options
   
2. GENETIC OPERATORS
   - Selection: Tournament selection (pressure toward quality solutions)
   - Crossover: Uniform crossover (50% probability per gene)
   - Mutation: Gaussian mutation with adaptive rates
   
3. LOCAL SEARCH (Memetic Component)
   - Hill climbing applied to all offspring
   - Move to larger diameter if cost decrease is offset by constraint satisfaction
   - Prevents getting stuck in local optima too early
   
4. MEME ACTIVATION STRATEGY
   - Apply local search after crossover/mutation (Baldwin effect)
   - Intensity: adaptive based on generation progress
   - Lamarckian update: Update individual if improvement found
   
5. TERMINATION
   - Generation-based (max_generations)
   - Stagnation detection (no improvement for N generations)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import random
from copy import deepcopy
from fitness_evaluator import FitnessEvaluator, AVAILABLE_DIAMETERS
from network_parser import WaterNetwork


BenchmarkScoreFn = Callable[[List[float]], Dict[str, float]]
FitnessScoreFn = Callable[[List[float]], float]


class Individual:
    """Represents an individual solution (pipe diameter configuration)."""
    
    def __init__(
        self,
        chromosome: List[int],
        fitness_evaluator: FitnessEvaluator,
        fitness_score_fn: Optional[FitnessScoreFn] = None
    ):
        """
        Args:
            chromosome: List of diameter indices for each pipe
            fitness_evaluator: FitnessEvaluator instance
            fitness_score_fn: Optional external objective function
        """
        self.chromosome = chromosome.copy()
        self.evaluator = fitness_evaluator
        self.fitness_score_fn = fitness_score_fn
        self._fitness = None
    
    @property
    def fitness(self) -> float:
        """Lazy evaluation of fitness."""
        if self._fitness is None:
            diameters = self.evaluator.indices_to_diameters(self.chromosome)
            if self.fitness_score_fn is None:
                self._fitness = self.evaluator.evaluate(diameters)
            else:
                self._fitness = float(self.fitness_score_fn(diameters))
        return self._fitness
    
    def invalidate_fitness(self):
        """Invalidate cached fitness (call after mutation)."""
        self._fitness = None
    
    def copy(self) -> 'Individual':
        """Return a deep copy of this individual."""
        return Individual(self.chromosome, self.evaluator, self.fitness_score_fn)


class MemeticGA:
    """
    Memetic Genetic Algorithm for network pipe optimization.
    
    Combines:
    - Genetic Algorithm: Population-based search, crossover, mutation
    - Local Search: Hill climbing to refine solutions
    - Lamarckian inheritance: Pass improved genes to offspring
    """
    
    def __init__(
        self,
        network: WaterNetwork,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        local_search_intensity: float = 1.0,
        diameter_options: Optional[List[float]] = None,
        unit_cost_lookup: Optional[Dict[float, float]] = None,
        fitness_score_fn: Optional[FitnessScoreFn] = None,
        benchmark_score_fn: Optional[BenchmarkScoreFn] = None,
        benchmark_eval_interval: int = 5,
        enable_early_stopping: bool = True,
        feasibility_checkpoint_interval: Optional[int] = None,
        feasibility_check_fn: Optional[Callable[[List[float]], bool]] = None,
        repair_fn: Optional[Callable[[List[float]], List[float]]] = None,
        seed: int = None
    ):
        """
        Initialize Memetic GA.
        
        Args:
            network: WaterNetwork instance
            population_size: Size of population
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Initial mutation rate (adaptive)
            local_search_intensity: Intensity of local search (0.0 to 1.0)
            diameter_options: Optional diameter catalog for this benchmark
            unit_cost_lookup: Optional benchmark unit-cost table
            fitness_score_fn: Optional external objective for optimization
            benchmark_score_fn: Optional external benchmark metric function
            benchmark_eval_interval: Periodicity for benchmark metric tracking
            enable_early_stopping: Whether to stop on stagnation
            feasibility_checkpoint_interval: Interval (generations) to check feasibility (None=disabled)
            feasibility_check_fn: Function to check if solution is feasible (takes diameters list)
            repair_fn: Function to repair infeasible solution (takes diameters, returns repaired diameters)
            seed: Random seed for reproducibility
        """
        self.network = network
        self.fitness_evaluator = FitnessEvaluator(
            network,
            diameter_options=diameter_options,
            unit_cost_lookup=unit_cost_lookup
        )
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_intensity = local_search_intensity
        self.num_pipes = network.get_pipe_count()
        self.num_diameter_options = self.fitness_evaluator.diameter_options
        self.fitness_score_fn = fitness_score_fn
        self.benchmark_score_fn = benchmark_score_fn
        self.benchmark_eval_interval = max(1, benchmark_eval_interval)
        self.enable_early_stopping = enable_early_stopping
        
        # Feasibility checkpoint mechanism
        self.feasibility_checkpoint_interval = feasibility_checkpoint_interval
        self.feasibility_check_fn = feasibility_check_fn
        self.repair_fn = repair_fn
        self.best_feasible_individual: Optional[Individual] = None
        self.best_feasible_generation: int = -1
        self.feasibility_checkpoint_history: List[Tuple[int, float]] = []  # (generation, best_cost)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.population: List[Individual] = []
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.benchmark_best_history: List[float] = []
        self.benchmark_avg_history: List[float] = []
        self.benchmark_eval_generations: List[int] = []
        self.benchmark_metric_name = 'universal_score'
        self.generation = 0

    def _check_feasibility_at_checkpoint(self):
        """
        Periodic feasibility checkpoint: check best individual for feasibility,
        attempt repair if needed, and track best feasible solution found.
        
        Called every feasibility_checkpoint_interval generations during optimization.
        """
        if not self.feasibility_check_fn or not self.repair_fn or not self.population:
            return
        
        # Get best individual by training fitness
        best_idx = np.argmin([ind.fitness for ind in self.population])
        candidate = self.population[best_idx]
        
        # Convert to diameters for feasibility check
        diameters = self.fitness_evaluator.indices_to_diameters(candidate.chromosome)
        
        # Check if already feasible
        if self.feasibility_check_fn(diameters):
            # Feasible - check if it's the best feasible yet
            cost = float(self.fitness_evaluator.calculate_total_cost(diameters))
            if self.best_feasible_individual is None:
                self.best_feasible_individual = candidate.copy()
                self.best_feasible_generation = self.generation
                self.feasibility_checkpoint_history.append((self.generation, cost))
                print(f"  [Checkpoint Gen {self.generation}] Found feasible solution: cost={cost:.2e}")
            else:
                best_cost = float(self.fitness_evaluator.calculate_total_cost(
                    self.fitness_evaluator.indices_to_diameters(self.best_feasible_individual.chromosome)
                ))
                if cost < best_cost:
                    self.best_feasible_individual = candidate.copy()
                    self.best_feasible_generation = self.generation
                    self.feasibility_checkpoint_history.append((self.generation, cost))
                    print(f"  [Checkpoint Gen {self.generation}] Improved feasible solution: cost={cost:.2e}")
        else:
            # Infeasible - attempt repair
            repaired_diameters = self.repair_fn(diameters)
            repaired_cost = float(self.fitness_evaluator.calculate_total_cost(repaired_diameters))
            
            # Check if repair succeeded
            if self.feasibility_check_fn(repaired_diameters):
                # Repair succeeded
                repaired_chromosome = [
                    self.fitness_evaluator.diameter_to_index(d)
                    for d in repaired_diameters
                ]
                repaired_individual = Individual(repaired_chromosome, self.fitness_evaluator, self.fitness_score_fn)
                
                if self.best_feasible_individual is None:
                    self.best_feasible_individual = repaired_individual
                    self.best_feasible_generation = self.generation
                    self.feasibility_checkpoint_history.append((self.generation, repaired_cost))
                    print(f"  [Checkpoint Gen {self.generation}] Repaired to feasible: cost={repaired_cost:.2e}")
                else:
                    best_cost = float(self.fitness_evaluator.calculate_total_cost(
                        self.fitness_evaluator.indices_to_diameters(self.best_feasible_individual.chromosome)
                    ))
                    if repaired_cost < best_cost:
                        self.best_feasible_individual = repaired_individual
                        self.best_feasible_generation = self.generation
                        self.feasibility_checkpoint_history.append((self.generation, repaired_cost))
                        print(f"  [Checkpoint Gen {self.generation}] Improved via repair: cost={repaired_cost:.2e}")
                
                # NOTE: We do NOT re-inject repaired solutions back into population.
                # We track them as best_feasible to preserve through elitism,
                # but allow SOTA selection to decide their fitness naturally.
                # This keeps the algorithm pure to the SOTA evaluation approach.
            else:
                print(f"  [Checkpoint Gen {self.generation}] Best individual infeasible; repair also failed")

    def _evaluate_benchmark_individual(self, individual: Individual) -> float:
        """Evaluate external benchmark score for one individual."""
        diameters = self.fitness_evaluator.indices_to_diameters(individual.chromosome)
        if self.benchmark_score_fn is None:
            benchmark = self.fitness_evaluator.evaluate_universal_score(diameters)
        else:
            benchmark = self.benchmark_score_fn(diameters)
        return float(benchmark.get('score', float('inf')))

    def _track_benchmark_scores(self, force: bool = False):
        """Track periodic benchmark score history for comparison plots."""
        if not self.population:
            return

        if not force and (self.generation % self.benchmark_eval_interval != 0):
            return

        benchmark_scores = [self._evaluate_benchmark_individual(ind) for ind in self.population]
        best_now = float(min(benchmark_scores))
        if self.benchmark_best_history:
            best_now = min(best_now, self.benchmark_best_history[-1])
        self.benchmark_best_history.append(best_now)
        self.benchmark_avg_history.append(float(np.mean(benchmark_scores)))
        self.benchmark_eval_generations.append(self.generation)
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            # Random diameter indices
            chromosome = [
                random.randint(0, self.num_diameter_options - 1)
                for _ in range(self.num_pipes)
            ]
            individual = Individual(chromosome, self.fitness_evaluator, self.fitness_score_fn)
            self.population.append(individual)
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """
        Tournament selection: pick best from random subset.
        
        Args:
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover: each gene from parent1 or parent2 with 50% probability.
        
        Args:
            parent1, parent2: Parent individuals
            
        Returns:
            Tuple of offspring individuals
        """
        child1_chr = []
        child2_chr = []
        
        for i in range(self.num_pipes):
            if random.random() < 0.5:
                child1_chr.append(parent1.chromosome[i])
                child2_chr.append(parent2.chromosome[i])
            else:
                child1_chr.append(parent2.chromosome[i])
                child2_chr.append(parent1.chromosome[i])
        
        return (
            Individual(child1_chr, self.fitness_evaluator, self.fitness_score_fn),
            Individual(child2_chr, self.fitness_evaluator, self.fitness_score_fn)
        )
    
    def _gaussian_mutation(self, individual: Individual, mutation_rate: float = None):
        """
        Gaussian mutation: add random noise to genes.
        
        Mutates individual in-place. For each gene:
        - With probability mutation_rate, add Gaussian noise
        - Noise drawn from N(0, 1.5) (limited to valid range)
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability per gene (uses self.mutation_rate if None)
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        for i in range(self.num_pipes):
            if random.random() < mutation_rate:
                # Add Gaussian noise
                noise = int(np.random.normal(0, 1.5))
                new_value = individual.chromosome[i] + noise
                
                # Clip to valid range
                individual.chromosome[i] = max(
                    0,
                    min(new_value, self.num_diameter_options - 1)
                )
        
        individual.invalidate_fitness()
    
    def _local_search_hillclimb(self, individual: Individual, num_iterations: int = 3):
        """
        FAST local search: Hill climbing with minimal iterations.
        
        Optimized for speed - randomly samples genes to improve (fair coverage).
        
        Args:
            individual: Individual to improve (modified in-place)
            num_iterations: Max iterations (reduced for speed)
        """
        iterations = max(1, int(num_iterations))
        max_attempts = min(self.num_pipes, 8 if self.num_pipes > 100 else 20)

        for _ in range(iterations):
            pipe_indices = random.sample(range(self.num_pipes), max_attempts)

            for pipe_idx in pipe_indices:
                base_gene = individual.chromosome[pipe_idx]
                current_fitness = individual.fitness
                best_gene = base_gene
                best_fitness = current_fitness

                # Keep large-network local search conservative unless we have a strict
                # external objective, in which case downsize moves are needed to refine cost.
                deltas = (-1, 1) if (self.num_pipes <= 100 or self.fitness_score_fn is not None) else (1,)

                # Explore one-step neighbors.
                for delta in deltas:
                    cand_gene = base_gene + delta
                    if cand_gene < 0 or cand_gene >= self.num_diameter_options:
                        continue

                    individual.chromosome[pipe_idx] = cand_gene
                    individual.invalidate_fitness()
                    cand_fitness = individual.fitness

                    if cand_fitness < best_fitness:
                        best_fitness = cand_fitness
                        best_gene = cand_gene

                if best_gene != base_gene:
                    individual.chromosome[pipe_idx] = best_gene
                    individual.invalidate_fitness()
                    # Warm the cache for subsequent neighborhood checks.
                    _ = individual.fitness
                else:
                    individual.chromosome[pipe_idx] = base_gene
                    individual.invalidate_fitness()
    
    def evolve_one_generation(self):
        """Execute one generation of evolution."""
        # Evaluate population
        fitnesses = [ind.fitness for ind in self.population]
        
        # Track statistics
        best_fitness = min(fitnesses)
        avg_fitness = np.mean(fitnesses)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self._track_benchmark_scores()
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        # Priority: prefer best_feasible_individual if it exists (never lose feasibility once found)
        best_idx = np.argmin(fitnesses)
        best_individual = self.population[best_idx].copy()
        
        if self.best_feasible_individual is not None:
            # We have a known feasible solution; prefer it over potentially infeasible candidates
            best_individual = self.best_feasible_individual.copy()
        
        new_population.append(best_individual)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._uniform_crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation
            self._gaussian_mutation(child1)
            self._gaussian_mutation(child2)
            
            # **MEMETIC COMPONENT**: Apply local search (Lamarckian)
            # Intensity increases with generation, but capped for larger networks
            local_search_prob = min(
                self.local_search_intensity,
                0.2 + 0.3 * (self.generation / max(1, self.max_generations))
            )

            if self.num_pipes > 100:
                # Large strict benchmarks are sensitive to premature local convergence,
                # but they still benefit from occasional refinement of both offspring.
                large_net_prob = min(0.22, 0.75 * local_search_prob)
                if random.random() < large_net_prob:
                    self._local_search_hillclimb(child1, 2)
                    if len(new_population) < self.population_size - 1:
                        self._local_search_hillclimb(child2, 1)
            elif random.random() < local_search_prob:
                ls_iterations = max(1, int(3 * self.local_search_intensity))
                self._local_search_hillclimb(child1, ls_iterations)
                if len(new_population) < self.population_size - 1:
                    self._local_search_hillclimb(child2, ls_iterations)
            
            # Add offspring to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Trim to population size (elitism already added)
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def run(self) -> Tuple[Individual, List[float], List[float], Dict[str, List[float]]]:
        """
        Run the Memetic GA algorithm.
        
        Returns:
            Tuple of (best_individual, best_fitness_history, avg_fitness_history,
            benchmark_history)
        """
        print("Initializing population...")
        self.initialize_population()
        
        print(f"Network: {self.network.get_network_stats()}")
        print(f"Population: {self.population_size}, Max Generations: {self.max_generations}")
        print(f"Initial population fitness: mean={np.mean([ind.fitness for ind in self.population]):.2e}")
        
        stagnation_count = 0
        if self.num_pipes <= 50:
            stagnation_threshold = 10
            min_generations_before_stop = 8
        elif self.num_pipes <= 200:
            stagnation_threshold = 14
            min_generations_before_stop = 15
        else:
            stagnation_threshold = 18
            min_generations_before_stop = 25
        
        print("\nRunning Memetic GA...")
        for gen in range(self.max_generations):
            self.evolve_one_generation()
            
            best_fit = self.best_fitness_history[-1]
            avg_fit = self.avg_fitness_history[-1]
            
            if gen > 0 and self.best_fitness_history[-1] >= (self.best_fitness_history[-2] - 1e-12):
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # Periodic feasibility checkpoint
            if self.feasibility_checkpoint_interval and self.generation % self.feasibility_checkpoint_interval == 0:
                self._check_feasibility_at_checkpoint()
            
            if gen % 10 == 0 or gen == self.max_generations - 1:
                print(f"Gen {gen:3d}: Best={best_fit:.2e}, Avg={avg_fit:.2e}, Stagnation={stagnation_count}")
            
            # Early stopping on stagnation
            if (
                self.enable_early_stopping and
                gen >= min_generations_before_stop and
                stagnation_count >= stagnation_threshold
            ):
                print(f"Stagnation detected at generation {gen}. Stopping.")
                break
        
        # Get best individual
        best_idx = np.argmin([ind.fitness for ind in self.population])
        best_individual = self.population[best_idx]
        self._track_benchmark_scores(force=True)

        benchmark_history = {
            'metric_name': self.benchmark_metric_name,
            'generations': self.benchmark_eval_generations,
            'best_history': self.benchmark_best_history,
            'avg_history': self.benchmark_avg_history,
            'feasibility_checkpoint_history': self.feasibility_checkpoint_history,
            'best_feasible_individual': self.best_feasible_individual,
            'best_feasible_generation': self.best_feasible_generation
        }
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {best_individual.fitness:.2e}")
        if self.best_feasible_individual is not None:
            best_feasible_cost = float(self.fitness_evaluator.calculate_total_cost(
                self.fitness_evaluator.indices_to_diameters(self.best_feasible_individual.chromosome)
            ))
            print(f"Best feasible solution found at Gen {self.best_feasible_generation}: cost={best_feasible_cost:.2e}")
        
        return best_individual, self.best_fitness_history, self.avg_fitness_history, benchmark_history
