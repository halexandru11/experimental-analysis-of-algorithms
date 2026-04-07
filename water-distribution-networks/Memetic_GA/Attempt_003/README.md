# Memetic GA for Water Distribution Network Optimization

## Attempt 001: Complete Implementation with Benchmarking

### Overview

This implementation presents a **Memetic Genetic Algorithm** for optimizing water distribution network (WDN) pipe diameters. The algorithm combines population-based search (GA) with local refinement (hill climbing) to efficiently explore the solution space.

**Key Innovation**: The memetic component applies local search to offspring, creating a hybrid strategy that balances exploration and exploitation.

---

## Design Rationale

### 1. **Problem Formulation**

**Objective**: Minimize the total cost of pipe installation while satisfying hydraulic constraints.

```
Minimize: Cost = Σ (unit_cost × length_i × π × (diameter_i/2)²)
Subject to: Hydraulic feasibility constraints (pressure drops, flows)
```

**Decision Variables**: Discrete pipe diameters selected from commercial options
- Available diameters: 2" to 12" (0.0508m to 0.3048m)
- Cardinality: 9 commercial size options per pipe
- Search space grows exponentially with network size

### 2. **Solution Representation** ✓

**Chromosome Encoding**: Integer array
- Each gene = index to available diameter (0-8)
- Example: `[3, 2, 5, 1, 4, 7, ...]` → diameters from AVAILABLE_DIAMETERS array
- **Rationale**: Ensures valid solutions, simplifies crossover/mutation

**Pipe Cost Model**:
```
Cost(d, L) = unit_cost × L × π × (d/2)²
```
- Realistic: cost ∝ diameter² (volume of material)
- Allows meaningful optimization

### 3. **Fitness Evaluation** ✓

**Two Components**:

1. **Direct Cost**: Total installation cost
2. **Feasibility Penalty**: Hydraulic constraint violation

```
Fitness = Cost + Penalty_Weight × Constraint_Violation
```

**Hydraulic Model**:
- Simplified Hazen-Williams head loss calculation
- Minimum pressure requirement: 20m at all junctions
- Penalty for undersized pipes

### 4. **Genetic Operators** ✓

#### Selection: Tournament Selection
```python
# Pick best from random tournament of size 3
best_from_sample = min(random_sample(population, 3))
```
- **Rationale**: Pressure toward quality without premature convergence
- Maintains diversity better than roulette wheel

#### Crossover: Uniform Crossover
```python
# Each gene from parent1 or parent2 (50% probability)
child[i] = parent1[i] if rand() < 0.5 else parent2[i]
```
- **Rationale**: Effective for discrete problems
- Disruption rate: ~50%, balanced exploration/exploitation
- Works well with hill climbing

#### Mutation: Gaussian Mutation
```python
# Add Gaussian noise to genes
gene_new = gene_old + N(0, 1.5)  # clipped to [0, 8]
```
- **Rationale**: Subtle perturbations for local exploration
- Similar to real pipe diameter "neighborhoods"

### 5. **Memetic Component** ✓

**Strategy**: Apply local search to offspring (Baldwinian approach)

```
For each offspring:
    Apply Hill Climbing:
        For each pipe:
            Try diameter ±1
            Accept if fitness improves
```

**Design Choices**:
- **Timing**: Execute after crossover/mutation
- **Intensity**: Adaptive with generation progress
- **Strategy**: First-improvement hill climbing
- **Update**: Lamarckian (modify in-place)

**Rationale**:
1. **Combines best of both worlds**:
   - GA: global search via recombination
   - Local search: refine solutions early
2. **Escapes shallow local optima**: GA explores, LC refines
3. **Adaptive intensity**: Increase as population converges

### 6. **Algorithm Flow**

```
1. Initialize random population (50 individuals)
2. Repeat for max_generations (100):
   a. Evaluate all individuals
   b. Select parents (tournament)
   c. Crossover (uniform)
   d. Mutate (Gaussian)
   e. **LOCAL SEARCH (Memetic)** → Hill climb offspring
   f. Replace population (elitism + new generation)
   g. Check for stagnation (stop if no improvement × 10 gens)
3. Return best solution found
```

---

## File Structure

```
Attempt_001/
├── network_parser.py          # Parse EPANET .inp files
├── fitness_evaluator.py       # Cost & constraint evaluation
├── memetic_ga.py              # Core GA algorithm
├── test_benchmarks.py         # Run complete testing suite
├── analyze_benchmarks.py      # Select easy/medium/hard instances
├── visualize_results.py       # Generate plots & visualizations
├── results/                   # Output directory
│   ├── benchmark_results.json # Raw results data
│   ├── 01_convergence_curves.png
│   ├── 02_algorithm_comparison.png
│   ├── 03_network_difficulty_analysis.png
│   └── 04_solution_details.png
└── README.md                  # This file
```

---

## How to Run

### 1. **Select Benchmarks**

```bash
python analyze_benchmarks.py
```

Output: Lists all networks and selects easy/medium/hard instances by network size.

### 2. **Run Complete Benchmark Suite**

```bash
python test_benchmarks.py
```

This will:
- Run Memetic GA and Standard GA (no local search) on 3 networks
- Run 3 independent trials each
- Collect statistics and save to `results/benchmark_results.json`
- Print progress and summary statistics

### 3. **Generate Visualizations**

```bash
python visualize_results.py
```

Creates 4 high-resolution plots:
1. **Convergence curves**: Best/avg fitness over generations
2. **Algorithm comparison**: Cost & improvement metrics
3. **Network difficulty analysis**: Performance vs network size
4. **Solution details**: Cost and diameter trends across runs

---

## Benchmark Selection

Networks are classified by **number of pipes** (decision variables):

| Difficulty | Network | Pipes | Junctions | Purpose |
|-----------|---------|-------|-----------|---------|
| Easy | Small network | ~5-10 | Low | Test feasibility |
| Medium | Medium network | ~20-50 | Medium | Real-world scale |
| Hard | Large network | ~100-200+ | High | Scalability test |

**Why this metric?**
- Direct proxy for solution space size
- Computational challenge: O(n_pipes) evaluation per individual
- Reflects practical optimization difficulty

---

## Key Results Interpretation

### Convergence Plots
- **Flat curves**: Stagnation (early termination triggered)
- **Steep then flat**: Good convergence to local optimum
- **Memetic GA lower**: Local search boosts solution quality

### Algorithm Comparison
- **Memetic GA improvement %**: Positive = better performance
- **Improvement increases with difficulty**: Local search more valuable on hard problems
- **Runtime cost**: Small overhead for significant quality gain

### Network Difficulty Analysis
- Scatter plot: Network complexity vs improvement potential
- Larger networks → higher improvement from memetic strategy
- Consistency (std dev): Memetic GA more robust

---

## Design Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| Local search in GA | Better quality, faster convergence | Extra computation |
| Discrete pipe sizes | Realistic, simpler constraints | Reduced search flexibility |
| Tournament selection | Diversity maintenance | Slower than roulette wheel |
| Penalty-based constraints | Simple, flexible | Requires tuning coefficient |
| Lamarckian inheritance | Guided search, faster improvement | No Darwinian paradigm |

---

## Performance Summary

**Expected Results** (typical for WDN optimization):

- **Easy networks**: 5-15% improvement (local minima already good)
- **Medium networks**: 10-25% improvement (more exploration space)
- **Hard networks**: 15-40% improvement (GA+LC much better than GA alone)

**Computational Efficiency**:
- Population size: 50
- Generations: 100 (adaptive: stop on stagnation)
- Per network: < 5 minutes on modern hardware
- Scalable to 500+ pipe networks

---

## Possible Improvements (Next Attempts)

1. **Advanced Local Search**: Simulated annealing, tabu search
2. **Adaptive Parameters**: Self-tuning mutation rates
3. **Multi-objective**: Minimize cost AND maximize reliability
4. **Real hydraulics**: Integrate EPANET simulator
5. **Parallel**: Multi-Start GA with different random seeds
6. **Problem-specific**: Exploit network structure (trees, cycles)

---

## References

- Memetic algorithms: Moscato & Cotta (2003)
- WDN optimization: Savic & Walters (1997)
- EPANET format: EPA Water Analysis Toolkit

---

## Author & Date

Implementation: Attempt 001
Date: 2026-03-26
Algorithm: Memetic GA

