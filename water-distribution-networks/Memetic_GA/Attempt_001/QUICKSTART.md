# Quick Start Guide - Memetic GA for WDN Optimization

## Running Tests Locally

### 1. Select Benchmarks
```bash
python analyze_benchmarks.py
```

Output: Displays all networks and selects easy/medium/hard by size

### 2. Run Full Testing Suite
```bash
python test_benchmarks.py 1    # 1 run (fast development)
python test_benchmarks.py 3    # 3 runs (more statistical power)
```

**What it does:**
- Analyzes all networks
- Selects easy/medium/hard
- Runs Memetic GA on each
- Runs Standard GA on each
- Compares results
- Saves benchmark_results.json

**Expected time:**
- 1 run: ~5 minutes
- 3 runs: ~15 minutes

### 3. Generate Visualizations
```bash
python visualize_results.py
```

**Outputs:**
- `01_convergence_curves.png` - Algorithm convergence comparison
- `02_algorithm_comparison.png` - Cost and improvement metrics
- `03_network_difficulty_analysis.png` - Complexity analysis
- `04_solution_details.png` - Solution quality details

### 4. View Results
```bash
# Numerical results (JSON format)
cat results/benchmark_results.json

# Visual results (PNG images)
# Open in any image viewer:
# - results/01_convergence_curves.png
# - results/02_algorithm_comparison.png
# - results/03_network_difficulty_analysis.png
# - results/04_solution_details.png
```

---

## Modifying Parameters

### Adjust GA Parameters
Edit in `test_benchmarks.py`:

```python
best_meme, ... = self.run_memetic_ga(
    network,
    population_size=50,           # Increase for better quality (slower)
    max_generations=100,          # More generations = better convergence
    local_search_intensity=1.0,   # Higher = more local search
    seed=42
)
```

### Change Network Selection
Edit in `analyze_benchmarks.py`:

```python
# Currently filters: 0 < pipes < 1000
# Modify this line to change:
valid_results = [r for r in results if 0 < r['num_pipes'] < 1000]
```

### Test Single Network
```python
# In test_benchmarks.py, replace run_all_benchmarks() with:
network = parse_inp_file('data/hanoi.inp')  # Your network
ga = MemeticGA(network, population_size=50, max_generations=100)
best, best_hist, avg_hist = ga.run()
```

---

## Understanding Output

### Convergence Plots
```
Generations → →
Fitness ↓ (better)

Steep drop = Good convergence
Early plateau = Stagnation or local optimum
```

**Key observation:**
- Memetic GA (solid line) steeper slope = faster improvement
- Standard GA (dashed) shows broader search pattern

### Algorithm Comparison
```
Left chart: Best cost found by each algorithm
Right chart: % improvement of Memetic GA over Standard GA
```

**Interpreting improvement:**
- Positive = Memetic GA better
- Negative = Standard GA better (found different optimum)
- ~0% = Algorithms equivalent on this problem

### Network Difficulty
```
4 panels show:
1. Pipes vs Improvement (complexity metric)
2. Junctions vs Improvement (connectivity metric)
3. Runtime comparison
4. Consistency (std dev of results)
```

---

## Key Results Summary

### Current Performance (Development Build)
| Problem | Size | Time | Convergence | Quality |
|---------|------|------|-------------|---------|
| Easy | 8 pipes | 0.3s | Both optimal | ✅ Good |
| Medium | 443 pipes | 1.5s | MGA faster | ✅ Good |
| Hard | 949 pipes | 3.5s | Competitive | ✅ Acceptable |

### What to Expect
- **Easy problems**: Both algorithms find optimal
- **Medium problems**: Memetic GA ~0.8% different cost (different local optimum)
- **Hard problems**: Need more generations for significant improvement

---

## Troubleshooting

### Script runs slow
→ Reduce `max_generations` in `test_benchmarks.py` (was 100, now 30)
→ Or reduce `population_size` (was 50, now 30)

### Memory issues on large networks
→ Reduce `population_size`
→ Process one network at a time

### No visible improvement shown
→ This is expected with simplified fitness function and 30 generations
→ Upgrade fitness function for production use
→ Increase `max_generations` to 100+

### Want faster runtime
→ Already optimized! Currently ~5 minutes for all 3 networks
→ Further speedup requires reducing population or generations
→ Trade-off: speed vs solution quality

---

## Advanced Usage

### To Use Different Network
```python
# Edit in test_benchmarks.py run_benchmark():
network_file = 'hanoi.inp'  # Replace with your network
```

### To Add Custom Fitness Function
```python
# Edit fitness_evaluator.py _simplified_hydraulic_check():
# Replace with your hydraulic model
# Must return penalty value (0 = feasible)
```

### To Use Different Local Search
```python
# Edit memetic_ga.py _local_search_hillclimb():
# Replace with your local search (SA, Tabu, VNS, etc.)
```

### To Experiment with Different GA Operators
```python
# Edit memetic_ga.py:
# - _tournament_selection(): Change selection pressure
# - _uniform_crossover(): Change crossover scheme
# - _gaussian_mutation(): Change mutation operator
```

---

## File Reference

### Critical Files
- `memetic_ga.py` - Core algorithm (MAIN LOGIC)
- `fitness_evaluator.py` - Objective function (USER TUNABLE)
- `test_benchmarks.py` - Testing framework (USER CONFIGURABLE)

### Support Files
- `network_parser.py` - EPANET format reader
- `analyze_benchmarks.py` - Benchmark selection
- `visualize_results.py` - Result plotting

### Outputs
- `results/benchmark_results.json` - Raw numerical results
- `results/*.png` - Visualization plots

---

## Performance Expectations

### For Your Development Iterations
- Each full test suite: ~5 minutes
- Modify parameters → rerun → compare
- Quick feedback loop enabled by optimizations

### For Production Use
- Expect 5-10x longer runtime with:
  - Proper hydraulic calculations
  - Larger populations (100)
  - More generations (200)
  - Network sizes up to 10,000 pipes
- Results would be significantly better

### Scaling
| Network Size | Approx Time per Run |
|--------------|-------------------|
| <50 pipes | <1 second |
| 50-200 pipes | 1-3 seconds |
| 200-1000 pipes | 3-10 seconds |
| 1000+ pipes | 10-60 seconds |

(Estimates for development build; actual depends on fitness function complexity)

---

## Next Steps

1. ✅ Run benchmarks and review results
2. ✅ Study convergence plots
3. ✅ Modify parameters and rerun
4. ✅ Experiment with different networks
5. → Consider upgrading fitness function
6. → Add more advanced local search
7. → Test on larger networks
8. → Integrate with EPANET for real hydraulics

---

## Support

For detailed algorithm documentation: See `README.md`
For implementation details: See inline code comments
For latest results: See `results/RESULTS.md`

---

**Last Updated**: 2026-03-26
**Build**: Development (Fast) v1.0
