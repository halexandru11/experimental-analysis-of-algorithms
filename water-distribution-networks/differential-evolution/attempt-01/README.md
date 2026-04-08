# Differential Evolution Attempt 01

This attempt implements **Differential Evolution (DE)** for a *simplified* water distribution network (WDN) pipe-diameter optimization model using the same representation and fitness surrogate as:
`Memetic_GA/Attempt_001`.

## Design choices (why this structure)

### Representation
- Decision variables are **discrete**: each pipe chooses one of `AVAILABLE_DIAMETERS` (9 options).
- For DE’s continuous mutation, we operate in **index space as floats**, then **round + clip** to integer option indices before evaluating fitness.

### Data structures
- `FastFitnessEvaluator` precomputes a `cost_matrix[pipe_i, diameter_option_k]` so fitness evaluation is:
  - `O(num_pipes)` with vectorized numpy indexing
  - avoids repeated math inside the DE loop.

### DE scheme
- Tested configurations:
  1. `DE/rand/1/bin` with fixed `(F, CR)`
  2. `DE/best/1/bin` with **jDE-style adaptation** of `F_i` and `CR_i`

### Visualization and reporting
- `run_benchmarks.py` runs DE on 3 auto-selected instances (easy/medium/hard) and saves raw results JSON.
- `visualize_results.py` generates plots in `attempt-01/results/`:
  - `01_de_convergence_curves.png`
  - `02_de_final_improvement.png`
  - `03_de_cost_summary.png`

## How to run

From this folder:

```bash
python run_benchmarks.py
python visualize_results.py
```

