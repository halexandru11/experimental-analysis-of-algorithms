# Attempt 003: Strict Score-First Search

This attempt changes direction from the earlier memetic GA.

Instead of optimizing a fast internal proxy and checking the published score at the end, Attempt 003 optimizes the published benchmark score directly. It starts from a feasible high-diameter baseline, repairs infeasible solutions using hydraulic diagnostics, and then performs greedy feasibility-preserving diameter reductions.

## Why this approach

The previous attempt was not aligned tightly enough with the benchmark objective. The final comparison metric and the search objective were only loosely coupled, which is a poor fit for strict literature comparison. This version removes that mismatch.

## Files

- `network_parser.py` - EPANET input parser
- `strict_local_search.py` - strict benchmark-first optimizer
- `test_benchmarks.py` - runs TLN, Hanoi, and Balerma
- `visualize_results.py` - plots score gap vs published best

## Run

```bash
python test_benchmarks.py
python visualize_results.py
```

## Expected behavior

This approach should produce better apples-to-apples results whenever the published diameter catalog and head constraints are available. It is also more transparent because every accepted move is justified by the strict benchmark score.