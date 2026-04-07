# Benchmark Protocol (Strict Comparison Track)

## Purpose
This attempt uses a strict apples-to-apples benchmark track for final-score comparison against published references.

## Selected Strict Benchmarks
- `TLN.inp` (Two-Loop Network)
- `hanoi.inp` (Hanoi Network)
- `BIN.inp` (Balerma Network)

These three are used because their benchmark formulations are sufficiently documented in literature and commonly replicated.

## Comparison Rules
A comparison is considered strict only when all of the following match the benchmark definition:
- Same network instance and hydraulic data
- Same decision variable model (single diameter choice per pipe)
- Same discrete diameter catalog
- Same cost model/table and units
- Same hydraulic constraint thresholds
- Same hydraulic model assumptions

If any of these differ, the result must be labeled exploratory/non-strict.

## Current Implementation Notes
- Optimization still uses an internal fast fitness for search.
- A separate external final comparison metric is logged for reporting.
- Plots include optional published-reference overlays from:
  - `results/published_reference_scores.json`

## Published Reference Values Used (Current)
- `TLN.inp`: 419000
- `hanoi.inp`: 6081000
- `BIN.inp`: 2306612.15

These values are treated as project references and should be re-verified in final report text with exact citations and assumptions.

## Report Framing Recommendation
Use two layers in write-up:
1. **Strict benchmark results**: TLN, Hanoi, Balerma (publishable comparison claims)
2. **Exploratory results**: any additional networks not fully matched to literature

This keeps claims rigorous while still showing broader experimentation.
