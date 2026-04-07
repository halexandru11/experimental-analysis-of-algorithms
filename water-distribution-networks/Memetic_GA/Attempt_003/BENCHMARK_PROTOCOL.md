# Benchmark Protocol - Attempt 003

Attempt 003 uses a strict score-first comparison strategy.

## Included benchmarks

- `TLN.inp`
- `hanoi.inp`
- `BIN.inp`

## Comparison rule

A solution is considered comparable only if it uses the published diameter catalog, published unit costs, and the benchmark pressure threshold for that network.

## Search rule

The optimizer is not allowed to accept a move that makes the solution infeasible under the benchmark hydraulic model.

## Reporting rule

The final report must show both:

1. the Attempt 003 score
2. the published best score

Any difference should be reported as a gap percentage.