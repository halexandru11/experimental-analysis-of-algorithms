# Attempt_005 Live Results Summary

Source: `Memetic_GA/Attempt_005/live_results.csv`
Snapshot date: 2026-04-11
Notes:
- Only finite values of `best_paper_score` are included in aggregates.
- Published references are taken from `Attempt_004/results/published_reference_scores.json`.

## Best Vs Published Reference (SOTA)

| Network | Algorithm | Best score | Published best | Gap (%) |
|---|---|---:|---:|---:|
| TLN.inp | Memetic GA | 419000.0 | 419000.0 | 0.00 |
| TLN.inp | Standard GA | 420000.0 | 419000.0 | 0.24 |
| hanoi.inp | Memetic GA | 6281488.3 | 6081000.0 | 3.30 |
| hanoi.inp | Standard GA | 6430614.6 | 6081000.0 | 5.75 |
| BIN.inp | Memetic GA | 3933733.4 | 1956000.0 | 97.04 |
| BIN.inp | Standard GA | 4427058.7 | 1956000.0 | 122.41 |

## Aggregate Over Recorded Runs

| Network | Algorithm | Runs | Mean score | Min score | Max score |
|---|---|---:|---:|---:|---:|
| TLN.inp | Memetic GA | 20 | 424150.0 | 419000.0 | 448000.0 |
| TLN.inp | Standard GA | 1 | 420000.0 | 420000.0 | 420000.0 |
| hanoi.inp | Memetic GA | 17 | 6778049.4 | 6281488.3 | 7581995.4 |
| hanoi.inp | Standard GA | 1 | 6430614.6 | 6430614.6 | 6430614.6 |
| BIN.inp | Memetic GA | 14 | 4234735.2 | 3933733.4 | 4717918.7 |
| BIN.inp | Standard GA | 1 | 4427058.7 | 4427058.7 | 4427058.7 |
