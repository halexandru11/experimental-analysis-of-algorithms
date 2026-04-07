"""
Run Attempt 003 on the strict benchmark set.

This runner evaluates the published-reference benchmarks directly and writes a
compact results file for visualization and comparison.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from network_parser import parse_inp_file
from strict_local_search import StrictBenchmarkOptimizer


STRICT_BENCHMARKS = ['TLN.inp', 'hanoi.inp', 'BIN.inp']


class BenchmarkRunner:
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.reference_scores = self._load_reference_scores()

    def _load_reference_scores(self) -> Dict:
        ref_path = os.path.join(self.results_dir, 'published_reference_scores.json')
        if not os.path.exists(ref_path):
            return {}
        with open(ref_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def run_benchmark(self, network_file: str, num_restarts: int = 8, seed: int = 42) -> Dict:
        inp_path = os.path.join(self.data_dir, network_file)
        network = parse_inp_file(inp_path)
        optimizer = StrictBenchmarkOptimizer(
            network_file=network_file,
            inp_filepath=inp_path,
            network=network,
            reference_scores=self.reference_scores,
            seed=seed,
            max_restarts=num_restarts,
            max_passes=8,
        )

        result = optimizer.optimize()
        reference_score = float(self.reference_scores.get(network_file, {}).get('published_best_universal_score', float('nan')))
        gap_percent = float('nan')
        if reference_score and reference_score == reference_score:
            gap_percent = ((result['score'] - reference_score) / reference_score) * 100.0

        return {
            'network_file': network_file,
            'benchmark_name': self.reference_scores.get(network_file, {}).get('benchmark_name', network_file),
            'published_best_universal_score': reference_score,
            'result': result,
            'gap_percent': gap_percent,
            'strict': True,
        }

    def run_all_benchmarks(self, num_restarts: int = 8) -> Dict:
        runs: List[Dict] = []
        for network_file in STRICT_BENCHMARKS:
            inp_path = os.path.join(self.data_dir, network_file)
            if not os.path.exists(inp_path):
                continue
            print(f'Running strict benchmark: {network_file}')
            runs.append(self.run_benchmark(network_file, num_restarts=num_restarts, seed=42))

        payload = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'benchmarks': runs,
        }

        output_path = os.path.join(self.results_dir, 'benchmark_results.json')
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

        print(f'Wrote results to {output_path}')
        return payload


def main():
    data_dir = r'c:\experimental-analysis-of-algorithms\water-distribution-networks\data'
    results_dir = os.path.dirname(__file__) + '\\results'
    runner = BenchmarkRunner(data_dir, results_dir)
    results = runner.run_all_benchmarks(num_restarts=3)

    print('\nStrict benchmark summary')
    print('=' * 72)
    for benchmark in results['benchmarks']:
        result = benchmark['result']
        print(
            f"{benchmark['network_file']:<10} "
            f"score={result['score']:.2f} "
            f"gap={benchmark['gap_percent']:.2f}% "
            f"feasible={result['feasible']}"
        )


if __name__ == '__main__':
    main()