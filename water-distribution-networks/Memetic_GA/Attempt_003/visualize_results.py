"""
Visualization helpers for Attempt 003.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


class ResultsVisualizer:
    def __init__(self, results_file: str, output_dir: str | None = None):
        with open(results_file, 'r', encoding='utf-8') as handle:
            self.results = json.load(handle)

        if output_dir is None:
            output_dir = os.path.dirname(results_file)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_strict_gap_comparison(self):
        benchmarks = self.results.get('benchmarks', [])
        if not benchmarks:
            return

        labels = [item['network_file'] for item in benchmarks]
        scores = [item['result']['score'] for item in benchmarks]
        reference = [item['published_best_universal_score'] for item in benchmarks]
        gaps = [item['gap_percent'] for item in benchmarks]

        x = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.bar(x - width / 2, scores, width, label='Attempt 003', color='#2E86AB')
        ax.bar(x + width / 2, reference, width, label='Published best', color='#2AA876')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Score')
        ax.set_title('Final score vs published best')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1]
        colors = ['#2AA876' if gap <= 0 else '#C73E1D' for gap in gaps]
        ax.bar(labels, gaps, color=colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Gap (%)')
        ax.set_title('Gap to published best')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'attempt_003_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {output_path}')

    def plot_all(self):
        self.plot_strict_gap_comparison()


def main():
    results_file = r'c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_003\results\benchmark_results.json'
    visualizer = ResultsVisualizer(results_file)
    visualizer.plot_all()


if __name__ == '__main__':
    main()