"""
Visualization of Memetic GA results on water distribution networks.

Generates:
- Convergence plots (best/average fitness over generations)
- Algorithm comparison plots
- Cost improvement bar charts
- Network difficulty vs improvement analysis
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List


class ResultsVisualizer:
    """Create visualizations of algorithm results."""
    
    def __init__(self, results_file: str, output_dir: str = None):
        """
        Args:
            results_file: Path to benchmark_results.json
            output_dir: Directory for output plots (default: same as results_file)
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        if output_dir is None:
            output_dir = os.path.dirname(results_file)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Optional file with published reference scores for apples-to-apples comparison.
        self.reference_scores = self._load_reference_scores()
        
        # Style
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10

    def _load_reference_scores(self) -> Dict[str, Dict]:
        """Load optional published reference scores keyed by network filename."""
        reference_path = os.path.join(self.output_dir, 'published_reference_scores.json')
        if not os.path.exists(reference_path):
            return {}

        with open(reference_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}
        return data
    
    def plot_convergence_curves(self):
        """Plot convergence of both algorithms on each benchmark."""
        
        benchmarks = self.results['benchmarks']
        n_benchmarks = len(benchmarks)
        
        fig, axes = plt.subplots(1, n_benchmarks, figsize=(16, 5))
        if n_benchmarks == 1:
            axes = [axes]
        
        colors_meme = ['#2E86AB', '#A23B72', '#1B9E77']  # Blue/Purple/Teal family
        colors_std = ['#C73E1D', '#EE6C4D', '#F4A261']   # Red/Orange family
        
        for ax_idx, benchmark in enumerate(benchmarks):
            ax = axes[ax_idx]
            network_file = benchmark['network']
            difficulty = benchmark['difficulty']
            
            # Get first run data
            meme_run = benchmark['memetic_ga_runs'][0]
            std_run = benchmark['standard_ga_runs'][0]
            
            use_universal = (
                'universal_best_history' in meme_run and
                len(meme_run['universal_best_history']) > 0 and
                'universal_best_history' in std_run and
                len(std_run['universal_best_history']) > 0
            )

            if use_universal:
                meme_best = meme_run['universal_best_history']
                meme_avg = meme_run['universal_avg_history']
                std_best = std_run['universal_best_history']
                std_avg = std_run['universal_avg_history']
                meme_gens = meme_run['universal_generations']
                std_gens = std_run['universal_generations']
                y_label = 'Universal Score (Cost + Penalty, $)'
                plot_title = 'Universal Comparison Metric'
            else:
                meme_best = meme_run['best_fitness_history']
                meme_avg = meme_run['avg_fitness_history']
                std_best = std_run['best_fitness_history']
                std_avg = std_run['avg_fitness_history']
                # Use actual history length (not all generations due to early stopping)
                meme_gens = range(len(meme_best))
                std_gens = range(len(std_best))
                y_label = 'Fitness (Cost, $)'
                plot_title = 'Training Fitness'
            
            ax.plot(
                meme_gens,
                meme_best,
                'o-',
                label='Memetic GA (best)',
                color=colors_meme[ax_idx],
                linewidth=2,
                markersize=4
            )
            
            ax.plot(
                meme_gens,
                meme_avg,
                '--',
                label='Memetic GA (avg)',
                color=colors_meme[ax_idx],
                alpha=0.6,
                linewidth=1.5
            )
            
            ax.plot(
                std_gens,
                std_best,
                's-',
                label='Standard GA (best)',
                color=colors_std[ax_idx],
                linewidth=2,
                markersize=4
            )
            
            ax.plot(
                std_gens,
                std_avg,
                '--',
                label='Standard GA (avg)',
                color=colors_std[ax_idx],
                alpha=0.6,
                linewidth=1.5
            )
            
            ax.set_xlabel('Generation')
            ax.set_ylabel(y_label)
            ax.set_title(f'{difficulty}: {network_file}\n'
                        f"({benchmark['network_stats']['num_pipes']} pipes, {plot_title})")
            # Large penalty spikes can hide normal costs on linear scale.
            # Log scale keeps both early spikes and converged values visible.
            ax.set_yscale('log')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '01_convergence_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_algorithm_comparison(self):
        """Bar chart comparing algorithm performance."""
        
        benchmarks = self.results['benchmarks']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        difficulties = [b['difficulty'] for b in benchmarks]
        meme_costs = [b['summary']['memetic_ga']['mean_cost'] for b in benchmarks]
        std_costs = [b['summary']['standard_ga']['mean_cost'] for b in benchmarks]
        improvements = [b['summary']['avg_improvement_percent'] for b in benchmarks]
        
        x = np.arange(len(difficulties))
        width = 0.35
        
        # Cost comparison
        ax = axes[0]
        bars1 = ax.bar(x - width/2, meme_costs, width, label='Memetic GA', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, std_costs, width, label='Standard GA', color='#C73E1D', alpha=0.8)
        
        ax.set_ylabel('Mean Cost ($)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Difficulty', fontsize=11, fontweight='bold')
        ax.set_title('Algorithm Cost Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'${height:.1e}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        # Cost Difference (Memetic vs Standard)
        # Positive = Memetic is cheaper (better)
        # Negative = Standard is cheaper
        ax = axes[1]
        cost_diff_pct = [(benchmarks[i]['summary']['standard_ga']['mean_cost'] - 
                         benchmarks[i]['summary']['memetic_ga']['mean_cost']) /
                        benchmarks[i]['summary']['standard_ga']['mean_cost'] * 100
                        for i in range(len(difficulties))]
        
        colors_diff = ['#06A77D' if d > 0 else '#C73E1D' for d in cost_diff_pct]
        bars = ax.bar(difficulties, cost_diff_pct, color=colors_diff, alpha=0.8)
        
        ax.set_ylabel('Cost Difference (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Difficulty', fontsize=11, fontweight='bold')
        ax.set_title('Cost Advantage: (Standard GA - Memetic GA)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#06A77D', alpha=0.8, label='Memetic GA Cheaper'),
                          Patch(facecolor='#C73E1D', alpha=0.8, label='Standard GA Cheaper')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '02_algorithm_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_network_difficulty_analysis(self):
        """Analyze performance vs network difficulty."""
        
        benchmarks = self.results['benchmarks']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        network_names = [b['network'] for b in benchmarks]
        num_pipes = [b['network_stats']['num_pipes'] for b in benchmarks]
        num_junctions = [b['network_stats']['num_junctions'] for b in benchmarks]
        improvements = [b['summary']['avg_improvement_percent'] for b in benchmarks]
        
        # Network size vs improvement
        ax = axes[0, 0]
        colors_imp = ['#06A77D', '#2E86AB', '#A23B72']
        ax.scatter(num_pipes, improvements, s=200, c=colors_imp, alpha=0.7, edgecolors='black', linewidth=2)
        for i, name in enumerate(network_names):
            ax.annotate(name, (num_pipes[i], improvements[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Number of Pipes', fontsize=11, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_title('Network Complexity vs Algorithm Improvement', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Junctions vs improvement
        ax = axes[0, 1]
        ax.scatter(num_junctions, improvements, s=200, c=colors_imp, alpha=0.7, edgecolors='black', linewidth=2)
        for i, name in enumerate(network_names):
            ax.annotate(name, (num_junctions[i], improvements[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Number of Junctions', fontsize=11, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_title('Network Nodes vs Algorithm Improvement', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Runtime analysis
        ax = axes[1, 0]
        runtimes_meme = [b['summary']['memetic_ga']['mean_runtime'] for b in benchmarks]
        runtimes_std = [b['summary']['standard_ga']['mean_runtime'] for b in benchmarks]
        
        x = np.arange(len(benchmarks))
        width = 0.35
        ax.bar(x - width/2, runtimes_meme, width, label='Memetic GA', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, runtimes_std, width, label='Standard GA', color='#C73E1D', alpha=0.8)
        
        ax.set_ylabel('Runtime (seconds)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Difficulty', fontsize=11, fontweight='bold')
        ax.set_title('Computational Time Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b['difficulty'] for b in benchmarks])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Solution quality variation (std dev)
        ax = axes[1, 1]
        stds_meme = [b['summary']['memetic_ga']['std_cost'] for b in benchmarks]
        stds_std = [b['summary']['standard_ga']['std_cost'] for b in benchmarks]
        
        # Check if we have meaningful std dev data (need > 1 run)
        has_variance = any(s > 0 for s in stds_meme + stds_std)
        
        if has_variance:
            x = np.arange(len(benchmarks))
            ax.bar(x - width/2, stds_meme, width, label='Memetic GA', color='#2E86AB', alpha=0.8)
            ax.bar(x + width/2, stds_std, width, label='Standard GA', color='#C73E1D', alpha=0.8)
            
            ax.set_ylabel('Cost Std Dev ($)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Difficulty', fontsize=11, fontweight='bold')
            ax.set_title('Solution Quality Consistency (Std Dev)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([b['difficulty'] for b in benchmarks])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        else:
            # Not enough runs to measure variability
            ax.text(0.5, 0.5, 'Insufficient Data\n(Single run - no variance)\n\nRun with num_runs > 1\nto see consistency analysis',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('Solution Quality Consistency (Std Dev)', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '03_network_difficulty_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_solution_details(self):
        """Detailed analysis of solutions found."""
        
        benchmarks = self.results['benchmarks']
        fig, axes = plt.subplots(len(benchmarks), 2, figsize=(14, 5 * len(benchmarks)))
        
        if len(benchmarks) == 1:
            axes = [axes]
        
        for idx, benchmark in enumerate(benchmarks):
            difficulty = benchmark['difficulty']
            
            # Best cost across runs
            ax = axes[idx][0]
            meme_costs = [r['evaluation']['cost'] for r in benchmark['memetic_ga_runs']]
            std_costs = [r['evaluation']['cost'] for r in benchmark['standard_ga_runs']]
            
            run_numbers = list(range(1, len(meme_costs) + 1))
            ax.plot(run_numbers, meme_costs, 'o-', label='Memetic GA', 
                   color='#2E86AB', linewidth=2, markersize=8)
            ax.plot(run_numbers, std_costs, 's-', label='Standard GA',
                   color='#C73E1D', linewidth=2, markersize=8)
            
            ax.set_xlabel('Run Number', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cost ($)', fontsize=10, fontweight='bold')
            ax.set_title(f'{difficulty}: Best Cost Across Runs', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            
            # Average diameter used
            ax = axes[idx][1]
            avg_diameters_meme = [r['evaluation']['avg_diameter'] for r in benchmark['memetic_ga_runs']]
            avg_diameters_std = [r['evaluation']['avg_diameter'] for r in benchmark['standard_ga_runs']]
            
            ax.plot(run_numbers, avg_diameters_meme, 'o-', label='Memetic GA',
                   color='#2E86AB', linewidth=2, markersize=8)
            ax.plot(run_numbers, avg_diameters_std, 's-', label='Standard GA',
                   color='#C73E1D', linewidth=2, markersize=8)
            
            ax.set_xlabel('Run Number', fontsize=10, fontweight='bold')
            ax.set_ylabel('Average Diameter (m)', fontsize=10, fontweight='bold')
            ax.set_title(f'{difficulty}: Average Pipe Diameter', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '04_solution_details.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_published_final_comparison(self):
        """
        Compare final universal score against published best (if provided).

        Uses references from published_reference_scores.json where available.
        """
        benchmarks = self.results['benchmarks']

        labels = []
        meme_scores = []
        std_scores = []
        pub_scores = []

        for benchmark in benchmarks:
            network = benchmark['network']
            if network not in self.reference_scores:
                continue

            ref = self.reference_scores[network]
            published_best = ref.get('published_best_universal_score')
            if published_best is None:
                continue

            labels.append(f"{benchmark['difficulty']}\n{network}")
            meme_score = benchmark['summary']['memetic_ga'].get('mean_paper_score')
            std_score = benchmark['summary']['standard_ga'].get('mean_paper_score')
            if meme_score is None:
                meme_score = benchmark['summary']['memetic_ga']['mean_universal_score']
            if std_score is None:
                std_score = benchmark['summary']['standard_ga']['mean_universal_score']

            meme_scores.append(float(meme_score) if np.isfinite(meme_score) else np.nan)
            std_scores.append(float(std_score) if np.isfinite(std_score) else np.nan)
            pub_scores.append(float(published_best))

        if not labels:
            print("⚠ Skipping published comparison plot (no published reference scores found).")
            return

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_meme = ax.bar(x - width, meme_scores, width, label='Memetic GA (mean final score)', color='#2E86AB', alpha=0.85)
        bars_std = ax.bar(x, std_scores, width, label='Standard GA (mean final score)', color='#C73E1D', alpha=0.85)
        bars_pub = ax.bar(x + width, pub_scores, width, label='Published Best (universal)', color='#1B9E77', alpha=0.9)

        ax.set_title('Final Score vs Published Best', fontsize=13, fontweight='bold')
        ax.set_ylabel('Final Score (benchmark currency)')
        ax.set_xlabel('Benchmark')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for i in range(len(labels)):
            pub = pub_scores[i]
            meme_gap = 100.0 * (meme_scores[i] - pub) / pub if pub > 0 and np.isfinite(meme_scores[i]) else np.nan
            std_gap = 100.0 * (std_scores[i] - pub) / pub if pub > 0 and np.isfinite(std_scores[i]) else np.nan

            if np.isfinite(meme_scores[i]) and np.isfinite(meme_gap):
                ax.text(x[i] - width, meme_scores[i], f"{meme_gap:+.1f}%", ha='center', va='bottom', fontsize=8)
            else:
                ax.text(x[i] - width, pub_scores[i] * 0.05, 'N/A', ha='center', va='bottom', fontsize=8)

            if np.isfinite(std_scores[i]) and np.isfinite(std_gap):
                ax.text(x[i], std_scores[i], f"{std_gap:+.1f}%", ha='center', va='bottom', fontsize=8)
            else:
                ax.text(x[i], pub_scores[i] * 0.08, 'N/A', ha='center', va='bottom', fontsize=8)

        output_path = os.path.join(self.output_dir, '05_final_universal_vs_published.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_published_gap_histogram(self):
        """
        Plot histogram-style bars of gap (%) to published best for final universal score.
        """
        benchmarks = self.results['benchmarks']

        labels = []
        meme_gaps = []
        std_gaps = []

        for benchmark in benchmarks:
            network = benchmark['network']
            if network not in self.reference_scores:
                continue

            ref = self.reference_scores[network]
            published_best = ref.get('published_best_universal_score')
            if published_best is None or published_best <= 0:
                continue

            meme = benchmark['summary']['memetic_ga'].get('mean_paper_score')
            std = benchmark['summary']['standard_ga'].get('mean_paper_score')
            if meme is None:
                meme = benchmark['summary']['memetic_ga']['mean_universal_score']
            if std is None:
                std = benchmark['summary']['standard_ga']['mean_universal_score']

            labels.append(f"{benchmark['difficulty']}\n{network}")
            meme_gaps.append(100.0 * (meme - published_best) / published_best if np.isfinite(meme) else np.nan)
            std_gaps.append(100.0 * (std - published_best) / published_best if np.isfinite(std) else np.nan)

        if not labels:
            print("⚠ Skipping published gap histogram (no published reference scores found).")
            return

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_meme = ax.bar(x - width / 2, meme_gaps, width, label='Memetic GA Gap (%)', color='#2E86AB', alpha=0.85)
        bars_std = ax.bar(x + width / 2, std_gaps, width, label='Standard GA Gap (%)', color='#C73E1D', alpha=0.85)

        ax.axhline(y=0, color='black', linewidth=1.2)
        ax.set_title('Final Score Gap to Published Best', fontsize=13, fontweight='bold')
        ax.set_ylabel('Gap to Published Best (%)')
        ax.set_xlabel('Benchmark')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        for bars in (bars_meme, bars_std):
            for bar in bars:
                h = bar.get_height()
                if not np.isfinite(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0,
                        'N/A',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:+.1f}%",
                    ha='center',
                    va='bottom' if h >= 0 else 'top',
                    fontsize=8
                )

        output_path = os.path.join(self.output_dir, '06_final_gap_to_published_hist.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_all(self):
        """Generate all visualization plots."""
        
        print("\nGenerating visualizations...")
        print("-" * 80)
        
        self.plot_convergence_curves()
        self.plot_algorithm_comparison()
        self.plot_network_difficulty_analysis()
        self.plot_solution_details()
        self.plot_published_final_comparison()
        self.plot_published_gap_histogram()
        
        print("-" * 80)
        print(f"✓ All visualizations saved to: {self.output_dir}")


def main():
    """Generate visualizations from benchmark results."""
    
    results_file = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_003\results\benchmark_results.json"
    
    visualizer = ResultsVisualizer(results_file)
    visualizer.plot_all()
    
    print("\nVisualization complete!")
    print(f"Results directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
