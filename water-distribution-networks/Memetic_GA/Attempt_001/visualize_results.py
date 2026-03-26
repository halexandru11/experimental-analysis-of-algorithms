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
        
        # Style
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 10
    
    def plot_convergence_curves(self):
        """Plot convergence of both algorithms on each benchmark."""
        
        benchmarks = self.results['benchmarks']
        n_benchmarks = len(benchmarks)
        
        fig, axes = plt.subplots(1, n_benchmarks, figsize=(16, 5))
        if n_benchmarks == 1:
            axes = [axes]
        
        colors_meme = ['#2E86AB', '#A23B72', '#F18F01']  # Blue family
        colors_std = ['#C73E1D', '#EE6C4D', '#F4A261']   # Red/Orange family
        
        for ax_idx, benchmark in enumerate(benchmarks):
            ax = axes[ax_idx]
            network_file = benchmark['network']
            difficulty = benchmark['difficulty']
            
            # Get first run data
            meme_run = benchmark['memetic_ga_runs'][0]
            std_run = benchmark['standard_ga_runs'][0]
            
            meme_best = meme_run['best_fitness_history']
            meme_avg = meme_run['avg_fitness_history']
            std_best = std_run['best_fitness_history']
            std_avg = std_run['avg_fitness_history']
            
            # Use actual history length (not all generations due to early stopping)
            meme_gens = range(len(meme_best))
            std_gens = range(len(std_best))
            
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
            ax.set_ylabel('Fitness (Cost, $)')
            ax.set_title(f'{difficulty}: {network_file}\n'
                        f"({benchmark['network_stats']['num_pipes']} pipes)")
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
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
    
    def plot_all(self):
        """Generate all visualization plots."""
        
        print("\nGenerating visualizations...")
        print("-" * 80)
        
        self.plot_convergence_curves()
        self.plot_algorithm_comparison()
        self.plot_network_difficulty_analysis()
        self.plot_solution_details()
        
        print("-" * 80)
        print(f"✓ All visualizations saved to: {self.output_dir}")


def main():
    """Generate visualizations from benchmark results."""
    
    results_file = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_001\results\benchmark_results.json"
    
    visualizer = ResultsVisualizer(results_file)
    visualizer.plot_all()
    
    print("\nVisualization complete!")
    print(f"Results directory: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
