"""
Test the Memetic GA on water distribution network benchmarks.

Runs the algorithm on easy/medium/hard networks and compares:
- Memetic GA (with local search)
- Standard GA (without local search)

Generates comprehensive results and statistics.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from network_parser import parse_inp_file, WaterNetwork
from fitness_evaluator import FitnessEvaluator, AVAILABLE_DIAMETERS
from memetic_ga import MemeticGA, Individual
from analyze_benchmarks import analyze_benchmark
from visualize_results import ResultsVisualizer


class BenchmarkRunner:
    """Runs and evaluates algorithms on benchmark instances."""
    
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def run_memetic_ga(
        self,
        network: WaterNetwork,
        population_size: int = 30,
        max_generations: int = 30,
        local_search_intensity: float = 0.5,
        seed: int = 42
    ) -> Tuple[Individual, List[float], List[float], float]:
        """
        Run Memetic GA on network.
        
        Returns:
            (best_individual, best_fitness_history, avg_fitness_history, runtime)
        """
        start_time = time.time()
        
        ga = MemeticGA(
            network,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            local_search_intensity=local_search_intensity,
            seed=seed
        )
        
        best_individual, best_hist, avg_hist = ga.run()
        runtime = time.time() - start_time
        
        return best_individual, best_hist, avg_hist, runtime
    
    def run_standard_ga(
        self,
        network: WaterNetwork,
        population_size: int = 30,
        max_generations: int = 30,
        seed: int = 42
    ) -> Tuple[Individual, List[float], List[float], float]:
        """
        Run GA without local search (standard GA).
        
        Returns:
            (best_individual, best_fitness_history, avg_fitness_history, runtime)
        """
        start_time = time.time()
        
        ga = MemeticGA(
            network,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            local_search_intensity=0.0,  # Disable local search
            seed=seed
        )
        
        best_individual, best_hist, avg_hist = ga.run()
        runtime = time.time() - start_time
        
        return best_individual, best_hist, avg_hist, runtime
    
    def evaluate_solution(
        self,
        individual: Individual,
        network: WaterNetwork
    ) -> Dict:
        """Evaluate solution and return detailed metrics."""
        
        diameters = individual.evaluator.indices_to_diameters(individual.chromosome)
        cost = individual.evaluator.calculate_total_cost(diameters)
        
        # Calculate some pipe statistics
        diameter_counts = {}
        for idx in individual.chromosome:
            d = AVAILABLE_DIAMETERS[idx]
            diameter_counts[d] = diameter_counts.get(d, 0) + 1
        
        return {
            'fitness': individual.fitness,
            'cost': cost,
            'avg_diameter': np.mean(diameters),
            'max_diameter': max(diameters),
            'min_diameter': min(diameters),
            'diameter_distribution': diameter_counts
        }
    
    def run_benchmark(
        self,
        network_file: str,
        difficulty: str,
        num_runs: int = 3
    ) -> Dict:
        """
        Run complete benchmark on network file.
        
        Tests both Memetic GA and Standard GA multiple times.
        """
        
        print(f"\n{'='*80}")
        print(f"Running benchmark: {network_file} ({difficulty})")
        print(f"{'='*80}")
        
        # Parse network
        filepath = os.path.join(self.data_dir, network_file)
        network = parse_inp_file(filepath)
        stats = network.get_network_stats()
        
        results = {
            'network': network_file,
            'difficulty': difficulty,
            'network_stats': stats,
            'memetic_ga_runs': [],
            'standard_ga_runs': []
        }
        
        # Run multiple instances
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Memetic GA
            print("Running Memetic GA (with local search)...")
            best_meme, best_hist_meme, avg_hist_meme, time_meme = self.run_memetic_ga(
                network,
                population_size=30,
                max_generations=30,
                local_search_intensity=0.5,
                seed=42 + run
            )
            
            eval_meme = self.evaluate_solution(best_meme, network)
            results['memetic_ga_runs'].append({
                'run': run + 1,
                'best_fitness_history': best_hist_meme,
                'avg_fitness_history': avg_hist_meme,
                'runtime': time_meme,
                'evaluation': eval_meme
            })
            
            print(f"  Memetic GA Best Cost: ${eval_meme['cost']:.2e}")
            
            # Standard GA
            print("Running Standard GA (no local search)...")
            best_std, best_hist_std, avg_hist_std, time_std = self.run_standard_ga(
                network,
                population_size=30,
                max_generations=30,
                seed=42 + run
            )
            
            eval_std = self.evaluate_solution(best_std, network)
            results['standard_ga_runs'].append({
                'run': run + 1,
                'best_fitness_history': best_hist_std,
                'avg_fitness_history': avg_hist_std,
                'runtime': time_std,
                'evaluation': eval_std
            })
            
            print(f"  Standard GA Best Cost: ${eval_std['cost']:.2e}")
            
            # Calculate improvement
            improvement = (
                (eval_std['fitness'] - eval_meme['fitness']) / eval_std['fitness'] * 100
            ) if eval_std['fitness'] > 0 else 0
            print(f"  Memetic GA Improvement: {improvement:.2f}%")
        
        # Calculate aggregate statistics
        meme_costs = [r['evaluation']['cost'] for r in results['memetic_ga_runs']]
        std_costs = [r['evaluation']['cost'] for r in results['standard_ga_runs']]
        
        results['summary'] = {
            'memetic_ga': {
                'mean_cost': np.mean(meme_costs),
                'std_cost': np.std(meme_costs),
                'min_cost': min(meme_costs),
                'max_cost': max(meme_costs),
                'mean_runtime': np.mean([r['runtime'] for r in results['memetic_ga_runs']])
            },
            'standard_ga': {
                'mean_cost': np.mean(std_costs),
                'std_cost': np.std(std_costs),
                'min_cost': min(std_costs),
                'max_cost': max(std_costs),
                'mean_runtime': np.mean([r['runtime'] for r in results['standard_ga_runs']])
            }
        }
        
        avg_improvement = (
            (results['summary']['standard_ga']['mean_cost'] - 
             results['summary']['memetic_ga']['mean_cost']) /
            results['summary']['standard_ga']['mean_cost'] * 100
        )
        results['summary']['avg_improvement_percent'] = avg_improvement
        
        print(f"\n--- Summary ---")
        print(f"Memetic GA (avg): ${results['summary']['memetic_ga']['mean_cost']:.2e}")
        print(f"Standard GA (avg): ${results['summary']['standard_ga']['mean_cost']:.2e}")
        print(f"Average Improvement: {avg_improvement:.2f}%")
        
        return results
    
    def run_all_benchmarks(self, num_runs: int = 3):
        """Run complete benchmark suite."""
        
        print("\nAnalyzing benchmarks to select easy/medium/hard instances...")
        easy_file, medium_file, hard_file = analyze_benchmark(self.data_dir)
        
        benchmarks = [
            (easy_file, 'Easy'),
            (medium_file, 'Medium'),
            (hard_file, 'Hard')
        ]
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': []
        }
        
        for network_file, difficulty in benchmarks:
            results = self.run_benchmark(network_file, difficulty, num_runs)
            all_results['benchmarks'].append(results)
        
        # Save results
        results_file = os.path.join(self.results_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json.dump(
                all_results,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
            )
        
        print(f"\n✓ Results saved to {results_file}")
        
        return all_results


def main():
    """Main execution."""
    
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    results_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_001\results"
    
    runner = BenchmarkRunner(data_dir, results_dir)
    results = runner.run_all_benchmarks(num_runs=3)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for benchmark in results['benchmarks']:
        print(f"\n{benchmark['network']} ({benchmark['difficulty']}):")
        print(f"  Network: {benchmark['network_stats']['num_pipes']} pipes, "
              f"{benchmark['network_stats']['num_junctions']} junctions")
        summary = benchmark['summary']
        print(f"  Memetic GA:  ${summary['memetic_ga']['mean_cost']:.2e} ±"
              f" ${summary['memetic_ga']['std_cost']:.2e}")
        print(f"  Standard GA: ${summary['standard_ga']['mean_cost']:.2e} ±"
              f" ${summary['standard_ga']['std_cost']:.2e}")
        print(f"  Improvement: {summary['avg_improvement_percent']:.2f}%")


if __name__ == "__main__":
    import sys
    
    # Allow passing number of runs as command line argument
    num_runs = 1  # Default: 1 run for faster development testing
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            pass
    
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    results_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_001\results"
    
    runner = BenchmarkRunner(data_dir, results_dir)
    results = runner.run_all_benchmarks(num_runs=num_runs)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for benchmark in results['benchmarks']:
        print(f"\n{benchmark['network']} ({benchmark['difficulty']}):")
        print(f"  Network: {benchmark['network_stats']['num_pipes']} pipes, "
              f"{benchmark['network_stats']['num_junctions']} junctions")
        summary = benchmark['summary']
        print(f"  Memetic GA:  ${summary['memetic_ga']['mean_cost']:.2e} ±"
              f" ${summary['memetic_ga']['std_cost']:.2e}")
        print(f"  Standard GA: ${summary['standard_ga']['mean_cost']:.2e} ±"
              f" ${summary['standard_ga']['std_cost']:.2e}")
        print(f"  Improvement: {summary['avg_improvement_percent']:.2f}%")
    
    # Automatically generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    results_file = os.path.join(results_dir, "benchmark_results.json")
    visualizer = ResultsVisualizer(results_file, results_dir)
    visualizer.plot_convergence_curves()
    visualizer.plot_algorithm_comparison()
    visualizer.plot_network_difficulty_analysis()
    visualizer.plot_solution_details()
    print("\n✓ All visualizations generated successfully!")
