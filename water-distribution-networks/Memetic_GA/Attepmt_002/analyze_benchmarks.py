"""
Analyze benchmark instances to select easy/medium/hard networks.

Difficulty is measured by:
- Number of pipes (decision variables)
- Network connectivity and structure
- Demand characteristics
"""

import os
import sys
from network_parser import parse_inp_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def analyze_benchmark(data_dir: str):
    """Analyze all .inp files and report statistics."""
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.inp')]
    files.sort()
    
    results = []
    
    print("Scanning benchmark instances...")
    print("-" * 80)
    print(f"{'File':<25} {'Pipes':<8} {'Junctions':<12} {'Demand':<12} {'Difficulty'}")
    print("-" * 80)
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            network = parse_inp_file(filepath)
            stats = network.get_network_stats()
            
            num_pipes = stats['num_pipes']
            num_junctions = stats['num_junctions']
            total_demand = stats['total_demand']
            
            # Difficulty score: primary is num_pipes (search space)
            difficulty_score = num_pipes
            
            results.append({
                'filename': filename,
                'num_pipes': num_pipes,
                'num_junctions': num_junctions,
                'total_demand': total_demand,
                'difficulty_score': difficulty_score
            })
            
            print(f"{filename:<25} {num_pipes:<8} {num_junctions:<12} {total_demand:<12.0f}")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    
    print("-" * 80)
    
    # Filter out networks with 0 pipes (invalid) and very large networks (> 1000 pipes)
    valid_results = [r for r in results if 0 < r['num_pipes'] < 1000]
    
    # Sort by difficulty (number of pipes)
    valid_results.sort(key=lambda x: x['difficulty_score'])
    
    # Select easy, medium, hard from valid networks
    n = len(valid_results)
    if n >= 3:
        # Easy: smallest network (< 50 pipes)
        easy = valid_results[0]
        # Medium: middle range (50-300 pipes)
        medium = valid_results[n // 2]
        # Hard: largest (300-1000 pipes)
        hard = valid_results[-1]
    elif n == 2:
        easy = valid_results[0]
        medium = valid_results[0]
        hard = valid_results[1]
    else:
        easy = medium = hard = valid_results[0]
    
    print("\n✓ Selected Benchmarks:")
    print("-" * 80)
    print(f"EASY:   {easy['filename']:<25} ({easy['num_pipes']} pipes, {easy['num_junctions']} junctions)")
    print(f"MEDIUM: {medium['filename']:<25} ({medium['num_pipes']} pipes, {medium['num_junctions']} junctions)")
    print(f"HARD:   {hard['filename']:<25} ({hard['num_pipes']} pipes, {hard['num_junctions']} junctions)")
    print("-" * 80)
    
    return easy['filename'], medium['filename'], hard['filename']


if __name__ == "__main__":
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    easy, medium, hard = analyze_benchmark(data_dir)
    print(f"\nSelected:\n  Easy:   {easy}\n  Medium: {medium}\n  Hard:   {hard}")
