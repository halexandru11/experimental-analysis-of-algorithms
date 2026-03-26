"""
Fitness evaluation for water distribution network optimization.

The objective is to minimize the total cost of pipes while satisfying
hydraulic constraints (pressure, flow requirements).

Design choices:
- Pipe cost: Cost ∝ diameter^2 * length (realistic cost model)
- Hydraulic feasibility: Simplified pressure drop calculation
- Pentalty-based approach for constraint violation
"""

import numpy as np
from typing import List, Tuple
from network_parser import WaterNetwork, Pipe


# Available commercial pipe diameters (in meters)
AVAILABLE_DIAMETERS = [
    0.0508,    # 2 inches
    0.0635,    # 2.5 inches
    0.0762,    # 3 inches
    0.1016,    # 4 inches
    0.1270,    # 5 inches
    0.1524,    # 6 inches
    0.2032,    # 8 inches
    0.2540,    # 10 inches
    0.3048,    # 12 inches
]

# Unit cost per unit volume of pipe material ($/m^3)
# Realistic cost model: cost increases with diameter
UNIT_COST = 1000.0  # Cost coefficient


class FitnessEvaluator:
    """
    Evaluates fitness of pipe diameter configurations.
    
    Design rationale:
    - Uses realistic cost model: cost ~ diameter^2 (material volume)
    - Simplified hydraulic feasibility check
    - Penalty-based constraint handling
    - Minimization problem (lower fitness = better solution)
    """
    
    def __init__(self, network: WaterNetwork):
        self.network = network
        self.num_pipes = network.get_pipe_count()
        self.diameter_options = len(AVAILABLE_DIAMETERS)
        self.min_pressure = 20.0  # Minimum pressure at junctions (meters)
        self.reference_cost = self._calculate_reference_cost()
        
    def _calculate_reference_cost(self) -> float:
        """Calculate cost with all pipes at maximum diameter (for normalization)."""
        max_diameter = AVAILABLE_DIAMETERS[-1]
        cost = 0.0
        for pipe in self.network.pipes_list:
            cost += self._pipe_cost(max_diameter, pipe.length)
        return cost
    
    @staticmethod
    def _pipe_cost(diameter: float, length: float) -> float:
        """
        Calculate pipe installation cost.
        
        Realistic model based on pipe volume (material cost):
        Cost = unit_cost * length * π * (diameter/2)^2
        """
        volume = length * np.pi * (diameter / 2.0) ** 2
        return UNIT_COST * volume
    
    def calculate_total_cost(self, diameters: List[float]) -> float:
        """
        Calculate total cost for a given diameter configuration.
        
        Args:
            diameters: List of diameters for each pipe
            
        Returns:
            Total cost in $.
        """
        total_cost = 0.0
        for i, diameter in enumerate(diameters):
            pipe = self.network.pipes_list[i]
            total_cost += self._pipe_cost(diameter, pipe.length)
        return total_cost
    
    def _simplified_hydraulic_check(self, diameters: List[float]) -> float:
        """
        FAST simplified hydraulic feasibility check (optimized for speed).
        
        Uses minimal computation for development speed:
        - Quick undersizing check
        - Simple diameter penalty
        
        Args:
            diameters: Pipe diameter configuration
            
        Returns:
            Penalty value (0 if all valid, else penalty).
        """
        min_valid_diameter = AVAILABLE_DIAMETERS[0]
        
        # Quick check: count undersized pipes
        undersized_count = sum(1 for d in diameters if d < min_valid_diameter)
        
        if undersized_count > 0:
            # Heavy penalty for invalid configurations
            return float(undersized_count * 1e10)
        
        # Light penalty: prefer larger diameters for feasibility
        # Penalize very small diameters (increase feasibility pressure)
        small_diameter_count = sum(1 for d in diameters if d <= AVAILABLE_DIAMETERS[1])
        
        return float(small_diameter_count * 1e9)  # Minimal penalty
    
    def evaluate(self, diameters: List[float]) -> float:
        """
        Evaluate fitness of a solution (diameter configuration).
        
        Design: Minimize cost with penalty for hydraulic infeasibility.
        
        Fitness = Cost + Penalty * Constraint_Violation
        
        Args:
            diameters: List of pipe diameters
            
        Returns:
            Fitness value (lower is better)
        """
        # Validate input
        if len(diameters) != self.num_pipes:
            return float('inf')
        
        # Calculate cost component (FAST)
        cost = self.calculate_total_cost(diameters)
        
        # Calculate feasibility component (FAST)
        feasibility_penalty = self._simplified_hydraulic_check(diameters)
        
        # Simple combination: cost + penalty
        # No complex weighting for speed
        fitness = cost + feasibility_penalty
        
        return fitness
    
    def diameter_to_index(self, diameter: float) -> int:
        """Convert diameter value to closest available diameter index."""
        # Find closest available diameter
        idx = np.argmin([abs(d - diameter) for d in AVAILABLE_DIAMETERS])
        return idx
    
    def index_to_diameter(self, idx: int) -> float:
        """Convert diameter index to actual diameter value."""
        return AVAILABLE_DIAMETERS[min(idx, len(AVAILABLE_DIAMETERS) - 1)]
    
    def indices_to_diameters(self, indices: List[int]) -> List[float]:
        """Convert list of indices to diameter values."""
        return [self.index_to_diameter(idx) for idx in indices]
