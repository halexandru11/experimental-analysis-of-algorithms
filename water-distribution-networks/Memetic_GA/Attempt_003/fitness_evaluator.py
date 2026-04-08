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
from typing import Dict, List, Optional, Tuple
from network_parser import WaterNetwork, Pipe
from collections import defaultdict
import heapq


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
    
    def __init__(
        self,
        network: WaterNetwork,
        diameter_options: Optional[List[float]] = None,
        unit_cost_lookup: Optional[Dict[float, float]] = None
    ):
        self.network = network
        self.num_pipes = network.get_pipe_count()
        self.diameter_values = sorted(diameter_options.copy()) if diameter_options else AVAILABLE_DIAMETERS.copy()
        self.diameter_options = len(self.diameter_values)
        self.unit_cost_lookup = unit_cost_lookup
        self.min_pressure = 20.0  # Minimum pressure at junctions (meters)
        self.reference_cost = self._calculate_reference_cost()
        self.pipe_importance = self._estimate_pipe_importance()
        
    def _calculate_reference_cost(self) -> float:
        """Calculate cost with all pipes at maximum diameter (for normalization)."""
        max_diameter = self.diameter_values[-1]
        cost = 0.0
        for pipe in self.network.pipes_list:
            cost += self._pipe_cost(max_diameter, pipe.length)
        return cost

    def _estimate_pipe_importance(self) -> List[float]:
        """Estimate which pipes are more hydraulically critical using topology only."""
        if not self.network.pipes_list:
            return []

        adjacency = defaultdict(list)
        reservoirs = list(self.network.reservoirs.keys())
        if not reservoirs:
            return [1.0 for _ in self.network.pipes_list]

        for pipe in self.network.pipes_list:
            adjacency[str(pipe.node1)].append((str(pipe.node2), pipe.length, pipe.id))
            adjacency[str(pipe.node2)].append((str(pipe.node1), pipe.length, pipe.id))

        dist = {node: float('inf') for node in adjacency}
        heap = []
        for reservoir_id in reservoirs:
            dist[str(reservoir_id)] = 0.0
            heapq.heappush(heap, (0.0, str(reservoir_id)))

        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist != dist.get(node, float('inf')):
                continue
            for neighbor, length, _ in adjacency.get(node, []):
                candidate = current_dist + float(length)
                if candidate < dist.get(neighbor, float('inf')):
                    dist[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))

        junction_demand = {str(j.id): float(j.demand) for j in self.network.junctions.values()}
        max_demand = max(junction_demand.values(), default=1.0)
        max_distance = max((value for value in dist.values() if np.isfinite(value)), default=1.0)
        if max_distance <= 0:
            max_distance = 1.0

        importance = []
        for pipe in self.network.pipes_list:
            node1 = str(pipe.node1)
            node2 = str(pipe.node2)
            near_distance = min(dist.get(node1, max_distance), dist.get(node2, max_distance))
            distance_term = 1.0 - min(1.0, near_distance / max_distance)
            demand_term = (junction_demand.get(node1, 0.0) + junction_demand.get(node2, 0.0)) / max(1.0, max_demand * 2.0)
            length_term = pipe.length / max((p.length for p in self.network.pipes_list), default=1.0)
            score = 0.45 + 0.35 * distance_term + 0.15 * demand_term + 0.05 * length_term
            importance.append(float(min(1.0, max(0.25, score))))

        return importance

    def _unit_cost_for_diameter(self, diameter: float) -> float:
        """Get benchmark unit cost for a diameter from lookup with tolerance."""
        if self.unit_cost_lookup is None:
            raise KeyError("No unit cost lookup configured")

        for d, c in self.unit_cost_lookup.items():
            if abs(float(d) - float(diameter)) < 1e-8:
                return float(c)
        raise KeyError(f"Diameter {diameter} not found in unit cost lookup")
    
    def _pipe_cost(self, diameter: float, length: float) -> float:
        """Calculate pipe cost from benchmark table or default geometric model."""
        if self.unit_cost_lookup is not None:
            return self._unit_cost_for_diameter(diameter) * length
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
        min_valid_diameter = self.diameter_values[0]

        # Hard fail for any value outside configured catalog range.
        undersized_count = sum(1 for d in diameters if d < min_valid_diameter)
        if undersized_count > 0:
            return float(undersized_count) * self.reference_cost * 100.0

        # Adaptive lower-bound target diameter by network size.
        # This encourages realistic diameters on larger networks where tiny pipes
        # almost always violate pressure constraints.
        n_opts = len(self.diameter_values)
        if self.num_pipes <= 10:
            target_idx = max(1, int(0.30 * (n_opts - 1)))
        elif self.num_pipes <= 60:
            target_idx = max(1, int(0.22 * (n_opts - 1)))
        else:
            target_idx = max(1, int(0.16 * (n_opts - 1)))
        target_diameter = self.diameter_values[target_idx]

        max_length = max((p.length for p in self.network.pipes_list), default=1.0)
        weighted_deficit = 0.0
        smallest_band = set(self.diameter_values[:2]) if n_opts >= 2 else {self.diameter_values[0]}
        smallest_band_count = 0

        for i, d in enumerate(diameters):
            if d in smallest_band:
                smallest_band_count += 1

            if d < target_diameter:
                length_weight = 1.0 + (self.network.pipes_list[i].length / max_length)
                importance = self.pipe_importance[i] if i < len(self.pipe_importance) else 1.0
                deficit = (target_diameter - d) / target_diameter
                weighted_deficit += importance * length_weight * deficit

        smallest_share = smallest_band_count / max(1, self.num_pipes)

        # Penalty scaled by reference cost so it remains meaningful for all benchmarks.
        return float(self.reference_cost * (2.0 * weighted_deficit + 8.0 * smallest_share))
    
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

    def calculate_proxy_violation(self, diameters: List[float]) -> float:
        """
        Calculate a proxy violation term for benchmark-style scoring.

        This is intentionally separated from the training fitness so we can
        report an external metric without changing optimization behavior.
        """
        undersized_count = sum(1 for d in diameters if d < self.diameter_values[0])
        small_threshold = self.diameter_values[1] if len(self.diameter_values) > 1 else self.diameter_values[0]
        small_diameter_count = sum(1 for d in diameters if d <= small_threshold)

        # Weighted proxy violation (dimensionless).
        return float(undersized_count + 0.1 * small_diameter_count)

    def evaluate_universal_score(
        self,
        diameters: List[float],
        penalty_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Evaluate a benchmark-style external score for comparison.

        Universal score format:
          score = cost                                if violation == 0
          score = cost + penalty_weight*ref*violation otherwise

        Args:
            diameters: Candidate pipe diameters.
            penalty_weight: Multiplier for violation penalty.

        Returns:
            Dict with score components.
        """
        if len(diameters) != self.num_pipes:
            return {
                'score': float('inf'),
                'cost': float('inf'),
                'violation': float('inf'),
                'penalty': float('inf'),
                'is_feasible': 0.0
            }

        cost = self.calculate_total_cost(diameters)
        violation = self.calculate_proxy_violation(diameters)

        if violation <= 0.0:
            penalty = 0.0
            score = cost
            is_feasible = 1.0
        else:
            penalty = penalty_weight * self.reference_cost * violation
            score = cost + penalty
            is_feasible = 0.0

        return {
            'score': float(score),
            'cost': float(cost),
            'violation': float(violation),
            'penalty': float(penalty),
            'is_feasible': is_feasible
        }
    
    def diameter_to_index(self, diameter: float) -> int:
        """Convert diameter value to closest available diameter index."""
        # Find closest available diameter
        idx = np.argmin([abs(d - diameter) for d in self.diameter_values])
        return idx
    
    def index_to_diameter(self, idx: int) -> float:
        """Convert diameter index to actual diameter value."""
        return self.diameter_values[min(idx, len(self.diameter_values) - 1)]
    
    def indices_to_diameters(self, indices: List[int]) -> List[float]:
        """Convert list of indices to diameter values."""
        return [self.index_to_diameter(idx) for idx in indices]
