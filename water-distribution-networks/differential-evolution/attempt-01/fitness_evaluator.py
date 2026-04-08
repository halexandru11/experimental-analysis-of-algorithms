"""
Simplified (but fast) fitness evaluation for WDN pipe-diameter optimization.

This mirrors the model used by `Memetic_GA/Attempt_001` but is optimized by
precomputing per-pipe costs for all candidate diameters.

Objective (minimization):
  fitness = total_pipe_cost + feasibility_penalty

Where feasibility_penalty uses a fast surrogate:
  - heavy penalty for choosing the smallest commercial sizes

This is not a full hydraulic simulation; it's consistent with the baseline
implementation so DE can be benchmarked similarly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from network_parser import WaterNetwork


# Available commercial pipe diameters (in meters)
AVAILABLE_DIAMETERS: List[float] = [
    0.0508,  # 2 inches
    0.0635,  # 2.5 inches
    0.0762,  # 3 inches
    0.1016,  # 4 inches
    0.1270,  # 5 inches
    0.1524,  # 6 inches
    0.2032,  # 8 inches
    0.2540,  # 10 inches
    0.3048,  # 12 inches
]

UNIT_COST: float = 1000.0  # Cost coefficient


@dataclass(frozen=True)
class PenaltyParams:
    big_penalty: float = 1e10
    small_penalty: float = 1e9
    # Penalize diameters up to and including this commercial option index.
    small_threshold_idx: int = 1


class FastFitnessEvaluator:
    """
    Fast fitness evaluator using precomputed costs.

    Decision variables:
      - for each pipe i, choose an integer diameter option index in [0, K-1]

    Input:
      - an array of integer indices (shape: (dim,)) or (pop, dim)
    Output:
      - scalar fitness or (pop,) fitness vector (lower is better)
    """

    def __init__(
        self,
        network: WaterNetwork,
        penalty: PenaltyParams = PenaltyParams(),
    ):
        self.network = network
        self.num_pipes = network.get_pipe_count()
        self.num_options = len(AVAILABLE_DIAMETERS)
        self.penalty = penalty

        # Precompute cost_matrix[i, k] = cost of choosing option k for pipe i.
        lengths = np.array([p.length for p in network.pipes_list], dtype=np.float64)
        diameters = np.array(AVAILABLE_DIAMETERS, dtype=np.float64)

        # Cost proportional to cross-sectional area => pi*(d/2)^2
        areas = np.pi * (diameters / 2.0) ** 2  # shape: (K,)
        self.cost_matrix = UNIT_COST * lengths[:, None] * areas[None, :]  # (dim, K)

    def index_to_diameter(self, idx: int) -> float:
        return AVAILABLE_DIAMETERS[int(np.clip(idx, 0, self.num_options - 1))]

    def indices_to_diameters(self, indices: Sequence[int]) -> List[float]:
        return [self.index_to_diameter(i) for i in indices]

    def evaluate_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for integer indices.

        Parameters
        ----------
        indices:
          - shape (dim,) for single solution
          - shape (pop, dim) for batch
        Returns
        -------
        fitness: shape () or (pop,)
        """

        idx = np.asarray(indices)
        if idx.ndim == 1:
            if idx.shape[0] != self.num_pipes:
                return np.array(float("inf"), dtype=np.float64)
            idx = np.clip(np.rint(idx).astype(np.int64), 0, self.num_options - 1)
            dim = self.num_pipes
            pipe_ids = np.arange(dim, dtype=np.int64)
            cost = np.sum(self.cost_matrix[pipe_ids, idx])
            penalty = self.penalty.small_penalty * float(np.sum(idx <= self.penalty.small_threshold_idx))
            return np.array(cost + penalty, dtype=np.float64)

        if idx.ndim == 2:
            if idx.shape[1] != self.num_pipes:
                return np.full((idx.shape[0],), float("inf"), dtype=np.float64)

            idx = np.clip(np.rint(idx).astype(np.int64), 0, self.num_options - 1)
            pop = idx.shape[0]
            dim = self.num_pipes

            pipe_ids = np.arange(dim, dtype=np.int64)[None, :]  # (1, dim)
            costs = self.cost_matrix[pipe_ids, idx]  # (pop, dim)
            cost_sum = np.sum(costs, axis=1)  # (pop,)

            penalty_sum = self.penalty.small_penalty * np.sum(
                idx <= self.penalty.small_threshold_idx, axis=1
            )  # (pop,)
            return cost_sum + penalty_sum

        raise ValueError(f"Unsupported indices shape: {idx.shape}")

    def evaluate_diameters(self, diameters: Sequence[float]) -> float:
        """Compatibility helper; expects a diameter list matching network pipes."""
        if len(diameters) != self.num_pipes:
            return float("inf")
        # Map each diameter to closest option index.
        options = np.array(AVAILABLE_DIAMETERS, dtype=np.float64)
        d = np.array(list(diameters), dtype=np.float64)[:, None]
        idx = np.argmin(np.abs(d - options[None, :]), axis=1)
        return float(self.evaluate_indices(idx))

