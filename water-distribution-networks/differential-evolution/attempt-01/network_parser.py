"""
Parser for EPANET water distribution network files (.inp format).

This is intentionally kept compatible with the parser used in
`Memetic_GA/Attempt_001` so that benchmarking across algorithms uses the
same network representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Junction:
    """Represents a demand junction in the network."""

    id: str
    elevation: float
    demand: float  # Flow demand at this junction
    pattern: str


@dataclass
class Pipe:
    """Represents a pipe segment connecting two nodes."""

    id: str
    node1: str
    node2: str
    length: float
    diameter: float  # Decision variable - will be optimized
    roughness: float  # Hazen-Williams coefficient
    minor_loss: float
    status: str


@dataclass
class Reservoir:
    """Represents a water supply source."""

    id: str
    head: float  # Pressure head
    pattern: str


class WaterNetwork:
    """Represents a complete water distribution network."""

    def __init__(self):
        self.junctions: Dict[str, Junction] = {}
        self.pipes: Dict[str, Pipe] = {}
        self.reservoirs: Dict[str, Reservoir] = {}

        # Ordered list for optimizer representation.
        self.pipes_list: List[Pipe] = []

    def get_pipe_count(self) -> int:
        """Return number of optimizable pipes in network."""

        return len(self.pipes_list)

    def get_total_demand(self) -> float:
        """Return total network demand."""

        return sum(j.demand for j in self.junctions.values())

    def get_network_stats(self) -> Dict:
        """Return basic network statistics."""

        return {
            "num_junctions": len(self.junctions),
            "num_pipes": len(self.pipes),
            "num_reservoirs": len(self.reservoirs),
            "total_demand": self.get_total_demand(),
            "total_pipe_length": sum(p.length for p in self.pipes_list),
        }


def parse_inp_file(filepath: str) -> WaterNetwork:
    """
    Parse an EPANET .inp file and return a WaterNetwork object.
    """

    network = WaterNetwork()

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_section = None

    for raw_line in lines:
        line = raw_line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(";"):
            continue

        # Section headers: [JUNCTIONS], [PIPES], ...
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].upper()
            continue

        if current_section == "JUNCTIONS":
            parts = line.split()
            if len(parts) >= 3:
                try:
                    junction = Junction(
                        id=parts[0],
                        elevation=float(parts[1]),
                        demand=float(parts[2]),
                        pattern=parts[3] if len(parts) > 3 else "",
                    )
                    network.junctions[junction.id] = junction
                except (ValueError, IndexError):
                    pass

        elif current_section == "PIPES":
            parts = line.split()
            if len(parts) >= 6:
                try:
                    pipe = Pipe(
                        id=parts[0],
                        node1=parts[1],
                        node2=parts[2],
                        length=float(parts[3]),
                        diameter=float(parts[4]),
                        roughness=float(parts[5]),
                        minor_loss=float(parts[6]) if len(parts) > 6 else 0.0,
                        status=parts[7] if len(parts) > 7 else "open",
                    )
                    network.pipes[pipe.id] = pipe
                    network.pipes_list.append(pipe)
                except (ValueError, IndexError):
                    pass

        elif current_section == "RESERVOIRS":
            parts = line.split()
            if len(parts) >= 2:
                try:
                    reservoir = Reservoir(
                        id=parts[0],
                        head=float(parts[1]),
                        pattern=parts[2] if len(parts) > 2 else "",
                    )
                    network.reservoirs[reservoir.id] = reservoir
                except (ValueError, IndexError):
                    pass

    return network

