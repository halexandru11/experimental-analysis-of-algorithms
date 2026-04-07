"""
Parser for EPANET water distribution network files (.inp format).

This is a small, self-contained parser used by Attempt 003.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Junction:
    id: str
    elevation: float
    demand: float
    pattern: str


@dataclass
class Pipe:
    id: str
    node1: str
    node2: str
    length: float
    diameter: float
    roughness: float
    minor_loss: float
    status: str


@dataclass
class Reservoir:
    id: str
    head: float
    pattern: str


class WaterNetwork:
    def __init__(self):
        self.junctions: Dict[str, Junction] = {}
        self.pipes: Dict[str, Pipe] = {}
        self.reservoirs: Dict[str, Reservoir] = {}
        self.pipes_list: List[Pipe] = []

    def get_pipe_count(self) -> int:
        return len(self.pipes_list)

    def get_total_demand(self) -> float:
        return sum(j.demand for j in self.junctions.values())

    def get_network_stats(self) -> Dict[str, float]:
        return {
            'num_junctions': len(self.junctions),
            'num_pipes': len(self.pipes_list),
            'num_reservoirs': len(self.reservoirs),
            'total_demand': self.get_total_demand(),
            'total_pipe_length': sum(p.length for p in self.pipes_list),
        }


def parse_inp_file(filepath: str) -> WaterNetwork:
    network = WaterNetwork()

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as handle:
        lines = handle.readlines()

    current_section = None

    for raw_line in lines:
        line = raw_line.strip()

        if not line or line.startswith(';'):
            continue

        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].upper()
            continue

        if current_section == 'JUNCTIONS':
            parts = line.split()
            if len(parts) >= 3:
                try:
                    junction = Junction(
                        id=parts[0],
                        elevation=float(parts[1]),
                        demand=float(parts[2]),
                        pattern=parts[3] if len(parts) > 3 else '',
                    )
                    network.junctions[junction.id] = junction
                except (ValueError, IndexError):
                    pass

        elif current_section == 'PIPES':
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
                        status=parts[7] if len(parts) > 7 else 'open',
                    )
                    network.pipes[pipe.id] = pipe
                    network.pipes_list.append(pipe)
                except (ValueError, IndexError):
                    pass

        elif current_section == 'RESERVOIRS':
            parts = line.split()
            if len(parts) >= 2:
                try:
                    reservoir = Reservoir(
                        id=parts[0],
                        head=float(parts[1]),
                        pattern=parts[2] if len(parts) > 2 else '',
                    )
                    network.reservoirs[reservoir.id] = reservoir
                except (ValueError, IndexError):
                    pass

    return network