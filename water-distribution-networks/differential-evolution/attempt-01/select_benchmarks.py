"""
Select benchmark instances (easy/medium/hard) from `data/`.

We use a pragmatic selection based on the number of decision variables
(pipes), since the simplified surrogate objective evaluates only per-pipe
costs/penalties.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from network_parser import parse_inp_file


def select_easy_medium_hard(
    data_dir: str,
    *,
    max_pipes: int = 1000,
    min_junctions: int = 1,
    medium_low: int = 20,
    medium_high: int = 100,
) -> Tuple[Dict, Dict, Dict]:
    """
    Returns three dicts: (easy, medium, hard).
    """

    p = Path(data_dir)
    files = sorted([f for f in p.iterdir() if f.suffix == ".inp"])
    candidates: List[Dict] = []

    for fp in files:
        try:
            net = parse_inp_file(str(fp))
            stats = net.get_network_stats()
            if 0 < stats["num_pipes"] < max_pipes and stats["num_junctions"] >= min_junctions:
                candidates.append(
                    {
                        "filename": fp.name,
                        "num_pipes": stats["num_pipes"],
                        "num_junctions": stats["num_junctions"],
                        "total_demand": stats["total_demand"],
                        "total_pipe_length": stats["total_pipe_length"],
                    }
                )
        except Exception:
            # Some instances in the dataset set might not match the simplified parser.
            pass

    if len(candidates) < 3:
        raise RuntimeError(f"Not enough valid benchmark instances found in {data_dir}. Found {len(candidates)}.")

    candidates.sort(key=lambda d: d["num_pipes"])

    easy = candidates[0]
    # Prefer an actually "medium" sized network if possible.
    medium_candidates = [c for c in candidates if medium_low <= c["num_pipes"] <= medium_high]
    if medium_candidates:
        medium = medium_candidates[0]
    else:
        # Fallback: median by pipe count among candidates.
        medium = candidates[len(candidates) // 2]

    hard = candidates[-1]
    return easy, medium, hard


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    easy, medium, hard = select_easy_medium_hard(str(data_dir))

    print("✓ Selected benchmarks:")
    for d in [easy, medium, hard]:
        print(f"  - {d['filename']}: {d['num_pipes']} pipes, {d['num_junctions']} junctions")


if __name__ == "__main__":
    main()

