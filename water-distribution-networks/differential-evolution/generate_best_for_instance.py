from __future__ import annotations

from pathlib import Path
import sys
import json
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def load_best_vector(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty best vector file: {path}")
    for line in text.splitlines():
        s = line.strip()
        if s:
            parts = s.split(",")
            return np.array([float(p) for p in parts], dtype=float)
    raise ValueError("No data in best vector file")


def snap_to_allowed(candidate: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    distances = np.abs(candidate[:, None] - allowed[None, :])
    idx = np.argmin(distances, axis=1)
    return allowed[idx]


def generate_for_results(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results folder not found: {results_dir}")

    exp_meta = json.loads(
        (results_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    de_conf = exp_meta.get("de_config", {})
    published_best_cost = float(exp_meta.get("published_best_cost", 0.0))

    root = results_dir.parent
    ref_path = root / "published_reference_scores.json"
    all_refs = json.loads(ref_path.read_text(encoding="utf-8"))
    instance_name = exp_meta.get("instance")
    bin_ref = all_refs.get(instance_name)
    if not bin_ref:
        raise SystemExit(f"No published reference for {instance_name}")
    allowed = np.array(
        [float(item["diameter_m"]) for item in bin_ref["diameter_set"]], dtype=float
    )

    # find best run id from run_summaries.csv if available
    run_cost = None
    summaries_path = results_dir / "run_summaries.csv"
    if summaries_path.exists():
        for line in summaries_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    rid = int(float(parts[0]))
                    cost = float(parts[2])
                except Exception:
                    continue
                if run_cost is None or cost < run_cost:
                    run_cost = cost
                    best_run_id = rid

    files = sorted(results_dir.glob("run_*_best_vectors.csv"))
    if not files:
        raise SystemExit("No best vector files found")
    best_vec_path = files[0]

    vec = load_best_vector(best_vec_path)
    snapped = snap_to_allowed(vec, allowed)

    # parse inp file for coordinates and pipe order
    from inp_parser import InpFileParser

    # instance inp is located in parent data folder
    inp_path = Path(__file__).resolve().parents[1] / "data" / instance_name
    parsed = InpFileParser(inp_path).parse()
    pipe_entries = parsed.pipes.entries if parsed.pipes else []

    positions = {}
    if parsed.coordinates and parsed.coordinates.entries:
        for c in parsed.coordinates.entries:
            positions[c.node] = (float(c.x), float(c.y))

    if not positions:
        fallback_nodes = set()
        for p in pipe_entries:
            fallback_nodes.add(p.node1)
            fallback_nodes.add(p.node2)
        names = sorted(list(fallback_nodes))
        n = len(names)
        for i, name in enumerate(names):
            theta = 2 * math.pi * i / max(1, n)
            positions[name] = (math.cos(theta), math.sin(theta))

    segments = []
    dia_vals = []
    for i, p in enumerate(pipe_entries):
        n1 = p.node1
        n2 = p.node2
        if n1 not in positions or n2 not in positions:
            continue
        x1, y1 = positions[n1]
        x2, y2 = positions[n2]
        segments.append([(x1, y1), (x2, y2)])
        dia = float(snapped[i]) if i < len(snapped) else float(snapped[-1])
        dia_vals.append(dia)

    if not segments:
        raise SystemExit("No segments to plot")

    dia_arr = np.array(dia_vals, dtype=float)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=float(np.min(allowed)), vmax=float(np.max(allowed)))
    colors = [cmap(norm(d)) for d in dia_arr]
    minw = 0.5
    maxw = 6.0
    if np.max(dia_arr) - np.min(dia_arr) < 1e-9:
        widths = [2.0 for _ in dia_arr]
    else:
        widths = list(
            minw
            + (dia_arr - dia_arr.min())
            / (dia_arr.max() - dia_arr.min())
            * (maxw - minw)
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=1)
    ax.add_collection(lc)
    node_x = [positions[n][0] for n in positions]
    node_y = [positions[n][1] for n in positions]
    ax.scatter(node_x, node_y, s=10, c="k", zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pipe diameter (m)")

    pop = de_conf.get("population_size")
    gens = de_conf.get("generations")
    F = de_conf.get("mutation_factor")
    CR = de_conf.get("crossover_rate")

    info_lines = [
        f"instance: {instance_name}",
        f"pop_size: {pop}",
        f"generations: {gens}",
        f"mutation (F): {F * 100:.0f}%",
        f"crossover (CR): {CR * 100:.0f}%",
    ]
    if run_cost is not None and published_best_cost > 0:
        distance_pct = max(0.0, ((run_cost / published_best_cost) - 1.0) * 100.0)
        info_lines.append(f"distance from published best: {distance_pct:.2f}%")
    else:
        info_lines.append("distance from published best: n/a")

    ax.text(
        0.01,
        0.01,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    out = results_dir / "best_run_solution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved best-run visualization to: {out}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Usage: generate_best_for_instance.py <results_dir>")
        sys.exit(1)
    generate_for_results(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
