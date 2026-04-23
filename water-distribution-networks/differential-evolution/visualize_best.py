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
    # file may contain one line or multiple; take first non-empty line
    for line in text.splitlines():
        s = line.strip()
        if s:
            parts = s.split(",")
            return np.array([float(p) for p in parts], dtype=float)
    raise ValueError("No data in best vector file")


def snap_to_allowed(candidate: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    # choose nearest allowed diameter per entry
    distances = np.abs(candidate[:, None] - allowed[None, :])
    idx = np.argmin(distances, axis=1)
    return allowed[idx]


def main() -> None:
    root = Path(__file__).resolve().parent
    results_dir = root / "results" / "BIN-3000-50-50-80"
    inp_path = root.parent / "data" / "BIN.inp"

    if not results_dir.exists():
        print(f"Results folder not found: {results_dir}")
        sys.exit(1)
    if not inp_path.exists():
        print(f"INP file not found: {inp_path}")
        sys.exit(1)

    # read experiment config for meta info
    exp_meta = json.loads(
        (results_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    de_conf = exp_meta.get("de_config", {})
    published_best_cost = float(exp_meta.get("published_best_cost", 0.0))

    # load allowed diameters from published_reference_scores.json
    ref_path = root / "results" / "published_reference_scores.json"
    if not ref_path.exists():
        raise SystemExit("published_reference_scores.json missing")
    all_refs = json.loads(ref_path.read_text(encoding="utf-8"))
    bin_ref = all_refs.get("BIN.inp")
    if not bin_ref:
        raise SystemExit("BIN entry missing in published_reference_scores.json")
    allowed = np.array(
        [float(item["diameter_m"]) for item in bin_ref["diameter_set"]], dtype=float
    )

    # prepare list of best vector files
    best_files = sorted(results_dir.glob("run_*_best_vector.csv"))
    if not best_files:
        print("No run best vector files found")
        sys.exit(1)

    # load run_summaries mapping run_id -> best_cost
    run_summaries_map = {}
    summaries_path = results_dir / "run_summaries.csv"
    if summaries_path.exists():
        for line in summaries_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    rid = int(float(parts[0]))
                    cost = float(parts[2])
                    run_summaries_map[rid] = cost
                except Exception:
                    continue

    # use local inp parser to get pipe order and coordinates
    sys.path.insert(0, str(root))
    from inp_parser import InpFileParser

    parsed = InpFileParser(inp_path).parse()
    pipe_entries = parsed.pipes.entries if parsed.pipes else []

    # build positions mapping from COORDINATES section if present
    positions = {}
    if parsed.coordinates and parsed.coordinates.entries:
        for c in parsed.coordinates.entries:
            positions[c.node] = (float(c.x), float(c.y))

    # fallback node set for circular layout
    fallback_nodes = set()
    for p in pipe_entries:
        fallback_nodes.add(p.node1)
        fallback_nodes.add(p.node2)

    for best_vec_path in best_files:
        vec = load_best_vector(best_vec_path)
        snapped = snap_to_allowed(vec, allowed)

        # ensure positions exist; if not, create circular layout of fallback nodes
        local_positions = dict(positions)
        if not local_positions:
            names = sorted(list(fallback_nodes))
            n = len(names)
            for i, name in enumerate(names):
                theta = 2 * math.pi * i / max(1, n)
                local_positions[name] = (math.cos(theta), math.sin(theta))

        segments = []
        dia_vals = []
        for i, p in enumerate(pipe_entries):
            n1 = p.node1
            n2 = p.node2
            if n1 not in local_positions or n2 not in local_positions:
                continue
            x1, y1 = local_positions[n1]
            x2, y2 = local_positions[n2]
            segments.append([(x1, y1), (x2, y2)])
            dia = float(snapped[i]) if i < len(snapped) else float(snapped[-1])
            dia_vals.append(dia)

        if not segments:
            print(f"No segments to plot for {best_vec_path.name}")
            continue

        dia_arr = np.array(dia_vals, dtype=float)

        cmap = plt.get_cmap("turbo")
        norm = plt.Normalize(vmin=float(np.min(allowed)), vmax=float(np.max(allowed)))
        colors = [cmap(norm(d)) for d in dia_arr]
        # thickness scaled by diameter
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

        # draw nodes
        node_x = []
        node_y = []
        for n, (x, y) in local_positions.items():
            node_x.append(x)
            node_y.append(y)
        ax.scatter(node_x, node_y, s=10, c="k", zorder=2)

        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

        # colorbar for diameters
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Pipe diameter (m)")

        # annotate with experiment meta and distance from published best
        run_name = best_vec_path.stem
        try:
            run_id = int(run_name.split("_")[1])
        except Exception:
            run_id = None

        this_cost = run_summaries_map.get(run_id)
        distance_pct = None
        if this_cost is not None and published_best_cost > 0:
            distance_pct = max(0.0, ((this_cost / published_best_cost) - 1.0) * 100.0)

        pop = de_conf.get("population_size")
        gens = de_conf.get("generations")
        F = de_conf.get("mutation_factor")
        CR = de_conf.get("crossover_rate")

        info_lines = [
            f"instance: {exp_meta.get('instance')}",
            f"pop_size: {pop}",
            f"generations: {gens}",
            f"mutation (F): {F * 100:.0f}%",
            f"crossover (CR): {CR * 100:.0f}%",
        ]
        if distance_pct is not None:
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

        out = results_dir / f"{run_name}_solution.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved visualization to: {out}")


if __name__ == "__main__":
    main()
