from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import messagebox, ttk

from fitness_evaluator import FitnessEvaluator
from network_parser import parse_inp_file
from persistence import RunPersistence


class SolutionEvolutionViewer(tk.Toplevel):
    def __init__(self, parent: tk.Misc, base_dir: Path, persistence: RunPersistence, run_id: str):
        super().__init__(parent)
        self.title(f"Solution Evolution Viewer - {run_id}")
        self.geometry("1400x900")

        self.base_dir = Path(base_dir)
        self.persistence = persistence
        self.run_id = run_id
        self.run_row = self.persistence.load_run(run_id)
        if not self.run_row:
            messagebox.showerror("Solution Viewer", f"Run {run_id} was not found.")
            self.destroy()
            return

        self.config_data = self.run_row.get("config", {})
        self.network_file = str(self.run_row.get("network_file", ""))
        self.network_path = self.base_dir / "data" / self.network_file
        self.network = parse_inp_file(str(self.network_path))

        # Use the persisted benchmark catalog for consistent diameter colors.
        self.diameter_values = self._load_diameter_catalog()
        self.evaluator = FitnessEvaluator(
            self.network,
            diameter_options=self.diameter_values,
            unit_cost_lookup=self._load_unit_cost_lookup(),
        )

        self.generations = self.persistence.load_generations(run_id)
        self.generations = [g for g in self.generations if g.get("best_chromosome") is not None]
        if not self.generations:
            messagebox.showwarning(
                "Solution Viewer",
                "This run does not have stored chromosome snapshots yet.\n"
                "Start a new run after the persistence update to use the slider viewer.",
            )
            self.destroy()
            return

        self._render_cache: Dict[int, Dict[str, Any]] = {}
        self._positions, self._node_values = self._load_node_layout()
        self._updating_slider = False
        self._slider_job: Optional[str] = None
        self._build_widgets()
        self._draw_generation(0)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        self.summary_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.summary_var, anchor="w").pack(fill="x")

        control = ttk.Frame(self)
        control.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Button(control, text="Prev", command=self._prev_generation).pack(side="left")
        ttk.Button(control, text="Next", command=self._next_generation).pack(side="left", padx=(6, 0))

        self.gen_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            control,
            from_=0,
            to=max(0, len(self.generations) - 1),
            orient="horizontal",
            command=self._on_slider_move,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=10)

        self.gen_label = ttk.Label(control, text="Gen 0")
        self.gen_label.pack(side="left")

        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [4.5, 1.7]})
        self.network_ax = self.axes[0]
        self.info_ax = self.axes[1]
        self.info_ax.axis("off")
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _load_diameter_catalog(self) -> List[float]:
        try:
            from test_benchmarks import BenchmarkRunner

            results_dir = self.base_dir / "Memetic_GA" / "Attempt_004" / "results"
            runner = BenchmarkRunner(str(self.base_dir / "data"), str(results_dir))
            diameter_options, _ = runner._get_benchmark_cost_spec(self.network_file)
            return [float(d) for d in diameter_options]
        except Exception:
            return []

    def _load_unit_cost_lookup(self) -> Optional[Dict[float, float]]:
        try:
            from test_benchmarks import BenchmarkRunner

            results_dir = self.base_dir / "Memetic_GA" / "Attempt_004" / "results"
            runner = BenchmarkRunner(str(self.base_dir / "data"), str(results_dir))
            _, unit_cost_lookup = runner._get_benchmark_cost_spec(self.network_file)
            return unit_cost_lookup if unit_cost_lookup else None
        except Exception:
            return None

    def _load_node_layout(self) -> Tuple[Dict[str, tuple[float, float]], Dict[str, float]]:
        try:
            import wntr

            wn = wntr.network.WaterNetworkModel(str(self.network_path))
            positions: Dict[str, tuple[float, float]] = {}
            values: Dict[str, float] = {}
            for name in wn.node_name_list:
                node = wn.get_node(name)
                coords = getattr(node, "coordinates", None)
                if coords and len(coords) >= 2:
                    positions[name] = (float(coords[0]), float(coords[1]))
                elevation = getattr(node, "elevation", None)
                if elevation is not None:
                    values[name] = float(elevation)
            return positions, values
        except Exception:
            return {}, {}

    def _on_slider_move(self, value: str) -> None:
        if self._updating_slider:
            return
        idx = int(round(float(value)))
        if self._slider_job is not None:
            try:
                self.after_cancel(self._slider_job)
            except Exception:
                pass
        self._slider_job = self.after(35, lambda: self._draw_generation(idx))

    def _prev_generation(self) -> None:
        idx = max(0, int(round(self.slider.get())) - 1)
        self.slider.set(idx)
        self._draw_generation(idx)

    def _next_generation(self) -> None:
        idx = min(len(self.generations) - 1, int(round(self.slider.get())) + 1)
        self.slider.set(idx)
        self._draw_generation(idx)

    def _chromosome_to_diameters(self, chromosome: List[int]) -> List[float]:
        return self.evaluator.indices_to_diameters([int(g) for g in chromosome])

    def _create_legend_elements(self, dia_norm, dia_cmap, node_norm, node_cmap):
        """Create visual legend elements for colormaps."""
        legend_elements = []
        
        # Pipe diameter legend
        pipe_samples = np.linspace(dia_norm.vmin, dia_norm.vmax, 5)
        pipe_colors = [dia_cmap(dia_norm(val)) for val in pipe_samples]
        legend_elements.append(Line2D([0], [0], color='white', label='Pipe diameter (color scale):'))
        for i, (val, color) in enumerate(zip(pipe_samples, pipe_colors)):
            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=4, 
                       label=f'  {val:.1f}' + (' (min)' if i == 0 else ' (max)' if i == len(pipe_samples)-1 else ''))
            )
        
        legend_elements.append(Line2D([0], [0], color='white', label=''))  # spacer
        
        # Node elevation legend
        node_samples = np.linspace(node_norm.vmin, node_norm.vmax, 5)
        node_colors = [node_cmap(node_norm(val)) for val in node_samples]
        legend_elements.append(Line2D([0], [0], color='white', label='Node elevation (color scale):'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=node_colors[0], markersize=8,
                                     label=f'  {node_samples[0]:.1f}m (low)'))
        for i in range(1, len(node_samples)-1):
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=node_colors[i], markersize=8,
                      label=f'  {node_samples[i]:.1f}m')
            )
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=node_colors[-1], markersize=8,
                  label=f'  {node_samples[-1]:.1f}m (high)')
        )
        
        return legend_elements

    def _chromosome_to_diameters(self, chromosome: List[int]) -> List[float]:
        return self.evaluator.indices_to_diameters([int(g) for g in chromosome])

    def _render_snapshot(self, generation_row: Dict[str, Any]) -> Dict[str, Any]:
        generation = int(generation_row["generation"])
        cached = self._render_cache.get(generation)
        if cached is not None:
            return cached

        chromosome = generation_row.get("best_chromosome") or []
        diameters = self._chromosome_to_diameters(chromosome)
        payload = {
            "chromosome": chromosome,
            "diameters": diameters,
        }
        self._render_cache[generation] = payload
        return payload

    def _draw_generation(self, index: int) -> None:
        index = max(0, min(index, len(self.generations) - 1))
        self._updating_slider = True
        try:
            self.slider.set(index)
        finally:
            self._updating_slider = False
        self.gen_label.configure(text=f"Gen {index + 1}/{len(self.generations)}")

        row = self.generations[index]
        snapshot = self._render_snapshot(row)
        diameters = snapshot["diameters"]
        generation = int(row["generation"])

        self.fig.clf()
        self.axes = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [4.5, 1.7]})
        self.network_ax = self.axes[0]
        self.info_ax = self.axes[1]
        self.info_ax.axis("off")

        # Build static color scales. Pipes are colored by diameter; nodes are colored by elevation.
        dia_norm = Normalize(vmin=float(np.min(self.diameter_values)), vmax=float(np.max(self.diameter_values)))
        dia_cmap = plt.get_cmap("turbo")
        node_values = self._node_values
        if node_values:
            node_vals_arr = np.array(list(node_values.values()), dtype=float)
            node_norm = Normalize(vmin=float(np.min(node_vals_arr)), vmax=float(np.max(node_vals_arr)))
        else:
            node_norm = Normalize(vmin=0.0, vmax=1.0)
        node_cmap = plt.get_cmap("coolwarm")

        # Draw pipes in one collection for speed.
        segments = []
        colors = []
        widths = []
        for pipe, diameter in zip(self.network.pipes_list, diameters):
            if pipe.node1 not in self._positions or pipe.node2 not in self._positions:
                continue
            x1, y1 = self._positions[pipe.node1]
            x2, y2 = self._positions[pipe.node2]
            segments.append([(x1, y1), (x2, y2)])
            colors.append(dia_cmap(dia_norm(float(diameter))))
            widths.append(1.1 + 3.0 * dia_norm(float(diameter)))

        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.95, zorder=1)
            self.network_ax.add_collection(lc)

        # Draw nodes colored by elevation. This is fast and still gives a useful
        # spatial cue for how the solution evolves over generations.
        node_x = []
        node_y = []
        node_c = []
        node_sizes = []
        for node_id, (x, y) in self._positions.items():
            node_x.append(x)
            node_y.append(y)
            node_c.append(float(node_values.get(node_id, np.nan)))
            node_sizes.append(26 if node_id not in self.network.reservoirs else 54)

        node_c_arr = np.array(node_c, dtype=float)
        junction_mask = np.isfinite(node_c_arr)
        if np.any(junction_mask):
            sc = self.network_ax.scatter(
                np.array(node_x)[junction_mask],
                np.array(node_y)[junction_mask],
                c=node_c_arr[junction_mask],
                cmap=node_cmap,
                norm=node_norm,
                s=np.array(node_sizes)[junction_mask],
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
            )
        else:
            sc = None
            self.network_ax.scatter(
                node_x,
                node_y,
                c="lightgray",
                s=node_sizes,
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
            )

        self.network_ax.set_title(f"{self.network_file} - Gen {generation}")
        self.network_ax.set_aspect("equal", adjustable="datalim")
        self.network_ax.axis("off")

        # Colorbars.
        sm_dia = ScalarMappable(norm=dia_norm, cmap=dia_cmap)
        sm_dia.set_array([])
        cbar_dia = self.fig.colorbar(sm_dia, ax=self.network_ax, fraction=0.03, pad=0.01)
        cbar_dia.set_label("Pipe diameter")

        if sc is not None:
            cbar_pr = self.fig.colorbar(sc, ax=self.network_ax, fraction=0.03, pad=0.06)
            cbar_pr.set_label("Node elevation")

        # Add visual legend to network plot
        legend_elements = self._create_legend_elements(dia_norm, dia_cmap, node_norm, node_cmap)
        self.network_ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.95)

        # Info panel.
        gap = row.get("gap_to_published_pct")
        gap_text = "n/a" if gap is None else f"{float(gap):+.2f}%"
        summary_lines = [
            f"Run: {self.run_id}",
            f"Algorithm: {self.run_row.get('algorithm', '')}",
            f"Generation: {generation}",
            f"Training fitness: {float(row.get('best_training_fitness', float('nan'))):.3e}",
            f"Paper score: {'inf' if row.get('best_paper_score') is None else f'{float(row.get('best_paper_score')):.3e}'}",
            f"Paper cost: {'inf' if row.get('best_paper_cost') is None else f'{float(row.get('best_paper_cost')):.3e}'}",
            f"Feasible: {'yes' if float(row.get('best_paper_feasible', 0.0) or 0.0) > 0.5 else 'no'}",
            f"Feasible in pop: {row.get('feasible_count', 'n/a')}",
            f"Best feasible delta vs published reference: {gap_text}",
            "",
            "Metrics shown:",
            "- Pipe color: diameter",
            "- Node color: elevation",
            "- Slider: generation",
        ]
        self.info_ax.text(
            0.0,
            1.0,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        self.summary_var.set(
            f"{self.network_file} | gen={generation} | paper={'inf' if row.get('best_paper_score') is None else f'{float(row.get('best_paper_score')):.3e}'} | feasible={'yes' if float(row.get('best_paper_feasible', 0.0) or 0.0) > 0.5 else 'no'}"
        )
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()