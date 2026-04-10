import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persistence import RunPersistence


NETWORK_ORDER = ["TLN.inp", "hanoi.inp", "BIN.inp"]
ALGO_ORDER = ["memetic", "standard"]
ALGO_LABEL = {
    "memetic": "Memetic GA",
    "standard": "Standard GA",
}
ALGO_COLOR = {
    "memetic": "#2E86AB",
    "standard": "#C73E1D",
}


class HistoryVisualizer:
    def __init__(self, persistence: RunPersistence, output_dir: Path, reference_scores_path: Path):
        self.persistence = persistence
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_scores = self._load_reference_scores(reference_scores_path)

    @staticmethod
    def _load_reference_scores(path: Path) -> Dict[str, Dict]:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _latest_runs_snapshot(self) -> Dict[str, Dict[str, Optional[Dict]]]:
        snapshot: Dict[str, Dict[str, Optional[Dict]]] = {}
        for net in NETWORK_ORDER:
            snapshot[net] = {
                "memetic": self.persistence.latest_completed_run_for_network_algorithm(net, "memetic"),
                "standard": self.persistence.latest_completed_run_for_network_algorithm(net, "standard"),
                "latest_any": self.persistence.latest_completed_run_for_network(net),
            }
        return snapshot

    def _load_generation_series(self, run: Optional[Dict]) -> List[Dict]:
        if not run:
            return []
        return self.persistence.load_generations(run["run_id"])

    def _best_finite_paper_score(self, gens: List[Dict]) -> Optional[float]:
        vals = [g.get("best_paper_score") for g in gens if g.get("best_paper_score") is not None]
        finite = [float(v) for v in vals if np.isfinite(float(v))]
        if not finite:
            return None
        return float(min(finite))

    def _last_finite_gap(self, gens: List[Dict]) -> Optional[float]:
        vals = [g.get("gap_to_published_pct") for g in gens if g.get("gap_to_published_pct") is not None]
        finite = [float(v) for v in vals if np.isfinite(float(v))]
        if not finite:
            return None
        return float(finite[-1])

    def plot_convergence(self, snapshot: Dict[str, Dict[str, Optional[Dict]]]) -> Optional[Path]:
        fig, axes = plt.subplots(1, len(NETWORK_ORDER), figsize=(18, 5))
        if len(NETWORK_ORDER) == 1:
            axes = [axes]

        any_plotted = False

        for i, net in enumerate(NETWORK_ORDER):
            ax = axes[i]
            for algo in ALGO_ORDER:
                run = snapshot[net][algo]
                gens = self._load_generation_series(run)
                if not gens:
                    continue
                x = [int(g["generation"]) for g in gens]
                y_train = [float(g["best_training_fitness"]) for g in gens]
                y_paper = [float(g["best_paper_score"]) for g in gens]

                ax.plot(x, y_train, linestyle="--", linewidth=1.5, color=ALGO_COLOR[algo], alpha=0.65,
                        label=f"{ALGO_LABEL[algo]} train")
                ax.plot(x, y_paper, linestyle="-", linewidth=2.0, color=ALGO_COLOR[algo],
                        label=f"{ALGO_LABEL[algo]} strict")
                any_plotted = True

            ax.set_title(net)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Score")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

        if not any_plotted:
            plt.close(fig)
            return None

        fig.suptitle("Attempt_005 History: Convergence (Latest Completed Runs)", fontsize=12)
        fig.tight_layout()
        out = self.output_dir / "01_history_convergence_curves.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_algorithm_comparison(self, snapshot: Dict[str, Dict[str, Optional[Dict]]]) -> Optional[Path]:
        labels = []
        memetic_scores = []
        standard_scores = []

        for net in NETWORK_ORDER:
            mg = self._load_generation_series(snapshot[net]["memetic"])
            sg = self._load_generation_series(snapshot[net]["standard"])
            if not mg and not sg:
                continue

            labels.append(net)
            memetic_scores.append(self._best_finite_paper_score(mg))
            standard_scores.append(self._best_finite_paper_score(sg))

        if not labels:
            return None

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))

        mg_vals = [v if v is not None else np.nan for v in memetic_scores]
        sg_vals = [v if v is not None else np.nan for v in standard_scores]

        ax.bar(x - width / 2, mg_vals, width, label="Memetic GA", color=ALGO_COLOR["memetic"], alpha=0.85)
        ax.bar(x + width / 2, sg_vals, width, label="Standard GA", color=ALGO_COLOR["standard"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Best strict paper score")
        ax.set_title("Attempt_005 History: Best Strict Score Comparison")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        out = self.output_dir / "02_history_algorithm_comparison.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_gap_to_sota(self, snapshot: Dict[str, Dict[str, Optional[Dict]]]) -> Optional[Path]:
        labels = []
        memetic_gaps = []
        standard_gaps = []

        for net in NETWORK_ORDER:
            mg = self._load_generation_series(snapshot[net]["memetic"])
            sg = self._load_generation_series(snapshot[net]["standard"])
            if not mg and not sg:
                continue
            labels.append(net)
            memetic_gaps.append(self._last_finite_gap(mg))
            standard_gaps.append(self._last_finite_gap(sg))

        if not labels:
            return None

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        mg_vals = [v if v is not None else np.nan for v in memetic_gaps]
        sg_vals = [v if v is not None else np.nan for v in standard_gaps]

        bars1 = ax.bar(x - width / 2, mg_vals, width, label="Memetic GA gap (%)", color=ALGO_COLOR["memetic"], alpha=0.85)
        bars2 = ax.bar(x + width / 2, sg_vals, width, label="Standard GA gap (%)", color=ALGO_COLOR["standard"], alpha=0.85)

        ax.axhline(0.0, color="black", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Gap to published best (%)")
        ax.set_title("Attempt_005 History: Gap to SOTA")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        for bars in (bars1, bars2):
            for b in bars:
                h = b.get_height()
                if np.isnan(h):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:+.1f}%", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)

        out = self.output_dir / "03_history_gap_to_sota.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_feasible_fraction(self, snapshot: Dict[str, Dict[str, Optional[Dict]]]) -> Optional[Path]:
        fig, axes = plt.subplots(1, len(NETWORK_ORDER), figsize=(18, 4))
        if len(NETWORK_ORDER) == 1:
            axes = [axes]

        any_plotted = False
        for i, net in enumerate(NETWORK_ORDER):
            ax = axes[i]
            for algo in ALGO_ORDER:
                run = snapshot[net][algo]
                gens = self._load_generation_series(run)
                if not gens:
                    continue
                pop_size = None
                if run and run.get("config"):
                    pop_size = run["config"].get("population_size")
                if not pop_size:
                    continue

                x = [int(g["generation"]) for g in gens]
                y = [float(g.get("feasible_count", 0) / max(1, int(pop_size))) for g in gens]
                ax.plot(x, y, linewidth=2, color=ALGO_COLOR[algo], label=ALGO_LABEL[algo])
                any_plotted = True

            ax.set_title(net)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Feasible fraction in population")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

        if not any_plotted:
            plt.close(fig)
            return None

        fig.suptitle("Attempt_005 History: Feasible Population Fraction", fontsize=12)
        fig.tight_layout()
        out = self.output_dir / "04_history_feasible_fraction.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def generate_all(self) -> List[Path]:
        snapshot = self._latest_runs_snapshot()

        outputs = []
        for fn in (
            self.plot_convergence,
            self.plot_algorithm_comparison,
            self.plot_gap_to_sota,
            self.plot_feasible_fraction,
        ):
            out = fn(snapshot)
            if out is not None:
                outputs.append(out)

        return outputs
