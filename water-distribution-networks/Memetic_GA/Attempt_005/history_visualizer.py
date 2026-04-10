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

    def _collect_group_runs(self, network_file: str, algorithm: str, latest_only: bool, latest_limit: int) -> List[Dict]:
        rows = self.persistence.list_runs(limit=5000)
        selected = [
            r for r in rows
            if r.network_file == network_file and r.algorithm == algorithm and r.status in ("completed", "stopped", "running")
        ]
        if latest_only:
            selected = selected[: max(1, int(latest_limit))]

        run_payloads: List[Dict] = []
        for r in selected:
            gens = self.persistence.load_generations(r.run_id)
            y = [
                float(g["best_paper_score"])
                for g in gens
                if g.get("best_paper_score") is not None and np.isfinite(float(g["best_paper_score"]))
            ]
            if not y:
                continue
            run_payloads.append({"run_id": r.run_id, "y": y})
        return run_payloads

    def plot_group_run_statistics(
        self,
        network_file: str,
        algorithm: str,
        latest_only: bool,
        latest_limit: int = 12,
    ) -> Optional[Path]:
        run_payloads = self._collect_group_runs(network_file, algorithm, latest_only, latest_limit)
        if len(run_payloads) < 2:
            return None

        min_len = min(len(r["y"]) for r in run_payloads)
        if min_len < 2:
            return None

        cropped = np.array([r["y"][:min_len] for r in run_payloads], dtype=float)
        x = np.arange(1, min_len + 1)

        median_curve = np.median(cropped, axis=0)
        p25 = np.percentile(cropped, 25, axis=0)
        p75 = np.percentile(cropped, 75, axis=0)

        finals = cropped[:, -1]
        best_idx = int(np.argmin(finals))
        worst_idx = int(np.argmax(finals))

        best_curve = cropped[best_idx]
        worst_curve = cropped[worst_idx]
        best_run_id = run_payloads[best_idx]["run_id"]
        worst_run_id = run_payloads[worst_idx]["run_id"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(x, p25, p75, color="#93c5fd", alpha=0.35, label="IQR (25-75%)")
        ax.plot(x, median_curve, color="#1d4ed8", linewidth=2.6, label="Median")
        ax.plot(x, best_curve, color="#15803d", linewidth=2.0, linestyle="-", label=f"Best run ({best_run_id})")
        ax.plot(x, worst_curve, color="#b91c1c", linewidth=2.0, linestyle="-", label=f"Worst run ({worst_run_id})")

        ax.set_xlabel("Generation (cropped to common min length)")
        ax.set_ylabel("Best strict paper score")
        ax.set_yscale("log")
        scope = f"latest {min(latest_limit, len(run_payloads))}" if latest_only else "all"
        ax.set_title(
            f"Run Statistics | {network_file} | {ALGO_LABEL.get(algorithm, algorithm)} | {scope} runs | n={len(run_payloads)}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        safe_net = network_file.replace(".inp", "")
        out_name = (
            f"10_stats_latest_{safe_net}_{algorithm}.png"
            if latest_only
            else f"11_stats_all_{safe_net}_{algorithm}.png"
        )
        out = self.output_dir / out_name
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out
