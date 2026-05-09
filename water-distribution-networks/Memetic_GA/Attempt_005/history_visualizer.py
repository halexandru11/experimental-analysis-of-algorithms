import csv
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
STANDARD_GA_NETWORKS = {"TLN.inp", "hanoi.inp"}


class HistoryVisualizer:
    def __init__(self, persistence: RunPersistence, output_dir: Path, reference_scores_path: Path):
        self.persistence = persistence
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_scores = self._load_reference_scores(
            reference_scores_path if reference_scores_path.exists() else self.output_dir / "published_reference_scores.json"
        )
        self._live_best_scores = self._load_live_results_best_scores()

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
                "standard": self.persistence.latest_completed_run_for_network_algorithm(net, "standard")
                if net in STANDARD_GA_NETWORKS
                else None,
                "latest_any": self.persistence.latest_completed_run_for_network(net),
            }
        return snapshot

    def _load_live_results_best_scores(self) -> Dict[tuple[str, str], float]:
        live_path = self.output_dir.parent / "live_results.csv"
        if not live_path.exists():
            return {}

        best_scores: Dict[tuple[str, str], float] = {}
        with live_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                network = str(row.get("network_file", "")).strip()
                algorithm = str(row.get("algorithm", "")).strip()
                if not network or not algorithm:
                    continue
                if not self._algorithm_allowed(network, algorithm):
                    continue
                try:
                    score = float(row.get("best_paper_score"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(score):
                    continue
                key = (network, algorithm)
                current = best_scores.get(key)
                if current is None or score < current:
                    best_scores[key] = score
        return best_scores

    def _best_score_for_network_algo(
        self, network_file: str, algorithm: str, gens: List[Dict]
    ) -> Optional[float]:
        key = (network_file, algorithm)
        if key in self._live_best_scores:
            return float(self._live_best_scores[key])
        return self._best_finite_paper_score(gens)

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

    @staticmethod
    def _strict_best_so_far(values: List[float]) -> List[float]:
        best = float("inf")
        series: List[float] = []
        for value in values:
            current = float(value)
            if np.isfinite(current) and current < best:
                best = current
            series.append(best)
        return series

    @staticmethod
    def _algorithm_allowed(network_file: str, algorithm: str) -> bool:
        if algorithm == "standard" and network_file not in STANDARD_GA_NETWORKS:
            return False
        return True

    def _last_finite_gap(self, network_file: str, gens: List[Dict]) -> Optional[float]:
        ref = self.reference_scores.get(network_file, {}).get("published_best_universal_score")
        if ref is None:
            return None
        try:
            ref_value = float(ref)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(ref_value) or ref_value <= 0.0:
            return None

        best_feasible = self._best_finite_paper_score(gens)
        if best_feasible is None or not np.isfinite(best_feasible):
            return None

        return float(100.0 * (best_feasible - ref_value) / ref_value)

    def _gap_to_published_best(self, network_file: str, best_score: Optional[float]) -> Optional[float]:
        if best_score is None or not np.isfinite(float(best_score)):
            return None
        ref = self.reference_scores.get(network_file, {}).get("published_best_universal_score")
        if ref is None:
            return None
        try:
            ref_value = float(ref)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(ref_value) or ref_value <= 0.0:
            return None
        return float(100.0 * (float(best_score) - ref_value) / ref_value)

    def plot_convergence(self, snapshot: Dict[str, Dict[str, Optional[Dict]]]) -> Optional[Path]:
        fig, axes = plt.subplots(1, len(NETWORK_ORDER), figsize=(18, 5))
        if len(NETWORK_ORDER) == 1:
            axes = [axes]

        any_plotted = False

        for i, net in enumerate(NETWORK_ORDER):
            ax = axes[i]
            for algo in ALGO_ORDER:
                if not self._algorithm_allowed(net, algo):
                    continue
                run = snapshot[net][algo]
                gens = self._load_generation_series(run)
                if not gens:
                    continue
                x = [int(g["generation"]) for g in gens]
                y_train = np.array([float(g["best_training_fitness"]) for g in gens], dtype=float)

                # Build strict best-so-far series with carry-forward for non-finite values.
                best = float("inf")
                seen_finite = False
                y_paper = []
                for g in gens:
                    raw = g.get("best_paper_score")
                    if raw is not None and np.isfinite(float(raw)):
                        val = float(raw)
                        if val < best:
                            best = val
                        seen_finite = True
                    y_paper.append(best if seen_finite else float("nan"))
                y_paper = np.array(y_paper, dtype=float)

                # Mask non-finite values so lines break instead of exploding the scale.
                y_train[~np.isfinite(y_train)] = np.nan
                y_paper[~np.isfinite(y_paper)] = np.nan

                max_paper = np.nanmax(y_paper) if np.any(np.isfinite(y_paper)) else None
                max_train = np.nanmax(y_train) if np.any(np.isfinite(y_train)) else None

                if max_paper is not None:
                    ax.plot(
                        x,
                        y_paper,
                        linestyle="-",
                        linewidth=2.0,
                        color=ALGO_COLOR[algo],
                        label=f"{ALGO_LABEL[algo]} strict",
                    )

                # Skip training curve if it dwarfs strict scores (keeps scale readable).
                if max_train is not None and max_paper is not None and max_train <= max_paper * 1000:
                    ax.plot(
                        x,
                        y_train,
                        linestyle="--",
                        linewidth=1.5,
                        color=ALGO_COLOR[algo],
                        alpha=0.65,
                        label=f"{ALGO_LABEL[algo]} train",
                    )
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
            sg = self._load_generation_series(snapshot[net]["standard"]) if self._algorithm_allowed(net, "standard") else []
            if not mg and not sg:
                continue

            labels.append(net)
            memetic_scores.append(self._best_score_for_network_algo(net, "memetic", mg) if mg else None)
            standard_scores.append(
                self._best_score_for_network_algo(net, "standard", sg) if sg else None
            )

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
            sg = self._load_generation_series(snapshot[net]["standard"]) if self._algorithm_allowed(net, "standard") else []
            if not mg and not sg:
                continue
            labels.append(net)
            mg_best = self._best_score_for_network_algo(net, "memetic", mg) if mg else None
            sg_best = self._best_score_for_network_algo(net, "standard", sg) if sg else None
            memetic_gaps.append(self._gap_to_published_best(net, mg_best))
            standard_gaps.append(self._gap_to_published_best(net, sg_best))

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
        ax.set_ylabel("Delta to published reference (%)")
        ax.set_title("Attempt_005 History: Delta to Published Reference")
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
                if not self._algorithm_allowed(net, algo):
                    continue
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
        if not self._algorithm_allowed(network_file, algorithm):
            return []
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
            if not gens:
                continue

            # Build a best-so-far series across all generations, carrying forward
            # the last finite value when strict score becomes non-finite.
            best = float("inf")
            series: List[float] = []
            seen_finite = False
            for g in gens:
                raw = g.get("best_paper_score")
                if raw is not None and np.isfinite(float(raw)):
                    val = float(raw)
                    if val < best:
                        best = val
                    seen_finite = True
                series.append(best if seen_finite else float("nan"))

            if not seen_finite:
                continue

            run_payloads.append({"run_id": r.run_id, "y": series})
        return run_payloads

    def _best_run_for_group(self, network_file: str, algorithm: str) -> Optional[Dict]:
        """Return best run payload (lowest strict best score) for network+algorithm."""
        if not self._algorithm_allowed(network_file, algorithm):
            return None
        live_key = (network_file, algorithm)
        live_score = self._live_best_scores.get(live_key)
        rows = self.persistence.list_runs(limit=5000)
        selected = [
            r for r in rows
            if r.network_file == network_file and r.algorithm == algorithm and r.status in ("completed", "stopped", "running")
        ]
        best_payload: Optional[Dict] = None
        best_score = float(live_score) if live_score is not None else float("inf")

        for r in selected:
            gens = self.persistence.load_generations(r.run_id)
            y_raw = [
                float(g["best_paper_score"])
                for g in gens
                if g.get("best_paper_score") is not None and np.isfinite(float(g["best_paper_score"]))
            ]
            if not y_raw:
                continue

            strict_series = self._strict_best_so_far(y_raw)
            final_score = float(strict_series[-1]) if strict_series else float("inf")
            if not np.isfinite(final_score):
                continue

            if final_score < best_score:
                best_score = final_score
                best_payload = {
                    "run_id": r.run_id,
                    "network_file": network_file,
                    "algorithm": algorithm,
                    "final_score": final_score,
                    "gens": gens,
                }
        if live_score is not None and best_payload is None:
            best_payload = {
                "run_id": "live_results",
                "network_file": network_file,
                "algorithm": algorithm,
                "final_score": float(live_score),
                "gens": [],
            }

        return best_payload

    def plot_best_of_best_summary(self) -> Optional[Path]:
        """
        Plot best run per (network, algorithm) group with:
        - gap to published best (%)
        - final strict best score (benchmark currency)
        """
        payloads: List[Dict] = []
        for net in NETWORK_ORDER:
            for algo in ALGO_ORDER:
                if not self._algorithm_allowed(net, algo):
                    continue
                best = self._best_run_for_group(net, algo)
                if best is None:
                    continue
                gap = self._gap_to_published_best(net, best.get("final_score"))
                payloads.append({
                    **best,
                    "gap_pct": gap,
                })

        if not payloads:
            return None

        labels = [f"{p['network_file']}\n{ALGO_LABEL.get(p['algorithm'], p['algorithm'])}" for p in payloads]
        x = np.arange(len(payloads))

        fig, (ax_gap, ax_score) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: gap-to-published-best (%).
        gap_vals = [p["gap_pct"] if p.get("gap_pct") is not None else np.nan for p in payloads]
        gap_colors = [ALGO_COLOR.get(p["algorithm"], "#4b5563") for p in payloads]
        bars_gap = ax_gap.bar(x, gap_vals, color=gap_colors, alpha=0.9)
        ax_gap.axhline(0.0, color="black", linewidth=1.2)
        ax_gap.set_xticks(x)
        ax_gap.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_gap.set_ylabel("Gap to published best (%)")
        ax_gap.set_title("Best-of-Best Gap by Network and Algorithm")
        ax_gap.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars_gap, gap_vals):
            if not np.isfinite(val):
                continue
            ax_gap.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:+.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )


        # Right: final strict best score (benchmark currency).
        score_vals = [float(p["final_score"]) for p in payloads]
        score_colors = [ALGO_COLOR.get(p["algorithm"], "#4b5563") for p in payloads]
        bars_score = ax_score.bar(x, score_vals, color=score_colors, alpha=0.9)
        ax_score.set_xticks(x)
        ax_score.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_score.set_ylabel("Final strict best score (benchmark currency)")
        ax_score.set_title("Best-of-Best Final Strict Score")
        ax_score.grid(True, axis="y", alpha=0.3)
        ax_score.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        for bar, val in zip(bars_score, score_vals):
            ax_score.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

        fig.suptitle("Attempt_005 Stats: Best-of-Best Across All Runs", fontsize=12)
        fig.tight_layout()
        out = self.output_dir / "12_stats_best_of_best_summary.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    def _available_networks_for_algorithm(self, algorithm: str) -> List[str]:
        rows = self.persistence.list_runs(limit=5000)
        available = []
        for net in NETWORK_ORDER:
            if not self._algorithm_allowed(net, algorithm):
                continue
            if any(r.network_file == net and r.algorithm == algorithm and r.status in ("completed", "stopped", "running") for r in rows):
                available.append(net)
        return available

    def _plot_group_statistics_for_networks(
        self,
        algorithm: str,
        networks: List[str],
        latest_only: bool,
        latest_limit: int = 12,
    ) -> Optional[Path]:
        active_networks = []
        per_network_payloads: Dict[str, List[Dict]] = {}
        for net in networks:
            payloads = self._collect_group_runs(net, algorithm, latest_only, latest_limit)
            if len(payloads) >= 2:
                active_networks.append(net)
                per_network_payloads[net] = payloads

        if not active_networks:
            return None

        fig, axes = plt.subplots(1, len(active_networks), figsize=(6 * len(active_networks), 6), squeeze=False)
        axes = axes[0]

        for ax, net in zip(axes, active_networks):
            run_payloads = per_network_payloads[net]
            min_len = min(len(r["y"]) for r in run_payloads)
            if min_len < 2:
                ax.set_visible(False)
                continue

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

            ax.fill_between(x, p25, p75, color="#93c5fd", alpha=0.35, label="IQR (25-75%)")
            ax.plot(x, median_curve, color="#1d4ed8", linewidth=2.6, label="Median")
            ax.plot(x, best_curve, color="#15803d", linewidth=2.0, label=f"Best ({best_run_id})")
            ax.plot(x, worst_curve, color="#b91c1c", linewidth=2.0, label=f"Worst ({worst_run_id})")
            ax.set_xlabel("Generation (cropped to common min length)")
            ax.set_ylabel("Best strict paper score")
            ax.set_yscale("log")
            ax.set_title(f"{net}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes[len(active_networks):]:
            ax.set_visible(False)

        algo_label = ALGO_LABEL.get(algorithm, algorithm)
        scope = "latest" if latest_only else "all"
        fig.suptitle(
            f"Run Statistics | {algo_label} | {scope} runs | networks: {', '.join(active_networks)}",
            fontsize=12,
        )

        safe_algo = algorithm.replace(" ", "_")
        net_tag = "_".join(net.replace(".inp", "") for net in active_networks)
        out_name = (
            f"10_stats_latest_{safe_algo}_{net_tag}.png"
            if latest_only
            else f"11_stats_all_{safe_algo}_{net_tag}.png"
        )
        out = self.output_dir / out_name
        fig.tight_layout()
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

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

    def generate_all_existing_group_statistics(self, latest_only: bool = False, latest_limit: int = 12) -> List[Path]:
        outputs: List[Path] = []
        for algo in ALGO_ORDER:
            networks = self._available_networks_for_algorithm(algo)
            out = self._plot_group_statistics_for_networks(
                algorithm=algo,
                networks=networks,
                latest_only=latest_only,
                latest_limit=latest_limit,
            )
            if out is not None:
                outputs.append(out)

        # For "all runs" view, also include best-of-best summary across groups.
        if not latest_only:
            best_summary = self.plot_best_of_best_summary()
            if best_summary is not None:
                outputs.append(best_summary)
        return outputs

    def plot_selected_runs_statistics(self, run_ids: List[str]) -> List[Path]:
        selected = [str(rid) for rid in run_ids if str(rid).strip()]
        if not selected:
            return []

        payloads = []
        for run_id in selected:
            run = self.persistence.load_run(run_id)
            if not run:
                continue

            gens = self.persistence.load_generations(run_id)
            finite_scores = [
                float(g["best_paper_score"])
                for g in gens
                if g.get("best_paper_score") is not None and np.isfinite(float(g.get("best_paper_score")))
            ]
            if not finite_scores:
                continue

            y = self._strict_best_so_far(finite_scores)
            final_best = float(y[-1]) if y else float("inf")
            payloads.append(
                {
                    "run_id": run_id,
                    "network": str(run.get("network_file") or "?"),
                    "algorithm": str(run.get("algorithm") or "?"),
                    "y": y,
                    "final": final_best,
                }
            )

        if not payloads:
            return []

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_curve, ax_bar = axes

        # Left panel: strict best-so-far curves by selected run.
        for idx, p in enumerate(payloads):
            x = np.arange(1, len(p["y"]) + 1)
            color = plt.cm.tab20(idx % 20)
            label = f"{p['run_id']} | {p['network']} | {p['algorithm']}"
            ax_curve.plot(x, p["y"], linewidth=2.0, color=color, label=label)

        ax_curve.set_xlabel("Generation (finite-score points)")
        ax_curve.set_ylabel("Best strict paper score")
        ax_curve.set_yscale("log")
        ax_curve.set_title("Selected Runs: Strict Best-So-Far Convergence")
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend(fontsize=7)

        # Right panel: final strict best score for each selected run.
        ordered = sorted(payloads, key=lambda p: p["final"])
        labels = [f"{p['run_id']}\n{p['network']}|{p['algorithm']}" for p in ordered]
        values = [p["final"] for p in ordered]
        x = np.arange(len(ordered))
        bars = ax_bar.bar(x, values, color="#2E86AB", alpha=0.9)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_bar.set_ylabel("Final strict best score")
        ax_bar.set_title("Selected Runs: Final Strict Score")
        ax_bar.grid(True, axis="y", alpha=0.3)
        ax_bar.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        for b, val in zip(bars, values):
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        fig.suptitle(f"Selected Run Statistics (n={len(payloads)})", fontsize=12)
        fig.tight_layout()
        out = self.output_dir / f"12_stats_selected_runs_{len(payloads)}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return [out]


if __name__ == "__main__":
    from persistence import RunPersistence
    
    # Setup paths relative to Attempt_005 folder
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "run_history.sqlite"
    output_dir = base_dir / "results"
    
    # published_reference_scores is in Attempt_005/results
    reference_path = output_dir / "published_reference_scores.json"
    
    print(f"Loading history from: {db_path}")
    persistence = RunPersistence(db_path)
    visualizer = HistoryVisualizer(persistence, output_dir, reference_path)
    
    print("Generating all visualizations...")
    outputs = visualizer.generate_all()
    
    # Also generate the group statistics (all runs)
    group_stats = visualizer.generate_all_existing_group_statistics(latest_only=False)
    outputs.extend(group_stats)
    
    print(f"Successfully generated {len(outputs)} visualizations in {output_dir}:")
    for out in outputs:
        print(f" - {out.name}")
