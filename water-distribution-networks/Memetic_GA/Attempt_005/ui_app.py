import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import os
import time
import warnings

# Suppress Tkinter cleanup warnings from background threads (harmless)
warnings.filterwarnings("ignore", message=".*main thread is not in main loop.*")

from interactive_runner import InteractiveRunManager, RunConfig


class MemeticUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WDN Strict Benchmark Runner (Attempt_005)")
        self.root.geometry("1250x860")
        self._closing = False
        self._refresh_after_id: Optional[str] = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        base_dir = Path(__file__).resolve().parents[2]
        self.manager = InteractiveRunManager(base_dir)

        self.selected_run_id: Optional[str] = None
        self.last_displayed_log_count: int = 0  # Track logs we've already shown
        self.current_logs_run_id: Optional[str] = None  # Track which run's logs we're showing
        self._build_widgets()
        self._refresh_loop()

    def _build_widgets(self) -> None:
        control = ttk.LabelFrame(self.root, text="Run Controls")
        control.pack(fill="x", padx=8, pady=8)

        self.network_var = tk.StringVar(value="BIN.inp")
        self.algorithm_var = tk.StringVar(value="memetic")
        self.pop_var = tk.IntVar(value=40)
        self.gen_var = tk.IntVar(value=250)
        self.local_var = tk.DoubleVar(value=0.8)
        self.seed_var = tk.IntVar(value=42)
        self.strict_obj_var = tk.BooleanVar(value=True)
        self.full_pop_var = tk.BooleanVar(value=True)
        self.repair_mode_var = tk.StringVar(value="every_generation")

        row0 = ttk.Frame(control)
        row0.pack(fill="x", padx=6, pady=4)
        ttk.Label(row0, text="Network").pack(side="left")
        ttk.Combobox(row0, textvariable=self.network_var, values=self.manager.supported_benchmarks(), width=12, state="readonly").pack(side="left", padx=6)

        ttk.Label(row0, text="Algorithm").pack(side="left", padx=(10, 0))
        ttk.Combobox(row0, textvariable=self.algorithm_var, values=["memetic", "standard"], width=10, state="readonly").pack(side="left", padx=6)

        ttk.Label(row0, text="Population").pack(side="left", padx=(10, 0))
        ttk.Entry(row0, textvariable=self.pop_var, width=8).pack(side="left", padx=6)

        ttk.Label(row0, text="Max Gens").pack(side="left", padx=(10, 0))
        ttk.Entry(row0, textvariable=self.gen_var, width=8).pack(side="left", padx=6)

        ttk.Label(row0, text="Local Search").pack(side="left", padx=(10, 0))
        ttk.Entry(row0, textvariable=self.local_var, width=8).pack(side="left", padx=6)

        ttk.Label(row0, text="Seed").pack(side="left", padx=(10, 0))
        ttk.Entry(row0, textvariable=self.seed_var, width=8).pack(side="left", padx=6)

        row1 = ttk.Frame(control)
        row1.pack(fill="x", padx=6, pady=4)
        ttk.Checkbutton(
            row1,
            text="Use strict paper objective for optimization",
            variable=self.strict_obj_var,
        ).pack(side="left", padx=(0, 10))

        ttk.Checkbutton(
            row1,
            text="Strict hydraulic check for full population each generation",
            variable=self.full_pop_var,
        ).pack(side="left", padx=(0, 10))

        ttk.Label(row1, text="Repair mode").pack(side="left", padx=(0, 6))
        ttk.Combobox(
            row1,
            textvariable=self.repair_mode_var,
            values=["none", "first_generation", "every_generation"],
            width=18,
            state="readonly",
        ).pack(side="left", padx=(0, 10))

        row2 = ttk.Frame(control)
        row2.pack(fill="x", padx=6, pady=4)
        ttk.Button(row2, text="Start Run", command=self._start_run).pack(side="left")
        ttk.Button(row2, text="Resume Selected Run", command=self._resume_selected_run).pack(side="left", padx=6)
        ttk.Button(row2, text="Stop Selected Run", command=self._stop_selected_run).pack(side="left", padx=6)
        ttk.Button(row2, text="Delete Selected Runs", command=self._delete_selected_runs).pack(side="left", padx=6)
        ttk.Button(row2, text="Force Delete from DB", command=self._force_delete_from_db).pack(side="left", padx=6)
        ttk.Button(row2, text="Refresh History", command=self._refresh_history).pack(side="left", padx=6)
        ttk.Button(row2, text="Generate Graphs", command=self._generate_graphs).pack(side="left", padx=6)

        status_frame = ttk.LabelFrame(self.root, text="Live Status")
        status_frame.pack(fill="x", padx=8, pady=6)
        self.status_var = tk.StringVar(value="No run selected")
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=6)

        split = ttk.Panedwindow(self.root, orient="vertical")
        split.pack(fill="both", expand=True, padx=8, pady=6)

        history_frame = ttk.LabelFrame(split, text="Persisted Run History")
        split.add(history_frame, weight=2)

        columns = ("run_id", "network", "algorithm", "status", "started", "ended", "best_paper", "best_gap")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=10, selectmode="extended")
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=140, stretch=True)
        self.history_tree.column("run_id", width=180)
        self.history_tree.column("started", width=180)
        self.history_tree.column("ended", width=180)
        self.history_tree.pack(fill="both", expand=True, padx=6, pady=6)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        logs_frame = ttk.LabelFrame(split, text="Logs")
        split.add(logs_frame, weight=3)
        self.log_text = tk.Text(logs_frame, height=20, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        self._refresh_history()

    def _build_config(self) -> RunConfig:
        return RunConfig(
            network_file=self.network_var.get(),
            algorithm=self.algorithm_var.get(),
            population_size=int(self.pop_var.get()),
            max_generations=int(self.gen_var.get()),
            local_search_intensity=float(self.local_var.get()),
            seed=int(self.seed_var.get()),
            strict_objective_for_optimization=bool(self.strict_obj_var.get()),
            strict_check_full_population_each_gen=bool(self.full_pop_var.get()),
            repair_mode=self.repair_mode_var.get(),
        )

    def _start_run(self) -> None:
        try:
            cfg = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        run_id = self.manager.start_run(cfg)
        self.selected_run_id = run_id
        self._refresh_history()
        self.status_var.set(f"Started {run_id}")

    def _resume_selected_run(self) -> None:
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo("Resume Run", "Select a run in history first.")
            return

        first = selected[0]
        values = self.history_tree.item(first, "values")
        if not values:
            messagebox.showinfo("Resume Run", "No valid run selected.")
            return

        run_id = values[0]
        status = values[3]
        force = False
        if status == "completed":
            messagebox.showinfo("Resume Run", "Completed runs cannot be resumed.")
            return
        if status == "running":
            force = messagebox.askyesno(
                "Force Resume",
                "This run is marked as running in history.\n"
                "Use FORCE resume only if the app was closed/crashed and no thread is actually running.",
            )
            if not force:
                return

        try:
            resumed_run_id = self.manager.resume_run(run_id, force=force)
        except Exception as exc:
            messagebox.showerror("Resume Run", f"Failed: {exc}")
            return

        self.selected_run_id = resumed_run_id
        self._refresh_history()
        self.status_var.set(f"Resumed {resumed_run_id}{' (forced)' if force else ''}")

    def _stop_selected_run(self) -> None:
        if not self.selected_run_id:
            return
        self.manager.stop_run(self.selected_run_id)

    def _force_delete_from_db(self) -> None:
        """Bypass all checks and nuke selected runs from DB."""
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo("Force Delete from DB", "Select one or more runs in history first.")
            return

        run_ids = [self.history_tree.item(item, "values")[0] for item in selected]
        
        confirm = messagebox.askyesno(
            "Force Delete from DB",
            f"PERMANENTLY delete {len(run_ids)} selected run(s) from database?\n"
            "This CANNOT be undone and will remove all logs and data.\n"
            "Runs: " + ", ".join(run_ids[:3]) + ("..." if len(run_ids) > 3 else ""),
        )
        if not confirm:
            return

        try:
            deleted = self.manager.force_delete_runs_from_db(run_ids)
        except Exception as exc:
            messagebox.showerror("Force Delete from DB", f"Failed: {exc}")
            return

        if self.selected_run_id in run_ids:
            self.selected_run_id = None
            self.log_text.delete("1.0", tk.END)

        self._refresh_history()
        self.status_var.set(f"Force-deleted {deleted} run(s) from DB")

    def _delete_selected_runs(self) -> None:
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo("Delete Runs", "Select one or more runs in history first.")
            return

        run_ids = [self.history_tree.item(item, "values")[0] for item in selected]
        statuses = [self.history_tree.item(item, "values")[3] for item in selected]

        running = [rid for rid, st in zip(run_ids, statuses) if st == "running"]
        if running:
            force = messagebox.askyesnocancel(
                "Delete Runs",
                "Cannot delete currently running runs:\n" + "\n".join(running) + 
                "\n\nYes = Force delete from DB\nNo = Cancel\nCancel = Show more help",
            )
            if force is None:
                messagebox.showinfo(
                    "Delete Runs - Help",
                    "To delete stuck/running runs:\n\n"
                    "Use 'Force Delete from DB' to remove them immediately from history.\n"
                    "This is permanent and also removes logs + generation data."
                )
                return
            if force is False:
                return
            # Force delete
            try:
                deleted = self.manager.force_delete_runs_from_db(run_ids)
            except Exception as exc:
                messagebox.showerror("Force Delete", f"Failed: {exc}")
                return
        else:
            confirm = messagebox.askyesno(
                "Delete Runs",
                f"Delete {len(run_ids)} selected run(s) from history?\n"
                "This will remove their logs and per-generation data.",
            )
            if not confirm:
                return

            try:
                deleted = self.manager.delete_runs(run_ids)
            except Exception as exc:
                messagebox.showerror("Delete Runs", f"Failed: {exc}")
                return

        if self.selected_run_id in run_ids:
            self.selected_run_id = None
            self.log_text.delete("1.0", tk.END)

        self._refresh_history()
        self.status_var.set(f"Deleted {deleted} run(s)")

    def _generate_graphs(self) -> None:
        try:
            outputs = self.manager.generate_history_visualizations()
        except Exception as exc:
            messagebox.showerror("Generate Graphs", f"Failed: {exc}")
            return

        if not outputs:
            messagebox.showinfo(
                "Generate Graphs",
                "No completed history runs found with enough data to plot yet."
            )
            return

        output_lines = "\n".join(str(p) for p in outputs)
        self.status_var.set(f"Generated {len(outputs)} graph(s)")
        messagebox.showinfo("Generate Graphs", f"Generated:\n{output_lines}")

        # Open output folder for convenience on Windows.
        try:
            os.startfile(str(outputs[0].parent))
        except Exception:
            pass

    def _on_history_select(self, _event) -> None:
        selected = self.history_tree.selection()
        if not selected:
            return
        run_id = self.history_tree.item(selected[0], "values")[0]
        self.selected_run_id = run_id
        self._refresh_logs(run_id)

    def _refresh_history(self) -> None:
        prev_selected_item_ids = self.history_tree.selection()
        prev_selected_run_ids = set()
        for item in prev_selected_item_ids:
            values = self.history_tree.item(item, "values")
            if values:
                prev_selected_run_ids.add(values[0])

        prev_focus_item = self.history_tree.focus()
        prev_focus_run_id = None
        if prev_focus_item:
            focus_values = self.history_tree.item(prev_focus_item, "values")
            if focus_values:
                prev_focus_run_id = focus_values[0]

        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        rows = self.manager.persistence.list_runs(limit=300)
        run_id_to_item = {}
        for row in rows:
            best_paper = "" if row.best_paper_score is None else f"{float(row.best_paper_score):.3e}"
            best_gap = "" if row.best_gap_pct is None else f"{float(row.best_gap_pct):+.2f}%"
            item_id = self.history_tree.insert(
                "",
                "end",
                values=(
                    row.run_id,
                    row.network_file,
                    row.algorithm,
                    row.status,
                    row.started_at,
                    row.ended_at or "",
                    best_paper,
                    best_gap,
                ),
            )
            run_id_to_item[row.run_id] = item_id

        # Restore selection/focus so periodic refresh does not auto-deselect rows.
        restored_items = [run_id_to_item[rid] for rid in prev_selected_run_ids if rid in run_id_to_item]
        if restored_items:
            self.history_tree.selection_set(restored_items)

        if prev_focus_run_id and prev_focus_run_id in run_id_to_item:
            self.history_tree.focus(run_id_to_item[prev_focus_run_id])

    def _refresh_logs(self, run_id: str) -> None:
        # If we switched runs, clear and rebuild
        if run_id != self.current_logs_run_id:
            self.log_text.delete("1.0", tk.END)
            self.last_displayed_log_count = 0
            self.current_logs_run_id = run_id
        
        all_logs = self.manager.persistence.load_logs(run_id)
        
        # Only append new logs since last refresh
        new_logs = all_logs[self.last_displayed_log_count:]
        if new_logs:
            if self.last_displayed_log_count == 0:
                # First load, insert all at once
                self.log_text.insert(tk.END, "\n".join(all_logs))
                # Auto-scroll to end on first load
                self.log_text.see(tk.END)
            else:
                # Append only new ones
                self.log_text.insert(tk.END, "\n" + "\n".join(new_logs))
            self.last_displayed_log_count = len(all_logs)
            # Only auto-scroll to end if user hasn't manually scrolled up
            # Check if scroll is near bottom
            try:
                scroll_pos = self.log_text.yview()[1]
                if scroll_pos > 0.95:  # If within 5% of bottom, auto-scroll
                    self.log_text.see(tk.END)
            except:
                pass

    def _refresh_live_status(self) -> None:
        if not self.selected_run_id:
            self.status_var.set("No run selected")
            return

        live = self.manager.get_live_state(self.selected_run_id)
        if live is not None:
            gap = live.get("best_gap_to_published_pct")
            gap_text = "n/a" if gap is None else f"{float(gap):+.2f}%"
            best_paper = live.get("best_paper_score")
            best_paper_text = "inf/n-a" if best_paper is None else f"{float(best_paper):.3e}"
            best_train = live.get("best_training_fitness")
            best_train_text = "n/a" if best_train is None else f"{float(best_train):.3e}"
            self.status_var.set(
                f"Run {live['run_id']} | status={live['status']} | gen={live['generation']} | "
                f"best_train={best_train_text} | best_paper={best_paper_text} | gap={gap_text}"
            )
        else:
            last = self.manager.persistence.load_last_generation(self.selected_run_id)
            if last is None:
                self.status_var.set(f"Run {self.selected_run_id}: no generation data yet")
            else:
                gap = last.get("gap_to_published_pct")
                gap_text = "n/a" if gap is None else f"{float(gap):+.2f}%"
                self.status_var.set(
                    f"Run {self.selected_run_id} | persisted gen={last.get('generation')} | "
                    f"best_train={float(last.get('best_training_fitness', 0.0)):.3e} | "
                    f"best_paper={float(last.get('best_paper_score', 0.0)):.3e} | gap={gap_text}"
                )

    def _refresh_loop(self) -> None:
        if self._closing:
            return
        try:
            self._refresh_history()
            if self.selected_run_id:
                self._refresh_logs(self.selected_run_id)
            self._refresh_live_status()
        except Exception as exc:
            # Keep UI alive even if one refresh cycle fails.
            self.status_var.set(f"UI refresh warning: {exc}")
        finally:
            if not self._closing:
                self._refresh_after_id = self.root.after(1200, self._refresh_loop)

    def _on_close(self) -> None:
        self._closing = True
        if self._refresh_after_id:
            try:
                self.root.after_cancel(self._refresh_after_id)
            except Exception:
                pass

        # Request stop for active runs so worker threads can exit cleanly.
        try:
            for run_id in self.manager.get_active_run_ids():
                self.manager.stop_run(run_id)
        except Exception:
            pass

        # Wait briefly for worker threads to drain before destroying Tk.
        self._wait_for_threads_then_close(time.time() + 5.0)

    def _wait_for_threads_then_close(self, deadline: float) -> None:
        try:
            active_ids = self.manager.get_active_run_ids()
        except Exception:
            active_ids = []

        if not active_ids:
            self.root.quit()
            self.root.destroy()
            return

        if time.time() >= deadline:
            # Force close after timeout if workers are still busy.
            self.root.quit()
            self.root.destroy()
            return

        self.status_var.set(
            f"Closing: waiting for {len(active_ids)} run(s) to stop..."
        )
        self.root.after(200, lambda: self._wait_for_threads_then_close(deadline))


def main() -> None:
    root = tk.Tk()
    app = MemeticUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
