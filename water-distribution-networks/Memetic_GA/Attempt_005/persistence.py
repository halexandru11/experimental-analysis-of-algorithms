import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunRow:
    run_id: str
    algorithm: str
    network_file: str
    status: str
    started_at: str
    ended_at: Optional[str]
    best_paper_score: Optional[float]
    best_gap_pct: Optional[float]


class RunPersistence:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    network_file TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    config_json TEXT NOT NULL,
                    summary_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    best_training_fitness REAL,
                    best_training_cost REAL,
                    best_paper_score REAL,
                    best_paper_cost REAL,
                    best_paper_feasible REAL,
                    feasible_count INTEGER,
                    gap_to_published_pct REAL,
                    ts TEXT NOT NULL,
                    PRIMARY KEY (run_id, generation),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )

    def create_run(self, run_id: str, algorithm: str, network_file: str, config: Dict[str, Any]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs (run_id, algorithm, network_file, status, started_at, config_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        algorithm,
                        network_file,
                        "running",
                        datetime.utcnow().isoformat(),
                        json.dumps(config),
                    ),
                )

    def append_log(self, run_id: str, level: str, message: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO logs (run_id, level, message, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, level, message, datetime.utcnow().isoformat()),
                )

    def append_generation(self, run_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO generations (
                        run_id,
                        generation,
                        best_training_fitness,
                        best_training_cost,
                        best_paper_score,
                        best_paper_cost,
                        best_paper_feasible,
                        feasible_count,
                        gap_to_published_pct,
                        ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(payload.get("generation", 0)),
                        payload.get("best_training_fitness"),
                        payload.get("best_training_cost"),
                        payload.get("best_paper_score"),
                        payload.get("best_paper_cost"),
                        payload.get("best_paper_feasible"),
                        payload.get("feasible_count"),
                        payload.get("gap_to_published_pct"),
                        datetime.utcnow().isoformat(),
                    ),
                )

    def finalize_run(self, run_id: str, status: str, summary: Dict[str, Any]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, ended_at = ?, summary_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        datetime.utcnow().isoformat(),
                        json.dumps(summary),
                        run_id,
                    ),
                )

    def mark_run_running(self, run_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            with self._connect() as conn:
                if config is None:
                    conn.execute(
                        """
                        UPDATE runs
                        SET status = ?, ended_at = ?
                        WHERE run_id = ?
                        """,
                        (
                            "running",
                            None,
                            run_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE runs
                        SET status = ?, ended_at = ?, config_json = ?
                        WHERE run_id = ?
                        """,
                        (
                            "running",
                            None,
                            json.dumps(config),
                            run_id,
                        ),
                    )

    def list_runs(self, limit: int = 200) -> List[RunRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.run_id,
                    r.algorithm,
                    r.network_file,
                    r.status,
                    r.started_at,
                    r.ended_at,
                    json_extract(r.summary_json, '$.best_paper_score') AS best_paper_score,
                    json_extract(r.summary_json, '$.best_gap_to_published_pct') AS best_gap_pct
                FROM runs r
                ORDER BY r.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            RunRow(
                run_id=row["run_id"],
                algorithm=row["algorithm"],
                network_file=row["network_file"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                best_paper_score=row["best_paper_score"],
                best_gap_pct=row["best_gap_pct"],
            )
            for row in rows
        ]

    def load_logs(self, run_id: str, limit: int = 1000) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, level, message
                FROM logs
                WHERE run_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()

        return [f"[{r['ts']}] {r['level'].upper()}: {r['message']}" for r in rows]

    def load_last_generation(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM generations
                WHERE run_id = ?
                ORDER BY generation DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def delete_runs(self, run_ids: List[str]) -> int:
        if not run_ids:
            return 0

        placeholders = ",".join(["?"] * len(run_ids))
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    f"DELETE FROM logs WHERE run_id IN ({placeholders})",
                    tuple(run_ids),
                )
                conn.execute(
                    f"DELETE FROM generations WHERE run_id IN ({placeholders})",
                    tuple(run_ids),
                )
                cur = conn.execute(
                    f"DELETE FROM runs WHERE run_id IN ({placeholders})",
                    tuple(run_ids),
                )
                return int(cur.rowcount if cur.rowcount is not None else 0)

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, algorithm, network_file, status, started_at, ended_at, config_json, summary_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["config"] = json.loads(result.pop("config_json") or "{}")
        result["summary"] = json.loads(result.pop("summary_json") or "{}")
        return result

    def load_generations(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT generation, best_training_fitness, best_training_cost,
                       best_paper_score, best_paper_cost, best_paper_feasible,
                       feasible_count, gap_to_published_pct, ts
                FROM generations
                WHERE run_id = ?
                ORDER BY generation ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_completed_run_for_network_algorithm(
        self,
        network_file: str,
        algorithm: str,
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, algorithm, network_file, status, started_at, ended_at, config_json, summary_json
                FROM runs
                WHERE network_file = ?
                  AND algorithm = ?
                  AND status IN ('completed', 'stopped')
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (network_file, algorithm),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["config"] = json.loads(result.pop("config_json") or "{}")
        result["summary"] = json.loads(result.pop("summary_json") or "{}")
        return result

    def latest_completed_run_for_network(self, network_file: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, algorithm, network_file, status, started_at, ended_at, config_json, summary_json
                FROM runs
                WHERE network_file = ?
                  AND status IN ('completed', 'stopped')
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (network_file,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["config"] = json.loads(result.pop("config_json") or "{}")
        result["summary"] = json.loads(result.pop("summary_json") or "{}")
        return result
