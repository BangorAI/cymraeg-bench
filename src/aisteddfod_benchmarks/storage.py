from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    seed INTEGER NOT NULL,
    max_cases INTEGER,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    status TEXT NOT NULL,
    scorer TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    expected TEXT NOT NULL,
    prediction TEXT,
    score REAL,
    scoring_json TEXT,
    provider_model TEXT,
    provider_response_json TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    stop_reason TEXT,
    error TEXT,
    PRIMARY KEY (run_id, model_id, suite_id, case_id, repetition),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_results_summary
ON results(run_id, model_id, suite_id, status);
"""


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def start_run(
        self,
        run_id: str,
        created_at: str,
        seed: int,
        max_cases: int | None,
        metadata: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO runs
               (run_id, created_at, status, seed, max_cases, metadata_json)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (run_id, created_at, seed, max_cases, json.dumps(metadata, ensure_ascii=False)),
        )
        self.connection.commit()

    def finish_run(self, run_id: str, status: str) -> None:
        self.connection.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
        self.connection.commit()

    def has_result(
        self, run_id: str, model_id: str, suite_id: str, case_id: str, repetition: int
    ) -> bool:
        row = self.connection.execute(
            """SELECT 1 FROM results
               WHERE run_id=? AND model_id=? AND suite_id=? AND case_id=? AND repetition=?
                 AND status != 'error'""",
            (run_id, model_id, suite_id, case_id, repetition),
        ).fetchone()
        return row is not None

    def save_result(self, values: dict[str, Any]) -> None:
        columns = list(values)
        placeholders = ",".join("?" for _ in columns)
        self.connection.execute(
            f"INSERT OR REPLACE INTO results ({','.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )
        self.connection.commit()

    def total_cost(self, run_id: str) -> float:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS cost FROM results WHERE run_id=?", (run_id,)
        ).fetchone()
        return float(row["cost"])
