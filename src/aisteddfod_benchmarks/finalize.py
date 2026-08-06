from __future__ import annotations

import json
import sqlite3
from pathlib import Path


OUTPUT_ERROR_PREFIXES = (
    "Ateb wedi'i dorri gan y terfyn tocynnau",
    "Ateb gwag gan y darparwr",
)

CCC_COMPONENTS = {
    "ccc_edit": ["cywiriad", "categori", "rheol", "esboniad"],
    "ccc_fidelity": ["ffeithiau", "dyfyniad", "cwmpas", "ffyddlondeb"],
    "ccc_grounded": ["cyfeiriadau", "ateb", "dim_rhithgyfeirio"],
    "ccc_sources": ["dewis_ffynonellau", "dim_ffynonellau_annerbyniol"],
}
CCC_CHOICE_COMPONENTS = {
    "ccc-priod-ddull": ["priod"],
    "ccc-termau": ["term"],
}


def _invalid_scoring(suite_id: str, scorer: str, reason: str) -> str:
    dimensions = CCC_COMPONENTS.get(scorer, CCC_CHOICE_COMPONENTS.get(suite_id, []))
    payload: dict[str, object] = {"invalid": True, "reason": reason}
    if dimensions:
        payload["components"] = {dimension: 0.0 for dimension in dimensions}
    return json.dumps(payload, ensure_ascii=False)


def finalize_output_errors(database: Path) -> tuple[int, int]:
    """Count persistent model-output failures as scored zero after retries.

    Network, authentication and other infrastructure errors remain untouched and
    continue to block publication.
    """
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT run_id, model_id, suite_id, case_id, repetition, scorer, error
           FROM results WHERE status = 'error'"""
    ).fetchall()
    converted = 0
    for row in rows:
        reason = row["error"] or ""
        if not reason.startswith(OUTPUT_ERROR_PREFIXES):
            continue
        connection.execute(
            """UPDATE results
               SET status='invalid', prediction='', score=0.0, scoring_json=?,
                   stop_reason='invalid_output'
               WHERE run_id=? AND model_id=? AND suite_id=? AND case_id=?
                 AND repetition=?""",
            (
                _invalid_scoring(row["suite_id"], row["scorer"], reason),
                row["run_id"], row["model_id"], row["suite_id"], row["case_id"],
                row["repetition"],
            ),
        )
        converted += 1
    remaining = connection.execute(
        "SELECT COUNT(*) FROM results WHERE status = 'error'"
    ).fetchone()[0]
    connection.execute(
        "UPDATE runs SET status=?",
        ("completed" if remaining == 0 else "completed_with_errors",),
    )
    connection.commit()
    connection.close()
    return converted, int(remaining)
