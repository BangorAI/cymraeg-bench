from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_report(database: Path, markdown: Path | None = None, csv_path: Path | None = None) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT model_id, suite_id, scorer, status, prediction, expected, score,
                  cost_usd, input_tokens, output_tokens
           FROM results ORDER BY model_id, suite_id, case_id, repetition"""
    ).fetchall()
    grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["suite_id"], row["scorer"])].append(row)
    summary: list[dict[str, Any]] = []
    for (model_id, suite_id, scorer), items in grouped.items():
        completed = [item for item in items if item["status"] in {"completed", "refusal"}]
        if scorer == "bleu" and completed:
            try:
                import sacrebleu
            except ImportError as exc:
                raise RuntimeError("Mae sacrebleu ar goll; gosodwch y prosiect gyda pip") from exc
            metric = sacrebleu.corpus_bleu(
                [item["prediction"] or "" for item in completed],
                [[item["expected"] for item in completed]],
            ).score
        else:
            scores = [item["score"] for item in completed if item["score"] is not None]
            metric = 100 * sum(scores) / len(scores) if scores else None
        summary.append(
            {
                "model": model_id,
                "suite": suite_id,
                "metric": "BLEU" if scorer == "bleu" else "accuracy_pct",
                "score": round(metric, 4) if metric is not None else "",
                "completed": len(completed),
                "errors": sum(item["status"] == "error" for item in items),
                "input_tokens": sum(item["input_tokens"] or 0 for item in items),
                "output_tokens": sum(item["output_tokens"] or 0 for item in items),
                "cost_usd": round(sum(item["cost_usd"] or 0 for item in items), 6),
            }
        )
    connection.close()
    markdown = markdown or database.with_suffix(".md")
    csv_path = csv_path or database.with_suffix(".csv")
    headers = [
        "model",
        "suite",
        "metric",
        "score",
        "completed",
        "errors",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Adroddiad AIsteddfod",
        "",
        "| Model | Set | Mesur | Sgôr | N | Gwallau | Cost USD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['model']} | {item['suite']} | {item['metric']} | "
            f"{item['score']} | {item['completed']} | {item['errors']} | {item['cost_usd']:.6f} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
