from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


OBJECTIVE_SUITES = {
    "arc-welsh-easy",
    "arc-welsh-challenge",
    "piqa-welsh",
    "truthfulqa-welsh-mc1",
    "xnli-brit-cy",
    "mgsm-cy",
    "copa-cy",
    "wnli-cy",
    "welsh-cefr",
}
TECHIAITH_SUITES = {
    "welsh-lexicon",
    "welsh-grammar",
    "welsh-yes-no",
    "welsh-obscenities",
    "welsh-bilingual-placenames",
    "welsh-registers",
    "welsh-mmlu-lite",
    "welsh-toxigen",
    "welsh-arc-easy-mini-cy",
}
TRANSLATION_SUITE = "welsh-legislation-translation"


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
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
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


def build_leaderboard(
    database: Path,
    output_dir: Path,
    final_models: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_report(
        database,
        output_dir / "suite-scores.md",
        output_dir / "suite-scores.csv",
    )
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        "SELECT run_id, created_at, status, seed, max_cases, metadata_json FROM runs LIMIT 1"
    ).fetchone()
    if run is None:
        connection.close()
        raise RuntimeError("Nid oes cofnod rhediad yn y gronfa ddata")
    if run["status"] != "completed":
        connection.close()
        raise RuntimeError(
            f"Ni ellir cyhoeddi rhediad â'r statws {run['status']}"
        )
    status_counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM results GROUP BY status"
        )
    }
    if status_counts.get("error", 0):
        connection.close()
        raise RuntimeError("Ni ellir cyhoeddi sgorfwrdd sydd â gwallau heb eu datrys")
    model_stats = {
        row["model_id"]: dict(row)
        for row in connection.execute(
            """SELECT model_id, COUNT(*) AS result_count,
                      SUM(status = 'refusal') AS refusals,
                      SUM(COALESCE(cost_usd, 0)) AS cost_usd
               FROM results GROUP BY model_id"""
        )
    }
    connection.close()

    metadata = json.loads(run["metadata_json"])
    labels = {item["id"]: item.get("label", item["id"]) for item in metadata.get("models", [])}
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for item in summary:
        if item["score"] != "":
            by_model[item["model"]][item["suite"]] = float(item["score"])

    expected_suites = OBJECTIVE_SUITES | TECHIAITH_SUITES | {TRANSLATION_SUITE}
    leaderboard: list[dict[str, Any]] = []
    for model_id, scores in by_model.items():
        missing = expected_suites - scores.keys()
        if missing:
            raise RuntimeError(
                f"Mae setiau ar goll ar gyfer {model_id}: {', '.join(sorted(missing))}"
            )
        reasoning = sum(scores[suite] for suite in OBJECTIVE_SUITES) / len(OBJECTIVE_SUITES)
        practical = sum(scores[suite] for suite in TECHIAITH_SUITES) / len(TECHIAITH_SUITES)
        overall = (reasoning + practical) / 2
        stats = model_stats[model_id]
        leaderboard.append(
            {
                "model_id": model_id,
                "model": labels.get(model_id, model_id),
                "overall_score": round(overall, 2),
                "welsh_reasoning": round(reasoning, 2),
                "practical_welsh": round(practical, 2),
                "translation_bleu": round(scores[TRANSLATION_SUITE], 2),
                "completed": stats["result_count"],
                "refusals": stats["refusals"],
                "refusal_pct": round(100 * stats["refusals"] / stats["result_count"], 2),
                "coverage_pct": 100.0,
                "cost_usd": round(stats["cost_usd"], 6),
            }
        )
    leaderboard.sort(key=lambda item: (-item["overall_score"], item["model"]))
    for rank, item in enumerate(leaderboard, start=1):
        item["rank"] = rank

    headers = [
        "rank", "model_id", "model", "overall_score", "welsh_reasoning",
        "practical_welsh", "translation_bleu", "completed", "refusals",
        "refusal_pct", "coverage_pct", "cost_usd",
    ]
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(leaderboard)

    lines = [
        f"# Sgorfwrdd {run['run_id']}",
        "",
        "Mae'r prif sgôr yn rhoi pwysau cyfartal i gymedr macro naw prawf "
        "rhesymu Cymraeg a chymedr macro naw prawf Cymraeg ymarferol. Dangosir "
        "BLEU y prawf cyfieithu deddfwriaeth ar wahân.",
        "",
        "| Safle | Model | Prif sgôr | Rhesymu | Cymraeg ymarferol | Cyfieithu BLEU | Gwrthodiadau | Cwmpas |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in leaderboard:
        lines.append(
            f"| {item['rank']} | {item['model']} | {item['overall_score']:.2f} | "
            f"{item['welsh_reasoning']:.2f} | {item['practical_welsh']:.2f} | "
            f"{item['translation_bleu']:.2f} | {item['refusal_pct']:.2f}% | "
            f"{item['coverage_pct']:.1f}% |"
        )
    (output_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    public_metadata = {
        "run_id": run["run_id"],
        "created_at": run["created_at"],
        "status": run["status"],
        "seed": run["seed"],
        "max_cases_per_suite": run["max_cases"],
        "techiaith_revision": metadata.get("techiaith_revision"),
        "models_at_run_start": metadata.get("models", []),
        "models": final_models or metadata.get("models", []),
        "suites": metadata.get("suites", []),
        "status_counts": status_counts,
        "retry_policy": (
            "Rows with status=error were retried in place. Successful rows were not rerun; "
            "the final model configuration records the higher output ceilings used for retries."
        ),
        "formula": {
            "overall_score": "50% macro mean of 9 objective suites + 50% macro mean of 9 Techiaith suites",
            "translation": "Corpus BLEU, reported separately",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(public_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return leaderboard
