import json
import tempfile
import unittest
import sqlite3
from pathlib import Path

from aisteddfod_benchmarks.finalize import finalize_output_errors
from aisteddfod_benchmarks.report import build_ccc_report, build_report
from aisteddfod_benchmarks.storage import Storage


class StorageReportTests(unittest.TestCase):
    def test_storage_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "run.sqlite3"
            store = Storage(database)
            store.start_run("r", "2026-08-05T00:00:00Z", 1, 1, {})
            values = {
                "run_id": "r", "model_id": "m", "suite_id": "s", "case_id": "c",
                "repetition": 1, "status": "completed", "scorer": "exact",
                "system_prompt": "system", "user_prompt": "user", "expected": "Y",
                "prediction": "Y", "score": 1.0, "scoring_json": json.dumps({}),
                "provider_model": "m", "provider_response_json": "{}", "input_tokens": 2,
                "output_tokens": 1, "cost_usd": 0.01, "latency_ms": 10,
                "stop_reason": "stop", "error": None,
            }
            failed = values | {
                "status": "error", "prediction": None, "score": None,
                "provider_model": None, "provider_response_json": None,
                "input_tokens": None, "output_tokens": None, "cost_usd": None,
                "latency_ms": None, "stop_reason": None, "error": "temporary",
            }
            store.save_result(failed)
            self.assertFalse(store.has_result("r", "m", "s", "c", 1))
            store.save_result(values)
            self.assertTrue(store.has_result("r", "m", "s", "c", 1))
            store.close()
            summary = build_report(database)
            self.assertEqual(summary[0]["score"], 100.0)
            self.assertTrue(database.with_suffix(".md").exists())
            self.assertTrue(database.with_suffix(".csv").exists())

    def test_ccc_report_aggregates_scoring_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ccc.sqlite3"
            store = Storage(database)
            store.start_run("r", "2026-08-06T00:00:00Z", 1, None, {})
            base = {
                "run_id": "r", "model_id": "m", "suite_id": "ccc-test",
                "repetition": 1, "status": "completed", "scorer": "ccc_edit",
                "system_prompt": "system", "user_prompt": "user", "expected": "{}",
                "prediction": "{}", "provider_model": "m", "provider_response_json": "{}",
                "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.01,
                "latency_ms": 10, "stop_reason": "stop", "error": None,
            }
            for case_id, score in (("c1", 1.0), ("c2", 0.0)):
                store.save_result(
                    base | {
                        "case_id": case_id,
                        "score": score,
                        "scoring_json": json.dumps({"components": {"cywiriad": score}}),
                    }
                )
            store.close()
            summary = build_ccc_report(database)
            self.assertEqual(summary[0]["score_pct"], 50.0)
            self.assertTrue(database.with_name("ccc-ccc.md").exists())

    def test_persistent_model_output_error_becomes_scored_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "invalid.sqlite3"
            store = Storage(database)
            store.start_run("r", "2026-08-06T00:00:00Z", 1, 1, {})
            store.save_result(
                {
                    "run_id": "r", "model_id": "m", "suite_id": "ccc-cywiro-esbonio",
                    "case_id": "c", "repetition": 1, "status": "error",
                    "scorer": "ccc_edit", "system_prompt": "system",
                    "user_prompt": "user", "expected": "{}", "prediction": None,
                    "score": None, "scoring_json": "{}", "provider_model": None,
                    "provider_response_json": None, "input_tokens": None,
                    "output_tokens": None, "cost_usd": None, "latency_ms": None,
                    "stop_reason": None,
                    "error": "Ateb wedi'i dorri gan y terfyn tocynnau (length)",
                }
            )
            store.finish_run("r", "completed_with_errors")
            store.close()

            self.assertEqual(finalize_output_errors(database), (1, 0))
            summary = build_report(database)
            self.assertEqual(summary[0]["score"], 0.0)
            self.assertEqual(summary[0]["invalid"], 1)
            connection = sqlite3.connect(database)
            status, run_status, scoring_json = connection.execute(
                """SELECT results.status, runs.status, results.scoring_json
                   FROM results JOIN runs USING (run_id)"""
            ).fetchone()
            connection.close()
            self.assertEqual((status, run_status), ("invalid", "completed"))
            self.assertEqual(
                set(json.loads(scoring_json)["components"]),
                {"cywiriad", "categori", "rheol", "esboniad"},
            )


if __name__ == "__main__":
    unittest.main()
