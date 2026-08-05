import json
import tempfile
import unittest
from pathlib import Path

from aisteddfod_benchmarks.report import build_report
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


if __name__ == "__main__":
    unittest.main()
