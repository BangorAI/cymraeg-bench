import json
import tempfile
import unittest
from pathlib import Path

from aisteddfod_benchmarks.report import (
    OBJECTIVE_SUITES,
    TECHIAITH_SUITES,
    TRANSLATION_SUITE,
    build_leaderboard,
)
from aisteddfod_benchmarks.storage import Storage


class LeaderboardTests(unittest.TestCase):
    def test_builds_ranked_public_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "official.sqlite3"
            store = Storage(database)
            store.start_run(
                "official",
                "2026-08-05T00:00:00Z",
                1,
                100,
                {
                    "models": [
                        {"id": "strong", "label": "Model Cryf"},
                        {"id": "baseline", "label": "Llinell Sylfaen"},
                    ],
                    "suites": sorted(OBJECTIVE_SUITES | TECHIAITH_SUITES | {TRANSLATION_SUITE}),
                    "techiaith_revision": "abc123",
                },
            )
            for model_id, score in (("strong", 1.0), ("baseline", 0.5)):
                for suite_id in OBJECTIVE_SUITES | TECHIAITH_SUITES | {TRANSLATION_SUITE}:
                    store.save_result(
                        {
                            "run_id": "official", "model_id": model_id,
                            "suite_id": suite_id, "case_id": "case", "repetition": 1,
                            "status": "completed",
                            "scorer": "bleu" if suite_id == TRANSLATION_SUITE else "exact",
                            "system_prompt": "system", "user_prompt": "user",
                            "expected": "Y", "prediction": "Y" if score == 1 else "N",
                            "score": score, "scoring_json": "{}", "provider_model": model_id,
                            "provider_response_json": "{}", "input_tokens": 1,
                            "output_tokens": 1, "cost_usd": 0.01, "latency_ms": 1,
                            "stop_reason": "stop", "error": None,
                        }
                    )
            store.finish_run("official", "completed")
            store.close()

            rows = build_leaderboard(database, root / "results")

            self.assertEqual(rows[0]["model"], "Model Cryf")
            self.assertEqual(rows[0]["overall_score"], 100.0)
            self.assertEqual(rows[1]["overall_score"], 50.0)
            self.assertTrue((root / "results" / "leaderboard.csv").exists())
            metadata = json.loads((root / "results" / "metadata.json").read_text())
            self.assertEqual(metadata["techiaith_revision"], "abc123")

    def test_rejects_an_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "interrupted.sqlite3"
            store = Storage(database)
            store.start_run("interrupted", "2026-08-05T00:00:00Z", 1, 100, {})
            store.finish_run("interrupted", "interrupted")
            store.close()

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                build_leaderboard(database, root / "results")


if __name__ == "__main__":
    unittest.main()
