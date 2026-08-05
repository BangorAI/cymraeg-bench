import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aisteddfod_benchmarks.config import ModelConfig, SuiteConfig
from aisteddfod_benchmarks.datasets import Case
from aisteddfod_benchmarks.providers import ProviderResponse
from aisteddfod_benchmarks.runner import run_evaluation


class RunnerTests(unittest.TestCase):
    def test_parallel_workers_write_checkpoints_in_main_thread(self):
        models = [
            ModelConfig(
                f"m{i}", f"Model {i}", "openai_compatible", f"model-{i}",
                "UNUSED", "http://localhost/v1",
            )
            for i in range(2)
        ]
        suite = SuiteConfig(
            "suite", "Suite", "local", "test", "exact", 1, "objective"
        )
        case = Case("case", "system", "user", "Y")

        def fake_generate(model, system, user, *, max_tokens):
            return ProviderResponse(
                "Y", model.api_model, 2, 1, 0.001, 5, "stop", {}
            )

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "aisteddfod_benchmarks.runner.load_cases", return_value=[case]
            ), patch(
                "aisteddfod_benchmarks.runner.generate", side_effect=fake_generate
            ):
                _, database, completed, errors = run_evaluation(
                    models=models,
                    suites=[suite],
                    root=Path(directory),
                    techiaith_revision="revision",
                    run_id="parallel",
                    workers=2,
                )

            self.assertEqual((completed, errors), (2, 0))
            self.assertTrue(database.exists())


if __name__ == "__main__":
    unittest.main()
