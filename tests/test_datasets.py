import unittest
import json

from aisteddfod_benchmarks.config import SuiteConfig
from aisteddfod_benchmarks.datasets import _adapt


def suite(adapter: str, scorer: str = "choice") -> SuiteConfig:
    return SuiteConfig("t", "T", "x", adapter, scorer, 1, "objective")


class DatasetTests(unittest.TestCase):
    def test_arc_adapter_preserves_non_letter_labels(self):
        case = _adapt(
            suite("arc"),
            {"id": "1", "question": "Q?", "choices": {"label": ["1", "2"], "text": ["x", "y"]}, "answerKey": "2"},
            0,
        )
        self.assertEqual(case.expected, "2")
        self.assertIn("2: y", case.user)

    def test_truthfulqa_mc1_uses_first_true_target(self):
        case = _adapt(
            suite("truthfulqa_mc1"),
            {"question": "Q?", "mc1_targets": {"choices": ["x", "y", "z"], "labels": [0, 1, 0]}},
            0,
        )
        self.assertEqual(case.expected, "B")

    def test_mgsm_uses_answer_number(self):
        case = _adapt(suite("mgsm", "number"), {"question": "2+2?", "answer_number": 4}, 0)
        self.assertEqual(case.expected, "4")

    def test_local_adapter_serializes_structured_target_and_metadata(self):
        case = _adapt(
            suite("local", "ccc_edit"),
            {
                "id": "ccc-1",
                "system": "System",
                "user": "Defnyddiwr",
                "ideal": {"category": "treiglad"},
                "metadata": {"dimension": "cywiro"},
            },
            3,
        )
        self.assertEqual(json.loads(case.expected), {"category": "treiglad"})
        self.assertEqual(case.metadata["dimension"], "cywiro")
        self.assertEqual(case.metadata["source_index"], 3)


if __name__ == "__main__":
    unittest.main()
