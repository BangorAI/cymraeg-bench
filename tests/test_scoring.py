import unittest

from aisteddfod_benchmarks.scoring import score_prediction


class ScoringTests(unittest.TestCase):
    def test_normalized_exact_match_handles_punctuation_and_case(self):
        score, _ = score_prediction("exact", "  YDW. ", "Ydw")
        self.assertEqual(score, 1.0)

    def test_choice_extracts_short_answer_from_sentence(self):
        score, metadata = score_prediction("choice", "Yr ateb yw C.", "C")
        self.assertEqual(score, 1.0)
        self.assertEqual(metadata["extracted"], "C")

    def test_cefr_label_is_not_truncated(self):
        score, metadata = score_prediction("choice", "B2", "B2")
        self.assertEqual(score, 1.0)
        self.assertEqual(metadata["extracted"], "B2")

    def test_numeric_match_uses_last_number(self):
        score, metadata = score_prediction("number", "Felly 3 * 6 = 18", "18")
        self.assertEqual(score, 1.0)
        self.assertEqual(metadata["extracted"], "18")


if __name__ == "__main__":
    unittest.main()
