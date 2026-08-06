import unittest
import json

from aisteddfod_benchmarks.scoring import score_prediction


class ScoringTests(unittest.TestCase):
    def test_normalized_exact_match_handles_punctuation_and_case(self):
        score, _ = score_prediction("exact", "  YDW. ", "Ydw")
        self.assertAlmostEqual(score, 1.0)

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

    def test_ccc_edit_scores_correction_and_explanation_separately(self):
        expected = json.dumps(
            {
                "accepted_corrections": ["Mae'r ateb yn gywir."],
                "category": "treiglad",
                "rule": "TM_YN_TRAETHIADOL",
                "explanation_concepts": [["treiglad meddal"], ["yn traethiadol"]],
            }
        )
        prediction = """```json
        {"cywiriad":"Mae'r ateb yn gywir.","categori":"treiglad",
         "rheol":"TM_YN_TRAETHIADOL","esboniad":"Treiglad meddal ar ôl yn traethiadol."}
        ```"""
        score, metadata = score_prediction("ccc_edit", prediction, expected)
        self.assertEqual(score, 1.0)
        self.assertEqual(metadata["components"]["esboniad"], 1.0)

    def test_ccc_fidelity_penalizes_unsupported_fact_and_changed_quote(self):
        expected = json.dumps(
            {
                "fact_ids": ["F1", "F2"],
                "allowed_fact_ids": ["F1", "F2"],
                "quote": "Dyma'r cam nesaf.",
                "summary_concepts": [["120"], ["Medi"]],
            }
        )
        prediction = json.dumps(
            {
                "crynodeb": "Cymerodd 120 ran; bydd y cam nesaf ym mis Medi.",
                "ffeithiau": ["F1", "F2", "F5"],
                "dyfyniad": "Dyma'r cam nesaf!",
            }
        )
        score, metadata = score_prediction("ccc_fidelity", prediction, expected)
        self.assertLess(score, 0.75)
        self.assertEqual(metadata["components"]["ffyddlondeb"], 0.0)

    def test_ccc_fidelity_accepts_exact_quote_with_typographic_wrappers(self):
        expected = json.dumps(
            {
                "fact_ids": ["F1"],
                "allowed_fact_ids": ["F1"],
                "quote": "Byddwn yn cyhoeddi'r data ym mis Medi.",
                "summary_concepts": [["Medi"]],
            }
        )
        prediction = json.dumps(
            {
                "crynodeb": "Cyhoeddir y data ym mis Medi.",
                "ffeithiau": ["F1"],
                "dyfyniad": "Dywedodd yr arweinydd, “Byddwn yn cyhoeddi'r data ym mis Medi.”",
            }
        )
        score, metadata = score_prediction("ccc_fidelity", prediction, expected)
        self.assertEqual(score, 1.0)
        self.assertEqual(metadata["components"]["dyfyniad"], 1.0)

    def test_ccc_grounded_rewards_abstention_without_citations(self):
        expected = json.dumps(
            {
                "source_ids": [],
                "allowed_source_ids": ["C1", "E1"],
                "answer_concepts": [["dim tystiolaeth", "nid oes digon o wybodaeth"]],
            }
        )
        prediction = json.dumps(
            {"ateb": "Nid oes digon o wybodaeth yn y catalog.", "cyfeiriadau": []}
        )
        score, _ = score_prediction("ccc_grounded", prediction, expected)
        self.assertAlmostEqual(score, 1.0)

    def test_ccc_sources_rejects_disallowed_language(self):
        expected = json.dumps(
            {"source_ids": ["CY1"], "allowed_source_ids": ["CY1"]}
        )
        prediction = json.dumps({"ffynonellau": ["CY1", "EN1"]})
        score, metadata = score_prediction("ccc_sources", prediction, expected)
        self.assertEqual(score, 0.0)
        self.assertEqual(metadata["unsupported_source_ids"], ["en1"])


if __name__ == "__main__":
    unittest.main()
