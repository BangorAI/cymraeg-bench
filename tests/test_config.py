import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from aisteddfod_benchmarks.config import load_models, load_suites, select_models
from aisteddfod_benchmarks.datasets import load_cases
from aisteddfod_benchmarks.scoring import score_prediction


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_catalog_has_all_scoreboard_models_and_public_suites(self):
        models = load_models(ROOT / "config")
        suites, revision = load_suites(ROOT / "config")
        scoreboard = {model.id for model in models if model.tier == "scoreboard"}
        self.assertEqual(
            scoreboard,
            {
                "gpt-5-6-sol", "claude-opus-4-8", "claude-fable-5", "grok-4-20",
                "kimi-k3", "glm-5-2", "deepseek-v4-flash-0731", "jupiter-n-120b",
                "caernarfon-3b", "caernarfon-3b-cymraeg-instruct-v0-1",
                "mwydryn", "bangor-mistral-cymraeg-v2",
                "mwydryn-7b-v2",
                "techiaith-llama-3-2-1b-sft",
            },
        )
        self.assertEqual(len([suite for suite in suites if suite.enabled]), 26)
        self.assertEqual(
            len([suite for suite in suites if suite.enabled and suite.group == "ccc"]),
            7,
        )
        self.assertEqual(revision, "47839d2147c97fd2f10a52dd36751608e5fa36bf")

    def test_explicit_selection_can_include_disabled_local_model(self):
        models = load_models(ROOT / "config")
        mwydryn = select_models(models, ["mwydryn"])[0]
        self.assertEqual(mwydryn.api_model, "BangorAI/phi2-mwydryn-1")
        self.assertEqual(mwydryn.min_output_tokens, 512)
        mistral_cymraeg = select_models(models, ["bangor-mistral-cymraeg-v2"])[0]
        self.assertEqual(
            mistral_cymraeg.api_model,
            "BangorAI/Mistral-7B-Cymraeg-Welsh-v2",
        )
        self.assertEqual(mistral_cymraeg.min_output_tokens, 512)
        mwydryn_7b = select_models(models, ["mwydryn-7b-v2"])[0]
        self.assertEqual(mwydryn_7b.api_model, "BangorAI/mwydryn-7b-fersiwn-2")
        self.assertEqual(mwydryn_7b.min_output_tokens, 512)
        techiaith_sft = select_models(models, ["techiaith-llama-3-2-1b-sft"])[0]
        self.assertEqual(
            techiaith_sft.api_model,
            "techiaith/llama-3.2-1b-welsh-sft",
        )
        self.assertEqual(techiaith_sft.min_output_tokens, 512)
        self.assertEqual(techiaith_sft.repetition_penalty, 1.15)
        jupiter = select_models(models, ["jupiter-n-120b"])[0]
        self.assertEqual(jupiter.min_output_tokens, 512)
        caernarfon_instruct = select_models(
            models, ["caernarfon-3b-cymraeg-instruct-v0-1"]
        )[0]
        self.assertEqual(caernarfon_instruct.min_output_tokens, 512)

    def test_ccc_files_match_catalog_counts_and_have_unique_ids(self):
        suites, revision = load_suites(ROOT / "config")
        ccc_suites = [suite for suite in suites if suite.group == "ccc"]
        all_ids: set[str] = set()
        for suite in ccc_suites:
            cases = load_cases(suite, root=ROOT, techiaith_revision=revision)
            self.assertEqual(len(cases), suite.expected_count, suite.id)
            self.assertEqual(len({case.id for case in cases}), len(cases), suite.id)
            self.assertFalse(all_ids & {case.id for case in cases}, suite.id)
            all_ids.update(case.id for case in cases)
        self.assertEqual(len(all_ids), 232)

        manifest = json.loads((ROOT / "data" / "ccc-audit" / "manifest.json").read_text())
        self.assertEqual(manifest["total_cases"], 232)
        for item in manifest["files"]:
            content = (ROOT / "data" / "ccc-audit" / item["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_ccc_reference_answers_receive_full_credit(self):
        suites, revision = load_suites(ROOT / "config")
        for suite in (suite for suite in suites if suite.group == "ccc"):
            for case in load_cases(suite, root=ROOT, techiaith_revision=revision):
                if suite.scorer == "choice":
                    prediction = case.expected
                else:
                    target = json.loads(case.expected)
                    if suite.scorer == "ccc_edit":
                        prediction = json.dumps(
                            {
                                "cywiriad": target["accepted_corrections"][0],
                                "categori": target["category"],
                                "rheol": target["rule"],
                                "esboniad": " ".join(group[0] for group in target["explanation_concepts"]),
                            }
                        )
                    elif suite.scorer == "ccc_fidelity":
                        prediction = json.dumps(
                            {
                                "crynodeb": " ".join(group[0] for group in target["summary_concepts"]),
                                "ffeithiau": target["fact_ids"],
                                "dyfyniad": target["quote"],
                            }
                        )
                    elif suite.scorer == "ccc_grounded":
                        prediction = json.dumps(
                            {
                                "ateb": " ".join(group[0] for group in target["answer_concepts"]),
                                "cyfeiriadau": target["source_ids"],
                            }
                        )
                    else:
                        prediction = json.dumps({"ffynonellau": target["source_ids"]})
                score, _ = score_prediction(
                    suite.scorer,
                    prediction,
                    case.expected,
                    case_payload={"metadata": case.metadata},
                )
                self.assertAlmostEqual(score, 1.0, msg=case.id)

    def test_ccc_coverage_is_balanced_across_failure_modes(self):
        suites, revision = load_suites(ROOT / "config")
        cases_by_suite = {
            suite.id: load_cases(suite, root=ROOT, techiaith_revision=revision)
            for suite in suites
            if suite.group == "ccc"
        }
        editing = cases_by_suite["ccc-cywiro-esbonio"]
        targets = [json.loads(case.expected) for case in editing]
        self.assertEqual(
            Counter(target["category"] for target in targets),
            Counter(
                {
                    "treiglad": 12,
                    "cysylltair": 12,
                    "arddodiad": 12,
                    "negyddiaeth": 12,
                    "cystrawen": 9,
                    "term": 7,
                }
            ),
        )
        self.assertGreaterEqual(len({target["rule"] for target in targets}), 10)
        citations = cases_by_suite["ccc-cyfeiriadau"]
        self.assertEqual(
            Counter(case.metadata["answerable"] for case in citations),
            {True: 24, False: 8},
        )
        for suite_id in ("ccc-cyfarwyddyd-cymraeg", "ccc-cyfarwyddyd-saesneg"):
            self.assertEqual(
                Counter(case.metadata["variant"] for case in cases_by_suite[suite_id]),
                {"rheolaeth": 8, "cymraeg_yn_unig": 8, "brawddeg_ccc": 8},
            )


if __name__ == "__main__":
    unittest.main()
