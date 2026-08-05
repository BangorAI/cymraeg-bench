import unittest
from pathlib import Path

from aisteddfod_benchmarks.config import load_models, load_suites, select_models


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
                "caernarfon-3b", "mwydryn",
            },
        )
        self.assertEqual(len([suite for suite in suites if suite.enabled]), 19)
        self.assertEqual(revision, "47839d2147c97fd2f10a52dd36751608e5fa36bf")

    def test_explicit_selection_can_include_disabled_local_model(self):
        models = load_models(ROOT / "config")
        self.assertEqual(
            select_models(models, ["mwydryn"])[0].api_model,
            "BangorAI/phi2-mwydryn-1",
        )


if __name__ == "__main__":
    unittest.main()
