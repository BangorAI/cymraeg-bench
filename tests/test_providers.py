import os
import unittest
from unittest.mock import patch

from aisteddfod_benchmarks.config import ModelConfig
from aisteddfod_benchmarks import providers


def model(provider: str, **kwargs) -> ModelConfig:
    return ModelConfig(
        "m", "M", provider, "api-model", "KEY", "https://example.test/v1",
        input_usd_per_million=1, output_usd_per_million=2,
        **kwargs,
    )


class ProviderTests(unittest.TestCase):
    def test_openrouter_response_and_cost(self):
        def fake_post(url, headers, payload, **kwargs):
            self.assertEqual(headers["Authorization"], "Bearer secret")
            self.assertEqual(payload["seed"], 1)
            return {
                "model": "upstream/model",
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001},
            }, 42

        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", fake_post):
            response = providers.generate(model("openrouter"), "s", "u", max_tokens=8)
        self.assertEqual(response.text, "A")
        self.assertEqual(response.model, "upstream/model")
        self.assertEqual(response.cost_usd, 0.001)

    def test_openai_extracts_output_blocks(self):
        fake = (
            {
                "model": "api-model", "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": "Y"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
            9,
        )
        def fake_post(url, headers, payload, **kwargs):
            self.assertEqual(payload["max_output_tokens"], 16)
            return fake

        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", fake_post):
            response = providers.generate(model("openai"), "s", "u", max_tokens=8)
        self.assertEqual(response.text, "Y")
        self.assertEqual(response.cost_usd, 0.000004)

    def test_anthropic_omits_deprecated_temperature(self):
        def fake_post(url, headers, payload, **kwargs):
            self.assertNotIn("temperature", payload)
            self.assertEqual(payload["max_tokens"], 8)
            return {
                "model": "api-model",
                "content": [{"type": "text", "text": "A"}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "stop_reason": "end_turn",
            }, 7

        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", fake_post):
            response = providers.generate(model("anthropic"), "s", "u", max_tokens=8)
        self.assertEqual(response.text, "A")

    def test_anthropic_controls_effort_and_minimum_output_budget(self):
        def fake_post(url, headers, payload, **kwargs):
            self.assertEqual(payload["max_tokens"], 256)
            self.assertEqual(payload["output_config"], {"effort": "low"})
            return {
                "model": "api-model",
                "content": [{"type": "text", "text": "A"}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "stop_reason": "end_turn",
            }, 7

        configured = model(
            "anthropic", reasoning_effort="low", min_output_tokens=256
        )
        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", fake_post):
            response = providers.generate(configured, "s", "u", max_tokens=8)
        self.assertEqual(response.text, "A")

    def test_openrouter_can_disable_reasoning(self):
        def fake_post(url, headers, payload, **kwargs):
            self.assertEqual(
                payload["reasoning"], {"effort": "none", "exclude": True}
            )
            return {
                "model": "upstream/model",
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }, 4

        configured = model("openrouter", reasoning_effort="none")
        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", fake_post):
            response = providers.generate(configured, "s", "u", max_tokens=8)
        self.assertEqual(response.text, "A")

    def test_truncated_response_is_an_error(self):
        fake = (
            {
                "model": "upstream/model",
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 8},
            },
            4,
        )
        with patch.dict(os.environ, {"KEY": "secret"}), patch.object(providers, "_post_json", return_value=fake):
            with self.assertRaises(providers.ProviderError):
                providers.generate(model("openrouter"), "s", "u", max_tokens=8)


if __name__ == "__main__":
    unittest.main()
