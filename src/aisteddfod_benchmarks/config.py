from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    provider: str
    api_model: str
    api_key_env: str
    base_url: str
    enabled: bool = True
    tier: str = "scoreboard"
    base_url_env: str | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    reasoning_effort: str | None = None
    min_output_tokens: int | None = None

    @property
    def resolved_base_url(self) -> str:
        if self.base_url_env and os.getenv(self.base_url_env):
            return os.environ[self.base_url_env].rstrip("/")
        return self.base_url.rstrip("/")

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or None


@dataclass(frozen=True)
class SuiteConfig:
    id: str
    label: str
    source: str
    adapter: str
    scorer: str
    expected_count: int
    group: str
    enabled: bool = True
    dataset: str | None = None
    dataset_config: str | None = None
    split: str | None = None
    path: str | None = None
    max_tokens: int = 64
    repetitions: int = 1
    trust_remote_code: bool = False
    scorer_command_env: str | None = None
    validator_only: bool = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_models(config_dir: Path | None = None) -> list[ModelConfig]:
    directory = config_dir or project_root() / "config"
    raw = _load_toml(directory / "models.toml")
    return [ModelConfig(**item) for item in raw["models"]]


def load_suites(config_dir: Path | None = None) -> tuple[list[SuiteConfig], str]:
    directory = config_dir or project_root() / "config"
    raw = _load_toml(directory / "suites.toml")
    return [SuiteConfig(**item) for item in raw["suites"]], raw["techiaith_revision"]


def select_models(models: list[ModelConfig], ids: list[str] | None) -> list[ModelConfig]:
    if not ids:
        return [model for model in models if model.enabled]
    wanted = set(ids)
    found = [model for model in models if model.id in wanted]
    missing = wanted - {model.id for model in found}
    if missing:
        raise ValueError(f"Modelau anhysbys: {', '.join(sorted(missing))}")
    return found


def select_suites(suites: list[SuiteConfig], ids: list[str] | None) -> list[SuiteConfig]:
    if not ids:
        return [suite for suite in suites if suite.enabled and not suite.validator_only]
    wanted = set(ids)
    found = [suite for suite in suites if suite.id in wanted]
    missing = wanted - {suite.id for suite in found}
    if missing:
        raise ValueError(f"Setiau anhysbys: {', '.join(sorted(missing))}")
    invalid = [suite.id for suite in found if suite.validator_only]
    if invalid:
        raise ValueError(
            "Mae'r rhain yn feincnodau dilysydd yn unig, nid tasgau model: "
            + ", ".join(invalid)
        )
    return found
