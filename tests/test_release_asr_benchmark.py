from __future__ import annotations

import importlib.util
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_release_module():
    script = ROOT / "scripts" / "run_release_asr_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_release_asr_benchmark", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_args(**updates: object) -> Namespace:
    values = {
        "release_status": ROOT / "missing-release-status.json",
        "zipformer_root": ROOT / "missing-zipformer",
        "experiment": "best-all-causal-standard-b900",
        "arfor_parquet": ROOT / "missing-arfor.parquet",
        "dewi_model_dir": None,
        "python": Path("/usr/bin/python3"),
        "device": "cuda:0",
        "timeout": 600.0,
        "poll_interval": 60,
        "access_poll_interval": 600,
        "model_attempts": 3,
        "model_retry_delay": 0,
    }
    values.update(updates)
    return Namespace(**values)


def test_release_model_list_covers_all_techiaith_asr_models_once() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())

    model_ids, revisions = benchmark.model_ids()
    techiaith = [model_id for model_id in model_ids if model_id.startswith("techiaith-")]

    assert model_ids[0] == "bangorai-zipformer-cy"
    assert len(techiaith) == 19
    assert len(techiaith) == len(set(techiaith))
    assert all(revisions[model_id] for model_id in model_ids)


def test_release_model_list_adds_dewi_only_when_configured() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(
        benchmark_args(dewi_model_dir=ROOT / "models" / "dewi")
    )

    model_ids, revisions = benchmark.model_ids()

    assert model_ids[-1] == "dewibrynjones-kaldi-cy-2606"
    assert revisions[model_ids[-1]]


def test_all_release_models_have_nonempty_revisions() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(
        benchmark_args(dewi_model_dir=ROOT / "models" / "dewi")
    )

    model_ids, revisions = benchmark.model_ids()
    selected = {model_id: revisions[model_id] for model_id in model_ids}

    assert len(selected) == 21
    assert all(selected.values())


def test_bangorai_revision_is_the_exact_release_fingerprint() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    fingerprint = "sha256:" + "a" * 64
    benchmark.release_revision = fingerprint

    _, revisions = benchmark.model_ids()

    assert revisions["bangorai-zipformer-cy"] == fingerprint


def test_prepare_zipformer_rejects_release_without_fingerprint() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())

    with pytest.raises(RuntimeError, match="release_revision"):
        benchmark.prepare_zipformer({"selection": {}}, {})


def test_access_probe_selects_the_largest_real_artifact() -> None:
    module = load_release_module()
    siblings = [
        SimpleNamespace(rfilename="README.md", size=100),
        SimpleNamespace(rfilename="model.bin", size=20_000),
        SimpleNamespace(rfilename="config.json", size=500),
    ]

    assert module.probe_filename(siblings) == "model.bin"


def test_access_probe_rejects_metadata_without_sized_artifacts() -> None:
    module = load_release_module()

    with pytest.raises(ValueError, match="artifact model"):
        module.probe_filename([SimpleNamespace(rfilename="README.md", size=None)])


def test_model_command_retries_failed_cases_then_succeeds(tmp_path: Path) -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    benchmark.status = tmp_path / "status.json"
    attempts = []

    def flaky_run(command, *, env=None, cwd=None):
        attempts.append(list(command))
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(1, command)

    benchmark.run = flaky_run
    benchmark.run_model_command(
        ["voice", "run"],
        env={},
        model_id="model-cy",
        stage="model_benchmark",
        model_count=21,
    )

    assert len(attempts) == 3
    status = module.json.loads(benchmark.status.read_text(encoding="utf-8"))
    assert status["stage"] == "model_benchmark"
    assert status["attempt"] == 3


def test_model_command_raises_after_bounded_attempts(tmp_path: Path) -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args(model_attempts=2))
    benchmark.status = tmp_path / "status.json"
    attempts = []

    def broken_run(command, *, env=None, cwd=None):
        attempts.append(list(command))
        raise subprocess.CalledProcessError(7, command)

    benchmark.run = broken_run
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_model_command(
            ["voice", "run"],
            env={},
            model_id="model-cy",
            stage="model_benchmark",
            model_count=21,
        )

    assert len(attempts) == 2
