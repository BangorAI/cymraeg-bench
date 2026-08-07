from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
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
        "install_python": None,
        "uv": None,
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


def test_access_gate_heads_exact_largest_pinned_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    benchmark.root = tmp_path
    config = tmp_path / "config"
    config.mkdir()
    (config / "techiaith-asr-catalog.json").write_text(
        module.json.dumps({
            "models": [{"id": "techiaith/model-cy", "revision": "abc123"}],
        }),
        encoding="utf-8",
    )
    calls = []

    class FakeApi:
        def model_info(self, repo_id, *, revision, files_metadata):
            calls.append(("info", repo_id, revision, files_metadata))
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename="README.md", size=100),
                SimpleNamespace(rfilename="model.bin", size=20_000),
            ])

    def fake_url(*, repo_id, filename, revision):
        calls.append(("url", repo_id, filename, revision))
        return "https://huggingface.co/techiaith/model-cy/model.bin"

    def fake_metadata(url, *, token):
        calls.append(("head", url, token))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi,
        get_hf_file_metadata=fake_metadata,
        hf_hub_url=fake_url,
    ))

    assert benchmark.check_model_access() == []
    assert calls[-2:] == [
        ("url", "techiaith/model-cy", "model.bin", "abc123"),
        ("head", "https://huggingface.co/techiaith/model-cy/model.bin", True),
    ]


def test_access_gate_can_defer_missing_model_without_sleep(tmp_path: Path) -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    benchmark.status = tmp_path / "status.json"
    failures = [{
        "model": "techiaith/kaldi-cy-2601",
        "revision": "abc123",
        "error": "GatedRepoError: 403",
    }]
    benchmark.check_model_access = lambda: failures

    assert benchmark.wait_for_model_access(allow_deferred=True) == failures
    status = module.json.loads(benchmark.status.read_text(encoding="utf-8"))
    assert status["stage"] == "model_access_deferred"
    assert status["accessible_techiaith_models"] == 18


def test_deferred_repository_maps_to_exact_benchmark_model() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())

    assert benchmark.model_ids_for_repositories({"techiaith/kaldi-cy-2601"}) == {
        "techiaith-kaldi-cy-2601",
    }


def test_asset_preparation_skips_only_deferred_repository(tmp_path: Path) -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    benchmark.root = tmp_path
    benchmark.results_dir = tmp_path / "results" / "voice-v0.1"
    env_dir = tmp_path / "models" / "techiaith"
    env_dir.mkdir(parents=True)
    artifact = tmp_path / "kaldi.tar.gz"
    artifact.write_bytes(b"kaldi-pinned")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "techiaith-asr-catalog.json").write_text(json.dumps({
        "models": [
            {"id": "techiaith/kaldi-cy", "runtime": "vosk", "revision": "kaldi-rev"},
            {"id": "techiaith/kaldi-cy-2601", "runtime": "vosk", "revision": "2601-rev"},
        ],
    }))
    (env_dir / "assets.json").write_text(json.dumps({"assets": [{
        "id": "techiaith/kaldi-cy",
        "revision": "kaldi-rev",
        "runtime": "vosk",
        "artifact_path": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }]}))
    (env_dir / "env.sh").write_text(
        "export TECHIAITH_KALDI_DIR=/models/kaldi-cy\n",
        encoding="utf-8",
    )
    commands = []
    benchmark.run = lambda command, *, env=None, cwd=None: commands.append(command)

    env = benchmark.prepare_assets(
        {},
        skip_repositories={"techiaith/kaldi-cy-2601"},
    )

    assert commands[1][-2:] == ["--skip-model", "techiaith/kaldi-cy-2601"]
    assert env["TECHIAITH_KALDI_DIR"] == "/models/kaldi-cy"
    copied = tmp_path / "results" / "voice-v0.1" / "techiaith-specialized-assets.json"
    assert copied.is_file()
    public_manifest = json.loads(copied.read_text())
    assert public_manifest["schema_version"] == "techiaith-specialized-assets-v1"
    assert public_manifest["assets"][0]["artifact_filename"] == "kaldi.tar.gz"
    assert "artifact_path" not in public_manifest["assets"][0]
    assert str(tmp_path) not in copied.read_text()


def test_partial_gate_is_explicit_and_uses_only_completed_revisions() -> None:
    module = load_release_module()
    benchmark = module.ReleaseBenchmark(benchmark_args())
    revisions = {
        "bangorai-zipformer-cy": "bangor-rev",
        "techiaith-one": "tech-rev",
        "techiaith-deferred": "missing-rev",
        "dewibrynjones-kaldi-cy-2606": "dewi-rev",
    }
    rows = [
        {
            "model_id": "bangorai-zipformer-cy",
            "model_revision": "bangor-rev",
            "suite_id": module.SUITE_ID,
            "cases": str(module.EXPECTED_CASES),
            "successful": str(module.EXPECTED_CASES),
            "wer": "0.20",
        },
        {
            "model_id": "techiaith-one",
            "model_revision": "tech-rev",
            "suite_id": module.SUITE_ID,
            "cases": str(module.EXPECTED_CASES),
            "successful": str(module.EXPECTED_CASES),
            "wer": "0.25",
        },
        {
            "model_id": "dewibrynjones-kaldi-cy-2606",
            "model_revision": "dewi-rev",
            "suite_id": module.SUITE_ID,
            "cases": str(module.EXPECTED_CASES),
            "successful": str(module.EXPECTED_CASES),
            "wer": "0.30",
        },
    ]
    completed = [
        "bangorai-zipformer-cy",
        "techiaith-one",
        "dewibrynjones-kaldi-cy-2606",
    ]
    deferred = [{"model": "techiaith/deferred", "error": "403"}]

    gate = benchmark.benchmark_gate(
        rows,
        model_ids=completed,
        revisions=revisions,
        expected_techiaith_models=1,
        provisional=True,
        deferred=deferred,
    )

    assert gate["passed"] is True
    assert gate["provisional"] is True
    assert gate["deferred_models"] == deferred
    assert set(gate["model_revisions"]) == set(completed)


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


def test_resolve_uv_prefers_verified_binary_beside_python(tmp_path: Path) -> None:
    module = load_release_module()
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    system_python = tmp_path / "system-python"
    system_python.touch()
    python = binary_dir / "python"
    python.symlink_to(system_python)
    uv = binary_dir / "uv"
    uv.write_text("#!/bin/sh\necho 'uv 0.5.29'\n", encoding="utf-8")
    uv.chmod(0o755)
    benchmark = module.ReleaseBenchmark(benchmark_args(python=python))

    assert benchmark.resolve_uv() == uv.resolve()


def test_resolve_uv_rejects_unpinned_version(tmp_path: Path) -> None:
    module = load_release_module()
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\necho 'uv 9.9.9'\n", encoding="utf-8")
    uv.chmod(0o755)
    benchmark = module.ReleaseBenchmark(benchmark_args(uv=uv))

    with pytest.raises(RuntimeError, match="Fersiwn uv annisgwyl"):
        benchmark.resolve_uv()


def test_linux_sherpa_core_is_directly_locked_for_require_hashes() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert "sherpa-onnx-core==1.13.4 ; sys_platform == 'linux'" in project
    assert 'name = "sherpa-onnx-core"' in lock
    assert "sha256:367aa06cee90b3fd7959d4e071d6fc821710b859af399b4987e5c3119ee6ae2a" in lock


def test_locked_runtime_uses_separate_builder_without_changing_model_python(
    tmp_path: Path,
) -> None:
    module = load_release_module()
    runtime_python = tmp_path / "runtime" / "python"
    builder_python = tmp_path / "builder" / "python"
    benchmark = module.ReleaseBenchmark(benchmark_args(
        python=runtime_python,
        install_python=builder_python,
    ))
    benchmark.root = tmp_path
    benchmark.target = tmp_path / "target"
    (tmp_path / "uv.lock").write_text("lock\n")
    requirements = tmp_path / "runs" / "voice-release-requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("# locked\n")
    uv = tmp_path / "uv"
    uv.touch()
    benchmark.resolve_uv = lambda: uv
    benchmark.python_abi = lambda python: (3, 12)
    commands = []
    benchmark.run = lambda command, *, env=None, cwd=None: commands.append(command)

    benchmark.install_runtime()

    pip_command = next(command for command in commands if "pip" in command)
    assert pip_command[0] == str(builder_python)
    assert benchmark.args.python == runtime_python


def test_locked_runtime_rejects_builder_with_different_python_abi(
    tmp_path: Path,
) -> None:
    module = load_release_module()
    runtime_python = tmp_path / "runtime-python"
    builder_python = tmp_path / "builder-python"
    benchmark = module.ReleaseBenchmark(benchmark_args(
        python=runtime_python,
        install_python=builder_python,
    ))
    benchmark.root = tmp_path
    benchmark.target = tmp_path / "target"
    (tmp_path / "uv.lock").write_text("lock\n")
    requirements = tmp_path / "runs" / "voice-release-requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("# locked\n")
    uv = tmp_path / "uv"
    uv.touch()
    benchmark.resolve_uv = lambda: uv
    benchmark.python_abi = lambda python: (
        (3, 13) if python == builder_python else (3, 12)
    )
    benchmark.run = lambda command, *, env=None, cwd=None: None

    with pytest.raises(RuntimeError, match="ABI Python installer"):
        benchmark.install_runtime()
