from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_techiaith_asr_assets",
    ROOT / "scripts" / "prepare_techiaith_asr_assets.py",
)
assert SPEC and SPEC.loader
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)

PREFETCH_SPEC = importlib.util.spec_from_file_location(
    "prefetch_techiaith_asr_snapshots",
    ROOT / "scripts" / "prefetch_techiaith_asr_snapshots.py",
)
assert PREFETCH_SPEC and PREFETCH_SPEC.loader
PREFETCH = importlib.util.module_from_spec(PREFETCH_SPEC)
PREFETCH_SPEC.loader.exec_module(PREFETCH)


def add_bytes(archive: tarfile.TarFile, name: str, content: bytes = b"x") -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


class VoiceAssetTests(unittest.TestCase):
    def test_safe_extract_and_find_vosk_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.tar.gz"
            with tarfile.open(archive, "w:gz") as destination:
                add_bytes(destination, "nested/model/am/final.mdl")
                add_bytes(destination, "nested/model/conf/model.conf")
            extracted = root / "extracted"
            extracted.mkdir()
            ASSETS.safe_extract(archive, extracted)
            self.assertEqual(
                ASSETS.vosk_model_root(extracted),
                extracted / "nested" / "model",
            )

    def test_finds_original_flat_vosk_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "nested" / "model"
            model.mkdir(parents=True)
            for name in ("final.mdl", "HCLG.fst", "words.txt", "mfcc.conf"):
                (model / name).write_bytes(b"x")
            self.assertEqual(ASSETS.vosk_model_root(root), model)

    def test_rejects_incomplete_flat_vosk_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "final.mdl").write_bytes(b"x")
            with self.assertRaises(FileNotFoundError):
                ASSETS.vosk_model_root(root)

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.tar.gz"
            with tarfile.open(archive, "w:gz") as destination:
                add_bytes(destination, "../escape")
            extracted = root / "extracted"
            extracted.mkdir()
            with self.assertRaisesRegex(ValueError, "tu allan"):
                ASSETS.safe_extract(archive, extracted)
            self.assertFalse((root / "escape").exists())

    def test_whisper_cpp_runtime_is_pinned(self) -> None:
        values = {}
        for line in (ROOT / "config" / "voice-runtimes.env").read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value
        self.assertRegex(values["WHISPER_CPP_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            values["WHISPER_CPP_REPO"],
            "https://github.com/ggml-org/whisper.cpp.git",
        )

    def test_whisper_cpp_bootstrap_finds_nvcc_outside_path(self) -> None:
        script = (ROOT / "scripts" / "bootstrap_whisper_cpp.sh").read_text()
        self.assertIn("/usr/local/cuda/bin/nvcc", script)
        self.assertIn('-DCMAKE_CUDA_COMPILER="$CUDACXX"', script)
        self.assertIn("Methu canfod nvcc", script)

    def test_snapshot_prefetch_uses_pinned_revisions_and_skips_special_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            status = root / "status.json"
            catalog.write_text(json.dumps({"models": [
                {
                    "id": "techiaith/whisper-base-ft-commonvoice-cy-cpp",
                    "revision": "cpp-rev",
                },
                {"id": "techiaith/model-one", "revision": "one-rev"},
                {"id": "techiaith/model-deferred", "revision": "deferred-rev"},
            ]}))
            calls = []

            completed = PREFETCH.prefetch(
                catalog_path=catalog,
                status_path=status,
                skipped={"techiaith/model-deferred"},
                downloader=lambda **kwargs: calls.append(kwargs),
            )

            self.assertEqual(completed, ["techiaith/model-one"])
            self.assertEqual(calls, [{
                "repo_id": "techiaith/model-one",
                "revision": "one-rev",
                "token": True,
            }])
            payload = json.loads(status.read_text())
            self.assertEqual(payload["stage"], "complete")
            self.assertEqual(payload["completed_models"], completed)


if __name__ == "__main__":
    unittest.main()
