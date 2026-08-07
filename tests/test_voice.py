from __future__ import annotations

import json
import csv
import sys
import tempfile
import tomllib
import unittest
import wave
from pathlib import Path

from aisteddfod_benchmarks.voice import (
    VoiceModel,
    _command,
    build_listening_pack,
    build_listening_report,
    build_voice_report,
    edit_counts,
    normalize_transcript,
    run_voice_benchmark,
    transcript_metrics,
    wav_metrics,
)


def write_wav(path: Path, samples: list[int], sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in samples))


class VoiceMetricsTests(unittest.TestCase):
    def test_python_adapter_uses_the_benchmark_interpreter(self) -> None:
        model = VoiceModel(
            id="adapter",
            label="Adapter",
            task="asr",
            command=("python", "adapter.py"),
            protocol="jsonl",
            enabled=False,
            variables={},
            source="",
            revision="",
            license="",
        )
        self.assertEqual(_command(model, {}), [sys.executable, "adapter.py"])

    def test_normalization_and_edit_counts(self) -> None:
        self.assertEqual(normalize_transcript("Mae’n iawn!"), "MAE'N IAWN")
        self.assertEqual(normalize_transcript("."), "")
        self.assertEqual(
            edit_counts(["MAE", "HI", "YMA"], ["MAE", "FO", "YMA", "RWAN"]),
            {"ref": 3, "ins": 1, "del": 0, "sub": 1},
        )
        metrics = transcript_metrics("Bore da.", "bore da")
        self.assertEqual(metrics["word_counts"]["sub"], 0)

    def test_wav_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            write_wav(path, [0] * 8000 + [32767] * 8000)
            metrics = wav_metrics(path)
            self.assertEqual(metrics["duration_seconds"], 1.0)
            self.assertEqual(metrics["sample_rate"], 16000)
            self.assertEqual(metrics["clipped_ratio"], 0.5)
            self.assertEqual(metrics["silence_ratio"], 0.5)

    def test_techiaith_asr_catalog_is_exhaustive_and_pinned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        snapshot = json.loads(
            (root / "config" / "techiaith-asr-catalog.json").read_text(encoding="utf-8")
        )
        with (root / "config" / "voice-models.toml").open("rb") as handle:
            configured = tomllib.load(handle)["models"]
        by_source = {
            item.get("source", "").removeprefix("https://huggingface.co/"): item
            for item in configured
            if item["task"] == "asr" and "/techiaith/" in item.get("source", "")
        }
        expected = {item["id"]: item for item in snapshot["models"]}
        self.assertEqual(snapshot["count"], len(expected))
        self.assertEqual(set(by_source), set(expected))
        for model_id, item in expected.items():
            self.assertEqual(by_source[model_id]["revision"], item["revision"])


class VoiceRunTests(unittest.TestCase):
    def test_asr_run_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            write_wav(audio, [0] * 1600)
            manifest = root / "asr.jsonl"
            manifest.write_text(
                json.dumps({"id": "one", "audio": "sample.wav", "reference": "Bore da"}) + "\n",
                encoding="utf-8",
            )
            adapter = root / "adapter.py"
            adapter.write_text('import json; print(json.dumps({"text": "bore da"}))\n', encoding="utf-8")
            models = root / "models.toml"
            models.write_text(
                "\n".join([
                    "[[models]]",
                    'id = "mock-asr"',
                    'label = "Mock"',
                    'task = "asr"',
                    "enabled = true",
                    f'command = ["{sys.executable}", "{adapter}", "{{audio}}"]',
                ]) + "\n",
                encoding="utf-8",
            )
            suites = root / "suites.toml"
            suites.write_text(
                "\n".join([
                    "[[suites]]",
                    'id = "mock-suite"',
                    'label = "Mock"',
                    'task = "asr"',
                    'manifest = "asr.jsonl"',
                ]) + "\n",
                encoding="utf-8",
            )
            output = root / "results.jsonl"
            completed, errors = run_voice_benchmark(
                models_path=models,
                suites_path=suites,
                root=root,
                output=output,
            )
            self.assertEqual((completed, errors), (1, 0))
            self.assertEqual(
                run_voice_benchmark(models_path=models, suites_path=suites, root=root, output=output),
                (0, 0),
            )
            summary = build_voice_report(output, root / "report.md", root / "report.csv")
            self.assertEqual(summary[0]["wer"], 0.0)
            self.assertEqual(summary[0]["cer"], 0.0)

            models.write_text(
                models.read_text(encoding="utf-8") + 'revision = "v2"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                run_voice_benchmark(
                    models_path=models,
                    suites_path=suites,
                    root=root,
                    output=output,
                ),
                (1, 0),
            )
            summary = build_voice_report(output, root / "report.md", root / "report.csv")
            self.assertEqual(
                {(row["model_revision"], row["cases"]) for row in summary},
                {("", 1), ("v2", 1)},
            )

            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "model_id": "mock-asr",
                    "model_revision": "v2",
                    "suite_id": "mock-suite",
                    "task": "asr",
                    "case_id": "one",
                    "status": "error",
                    "latency_seconds": 0.0,
                }) + "\n")
            summary = build_voice_report(output, root / "report.md", root / "report.csv")
            v2 = next(row for row in summary if row["model_revision"] == "v2")
            self.assertEqual((v2["cases"], v2["successful"], v2["failure_rate"]), (1, 0, 1.0))

    def test_jsonl_adapter_is_loaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            write_wav(audio, [0] * 1600)
            manifest = root / "asr.jsonl"
            manifest.write_text("\n".join([
                json.dumps({"id": "one", "audio": "sample.wav", "reference": "Bore da"}),
                json.dumps({"id": "two", "audio": "sample.wav", "reference": "Bore da"}),
            ]) + "\n", encoding="utf-8")
            marker = root / "loads.txt"
            adapter = root / "adapter.py"
            adapter.write_text(
                "\n".join([
                    "import json, sys",
                    "from pathlib import Path",
                    "marker = Path(sys.argv[1])",
                    "marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')",
                    "print(json.dumps({'ready': True}), flush=True)",
                    "for line in sys.stdin:",
                    "    request = json.loads(line)",
                    "    if request.get('command') == 'shutdown': break",
                    "    print(json.dumps({'text': 'bore da'}), flush=True)",
                ]) + "\n",
                encoding="utf-8",
            )
            models = root / "models.toml"
            models.write_text("\n".join([
                "[[models]]",
                'id = "persistent-asr"',
                'label = "Persistent"',
                'task = "asr"',
                'protocol = "jsonl"',
                "enabled = true",
                f'command = ["{sys.executable}", "{adapter}", "{marker}"]',
            ]) + "\n", encoding="utf-8")
            suites = root / "suites.toml"
            suites.write_text("\n".join([
                "[[suites]]",
                'id = "mock-suite"',
                'label = "Mock"',
                'task = "asr"',
                'manifest = "asr.jsonl"',
            ]) + "\n", encoding="utf-8")
            completed, errors = run_voice_benchmark(
                models_path=models, suites_path=suites, root=root, output=root / "results.jsonl"
            )
            self.assertEqual((completed, errors), (2, 0))
            self.assertEqual(marker.read_text(), "1")

    def test_crashed_jsonl_adapter_restarts_and_failed_case_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            write_wav(audio, [0] * 1600)
            manifest = root / "asr.jsonl"
            manifest.write_text("\n".join([
                json.dumps({"id": "one", "audio": "sample.wav", "reference": "Bore da"}),
                json.dumps({"id": "two", "audio": "sample.wav", "reference": "Bore da"}),
            ]) + "\n", encoding="utf-8")
            marker = root / "crashed"
            adapter = root / "adapter.py"
            adapter.write_text("\n".join([
                "import json, sys",
                "from pathlib import Path",
                "marker = Path(sys.argv[1])",
                "print(json.dumps({'ready': True}), flush=True)",
                "for line in sys.stdin:",
                "    request = json.loads(line)",
                "    if request.get('command') == 'shutdown': break",
                "    if not marker.exists():",
                "        marker.write_text('once')",
                "        raise SystemExit(7)",
                "    print(json.dumps({'text': 'bore da'}), flush=True)",
            ]) + "\n", encoding="utf-8")
            models = root / "models.toml"
            models.write_text("\n".join([
                "[[models]]",
                'id = "crashy-asr"',
                'label = "Crashy"',
                'task = "asr"',
                'protocol = "jsonl"',
                "enabled = true",
                f'command = ["{sys.executable}", "{adapter}", "{marker}"]',
            ]) + "\n", encoding="utf-8")
            suites = root / "suites.toml"
            suites.write_text("\n".join([
                "[[suites]]",
                'id = "mock-suite"',
                'label = "Mock"',
                'task = "asr"',
                'manifest = "asr.jsonl"',
            ]) + "\n", encoding="utf-8")
            output = root / "results.jsonl"
            self.assertEqual(
                run_voice_benchmark(
                    models_path=models, suites_path=suites, root=root, output=output
                ),
                (1, 1),
            )
            self.assertEqual(
                run_voice_benchmark(
                    models_path=models, suites_path=suites, root=root, output=output
                ),
                (1, 0),
            )
            report = build_voice_report(output, root / "report.md", root / "report.csv")
            self.assertEqual(
                (report[0]["cases"], report[0]["successful"], report[0]["failure_rate"]),
                (2, 2, 0.0),
            )

    def test_blind_listening_pack_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "tts.wav"
            write_wav(audio, [0, 1000, -1000] * 100)
            results = root / "results.jsonl"
            results.write_text(json.dumps({
                "task": "tts",
                "status": "ok",
                "model_id": "voice-a",
                "suite_id": "tts",
                "case_id": "one",
                "text": "Bore da",
                "output_wav": str(audio),
            }) + "\n", encoding="utf-8")
            ratings_path, key_path = build_listening_pack(results, root / "pack")
            with ratings_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0].update({
                "listener_id": "g01",
                "dealladwyedd_1_5": "5",
                "naturioldeb_1_5": "4",
                "ynganu_1_5": "5",
                "rhythm_1_5": "4",
            })
            with ratings_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            summary = build_listening_report(
                [ratings_path], key_path, root / "listening.md", root / "listening.csv"
            )
            self.assertEqual(summary[0]["model_id"], "voice-a")
            self.assertEqual(summary[0]["naturioldeb"], 4.0)


if __name__ == "__main__":
    unittest.main()
