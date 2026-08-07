from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from aisteddfod_benchmarks.voice import (
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
