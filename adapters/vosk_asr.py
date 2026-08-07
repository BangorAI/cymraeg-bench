#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    from vosk import KaldiRecognizer, Model

    model = Model(str(args.model_dir))

    def transcribe(audio: Path) -> str:
        with wave.open(str(audio), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Mae Vosk yn disgwyl WAV mono PCM16")
            recognizer = KaldiRecognizer(model, handle.getframerate())
            while data := handle.readframes(4000):
                recognizer.AcceptWaveform(data)
        return str(json.loads(recognizer.FinalResult()).get("text", ""))

    if args.serve_jsonl:
        print('{"ready":true}', flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    break
                print(json.dumps({"text": transcribe(Path(request["audio"]))}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
        return
    if args.audio is None:
        parser.error("mae angen --audio heb --serve-jsonl")
    print(json.dumps({"text": transcribe(args.audio)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
