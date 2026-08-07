#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    from vosk import KaldiRecognizer, Model

    with wave.open(str(args.audio), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Mae Vosk yn disgwyl WAV mono PCM16")
        recognizer = KaldiRecognizer(Model(str(args.model_dir)), handle.getframerate())
        while data := handle.readframes(4000):
            recognizer.AcceptWaveform(data)
    result = json.loads(recognizer.FinalResult())
    print(json.dumps({"text": result.get("text", "")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
