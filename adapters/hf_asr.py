#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path


def read_pcm16_wav(path: str | Path) -> dict[str, object]:
    """Return the in-memory ASR pipeline input without requiring ffmpeg."""
    import numpy as np

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"Disgwyl WAV PCM16; cafwyd sample width {sample_width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        if samples.size % channels:
            raise ValueError("Nid yw nifer samples WAV yn lluosrif o'r sianeli")
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return {"array": samples, "sampling_rate": sample_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--language")
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    device = os.getenv("VOICE_BENCH_DEVICE", "cuda:0")
    # Passing revision= to transformers pins the acoustic model, but its
    # pyctcdecode integration historically fetched KenLM files from main.
    # Resolve the complete snapshot first so every component uses one SHA.
    model_path = snapshot_download(repo_id=args.model, revision=args.revision)
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
    )
    def transcribe(audio: str) -> str:
        pipeline_input = read_pcm16_wav(audio)
        if args.language:
            result = recognizer(
                pipeline_input,
                generate_kwargs={"language": args.language, "task": "transcribe"},
            )
        else:
            result = recognizer(pipeline_input)
        return str(result["text"])

    if args.serve_jsonl:
        print('{"ready":true}', flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    break
                print(json.dumps({"text": transcribe(str(request["audio"]))}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
        return
    if not args.audio:
        parser.error("mae angen --audio heb --serve-jsonl")
    print(json.dumps({"text": transcribe(args.audio)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
