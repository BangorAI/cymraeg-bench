#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language")
    args = parser.parse_args()
    from transformers import pipeline

    device = os.getenv("VOICE_BENCH_DEVICE", "cuda:0")
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        revision=args.revision,
        device=device,
    )
    if args.language:
        result = recognizer(
            args.audio,
            generate_kwargs={"language": args.language, "task": "transcribe"},
        )
    else:
        result = recognizer(args.audio)
    print(json.dumps({"text": result["text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
