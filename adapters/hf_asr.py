#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys


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
        if args.language:
            result = recognizer(
                audio,
                generate_kwargs={"language": args.language, "task": "transcribe"},
            )
        else:
            result = recognizer(audio)
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
