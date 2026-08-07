#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def runtime_options(requested_device: str) -> tuple[str, int, str]:
    """Choose a quality-preserving CTranslate2 backend for the benchmark."""
    if requested_device.startswith("cuda"):
        device = "cuda"
        device_index = int(requested_device.partition(":")[2] or "0")
        # Do not quantize Techiaith comparators merely to make them cheaper to
        # beat.  The release benchmark runs after training and the RTX 4090 can
        # hold these models in FP16.
        default_compute_type = "float16"
    else:
        device = requested_device
        device_index = 0
        default_compute_type = "int8"
    compute_type = os.getenv("VOICE_BENCH_CT2_COMPUTE_TYPE", default_compute_type)
    return device, device_index, compute_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="cy")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=args.model, revision=args.revision)
    requested_device = os.getenv("VOICE_BENCH_DEVICE", "cuda:0")
    device, device_index, compute_type = runtime_options(requested_device)
    recognizer = WhisperModel(
        model_path,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        cpu_threads=int(os.getenv("VOICE_BENCH_THREADS", "4")),
        num_workers=1,
    )

    def transcribe(audio: Path) -> str:
        segments, _ = recognizer.transcribe(
            str(audio),
            language=args.language,
            task="transcribe",
            beam_size=args.beam_size,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    if args.serve_jsonl:
        print('{"ready":true}', flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    break
                print(
                    json.dumps(
                        {"text": transcribe(Path(request["audio"]))},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return
    if args.audio is None:
        parser.error("mae angen --audio heb --serve-jsonl")
    print(json.dumps({"text": transcribe(args.audio)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
