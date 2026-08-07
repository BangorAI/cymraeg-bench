#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--decoding-method",
        choices=("greedy_search", "modified_beam_search"),
        default=os.getenv("CYMRAEG_ZIPFORMER_DECODING_METHOD", "greedy_search"),
    )
    parser.add_argument(
        "--max-active-paths",
        type=int,
        default=int(os.getenv("CYMRAEG_ZIPFORMER_MAX_ACTIVE_PATHS", "4")),
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import sherpa_onnx

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(args.model_dir / "tokens.txt"),
        encoder=str(args.model_dir / "encoder.onnx"),
        decoder=str(args.model_dir / "decoder.onnx"),
        joiner=str(args.model_dir / "joiner.onnx"),
        num_threads=args.num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        enable_endpoint_detection=False,
    )
    def transcribe(audio: Path) -> str:
        with wave.open(str(audio), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Mae'r addasydd sherpa yn disgwyl WAV mono PCM16")
            sample_rate = handle.getframerate()
            samples = np.frombuffer(
                handle.readframes(handle.getnframes()), dtype="<i2"
            ).astype(np.float32) / 32768.0
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        return str(getattr(result, "text", result))

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
