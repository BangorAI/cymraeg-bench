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
    import numpy as np
    import sherpa_onnx

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(args.model_dir / "tokens.txt"),
        encoder=str(args.model_dir / "encoder.onnx"),
        decoder=str(args.model_dir / "decoder.onnx"),
        joiner=str(args.model_dir / "joiner.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )
    with wave.open(str(args.audio), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("Mae'r addasydd sherpa yn disgwyl WAV mono PCM16")
        sample_rate = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result(stream)
    print(json.dumps({"text": getattr(result, "text", str(result))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
