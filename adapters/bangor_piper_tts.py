#!/usr/bin/env python3
from __future__ import annotations

import argparse
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort
    from techiaith.g2p import BangorG2P

    text = args.text_file.read_text(encoding="utf-8").strip()
    ids = BangorG2P(english_mode="native").text_to_ids(text)
    session = ort.InferenceSession(str(args.model))
    audio = session.run(None, {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([len(ids)], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        "sid": np.array([args.speaker], dtype=np.int64),
    })[0].squeeze()
    audio = (audio * (32767 / max(0.01, float(np.max(np.abs(audio)))))).astype("<i2")
    with wave.open(str(args.output_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(audio.tobytes())


if __name__ == "__main__":
    main()
