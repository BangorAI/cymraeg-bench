#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any


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


def ctc_decoder_labels(
    model_path: str | Path,
    tokenizer: Any,
    *,
    vocab_size: int,
    blank_token_id: int,
) -> list[str]:
    """Reconstruct a CTC alphabet from the acoustic model's pinned vocabulary.

    Some historical Wav2Vec2+KenLM repositories contain an ``alphabet.json``
    copied from a different tokenizer.  The acoustic ``vocab.json`` and model
    blank id remain the authoritative mapping for the output logits.
    """
    model_path = Path(model_path)
    vocabulary = json.loads((model_path / "vocab.json").read_text())
    labels: list[str | None] = [None] * vocab_size
    special_tokens = {"[UNK]", "[PAD]", "<s>", "</s>"}
    delimiter = getattr(tokenizer, "word_delimiter_token", "|")

    for token, token_id in vocabulary.items():
        token_id = int(token_id)
        if not 0 <= token_id < vocab_size or token in special_tokens:
            continue
        labels[token_id] = " " if token == delimiter else str(token)

    if not 0 <= blank_token_id < vocab_size:
        raise ValueError(
            f"CTC blank token id {blank_token_id} outside vocabulary of {vocab_size}"
        )
    labels[blank_token_id] = ""

    used = {label for label in labels if label is not None}
    for token_id, label in enumerate(labels):
        if label is not None:
            continue
        token = str(tokenizer.convert_ids_to_tokens(token_id))
        if token == delimiter:
            candidate = " "
        elif token == getattr(tokenizer, "unk_token", "[UNK]"):
            candidate = "⁇"
        elif token == getattr(tokenizer, "pad_token", "[PAD]"):
            candidate = f"<unused-pad-{token_id}>"
        else:
            candidate = token
        if not candidate or candidate in used:
            candidate = f"<unused-{token_id}>"
        labels[token_id] = candidate
        used.add(candidate)

    result = [str(label) for label in labels]
    if len(result) != len(set(result)):
        raise ValueError("Reconstructed CTC alphabet contains duplicate labels")
    return result


def repair_misaligned_lm_decoder(recognizer: Any, model_path: str | Path) -> bool:
    """Replace a packaged LM decoder only when it cannot match model logits."""
    decoder = getattr(recognizer, "decoder", None)
    alphabet = getattr(decoder, "_alphabet", None)
    packaged_labels = getattr(alphabet, "labels", None)
    if packaged_labels is None:
        return False

    vocab_size = int(recognizer.model.config.vocab_size)
    if len(packaged_labels) == vocab_size:
        return False

    blank_token_id = int(recognizer.model.config.pad_token_id)
    labels = ctc_decoder_labels(
        model_path,
        recognizer.tokenizer,
        vocab_size=vocab_size,
        blank_token_id=blank_token_id,
    )
    language_model_path = Path(model_path) / "language_model"
    if not language_model_path.is_dir():
        raise ValueError(
            "Packaged CTC alphabet does not match logits and no pinned language model exists"
        )

    from pyctcdecode import BeamSearchDecoderCTC
    from pyctcdecode.alphabet import Alphabet
    from pyctcdecode.language_model import LanguageModel

    language_model = LanguageModel.load_from_dir(str(language_model_path))
    recognizer.decoder = BeamSearchDecoderCTC(
        Alphabet.build_alphabet(labels),
        language_model=language_model,
    )
    print(
        "Rebuilt mismatched packaged CTC alphabet "
        f"({len(packaged_labels)} labels -> {vocab_size} logits) from pinned vocab/LM",
        file=sys.stderr,
        flush=True,
    )
    return True


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
    repair_misaligned_lm_decoder(recognizer, model_path)

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
