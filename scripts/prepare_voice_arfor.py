#!/usr/bin/env python3
"""Echdynnu test_clean ARFOR pinned i faniffest preifat CymraegBench Voice."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

from aisteddfod_benchmarks.voice import normalize_transcript


REVISION = "0665ea3e755d9864985344512b7d346363b9b806"
EXPECTED_SHA256 = "9f21512368e70237ad96cd938f4e352ef5ea99a403f049b5ef67813a67633d06"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/private/voice"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA-256 anghywir: {digest}")
    import pyarrow.parquet as parquet

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "arfor-test-clean.jsonl"
    temporary = manifest.with_suffix(".tmp.jsonl")
    count = 0
    duration = 0.0
    with temporary.open("w", encoding="utf-8") as output:
        source_index = 0
        source = parquet.ParquetFile(args.parquet)
        for batch in source.iter_batches(
            batch_size=256,
            columns=("sentence", "audio", "accent", "language"),
        ):
            for row in batch.to_pylist():
                current_index = source_index
                source_index += 1
                sentence = str(row.get("sentence", ""))
                if row.get("language") != "cy" or not normalize_transcript(sentence):
                    continue
                case_id = f"arfor-test-{current_index:06d}"
                audio = audio_dir / f"{case_id}.wav"
                raw = row["audio"].get("bytes")
                if raw is None:
                    raise RuntimeError(f"Bytes sain ar goll: {case_id}")
                audio.write_bytes(raw)
                with wave.open(str(audio), "rb") as handle:
                    clip_duration = handle.getnframes() / handle.getframerate()
                item = {
                    "id": case_id,
                    "audio": f"audio/{audio.name}",
                    "reference": sentence,
                    "metadata": {
                        "accent": row.get("accent", ""),
                        "language": "cy",
                        "source_row": current_index,
                    },
                }
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                duration += clip_duration
                if args.limit > 0 and count >= args.limit:
                    break
            if args.limit > 0 and count >= args.limit:
                break
    temporary.replace(manifest)
    summary = {
        "dataset": "cymen-arfor/lleisiau-arfor",
        "revision": REVISION,
        "split": "test_clean",
        "license": "CC0-1.0",
        "source_sha256": digest,
        "count": count,
        "duration_seconds": round(duration, 3),
    }
    (args.output_dir / "arfor-test-clean.metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
