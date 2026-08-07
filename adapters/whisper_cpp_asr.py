#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--language", default="cy")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()

    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    def transcribe(audio: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="cymraeg-bench-whisper-") as directory:
            output_base = Path(directory) / "prediction"
            process = subprocess.run(
                [
                    str(args.binary),
                    "--model",
                    str(args.model),
                    "--file",
                    str(audio),
                    "--language",
                    args.language,
                    "--threads",
                    str(args.threads),
                    "--output-txt",
                    "--output-file",
                    str(output_base),
                    "--no-prints",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            prediction = output_base.with_suffix(".txt")
            if process.returncode or not prediction.is_file():
                detail = process.stderr.strip()[-2000:]
                raise RuntimeError(
                    f"whisper.cpp exit={process.returncode}; {detail or 'dim allbwn testun'}"
                )
            return prediction.read_text(encoding="utf-8").strip()

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
