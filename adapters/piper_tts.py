#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-wav", type=Path, required=True)
    args = parser.parse_args()
    text = args.text_file.read_text(encoding="utf-8")
    subprocess.run(
        ["piper", "--model", str(args.model), "--output_file", str(args.output_wav)],
        input=text,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
