#!/usr/bin/env python3
"""Prefetch pinned HF snapshots used directly by ASR adapters, resumably."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SPECIALIZED_ASSETS = {
    "techiaith/whisper-base-ft-commonvoice-cy-en-cpp",
    "techiaith/whisper-base-ft-verbatim-cy-en-cpp",
    "techiaith/whisper-base-ft-commonvoice-cy-cpp",
    "techiaith/kaldi-cy",
    "techiaith/kaldi-cy-2601",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, stage: str, **details: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now(), "stage": stage, **details}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prefetch(
    *,
    catalog_path: Path,
    status_path: Path,
    skipped: set[str],
    downloader: Callable[..., object],
) -> list[str]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    models = [
        item
        for item in catalog["models"]
        if item["id"] not in SPECIALIZED_ASSETS | skipped
    ]
    completed: list[str] = []
    for index, item in enumerate(models, 1):
        repo_id = str(item["id"])
        revision = str(item["revision"])
        write_status(
            status_path,
            "downloading",
            current_model=repo_id,
            current_revision=revision,
            model_index=index,
            model_count=len(models),
            completed_models=completed,
        )
        print(f"[{utc_now()}] PREFETCH {index}/{len(models)} {repo_id}", flush=True)
        try:
            downloader(repo_id=repo_id, revision=revision, token=True)
        except BaseException as error:
            write_status(
                status_path,
                "failed",
                current_model=repo_id,
                current_revision=revision,
                model_index=index,
                model_count=len(models),
                completed_models=completed,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        completed.append(repo_id)
    write_status(
        status_path,
        "complete",
        model_count=len(models),
        completed_models=completed,
    )
    return completed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "config" / "techiaith-asr-catalog.json",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=root / "runs" / "techiaith-prefetch-status.json",
    )
    parser.add_argument("--skip-model", action="append", default=[])
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    prefetch(
        catalog_path=args.catalog,
        status_path=args.status,
        skipped=set(args.skip_model),
        downloader=snapshot_download,
    )


if __name__ == "__main__":
    main()
