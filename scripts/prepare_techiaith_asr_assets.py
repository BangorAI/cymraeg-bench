#!/usr/bin/env python3
"""Download pinned Techiaith Vosk and whisper.cpp assets safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import tarfile
from pathlib import Path


CPP_VARIABLES = {
    "techiaith/whisper-base-ft-commonvoice-cy-en-cpp":
        "TECHIAITH_WHISPER_BASE_CV_CY_EN_CPP_MODEL",
    "techiaith/whisper-base-ft-verbatim-cy-en-cpp":
        "TECHIAITH_WHISPER_BASE_VERBATIM_CY_EN_CPP_MODEL",
    "techiaith/whisper-base-ft-commonvoice-cy-cpp":
        "TECHIAITH_WHISPER_BASE_CV_CY_CPP_MODEL",
}
VOSK_VARIABLES = {
    "techiaith/kaldi-cy": "TECHIAITH_KALDI_DIR",
    "techiaith/kaldi-cy-2601": "TECHIAITH_KALDI_2601_DIR",
}


def artifact_metadata(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "artifact_path": str(path.resolve()),
        "artifact_size_bytes": path.stat().st_size,
        "artifact_sha256": digest.hexdigest(),
    }


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Tar member anniogel: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Tar member tu allan i'r destination: {member.name}")
        source.extractall(destination, members=members)


def vosk_model_root(destination: Path) -> Path:
    # Mae modelau Vosk diweddarach yn defnyddio am/, conf/ a graph/, ond mae'r
    # ddau fodel Kaldi Cymraeg yn yr archif pinned yn defnyddio'r layout Vosk
    # gwreiddiol, gwastad. Mae Vosk 0.3.45 yn cefnogi'r ddau.
    candidates = sorted(path.parent.parent for path in destination.rglob("am/final.mdl"))
    for candidate in candidates:
        if (candidate / "conf").is_dir():
            return candidate
    flat_required = ("final.mdl", "HCLG.fst", "words.txt", "mfcc.conf")
    flat_candidates = sorted(path.parent for path in destination.rglob("final.mdl"))
    for candidate in flat_candidates:
        if all((candidate / name).is_file() for name in flat_required):
            return candidate
    raise FileNotFoundError(f"Methu canfod gwraidd model Vosk o dan {destination}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=root / "config" / "techiaith-asr-catalog.json",
    )
    parser.add_argument("--models-dir", type=Path, default=root / "models" / "techiaith")
    parser.add_argument("--downloads-dir", type=Path, default=root / "downloads" / "techiaith-asr")
    parser.add_argument(
        "--whisper-cli",
        type=Path,
        default=root / "vendor" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="Repo Hugging Face i'w ohirio os nad oes mynediad eto (ailadroddadwy)",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    models = {item["id"]: item for item in catalog["models"]}
    expected = set(CPP_VARIABLES) | set(VOSK_VARIABLES)
    missing = expected - set(models)
    if missing:
        raise RuntimeError(f"Modelau asset ar goll o'r snapshot: {sorted(missing)}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.downloads_dir.mkdir(parents=True, exist_ok=True)
    environment: dict[str, Path] = {}
    assets: list[dict[str, str]] = []
    skipped = set(args.skip_model)

    for repo_id, variable in CPP_VARIABLES.items():
        if repo_id in skipped:
            continue
        item = models[repo_id]
        destination = args.models_dir / repo_id.split("/", 1)[1]
        destination.mkdir(parents=True, exist_ok=True)
        model_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename="ggml-model.bin",
            revision=item["revision"],
            local_dir=destination,
        )).resolve()
        environment[variable] = model_path
        assets.append({
            "id": repo_id,
            "revision": item["revision"],
            "runtime": item["runtime"],
            "path": str(model_path),
            **artifact_metadata(model_path),
        })

    for repo_id, variable in VOSK_VARIABLES.items():
        if repo_id in skipped:
            continue
        item = models[repo_id]
        slug = repo_id.split("/", 1)[1]
        download_dir = args.downloads_dir / slug
        destination = args.models_dir / slug
        download_dir.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        archive = Path(hf_hub_download(
            repo_id=repo_id,
            filename="model_cy.tar.gz",
            revision=item["revision"],
            local_dir=download_dir,
        ))
        marker = destination / ".cymraeg-bench-revision"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != item["revision"]:
            safe_extract(archive, destination)
            marker.write_text(item["revision"] + "\n", encoding="utf-8")
        model_path = vosk_model_root(destination).resolve()
        environment[variable] = model_path
        assets.append({
            "id": repo_id,
            "revision": item["revision"],
            "runtime": item["runtime"],
            "path": str(model_path),
            **artifact_metadata(archive),
        })

    environment["WHISPER_CPP_CLI"] = args.whisper_cli.resolve()
    env_path = args.models_dir / "env.sh"
    env_path.write_text(
        "# Crëwyd gan prepare_techiaith_asr_assets.py; peidiwch â commitio.\n"
        + "\n".join(
            f"export {name}={shlex.quote(str(path))}"
            for name, path in sorted(environment.items())
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = args.models_dir / "assets.json"
    manifest.write_text(
        json.dumps({"catalog": str(args.catalog), "assets": assets}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(env_path)
    print(manifest)


if __name__ == "__main__":
    main()
