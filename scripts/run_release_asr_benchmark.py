#!/usr/bin/env python3
"""Run the sealed CymraegBench ASR release comparison after model finalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path


SUITE_ID = "arfor-test-clean-v0.1"
EXPECTED_CASES = 3445
UV_VERSION = "0.5.29"


def probe_filename(siblings: list[object]) -> str:
    """Choose a real large artifact so gated metadata alone cannot pass."""
    candidates = [
        sibling
        for sibling in siblings
        if int(getattr(sibling, "size", 0) or 0) > 0
        and str(getattr(sibling, "rfilename", ""))
    ]
    if not candidates:
        raise ValueError("Nid oes artifact model â maint hysbys")
    selected = max(
        candidates,
        key=lambda sibling: (
            int(getattr(sibling, "size", 0) or 0),
            str(getattr(sibling, "rfilename", "")),
        ),
    )
    return str(getattr(selected, "rfilename"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReleaseBenchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.status = self.root / "runs" / "voice-v0.1-status.json"
        self.output = self.root / "runs" / "voice-v0.1.jsonl"
        self.results_dir = self.root / "results" / "voice-v0.1"
        self.target = self.root / ".voice-packages-release"
        self.release_status = args.release_status.resolve()
        self.completed_models: list[str] = []
        self.release_revision: str | None = None

    def write_status(self, stage: str, **extra: object) -> None:
        payload = {
            "updated_at": utc_now(),
            "stage": stage,
            "suite": SUITE_ID,
            "expected_cases_per_model": EXPECTED_CASES,
            "release_status": str(self.release_status),
            "completed_models": self.completed_models,
            **extra,
        }
        self.status.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.status)

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        print(f"[{utc_now()}] Rhedeg: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=cwd or self.root, env=env, check=True)

    def run_model_command(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        model_id: str,
        stage: str,
        model_count: int,
    ) -> None:
        """Retry only failed/missing cases; the voice runner skips durable successes."""
        for attempt in range(1, self.args.model_attempts + 1):
            self.write_status(
                stage,
                current_model=model_id,
                model_count=model_count,
                attempt=attempt,
                max_attempts=self.args.model_attempts,
            )
            try:
                self.run(command, env=env)
                return
            except subprocess.CalledProcessError as error:
                if attempt >= self.args.model_attempts:
                    raise
                self.write_status(
                    "model_retry_wait",
                    current_model=model_id,
                    model_count=model_count,
                    failed_stage=stage,
                    failed_attempt=attempt,
                    max_attempts=self.args.model_attempts,
                    returncode=error.returncode,
                    retry_delay_seconds=self.args.model_retry_delay,
                )
                print(
                    f"[{utc_now()}] {model_id}: methodd {stage} attempt "
                    f"{attempt}/{self.args.model_attempts}; yn ailafael mewn "
                    f"{self.args.model_retry_delay}s",
                    flush=True,
                )
                time.sleep(self.args.model_retry_delay)

    def resolve_uv(self) -> Path:
        """Resolve and verify the exact uv used to export the locked runtime."""
        candidates: list[Path] = []
        if self.args.uv is not None:
            candidates.append(self.args.uv.expanduser())
        # Peidio â defnyddio resolve() yma: mae python mewn venv fel arfer yn
        # symlink i /usr/bin/python, ond mae'r uv perthnasol wrth ymyl y
        # symlink yn <venv>/bin.
        python_path = self.args.python.expanduser()
        if not python_path.is_absolute():
            python_path = (Path.cwd() / python_path).absolute()
        candidates.append(python_path.parent / "uv")
        on_path = shutil.which("uv")
        if on_path:
            candidates.append(Path(on_path))
        selected = next((path.resolve() for path in candidates if path.is_file()), None)
        if selected is None:
            raise FileNotFoundError(
                f"uv {UV_VERSION} ar goll; gosodwch ef wrth ymyl --python neu rhowch --uv"
            )
        version = subprocess.run(
            [str(selected), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version.split()[:2] != ["uv", UV_VERSION]:
            raise RuntimeError(
                f"Fersiwn uv annisgwyl: {version!r}; disgwyl uv {UV_VERSION}"
            )
        return selected

    def wait_for_release(self) -> dict[str, object]:
        self.write_status("waiting_for_release_bundle")
        last_report = 0.0
        while True:
            if self.release_status.is_file():
                status = json.loads(self.release_status.read_text(encoding="utf-8"))
                stage = status.get("stage")
                if stage == "complete":
                    return status
                if stage == "failed":
                    raise RuntimeError(f"Methodd finalization y model: {status.get('error')}")
            now = time.monotonic()
            if now - last_report >= 600:
                print(f"[{utc_now()}] Yn dal i aros am y release bundle", flush=True)
                last_report = now
            time.sleep(self.args.poll_interval)

    def check_model_access(self) -> list[dict[str, str]]:
        """HEAD the largest pinned artifact in every Techiaith ASR repo."""
        from huggingface_hub import HfApi, hf_hub_download

        catalog = json.loads(
            (self.root / "config" / "techiaith-asr-catalog.json").read_text(
                encoding="utf-8"
            )
        )
        failures: list[dict[str, str]] = []
        api = HfApi()
        for item in catalog["models"]:
            repo_id = str(item["id"])
            revision = str(item["revision"])
            try:
                info = api.model_info(
                    repo_id,
                    revision=revision,
                    files_metadata=True,
                )
                filename = probe_filename(list(info.siblings))
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    dry_run=True,
                )
            except Exception as error:
                failures.append({
                    "model": repo_id,
                    "revision": revision,
                    "error": f"{type(error).__name__}: {str(error).splitlines()[0]}",
                })
        return failures

    def wait_for_model_access(self) -> None:
        """Wait without touching the test manifest until every weight is readable."""
        while True:
            self.write_status("checking_model_access")
            failures = self.check_model_access()
            if not failures:
                self.write_status("model_access_verified", techiaith_models=19)
                print(f"[{utc_now()}] Mynediad i bob un o'r 19 model wedi'i wirio", flush=True)
                return
            self.write_status(
                "waiting_for_model_access",
                inaccessible_models=failures,
            )
            names = ", ".join(item["model"] for item in failures)
            print(
                f"[{utc_now()}] Yn aros am fynediad Hugging Face: {names}",
                flush=True,
            )
            time.sleep(self.args.access_poll_interval)

    def install_runtime(self) -> dict[str, str]:
        lock_hash = sha256_file(self.root / "uv.lock")
        marker = self.target / ".cymraeg-bench-lock-sha256"
        requirements = self.root / "runs" / "voice-release-requirements.txt"
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != lock_hash:
            self.write_status("installing_locked_runtime", lock_sha256=lock_hash)
            uv = self.resolve_uv()
            self.run([
                str(uv),
                "export",
                "--locked",
                "--extra",
                "voice-asr",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ])
            self.target.mkdir(parents=True, exist_ok=True)
            self.run([
                str(self.args.python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--require-hashes",
                "--target",
                str(self.target),
                "--requirement",
                str(requirements),
            ])
            marker.write_text(lock_hash + "\n", encoding="utf-8")
        env = os.environ.copy()
        pythonpath = [str(self.target), str(self.root / "src")]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env.update({
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "VOICE_BENCH_DEVICE": self.args.device,
            "TOKENIZERS_PARALLELISM": "false",
        })
        return env

    def prepare_assets(self, env: dict[str, str]) -> dict[str, str]:
        self.write_status("preparing_pinned_assets")
        self.run([str(self.root / "scripts" / "bootstrap_whisper_cpp.sh")], env=env)
        self.run([
            str(self.args.python),
            str(self.root / "scripts" / "prepare_techiaith_asr_assets.py"),
        ], env=env)
        env_file = self.root / "models" / "techiaith" / "env.sh"
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if not line.startswith("export "):
                continue
            assignment = shlex.split(line.removeprefix("export "))[0]
            name, value = assignment.split("=", 1)
            env[name] = value
        if self.args.dewi_model_dir:
            env["DEWI_KALDI_2606_DIR"] = str(self.args.dewi_model_dir.resolve())
        return env

    def prepare_manifest(self, env: dict[str, str]) -> None:
        self.write_status("preparing_sealed_manifest")
        self.run([
            str(self.args.python),
            str(self.root / "scripts" / "prepare_voice_arfor.py"),
            "--parquet",
            str(self.args.arfor_parquet),
        ], env=env)
        metadata_path = self.root / "data" / "private" / "voice" / "arfor-test-clean.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = self.root / "data" / "private" / "voice" / "arfor-test-clean.jsonl"
        line_count = sum(1 for _ in manifest.open(encoding="utf-8"))
        if metadata.get("count") != EXPECTED_CASES or line_count != EXPECTED_CASES:
            raise RuntimeError(
                f"Manifest annisgwyl: metadata={metadata.get('count')}, lines={line_count}"
            )

    def prepare_zipformer(
        self,
        release: dict[str, object],
        env: dict[str, str],
    ) -> dict[str, str]:
        selection = release.get("selection")
        if not isinstance(selection, dict):
            raise RuntimeError("Selection ar goll o release-finalization-status.json")
        release_revision = str(release.get("release_revision", ""))
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", release_revision):
            raise RuntimeError(f"release_revision annilys: {release_revision!r}")
        self.release_revision = release_revision
        epoch = int(selection["epoch"])
        avg = int(selection["avg"])
        chunk = int(selection["chunk"])
        left_context = int(selection["left_context"])
        method = str(selection["method"])
        beam = int(selection["beam"])
        name = f"epoch-{epoch}-avg-{avg}-chunk-{chunk}-left-{left_context}"
        exp = self.args.zipformer_root / "exp" / self.args.experiment
        sources = {
            "encoder.onnx": exp / f"encoder-{name}.int8.onnx",
            "decoder.onnx": exp / f"decoder-{name}.onnx",
            "joiner.onnx": exp / f"joiner-{name}.int8.onnx",
            "tokens.txt": (
                self.args.zipformer_root
                / "data"
                / "cy-best-all"
                / "lang_bpe_500"
                / "tokens.txt"
            ),
        }
        destination = self.root / "models" / "bangorai-zipformer-cy"
        destination.mkdir(parents=True, exist_ok=True)
        for filename, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / filename)
        env.update({
            "CYMRAEG_ZIPFORMER_DIR": str(destination),
            "CYMRAEG_ZIPFORMER_DECODING_METHOD": method,
            "CYMRAEG_ZIPFORMER_MAX_ACTIVE_PATHS": str(beam if method == "modified_beam_search" else 4),
            "CYMRAEG_ZIPFORMER_REVISION": release_revision,
        })
        return env

    def model_ids(self) -> tuple[list[str], dict[str, str]]:
        with (self.root / "config" / "voice-models.toml").open("rb") as source:
            models = tomllib.load(source)["models"]
        revisions = {
            str(item["id"]): os.path.expandvars(str(item.get("revision", "")))
            for item in models
        }
        if self.release_revision is not None:
            revisions["bangorai-zipformer-cy"] = self.release_revision
        techiaith = [
            str(item["id"])
            for item in models
            if item["task"] == "asr"
            and str(item.get("source", "")).startswith("https://huggingface.co/techiaith/")
        ]
        priority = [
            "techiaith-whisper-large-cy-en-2607",
            "techiaith-whisper-large-cy-en-2607-ct2",
            "techiaith-whisper-large-v3-verbatim",
            "techiaith-whisper-large-v3-verbatim-ct2",
            "techiaith-kaldi-cy-2601",
        ]
        ordered = [item for item in priority if item in techiaith]
        ordered.extend(sorted(set(techiaith) - set(ordered)))
        all_models = ["bangorai-zipformer-cy", *ordered]
        if self.args.dewi_model_dir:
            all_models.append("dewibrynjones-kaldi-cy-2606")
        if len(techiaith) != 19:
            raise RuntimeError(f"Disgwyl 19 model Techiaith; cafwyd {len(techiaith)}")
        return all_models, revisions

    def voice_command(self, model_id: str, max_cases: int | None = None) -> list[str]:
        command = [
            str(self.args.python),
            "-m",
            "aisteddfod_benchmarks.cli",
            "voice",
            "run",
            "--root",
            str(self.root),
            "--model-ids",
            model_id,
            "--suite-ids",
            SUITE_ID,
            "--timeout",
            str(self.args.timeout),
            "--output",
            str(self.output),
        ]
        if max_cases is not None:
            command.extend(["--max-cases", str(max_cases)])
        return command

    def report(self, env: dict[str, str]) -> list[dict[str, str]]:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.results_dir / "leaderboard.csv"
        self.run([
            str(self.args.python),
            "-m",
            "aisteddfod_benchmarks.cli",
            "voice",
            "report",
            str(self.output),
            "--markdown",
            str(self.results_dir / "leaderboard.md"),
            "--csv",
            str(csv_path),
        ], env=env)
        with csv_path.open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))

    def execute(self) -> None:
        self.wait_for_model_access()
        release = self.wait_for_release()
        env = self.install_runtime()
        env = self.prepare_assets(env)
        self.prepare_manifest(env)
        env = self.prepare_zipformer(release, env)
        self.run([
            str(self.args.python),
            "-m",
            "aisteddfod_benchmarks.cli",
            "voice",
            "validate",
            "--root",
            str(self.root),
        ], env=env)

        model_ids, revisions = self.model_ids()
        for model_id in model_ids:
            self.run_model_command(
                self.voice_command(model_id, max_cases=1),
                env=env,
                model_id=model_id,
                stage="model_preflight",
                model_count=len(model_ids),
            )
            self.run_model_command(
                self.voice_command(model_id),
                env=env,
                model_id=model_id,
                stage="model_benchmark",
                model_count=len(model_ids),
            )
            self.completed_models.append(model_id)
            self.report(env)

        rows = self.report(env)
        current_rows = {
            row["model_id"]: row
            for row in rows
            if row["suite_id"] == SUITE_ID
            and row["model_revision"] == revisions.get(row["model_id"], "")
        }
        missing = [model_id for model_id in model_ids if model_id not in current_rows]
        incomplete = [
            model_id
            for model_id, row in current_rows.items()
            if model_id in model_ids
            and (int(row["cases"]) != EXPECTED_CASES or int(row["successful"]) != EXPECTED_CASES)
        ]
        if missing or incomplete:
            raise RuntimeError(f"Leaderboard anghyflawn: missing={missing}, incomplete={incomplete}")
        bangor_wer = float(current_rows["bangorai-zipformer-cy"]["wer"])
        tech_rows = {
            model_id: row
            for model_id, row in current_rows.items()
            if model_id.startswith("techiaith-")
        }
        if len(tech_rows) != 19:
            raise RuntimeError(
                f"Disgwyl 19 canlyniad Techiaith; cafwyd {len(tech_rows)}"
            )
        strongest_id, strongest_row = min(
            tech_rows.items(), key=lambda item: float(item[1]["wer"])
        )
        strongest_wer = float(strongest_row["wer"])
        passed = bangor_wer < strongest_wer
        gate = {
            "passed": passed,
            "bangorai_wer": bangor_wer,
            "strongest_techiaith_model": strongest_id,
            "strongest_techiaith_wer": strongest_wer,
            "margin": strongest_wer - bangor_wer,
            "techiaith_models": len(tech_rows),
            "model_revisions": {
                model_id: revisions[model_id]
                for model_id in model_ids
            },
        }
        self.write_status("complete" if passed else "gate_failed", gate=gate)
        print(json.dumps(gate, indent=2), flush=True)
        if not passed:
            raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-status", type=Path, required=True)
    parser.add_argument("--zipformer-root", type=Path, required=True)
    parser.add_argument("--experiment", default="best-all-causal-standard-b900")
    parser.add_argument("--arfor-parquet", type=Path, required=True)
    parser.add_argument("--dewi-model-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--access-poll-interval", type=int, default=600)
    parser.add_argument("--model-attempts", type=int, default=3)
    parser.add_argument("--model-retry-delay", type=int, default=30)
    args = parser.parse_args()
    if args.model_attempts < 1:
        parser.error("Rhaid i --model-attempts fod yn bositif")
    if args.model_retry_delay < 0:
        parser.error("Ni all --model-retry-delay fod yn negatif")
    benchmark = ReleaseBenchmark(args)
    try:
        benchmark.execute()
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code != 2:
            benchmark.write_status(
                "failed",
                error=f"{type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    main()
