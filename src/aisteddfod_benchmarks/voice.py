from __future__ import annotations

import array
import csv
import json
import math
import os
import re
import random
import selectors
import shutil
import subprocess
import time
import tomllib
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VoiceModel:
    id: str
    label: str
    task: str
    command: tuple[str, ...]
    protocol: str
    enabled: bool
    variables: dict[str, str]
    source: str
    revision: str
    license: str


@dataclass(frozen=True)
class VoiceSuite:
    id: str
    label: str
    task: str
    manifest: Path
    language: str
    source: str
    revision: str
    license: str


def _table_items(path: Path, key: str) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return list(tomllib.load(handle).get(key, []))


def load_voice_models(path: Path) -> list[VoiceModel]:
    return [
        VoiceModel(
            id=str(item["id"]),
            label=str(item["label"]),
            task=str(item["task"]),
            command=tuple(str(part) for part in item["command"]),
            protocol=str(item.get("protocol", "oneshot")),
            enabled=bool(item.get("enabled", False)),
            variables={str(k): os.path.expandvars(str(v)) for k, v in item.get("variables", {}).items()},
            source=str(item.get("source", "")),
            revision=str(item.get("revision", "")),
            license=str(item.get("license", "")),
        )
        for item in _table_items(path, "models")
    ]


def load_voice_suites(path: Path, root: Path) -> list[VoiceSuite]:
    return [
        VoiceSuite(
            id=str(item["id"]),
            label=str(item["label"]),
            task=str(item["task"]),
            manifest=(root / str(item["manifest"])).resolve(),
            language=str(item.get("language", "cy")),
            source=str(item.get("source", "")),
            revision=str(item.get("revision", "")),
            license=str(item.get("license", "")),
        )
        for item in _table_items(path, "suites")
    ]


def normalize_transcript(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("’", "'").upper()
    value = "".join(character if character.isalnum() or character == "'" else " " for character in value)
    return re.sub(r"\s+", " ", value).strip()


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    ops = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], ops[i][0] = i, "del"
    for j in range(1, columns):
        costs[0][j], ops[0][j] = j, "ins"
    for i in range(1, rows):
        for j in range(1, columns):
            candidates = [
                (costs[i - 1][j] + 1, "del"),
                (costs[i][j - 1] + 1, "ins"),
                (costs[i - 1][j - 1] + int(reference[i - 1] != hypothesis[j - 1]),
                 "ok" if reference[i - 1] == hypothesis[j - 1] else "sub"),
            ]
            costs[i][j], ops[i][j] = min(candidates, key=lambda item: item[0])
    counts = {"ref": len(reference), "ins": 0, "del": 0, "sub": 0}
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = ops[i][j]
        if operation in {"ok", "sub"}:
            i -= 1
            j -= 1
        elif operation == "del":
            i -= 1
        elif operation == "ins":
            j -= 1
        else:
            raise RuntimeError("Llwybr Levenshtein annilys")
        if operation in counts:
            counts[operation] += 1
    return counts


def transcript_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref = normalize_transcript(reference)
    hyp = normalize_transcript(hypothesis)
    words = edit_counts(ref.split(), hyp.split())
    chars = edit_counts(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return {
        "reference_normalized": ref,
        "hypothesis_normalized": hyp,
        "word_counts": words,
        "char_counts": chars,
    }


def wav_metrics(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)
    result: dict[str, Any] = {
        "duration_seconds": frame_count / sample_rate if sample_rate else 0.0,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "clipped_ratio": None,
        "silence_ratio": None,
    }
    if sample_width != 2 or not frames:
        return result
    samples = array.array("h")
    samples.frombytes(frames)
    if os.sys.byteorder != "little":
        samples.byteswap()
    maximum = 32767
    silence_threshold = maximum * math.pow(10.0, -50.0 / 20.0)
    result["clipped_ratio"] = sum(abs(value) >= maximum for value in samples) / len(samples)
    result["silence_ratio"] = sum(abs(value) <= silence_threshold for value in samples) / len(samples)
    return result


def _read_jsonl(path: Path, *, require_id: bool = True) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                item = json.loads(line)
                if require_id and "id" not in item:
                    raise ValueError(f"ID ar goll yn {path}:{line_number}")
                rows.append(item)
    return rows


def _select(items: Iterable[Any], wanted: set[str] | None) -> list[Any]:
    selected = [item for item in items if wanted is None or item.id in wanted]
    if wanted:
        missing = wanted - {item.id for item in selected}
        if missing:
            raise ValueError("IDs anhysbys: " + ", ".join(sorted(missing)))
    return selected


def _command(model: VoiceModel, values: dict[str, str]) -> list[str]:
    replacements = {**model.variables, **values}
    try:
        return [part.format_map(replacements) for part in model.command]
    except KeyError as exc:
        raise ValueError(f"Newidyn gorchymyn ar goll ar gyfer {model.id}: {exc.args[0]}") from exc


def _prediction(stdout: str) -> str:
    value = stdout.strip()
    if not value:
        return ""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])
    return value


def _read_process_json(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("Nid oes stdout ar yr addasydd JSONL")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        if not selector.select(timeout):
            raise TimeoutError(f"Dim ymateb gan yr addasydd ar ôl {timeout:.1f}s")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        code = process.poll()
        raise RuntimeError(f"Daeth yr addasydd JSONL i ben (cod {code})")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("Rhaid i ymateb yr addasydd JSONL fod yn wrthrych")
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _start_jsonl_adapter(model: VoiceModel, root: Path, timeout: float) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        _command(model, {"root": str(root)}),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = _read_process_json(process, timeout)
        if ready.get("ready") is not True:
            raise RuntimeError(f"Ymateb parod annilys gan {model.id}: {ready}")
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise
    return process


def _jsonl_request(
    process: subprocess.Popen[str], request: dict[str, str], timeout: float
) -> dict[str, Any]:
    if process.stdin is None:
        raise RuntimeError("Nid oes stdin ar yr addasydd JSONL")
    process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
    process.stdin.flush()
    return _read_process_json(process, timeout)


def _stop_jsonl_adapter(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            process.stdin.write('{"command":"shutdown"}\n')
            process.stdin.flush()
        process.wait(timeout=10)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_voice_catalog(models_path: Path, suites_path: Path, root: Path) -> tuple[int, list[str]]:
    models = load_voice_models(models_path)
    suites = load_voice_suites(suites_path, root)
    errors = []
    for task in ("asr", "tts"):
        if not any(item.task == task for item in models):
            errors.append(f"Dim model {task}")
        if not any(item.task == task for item in suites):
            errors.append(f"Dim set {task}")
    for model in models:
        if model.protocol not in {"oneshot", "jsonl"}:
            errors.append(f"Protocol anhysbys ar gyfer {model.id}: {model.protocol}")
    if len({item.id for item in models}) != len(models):
        errors.append("ID model llais wedi'i ddyblygu")
    if len({item.id for item in suites}) != len(suites):
        errors.append("ID set llais wedi'i ddyblygu")
    for suite in suites:
        if not suite.manifest.is_file():
            errors.append(f"Manifest ar goll: {suite.manifest}")
            continue
        try:
            rows = _read_jsonl(suite.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Manifest annilys {suite.id}: {exc}")
            continue
        for row in rows:
            if suite.task == "asr" and not {"audio", "reference"} <= row.keys():
                errors.append(f"Mae {suite.id}/{row['id']} heb audio/reference")
            if suite.task == "tts" and "text" not in row:
                errors.append(f"Mae {suite.id}/{row['id']} heb text")
    return len(models) + len(suites), errors


def run_voice_benchmark(
    *,
    models_path: Path,
    suites_path: Path,
    root: Path,
    output: Path,
    model_ids: set[str] | None = None,
    suite_ids: set[str] | None = None,
    max_cases: int | None = None,
    timeout: float = 600.0,
) -> tuple[int, int]:
    models = _select(load_voice_models(models_path), model_ids)
    if model_ids is None:
        models = [model for model in models if model.enabled]
    suites = _select(load_voice_suites(suites_path, root), suite_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = output.with_suffix("").with_name(output.stem + "-audio")
    completed_keys = set()
    if output.is_file():
        completed_keys = {
            (row["model_id"], row["suite_id"], row["case_id"])
            for row in _read_jsonl(output, require_id=False)
            if row.get("status") == "ok"
        }
    completed = errors = 0
    for model in models:
        adapter: subprocess.Popen[str] | None = None
        try:
            for suite in suites:
                if model.task != suite.task:
                    continue
                cases = _read_jsonl(suite.manifest)
                if max_cases is not None:
                    cases = cases[:max_cases]
                for case in cases:
                    key = (model.id, suite.id, str(case["id"]))
                    if key in completed_keys:
                        continue
                    row: dict[str, Any] = {
                        "schema_version": "voice-v0.1",
                        "model_id": model.id,
                        "model_label": model.label,
                        "model_source": model.source,
                        "model_revision": model.revision,
                        "model_license": model.license,
                        "suite_id": suite.id,
                        "suite_source": suite.source,
                        "suite_revision": suite.revision,
                        "suite_license": suite.license,
                        "task": suite.task,
                        "case_id": str(case["id"]),
                        "metadata": case.get("metadata", {}),
                    }
                    started = time.perf_counter()
                    try:
                        if suite.task == "asr":
                            audio = (suite.manifest.parent / str(case["audio"])).resolve()
                            if not audio.is_file():
                                raise FileNotFoundError(audio)
                            audio_stats = wav_metrics(audio)
                            if model.protocol == "jsonl":
                                if adapter is None:
                                    adapter = _start_jsonl_adapter(model, root, timeout)
                                    started = time.perf_counter()
                                response = _jsonl_request(adapter, {"audio": str(audio)}, timeout)
                                prediction = str(response.get("text", ""))
                            else:
                                command = _command(model, {"audio": str(audio), "root": str(root)})
                                process = subprocess.run(
                                    command,
                                    capture_output=True,
                                    text=True,
                                    timeout=timeout,
                                    check=True,
                                )
                                prediction = _prediction(process.stdout)
                            latency = time.perf_counter() - started
                            row.update(transcript_metrics(str(case["reference"]), prediction))
                            row.update({
                                "reference": str(case["reference"]),
                                "prediction": prediction,
                                "audio": str(audio),
                                "audio_metrics": audio_stats,
                                "latency_seconds": latency,
                                "real_time_factor": latency / audio_stats["duration_seconds"]
                                if audio_stats["duration_seconds"] else None,
                            })
                        else:
                            destination = audio_dir / model.id / suite.id / f"{case['id']}.wav"
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            text_file = destination.with_suffix(".txt")
                            text_file.write_text(str(case["text"]) + "\n", encoding="utf-8")
                            command = _command(model, {
                                "text": str(case["text"]),
                                "text_file": str(text_file),
                                "output_wav": str(destination),
                                "root": str(root),
                            })
                            subprocess.run(
                                command,
                                capture_output=True,
                                text=True,
                                timeout=timeout,
                                check=True,
                            )
                            latency = time.perf_counter() - started
                            if not destination.is_file():
                                raise RuntimeError(f"Ni chrewyd {destination}")
                            audio_stats = wav_metrics(destination)
                            row.update({
                                "text": str(case["text"]),
                                "output_wav": str(destination),
                                "audio_metrics": audio_stats,
                                "latency_seconds": latency,
                                "real_time_factor": latency / audio_stats["duration_seconds"]
                                if audio_stats["duration_seconds"] else None,
                            })
                        row["status"] = "ok"
                        completed += 1
                    except Exception as exc:  # each case is a durable checkpoint
                        row.update({
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "latency_seconds": time.perf_counter() - started,
                        })
                        errors += 1
                    _append_jsonl(output, row)
        finally:
            _stop_jsonl_adapter(adapter)
    return completed, errors


def _ratio(counts: dict[str, int]) -> float | None:
    return (counts["ins"] + counts["del"] + counts["sub"]) / counts["ref"] if counts["ref"] else None


def build_voice_report(results: Path, markdown: Path, csv_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(results, require_id=False)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["suite_id"], row["task"]), []).append(row)
    summary = []
    for (model_id, suite_id, task), items in sorted(groups.items()):
        successful = [item for item in items if item.get("status") == "ok"]
        result: dict[str, Any] = {
            "model_id": model_id,
            "suite_id": suite_id,
            "task": task,
            "cases": len(items),
            "successful": len(successful),
            "failure_rate": (len(items) - len(successful)) / len(items),
            "latency_mean_seconds": sum(item["latency_seconds"] for item in successful) / len(successful)
            if successful else None,
            "rtf_mean": sum(item["real_time_factor"] for item in successful if item.get("real_time_factor") is not None)
            / len([item for item in successful if item.get("real_time_factor") is not None])
            if any(item.get("real_time_factor") is not None for item in successful) else None,
            "wer": None,
            "cer": None,
        }
        if task == "asr" and successful:
            word_counts = {name: sum(item["word_counts"][name] for item in successful) for name in ("ref", "ins", "del", "sub")}
            char_counts = {name: sum(item["char_counts"][name] for item in successful) for name in ("ref", "ins", "del", "sub")}
            result["wer"] = _ratio(word_counts)
            result["cer"] = _ratio(char_counts)
        summary.append(result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["model_id"])
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# CymraegBench Voice v0.1",
        "",
        "| Model | Set | Tasg | Achosion | WER | CER | RTF | Methiannau |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        percentage = lambda value: "—" if value is None else f"{100 * value:.2f}%"
        number = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {item['model_id']} | {item['suite_id']} | {item['task']} | {item['successful']}/{item['cases']} "
            f"| {percentage(item['wer'])} | {percentage(item['cer'])} | {number(item['rtf_mean'])} "
            f"| {percentage(item['failure_rate'])} |"
        )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_listening_pack(results: Path, output_dir: Path, seed: int = 1) -> tuple[Path, Path]:
    rows = [
        row for row in _read_jsonl(results, require_id=False)
        if row.get("task") == "tts" and row.get("status") == "ok"
    ]
    if not rows:
        raise ValueError("Dim canlyniadau TTS llwyddiannus")
    random.Random(seed).shuffle(rows)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / "allwedd-breifat.jsonl"
    ratings_path = output_dir / "sgoriau-gwrando.csv"
    key_rows = []
    rating_rows = []
    for index, row in enumerate(rows, 1):
        blind_id = f"llais-{index:05d}"
        source = Path(row["output_wav"])
        destination = audio_dir / f"{blind_id}.wav"
        shutil.copy2(source, destination)
        key_rows.append({
            "id": blind_id,
            "model_id": row["model_id"],
            "suite_id": row["suite_id"],
            "case_id": row["case_id"],
            "text": row["text"],
            "source_wav": str(source),
        })
        rating_rows.append({
            "listener_id": "",
            "id": blind_id,
            "audio": f"audio/{blind_id}.wav",
            "text": row["text"],
            "dealladwyedd_1_5": "",
            "naturioldeb_1_5": "",
            "ynganu_1_5": "",
            "rhythm_1_5": "",
            "nodiadau": "",
        })
    with key_path.open("w", encoding="utf-8") as handle:
        for row in key_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with ratings_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    return ratings_path, key_path


def build_listening_report(
    ratings_paths: list[Path], key_path: Path, markdown: Path, csv_path: Path
) -> list[dict[str, Any]]:
    key = {row["id"]: row for row in _read_jsonl(key_path)}
    dimensions = ("dealladwyedd_1_5", "naturioldeb_1_5", "ynganu_1_5", "rhythm_1_5")
    grouped: dict[str, list[dict[str, float]]] = {}
    listeners: dict[str, set[str]] = {}
    for ratings_path in ratings_paths:
        with ratings_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("id") not in key:
                    raise ValueError(f"ID gwrando anhysbys: {row.get('id')}")
                try:
                    scores = {dimension: float(row[dimension]) for dimension in dimensions}
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Sgôr annilys ar gyfer {row.get('id')}") from exc
                if any(not 1.0 <= value <= 5.0 for value in scores.values()):
                    raise ValueError(f"Rhaid i bob sgôr fod rhwng 1 a 5: {row.get('id')}")
                model_id = key[row["id"]]["model_id"]
                grouped.setdefault(model_id, []).append(scores)
                listeners.setdefault(model_id, set()).add(row.get("listener_id") or ratings_path.stem)
    summary = []
    for model_id, scores in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model_id": model_id,
            "ratings": len(scores),
            "listeners": len(listeners[model_id]),
        }
        for dimension in dimensions:
            result[dimension.removesuffix("_1_5")] = sum(row[dimension] for row in scores) / len(scores)
        summary.append(result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["model_id"])
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Prawf gwrando CymraegBench Voice v0.1",
        "",
        "| Model | Gwrandawyr | Sgoriau | Dealladwyedd | Naturioldeb | Ynganu | Rhythm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model_id']} | {row['listeners']} | {row['ratings']} | "
            f"{row['dealladwyedd']:.2f} | {row['naturioldeb']:.2f} | "
            f"{row['ynganu']:.2f} | {row['rhythm']:.2f} |"
        )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
