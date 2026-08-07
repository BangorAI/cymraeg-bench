from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .config import (
    load_models,
    load_suites,
    project_root,
    select_models,
    select_suites,
)
from .finalize import finalize_output_errors
from .report import build_ccc_report, build_leaderboard, build_report
from .runner import planned_calls, run_evaluation
from .voice import (
    build_listening_pack,
    build_listening_report,
    build_voice_report,
    run_voice_benchmark,
    validate_voice_catalog,
)


def _ids(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _catalog(config_dir: Path):
    models = load_models(config_dir)
    suites, revision = load_suites(config_dir)
    return models, suites, revision


def _remote_openrouter_ids() -> set[str]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "aisteddfod-benchmarks/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {item["id"] for item in payload["data"]}


def command_validate(args: argparse.Namespace) -> int:
    models, suites, revision = _catalog(args.config_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if len({model.id for model in models}) != len(models):
        errors.append("Mae ID model wedi'i ddyblygu")
    if len({suite.id for suite in suites}) != len(suites):
        errors.append("Mae ID set brawf wedi'i ddyblygu")
    for model in models:
        if model.enabled and not os.getenv(model.api_key_env):
            message = f"{model.api_key_env} ar goll ({model.label})"
            (errors if args.require_keys else warnings).append(message)
    if args.check_remote:
        try:
            available = _remote_openrouter_ids()
            for model in models:
                if model.provider == "openrouter" and model.api_model not in available:
                    errors.append(f"Nid yw {model.api_model} yng nghatalog OpenRouter")
        except Exception as exc:  # network diagnostics belong in CLI output
            errors.append(f"Methu darllen catalog OpenRouter: {exc}")
    print(f"{len(models)} model; {len(suites)} set; Techiaith {revision[:12]}")
    for warning in warnings:
        print(f"RHYBUDD: {warning}")
    for error in errors:
        print(f"GWALL: {error}", file=sys.stderr)
    print("Dilys." if not errors else "Methodd y dilysu.")
    return 1 if errors else 0


def command_plan(args: argparse.Namespace) -> int:
    models, suites, _ = _catalog(args.config_dir)
    selected_models = select_models(models, _ids(args.models))
    selected_suites = select_suites(suites, _ids(args.suites))
    total, rows = planned_calls(selected_models, selected_suites, args.max_cases)
    print(f"Modelau: {len(selected_models)}")
    print(f"Setiau: {len(selected_suites)}")
    for suite_id, calls in rows:
        print(f"  {suite_id:<36} {calls:>9,} galwad")
    print(f"Cyfanswm: {total:,} galwad fodel")
    print("Nid oes galwad API wedi'i gwneud.")
    return 0


def command_run(args: argparse.Namespace) -> int:
    models, suites, revision = _catalog(args.config_dir)
    selected_models = select_models(models, _ids(args.models))
    selected_suites = select_suites(suites, _ids(args.suites))
    missing = [
        f"{model.api_key_env} ({model.label})"
        for model in selected_models
        if model.provider != "openai_compatible" and not os.getenv(model.api_key_env)
    ]
    if missing:
        raise SystemExit("Allweddi API ar goll: " + ", ".join(missing))
    run_id, path, completed, errors = run_evaluation(
        models=selected_models,
        suites=selected_suites,
        root=args.config_dir.parent,
        techiaith_revision=revision,
        run_id=args.run_id,
        max_cases=args.max_cases,
        max_usd=args.max_usd,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"Rhediad {run_id}: {completed} wedi cwblhau; {errors} gwall")
    print(path)
    return 1 if errors else 0


def command_report(args: argparse.Namespace) -> int:
    summary = build_report(args.database, args.markdown, args.csv)
    print(f"{len(summary)} rhes grynhoi")
    print(args.markdown or args.database.with_suffix(".md"))
    print(args.csv or args.database.with_suffix(".csv"))
    return 0


def command_ccc_report(args: argparse.Namespace) -> int:
    summary = build_ccc_report(args.database, args.markdown, args.csv)
    print(f"{len(summary)} rhes dimensiwn CCC")
    print(args.markdown or args.database.with_name(f"{args.database.stem}-ccc.md"))
    print(args.csv or args.database.with_name(f"{args.database.stem}-ccc.csv"))
    return 0


def command_finalize_output_errors(args: argparse.Namespace) -> int:
    converted, remaining = finalize_output_errors(args.database)
    print(f"{converted} allbwn model annilys wedi'u sgorio'n sero")
    print(f"{remaining} gwall seilwaith heb ei ddatrys")
    return 1 if remaining else 0


def command_leaderboard(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or project_root() / "results" / args.database.stem
    models, _, _ = _catalog(args.config_dir)
    # Mae build_leaderboard yn hidlo'r catalog i'r modelau sydd yn y DB. Mae
    # angen cynnwys modelau lleol analluog yma pan gyfunir rhediadau cyhoeddi.
    final_models = [asdict(model) for model in models]
    rows = build_leaderboard(args.database, output_dir, final_models)
    print(f"{len(rows)} model yn y sgorfwrdd")
    print(output_dir)
    return 0


def command_voice_validate(args: argparse.Namespace) -> int:
    total, errors = validate_voice_catalog(args.models, args.suites, args.root)
    print(f"{total} eitem yng nghatalog CymraegBench Voice")
    for error in errors:
        print(f"GWALL: {error}", file=sys.stderr)
    print("Dilys." if not errors else "Methodd y dilysu.")
    return 1 if errors else 0


def command_voice_run(args: argparse.Namespace) -> int:
    completed, errors = run_voice_benchmark(
        models_path=args.models,
        suites_path=args.suites,
        root=args.root,
        output=args.output,
        model_ids=set(_ids(args.model_ids) or []) or None,
        suite_ids=set(_ids(args.suite_ids) or []) or None,
        max_cases=args.max_cases,
        timeout=args.timeout,
    )
    print(f"{completed} achos llais wedi cwblhau; {errors} gwall")
    print(args.output)
    return 1 if errors else 0


def command_voice_report(args: argparse.Namespace) -> int:
    rows = build_voice_report(args.results, args.markdown, args.csv)
    print(f"{len(rows)} rhes grynhoi llais")
    print(args.markdown)
    print(args.csv)
    return 0


def command_voice_listening_pack(args: argparse.Namespace) -> int:
    ratings, key = build_listening_pack(args.results, args.output_dir, args.seed)
    print(ratings)
    print(f"Cadwch yr allwedd ar wahân: {key}")
    return 0


def command_voice_listening_report(args: argparse.Namespace) -> int:
    rows = build_listening_report(args.ratings, args.key, args.markdown, args.csv)
    print(f"{len(rows)} model yn yr adroddiad gwrando")
    print(args.markdown)
    print(args.csv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meincnodau Cymraeg AIsteddfod")
    parser.add_argument(
        "--config-dir", type=Path, default=project_root() / "config", help="Ffolder y catalogau TOML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Dilysu'r catalog a'r cyfrinachau")
    validate.add_argument("--require-keys", action="store_true")
    validate.add_argument("--check-remote", action="store_true")
    validate.set_defaults(func=command_validate)

    plan = subparsers.add_parser("plan", help="Cyfrif y matrics heb alw API")
    plan.add_argument("--models", help="IDs wedi'u gwahanu â choma")
    plan.add_argument("--suites", help="IDs wedi'u gwahanu â choma")
    plan.add_argument("--max-cases", type=int)
    plan.set_defaults(func=command_plan)

    run = subparsers.add_parser("run", help="Rhedeg a chadw checkpoint SQLite")
    run.add_argument("--models", help="IDs wedi'u gwahanu â choma")
    run.add_argument("--suites", help="IDs wedi'u gwahanu â choma")
    run.add_argument("--max-cases", type=int)
    run.add_argument("--max-usd", type=float)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--run-id")
    run.set_defaults(func=command_run)

    report = subparsers.add_parser("report", help="Creu crynodebau Markdown a CSV")
    report.add_argument("database", type=Path)
    report.add_argument("--markdown", type=Path)
    report.add_argument("--csv", type=Path)
    report.set_defaults(func=command_report)

    ccc_report = subparsers.add_parser(
        "ccc-report", help="Creu adroddiad dimensiynau archwiliad iaith CCC"
    )
    ccc_report.add_argument("database", type=Path)
    ccc_report.add_argument("--markdown", type=Path)
    ccc_report.add_argument("--csv", type=Path)
    ccc_report.set_defaults(func=command_ccc_report)

    finalize = subparsers.add_parser(
        "finalize-output-errors",
        help="Sgorio allbynnau gwag neu doredig parhaol yn sero ar ôl ailbrofi",
    )
    finalize.add_argument("database", type=Path)
    finalize.set_defaults(func=command_finalize_output_errors)

    leaderboard = subparsers.add_parser(
        "leaderboard", help="Creu sgorfwrdd cyhoeddadwy o rediad cyflawn"
    )
    leaderboard.add_argument("database", type=Path)
    leaderboard.add_argument("--output-dir", type=Path)
    leaderboard.set_defaults(func=command_leaderboard)

    voice = subparsers.add_parser("voice", help="Meincnodi ASR a TTS Cymraeg")
    voice_subparsers = voice.add_subparsers(dest="voice_command", required=True)

    voice_validate = voice_subparsers.add_parser("validate", help="Dilysu catalog llais")
    voice_validate.add_argument("--root", type=Path, default=project_root())
    voice_validate.add_argument(
        "--models", type=Path, default=project_root() / "config" / "voice-models.toml"
    )
    voice_validate.add_argument(
        "--suites", type=Path, default=project_root() / "config" / "voice-suites.toml"
    )
    voice_validate.set_defaults(func=command_voice_validate)

    voice_run = voice_subparsers.add_parser("run", help="Rhedeg ASR/TTS drwy addaswyr gorchymyn")
    voice_run.add_argument("--root", type=Path, default=project_root())
    voice_run.add_argument(
        "--models", type=Path, default=project_root() / "config" / "voice-models.toml"
    )
    voice_run.add_argument(
        "--suites", type=Path, default=project_root() / "config" / "voice-suites.toml"
    )
    voice_run.add_argument("--model-ids", help="IDs model wedi'u gwahanu â choma")
    voice_run.add_argument("--suite-ids", help="IDs set wedi'u gwahanu â choma")
    voice_run.add_argument("--max-cases", type=int)
    voice_run.add_argument("--timeout", type=float, default=600.0)
    voice_run.add_argument("--output", type=Path, required=True)
    voice_run.set_defaults(func=command_voice_run)

    voice_report = voice_subparsers.add_parser("report", help="Creu adroddiad WER/CER/RTF")
    voice_report.add_argument("results", type=Path)
    voice_report.add_argument("--markdown", type=Path, required=True)
    voice_report.add_argument("--csv", type=Path, required=True)
    voice_report.set_defaults(func=command_voice_report)

    listening_pack = voice_subparsers.add_parser(
        "listening-pack", help="Creu pecyn WAV dall a thaflen sgorio TTS"
    )
    listening_pack.add_argument("results", type=Path)
    listening_pack.add_argument("--output-dir", type=Path, required=True)
    listening_pack.add_argument("--seed", type=int, default=1)
    listening_pack.set_defaults(func=command_voice_listening_pack)

    listening_report = voice_subparsers.add_parser(
        "listening-report", help="Agor yr allwedd a chyfuno sgoriau gwrandawyr"
    )
    listening_report.add_argument("--ratings", type=Path, nargs="+", required=True)
    listening_report.add_argument("--key", type=Path, required=True)
    listening_report.add_argument("--markdown", type=Path, required=True)
    listening_report.add_argument("--csv", type=Path, required=True)
    listening_report.set_defaults(func=command_voice_listening_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    raise SystemExit(code)
