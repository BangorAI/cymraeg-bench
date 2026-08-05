from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from .config import (
    load_models,
    load_suites,
    project_root,
    select_models,
    select_suites,
)
from .report import build_report
from .runner import planned_calls, run_evaluation


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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    raise SystemExit(code)
