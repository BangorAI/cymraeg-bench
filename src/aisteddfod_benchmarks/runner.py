from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ModelConfig, SuiteConfig
from .datasets import load_cases
from .providers import ProviderError, generate
from .scoring import score_prediction
from .storage import Storage


def default_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _evaluate_one(
    model: ModelConfig,
    suite: SuiteConfig,
    case: Any,
    base: dict[str, Any],
) -> dict[str, Any]:
    response = generate(
        model,
        case.system,
        case.user,
        max_tokens=suite.max_tokens,
    )
    refused = response.stop_reason == "refusal"
    if refused:
        score = 0.0
        scoring = {"refusal": True}
    else:
        score, scoring = score_prediction(
            suite.scorer,
            response.text,
            case.expected,
            case_payload={
                "case_id": case.id,
                "metadata": case.metadata,
                "expected": case.expected,
            },
            command_env=suite.scorer_command_env,
        )
    return base | {
        "status": "refusal" if refused else "completed",
        "prediction": response.text,
        "score": score,
        "scoring_json": json.dumps(scoring, ensure_ascii=False),
        "provider_model": response.model,
        "provider_response_json": json.dumps(response.raw, ensure_ascii=False),
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "stop_reason": response.stop_reason,
        "error": None,
    }


def _error_result(base: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return base | {
        "status": "error",
        "prediction": None,
        "score": None,
        "scoring_json": "{}",
        "provider_model": None,
        "provider_response_json": None,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
        "latency_ms": None,
        "stop_reason": None,
        "error": str(exc)[:2000],
    }


def run_evaluation(
    *,
    models: list[ModelConfig],
    suites: list[SuiteConfig],
    root: Path,
    techiaith_revision: str,
    run_id: str | None = None,
    max_cases: int | None = None,
    max_usd: float | None = None,
    seed: int = 1,
    workers: int = 1,
) -> tuple[str, Path, int, int]:
    if workers < 1:
        raise ValueError("Rhaid cael o leiaf un gweithiwr")
    actual_id = run_id or default_run_id()
    database_path = root / "runs" / f"{actual_id}.sqlite3"
    store = Storage(database_path)
    created = datetime.now(timezone.utc).isoformat()
    store.start_run(
        actual_id,
        created,
        seed,
        max_cases,
        {
            "models": [asdict(model) | {"api_key_env": model.api_key_env} for model in models],
            "suites": [suite.id for suite in suites],
            "techiaith_revision": techiaith_revision,
            "workers": workers,
        },
    )
    completed = 0
    errors = 0
    processed = 0
    tasks: list[tuple[ModelConfig, SuiteConfig, Any, dict[str, Any]]] = []
    try:
        for suite in suites:
            cases = load_cases(
                suite,
                root=root,
                techiaith_revision=techiaith_revision,
                max_cases=max_cases,
                seed=seed,
            )
            for case in cases:
                for repetition in range(1, suite.repetitions + 1):
                    for model in models:
                        if store.has_result(actual_id, model.id, suite.id, case.id, repetition):
                            processed += 1
                            continue
                        base = {
                            "run_id": actual_id,
                            "model_id": model.id,
                            "suite_id": suite.id,
                            "case_id": case.id,
                            "repetition": repetition,
                            "scorer": suite.scorer,
                            "system_prompt": case.system,
                            "user_prompt": case.user,
                            "expected": case.expected,
                        }
                        tasks.append((model, suite, case, base))

        total = processed + len(tasks)
        if processed:
            print(f"Checkpoint: {processed:,}/{total:,} eisoes wedi'u cwblhau", flush=True)

        def record(values: dict[str, Any]) -> None:
            nonlocal completed, errors, processed
            store.save_result(values)
            if values["status"] == "error":
                errors += 1
            else:
                completed += 1
            processed += 1
            if processed % 25 == 0 or processed == total:
                cost = store.total_cost(actual_id)
                print(
                    f"Cynnydd: {processed:,}/{total:,} ({100 * processed / total:.1f}%); "
                    f"{completed:,} newydd; {errors:,} gwall; ${cost:.2f}",
                    flush=True,
                )

        def cost_limit_reached() -> bool:
            return max_usd is not None and store.total_cost(actual_id) >= max_usd

        if workers == 1:
            for model, suite, case, base in tasks:
                if cost_limit_reached():
                    raise RuntimeError(f"Cyrhaeddwyd y terfyn cost ${max_usd:.2f}")
                try:
                    record(_evaluate_one(model, suite, case, base))
                except (ProviderError, RuntimeError, ValueError) as exc:
                    record(_error_result(base, exc))
        else:
            next_task = 0
            pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            limit_hit = False
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while pending or (next_task < len(tasks) and not limit_hit):
                    while len(pending) < workers and next_task < len(tasks) and not limit_hit:
                        if cost_limit_reached():
                            limit_hit = True
                            break
                        model, suite, case, base = tasks[next_task]
                        next_task += 1
                        future = executor.submit(_evaluate_one, model, suite, case, base)
                        pending[future] = base
                    if not pending:
                        break
                    future = next(as_completed(pending))
                    base = pending.pop(future)
                    try:
                        record(future.result())
                    except (ProviderError, RuntimeError, ValueError) as exc:
                        record(_error_result(base, exc))
                    if cost_limit_reached():
                        limit_hit = True
            if limit_hit and next_task < len(tasks):
                raise RuntimeError(f"Cyrhaeddwyd y terfyn cost ${max_usd:.2f}")
        store.finish_run(actual_id, "completed" if not errors else "completed_with_errors")
    except BaseException:
        store.finish_run(actual_id, "interrupted")
        raise
    finally:
        store.close()
    return actual_id, database_path, completed, errors


def planned_calls(
    models: Iterable[ModelConfig], suites: Iterable[SuiteConfig], max_cases: int | None
) -> tuple[int, list[tuple[str, int]]]:
    model_count = len(list(models))
    rows = []
    total = 0
    for suite in suites:
        cases = min(suite.expected_count, max_cases) if max_cases is not None else suite.expected_count
        calls = cases * suite.repetitions * model_count
        rows.append((suite.id, calls))
        total += calls
    return total, rows
