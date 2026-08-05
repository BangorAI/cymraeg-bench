from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig


@dataclass
class ProviderResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: int
    stop_reason: str | None
    raw: dict[str, Any]


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    attempts: int = 5,
    timeout: int = 180,
) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    retryable = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                return parsed, round((time.monotonic() - started) * 1000)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code not in retryable or attempt == attempts - 1:
                raise ProviderError(f"HTTP {exc.code}: {detail}", exc.code) from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts - 1:
                raise ProviderError(f"Gwall rhwydwaith: {exc}") from exc
            delay = 2**attempt
        time.sleep(min(60.0, delay + random.random() * 0.25))
    raise AssertionError("unreachable")


def _estimated_cost(model: ModelConfig, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if (
        input_tokens is None
        or output_tokens is None
        or model.input_usd_per_million is None
        or model.output_usd_per_million is None
    ):
        return None
    return (
        input_tokens * model.input_usd_per_million
        + output_tokens * model.output_usd_per_million
    ) / 1_000_000


def _require_key(model: ModelConfig) -> str:
    key = model.api_key
    if key:
        return key
    if model.provider == "openai_compatible":
        return "not-required"
    raise ProviderError(f"Mae {model.api_key_env} ar goll ar gyfer {model.label}")


def generate(
    model: ModelConfig,
    system: str,
    user: str,
    *,
    max_tokens: int,
) -> ProviderResponse:
    key = _require_key(model)
    effective_max_tokens = max(max_tokens, model.min_output_tokens or 0)
    if model.provider == "openai":
        response = _openai(model, key, system, user, effective_max_tokens)
    elif model.provider == "anthropic":
        response = _anthropic(model, key, system, user, effective_max_tokens)
    elif model.provider == "openrouter":
        response = _chat(model, key, system, user, effective_max_tokens, openrouter=True)
    elif model.provider == "openai_compatible":
        response = _chat(model, key, system, user, effective_max_tokens, openrouter=False)
    else:
        raise ProviderError(f"Darparwr anhysbys: {model.provider}")
    if response.stop_reason in {"length", "max_tokens", "incomplete"}:
        raise ProviderError(
            f"Ateb wedi'i dorri gan y terfyn tocynnau ({response.stop_reason})"
        )
    if response.stop_reason != "refusal" and not response.text.strip():
        raise ProviderError("Ateb gwag gan y darparwr")
    return response


def _openai(model: ModelConfig, key: str, system: str, user: str, max_tokens: int) -> ProviderResponse:
    payload: dict[str, Any] = {
        "model": model.api_model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # The Responses API rejects values below 16, even for one-token
        # classification suites.
        "max_output_tokens": max(16, max_tokens),
        "store": False,
    }
    if model.reasoning_effort:
        payload["reasoning"] = {"effort": model.reasoning_effort}
    raw, latency = _post_json(
        f"{model.resolved_base_url}/responses",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload,
    )
    text = raw.get("output_text", "")
    refused = False
    if not text:
        parts: list[str] = []
        for item in raw.get("output", []):
            for block in item.get("content", []):
                if block.get("type") in {"output_text", "text"}:
                    parts.append(block.get("text", ""))
                elif block.get("type") == "refusal":
                    refused = True
                    parts.append(block.get("refusal", ""))
        text = "".join(parts)
    usage = raw.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    return ProviderResponse(
        text=text,
        model=raw.get("model", model.api_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimated_cost(model, input_tokens, output_tokens),
        latency_ms=latency,
        stop_reason="refusal" if refused else raw.get("status"),
        raw=raw,
    )


def _anthropic(model: ModelConfig, key: str, system: str, user: str, max_tokens: int) -> ProviderResponse:
    payload: dict[str, Any] = {
        "model": model.api_model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if model.reasoning_effort:
        payload["output_config"] = {"effort": model.reasoning_effort}
    raw, latency = _post_json(
        f"{model.resolved_base_url}/messages",
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        payload,
    )
    text = "".join(block.get("text", "") for block in raw.get("content", []) if block.get("type") == "text")
    usage = raw.get("usage", {})
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    return ProviderResponse(
        text=text,
        model=raw.get("model", model.api_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimated_cost(model, input_tokens, output_tokens),
        latency_ms=latency,
        stop_reason=raw.get("stop_reason"),
        raw=raw,
    )


def _chat(
    model: ModelConfig,
    key: str,
    system: str,
    user: str,
    max_tokens: int,
    *,
    openrouter: bool,
) -> ProviderResponse:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
    }
    if openrouter:
        headers.update(
            {
                "HTTP-Referer": "https://aisteddfod.com",
                "X-Title": "AIsteddfod Welsh Benchmarks",
                "X-OpenRouter-Metadata": "enabled",
            }
        )
        payload["provider"] = {"require_parameters": True}
        if model.reasoning_effort:
            payload["reasoning"] = {
                "effort": model.reasoning_effort,
                "exclude": True,
            }
    raw, latency = _post_json(
        f"{model.resolved_base_url}/chat/completions", headers, payload
    )
    choice = raw.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = str(content or "")
    usage = raw.get("usage", {})
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    cost = usage.get("cost")
    if cost is None:
        cost = _estimated_cost(model, input_tokens, output_tokens)
    return ProviderResponse(
        text=text,
        model=raw.get("model", model.api_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=float(cost) if cost is not None else None,
        latency_ms=latency,
        stop_reason=choice.get("finish_reason"),
        raw=raw,
    )
