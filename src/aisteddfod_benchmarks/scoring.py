from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text).strip().casefold()
    value = re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’.,!?;:]+$", "", value)
    return re.sub(r"\s+", " ", value)


def _first_label(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"(?<![\w])([ABC][12]|[A-Za-z]|[0-9]+)(?![\w])", stripped)
    return match.group(1) if match else stripped


def _number(text: str) -> Decimal | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def score_prediction(
    scorer: str,
    prediction: str,
    expected: str,
    *,
    case_payload: dict[str, Any] | None = None,
    command_env: str | None = None,
) -> tuple[float | None, dict[str, Any]]:
    if scorer == "exact":
        return float(normalize(prediction) == normalize(expected)), {}
    if scorer == "choice":
        return float(normalize(_first_label(prediction)) == normalize(expected)), {
            "extracted": _first_label(prediction)
        }
    if scorer == "number":
        actual, target = _number(prediction), _number(expected)
        return float(actual is not None and target is not None and actual == target), {
            "extracted": str(actual) if actual is not None else None
        }
    if scorer == "bleu":
        return None, {}
    if scorer == "external_json":
        if not command_env or not os.getenv(command_env):
            raise RuntimeError(f"Mae {command_env or 'gorchymyn dilysu'} ar goll")
        payload = dict(case_payload or {})
        payload["prediction"] = prediction
        result = subprocess.run(
            shlex.split(os.environ[command_env]),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        metadata = json.loads(result.stdout)
        primary = metadata.get("addas_dilys_newydd", metadata.get("valid", False))
        return float(bool(primary)), metadata
    raise ValueError(f"Sgoriwr anhysbys: {scorer}")
