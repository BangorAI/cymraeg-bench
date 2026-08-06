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


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _concept_score(text: str, groups: list[list[str]]) -> float:
    if not groups:
        return 1.0
    value = normalize(text)
    hits = sum(any(normalize(option) in value for option in group) for group in groups)
    return hits / len(groups)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {normalize(str(item)) for item in value}


def _set_f1(actual: set[str], target: set[str]) -> float:
    if not actual and not target:
        return 1.0
    if not actual or not target:
        return 0.0
    overlap = len(actual & target)
    precision, recall = overlap / len(actual), overlap / len(target)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _quotation_matches(actual: Any, expected: str) -> bool:
    """Allow typographic quote wrappers or attribution, but not changed text."""
    value = unicodedata.normalize("NFC", str(actual)).strip()
    target = unicodedata.normalize("NFC", expected).strip()
    if value == target:
        return True
    quoted_spans = re.findall(r'“([^”]+)”|‘([^’]+)’|«([^»]+)»|"([^"]+)"', value)
    return any(
        unicodedata.normalize("NFC", next(part for part in match if part)).strip()
        == target
        for match in quoted_spans
    )


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
        value = float(normalize(_first_label(prediction)) == normalize(expected))
        details: dict[str, Any] = {"extracted": _first_label(prediction)}
        dimension = (case_payload or {}).get("metadata", {}).get("dimension")
        if dimension in {"priod", "term"}:
            details["components"] = {str(dimension): value}
        return value, details
    if scorer == "number":
        actual, target = _number(prediction), _number(expected)
        return float(actual is not None and target is not None and actual == target), {
            "extracted": str(actual) if actual is not None else None
        }
    if scorer == "bleu":
        return None, {}
    if scorer == "ccc_edit":
        target = json.loads(expected)
        actual = _json_object(prediction)
        correction = float(
            normalize(str(actual.get("cywiriad", "")))
            in {normalize(item) for item in target["accepted_corrections"]}
        )
        category = float(normalize(str(actual.get("categori", ""))) == normalize(target["category"]))
        rule = float(normalize(str(actual.get("rheol", ""))) == normalize(target["rule"]))
        explanation = _concept_score(
            str(actual.get("esboniad", "")), target.get("explanation_concepts", [])
        )
        components = {
            "cywiriad": correction,
            "categori": category,
            "rheol": rule,
            "esboniad": explanation,
        }
        return 0.50 * correction + 0.15 * category + 0.25 * rule + 0.10 * explanation, {
            "components": components,
            "parsed": actual,
        }
    if scorer == "ccc_fidelity":
        target = json.loads(expected)
        actual = _json_object(prediction)
        facts = _set_f1(_string_set(actual.get("ffeithiau")), _string_set(target["fact_ids"]))
        quotation = float(_quotation_matches(actual.get("dyfyniad", ""), target["quote"]))
        coverage = _concept_score(str(actual.get("crynodeb", "")), target["summary_concepts"])
        unsupported = _string_set(actual.get("ffeithiau")) - _string_set(target["allowed_fact_ids"])
        faithfulness = float(not unsupported)
        components = {
            "ffeithiau": facts,
            "dyfyniad": quotation,
            "cwmpas": coverage,
            "ffyddlondeb": faithfulness,
        }
        return 0.30 * facts + 0.25 * quotation + 0.40 * coverage + 0.05 * faithfulness, {
            "components": components,
            "unsupported_fact_ids": sorted(unsupported),
            "parsed": actual,
        }
    if scorer == "ccc_grounded":
        target = json.loads(expected)
        actual = _json_object(prediction)
        cited = _string_set(actual.get("cyfeiriadau"))
        wanted = _string_set(target["source_ids"])
        allowed = _string_set(target["allowed_source_ids"])
        citations = float(cited == wanted)
        answer = _concept_score(str(actual.get("ateb", "")), target.get("answer_concepts", []))
        no_invention = float(not (cited - allowed))
        components = {
            "cyfeiriadau": citations,
            "ateb": answer,
            "dim_rhithgyfeirio": no_invention,
        }
        return 0.60 * citations + 0.30 * answer + 0.10 * no_invention, {
            "components": components,
            "unsupported_source_ids": sorted(cited - allowed),
            "parsed": actual,
        }
    if scorer == "ccc_sources":
        target = json.loads(expected)
        actual = _json_object(prediction)
        selected = _string_set(actual.get("ffynonellau"))
        wanted = _string_set(target["source_ids"])
        allowed = _string_set(target["allowed_source_ids"])
        selection = float(selected == wanted)
        no_invention = float(not (selected - allowed))
        components = {
            "dewis_ffynonellau": selection,
            "dim_ffynonellau_annerbyniol": no_invention,
        }
        return 0.80 * selection + 0.20 * no_invention, {
            "components": components,
            "unsupported_source_ids": sorted(selected - allowed),
            "parsed": actual,
        }
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
