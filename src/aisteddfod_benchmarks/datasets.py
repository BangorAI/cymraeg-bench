from __future__ import annotations

import hashlib
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import SuiteConfig


@dataclass(frozen=True)
class Case:
    id: str
    system: str
    user: str
    expected: str
    metadata: dict[str, Any] = field(default_factory=dict)


CHOICE_SYSTEM = (
    "Ateba'r cwestiwn amlddewis. Rho label yr unig ateb cywir yn unig, "
    "heb esboniad nac atalnodi."
)


def _stable_id(suite_id: str, index: int, row: dict[str, Any]) -> str:
    explicit = row.get("id") or row.get("idx")
    if explicit is not None:
        return f"{suite_id}:{explicit}"
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return f"{suite_id}:{index}:{digest}"


def _format_choices(labels: Iterable[str], texts: Iterable[str]) -> str:
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))


def _adapt(suite: SuiteConfig, row: dict[str, Any], index: int) -> Case:
    case_id = _stable_id(suite.id, index, row)
    adapter = suite.adapter
    if adapter == "techiaith":
        messages = row["input"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        return Case(case_id, system, user, str(row["ideal"]), {"source_index": index})
    if adapter == "arc":
        labels = [str(x) for x in row["choices"]["label"]]
        user = f"{row['question']}\n\n{_format_choices(labels, row['choices']['text'])}"
        return Case(case_id, CHOICE_SYSTEM, user, str(row["answerKey"]), {"labels": labels})
    if adapter == "piqa":
        labels = ["1", "2"]
        user = f"{row['goal']}\n\n1: {row['sol1']}\n2: {row['sol2']}"
        return Case(case_id, CHOICE_SYSTEM, user, str(int(row["label"]) + 1), {"labels": labels})
    if adapter == "truthfulqa_mc1":
        target = row["mc1_targets"]
        labels = [chr(65 + i) for i in range(len(target["choices"]))]
        answer = labels[target["labels"].index(1)]
        user = f"{row['question']}\n\n{_format_choices(labels, target['choices'])}"
        return Case(case_id, CHOICE_SYSTEM, user, answer, {"labels": labels})
    if adapter == "xnli":
        system = (
            "Dosbartha'r berthynas rhwng gosodiad a rhagdybiaeth. "
            "Ateba 0 (mae'n dilyn), 1 (niwtral), neu 2 (gwrthddywediad) yn unig."
        )
        user = f"Gosodiad: {row['premise']}\nRhagdybiaeth: {row['hypothesis']}"
        return Case(case_id, system, user, str(row["label"]), {"labels": ["0", "1", "2"]})
    if adapter == "mgsm":
        system = "Datrysa'r broblem. Ateba â'r rhif terfynol yn unig, heb esboniad."
        return Case(case_id, system, row["question"], str(row["answer_number"]), {})
    if adapter == "copa":
        relation = "achos" if row["question"] == "cause" else "canlyniad"
        system = f"Dewisa'r {relation} mwyaf tebygol. Ateba 1 neu 2 yn unig."
        user = f"{row['premise']}\n\n1: {row['choice1']}\n2: {row['choice2']}"
        return Case(case_id, system, user, str(int(row["label"]) + 1), {"labels": ["1", "2"]})
    if adapter == "wnli":
        system = (
            "Dyma dasg WNLI. Penderfyna a yw'r ail frawddeg yn dilyn o'r gyntaf. "
            "Ateba â label y set ddata, 0 neu 1, yn unig."
        )
        user = f"Brawddeg 1: {row['sentence1']}\nBrawddeg 2: {row['sentence2']}"
        return Case(case_id, system, user, str(row["label"]), {"labels": ["0", "1"]})
    if adapter == "cefr":
        system = (
            "Dosbartha lefel CEFR y testun Cymraeg. Ateba ag un label yn unig: "
            "A1, A2, B1, B2, C1 neu C2."
        )
        return Case(case_id, system, row["text"], row["cefr_level"], {})
    if adapter == "cynghanedd":
        system = row.get(
            "system",
            "Cyfansodda un llinell Gymraeg saith sillaf sy'n cynnwys cynghanedd ddilys. "
            "Rho'r llinell yn unig, heb esboniad.",
        )
        theme = row.get("theme") or row.get("thema")
        return Case(case_id, system, row.get("user", f"Thema: {theme}"), "", {"theme": theme})
    raise ValueError(f"Addasydd set ddata anhysbys: {adapter}")


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "aisteddfod-benchmarks/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def _techiaith_rows(suite: SuiteConfig, root: Path, revision: str) -> list[dict[str, Any]]:
    assert suite.path
    cache = root / "data" / "cache" / "llm-evals-cy" / revision / suite.path
    url = (
        "https://raw.githubusercontent.com/techiaith/llm-evals-cy/"
        f"{revision}/evals-cymraeg/{suite.path}"
    )
    path = _download(url, cache)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _piqa_rows(root: Path) -> list[dict[str, Any]]:
    archive = _download(
        "https://huggingface.co/datasets/britllm/piqa_welsh/resolve/main/physicaliqa-train-dev-welsh.zip",
        root / "data" / "cache" / "piqa_welsh" / "physicaliqa-train-dev-welsh.zip",
    )
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        data_name = next(name for name in names if name.endswith("dev.jsonl"))
        label_name = next(name for name in names if name.endswith("dev-labels.lst"))
        data = zipped.read(data_name).decode("utf-8").splitlines()
        labels = zipped.read(label_name).decode("utf-8").splitlines()
    rows = []
    for raw, label in zip(data, labels):
        row = json.loads(raw)
        row["label"] = int(label)
        rows.append(row)
    return rows


def _hf_rows(suite: SuiteConfig) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Gosodwch y prosiect gyda `python -m pip install -e .`") from exc
    assert suite.dataset and suite.split
    kwargs: dict[str, Any] = {"split": suite.split}
    if suite.trust_remote_code:
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(suite.dataset, suite.dataset_config, **kwargs)
    return [dict(row) for row in dataset]


def load_cases(
    suite: SuiteConfig,
    *,
    root: Path,
    techiaith_revision: str,
    max_cases: int | None = None,
    seed: int = 1,
) -> list[Case]:
    if suite.source == "techiaith_jsonl":
        rows = _techiaith_rows(suite, root, techiaith_revision)
    elif suite.source == "huggingface" and suite.adapter == "piqa":
        rows = _piqa_rows(root)
    elif suite.source == "huggingface":
        rows = _hf_rows(suite)
    elif suite.source == "private_jsonl":
        assert suite.path
        path = root / suite.path
        if not path.exists():
            raise FileNotFoundError(f"Data preifat ar goll: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"Ffynhonnell anhysbys: {suite.source}")
    indexed = list(enumerate(rows))
    if max_cases is not None and max_cases < len(indexed):
        chosen = set(random.Random(seed).sample(range(len(indexed)), max_cases))
        indexed = [pair for pair in indexed if pair[0] in chosen]
    return [_adapt(suite, row, index) for index, row in indexed]
