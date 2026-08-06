"""OpenAI-compatible, pinned Transformers server for legacy Mwydryn weights."""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "BangorAI/phi2-mwydryn-1"
MAX_CONTEXT = 2048
PORT = int(os.environ.get("MWYDRYN_PORT", "8003"))

torch.set_grad_enabled(False)
torch.manual_seed(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.to("cuda")
model.eval()

app = FastAPI(title="Mwydryn OpenAI-compatible API")
generation_lock = threading.Lock()


def render_prompt(messages: list[dict[str, Any]]) -> str:
    instruction: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
            )
        content = str(content).strip()
        if role in {"system", "user"} and content:
            instruction.append(content)
    if not instruction:
        raise HTTPException(status_code=400, detail="Mae angen neges system neu user")
    return "### Instruction:\n" + "\n\n".join(instruction) + "\n\n### Response:\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "BangorAI"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("model") not in {None, MODEL_ID}:
        raise HTTPException(status_code=404, detail="Model anhysbys")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Mae messages yn ofynnol")

    prompt = render_prompt(messages)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    requested = max(1, int(payload.get("max_tokens", 128)))
    available = MAX_CONTEXT - input_ids.shape[-1]
    if available < 1:
        raise HTTPException(status_code=400, detail="Mae'r prompt yn hwy na 2,048 tocyn")
    max_new_tokens = min(requested, available)
    seed = int(payload.get("seed", 1))

    started = time.time()
    with generation_lock, torch.inference_mode():
        torch.manual_seed(seed)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[0, input_ids.shape[-1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    hit_limit = completion_ids.numel() >= max_new_tokens
    finish_reason = "length" if hit_limit else "stop"
    prompt_tokens = int(input_ids.shape[-1])
    completion_tokens = int(completion_ids.numel())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(started),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
