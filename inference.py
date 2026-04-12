#!/usr/bin/env python3

import json
import os
import sys
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

sys.path.insert(0, os.path.dirname(__file__))

try:
    from models import EmailAction, EmailCategory, UrgencyLevel
except Exception:
    EmailAction = EmailCategory = UrgencyLevel = None


app = FastAPI()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = (
    os.environ.get("API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or ""
)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if (OpenAI and API_KEY) else None

SYSTEM_PROMPT = """You are an expert email triage assistant.

Return only valid JSON with:
- urgency
- category
- action
- draft_reply (if reply)
- forward_to (if forward/escalate)
- reasoning
"""


class InputData(BaseModel):
    input: Dict[str, Any]


def clamp_enum(value: str, enum_cls, default: str) -> str:
    if enum_cls is None:
        return value or default
    valid = {e.value for e in enum_cls}
    if value in valid:
        return value
    return default


def fallback_decision() -> Dict[str, Any]:
    return {
        "urgency": "medium",
        "category": "other",
        "action": "archive",
        "draft_reply": None,
        "forward_to": None,
        "reasoning": "fallback",
    }


def agent_decide(email_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if client is None:
            return fallback_decision()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(email_data)},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception:
        return fallback_decision()


def normalize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "urgency": clamp_enum(
            decision.get("urgency", "medium"),
            UrgencyLevel,
            "medium",
        ),
        "category": clamp_enum(
            decision.get("category", "other"),
            EmailCategory,
            "other",
        ),
        "action": clamp_enum(
            decision.get("action", "archive"),
            EmailAction,
            "archive",
        ),
        "draft_reply": decision.get("draft_reply"),
        "forward_to": decision.get("forward_to"),
        "reasoning": decision.get("reasoning", ""),
    }


def emit_block(tag: str, **fields: Any) -> None:
    parts = [f"{key}={json.dumps(value)}" for key, value in fields.items()]
    print(f"[{tag}] " + " ".join(parts), flush=True)


@app.post("/reset")
def reset() -> Dict[str, str]:
    return {"status": "reset successful"}


@app.post("/predict")
def predict(data: InputData) -> Dict[str, Any]:
    return normalize_decision(agent_decide(data.input))


def run_cli() -> int:
    try:
        raw_input = sys.stdin.read().strip()
        payload = json.loads(raw_input) if raw_input else {}
    except Exception:
        payload = {}

    email_data = payload.get("input", payload) if isinstance(payload, dict) else {}

    emit_block("START", task="email-triage", model=MODEL_NAME)

    try:
        decision = normalize_decision(agent_decide(email_data))
        emit_block("STEP", step=1, reward=0.0, action=decision["action"])
        emit_block("END", task="email-triage", score=0.0, steps=1)
        print(json.dumps(decision), flush=True)
        return 0
    except Exception as exc:
        safe_message = str(exc).replace("\n", " ")
        emit_block("STEP", step=1, reward=0.0, error=safe_message)
        emit_block("END", task="email-triage", score=0.0, steps=1)
        print(json.dumps(fallback_decision()), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
