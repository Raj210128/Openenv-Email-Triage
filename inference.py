#!/usr/bin/env python3

import os
import sys
import json
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# ─── FastAPI App ─────────────────────────────────────────
app = FastAPI()

# ─── Environment Variables ───────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

# ❗ Prevent crash if token missing
if not HF_TOKEN:
    HF_TOKEN = "dummy-key"

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─── Safe Imports (IMPORTANT FIX) ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

try:
    from models import UrgencyLevel, EmailCategory, EmailAction
except Exception:
    # fallback if import fails (prevents uvicorn crash)
    UrgencyLevel = EmailCategory = EmailAction = None

# ─── Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant.

Return ONLY valid JSON with:
- urgency
- category
- action
- draft_reply (if reply)
- forward_to (if forward/escalate)
- reasoning
"""

# ─── Request Schema ──────────────────────────────────────
class InputData(BaseModel):
    input: Dict[str, Any]

# ─── Helper Function ─────────────────────────────────────
def clamp_enum(value: str, enum_cls):
    if enum_cls is None:
        return value  # fallback if enums not available

    valid = {e.value for e in enum_cls}
    return value if value in valid else list(enum_cls)[0].value

# ─── Agent Logic ─────────────────────────────────────────
def agent_decide(email_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
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
        return {
            "urgency": "medium",
            "category": "other",
            "action": "archive",
            "draft_reply": None,
            "forward_to": None,
            "reasoning": "fallback"
        }

# ─── REQUIRED ENDPOINTS ──────────────────────────────────

# ✅ FIXES YOUR ERROR
@app.post("/reset")
def reset():
    return {"status": "reset successful"}


@app.post("/predict")
def predict(data: InputData):
    email_data = data.input

    decision = agent_decide(email_data)

    urgency  = clamp_enum(decision.get("urgency", "medium"), UrgencyLevel)
    category = clamp_enum(decision.get("category", "other"), EmailCategory)
    action   = clamp_enum(decision.get("action", "archive"), EmailAction)

    return {
        "urgency": urgency,
        "category": category,
        "action": action,
        "draft_reply": decision.get("draft_reply"),
        "forward_to": decision.get("forward_to"),
        "reasoning": decision.get("reasoning", "")
    }