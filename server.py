from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models import Action, UrgencyLevel, EmailCategory, EmailAction
from environment import EmailTriageEnv

# ─── App setup ─────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv Email Triage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnv()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─── Endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata():
    return {
        "name": "OpenEnv Email Triage",
        "description": "AI-powered email triage environment that classifies emails by urgency, category, and action."
    }

@app.post("/mcp")
async def mcp():
    return {
        "jsonrpc": "2.0",
        "result": {
            "message": "MCP endpoint active"
        },
        "id": 1
    }
@app.get("/schema")
async def schema():
    return {
        "action": {
            "urgency": [e.value for e in UrgencyLevel],
            "category": [e.value for e in EmailCategory],
            "action": [e.value for e in EmailAction],
            "draft_reply": "string (optional)",
            "forward_to": "string (optional)",
            "reasoning": "string (optional)"
        },
        "observation": {
            "current_email": "object",
            "done": "boolean",
            "info": "object"
        },
        "state": {
            "emails_processed": "int",
            "current_step": "int",
            "task_id": "string"
        }
    }

# ✅ FIXED RESET (IMPORTANT)
@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id", "task_easy") if body else "task_easy"
    except:
        task_id = "task_easy"

    obs = env.reset(task_id=task_id)
    return obs.model_dump()


# ─── STEP ENDPOINT ─────────────────────────────────────────

@app.post("/step")
async def step(request: Request):
    try:
        data = await request.json()

        urgency  = UrgencyLevel(data.get("urgency", "medium"))
        category = EmailCategory(data.get("category", "other"))
        action   = EmailAction(data.get("action", "archive"))

        act = Action(
            urgency=urgency,
            category=category,
            action=action,
            draft_reply=data.get("draft_reply"),
            forward_to=data.get("forward_to"),
            reasoning=data.get("reasoning"),
        )

        result = env.step(act)
        return result.model_dump()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── OTHER ENDPOINTS ───────────────────────────────────────

@app.get("/state")
async def state():
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Binary Spam Detection",
                "difficulty": "easy",
                "grader": "grade_task_easy",
            },
            {
                "id": "task_medium",
                "name": "Priority Inbox Triage",
                "difficulty": "medium",
                "grader": "grade_task_medium",
            },
            {
                "id": "task_hard",
                "name": "Full Triage with Response Drafting",
                "difficulty": "hard",
                "grader": "grade_task_hard",
            },
        ]
    }


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "OpenEnv Email Triage API running"}
