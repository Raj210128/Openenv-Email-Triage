"""
OpenEnv Email Triage — FastAPI Server
Exposes: POST /reset, POST /step, GET /state, GET /health, GET /
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, UrgencyLevel, EmailCategory, EmailAction
from environment import EmailTriageEnv

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv Email Triage",
    description="Real-world email triage environment implementing the OpenEnv spec.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful per-process)
env = EmailTriageEnv()


# ─── Request/Response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class ActionRequest(BaseModel):
    urgency: str
    category: str
    action: str
    draft_reply: Optional[str] = None
    forward_to: Optional[str] = None
    reasoning: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — must return 200."""
    return {"status": "ok", "env": "email-triage-env", "version": "1.0.0"}


@app.post("/reset")
async def reset(req: ResetRequest):
    """
    Start a new episode for the given task.
    task_id: one of task_easy | task_medium | task_hard
    """
    try:
        obs = env.reset(task_id=req.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: ActionRequest):
    """
    Submit an action for the current email.
    Returns observation, reward, done, info.
    """
    try:
        # Validate enums
        try:
            urgency  = UrgencyLevel(req.urgency)
            category = EmailCategory(req.category)
            action   = EmailAction(req.action)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid enum value: {e}")

        act = Action(
            urgency=urgency,
            category=category,
            action=action,
            draft_reply=req.draft_reply,
            forward_to=req.forward_to,
            reasoning=req.reasoning,
        )
        result = env.step(act)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state():
    """Return current internal environment state."""
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """List available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Binary Spam Detection",
                "difficulty": "easy",
                "num_emails": 10,
                "description": "Classify emails as spam vs legitimate and assign basic urgency.",
            },
            {
                "id": "task_medium",
                "name": "Priority Inbox Triage",
                "difficulty": "medium",
                "num_emails": 15,
                "description": "Full classification: urgency (5 levels), category (10 types), action (6 types).",
            },
            {
                "id": "task_hard",
                "name": "Full Triage with Response Drafting",
                "difficulty": "hard",
                "num_emails": 20,
                "description": "Complete pipeline: triage + draft contextually appropriate replies.",
            },
        ]
    }


@app.get("/action_space")
async def action_space():
    """Describe the action space for agents."""
    return {
        "urgency":    [e.value for e in UrgencyLevel],
        "category":   [e.value for e in EmailCategory],
        "action":     [e.value for e in EmailAction],
        "draft_reply": "string (optional, required when action=reply)",
        "forward_to":  "string (optional, required when action=forward or escalate)",
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive UI for the environment."""
    html = open("/app/static/index.html").read() if os.path.exists("/app/static/index.html") else ""
    if html:
        return HTMLResponse(html)
    return HTMLResponse("""
    <html><head><title>OpenEnv Email Triage</title></head>
    <body style="font-family:monospace;max-width:800px;margin:40px auto;padding:20px">
    <h1>📧 OpenEnv Email Triage</h1>
    <p>Real-world email triage environment implementing the OpenEnv spec.</p>
    <h2>Endpoints</h2>
    <ul>
    <li><b>POST /reset</b> — Start new episode (body: {"task_id": "task_easy"})</li>
    <li><b>POST /step</b> — Submit action for current email</li>
    <li><b>GET /state</b> — Get current environment state</li>
    <li><b>GET /tasks</b> — List available tasks</li>
    <li><b>GET /action_space</b> — Describe valid actions</li>
    <li><b>GET /health</b> — Health check</li>
    <li><b>GET /docs</b> — Interactive API docs (Swagger)</li>
    </ul>
    <p>Run <code>inference.py</code> to see baseline agent scores.</p>
    </body></html>
    """)
