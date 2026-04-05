#!/usr/bin/env python3
"""
inference.py — Baseline inference script for OpenEnv Email Triage
Uses OpenAI client (via API_BASE_URL + MODEL_NAME) to run an LLM agent
against all 3 tasks and produces reproducible scores.

Required env vars:
  API_BASE_URL  — LLM API base URL (OpenAI-compatible)
  MODEL_NAME    — Model identifier
  HF_TOKEN      — Hugging Face / API key  (used as openai api_key)

Stdout format: strictly [START], [STEP], [END] as specified.
"""
import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

# ─── Environment / config ─────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable is required.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─── Import environment directly (no HTTP required for inference) ──────────────
sys.path.insert(0, os.path.dirname(__file__))
from environment import EmailTriageEnv
from models import Action, UrgencyLevel, EmailCategory, EmailAction

logging.basicConfig(level=logging.WARNING)


# ─── Agent prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant. For each email you receive,
you must classify it and recommend an action. 

You MUST respond with valid JSON only — no markdown, no explanation, just the JSON object.

Required fields:
- urgency: one of [critical, high, medium, low, ignore]
- category: one of [customer_complaint, sales_inquiry, internal_ops, hr, finance, spam, support, legal, pr, other]
- action: one of [reply, forward, archive, delete, escalate, flag_review]
- draft_reply: string (REQUIRED if action is "reply", otherwise omit or set null)
- forward_to: string (REQUIRED if action is "forward" or "escalate", otherwise omit or set null)
- reasoning: brief explanation of your decision

Guidelines:
- Mark spam/phishing as urgency=ignore, category=spam, action=delete
- Mark production outages, legal issues, press inquiries as urgency=critical
- Always draft a reply when action=reply (aim for 100+ words, professional tone)
- For complaints: acknowledge, apologize, provide resolution path
- For enterprise sales: express enthusiasm, offer to connect with sales team
"""

def agent_decide(email_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Call LLM to decide how to triage an email."""
    user_msg = f"""Task: {task_id}

Email to triage:
Subject: {email_data.get('subject', '')}
From: {email_data.get('sender', '')} ({email_data.get('sender_domain', '')})
Date: {email_data.get('timestamp', '')}
Has Attachment: {email_data.get('has_attachment', False)}
Thread Length: {email_data.get('thread_length', 1)}

Body:
{email_data.get('body', '')}

Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: safe default
        return {
            "urgency": "medium", "category": "other", "action": "archive",
            "draft_reply": None, "forward_to": None, "reasoning": "parse error fallback"
        }
    except Exception as e:
        return {
            "urgency": "medium", "category": "other", "action": "archive",
            "draft_reply": None, "forward_to": None, "reasoning": f"error: {e}"
        }


def clamp_enum(value: str, enum_cls) -> str:
    """Return value if valid, else first member."""
    valid = {e.value for e in enum_cls}
    return value if value in valid else list(enum_cls)[2].value


def run_task(task_id: str) -> Dict[str, Any]:
    """Run full episode for one task. Returns result dict."""
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id)

    step_results = []
    step_num = 0

    while not obs.done:
        step_num += 1
        email_data = obs.current_email or {}

        # Agent decides
        decision = agent_decide(email_data, task_id)

        # Build Action (clamp invalid enums to safe defaults)
        urgency  = clamp_enum(str(decision.get("urgency", "medium")),  UrgencyLevel)
        category = clamp_enum(str(decision.get("category", "other")),  EmailCategory)
        action   = clamp_enum(str(decision.get("action", "archive")),  EmailAction)

        act = Action(
            urgency=UrgencyLevel(urgency),
            category=EmailCategory(category),
            action=EmailAction(action),
            draft_reply=decision.get("draft_reply"),
            forward_to=decision.get("forward_to"),
            reasoning=decision.get("reasoning"),
        )

        # Step environment
        result = env.step(act)
        reward_val = result.reward.value
        step_results.append(reward_val)

        # ── [STEP] log ─────────────────────────────────────────────────────
        print(json.dumps({
            "type":        "[STEP]",
            "task_id":     task_id,
            "step":        step_num,
            "email_id":    email_data.get("id", ""),
            "subject":     email_data.get("subject", "")[:60],
            "agent_action": {
                "urgency":  urgency,
                "category": category,
                "action":   action,
            },
            "reward":      round(reward_val, 4),
            "feedback":    result.reward.feedback[:120],
            "done":        result.done,
        }))

        obs = result.observation

    final_score = round(sum(step_results) / len(step_results), 4) if step_results else 0.0
    return {
        "task_id":     task_id,
        "final_score": final_score,
        "steps":       len(step_results),
        "step_scores": [round(s, 4) for s in step_results],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    tasks = ["task_easy", "task_medium", "task_hard"]
    all_results = {}

    # ── [START] log ────────────────────────────────────────────────────────────
    print(json.dumps({
        "type":       "[START]",
        "env":        "email-triage-env",
        "version":    "1.0.0",
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
        "tasks":      tasks,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))

    for task_id in tasks:
        print(json.dumps({"type": "[TASK_START]", "task_id": task_id}))
        t0 = time.time()
        result = run_task(task_id)
        elapsed = round(time.time() - t0, 2)
        result["elapsed_seconds"] = elapsed
        all_results[task_id] = result
        print(json.dumps({
            "type":          "[TASK_END]",
            "task_id":       task_id,
            "final_score":   result["final_score"],
            "steps":         result["steps"],
            "elapsed":       elapsed,
        }))

    overall = round(
        sum(r["final_score"] for r in all_results.values()) / len(all_results), 4
    )

    # ── [END] log ──────────────────────────────────────────────────────────────
    print(json.dumps({
        "type":          "[END]",
        "overall_score": overall,
        "task_scores": {
            t: all_results[t]["final_score"] for t in tasks
        },
        "total_steps":   sum(r["steps"] for r in all_results.values()),
        "model":         MODEL_NAME,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status":        "success",
    }))

    return overall


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 0.0 else 1)
