# 📧 OpenEnv Email Triage Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-00d4ff?style=flat-square)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)](https://fastapi.tiangolo.com)

A real-world **email triage environment** for AI agents. Agents must classify, prioritize, and respond to business emails — simulating one of the most common daily tasks for knowledge workers. Built to the full OpenEnv spec with three progressively harder tasks.

---

## Why Email Triage?

Email overload is a massive real-world problem. Studies show knowledge workers spend 28% of their day on email. The ability to triage — classify urgency, categorize, decide on actions, and draft replies — is a core productivity skill that AI agents can meaningfully assist with. This environment gives agents a realistic, varied inbox with clear success metrics.

---

## Environment Description

The agent receives emails one at a time from a simulated inbox. For each email, it must:

1. **Classify urgency** — 5 levels (critical → ignore)
2. **Assign category** — 10 types (complaint, spam, finance, HR, legal, etc.)
3. **Recommend action** — 6 options (reply, forward, archive, delete, escalate, flag)
4. **Draft a reply** *(Hard task only)* — contextually appropriate response

Rewards are given per-step with partial credit, not just at the end of an episode.

---

## Observation Space

```json
{
  "current_email": {
    "id": "string",
    "subject": "string",
    "sender": "string",
    "sender_domain": "string",
    "body": "string",
    "timestamp": "ISO8601 string",
    "has_attachment": "boolean",
    "thread_length": "integer"
  },
  "emails_processed": "integer",
  "emails_remaining": "integer",
  "score_so_far":     "float [0.0–1.0]",
  "task_id":          "string",
  "done":             "boolean",
  "message":          "string"
}
```

---

## Action Space

```json
{
  "urgency":     "critical | high | medium | low | ignore",
  "category":    "customer_complaint | sales_inquiry | internal_ops | hr | finance | spam | support | legal | pr | other",
  "action":      "reply | forward | archive | delete | escalate | flag_review",
  "draft_reply": "string (required when action=reply)",
  "forward_to":  "string (required when action=forward or escalate)",
  "reasoning":   "string (optional, not scored)"
}
```

---

## Tasks

### Task 1 — Binary Spam Detection (Easy)
- **Emails**: 10
- **Objective**: Identify spam vs legitimate email; assign basic urgency
- **Reward**: Full (1.0) for correct spam detection; proportional for legit emails
- **Success Threshold**: 0.75
- **Baseline Score**: ~0.82

### Task 2 — Priority Inbox Triage (Medium)
- **Emails**: 15
- **Objective**: Full 3-way classification (urgency × category × action)
- **Reward**: Weighted sum with partial credit for close classifications
- **Success Threshold**: 0.65
- **Baseline Score**: ~0.67

### Task 3 — Full Triage + Response Drafting (Hard)
- **Emails**: 20
- **Objective**: Complete triage + draft contextually appropriate replies
- **Reward**: 50% triage quality + 50% reply quality (keyword coverage, tone, length)
- **Success Threshold**: 0.55
- **Baseline Score**: ~0.54

---

## Reward Function

Rewards are **per-step** (not end-of-episode) and provide partial progress signals:

```
reward = urgency_score × 0.30
       + category_score × 0.40
       + action_score × 0.30
       − penalty
```

**Partial credit:**
- Urgency: exact=1.0, off-by-one=0.5, off-by-two=0.2
- Category: exact=1.0, semantically related=0.4, unrelated=0.0
- Action: exact=1.0, acceptable alternative=0.5

**Penalties:**
- Missing critical email: −0.25 to −0.30
- Marking legit email as spam: −0.15 to −0.30

**Reply quality (Hard task):** graded on non-empty content, length ≥100 chars, keyword coverage, and professional tone markers.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start episode: `{"task_id": "task_easy"}` |
| `POST` | `/step` | Submit action, get reward |
| `GET`  | `/state` | Full internal state |
| `GET`  | `/health` | Health check (returns 200) |
| `GET`  | `/tasks` | List tasks |
| `GET`  | `/action_space` | Valid action enum values |
| `GET`  | `/docs` | Swagger UI |

---

## Setup & Usage

### Docker (recommended)

```bash
# Build
docker build -t openenv-email-triage .

# Run
docker run -p 7860:7860 openenv-email-triage

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_easy"}'
```

### Local Python

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"

python inference.py
```

### Run Validation

```bash
python validate.py
```

---

## Baseline Scores

Scores produced by `gpt-4o-mini` (temperature=0.1):

| Task | Score | Threshold | Pass |
|------|-------|-----------|------|
| task_easy   | 0.82 | 0.75 | ✅ |
| task_medium | 0.67 | 0.65 | ✅ |
| task_hard   | 0.54 | 0.55 | ❌ (close) |
| **Overall** | **0.68** | — | — |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | Hugging Face / API key |

---

## Project Structure

```
openenv-email-triage/
├── openenv.yaml       # OpenEnv spec metadata
├── models.py          # Pydantic typed models (Observation, Action, Reward)
├── dataset.py         # Email dataset with ground truth labels
├── graders.py         # Deterministic graders for all 3 tasks
├── environment.py     # EmailTriageEnv with step()/reset()/state()
├── server.py          # FastAPI server exposing REST endpoints
├── inference.py       # Baseline inference script (OpenAI client)
├── validate.py        # Pre-submission validation script
├── app.py             # HF Spaces entry point
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
├── static/
│   └── index.html     # Interactive environment UI
└── README.md          # This file
```

---

## License

MIT © 2024 OpenEnv Hackathon Team
