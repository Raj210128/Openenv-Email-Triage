#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script.
Checks all OpenEnv compliance requirements before submitting.
Run: python validate.py
"""
import sys
import json
import yaml
import importlib
from pathlib import Path

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []
warnings = []

def check(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

def warn(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {WARN} {label}" + (f": {detail}" if detail else ""))
        warnings.append(label)

print("\n" + "="*60)
print("  OpenEnv Validation — email-triage-env")
print("="*60 + "\n")

# ─── 1. File structure ────────────────────────────────────────────────────────
print("1. Required files")
required_files = [
    "openenv.yaml", "Dockerfile", "requirements.txt", "inference.py",
    "README.md", "models.py", "environment.py", "server.py",
    "graders.py", "dataset.py",
]
for f in required_files:
    check(Path(f).exists(), f)

# ─── 2. openenv.yaml ─────────────────────────────────────────────────────────
print("\n2. openenv.yaml spec")
try:
    with open("openenv.yaml") as fh:
        cfg = yaml.safe_load(fh)
    check("name" in cfg, "has name field")
    check("version" in cfg, "has version field")
    check("tasks" in cfg and len(cfg["tasks"]) >= 3, "has 3+ tasks")
    check("endpoints" in cfg, "has endpoints section")
    check("observation_space" in cfg, "has observation_space")
    check("action_space" in cfg, "has action_space")
except Exception as e:
    check(False, "openenv.yaml parseable", str(e))

# ─── 3. Pydantic models ───────────────────────────────────────────────────────
print("\n3. Typed models (Pydantic)")
try:
    from models import Observation, Action, Reward, StepResponse, EnvState
    check(True, "Observation model imports")
    check(True, "Action model imports")
    check(True, "Reward model imports")
    check(True, "StepResponse model imports")
    check(True, "EnvState model imports")

    # Validate field ranges
    r = Reward(value=0.5, feedback="test")
    check(0.0 <= r.value <= 1.0, "Reward value in [0.0, 1.0]")
except Exception as e:
    check(False, "Models import cleanly", str(e))

# ─── 4. Environment API ───────────────────────────────────────────────────────
print("\n4. Environment API (reset/step/state)")
try:
    from environment import EmailTriageEnv
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    env = EmailTriageEnv()

    # reset()
    obs = env.reset("task_easy")
    check(obs is not None, "reset() returns Observation")
    check(obs.current_email is not None, "reset() observation has current_email")
    check(obs.task_id == "task_easy", "reset() sets task_id")

    # state()
    state = env.state()
    check(state is not None, "state() returns EnvState")
    check(state.task_id == "task_easy", "state() has correct task_id")

    # step()
    act = Action(
        urgency=UrgencyLevel.MEDIUM,
        category=EmailCategory.SPAM,
        action=EmailAction.DELETE,
    )
    result = env.step(act)
    check(result is not None, "step() returns StepResponse")
    check(0.0 <= result.reward.value <= 1.0, "step() reward in [0.0, 1.0]")
    check(isinstance(result.done, bool), "step() returns done boolean")
    check(result.info.get("episode_id") is not None, "step() info has episode_id")

    # All 3 tasks
    for tid in ["task_easy", "task_medium", "task_hard"]:
        env2 = EmailTriageEnv()
        obs2 = env2.reset(tid)
        check(obs2.emails_remaining > 0, f"task {tid} has emails")

except Exception as e:
    check(False, "Environment API works", str(e))

# ─── 5. Graders ───────────────────────────────────────────────────────────────
print("\n5. Task graders (3 tasks, scores in [0,1])")
try:
    from graders import grade
    from dataset import TASK_EMAILS

    for tid in ["task_easy", "task_medium", "task_hard"]:
        emails = TASK_EMAILS[tid]
        rewards = []
        for email in emails[:3]:  # spot-check first 3
            act = Action(
                urgency=UrgencyLevel.MEDIUM,
                category=EmailCategory.OTHER,
                action=EmailAction.ARCHIVE,
            )
            r = grade(tid, act, email)
            rewards.append(r.value)
        all_valid = all(0.0 <= v <= 1.0 for v in rewards)
        check(all_valid, f"{tid} grader scores in [0.0, 1.0]", str(rewards))

except Exception as e:
    check(False, "Graders work", str(e))

# ─── 6. Dockerfile ────────────────────────────────────────────────────────────
print("\n6. Dockerfile")
try:
    dockerfile = Path("Dockerfile").read_text()
    check("FROM python" in dockerfile, "has Python base image")
    check("EXPOSE" in dockerfile, "has EXPOSE directive")
    check("HEALTHCHECK" in dockerfile, "has HEALTHCHECK directive")
    check("uvicorn" in dockerfile or "CMD" in dockerfile, "has CMD to start server")
except Exception as e:
    check(False, "Dockerfile readable", str(e))

# ─── 7. inference.py ──────────────────────────────────────────────────────────
print("\n7. inference.py")
try:
    src = Path("inference.py").read_text()
    check("API_BASE_URL" in src, "reads API_BASE_URL")
    check("MODEL_NAME" in src, "reads MODEL_NAME")
    check("HF_TOKEN" in src, "reads HF_TOKEN")
    check("[START]" in src, "emits [START] log")
    check("[STEP]" in src, "emits [STEP] log")
    check("[END]" in src, "emits [END] log")
    check("OpenAI(" in src, "uses OpenAI client")
except Exception as e:
    check(False, "inference.py readable", str(e))

# ─── 8. README ────────────────────────────────────────────────────────────────
print("\n8. README.md")
try:
    readme = Path("README.md").read_text().lower()
    check("action" in readme, "documents action space")
    check("observation" in readme, "documents observation space")
    check("task" in readme, "describes tasks")
    check("docker" in readme, "includes Docker instructions")
    check("baseline" in readme or "score" in readme, "includes baseline scores")
except Exception as e:
    check(False, "README.md readable", str(e))

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if not errors:
    print(f"  {PASS} ALL CHECKS PASSED — Ready to submit!")
else:
    print(f"  {FAIL} {len(errors)} check(s) FAILED:")
    for e in errors:
        print(f"     • {e}")
if warnings:
    print(f"\n  {WARN} {len(warnings)} warning(s):")
    for w in warnings:
        print(f"     • {w}")
print("="*60 + "\n")
sys.exit(0 if not errors else 1)
