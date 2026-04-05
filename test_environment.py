"""
Unit tests for the Email Triage OpenEnv environment.
Run: python -m pytest tests/ -v
(requires pydantic, pytest installed)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# ─── These tests run after pip install -r requirements.txt ────────────────────

def test_imports():
    from models import Observation, Action, Reward, StepResponse, EnvState
    from models import UrgencyLevel, EmailCategory, EmailAction
    assert UrgencyLevel.CRITICAL.value == "critical"
    assert EmailCategory.SPAM.value == "spam"
    assert EmailAction.DELETE.value == "delete"


def test_reward_range():
    from models import Reward, RewardBreakdown
    r = Reward(value=0.75, feedback="ok")
    assert 0.0 <= r.value <= 1.0


def test_reward_clamp():
    from models import Reward
    with pytest.raises(Exception):
        Reward(value=1.5, feedback="out of range")


def test_reset_all_tasks():
    from environment import EmailTriageEnv
    env = EmailTriageEnv()
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        obs = env.reset(task_id)
        assert obs.task_id == task_id
        assert obs.current_email is not None
        assert obs.emails_remaining > 0
        assert not obs.done


def test_reset_invalid_task():
    from environment import EmailTriageEnv
    env = EmailTriageEnv()
    with pytest.raises(ValueError):
        env.reset("task_nonexistent")


def test_full_easy_episode():
    from environment import EmailTriageEnv
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    env = EmailTriageEnv()
    obs = env.reset("task_easy")
    steps = 0
    rewards = []

    while not obs.done:
        act = Action(
            urgency=UrgencyLevel.MEDIUM,
            category=EmailCategory.OTHER,
            action=EmailAction.ARCHIVE,
        )
        result = env.step(act)
        rewards.append(result.reward.value)
        assert 0.0 <= result.reward.value <= 1.0
        assert isinstance(result.done, bool)
        obs = result.observation
        steps += 1

    assert steps == 10  # task_easy has 10 emails
    assert len(rewards) == 10
    final = sum(rewards) / len(rewards)
    assert 0.0 <= final <= 1.0


def test_step_after_done_raises():
    from environment import EmailTriageEnv
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    env = EmailTriageEnv()
    env.reset("task_easy")
    # exhaust episode
    for _ in range(10):
        act = Action(urgency=UrgencyLevel.LOW, category=EmailCategory.SPAM, action=EmailAction.DELETE)
        env.step(act)

    with pytest.raises(RuntimeError):
        env.step(act)


def test_perfect_spam_score():
    from graders import grade_task_easy
    from dataset import TASK_EASY_EMAILS
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    # e001 is spam
    spam_email = next(e for e in TASK_EASY_EMAILS if e.id == "e001")
    act = Action(
        urgency=UrgencyLevel.IGNORE,
        category=EmailCategory.SPAM,
        action=EmailAction.DELETE,
    )
    reward = grade_task_easy(act, spam_email)
    assert reward.value == 1.0


def test_missed_spam_penalty():
    from graders import grade_task_easy
    from dataset import TASK_EASY_EMAILS
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    spam_email = next(e for e in TASK_EASY_EMAILS if e.id == "e001")
    act = Action(
        urgency=UrgencyLevel.HIGH,
        category=EmailCategory.OTHER,
        action=EmailAction.REPLY,
    )
    reward = grade_task_easy(act, spam_email)
    assert reward.value == 0.0
    assert reward.breakdown.penalty > 0


def test_state_reflects_progress():
    from environment import EmailTriageEnv
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    env = EmailTriageEnv()
    env.reset("task_easy")
    state0 = env.state()
    assert state0.step_count == 0

    act = Action(urgency=UrgencyLevel.MEDIUM, category=EmailCategory.OTHER, action=EmailAction.ARCHIVE)
    env.step(act)
    state1 = env.state()
    assert state1.step_count == 1
    assert state1.emails_processed == 1


def test_reply_quality_grader():
    from graders import reply_quality_score
    from dataset import TASK_HARD_EMAILS
    from models import Email

    # h001 is a customer complaint — needs apology, resolution, etc.
    email = next(e for e in TASK_HARD_EMAILS if e.id == "h001")

    # Good reply
    good = "Thank you for reaching out. We sincerely apologize for the damaged item you received. We understand your frustration and will issue a full refund within 2 business days. Please reply with your order number to expedite. We look forward to making this right."
    score_good = reply_quality_score(good, email)

    # Bad reply
    bad = "ok"
    score_bad = reply_quality_score(bad, email)

    assert score_good > score_bad
    assert score_good >= 0.4
    assert score_bad == 0.0


def test_task_email_counts():
    from dataset import TASK_EMAILS
    assert len(TASK_EMAILS["task_easy"])   == 10
    assert len(TASK_EMAILS["task_medium"]) == 15
    assert len(TASK_EMAILS["task_hard"])   == 20


def test_all_graders_return_valid_range():
    from graders import grade
    from dataset import TASK_EMAILS
    from models import Action, UrgencyLevel, EmailCategory, EmailAction

    act = Action(urgency=UrgencyLevel.HIGH, category=EmailCategory.FINANCE, action=EmailAction.ESCALATE)
    for task_id, emails in TASK_EMAILS.items():
        for email in emails:
            reward = grade(task_id, act, email)
            assert 0.0 <= reward.value <= 1.0, f"{task_id}/{email.id}: {reward.value}"
