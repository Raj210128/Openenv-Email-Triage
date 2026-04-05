"""
EmailTriageEnv — Full OpenEnv-compliant environment.
Implements step() / reset() / state() interface.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, Optional, Tuple
from models import (
    Action, Observation, Reward, StepResponse, EnvState, Email
)
from dataset import TASK_EMAILS
from graders import grade


class EmailTriageEnv:
    """
    Real-world email triage environment.
    
    An AI agent receives emails one at a time and must classify each email
    by urgency, category, and recommended action — and optionally draft a reply.
    
    Usage:
        env = EmailTriageEnv()
        obs = env.reset("task_medium")
        while not obs.done:
            action = agent.decide(obs)
            step_result = env.step(action)
            obs = step_result.observation
    """

    TASK_IDS = ["task_easy", "task_medium", "task_hard"]

    def __init__(self):
        self._task_id: str = "task_easy"
        self._episode_id: str = ""
        self._emails: list[Email] = []
        self._email_index: int = 0
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._scores_history: list[float] = []
        self._done: bool = True

    # ─── OpenEnv Core API ─────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> Observation:
        """
        Start a new episode for the given task.
        Returns the initial observation (first email).
        """
        if task_id not in self.TASK_IDS:
            raise ValueError(
                f"Unknown task '{task_id}'. Choose from: {self.TASK_IDS}"
            )
        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())[:8]
        self._emails = list(TASK_EMAILS[task_id])  # fresh copy
        self._email_index = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._scores_history = []
        self._done = False

        return self._make_observation(
            message=f"Episode started. Inbox has {len(self._emails)} emails to triage."
        )

    def step(self, action: Action) -> StepResponse:
        """
        Submit an Action for the current email.
        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_email = self._current_email()
        if current_email is None:
            raise RuntimeError("No current email. The inbox may be empty.")

        # Grade the action
        reward: Reward = grade(self._task_id, action, current_email)

        # Update state
        self._total_reward += reward.value
        self._scores_history.append(reward.value)
        self._email_index += 1
        self._step_count += 1

        # Check done
        done = self._email_index >= len(self._emails)
        self._done = done

        # Build next observation
        if done:
            episode_avg = self._total_reward / len(self._emails)
            obs = Observation(
                current_email=None,
                emails_processed=self._email_index,
                emails_remaining=0,
                score_so_far=round(episode_avg, 4),
                task_id=self._task_id,
                done=True,
                message=f"Episode complete! Final score: {episode_avg:.3f}"
            )
        else:
            obs = self._make_observation(
                message=f"Step {self._step_count}: reward={reward.value:.3f} — {reward.feedback}"
            )

        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_id":     self._episode_id,
                "step":           self._step_count,
                "email_id":       current_email.id,
                "task_id":        self._task_id,
                "reward_breakdown": reward.breakdown.model_dump(),
            }
        )

    def state(self) -> EnvState:
        """Return full internal state for debugging/logging."""
        return EnvState(
            task_id=self._task_id,
            episode_id=self._episode_id,
            step_count=self._step_count,
            total_reward=self._total_reward,
            emails_processed=self._email_index,
            emails_remaining=max(0, len(self._emails) - self._email_index),
            done=self._done,
            scores_history=list(self._scores_history),
        )

    # ─── Helper methods ───────────────────────────────────────────────────────

    def _current_email(self) -> Optional[Email]:
        if 0 <= self._email_index < len(self._emails):
            return self._emails[self._email_index]
        return None

    def _make_observation(self, message: str = "") -> Observation:
        email = self._current_email()
        return Observation(
            current_email=email.to_agent_view() if email else None,
            emails_processed=self._email_index,
            emails_remaining=max(0, len(self._emails) - self._email_index),
            score_so_far=round(
                self._total_reward / max(1, self._email_index), 4
            ),
            task_id=self._task_id,
            done=self._done,
            message=message,
        )

    def episode_score(self) -> float:
        """Average reward across all processed emails."""
        if not self._scores_history:
            return 0.0
        return round(sum(self._scores_history) / len(self._scores_history), 4)
