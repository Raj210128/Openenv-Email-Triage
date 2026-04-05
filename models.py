"""
OpenEnv Email Triage — Typed Models
Implements the full OpenEnv spec with Pydantic v2 models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class UrgencyLevel(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    IGNORE   = "ignore"


class EmailCategory(str, Enum):
    CUSTOMER_COMPLAINT = "customer_complaint"
    SALES_INQUIRY      = "sales_inquiry"
    INTERNAL_OPS       = "internal_ops"
    HR                 = "hr"
    FINANCE            = "finance"
    SPAM               = "spam"
    SUPPORT            = "support"
    LEGAL              = "legal"
    PR                 = "pr"
    OTHER              = "other"


class EmailAction(str, Enum):
    REPLY        = "reply"
    FORWARD      = "forward"
    ARCHIVE      = "archive"
    DELETE       = "delete"
    ESCALATE     = "escalate"
    FLAG_REVIEW  = "flag_review"


# ─── Email Data ───────────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    timestamp: str
    has_attachment: bool = False
    thread_length: int = 1
    # Ground truth (hidden from agent, used by grader)
    _gt_urgency: Optional[str] = None
    _gt_category: Optional[str] = None
    _gt_action: Optional[str] = None
    _gt_keywords: Optional[List[str]] = None

    model_config = {"populate_by_name": True}

    def to_agent_view(self) -> Dict[str, Any]:
        """Return only what the agent should see."""
        return {
            "id": self.id,
            "subject": self.subject,
            "sender": self.sender,
            "sender_domain": self.sender_domain,
            "body": self.body,
            "timestamp": self.timestamp,
            "has_attachment": self.has_attachment,
            "thread_length": self.thread_length,
        }


# ─── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    current_email: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The email currently being processed (agent view only)"
    )
    emails_processed: int = Field(
        default=0,
        description="How many emails have been triaged so far"
    )
    emails_remaining: int = Field(
        default=0,
        description="How many emails remain in the inbox"
    )
    score_so_far: float = Field(
        default=0.0,
        description="Cumulative reward earned this episode"
    )
    task_id: str = Field(
        default="",
        description="Which task is active"
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended"
    )
    message: str = Field(
        default="",
        description="Optional status/feedback message"
    )


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """What the agent submits for the current email."""
    urgency: UrgencyLevel = Field(
        ...,
        description="Urgency classification: critical/high/medium/low/ignore"
    )
    category: EmailCategory = Field(
        ...,
        description="Email category"
    )
    action: EmailAction = Field(
        ...,
        description="Recommended action"
    )
    draft_reply: Optional[str] = Field(
        default=None,
        description="Draft reply text (required when action=reply)"
    )
    forward_to: Optional[str] = Field(
        default=None,
        description="Who to forward/escalate to"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional agent reasoning (not scored)"
    )


# ─── Reward ───────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    category_score: float = Field(default=0.0, ge=0.0, le=1.0)
    action_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reply_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty: float = Field(default=0.0, ge=0.0, le=1.0)


class Reward(BaseModel):
    """Scalar reward with breakdown for interpretability."""
    value: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall reward for this step (0.0–1.0)"
    )
    breakdown: RewardBreakdown = Field(
        default_factory=RewardBreakdown,
        description="Per-component reward breakdown"
    )
    feedback: str = Field(
        default="",
        description="Human-readable explanation of the reward"
    )


# ─── Step Response ────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── State ────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Full internal state (for debugging/logging)."""
    task_id: str
    episode_id: str
    step_count: int
    total_reward: float
    emails_processed: int
    emails_remaining: int
    done: bool
    scores_history: List[float] = Field(default_factory=list)
