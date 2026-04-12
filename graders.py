"""
Deterministic graders for all three tasks.
Each grader returns a Reward object with a value in [0.0, 1.0]
and a human-readable breakdown.
"""
from __future__ import annotations
from typing import Optional
from models import Action, Reward, RewardBreakdown, Email


# ─── Urgency proximity map (partial credit for close guesses) ─────────────────

URGENCY_ORDER = ["ignore", "low", "medium", "high", "critical"]

def urgency_score(predicted: str, ground_truth: str) -> float:
    """Partial credit: exact=1.0, off-by-one=0.5, off-by-two=0.2, further=0.0"""
    try:
        p = URGENCY_ORDER.index(predicted)
        g = URGENCY_ORDER.index(ground_truth)
        diff = abs(p - g)
        if diff == 0:   return 1.0
        elif diff == 1: return 0.5
        elif diff == 2: return 0.2
        else:           return 0.0
    except ValueError:
        return 0.0


def category_score(predicted: str, ground_truth: str) -> float:
    """Exact match = 1.0; related categories get 0.4"""
    if predicted == ground_truth:
        return 1.0
    # Semantic proximity groups
    related = {
        frozenset({"customer_complaint", "support"}): 0.4,
        frozenset({"sales_inquiry", "pr"}): 0.4,
        frozenset({"hr", "legal"}): 0.3,
        frozenset({"finance", "legal"}): 0.3,
        frozenset({"internal_ops", "support"}): 0.4,
    }
    pair = frozenset({predicted, ground_truth})
    return related.get(pair, 0.0)


def action_score(predicted: str, ground_truth: str) -> float:
    """Exact match = 1.0; acceptable alternatives get 0.5"""
    if predicted == ground_truth:
        return 1.0
    acceptable = {
        ("reply", "forward"):   0.5,
        ("forward", "reply"):   0.5,
        ("escalate", "forward"): 0.5,
        ("forward", "escalate"): 0.5,
        ("archive", "delete"):  0.3,
        ("delete", "archive"):  0.3,
    }
    return acceptable.get((predicted, ground_truth), 0.0)


def reply_quality_score(draft: Optional[str], email: Email) -> float:
    """
    Grade reply quality 0.0–1.0 based on:
    - Non-empty (0.15)
    - Minimum length (0.15)
    - Contains required keywords from ground truth (0.5, proportional)
    - Professional tone markers (0.2)
    """
    if not draft or len(draft.strip()) < 20:
        return 0.0

    score = 0.15  # non-empty bonus
    
    # Length bonus
    if len(draft) >= 100:
        score += 0.15

    # Keyword coverage
    required_kws = email._gt_keywords or []
    if required_kws:
        draft_lower = draft.lower()
        hits = sum(1 for kw in required_kws if kw.lower() in draft_lower)
        keyword_ratio = hits / len(required_kws)
        score += 0.5 * keyword_ratio

    # Tone markers
    tone_markers = [
        "thank", "apolog", "understand", "assist", "help",
        "please", "we will", "we are", "sincerely", "regards",
        "look forward", "happy to", "reach out"
    ]
    draft_lower = draft.lower()
    tone_hits = sum(1 for m in tone_markers if m in draft_lower)
    score += min(0.2, tone_hits * 0.05)

    return min(1.0, score)


# ─── TASK 1 GRADER: Easy — Spam Detection ─────────────────────────────────────

def grade_task_easy(action: Action, email: Email) -> Reward:
    """
    Binary task: correctly identify spam vs non-spam.
    - Spam correctly classified: 1.0
    - Spam classified as low/ignore but not deleted: 0.6
    - Non-spam correctly classified: full urgency + action scoring
    - Spam classified as legitimate: 0.0 + penalty
    """
    gt_cat = email._gt_category
    is_spam = (gt_cat == "spam")
    predicted_is_spam = (action.category == "spam" or action.action == "delete")

    if is_spam:
        if action.category == "spam" and action.action == "delete":
            return Reward(
                value=1.0,
                breakdown=RewardBreakdown(
                    urgency_score=1.0, category_score=1.0,
                    action_score=1.0, reply_quality_score=0.0, penalty=0.0
                ),
                feedback="✓ Correctly identified and deleted spam."
            )
        elif action.category == "spam" or action.action == "delete":
            return Reward(
                value=0.6,
                breakdown=RewardBreakdown(
                    urgency_score=0.5, category_score=0.8,
                    action_score=0.5, reply_quality_score=0.0, penalty=0.0
                ),
                feedback="Partial: spam identified but action suboptimal."
            )
        else:
            # Failed to identify spam — worst case
            return Reward(
                value=0.0,
                breakdown=RewardBreakdown(
                    urgency_score=0.0, category_score=0.0,
                    action_score=0.0, reply_quality_score=0.0, penalty=0.2
                ),
                feedback="✗ Failed to identify spam email. Penalty applied."
            )
    else:
        # Legitimate email — score normally
        u_score = urgency_score(action.urgency, email._gt_urgency or "medium")
        c_score = category_score(action.category, gt_cat or "other")
        a_score = action_score(action.action, email._gt_action or "archive")
        
        # If agent marks legit email as spam — penalize
        penalty = 0.3 if action.category == "spam" else 0.0
        
        total = max(0.0, (u_score * 0.3 + c_score * 0.4 + a_score * 0.3) - penalty)
        return Reward(
            value=round(total, 3),
            breakdown=RewardBreakdown(
                urgency_score=u_score, category_score=c_score,
                action_score=a_score, reply_quality_score=0.0, penalty=penalty
            ),
            feedback=f"Urgency:{u_score:.1f} Category:{c_score:.1f} Action:{a_score:.1f} Penalty:{penalty:.1f}"
        )


# ─── TASK 2 GRADER: Medium — Priority Triage ──────────────────────────────────

def grade_task_medium(action: Action, email: Email) -> Reward:
    """
    Full classification: urgency (30%) + category (40%) + action (30%).
    Partial credit on all dimensions. No reply quality check.
    """
    u_score = urgency_score(action.urgency, email._gt_urgency or "medium")
    c_score = category_score(action.category, email._gt_category or "other")
    a_score = action_score(action.action, email._gt_action or "archive")

    # Penalty: if critical email is triaged as low/ignore
    penalty = 0.0
    if email._gt_urgency == "critical" and action.urgency in ("low", "ignore"):
        penalty = 0.25
    elif email._gt_category == "spam" and action.category != "spam":
        penalty = 0.15

    total = max(0.0, (u_score * 0.3 + c_score * 0.4 + a_score * 0.3) - penalty)
    return Reward(
        value=round(total, 3),
        breakdown=RewardBreakdown(
            urgency_score=u_score, category_score=c_score,
            action_score=a_score, reply_quality_score=0.0, penalty=penalty
        ),
        feedback=(
            f"Urgency:{u_score:.1f}({action.urgency}→{email._gt_urgency}) "
            f"Category:{c_score:.1f}({action.category}→{email._gt_category}) "
            f"Action:{a_score:.1f}({action.action}→{email._gt_action})"
            + (f" PENALTY:{penalty:.2f}" if penalty else "")
        )
    )


# ─── TASK 3 GRADER: Hard — Full Triage + Reply ────────────────────────────────

def grade_task_hard(action: Action, email: Email) -> Reward:
    """
    Full pipeline: triage (50%) + reply quality (50%) when action=reply.
    Non-reply emails still scored on triage quality only.
    """
    u_score = urgency_score(action.urgency, email._gt_urgency or "medium")
    c_score = category_score(action.category, email._gt_category or "other")
    a_score = action_score(action.action, email._gt_action or "archive")
    triage_score = u_score * 0.3 + c_score * 0.4 + a_score * 0.3

    # Penalty for mis-triaging critical emails
    penalty = 0.0
    if email._gt_urgency == "critical" and action.urgency in ("low", "ignore"):
        penalty = 0.3
    elif email._gt_category == "spam" and action.category != "spam":
        penalty = 0.15

    # Reply quality: weighted 50% when reply expected
    needs_reply = email._gt_action == "reply"
    r_score = 0.0
    if needs_reply:
        if action.action == "reply" and action.draft_reply:
            r_score = reply_quality_score(action.draft_reply, email)
        elif action.action != "reply":
            penalty += 0.1  # small penalty for wrong action on reply-needed emails
        
        total = max(0.0, (triage_score * 0.5 + r_score * 0.5) - penalty)
    else:
        total = max(0.0, triage_score - penalty)

    return Reward(
        value=round(total, 3),
        breakdown=RewardBreakdown(
            urgency_score=u_score, category_score=c_score,
            action_score=a_score, reply_quality_score=r_score, penalty=penalty
        ),
        feedback=(
            f"Triage:{triage_score:.2f} Reply:{r_score:.2f} Penalty:{penalty:.2f} "
            f"| U:{action.urgency}→{email._gt_urgency} "
            f"C:{action.category}→{email._gt_category} "
            f"A:{action.action}→{email._gt_action}"
        )
    )


# ─── Grader dispatch ──────────────────────────────────────────────────────────

GRADERS = {
    "task_easy":   grade_task_easy,
    "task_medium": grade_task_medium,
    "task_hard":   grade_task_hard,
}

def grade(task_id: str, action: Action, email: Email) -> Reward:
    grader = GRADERS.get(task_id)
    if not grader:
        raise ValueError(f"Unknown task: {task_id}")
    return grader(action, email)
