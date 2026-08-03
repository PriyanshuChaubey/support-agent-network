"""Revision bookkeeping + final output assembly."""
from __future__ import annotations

import logging

from ..state import AgentState

logger = logging.getLogger("support_agent.finalize")

MAX_REVISIONS = 1  # loop-protection: at most one revise-and-reverify pass


def prepare_revision(state: AgentState) -> dict:
    """Runs when verification failed and we still have a retry budget.
    Bumps revision_count and stashes verifier feedback for the next
    generation pass. This being a separate node (rather than a self-loop
    on `generate`) keeps the increment explicit and easy to unit test."""
    trace = list(state.get("node_trace", [])) + ["prepare_revision"]
    count = state.get("revision_count", 0) + 1
    feedback = state["verification"]["notes"]
    logger.info("prepare_revision: attempt %d, feedback=%s", count, feedback)
    return {
        "revision_count": count,
        "generation_feedback": feedback,
        "node_trace": trace,
    }


def finalize(state: AgentState) -> dict:
    trace = list(state.get("node_trace", [])) + ["finalize"]
    classification = state.get("classification", "requires_escalation")
    retrieved = state.get("retrieved", [])

    if classification == "answerable":
        verification = state.get("verification") or {}
        if verification.get("passed"):
            answer = state["draft_answer"]
            sources = [
                {"document": r["document"], "passage": r["passage"][:220]}
                for r in retrieved[:3]
            ]
            confidence = round(min(0.95, 0.55 + retrieved[0]["score"] * 0.4), 2) if retrieved else 0.5
            requires_human = False
            reason = "Answer generated from retrieved knowledge-base evidence and passed verification."
        else:
            # Safe failure: verification failed even after the retry budget.
            classification = "requires_escalation"
            answer = (
                "I found some potentially relevant documentation, but I could not "
                "generate an answer I'm confident is fully supported by it. "
                "I'm routing this to a human support engineer rather than guessing."
            )
            sources = [
                {"document": r["document"], "passage": r["passage"][:220]}
                for r in retrieved[:3]
            ]
            confidence = 0.2
            requires_human = True
            reason = f"Verification failed after {state.get('revision_count', 0)} revision attempt(s): " \
                      f"{(state.get('verification') or {}).get('notes', 'unknown')}"

    elif classification == "requires_clarification":
        answer = (
            "Could you share a bit more detail? For example: which feature this "
            "involves (exports, alerts, API access, billing, etc.), and any error "
            "message or exact steps you've already tried. That will let me search "
            "the right documentation."
        )
        sources = []
        confidence = 0.3
        requires_human = False
        reason = state.get("triage_reason", "Question was too ambiguous to confidently retrieve evidence for.")

    elif classification == "out_of_scope":
        answer = (
            "This request is outside what I can safely handle automatically "
            "(it involves billing/refund/legal action). I'm not able to issue or "
            "promise a refund myself — I've flagged this for a human on the "
            "billing team to review."
        )
        sources = []
        confidence = 0.95
        requires_human = True
        reason = state.get("triage_reason", "Request matched an out-of-scope pattern.")

    else:  # requires_escalation
        answer = (
            "This needs a human support engineer. Please have the following ready: "
            "your workspace ID, the exact report/export name, timestamps, any error "
            "text, and the steps you've already tried."
        )
        sources = []
        confidence = 0.9
        requires_human = True
        reason = state.get("triage_reason", "Escalation requested or required.")

    output = {
        "classification": classification,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "requires_human": requires_human,
        "reason": reason,
    }
    logger.info("finalize: classification=%s requires_human=%s", classification, requires_human)
    return {"final_output": output, "requires_human": requires_human, "node_trace": trace}
