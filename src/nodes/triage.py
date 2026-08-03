"""Triage nodes.

Split into two passes so the graph can short-circuit cheaply before paying
for retrieval:

1. `triage_intent` — deterministic keyword/pattern checks that catch the
   two cases that should NEVER go anywhere near retrieval or generation:
   requests that are clearly out of scope (billing/refunds/legal/etc.)
   and requests that explicitly ask for a human. This is intentionally
   plain Python, not a model call — it's cheap, auditable, and doesn't
   need to be "smart" to catch "write a refund for my subscription."

2. `triage_finalize` — runs AFTER retrieval, and turns retrieval
   confidence into a final classification of `answerable` vs
   `requires_clarification` for anything the first pass didn't already
   resolve.
"""
from __future__ import annotations

import logging

from ..state import AgentState

logger = logging.getLogger("support_agent.triage")

OUT_OF_SCOPE_PATTERNS = [
    "refund", "chargeback", "reimburse", "money back",
    "lawsuit", "legal action", "sue you", "gdpr request", "subpoena",
    "discount code", "promo code", "free trial extension",
    "delete all my personal data",
]

ESCALATION_REQUEST_PATTERNS = [
    "talk to a human", "speak to a person", "human agent",
    "escalate this", "connect me with support", "real person",
]


def triage_intent(state: AgentState) -> dict:
    question = state["question"].lower()
    trace = list(state.get("node_trace", [])) + ["triage_intent"]

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern in question:
            logger.info("triage_intent: matched out-of-scope pattern '%s'", pattern)
            return {
                "classification": "out_of_scope",
                "triage_reason": f"Request matches an out-of-scope pattern ('{pattern}') "
                                  "that the support knowledge base intentionally does not cover.",
                "requires_human": True,
                "node_trace": trace,
            }

    for pattern in ESCALATION_REQUEST_PATTERNS:
        if pattern in question:
            logger.info("triage_intent: matched explicit escalation request '%s'", pattern)
            return {
                "classification": "requires_escalation",
                "triage_reason": "User explicitly asked for a human agent.",
                "requires_human": True,
                "node_trace": trace,
            }

    return {"classification": "pending", "node_trace": trace}


ANSWERABLE_SCORE_THRESHOLD_DEFAULT = 0.08
# Note: this default suits MockBackend's crude bag-of-words overlap score
# (small numbers, since it's an exact-token-overlap ratio). The real
# sentence-transformers cosine similarities run noticeably higher for
# genuinely relevant passages — main.py exposes --threshold so it can be
# retuned per backend/dataset without touching this file.


def triage_finalize(state: AgentState, threshold: float = ANSWERABLE_SCORE_THRESHOLD_DEFAULT) -> dict:
    trace = list(state.get("node_trace", [])) + ["triage_finalize"]

    if state.get("classification") != "pending":
        # Already resolved by triage_intent; nothing to do.
        return {"node_trace": trace}

    retrieved = state.get("retrieved", [])
    top_score = retrieved[0]["score"] if retrieved else 0.0

    # Also require a minimum amount of signal: a lone barely-relevant hit
    # shouldn't count as "answerable" just because nothing beat it.
    if len(state["question"].split()) <= 3:
        # Very short/underspecified questions ("it's broken", "help") are
        # ambiguous almost by construction, regardless of retrieval score.
        classification = "requires_clarification"
        reason = "The question is too short/underspecified to identify what the user needs."
    elif top_score >= threshold:
        classification = "answerable"
        reason = f"Top retrieved passage score {top_score:.3f} >= threshold {threshold}."
    else:
        classification = "requires_clarification"
        reason = f"Best retrieval score {top_score:.3f} is below the confidence threshold {threshold}."

    logger.info("triage_finalize: classification=%s (%s)", classification, reason)
    return {
        "classification": classification,
        "triage_reason": reason,
        "requires_human": False,
        "node_trace": trace,
    }
