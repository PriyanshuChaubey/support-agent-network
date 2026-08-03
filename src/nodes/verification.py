"""Verification node.

Deliberately implemented as plain deterministic Python, not another model
call: the checks below are things code can verify directly (string
matching, token overlap, numeric grounding) which is both cheaper and more
auditable than asking a second LLM "was this answer good?".

Four checks, all of which must pass for `passed=True`:
  1. has_source_references — the answer cites at least one retrieved doc.
  2. supported_by_evidence  — enough lexical overlap between the answer
     and the retrieved evidence that the answer isn't clearly drawing on
     outside knowledge.
  3. no_invented_instructions — any concrete numbers ("15 minutes",
     "90 days") or "Settings -> ..." style UI paths mentioned in the
     answer also appear somewhere in the evidence.
  4. follows_schema — the pieces needed to build the required output JSON
     are all present (non-empty answer, at least one source when the
     classification is "answerable").
"""
from __future__ import annotations

import logging
import re

from ..state import AgentState

logger = logging.getLogger("support_agent.verification")

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "on",
    "for", "your", "you", "this", "that", "it", "with", "as", "be", "was",
    "were", "will", "can", "if", "not", "based", "documentation", "answer",
}


def _content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _numeric_and_path_claims(text: str):
    numbers = re.findall(r"\b\d+\s?(?:minutes?|days?|hours?|%|seconds?)\b", text.lower())
    paths = re.findall(r"settings\s*[→>-]{1,2}\s*[\w \-→>]+", text.lower())
    return numbers + paths


def verify(state: AgentState) -> dict:
    trace = list(state.get("node_trace", [])) + ["verification"]
    answer = state.get("draft_answer", "") or ""
    retrieved = state.get("retrieved", [])
    evidence_text = "\n".join(r["passage"] for r in retrieved)
    doc_names = {r["document"] for r in retrieved}

    # 1. Source references
    cited = set(re.findall(r"\[Source:\s*([^\]]+)\]", answer))
    has_source_references = any(c.strip() in doc_names for c in cited) if retrieved else False

    # 2. Lexical grounding
    answer_words = _content_words(answer)
    evidence_words = _content_words(evidence_text)
    if answer_words:
        overlap_ratio = len(answer_words & evidence_words) / len(answer_words)
    else:
        overlap_ratio = 0.0
    supported_by_evidence = overlap_ratio >= 0.35

    # 3. No invented concrete claims
    answer_claims = _numeric_and_path_claims(answer)
    invented = [c for c in answer_claims if c not in evidence_text.lower()]
    no_invented_instructions = len(invented) == 0

    # 4. Schema-readiness
    follows_schema = bool(answer.strip()) and (len(retrieved) > 0)

    passed = all(
        [has_source_references, supported_by_evidence, no_invented_instructions, follows_schema]
    )

    notes_parts = []
    if not has_source_references:
        notes_parts.append("missing a [Source: ...] citation matching a retrieved document")
    if not supported_by_evidence:
        notes_parts.append(f"low lexical overlap with evidence ({overlap_ratio:.2f} < 0.35)")
    if not no_invented_instructions:
        notes_parts.append(f"claims not found in evidence: {invented}")
    if not follows_schema:
        notes_parts.append("answer or retrieved evidence missing/empty")
    notes = "; ".join(notes_parts) if notes_parts else "all checks passed"

    logger.info("verification: passed=%s (%s)", passed, notes)

    result = {
        "supported_by_evidence": supported_by_evidence,
        "has_source_references": has_source_references,
        "follows_schema": follows_schema,
        "no_invented_instructions": no_invented_instructions,
        "passed": passed,
        "notes": notes,
    }
    return {"verification": result, "node_trace": trace}
