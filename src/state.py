"""Shared typed state passed between every node in the graph.

Keeping this in one place is what lets the graph nodes stay small and
purely functional: each node reads a slice of this state and returns a
partial update, and LangGraph merges updates back into the shared state.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict


class RetrievedPassage(TypedDict):
    document: str
    passage: str
    score: float
    source_type: str  # "kb" or "resolved_case"


class VerificationResult(TypedDict):
    supported_by_evidence: bool
    has_source_references: bool
    follows_schema: bool
    no_invented_instructions: bool
    passed: bool
    notes: str


class AgentState(TypedDict, total=False):
    # ---- input ----
    question: str

    # ---- triage ----
    classification: Optional[str]  # answerable | requires_clarification | requires_escalation | out_of_scope
    triage_reason: Optional[str]

    # ---- retrieval ----
    retrieved: List[RetrievedPassage]

    # ---- generation ----
    draft_answer: Optional[str]
    revision_count: int
    generation_feedback: Optional[str]  # fed back in on a revision pass

    # ---- verification ----
    verification: Optional[VerificationResult]

    # ---- output ----
    final_output: Optional[dict]
    requires_human: bool

    # ---- observability ----
    node_trace: List[str]  # ordered list of node names that executed
