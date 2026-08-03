"""Automated tests for graph ROUTING, deliberately independent of the
exact wording a model produces. Every assertion here checks `node_trace`,
`classification`, `requires_human`, or `verification` fields — never
string-matches the generated answer text — per the assignment's
requirement: "At least one automated test must verify graph routing
without depending on the exact wording produced by the model."

Runs fully offline via MockBackend: no model weights or network needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.graph import build_graph
from src.knowledge_base import load_passages
from src.models import MockBackend
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    passages = load_passages(str(ROOT / "data" / "kb"), str(ROOT / "data" / "resolved_cases.json"))
    backend = MockBackend()
    retriever = Retriever(backend, passages)
    retriever.build_index()
    return build_graph(backend, retriever, k=4, answerable_threshold=0.08)


def run(app, question):
    state = {"question": question, "revision_count": 0, "node_trace": []}
    return app.invoke(state, config={"recursion_limit": 25})


# --- Test case 1: directly answerable, single document ---------------------
def test_answerable_single_doc_routes_through_generation(app):
    result = run(app, "Can a read-only user create API credentials?")
    assert result["classification"] == "answerable"
    assert "retrieval" in result["node_trace"]
    assert "generation" in result["node_trace"]
    assert "verification" in result["node_trace"]
    assert result["final_output"]["requires_human"] is False
    assert len(result["final_output"]["sources"]) > 0


# --- Test case 2: needs info from two documents -----------------------------
def test_answerable_pulls_multiple_documents(app):
    result = run(
        app,
        "My scheduled exports stopped after I changed my workspace timezone. What should I check?",
    )
    assert result["classification"] == "answerable"
    docs_used = {r["document"] for r in result["retrieved"][:3]}
    assert len(docs_used) >= 2, f"expected evidence from >=2 docs, got {docs_used}"


# --- Test case 3: ambiguous, needs clarification ----------------------------
def test_ambiguous_question_requires_clarification(app):
    result = run(app, "It's broken, please fix it.")
    assert result["classification"] == "requires_clarification"
    assert "generation" not in result["node_trace"]  # never should reach generation
    assert result["final_output"]["requires_human"] is False


# --- Test case 4: out of scope ----------------------------------------------
def test_refund_request_is_out_of_scope(app):
    result = run(app, "Write a refund for my subscription.")
    assert result["classification"] == "out_of_scope"
    assert result["final_output"]["requires_human"] is True
    # out-of-scope requests must never reach retrieval/generation
    assert "retrieval" not in result["node_trace"]
    assert "generation" not in result["node_trace"]


# --- Test case 5: initial answer fails verification, triggers a retry ------
def test_failed_verification_triggers_revision_then_passes(app):
    result = run(
        app,
        "What should I check if my export fails? VERIFY_FAIL_DEMO",
    )
    # generation must have run twice (initial + one revision)
    assert result["node_trace"].count("generation") == 2
    assert "prepare_revision" in result["node_trace"]
    assert result["revision_count"] == 1
    # after the revision, the safety net (finalize) still produces valid output
    assert result["final_output"]["classification"] in {"answerable", "requires_escalation"}


# --- Explicit escalation request --------------------------------------------
def test_explicit_escalation_request_skips_retrieval(app):
    result = run(app, "Please connect me with support, I want to talk to a human.")
    assert result["classification"] == "requires_escalation"
    assert result["final_output"]["requires_human"] is True
    assert "retrieval" not in result["node_trace"]


# --- Loop protection ---------------------------------------------------------
def test_revision_count_never_exceeds_max(app):
    result = run(app, "What should I check if my export fails? VERIFY_FAIL_DEMO")
    assert result["revision_count"] <= 1  # MAX_REVISIONS in src/nodes/finalize.py
    # graph must terminate (invoke() returning at all proves no infinite loop,
    # this also checks it didn't hit the recursion_limit safety net)
    assert result["node_trace"][-1] == "finalize"


# --- Schema sanity on every route -------------------------------------------
@pytest.mark.parametrize("question", [
    "Can a read-only user create API credentials?",
    "It's broken, please fix it.",
    "Write a refund for my subscription.",
])
def test_output_always_matches_schema(app, question):
    result = run(app, question)
    out = result["final_output"]
    for key in ["classification", "answer", "sources", "confidence", "requires_human", "reason"]:
        assert key in out
    assert isinstance(out["sources"], list)
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["requires_human"], bool)
