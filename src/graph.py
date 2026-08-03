"""Builds the LangGraph StateGraph.

    START
      |
  triage_intent -----------------------------+ (out_of_scope / requires_escalation)
      | (pending)                            |
   retrieval                                 |
      |                                      |
  triage_finalize --------------------------+| (requires_clarification)
      | (answerable)                        ||
   generation                                ||
      |                                      ||
  verification                               ||
      | pass          | fail, retry left      | fail, no retries left
      |               v                       |
      |         prepare_revision              |
      |               |                       |
      |          (back to generation)         |
      v                                       v
   finalize <-------------------------------- +
      |
     END

Loop protection: `prepare_revision` only fires when
`revision_count < MAX_REVISIONS` (checked in the conditional edge), so the
generation<->verification cycle can run at most twice total. As a second,
independent backstop, the graph is invoked with a low `recursion_limit`
in main.py, so even a bug in the counter logic can't spin forever.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import AgentState
from functools import partial

from .nodes.triage import triage_intent, triage_finalize, ANSWERABLE_SCORE_THRESHOLD_DEFAULT
from .nodes.retrieval import make_retrieval_node
from .nodes.generation import make_generation_node
from .nodes.verification import verify
from .nodes.finalize import prepare_revision, finalize, MAX_REVISIONS


def route_after_triage_intent(state: AgentState) -> str:
    if state["classification"] == "pending":
        return "retrieval"
    return "finalize"


def route_after_triage_finalize(state: AgentState) -> str:
    if state["classification"] == "answerable":
        return "generation"
    return "finalize"


def route_after_verification(state: AgentState) -> str:
    verification = state["verification"]
    if verification["passed"]:
        return "finalize"
    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "prepare_revision"
    return "finalize"


def build_graph(backend, retriever, k: int = 4, answerable_threshold: float = ANSWERABLE_SCORE_THRESHOLD_DEFAULT):
    graph = StateGraph(AgentState)

    graph.add_node("triage_intent", triage_intent)
    graph.add_node("retrieval", make_retrieval_node(retriever, k=k))
    graph.add_node("triage_finalize", partial(triage_finalize, threshold=answerable_threshold))
    graph.add_node("generation", make_generation_node(backend))
    graph.add_node("verification", verify)
    graph.add_node("prepare_revision", prepare_revision)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("triage_intent")

    graph.add_conditional_edges(
        "triage_intent", route_after_triage_intent,
        {"retrieval": "retrieval", "finalize": "finalize"},
    )
    graph.add_edge("retrieval", "triage_finalize")
    graph.add_conditional_edges(
        "triage_finalize", route_after_triage_finalize,
        {"generation": "generation", "finalize": "finalize"},
    )
    graph.add_edge("generation", "verification")
    graph.add_conditional_edges(
        "verification", route_after_verification,
        {"finalize": "finalize", "prepare_revision": "prepare_revision"},
    )
    graph.add_edge("prepare_revision", "generation")
    graph.add_edge("finalize", END)

    return graph.compile()
