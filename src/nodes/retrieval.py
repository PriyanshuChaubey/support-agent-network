"""Retrieval node — pulls the top-k passages for the question using the
local embedding model. Pure wrapper around Retriever.search so the node
itself stays a thin, testable function."""
from __future__ import annotations

import logging

from ..state import AgentState

logger = logging.getLogger("support_agent.retrieval")


def make_retrieval_node(retriever, k: int = 4):
    def retrieval_node(state: AgentState) -> dict:
        trace = list(state.get("node_trace", [])) + ["retrieval"]
        results = retriever.search(state["question"], k=k)
        logger.info(
            "retrieval: top result doc=%s score=%.3f",
            results[0]["document"] if results else "-",
            results[0]["score"] if results else 0.0,
        )
        return {"retrieved": results, "node_trace": trace}

    return retrieval_node
