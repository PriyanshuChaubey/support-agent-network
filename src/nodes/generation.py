"""Generation node — asks the local LLM to answer strictly from the
retrieved evidence, and to cite the document(s) it drew from.

On a revision pass (after a failed verification), the prompt includes the
verifier's feedback and an explicit instruction to fix it — this is what
the "revise the answer once" requirement looks like in code.

Small local models (e.g. flan-t5-small) are unreliable at following an
exact string-formatting instruction like "[Source: doc.md]" every time,
even when told to. Rather than let a pure formatting miss burn the one
retry budget and trigger an unnecessary safe-failure, this node
deterministically appends the citation itself when the model's raw answer
doesn't already include one — the correct source is already known from
retrieval, so there's no reason to gamble on the model repeating it
verbatim. Verification still independently checks grounding and
no-invented-instructions against the model's own generated content, so
this doesn't weaken verification — it only removes a brittle formatting
dependency on a small model.
"""
from __future__ import annotations

import logging
import re

from ..state import AgentState

logger = logging.getLogger("support_agent.generation")

PROMPT_TEMPLATE = """You are a support assistant. Answer the question using ONLY the evidence
below. Do not invent steps, settings, or facts that are not in the evidence.
End your answer by citing the source document(s) you used, formatted like
[Source: <document name>].

EVIDENCE:
{evidence}

QUESTION:
{question}

ANSWER:"""

REVISION_SUFFIX = """

REVISION_FEEDBACK: Your previous answer failed a verification check:
{feedback}
Rewrite the answer, fixing this issue, while still only using the evidence above.
CITE_SOURCES_STRICT: yes"""


def _ensure_citation(answer: str, retrieved: list) -> str:
    """If the model's raw answer doesn't already contain a [Source: ...]
    tag matching one of the retrieved documents, append one deterministically."""
    if not retrieved:
        return answer
    doc_names = {r["document"] for r in retrieved}
    cited = set(re.findall(r"\[Source:\s*([^\]]+)\]", answer))
    if any(c.strip() in doc_names for c in cited):
        return answer  # model already cited correctly, leave it alone
    top_doc = retrieved[0]["document"]
    return f"{answer.rstrip()} [Source: {top_doc}]"


def make_generation_node(backend):
    def generation_node(state: AgentState) -> dict:
        trace = list(state.get("node_trace", [])) + ["generation"]
        retrieved = state.get("retrieved", [])
        evidence = "\n\n".join(
            f"[{r['document']}] {r['passage']}" for r in retrieved
        ) or "(no evidence retrieved)"

        prompt = PROMPT_TEMPLATE.format(evidence=evidence, question=state["question"])

        feedback = state.get("generation_feedback")
        if feedback:
            prompt += REVISION_SUFFIX.format(feedback=feedback)
            logger.info("generation: revising with feedback: %s", feedback)

        raw_answer = backend.generate(prompt)
        answer = _ensure_citation(raw_answer, retrieved)
        if answer != raw_answer:
            logger.info("generation: model omitted a citation tag; appended one deterministically")
        logger.info("generation: produced %d-char draft answer", len(answer))

        return {"draft_answer": answer, "node_trace": trace}

    return generation_node
