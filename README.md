# Local-First Support Agent Network

A graph-based support agent for a fictional SaaS product ("Flowlytics") that
answers questions from a local knowledge base using **only local Hugging
Face models** — no OpenAI/Anthropic/Gemini or any other hosted LLM API is
used anywhere in this repo.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) for
orchestration, `sentence-transformers` for retrieval, and a local
`transformers` model for generation.

---

## 1. What this is

Four responsibilities, each a distinct graph node (see `src/graph.py`):

1. **Triage** — classifies each question as `answerable`,
   `requires_clarification`, `requires_escalation`, or `out_of_scope`.
   Split into two passes: a cheap keyword pass (`triage_intent`) that
   catches things like refund requests before they ever reach retrieval,
   and a retrieval-informed pass (`triage_finalize`) that turns retrieval
   confidence into `answerable` vs `requires_clarification`.
2. **Retrieval** — embeds the question with a local sentence-transformer
   and does cosine-similarity search over the knowledge base + resolved
   cases (`src/retriever.py`). No vector DB — a small in-memory index is
   plenty at this scale.
3. **Response generation** — a local seq2seq model answers strictly from
   the retrieved evidence and is asked to cite its source(s)
   (`src/nodes/generation.py`).
4. **Verification** — deterministic (non-model) checks on the draft
   answer: is it cited, is it lexically grounded in the evidence, does it
   avoid inventing numbers/UI paths not present in the evidence, does it
   have everything needed to fill the output schema
   (`src/nodes/verification.py`). If verification fails, the graph revises
   the answer once with the verifier's feedback; if it fails again, it
   returns a safe "escalate to a human" response instead of guessing.

See `diagrams/graph.png` for the full wiring diagram.

## 2. Models used

| Purpose | Model | Revision | Notes |
|---|---|---|---|
| Embeddings (retrieval + triage confidence) | `sentence-transformers/all-MiniLM-L6-v2` | `main` (pin a commit hash for production) | ~90MB, 384-dim, CPU-friendly |
| Generation | `google/flan-t5-small` | `main` (pin a commit hash for production) | ~300MB, CPU-friendly, instruction-tuned seq2seq |

Both are loaded through Hugging Face libraries (`sentence-transformers`,
`transformers`) in `src/models.py`, which is the **only** file that touches
model weights — every node calls through the `ModelBackend` interface, not
the libraries directly. This is the "clear separation between
deterministic code and model reasoning" the assignment asks for.

Swap `GENERATION_MODEL_NAME` in `src/models.py` for a larger/quantized
model if you have more RAM/GPU and want better answer quality — nothing
else needs to change.

### Load time / latency

Measured on the author's machine (CPU-only, see Hardware section) after
models were cached locally, using the corrected post-run `Model stats`
logging (`main.py` logs this again after the generation model has
actually loaded, since logging it at startup — before generation has run
once — under-reports the generator's load time). Full logs for these runs
are reproducible via `python main.py --file data/sample_questions.json`
and `python main.py --require-offline "..."`.

| Metric | Value |
|---|---|
| Embedder load time (`all-MiniLM-L6-v2`) | ~0.9s (cache warm, offline) to ~22s (online, Hub freshness checks against huggingface.co add overhead even for cached files) |
| Generator load time (`flan-t5-small`) | ~8.0s |
| Knowledge-base index build (40 passages, one-time at startup) | ~20–53s (dominated by the embedder load time above, which happens inside this step) |
| Generation call latency alone (`backend.generate()`, per question) | ~17.6s on CPU (beam search, `num_beams=4`, `max_new_tokens=220`) |
| Single-question latency, `answerable` route (embed + generate + verify, end to end) | ~25–38s on CPU |
| Single-question latency, `requires_clarification` / `out_of_scope` route (no generation call) | <0.1s |

Sample real measurement (one `answerable` question, online run):
```json
{"embedder_load_seconds": 22.335, "generator_load_seconds": 8.012,
 "last_embed_latency_seconds": 0.074, "last_generate_latency_seconds": 17.599}
```

Generation dominates latency, as expected for a seq2seq model doing beam
search on CPU. The clarification/out-of-scope routes are near-instant
because they skip retrieval and/or generation entirely by design — see
the "shortcut routes" in `diagrams/graph.png`. Model load time is also
noticeably higher when online (Hugging Face Hub freshness checks add
several seconds even for already-cached files) versus fully offline via
`--require-offline`, where the embedder load dropped to under 1 second in
testing.

## 3. Running it

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first real run downloads the two models above (needs internet once).
After that, use `--require-offline` to force fully offline operation and
get visual proof no network calls are made:

```bash
python main.py --require-offline "Can a read-only user create API credentials?"
```

Every run — with or without this flag — starts by printing an **OFFLINE
STATUS CHECK** banner showing whether `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE` are set, and a live reachability probe against
`huggingface.co`, e.g.:

```
======================================================================
OFFLINE STATUS CHECK
  HF_HUB_OFFLINE=1        : True
  TRANSFORMERS_OFFLINE=1  : True
  huggingface.co reachable: False
  -> Fully offline: env vars force no network calls, AND no network is reachable.
======================================================================
```

To actually **demonstrate** offline operation (e.g. on video): run once
online to cache the models, then physically disconnect your network
(Wi-Fi off / airplane mode — not just closing a browser), and run again
with `--require-offline`. That flag sets the env vars *before* `transformers`
is imported, so if anything tried to reach the network it would raise an
error immediately instead of silently falling back — a genuine offline
run, not just an assumption.

### Modes

```bash
# Single question, real local models
python main.py "Can a read-only user create API credentials?"

# Batch over the 5 required sample questions, real local models
python main.py --file data/sample_questions.json

# Interactive REPL
python main.py --interactive

# Mock backend — no model downloads, deterministic, useful for a fast
# sanity check of routing logic or CI. NOT what the real answers look like.
python main.py --mock --file data/sample_questions.json
```

Every run prints:
- the **node trace** (which nodes executed, in order — e.g.
  `triage_intent -> retrieval -> triage_finalize -> generation -> verification -> finalize`)
- latency
- the structured JSON output

Full `logging` output (also visible on stderr during any run) additionally
shows model load times, retrieval scores, verification pass/fail reasons,
and revision attempts.

### Tests

```bash
pytest tests/ -v
```

All 10 tests run **offline** against the `MockBackend` and assert on
`node_trace`, `classification`, and `requires_human` — never on the exact
text a model produced — per the assignment's requirement for a routing
test that doesn't depend on model wording. See
`tests/test_graph.py` docstring for details.

A full transcript of running all 5 required sample questions (plus the
verification-retry and explicit-escalation demos) through the mock backend
is saved in `sample_outputs_mock_backend.log` for quick inspection without
running anything.

## 4. Required test cases — where to find each one

| # | Case | Command / test |
|---|---|---|
| 1 | Directly answerable | `python main.py "Can a read-only user create API credentials?"` / `test_answerable_single_doc_routes_through_generation` |
| 2 | Needs two documents | `python main.py "My scheduled exports stopped after I changed my workspace timezone. What should I check?"` / `test_answerable_pulls_multiple_documents` (asserts evidence spans ≥2 distinct docs) |
| 3 | Ambiguous, needs clarification | `python main.py "It's broken, please fix it."` / `test_ambiguous_question_requires_clarification` |
| 4 | Out of scope | `python main.py "Write a refund for my subscription."` / `test_refund_request_is_out_of_scope` |
| 5 | Initial answer fails verification, triggers retry | `python main.py --mock "What should I check if my export fails? VERIFY_FAIL_DEMO"` / `test_failed_verification_triggers_revision_then_passes` — the `VERIFY_FAIL_DEMO` marker is a deliberate hook in `MockBackend.generate()` that returns an uncited, ungrounded first draft so the retry path is exercised deterministically in tests, without depending on a real model's non-determinism. The real `HFBackend` needs no such hook — verification failures happen naturally with a small model; the mock hook exists purely to make the retry path a reliable, repeatable **test**. |

Plus two extra cases exercised beyond the minimum: an explicit
"talk to a human" request (`requires_escalation`, bypasses retrieval
entirely) and a loop-protection check confirming `revision_count` never
exceeds 1 and the graph always terminates at `finalize`.

## 5. Orchestration requirements — where each lives

- **Shared typed state**: `src/state.py` (`AgentState` TypedDict)
- **Conditional routing**: `src/graph.py`, three `add_conditional_edges` calls
  (`route_after_triage_intent`, `route_after_triage_finalize`,
  `route_after_verification`)
- **Retry / revision path**: `verification` → `prepare_revision` →
  back to `generation`, capped at `MAX_REVISIONS = 1`
  (`src/nodes/finalize.py`)
- **Deterministic vs model code**: all model calls are isolated in
  `src/models.py`; every node is otherwise plain Python
- **Node execution logs**: `node_trace` in the returned state, plus
  standard `logging` output from every node
- **Loop protection**: two independent layers — (a) `route_after_verification`
  only allows a revision when `revision_count < MAX_REVISIONS`, so the
  generation↔verification cycle can run at most twice; (b) `main.py` also
  invokes the graph with `recursion_limit=25` as a hard backstop in case
  of a bug in the counter logic

## 6. Output schema

```json
{
  "classification": "answerable",
  "answer": "Suggested response, including a [Source: doc.md] citation",
  "sources": [{"document": "workspace-settings.md", "passage": "..."}],
  "confidence": 0.82,
  "requires_human": false,
  "reason": "Brief explanation"
}
```

Assembled deterministically in `src/nodes/finalize.py` for every route
(answerable / clarification / out-of-scope / escalation / safe-failure), so
the schema is guaranteed even when the model output is discarded (e.g. the
safe-failure path never uses the model's raw text).

## 7. Hardware

| | |
|---|---|
| CPU | AMD Ryzen 3 3250U with Radeon Graphics (2 cores / 4 threads) |
| RAM | 6 GB |
| GPU / accelerator | None — CPU-only run (the "Radeon Graphics" in the CPU name is an integrated GPU, not used for model inference here; `torch` ran in CPU mode) |
| OS | Windows 11 Home Single Language |

This is a genuinely low-resource machine for this kind of workload — a
dual-core CPU and 6GB RAM. It directly informed model selection: both
`all-MiniLM-L6-v2` (~90MB) and `flan-t5-small` (~300MB) were chosen
specifically because they're runnable on hardware like this, where a
larger embedding model or a multi-billion-parameter local LLM would
either fail to load or make each answer take minutes instead of seconds.
The measured load/latency numbers above (Section 2) reflect this hardware
— on a typical 16GB+ multi-core dev machine, expect meaningfully faster
model loads and generation.

## 8. AI assistant disclosure

This project was built with the help of an AI coding assistant (Claude),
which was used to scaffold the LangGraph wiring, write the knowledge base
documents, verification heuristics, and tests. All code was reviewed and
is understood by the author before submission. *(Adjust this paragraph to
accurately describe your own usage before submitting.)*

## 9. Known limitations & what I'd improve with more time

- **Small-model beam-search repetition (found and fixed during testing).**
  Early real-model runs with `flan-t5-small` + `num_beams=4` on longer
  answers showed the classic small-model degeneration pattern — repeating
  the same phrase or sentence 2-3 times in a single answer (observed: a
  775-character answer that repeated its opening sentence three times).
  Fixed by adding `repetition_penalty=1.3` and `no_repeat_ngram_size=3` to
  the generation call (`src/models.py`); the same question afterward
  produced a clean 174-character answer with no repetition. Worth knowing
  about even after the fix, since it's a good example of a small-model
  failure mode distinct from hallucination/ungroundedness — this one is
  about *decoding strategy*, not the model's knowledge, and verification's
  grounding checks wouldn't have caught it (a repeated-but-grounded
  sentence still passes token-overlap grounding). With more time I'd add a
  dedicated verification check for excessive repetition as a fourth
  failure mode alongside the existing four.
- **Citation formatting is deterministic, not model-verified.** Small local
  models are unreliable at reproducing an exact `[Source: ...]` string
  every time even when instructed to, so the citation tag is appended in
  code (`src/nodes/generation.py::_ensure_citation`) whenever the model's
  raw output doesn't already include one that matches a retrieved
  document. This trades a small amount of "did the model actually cite
  correctly" signal for a system that doesn't unnecessarily escalate to a
  human over pure formatting. Verification still independently checks
  grounding and no-invented-instructions against the model's own text, so
  genuine hallucination is still caught.
- **Verification is heuristic, not semantic.** The grounding check uses
  lexical (token-overlap) similarity rather than an entailment/NLI model.
  A small local NLI model (e.g. a distilled MNLI checkpoint) would catch
  paraphrased-but-ungrounded claims that token overlap misses.
- **Triage out-of-scope detection is keyword-based.** It's fast and
  auditable but will miss out-of-scope requests phrased without any of the
  listed patterns. A zero-shot classification pass (still local — e.g.
  `facebook/bart-large-mnli`) would generalize better at the cost of
  latency and RAM.
- **Retrieval has no reranking step.** For a KB this small it doesn't
  matter, but a cross-encoder reranker would help precision at larger
  scale.
- **Answerable-vs-clarification threshold is a single global constant**
  (`ANSWERABLE_SCORE_THRESHOLD_DEFAULT`), tuned per-backend by hand. A
  calibrated/learned threshold (or a small held-out labeled set) would be
  more robust than a hand-picked cutoff.
- **No persistent conversation memory** — every question is handled
  independently; a real support bot would want to remember earlier turns
  within a session, particularly for the `requires_clarification` path
  (right now, a follow-up answer to a clarifying question is treated as a
  brand-new, unrelated question).

## 10. Repository structure

```
.
├── main.py                      # CLI entrypoint
├── requirements.txt
├── data/
│   ├── kb/                      # 9 fictional knowledge-base docs
│   ├── resolved_cases.json      # previously resolved support cases
│   └── sample_questions.json    # the 5 required sample questions
├── src/
│   ├── state.py                 # shared typed graph state
│   ├── models.py                # ONLY place model weights are loaded/called
│   ├── knowledge_base.py        # markdown/JSON loading + chunking
│   ├── retriever.py             # embedding index + cosine search
│   ├── graph.py                 # LangGraph wiring, routing functions
│   └── nodes/
│       ├── triage.py
│       ├── retrieval.py
│       ├── generation.py
│       ├── verification.py
│       └── finalize.py
├── tests/
│   └── test_graph.py            # 10 offline routing tests (MockBackend)
├── diagrams/
│   └── graph.png
├── scripts/
│   └── draw_diagram.py          # regenerates diagrams/graph.png
└── sample_outputs_mock_backend.log  # full transcript, all required cases
```
