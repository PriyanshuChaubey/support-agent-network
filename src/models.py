"""Local Hugging Face model backends.

This module is the ONLY place model weights are loaded or called. Every
graph node talks to the models through the `ModelBackend` interface below,
never directly to `transformers` / `sentence-transformers`. That is the
"clear separation between deterministic code and model reasoning" the
assignment asks for: nodes contain plain Python control flow, this file
contains the only non-deterministic calls.

Two backends are provided:

- `HFBackend`   - the real local models (sentence-transformers embedder +
                  a small local HF text2text model for generation).
                  Requires the models to have been downloaded once while
                  online; runs fully offline afterwards
                  (set HF_HUB_OFFLINE=1 to force this).
- `MockBackend` - deterministic stand-ins with the exact same interface,
                  used by the automated tests and for demoing/validating
                  graph routing without waiting on model downloads or a
                  GPU. This is what makes it possible to unit-test the
                  graph's routing logic without depending on the wording
                  a real model would produce.

Which backend is used is controlled by one flag (`use_mock`) so swapping
between them never touches node or graph code.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger("support_agent.models")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "main"  # pin a commit hash in production use

GENERATION_MODEL_NAME = "google/flan-t5-small"
GENERATION_MODEL_REVISION = "main"  # pin a commit hash in production use


class ModelBackend:
    """Interface every node relies on. Swap implementations, not call sites."""

    def embed(self, texts: List[str]):
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        raise NotImplementedError

    def load_stats(self) -> dict:
        raise NotImplementedError


@dataclass
class _Timing:
    embedder_load_s: float = 0.0
    generator_load_s: float = 0.0
    last_generate_latency_s: float = 0.0
    last_embed_latency_s: float = 0.0


class HFBackend(ModelBackend):
    """Real local Hugging Face models.

    Embedder: sentence-transformers/all-MiniLM-L6-v2 (~90MB, CPU-friendly,
    384-dim embeddings) used for both KB/case retrieval and the
    similarity-based triage step.

    Generator: google/flan-t5-small (~300MB, CPU-friendly, instruction
    tuned) used for answer generation. Swap GENERATION_MODEL_NAME for a
    larger/quantized model if more quality is needed and hardware allows.
    """

    def __init__(self, lazy: bool = True):
        self._embedder = None
        self._generator = None
        self._tokenizer = None
        self.timing = _Timing()
        if not lazy:
            self._load_embedder()
            self._load_generator()

    def _load_embedder(self):
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL_NAME)
        t0 = time.time()
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.timing.embedder_load_s = time.time() - t0
        logger.info("Embedding model loaded in %.2fs", self.timing.embedder_load_s)

    def _load_generator(self):
        if self._generator is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading generation model %s ...", GENERATION_MODEL_NAME)
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        self._generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
        self.timing.generator_load_s = time.time() - t0
        logger.info("Generation model loaded in %.2fs", self.timing.generator_load_s)

    def embed(self, texts: List[str]):
        self._load_embedder()
        t0 = time.time()
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        self.timing.last_embed_latency_s = time.time() - t0
        return vectors

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        self._load_generator()
        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self._generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.3,   # penalizes re-emitting tokens already generated
            no_repeat_ngram_size=3,   # hard-blocks repeating any 3-token sequence
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.timing.last_generate_latency_s = time.time() - t0
        return text

    def load_stats(self) -> dict:
        return {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "generation_model": GENERATION_MODEL_NAME,
            "embedder_load_seconds": round(self.timing.embedder_load_s, 3),
            "generator_load_seconds": round(self.timing.generator_load_s, 3),
            "last_embed_latency_seconds": round(self.timing.last_embed_latency_s, 3),
            "last_generate_latency_seconds": round(self.timing.last_generate_latency_s, 3),
        }


class MockBackend(ModelBackend):
    """Deterministic backend for tests / offline routing demos.

    Uses simple bag-of-words overlap as a stand-in for embedding similarity
    (same [0, 1] scale, same call signature) and a template-based stand-in
    for generation. This lets the graph's *routing logic* be exercised and
    unit-tested independently of real model weights or network access.
    """

    def __init__(self):
        self.timing = _Timing()

    @staticmethod
    def _tokenize(text: str) -> set:
        return {w.strip(".,?!'\"").lower() for w in text.split() if len(w.strip(".,?!'\"")) > 2}

    def embed(self, texts: List[str]):
        # Return the token-set itself; downstream similarity() knows how
        # to compare these. Keeping the same method name/signature as the
        # real backend is what makes the swap transparent to node code.
        return [self._tokenize(t) for t in texts]

    def similarity(self, a, b) -> float:
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        return overlap / max(len(a | b), 1)

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        # Extract the EVIDENCE and QUESTION blocks the prompt template
        # always includes, so tests can assert on structure without
        # depending on real model wording.
        evidence = ""
        if "EVIDENCE:" in prompt:
            evidence = prompt.split("EVIDENCE:", 1)[1].split("QUESTION:")[0].strip()
        question = ""
        if "QUESTION:" in prompt:
            question = prompt.split("QUESTION:", 1)[1].split("ANSWER:")[0].strip()

        is_revision = "CITE_SOURCES_STRICT" in prompt

        # Demo/test hook: a question containing this marker deliberately
        # produces an ungrounded, uncited first draft (fails verification),
        # then a clean, grounded, cited draft once revised. This is what
        # exercises the retry path in tests/test_graph.py without relying
        # on the whims of a real model's non-determinism.
        if "VERIFY_FAIL_DEMO" in question and not is_revision:
            return (
                "You should wait 24 hours and then call our billing hotline at "
                "1-800-000-0000. Settings -> Legacy Panel -> Force Resync."
            )

        first_doc, first_line = "unknown-document", "No evidence available."
        if evidence:
            first_block = evidence.split("\n\n")[0]
            if first_block.startswith("["):
                first_doc = first_block[1:].split("]", 1)[0]
                first_line = first_block.split("]", 1)[1].strip().splitlines()[0]
            else:
                first_line = first_block.splitlines()[0]

        return f"Based on the documentation: {first_line} [Source: {first_doc}]"

    def load_stats(self) -> dict:
        return {
            "embedding_model": "mock-bow-overlap",
            "generation_model": "mock-template",
            "embedder_load_seconds": 0.0,
            "generator_load_seconds": 0.0,
        }


def cosine_sim(u, v) -> float:
    import numpy as np

    u = np.asarray(u)
    v = np.asarray(v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) or 1e-8
    return float(np.dot(u, v) / denom)


def get_backend(use_mock: bool) -> ModelBackend:
    if use_mock:
        logger.info("Using MockBackend (no real model weights loaded).")
        return MockBackend()
    logger.info("Using HFBackend (real local Hugging Face models).")
    return HFBackend(lazy=True)
