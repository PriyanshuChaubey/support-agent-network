"""Wraps the model backend + knowledge base into a simple search() call.

Embeddings are computed once at startup (`build_index`) and reused for
every query — this is the "no managed vector DB required" approach: a
small local Python list is a perfectly adequate index at this scale.
"""
from __future__ import annotations

from typing import List

from .knowledge_base import Passage
from .models import ModelBackend, MockBackend, cosine_sim


class Retriever:
    def __init__(self, backend: ModelBackend, passages: List[Passage]):
        self.backend = backend
        self.passages = passages
        self._vectors = None

    def build_index(self):
        texts = [p.text for p in self.passages]
        self._vectors = self.backend.embed(texts)

    def _score(self, query_vec, doc_vec) -> float:
        if isinstance(self.backend, MockBackend):
            return self.backend.similarity(query_vec, doc_vec)
        return cosine_sim(query_vec, doc_vec)

    def search(self, query: str, k: int = 4):
        if self._vectors is None:
            self.build_index()
        query_vec = self.backend.embed([query])[0]

        scored = [
            (self._score(query_vec, doc_vec), passage)
            for doc_vec, passage in zip(self._vectors, self.passages)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, passage in scored[:k]:
            results.append(
                {
                    "document": passage.document,
                    "passage": passage.text,
                    "score": round(float(score), 4),
                    "source_type": passage.source_type,
                }
            )
        return results
