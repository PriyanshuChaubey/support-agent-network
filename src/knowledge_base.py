"""Loads and chunks the markdown knowledge base + resolved-cases JSON into
a flat list of passages that the retrieval node can embed and search over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Passage:
    document: str       # source file / case id
    passage_id: str      # short identifier, e.g. "workspace-settings.md#2"
    text: str
    source_type: str     # "kb" or "resolved_case"


def _chunk_markdown(path: Path) -> List[str]:
    """Split a markdown doc into passages on `##` section boundaries.
    Falls back to paragraph splitting if there are no headings."""
    raw = path.read_text(encoding="utf-8")
    sections = []
    current = []
    for line in raw.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    sections = [s for s in sections if s.strip()]
    if len(sections) <= 1:
        sections = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return sections


def load_passages(kb_dir: str, resolved_cases_path: str) -> List[Passage]:
    passages: List[Passage] = []

    kb_path = Path(kb_dir)
    for md_file in sorted(kb_path.glob("*.md")):
        chunks = _chunk_markdown(md_file)
        for i, chunk in enumerate(chunks):
            passages.append(
                Passage(
                    document=md_file.name,
                    passage_id=f"{md_file.name}#{i}",
                    text=chunk,
                    source_type="kb",
                )
            )

    cases_path = Path(resolved_cases_path)
    if cases_path.exists():
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        for case in cases:
            text = (
                f"Q: {case['question']}\nResolution: {case['resolution']}"
            )
            passages.append(
                Passage(
                    document=case["case_id"],
                    passage_id=case["case_id"],
                    text=text,
                    source_type="resolved_case",
                )
            )

    return passages
