"""Question answering on top of search: find passages, pick the answer sentence, cite it, or refuse.

The answer is EXTRACTIVE by default: it is a sentence copied from a document, so it cannot be invented.
`generate()` can ask a language model to write a fluent answer from the allowed passages instead; it is optional.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np

from .search import Index

_sent = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _sent.split(text) if len(s.strip()) > 3]


@dataclass
class Answer:
    question: str
    abstained: bool
    confidence: float
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    message: str = ""


class QA:
    def __init__(self, index: Index, threshold: float, candidates: int = 8):
        self.index, self.threshold, self.candidates = index, threshold, candidates

    def retrieve(self, question: str, groups):
        """Hybrid search, then rerank. Returns (chunk ids, scores), best first, already permission-filtered."""
        ix = self.index
        mask = ix.allowed_mask(groups)
        qvec = ix.embed_queries([question])[0]
        ids, _ = ix.hybrid_search(question, qvec, mask, self.candidates)
        if ix.reranker is not None:
            return ix.rerank(question, ids, 5)
        d, s = ix.dense_search(qvec, mask, 5)
        return list(d), s

    def ask(self, question: str, groups) -> Answer:
        ids, scores = self.retrieve(question, groups)
        if len(ids) == 0:
            return Answer(question, True, 0.0, message="You do not have access to any documents that could answer this.")
        conf = float(scores[0])
        if conf < self.threshold:
            return Answer(question, True, conf, message="I don't know. The documents I can access do not answer this clearly enough.",
                          citations=[self._cite(ids[0])])
        sentence = self._best_sentence(question, ids[0])
        return Answer(question, False, conf, answer=sentence, citations=[self._cite(i) for i in ids[:3]])

    def _best_sentence(self, question: str, chunk_id: int) -> str:
        sents = split_sentences(self.index.chunks[chunk_id].text)
        if len(sents) <= 1:
            return sents[0] if sents else ""
        ix = self.index
        if ix.reranker is not None:
            sc = np.array(list(ix.reranker.rerank(question, sents)), dtype=float)
        else:
            q = ix.embed_queries([question])[0]
            m = np.array(list(ix.dense.embed(sents)), dtype=np.float32)
            sc = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
        return sents[int(np.argmax(sc))]

    def _cite(self, chunk_id: int) -> dict:
        c = self.index.chunks[chunk_id]
        return dict(chunk_id=c.chunk_id, title=c.title, team=c.team, snippet=c.text[:220] + ("..." if len(c.text) > 220 else ""))


def build_prompt(question: str, chunks) -> tuple[str, str]:
    """Prompt for an optional language model. It only ever receives chunks the user is allowed to see."""
    system = ("Answer the question using ONLY the numbered passages. Cite passage numbers like [1]. "
              "If the passages do not contain the answer, reply exactly: I don't know.")
    body = "\n\n".join(f"[{i}] ({c.title}) {c.text}" for i, c in enumerate(chunks, start=1))
    return system, f"Passages:\n{body}\n\nQuestion: {question}"


def generate(qa: QA, question: str, groups, llm) -> str:
    """llm is any object with complete(system, user) -> str."""
    ids, scores = qa.retrieve(question, groups)
    if len(ids) == 0 or float(scores[0]) < qa.threshold:
        return "I don't know."
    system, user = build_prompt(question, [qa.index.chunks[i] for i in ids[:3]])
    return llm.complete(system, user)
