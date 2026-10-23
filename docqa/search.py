"""Search: keyword (BM25), meaning (embeddings), both combined, and an optional reranker.

Why combine? Keyword search is great when the question uses the same words as the document (names, numbers, codes)
but misses paraphrases. Embedding search finds paraphrases but can miss exact terms. Reciprocal Rank Fusion (RRF) merges
the two ranked lists without having to compare their incompatible scores.

Permissions are applied BEFORE ranking: chunks a user may not see never enter the candidate list, so they cannot leak
through scores, rankings or citations.
"""
from __future__ import annotations
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_tok = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from has have in is it its of on or that the to was were what when where which who why how with did do does".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _tok.findall(text.lower()) if t not in STOP]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Each item scores 1/(k+rank) in every list it appears in; sum and sort."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])


class Index:
    def __init__(self, chunks, use_dense: bool = True, use_rerank: bool = True):
        self.chunks = chunks
        self.teams = np.array([c.team for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.dense = self.reranker = None
        self.emb = None
        if use_dense:
            from fastembed import TextEmbedding
            self.dense = TextEmbedding(EMBED_MODEL)
            self.emb = self._chunk_embeddings()
        if use_rerank:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self.reranker = TextCrossEncoder(RERANK_MODEL)
            except Exception as e:                      # the reranker is optional; the system works without it
                print("reranker unavailable:", e)

    # ---------- embeddings ----------
    def _chunk_embeddings(self) -> np.ndarray:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"chunks_{len(self.chunks)}.npy")
        if os.path.exists(path):
            return np.load(path)
        vecs = np.array(list(self.dense.embed([c.text for c in self.chunks], batch_size=64)), dtype=np.float32)
        np.save(path, vecs)
        return vecs

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.array(list(self.dense.query_embed(queries)), dtype=np.float32)

    # ---------- permission filter ----------
    def allowed_mask(self, groups) -> np.ndarray:
        if groups is None:                    # None means "admin: everything"
            return np.ones(len(self.chunks), dtype=bool)
        return np.isin(self.teams, list(groups))

    # ---------- individual searches: each returns (chunk ids, scores), best first ----------
    def bm25_search(self, query: str, mask: np.ndarray, k: int = 20):
        s = np.array(self.bm25.get_scores(tokenize(query)))
        s = np.where(mask, s, -np.inf)
        order = np.argsort(-s)[:k]
        order = [i for i in order if np.isfinite(s[i])]
        return order, s[order] if len(order) else np.array([])

    def dense_search(self, qvec: np.ndarray, mask: np.ndarray, k: int = 20):
        s = self.emb @ qvec / (np.linalg.norm(self.emb, axis=1) * np.linalg.norm(qvec) + 1e-9)
        s = np.where(mask, s, -np.inf)
        order = np.argsort(-s)[:k]
        order = [i for i in order if np.isfinite(s[i])]
        return order, s[order] if len(order) else np.array([])

    def hybrid_search(self, query: str, qvec: np.ndarray, mask: np.ndarray, k: int = 20):
        b, _ = self.bm25_search(query, mask, 50)
        d, _ = self.dense_search(qvec, mask, 50)
        fused = rrf([list(b), list(d)])[:k]
        return [i for i, _ in fused], np.array([s for _, s in fused])

    def rerank(self, query: str, ids: list[int], k: int = 10):
        """A cross-encoder reads the question and each candidate together, which is slower but more accurate."""
        if not ids:
            return [], np.array([])
        scores = np.array(list(self.reranker.rerank(query, [self.chunks[i].text for i in ids])), dtype=float)
        order = np.argsort(-scores)[:k]
        return [ids[j] for j in order], scores[order]
