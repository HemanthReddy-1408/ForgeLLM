"""Retrieval: BM25 from scratch, optional dense embeddings, reciprocal-rank fusion. RAG is deliberately the
*knowledge* path of this system: facts that change live here (re-index, no retraining); skills live in adapters."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_TOK = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_STOP = {"the", "a", "an", "of", "is", "are", "in", "on", "to", "for", "and", "or", "what", "how", "does", "do", "it", "its", "this", "that", "with", "as", "at", "by", "be", "i", "me", "my", "you", "your", "current", "release"}


def tokenize(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    out = []
    for t in toks:
        out.append(t)
        if "-" in t or "_" in t:  # index both the compound and its parts ("atlas-gateway" -> atlas, gateway)
            out.extend(p for p in re.split(r"[-_]", t) if p)
    return [t for t in out if t not in _STOP]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    meta: dict[str, Any]


class BM25:
    """Okapi BM25:  score(q,d) = Σ idf(t) · f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl))"""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=np.float32)
        self.avgdl = float(self.len.mean()) if len(texts) else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(texts)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.inv: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.inv.setdefault(t, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
        return s


class DenseIndex:
    """Sentence-embedding index (optional; sentence-transformers). Falls back gracefully when unavailable."""

    def __init__(self, texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.emb = self.model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.emb @ q


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion — fuses *ranks*, so scores from different scales never need calibrating."""
    out: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            out[i] = out.get(i, 0.0) + 1.0 / (k + pos + 1)
    return out


class Retriever:
    def __init__(self, docs: list[dict[str, Any]], dense: bool = False) -> None:
        self.docs = [Doc(d["id"], d.get("title", ""), d["text"], {k: v for k, v in d.items() if k not in ("id", "title", "text")}) for d in docs]
        texts = [f"{d.title}. {d.text}" for d in self.docs]
        self.bm25 = BM25(texts)
        self.dense: DenseIndex | None = None
        if dense:
            try:
                self.dense = DenseIndex(texts)
            except Exception:
                self.dense = None

    @classmethod
    def from_jsonl(cls, *paths: str | Path, dense: bool = False) -> Retriever:
        from forgellm.utils import read_jsonl

        docs: list[dict[str, Any]] = []
        for p in paths:
            docs += read_jsonl(p)
        return cls(docs, dense)

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        bm = self.bm25.scores(query)
        order_bm = [int(i) for i in np.argsort(-bm)[: max(k * 3, 10)] if bm[i] > 0]
        if self.dense is not None:
            dn = self.dense.scores(query)
            order_dn = [int(i) for i in np.argsort(-dn)[: max(k * 3, 10)]]
            fused = rrf([order_bm, order_dn])
            ranked = sorted(fused, key=lambda i: -fused[i])[:k]
            score = {i: fused[i] for i in ranked}
        else:
            ranked = order_bm[:k]
            score = {i: float(bm[i]) for i in ranked}
        return [{"id": self.docs[i].id, "title": self.docs[i].title, "text": self.docs[i].text, "score": round(score[i], 4)} for i in ranked]

    def __len__(self) -> int:
        return len(self.docs)


def recall_at_k(retriever: Retriever, queries: list[tuple[str, str]], k: int = 3) -> float:
    """queries: (query, id of the document that must be retrieved)."""
    hit = sum(any(r["id"] == gold for r in retriever.search(q, k)) for q, gold in queries)
    return hit / max(1, len(queries))
