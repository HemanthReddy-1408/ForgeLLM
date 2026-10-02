"""Deduplication: exact (normalised hash) + near-duplicate (MinHash signatures, LSH banding, verified Jaccard)."""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np

from forgellm.data.schemas import Example
from forgellm.utils import stable_hash

_NORM = re.compile(r"[^a-z0-9 ]+")
_P = np.uint64((1 << 31) - 1)


def prompt_key(ex: Example) -> str:
    """The text that identifies *what is being asked*. Boilerplate inputs (tool lists, shared passages) are excluded
    so that they do not make unrelated prompts look similar."""
    if ex.task_type in ("extraction",):
        return ex.input
    if ex.task_type in ("coding",):
        return f"{ex.instruction}\n{ex.input}"
    if ex.task_type == "grounded_qa":  # instance identity = question + exact passage set (passages are templated)
        return f"{ex.instruction} #{stable_hash(ex.input, n=10)}"
    return ex.instruction


def normalize_key(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())


def shingles(text: str, n: int = 3) -> set[int]:
    toks = normalize_key(text).split()
    if len(toks) <= n:
        return {zlib.crc32(" ".join(toks).encode())}
    return {zlib.crc32(" ".join(toks[i : i + n]).encode()) for i in range(len(toks) - n + 1)}


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_P), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_P), size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        h = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
        return ((self.a[:, None] * h[None, :] + self.b[:, None]) % _P).min(axis=1)


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def bucket_of(ex: Example) -> tuple:
    return (ex.task_type, ex.meta.get("stage"), ex.meta.get("subtype"))


def dedup(examples: list[Example], threshold: float = 0.85, num_perm: int = 128, bands: int = 32) -> tuple[list[Example], dict[str, int]]:
    """Keep one representative per duplicate cluster — the highest quality, earliest on ties."""
    stats = {"input": len(examples), "exact": 0, "near": 0}
    # --- exact: identical (prompt, output) after normalisation -------------------------------------
    seen: dict[tuple, int] = {}
    uniq: list[Example] = []
    for ex in examples:
        key = (bucket_of(ex), normalize_key(prompt_key(ex)), normalize_key(ex.output))
        if key in seen:
            j = seen[key]
            if (ex.quality_score or 0) > (uniq[j].quality_score or 0):
                uniq[j] = ex
            stats["exact"] += 1
        else:
            seen[key] = len(uniq)
            uniq.append(ex)
    # --- near: MinHash + LSH banding within a (task, stage, subtype) bucket ------------------------
    mh = MinHasher(num_perm)
    rows = num_perm // bands
    sets = [shingles(prompt_key(ex)) for ex in uniq]
    sigs = [mh.signature(s) for s in sets]
    uf = _UF(len(uniq))
    tables: list[dict[tuple, list[int]]] = [defaultdict(list) for _ in range(bands)]
    for i, sig in enumerate(sigs):
        for b in range(bands):
            key = (bucket_of(uniq[i]), tuple(sig[b * rows : (b + 1) * rows].tolist()))
            for j in tables[b][key]:
                if uf.find(i) != uf.find(j) and jaccard(sets[i], sets[j]) >= threshold:
                    uf.union(i, j)
            tables[b][key].append(i)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(uniq)):
        clusters[uf.find(i)].append(i)
    keep_idx = []
    for members in clusters.values():
        best = max(members, key=lambda m: ((uniq[m].quality_score or 0), -m))
        keep_idx.append(best)
        stats["near"] += len(members) - 1
    keep_idx.sort()
    stats["output"] = len(keep_idx)
    return [uniq[i] for i in keep_idx], stats
