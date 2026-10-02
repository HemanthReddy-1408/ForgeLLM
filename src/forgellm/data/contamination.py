"""Contamination detection between training prompts and every evaluation prompt / benchmark item.

Three signals, strongest first:
  1. exact match after normalisation
  2. containment: one prompt is the other plus at most a few extra tokens (polite prefix, suffix, quoting)
  3. 8-gram overlap fraction, only for long prompts (documents, code) where n-grams are meaningful

Short templated prompts ("What is X?" vs "What is Y?") are *not* flagged: sharing a template is not leakage."""

from __future__ import annotations

import zlib
from collections import defaultdict

from forgellm.data.dedup import normalize_key, prompt_key
from forgellm.data.schemas import Example

MIN_LONG = 40          # tokens before n-gram overlap is considered meaningful
MAX_EXTRA = 3          # containment only counts when the container adds at most this many tokens


def _grams(toks: list[str], n: int) -> set[int]:
    return {zlib.crc32(" ".join(toks[i : i + n]).encode()) for i in range(len(toks) - n + 1)}


class ContaminationIndex:
    def __init__(self, eval_texts: list[str], ngram: int = 8) -> None:
        self.ngram = ngram
        self.norm = [normalize_key(t) for t in eval_texts]
        self.exact = set(self.norm)
        self.grams: set[int] = set()
        self.by_len: dict[int, list[str]] = defaultdict(list)
        for t in self.norm:
            toks = t.split()
            self.by_len[len(toks)].append(f" {t} ")
            if len(toks) >= MIN_LONG:
                self.grams |= _grams(toks, ngram)
        self.lens = sorted(self.by_len)

    def overlap(self, text: str) -> float:
        t = normalize_key(text)
        if not t:
            return 0.0
        if t in self.exact:
            return 1.0
        toks = t.split()
        padded = f" {t} "
        n = len(toks)
        # containment, both directions, restricted to comparable lengths
        for ln in self.lens:
            if ln < 3:
                continue
            if ln <= n <= ln + MAX_EXTRA:  # eval prompt inside this one
                if any(e in padded for e in self.by_len[ln]):
                    return 1.0
            elif n >= 3 and n < ln <= n + MAX_EXTRA:  # this prompt inside an eval prompt
                if any(padded in e for e in self.by_len[ln]):
                    return 1.0
        if n >= MIN_LONG:
            g = _grams(toks, self.ngram)
            return len(g & self.grams) / len(g) if g else 0.0
        return 0.0


def decontaminate(train: list[Example], eval_examples: list[Example], extra_texts: list[str] | None = None,
                  ngram: int = 8, threshold: float = 0.9) -> tuple[list[Example], dict[str, object]]:
    eval_texts = [prompt_key(e) for e in eval_examples] + list(extra_texts or [])
    idx = ContaminationIndex(eval_texts, ngram)
    kept: list[Example] = []
    removed: dict[str, int] = defaultdict(int)
    flagged: list[dict[str, object]] = []
    for ex in train:
        ov = idx.overlap(prompt_key(ex))
        if ov >= threshold:
            removed[ex.task_type] += 1
            if len(flagged) < 12:
                flagged.append({"id": ex.id, "task": ex.task_type, "overlap": round(ov, 2), "prompt": prompt_key(ex)[:90]})
        else:
            kept.append(ex)
    return kept, {"removed": sum(removed.values()), "by_task": dict(removed), "examples": flagged,
                  "eval_items": len(eval_texts), "threshold": threshold}
