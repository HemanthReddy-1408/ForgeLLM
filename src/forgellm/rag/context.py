"""Context assembly: turn retrieved documents into the numbered passage block the grounded-QA adapter was trained on."""

from __future__ import annotations

from typing import Any

from forgellm.data.synth import format_passages


def build_context(hits: list[dict[str, Any]], budget_chars: int = 2400) -> tuple[str, list[dict[str, Any]]]:
    """Dedupe by id, keep rank order, and stop before the character budget is exceeded. Returns (passages, kept hits)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    used = 0
    for h in hits:
        if h["id"] in seen:
            continue
        cost = len(h["title"]) + len(h["text"]) + 8
        if kept and used + cost > budget_chars:
            break
        seen.add(h["id"])
        kept.append(h)
        used += cost
    return format_passages(kept), kept


def cited_documents(answer_text: str, kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import re

    idx = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}
    return [kept[i - 1] for i in sorted(idx) if 1 <= i <= len(kept)]
