"""Output parsing: pull structured content (JSON, code, final answers, citations) out of free-form model text."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|python|py)?\s*\n?(.*?)```", re.S)


def _balanced_object(text: str) -> str | None:
    """First balanced {...} span, aware of JSON strings and escapes."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(text: str, repair: bool = True) -> tuple[Any | None, str]:
    """Returns (object, status) where status is 'strict' (whole text was JSON), 'embedded' (found inside prose/fences),
    'repaired' (needed trailing-comma / quote fixes) or 'failed'."""
    t = text.strip()
    try:
        return json.loads(t), "strict"
    except json.JSONDecodeError:
        pass
    fence = _FENCE.search(t)
    cand = fence.group(1).strip() if fence else t
    blob = _balanced_object(cand)
    if blob is None:
        return None, "failed"
    try:
        return json.loads(blob), "embedded"
    except json.JSONDecodeError:
        if not repair:
            return None, "failed"
    fixed = re.sub(r",\s*([}\]])", r"\1", blob)
    fixed = re.sub(r"(?<![\w\"])'([^']*)'(?![\w\"])", r'"\1"', fixed)
    try:
        return json.loads(fixed), "repaired"
    except json.JSONDecodeError:
        return None, "failed"


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip("\n")


_ANS = re.compile(r"####\s*(.+?)\s*$", re.S)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?")


def extract_final_answer(text: str) -> str | None:
    m = _ANS.search(text.strip())
    return m.group(1).strip() if m else None


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    m = _NUM.search(s.replace(",", ""))
    try:
        return float(m.group(0)) if m else None
    except ValueError:
        return None


def last_number(text: str) -> float | None:
    nums = _NUM.findall(text.replace(",", ""))
    try:
        return float(nums[-1]) if nums else None
    except ValueError:
        return None


def extract_citations(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]
