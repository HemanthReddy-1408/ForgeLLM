"""Cleaning: normalise text, strip markup, redact PII, and drop records that cannot be repaired."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from forgellm.data.schemas import Example, validate_record

_TAG = re.compile(r"</?(?:div|p|span|br|html|body|a|b|i|ul|li|table|tr|td)\b[^>]*>", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.[\w.-]+)")
_MOJIBAKE = re.compile(r"Ã[\x80-\xbf©¨¤¶¼§]|â€[™œ\x9d\x9c]|Â[\xa0-\xbf]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BOILER = re.compile(r"^\s*(as an ai language model|i('| a)m (sorry|unable)|n/?a\s*$|lorem ipsum)", re.I)
SAFE_EMAIL_DOMAINS = {"example.com", "example.org"}


def _redact_email(m: re.Match[str]) -> str:
    return m.group(0) if m.group(1).lower() in SAFE_EMAIL_DOMAINS else "[EMAIL]"


def normalize(text: str, multiline: bool) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _CTRL.sub("", t)
    t = _TAG.sub("", t)
    if multiline:  # keep indentation (code / JSON / passages); only trim trailing blanks and runaway newlines
        t = "\n".join(line.rstrip() for line in t.replace("\r\n", "\n").split("\n"))
        t = re.sub(r"\n{3,}", "\n\n", t)
    else:
        t = re.sub(r"\s+", " ", t)
    return t.strip()


def redact_pii(text: str) -> tuple[str, int]:
    n = 0
    for pat, tag in ((_SSN, "[SSN]"), (_PHONE, "[PHONE]")):
        text, k = pat.subn(tag, text)
        n += k
    text, k = _EMAIL.subn(_redact_email, text)
    return text, n + (1 if "[EMAIL]" in text and k else 0)


def non_ascii_ratio(text: str) -> float:
    return sum(ord(c) > 127 for c in text) / len(text) if text else 0.0


def clean_record(d: dict[str, Any]) -> tuple[Example | None, str]:
    """Returns (example, "ok"|"redacted") or (None, drop_reason)."""
    errs = validate_record(d)
    if errs:
        return None, "schema_invalid"
    if _MOJIBAKE.search(d["instruction"] + d.get("input", "") + d["output"]):
        return None, "mojibake"  # the original characters are unrecoverable, so drop rather than "repair"
    multi_out = d["task_type"] in ("coding", "reasoning", "tool_use", "grounded_qa", "extraction")
    instr = normalize(d["instruction"], multiline=False)
    inp = normalize(d.get("input", ""), multiline=True)
    out = normalize(d["output"], multiline=multi_out)
    if len(out) < 2 or (out.lower() in {"n/a", "none", "null", "tbd"} and d["task_type"] != "extraction"):
        return None, "empty_output"
    if len(instr) < 4:
        return None, "empty_instruction"
    if _BOILER.search(out) and d["task_type"] != "safety":
        return None, "boilerplate_output"
    if non_ascii_ratio(out) > 0.2 and d["task_type"] != "general":
        return None, "non_ascii_junk"
    if len(instr) + len(inp) + len(out) > 12000:
        return None, "too_long"
    redacted = False
    for field_name in ("instr", "inp", "out"):
        val = {"instr": instr, "inp": inp, "out": out}[field_name]
        new, _n = redact_pii(val)
        if new != val:
            redacted = True
            if field_name == "instr":
                instr = new
            elif field_name == "inp":
                inp = new
            else:
                out = new
    meta = dict(d.get("meta") or {})
    ex = Example(instr, out, d["task_type"], inp, d.get("difficulty", "medium"), d.get("source", "synthetic"),
                 d.get("quality_score"), meta)
    return ex, "redacted" if redacted else "ok"


def clean_all(rows: list[dict[str, Any]]) -> tuple[list[Example], Counter]:
    kept: list[Example] = []
    stats: Counter = Counter()
    for d in rows:
        ex, status = clean_record(d)
        stats[status] += 1
        if ex is not None:
            kept.append(ex)
    return kept, stats
