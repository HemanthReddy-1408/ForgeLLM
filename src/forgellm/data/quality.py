"""Quality scoring: task-specific validity checks (parse the JSON, compile the code, validate the tool call)
combined with generic text-quality signals into one score in [0, 1]."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter

from forgellm.data.cleaning import non_ascii_ratio
from forgellm.data.schemas import Example
from forgellm.tools.execution import parse_tool_call
from forgellm.tools.registry import TOOLS
from forgellm.tools.schemas import validate_arguments, validate_json_schema

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_WORD = re.compile(r"\w+")


def distinct_ngram_ratio(text: str, n: int = 3) -> float:
    toks = _WORD.findall(text.lower())
    if len(toks) < n + 1:
        return 1.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def jaccard_words(a: str, b: str) -> float:
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def task_validity(ex: Example) -> tuple[float, str]:
    """1.0 = the target is verifiably well-formed for its task; lower = malformed / truncated."""
    t, out = ex.task_type, ex.output
    if t == "extraction":
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            return 0.0, "invalid_json"
        errs = validate_json_schema(obj, ex.meta.get("schema", {})) if ex.meta.get("schema") else []
        return (0.0, "schema_mismatch") if errs else (1.0, "")
    if t == "tool_use":
        stage = ex.meta.get("stage")
        if stage == "call":
            call = parse_tool_call(out)
            if call is None:
                return 0.0, "unparseable_tool_call"
            spec = TOOLS.get(call.name)
            if spec is None or not validate_arguments(spec, call.arguments).ok:
                return 0.0, "invalid_tool_call"
            return 1.0, ""
        if "<tool_call>" in out:
            return 0.2, "unexpected_tool_call"
        return (1.0, "") if out.strip()[-1:] in ".!?)" else (0.6, "no_terminal_punct")
    if t == "coding":
        if ex.meta.get("subtype") == "trace":
            return (1.0, "") if "####" in out else (0.3, "missing_answer")
        m = _CODE_BLOCK.search(out)
        if not m:
            return 0.2, "no_code_block"
        try:
            ast.parse(m.group(1))
        except SyntaxError:
            return 0.0, "syntax_error"
        if len(out) < 40:
            return 0.4, "too_short"
        return 1.0, ""
    if t == "reasoning":
        m = re.search(r"####\s*(-?[\d.]+)\s*$", out)
        return (1.0, "") if m else (0.0, "missing_final_answer")
    if t == "grounded_qa":
        if ex.meta.get("cite") is None and ex.meta.get("value") is None:
            return (1.0, "") if "not say" in out or "do not" in out else (0.5, "")
        return (1.0, "") if re.search(r"\[\d+\]\.?$", out.strip()) or ex.meta.get("closed_book") else (0.3, "missing_citation")
    if t in ("technical_qa", "safety", "general"):
        s = out.strip()
        if len(s.split()) < 4:
            return 0.4, "too_short"
        return (1.0, "") if s[-1] in ".!?)`\"'" else (0.5, "truncated")
    return 0.8, ""


def quality_score(ex: Example) -> tuple[float, dict[str, float | str]]:
    validity, reason = task_validity(ex)
    rep = distinct_ngram_ratio(ex.output)
    rep_pen = 0.0 if rep >= 0.6 else (0.6 - rep)
    copy = jaccard_words(ex.instruction, ex.output) if ex.task_type in ("technical_qa", "safety") else 0.0
    copy_pen = 0.3 if copy > 0.9 else 0.0
    ascii_pen = min(0.3, non_ascii_ratio(ex.output))
    length_pen = 0.15 if len(ex.output) > 3500 else 0.0
    score = max(0.0, min(1.0, 0.1 + 0.9 * validity - rep_pen - copy_pen - ascii_pen - length_pen))
    return round(score, 3), {"validity": validity, "reason": reason, "distinct3": round(rep, 3)}


def score_all(examples: list[Example]) -> Counter:
    reasons: Counter = Counter()
    for ex in examples:
        ex.quality_score, feats = quality_score(ex)
        if feats["reason"]:
            reasons[f"{ex.task_type}:{feats['reason']}"] += 1
    return reasons
