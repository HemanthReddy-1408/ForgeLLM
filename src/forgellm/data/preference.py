"""Preference-pair construction: the gold output is *chosen*; *rejected* is a realistic failure of the same task
(wrong arithmetic, malformed JSON, wrong tool, buggy code, off-topic answer, ungrounded answer)."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from forgellm.data import knowledge as kb
from forgellm.data.schemas import Example, PreferencePair
from forgellm.tools.registry import TOOLS


def _wrong_number(r: random.Random, s: str) -> str:
    n = float(s)
    delta = r.choice([-2, -1, 1, 2, 10]) if n.is_integer() else r.choice([-0.5, 0.5, 1])
    v = n * r.choice([0.5, 2]) if r.random() < 0.4 else n + delta
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def _rej_reasoning(ex: Example, r: random.Random) -> str | None:
    m = re.search(r"####\s*(-?[\d.]+)\s*$", ex.output)
    if not m:
        return None
    bad = _wrong_number(r, m.group(1))
    body = ex.output[: m.start()]
    body = re.sub(re.escape(m.group(1)), bad, body) if r.random() < 0.5 else body
    return f"{body}#### {bad}"


def _rej_extraction(ex: Example, r: random.Random) -> str | None:
    try:
        obj = json.loads(ex.output)
    except json.JSONDecodeError:
        return None
    mode = r.choice(["wrong_value", "drop_key", "prose", "trailing_comma", "extra_key"])
    keys = list(obj)
    if mode == "wrong_value":
        k = r.choice(keys)
        obj[k] = (obj[k] + 1) if isinstance(obj[k], (int, float)) and not isinstance(obj[k], bool) else "unknown"
    elif mode == "drop_key" and len(keys) > 1:
        obj.pop(r.choice(keys))
    elif mode == "extra_key":
        obj["notes"] = "extracted automatically"
    text = json.dumps(obj, ensure_ascii=False)
    if mode == "prose":
        return f"Sure! Here is the JSON you asked for:\n{text}\nLet me know if you need anything else."
    if mode == "trailing_comma":
        return text[:-1] + ",}"
    return text


def _rej_tool(ex: Example, r: random.Random) -> str | None:
    stage = ex.meta.get("stage")
    if stage == "call":
        mode = r.choice(["wrong_tool", "wrong_arg", "prose", "bad_json"])
        name, args = ex.meta["tool"], dict(ex.meta["arguments"])
        if mode == "wrong_tool":
            name = r.choice([t for t in ex.meta["tools"] if t != name])
        elif mode == "wrong_arg":
            k = r.choice(list(args))
            args[k] = args[k] + 1 if isinstance(args[k], int) and not isinstance(args[k], bool) else f"{args[k]}-x"
        elif mode == "prose":
            return "I don't have access to live data, but I can try to answer from memory."
        elif mode == "bad_json":
            return "<tool_call>\n{" + f"'name': '{name}', 'arguments': {args}" + "}\n</tool_call>"
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
    if stage == "final":
        return "The tool call finished, but I'm not able to tell you what it returned."
    return "<tool_call>\n" + json.dumps({"name": r.choice(list(TOOLS)), "arguments": {}}) + "\n</tool_call>"


def _rej_coding(ex: Example, r: random.Random) -> str | None:
    if ex.meta.get("subtype") == "trace":
        m = re.search(r"####\s*(.+)$", ex.output)
        return None if not m else f"The function returns something else.\n#### {r.choice(['0', '[]', 'None'])}"
    swaps = [("sum(", "max("), (" % 2 == 0", " % 2 == 1"), ("x * x", "x + x"), ("x > 0", "x >= 0"), ("max(", "min("), ("len(", "sum(")]
    out = ex.output
    for a, b in r.sample(swaps, len(swaps)):
        if a in out and ex.meta.get("subtype") != "fix":
            return out.replace(a, b, 1)
    if ex.meta.get("subtype") == "fix":
        return "The bug: the code is fine as written. Fixed code:\n" + ex.input
    return re.sub(r"return .*", "return None", out, count=1)


def _rej_qa(ex: Example, r: random.Random, pool: list[Example]) -> str | None:
    mode = r.choice(["other_concept", "vague"])
    if mode == "vague":
        return "It depends on many factors, and there is no single answer that applies in every case."
    others = [p for p in pool if p.task_type == ex.task_type and p.output != ex.output]
    return r.choice(others).output if others else None


def _rej_grounded(ex: Example, r: random.Random) -> str | None:
    val = ex.meta.get("value")
    if val is None:
        return f"The {ex.meta.get('param', 'value')} is 42 [1]."
    mode = r.choice(["wrong_value", "no_cite", "idk"])
    if mode == "idk":
        return "The provided documents do not contain this information."
    if mode == "no_cite":
        return ex.output.rsplit(" [", 1)[0] + "."
    other = kb.PARAMS[ex.meta["param"]][1](r)
    return ex.output.replace(str(val), f"{other}{kb.PARAMS[ex.meta['param']][2]}")


def make_pairs(examples: list[Example], n: int, seed: int = 0, tasks: set[str] | None = None) -> list[PreferencePair]:
    r = random.Random(seed)
    usable = [e for e in examples if e.task_type not in ("safety", "general") and (not tasks or e.task_type in tasks)]
    r.shuffle(usable)
    pairs: list[PreferencePair] = []
    for ex in usable:
        rej: str | None = None
        t = ex.task_type
        if t == "reasoning":
            rej = _rej_reasoning(ex, r)
        elif t == "extraction":
            rej = _rej_extraction(ex, r)
        elif t == "tool_use":
            rej = _rej_tool(ex, r)
        elif t == "coding":
            rej = _rej_coding(ex, r)
        elif t == "grounded_qa":
            rej = _rej_grounded(ex, r)
        elif t == "technical_qa":
            rej = _rej_qa(ex, r, usable)
        if rej and rej.strip() != ex.output.strip():
            meta: dict[str, Any] = {k: v for k, v in ex.meta.items() if k in ("history", "stage", "tools", "tool", "arguments", "schema", "tests", "entry", "subtype", "value", "param")}
            pairs.append(PreferencePair(ex.instruction, ex.input, ex.output, rej, t, meta))
            if len(pairs) >= n:
                break
    return pairs
