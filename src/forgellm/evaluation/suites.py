"""Capability suites. Each one turns (examples, model outputs) into per-item scores in [0, 1] plus named metrics.

    technical_qa / grounded_qa   domain knowledge & grounding        reasoning   numeric answers
    coding                       executed unit tests                  extraction  JSON validity / schema / field F1
    tool_use                     tool choice + arguments + abstention + observation use"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from forgellm.data.schemas import Example
from forgellm.evaluation import sandbox
from forgellm.inference.parser import (
    extract_citations,
    extract_code,
    extract_final_answer,
    extract_json,
    last_number,
    parse_number,
)
from forgellm.tools.execution import parse_tool_call
from forgellm.tools.registry import TOOLS
from forgellm.tools.schemas import validate_arguments, validate_json_schema

_STOP = set(["the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "is", "are", "be", "as", "that", "this", "it", "its", "by", "from", "at", "which", "can", "has", "have", "not", "but", "if", "you", "your"])
_W = re.compile(r"[a-z0-9]+")


def _tokens(t: str) -> list[str]:
    return _W.findall(t.lower())


def token_f1(pred: str, ref: str) -> float:
    p, r = Counter(_tokens(pred)), Counter(_tokens(ref))
    common = sum((p & r).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def keyterm_recall(pred: str, ref: str) -> float:
    terms = {t for t in _tokens(ref) if len(t) >= 5 and t not in _STOP}
    if not terms:
        return 1.0
    pt = set(_tokens(pred))
    return sum(t in pt for t in terms) / len(terms)


def _norm_val(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.lower().split())
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)


# --------------------------------------------------------------------------------------------------------------
def score_technical_qa(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    scores, f1s, recs = [], [], []
    for ex, o in zip(exs, outs):
        f1, rec = token_f1(o, ex.output), keyterm_recall(o, ex.output)
        f1s.append(f1)
        recs.append(rec)
        scores.append(0.5 * f1 + 0.5 * rec)
    return {"scores": scores, "metrics": {"token_f1": _m(f1s), "keyterm_recall": _m(recs)}}


def score_reasoning(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    scores, fmt = [], []
    for ex, o in zip(exs, outs):
        gold = float(ex.meta["answer"])
        ans = extract_final_answer(o)
        pred = parse_number(ans) if ans is not None else last_number(o)
        fmt.append(float(ans is not None))
        scores.append(float(pred is not None and _close(pred, gold)))
    return {"scores": scores, "metrics": {"accuracy": _m(scores), "format_rate": _m(fmt)}, "format": fmt}


def score_coding(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    jobs, idx = [], []
    scores = [0.0] * len(exs)
    syntax = [0.0] * len(exs)
    for i, (ex, o) in enumerate(zip(exs, outs)):
        if ex.meta.get("subtype") == "trace":
            ans = extract_final_answer(o)
            scores[i] = float(ans is not None and ans.strip() == ex.meta["answer"].strip())
            syntax[i] = float(ans is not None)
            continue
        code = extract_code(o) if "```" in o else o
        # fix-mode answers contain prose before the code block; only the fenced block is run
        try:
            compile(code, "<c>", "exec")
            syntax[i] = 1.0
        except SyntaxError:
            continue
        jobs.append((code, ex.meta["entry"], ex.meta["tests"]))
        idx.append(i)
    for i, r in zip(idx, sandbox.run_many(jobs)):
        scores[i] = float(r["ok"])
    sub: dict[str, list[float]] = {}
    for ex, s in zip(exs, scores):
        sub.setdefault(ex.meta.get("subtype", "write"), []).append(s)
    return {"scores": scores, "metrics": {"pass@1": _m(scores), "syntax_valid": _m(syntax),
                                          **{f"pass@1[{k}]": _m(v) for k, v in sub.items()}}, "format": syntax}


def score_extraction(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    scores, strict, anyj, schema_ok, exact = [], [], [], [], []
    for ex, o in zip(exs, outs):
        gold = json.loads(ex.output)
        obj, status = extract_json(o)
        strict.append(float(status == "strict"))
        anyj.append(float(obj is not None))
        if not isinstance(obj, dict):
            scores.append(0.0)
            schema_ok.append(0.0)
            exact.append(0.0)
            continue
        schema_ok.append(float(not validate_json_schema(obj, ex.meta["schema"])))
        hits = sum(_norm_val(obj.get(k)) == _norm_val(v) for k, v in gold.items())
        f1 = hits / len(gold)
        extra = sum(k not in gold for k in obj)
        exact.append(float(hits == len(gold) and not extra))
        scores.append(f1)
    return {"scores": scores, "metrics": {"field_accuracy": _m(scores), "valid_json_strict": _m(strict), "valid_json_any": _m(anyj),
                                          "schema_valid": _m(schema_ok), "exact_object": _m(exact)}, "format": strict}


_OBS_KEYS = {"get_gpu_status": ["state", "utilization_pct", "temperature_c"], "get_run_metric": ["value"], "calculate": ["result"],
             "convert_units": ["result"], "get_weather": ["temperature_c", "condition"], "schedule_job": ["job_id"],
             "search_papers": ["count"], "search_docs": []}


def score_tool_use(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    scores, by_stage = [], {"call": [], "final": [], "abstain": []}
    name_ok, args_ok, valid = [], [], []
    for ex, o in zip(exs, outs):
        stage = ex.meta.get("stage")
        if stage == "call":
            call = parse_tool_call(o + ("</tool_call>" if "<tool_call>" in o and "</tool_call>" not in o else ""))
            spec = TOOLS.get(call.name) if call else None
            v = float(call is not None and spec is not None and validate_arguments(spec, call.arguments).ok)
            valid.append(v)
            n = float(call is not None and call.name == ex.meta["tool"])
            a = float(n and {k: _norm_val(x) for k, x in call.arguments.items()} == {k: _norm_val(x) for k, x in ex.meta["arguments"].items()})
            name_ok.append(n)
            args_ok.append(a)
            s = a
        elif stage == "abstain":
            s = float("<tool_call>" not in o and len(o.strip()) > 0)
        else:
            obs = ex.meta.get("observation") or {}
            keys = _OBS_KEYS.get(ex.meta["tool"], [])
            vals = [str(obs[k]).lower() for k in keys if k in obs]
            low = o.lower().replace(",", "")
            s = float("<tool_call>" not in o and (all(v in low for v in vals) if vals else len(o.strip()) > 0))
        by_stage[stage].append(s)
        scores.append(s)
    met = {"call_exact": _m(by_stage["call"]), "abstain_acc": _m(by_stage["abstain"]), "final_uses_observation": _m(by_stage["final"]),
           "tool_name_acc": _m(name_ok), "args_exact": _m(args_ok), "valid_call_rate": _m(valid)}
    return {"scores": scores, "metrics": met, "format": valid}


def score_grounded(exs: list[Example], outs: list[str]) -> dict[str, Any]:
    scores, cite = [], []
    for ex, o in zip(exs, outs):
        val = ex.meta.get("value")
        low = o.lower().replace(",", "")
        if val is None:
            s = float(any(p in low for p in ("do not", "does not", "not say", "no information", "not contain", "cannot", "not mentioned")))
            scores.append(s)
            continue
        num = str(val).split()[0].lower()   # the unit is implied by the parameter; the number is the fact
        s = float(re.search(rf"(?<![\d.]){re.escape(num)}(?![\d]|\.\d)", low) is not None)
        scores.append(s)
        cites = extract_citations(o)
        cite.append(float(ex.meta.get("cite") in cites))
    return {"scores": scores, "metrics": {"answer_match": _m(scores), "citation_acc": _m(cite)}}


SCORERS = {"technical_qa": score_technical_qa, "reasoning": score_reasoning, "coding": score_coding,
           "extraction": score_extraction, "tool_use": score_tool_use, "grounded_qa": score_grounded}


def _m(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else float("nan")
