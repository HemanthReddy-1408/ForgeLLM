"""Tool registry: specs + deterministic simulated backends so tool-use can be trained and tested offline."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from forgellm.tools.schemas import ToolSpec

GPU_FLEET = {  # (host, gpu_id) -> status. Deterministic stand-in for a cluster API.
    i: {"utilization_pct": (17 * i + 23) % 100, "memory_used_gb": round(((i * 7) % 24) + 0.5, 1),
        "temperature_c": 48 + (i * 5) % 30, "state": "busy" if i % 3 else "idle"}
    for i in range(8)
}
RUN_METRICS = {
    "loss": lambda rid: round(0.3 + (sum(map(ord, rid)) % 90) / 100, 3),
    "accuracy": lambda rid: round(60 + (sum(map(ord, rid)) % 350) / 10, 1),
    "perplexity": lambda rid: round(3 + (sum(map(ord, rid)) % 200) / 10, 1),
    "f1": lambda rid: round(0.4 + (sum(map(ord, rid)) % 55) / 100, 2),
}
UNIT_FACTORS = {  # to a base unit per family
    ("gb", "mb"): 1000.0, ("gb", "kb"): 1_000_000.0, ("mb", "kb"): 1000.0, ("tb", "gb"): 1000.0,
    ("gib", "mib"): 1024.0, ("s", "ms"): 1000.0, ("min", "s"): 60.0, ("h", "min"): 60.0,
    ("tflops", "gflops"): 1000.0, ("km", "m"): 1000.0, ("m", "cm"): 100.0,
}
CITIES = ["Hyderabad", "Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune", "London", "Berlin",
          "Tokyo", "Singapore", "Austin", "Seattle", "Toronto", "Dublin", "Sydney", "Paris"]

TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("get_gpu_status", "Get utilisation, memory and state of a GPU.",
             {"gpu_id": {"type": "integer", "minimum": 0, "maximum": 7}}, ("gpu_id",)),
    ToolSpec("get_run_metric", "Get the latest value of a metric for a training run.",
             {"run_id": {"type": "string"},
              "metric": {"type": "string", "enum": ["loss", "accuracy", "perplexity", "f1"]}},
             ("run_id", "metric")),
    ToolSpec("search_docs", "Search the internal engineering documentation.",
             {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
             ("query",)),
    ToolSpec("calculate", "Evaluate an arithmetic expression.",
             {"expression": {"type": "string"}}, ("expression",)),
    ToolSpec("convert_units", "Convert a value between units.",
             {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
             ("value", "from_unit", "to_unit")),
    ToolSpec("get_weather", "Get the current weather for a city.",
             {"city": {"type": "string"}}, ("city",)),
    ToolSpec("schedule_job", "Queue a training job on the cluster.",
             {"name": {"type": "string"}, "gpu_count": {"type": "integer", "minimum": 1, "maximum": 8},
              "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
             ("name", "gpu_count")),
    ToolSpec("search_papers", "Search research papers by topic, optionally from a given year.",
             {"query": {"type": "string"}, "year": {"type": "integer", "minimum": 2015, "maximum": 2026}},
             ("query",)),
]}

_KEYWORDS = {'get_gpu_status': 'gpu utilization utilisation busy free idle memory temperature hot status usage', 'get_run_metric': 'run training metric loss accuracy f1 perplexity score latest current doing', 'search_docs': 'docs documentation runbook wiki internal guide policy find lookup search', 'calculate': 'calculate compute math arithmetic times plus minus divided multiply multiplied sum product what is', 'convert_units': 'convert conversion units gb mb kb tb ms seconds minutes hours km meters how many', 'get_weather': 'weather rain raining rainy temperature forecast climate hot cold sunny humid city today', 'schedule_job': 'schedule queue start launch submit job training gpus priority cluster run', 'search_papers': 'papers paper research arxiv literature study studies publications year'}
TOOLS = {k: ToolSpec(t.name, t.description, t.parameters, t.required, _KEYWORDS.get(k, '')) for k, t in TOOLS.items()}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only (AST whitelist, bounded exponent) — never calls eval()."""
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    if len(expr) > 200:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _get_gpu_status(gpu_id: int) -> dict[str, Any]:
    return {"gpu_id": gpu_id, **GPU_FLEET[gpu_id]}


def _get_run_metric(run_id: str, metric: str) -> dict[str, Any]:
    return {"run_id": run_id, "metric": metric, "value": RUN_METRICS[metric](run_id)}


def _calculate(expression: str) -> dict[str, Any]:
    v = safe_eval(expression)
    return {"expression": expression, "result": round(v, 6) if isinstance(v, float) else v}


def _convert_units(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    f, t = from_unit.lower(), to_unit.lower()
    if (f, t) in UNIT_FACTORS:
        r = value * UNIT_FACTORS[(f, t)]
    elif (t, f) in UNIT_FACTORS:
        r = value / UNIT_FACTORS[(t, f)]
    elif f == t:
        r = value
    else:
        raise ValueError(f"cannot convert {from_unit} to {to_unit}")
    return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": round(r, 6)}


def _get_weather(city: str) -> dict[str, Any]:
    h = sum(map(ord, city.lower()))
    return {"city": city, "temperature_c": 12 + h % 24, "condition": ["clear", "cloudy", "rain", "haze"][h % 4]}


def _schedule_job(name: str, gpu_count: int, priority: str = "normal") -> dict[str, Any]:
    return {"job_id": f"job-{sum(map(ord, name)) % 9000 + 1000}", "name": name,
            "gpu_count": gpu_count, "priority": priority, "status": "queued"}


def _search_papers(query: str, year: int | None = None) -> dict[str, Any]:
    h = sum(map(ord, query.lower())) + (year or 0)
    return {"query": query, "year": year, "count": 5 + h % 40,
            "top_title": f"A Study of {query.title()} at Scale"}


def default_backends(search_docs: Callable[[str, int], list[dict[str, Any]]] | None = None) -> dict[str, Callable[..., Any]]:
    def _search_docs(query: str, top_k: int = 3) -> dict[str, Any]:
        hits = search_docs(query, top_k) if search_docs else []
        return {"query": query, "results": hits}

    return {"get_gpu_status": _get_gpu_status, "get_run_metric": _get_run_metric, "search_docs": _search_docs,
            "calculate": _calculate, "convert_units": _convert_units, "get_weather": _get_weather,
            "schedule_job": _schedule_job, "search_papers": _search_papers}


class ToolRegistry:
    def __init__(self, tools: dict[str, ToolSpec] | None = None,
                 backends: dict[str, Callable[..., Any]] | None = None) -> None:
        self.tools = dict(tools or TOOLS)
        self.backends = backends or default_backends()

    def spec(self, name: str) -> ToolSpec | None:
        return self.tools.get(name)

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        self.tools[spec.name] = spec
        self.backends[spec.name] = fn

    def prompt(self, names: list[str] | None = None) -> str:
        return "\n".join(self.tools[n].to_prompt() for n in (names or list(self.tools)))

