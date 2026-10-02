"""Safe tool execution: parse -> validate -> run with a timeout -> structured observation."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from forgellm.tools.registry import ToolRegistry
from forgellm.tools.schemas import validate_arguments

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ToolResult:
    ok: bool
    call: ToolCall | None
    observation: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0

    def as_text(self) -> str:
        payload = self.observation if self.ok else {"error": self.error}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def parse_tool_call(text: str) -> ToolCall | None:
    m = TOOL_CALL_RE.search(text)
    blob = m.group(1) if m else None
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    return ToolCall(obj["name"], args if isinstance(args, dict) else {}, blob)


def execute(call: ToolCall, registry: ToolRegistry, timeout_s: float = 5.0) -> ToolResult:
    spec = registry.spec(call.name)
    if spec is None:
        return ToolResult(False, call, error=f"unknown tool '{call.name}'")
    v = validate_arguments(spec, call.arguments)
    if not v.ok:
        return ToolResult(False, call, error="; ".join(v.errors))
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["out"] = registry.backends[call.name](**call.arguments)
        except Exception as e:  # tool errors are observations, not crashes
            box["err"] = f"{type(e).__name__}: {e}"

    t0 = time.perf_counter()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    ms = (time.perf_counter() - t0) * 1000
    if th.is_alive():
        return ToolResult(False, call, error=f"timeout after {timeout_s}s", latency_ms=ms)
    if "err" in box:
        return ToolResult(False, call, error=box["err"], latency_ms=ms)
    return ToolResult(True, call, observation=box["out"], latency_ms=ms)
