"""Dataset schema: the single record type every stage of the data pipeline reads and writes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from forgellm.utils import stable_hash

TASK_TYPES = ("technical_qa", "reasoning", "coding", "extraction", "tool_use", "grounded_qa")
EXTRA_TASK_TYPES = ("safety", "general")          # replay data, used only to protect existing capabilities
ALL_TASK_TYPES = TASK_TYPES + EXTRA_TASK_TYPES
DIFFICULTIES = ("easy", "medium", "hard")

# Which LoRA adapter owns which task type (the router maps queries to these names).
TASK_TO_ADAPTER = {
    "technical_qa": "technical_reasoning", "reasoning": "technical_reasoning",
    "grounded_qa": "technical_reasoning", "coding": "coding",
    "extraction": "extraction", "tool_use": "tool_use",
}
ADAPTERS = ("technical_reasoning", "coding", "extraction", "tool_use")


@dataclass
class Example:
    instruction: str
    output: str
    task_type: str
    input: str = ""
    difficulty: str = "medium"
    source: str = "synthetic"
    quality_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = stable_hash(self.task_type, self.instruction, self.input, self.output)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        known = {k: d[k] for k in ("instruction", "output", "task_type", "input", "difficulty",
                                   "source", "quality_score", "meta", "id") if k in d}
        return cls(**known)

    @property
    def group(self) -> str:
        return str(self.meta.get("group") or self.id)


@dataclass
class PreferencePair:
    instruction: str
    input: str
    chosen: str
    rejected: str
    task_type: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_example(self, which: str) -> Example:
        return Example(self.instruction, self.chosen if which == "chosen" else self.rejected,
                       self.task_type, self.input, meta=dict(self.meta))


def validate_record(d: Any) -> list[str]:
    """Structural validation of one raw record (before it becomes an Example)."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["record is not an object"]
    for key in ("instruction", "output", "task_type"):
        v = d.get(key)
        if not isinstance(v, str):
            errs.append(f"'{key}' missing or not a string")
        elif key != "input" and not v.strip():
            errs.append(f"'{key}' is empty")
    if d.get("task_type") not in ALL_TASK_TYPES:
        errs.append(f"unknown task_type {d.get('task_type')!r}")
    if "input" in d and not isinstance(d["input"], str):
        errs.append("'input' must be a string")
    if d.get("difficulty", "medium") not in DIFFICULTIES:
        errs.append(f"unknown difficulty {d.get('difficulty')!r}")
    q = d.get("quality_score")
    if q is not None and not (isinstance(q, (int, float)) and 0 <= q <= 1):
        errs.append("quality_score must be in [0, 1]")
    if "meta" in d and not isinstance(d["meta"], dict):
        errs.append("'meta' must be an object")
    try:
        json.dumps(d)
    except TypeError:
        errs.append("record is not JSON-serialisable")
    return errs
