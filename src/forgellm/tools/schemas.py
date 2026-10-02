"""Tool specifications and a small JSON-Schema-subset validator (no external dependency)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_PY_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,),
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]   # param -> {"type", "description", "enum"?, "minimum"?, "maximum"?}
    required: tuple[str, ...] = ()
    keywords: str = ""      # synonyms used only for tool *selection* (never shown to the model)

    def to_prompt(self) -> str:
        """Compact one-line JSON signature used inside system prompts (kept short: it is paid on every call)."""
        props = {}
        for k, v in self.parameters.items():
            p = {"type": v["type"]}
            if "enum" in v:
                p["enum"] = v["enum"]
            props[k] = p
        return json.dumps({"name": self.name, "description": self.description,
                           "parameters": props, "required": list(self.required)}, separators=(",", ":"))


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_arguments(spec: ToolSpec, args: Any) -> ValidationResult:
    errs: list[str] = []
    if not isinstance(args, dict):
        return ValidationResult(False, ["arguments must be an object"])
    for r in spec.required:
        if r not in args:
            errs.append(f"missing required argument '{r}'")
    for k, v in args.items():
        if k not in spec.parameters:
            errs.append(f"unexpected argument '{k}'")
            continue
        p = spec.parameters[k]
        types = _PY_TYPES[p["type"]]
        if isinstance(v, bool) and p["type"] in ("integer", "number"):
            errs.append(f"'{k}' must be {p['type']}, got boolean")
        elif not isinstance(v, types):
            errs.append(f"'{k}' must be {p['type']}, got {type(v).__name__}")
        else:
            if "enum" in p and v not in p["enum"]:
                errs.append(f"'{k}' must be one of {p['enum']}")
            if "minimum" in p and isinstance(v, (int, float)) and v < p["minimum"]:
                errs.append(f"'{k}' below minimum {p['minimum']}")
            if "maximum" in p and isinstance(v, (int, float)) and v > p["maximum"]:
                errs.append(f"'{k}' above maximum {p['maximum']}")
    return ValidationResult(not errs, errs)


def validate_json_schema(obj: Any, schema: dict[str, Any]) -> list[str]:
    """Validate extraction output against {"field": "type"} style schemas: string|integer|number|boolean|null-able."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["output must be a JSON object"]
    for key, tp in schema.items():
        if key not in obj:
            errs.append(f"missing key '{key}'")
            continue
        v = obj[key]
        if v is None:
            continue
        base = tp.rstrip("?")
        if base not in _PY_TYPES:
            continue
        if isinstance(v, bool) and base in ("integer", "number"):
            errs.append(f"'{key}' expected {base}, got boolean")
        elif not isinstance(v, _PY_TYPES[base]):
            errs.append(f"'{key}' expected {base}, got {type(v).__name__}")
    for key in obj:
        if key not in schema:
            errs.append(f"unexpected key '{key}'")
    return errs
