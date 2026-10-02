"""Run model-generated Python against unit tests in an isolated subprocess (no site packages, CPU/memory limits,
wall-clock timeout, static screen for obviously dangerous constructs)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_DANGEROUS = re.compile(r"\b(import\s+(os|sys|subprocess|shutil|socket|requests|pathlib)|__import__|open\s*\(|eval\s*\(|exec\s*\(|compile\s*\(|input\s*\()")

_RUNNER = r"""
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
except Exception:
    pass
spec = json.loads(sys.stdin.read())
ns = {}
try:
    exec(compile(spec["code"], "<candidate>", "exec"), ns)
    fn = ns[spec["entry"]]
except Exception as e:
    print(json.dumps({"ok": False, "error": "load: " + type(e).__name__, "passed": 0, "total": len(spec["tests"])}))
    raise SystemExit(0)
passed = 0
for t in spec["tests"]:
    try:
        if fn(*t["args"]) == t["expected"]:
            passed += 1
    except Exception:
        pass
print(json.dumps({"ok": passed == len(spec["tests"]), "passed": passed, "total": len(spec["tests"])}))
"""


def run_tests(code: str, entry: str, tests: list[dict[str, Any]], timeout: float = 6.0) -> dict[str, Any]:
    if _DANGEROUS.search(code):
        return {"ok": False, "error": "blocked: unsafe construct", "passed": 0, "total": len(tests)}
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=json.dumps({"code": code, "entry": entry, "tests": tests}),
                           capture_output=True, text=True, timeout=timeout)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "passed": 0, "total": len(tests)}
    except Exception as e:
        return {"ok": False, "error": f"runner: {type(e).__name__}", "passed": 0, "total": len(tests)}


def run_many(jobs: list[tuple[str, str, list[dict[str, Any]]]], workers: int = 6) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: run_tests(*j), jobs))
