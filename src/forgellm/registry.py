"""Model registry: versioned adapters with lineage, and promotion gated by the regression evaluation.

    candidate --(eval vs base + regression gate)--> production | rejected

An adapter is only served as `production` if it improved its target skill and did not damage anything it was
not meant to touch (general ability, safety, other skills)."""

from __future__ import annotations

from typing import Any

from forgellm.data.schemas import TASK_TO_ADAPTER
from forgellm.evaluation.regression import GateResult, compare_reports, render_comparison
from forgellm.store import Store, get_store

ADAPTER_TARGETS: dict[str, list[str]] = {}
for _task, _ad in TASK_TO_ADAPTER.items():
    ADAPTER_TARGETS.setdefault(_ad, []).append(_task)


def promote(name: str, version: int | None, base_report: dict[str, Any], tuned_report: dict[str, Any],
            targets: list[str] | None = None, store: Store | None = None, min_gain: float = 0.03,
            max_drop: float = 0.03) -> tuple[GateResult, str]:
    store = store or get_store()
    rows = store.adapters(name)
    if not rows:
        raise KeyError(f"adapter '{name}' is not registered")
    version = version or rows[-1]["version"]
    targets = targets or ADAPTER_TARGETS.get(name, [])
    gate = compare_reports(base_report, tuned_report, targets, min_gain=min_gain, max_drop=max_drop)
    status = "production" if gate.passed else "rejected"
    if gate.passed:  # only one production version per adapter name
        for r in rows:
            if r["status"] == "production" and r["version"] != version:
                store.set_adapter_status(name, r["version"], "archived")
    store.set_adapter_status(name, version, status, gate.to_dict())
    store.save_eval(f"{name}-v{version}-{tuned_report['label']}", f"{name}:v{version}", "full", tuned_report)
    return gate, status


def lineage(name: str, store: Store | None = None) -> list[dict[str, Any]]:
    store = store or get_store()
    return [{"name": r["name"], "version": r["version"], "status": r["status"], "base_model": r["base_model"], "method": r["method"],
             "task": r["task"], "params": r["num_parameters"], "data_hash": r["data_hash"], "config_hash": r["config_hash"],
             "gate": (r["gate"] or {}).get("passed")} for r in store.adapters(name)]


def render_registry(store: Store | None = None) -> str:
    store = store or get_store()
    rows = store.adapters()
    if not rows:
        return "(registry empty)"
    lines = [f"{'adapter':<22} {'ver':>3} {'status':<11} {'method':<10} {'params':>10} {'task':<14} gate"]
    for r in rows:
        g = r["gate"]
        lines.append(f"{r['name']:<22} {r['version']:>3} {r['status']:<11} {r['method']:<10} {r['num_parameters']:>10,} {r['task']:<14} "
                     f"{'-' if g is None else ('pass' if g['passed'] else 'FAIL')}")
    return "\n".join(lines)


__all__ = ["ADAPTER_TARGETS", "lineage", "promote", "render_comparison", "render_registry"]
