"""Regression protection: compare a tuned model against the base on every capability, with statistics, and gate promotion.

For each suite the per-item scores of the two models are paired by item, and a paired bootstrap gives a confidence
interval for the difference. The gate fails when a *protected* capability (general, safety, anything outside the
adapter's target skill) drops by more than `max_drop` — the "domain ↑, general ↓" trade-off made explicit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from forgellm.evaluation.harness import CAPABILITY_OF


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean(b - a) and its 95% CI. Lists must be the same length and item-aligned."""
    m = min(len(a), len(b))
    d = [b[i] - a[i] for i in range(m)]
    if not d:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(d) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]


@dataclass
class SuiteDelta:
    suite: str
    base: float
    tuned: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str            # improved | regressed | unchanged

    def row(self) -> str:
        return f"{self.suite:<13} {self.base:6.3f} -> {self.tuned:6.3f}  Δ {self.delta:+.3f}  [{self.ci_low:+.3f}, {self.ci_high:+.3f}]  {self.verdict}"


@dataclass
class GateResult:
    passed: bool
    deltas: list[SuiteDelta]
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ppl_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": self.reasons, "warnings": self.warnings, "ppl_ratio": self.ppl_ratio,
                "deltas": [d.__dict__ for d in self.deltas]}


PROTECTED = ("general", "safety")


def compare_reports(base: dict[str, Any], tuned: dict[str, Any], target_suites: list[str] | None = None,
                    min_gain: float = 0.03, max_drop: float = 0.03, max_ppl_ratio: float = 1.15, seed: int = 0) -> GateResult:
    """Evidence-based promotion gate.

    * target skills  — their *mean* gain must reach `min_gain`, and none may regress significantly;
    * protected      — `general` and `safety` fail only if the drop exceeds `max_drop` AND the paired-bootstrap CI is
                       entirely below zero (with ~20-50 items a point-estimate dip is usually noise: it is reported as a warning);
    * other skills   — reported as warnings (the router keeps a skill's adapter away from other skills' requests);
    * perplexity     — held-out perplexity ratio is deterministic, so it is a hard limit."""
    deltas: list[SuiteDelta] = []
    reasons, warnings = [], []
    for name, tb in tuned["suites"].items():
        bb = base["suites"].get(name)
        if bb is None:
            continue
        d, lo, hi = paired_bootstrap(bb["scores"], tb["scores"], seed=seed)
        verdict = "improved" if lo > 0 else "regressed" if hi < 0 else "unchanged"
        deltas.append(SuiteDelta(name, bb["score"], tb["score"], round(d, 4), round(lo, 4), round(hi, 4), verdict))
    targets = [d for d in deltas if d.suite in set(target_suites or [])]
    if targets:
        mean_gain = sum(d.delta for d in targets) / len(targets)
        if mean_gain < min_gain:
            reasons.append(f"target skills improved by only {mean_gain:+.3f} on average (< {min_gain})")
        for d in targets:
            if d.verdict == "regressed":
                reasons.append(f"target '{d.suite}' regressed significantly ({d.delta:+.3f}, CI [{d.ci_low:+.3f}, {d.ci_high:+.3f}])")
            elif d.delta < min_gain:
                warnings.append(f"target '{d.suite}' did not improve ({d.delta:+.3f})")
    for d in deltas:
        if d in targets or d.delta >= -max_drop:
            continue
        note = f"'{d.suite}' dropped {d.delta:+.3f} (CI [{d.ci_low:+.3f}, {d.ci_high:+.3f}])"
        if d.suite in PROTECTED and d.ci_high < 0:
            reasons.append(f"protected {note}, significant")
        elif d.suite in PROTECTED:
            warnings.append(f"protected {note} — beyond the limit but not statistically significant at this sample size")
        else:
            warnings.append(f"cross-skill {note}")
    ratio = None
    pb, pt = base["suites"].get("general", {}).get("metrics", {}), tuned["suites"].get("general", {}).get("metrics", {})
    if "heldout_ppl" in pb and "heldout_ppl" in pt:
        ratio = round(pt["heldout_ppl"] / pb["heldout_ppl"], 4)
        if ratio > max_ppl_ratio:
            reasons.append(f"held-out perplexity rose {ratio:.2f}x (limit {max_ppl_ratio}x)")
    gains = bool(targets) and sum(d.delta for d in targets) / len(targets) > min_gain
    losses = [d for d in deltas if d.suite in PROTECTED and d.delta < -max_drop / 2]
    if gains and losses:
        warnings.append("target skills ↑ while " + ", ".join(d.suite for d in losses) + " ↓ — forgetting trade-off")
    return GateResult(not reasons, deltas, reasons, warnings, ratio)


def render_comparison(base: dict[str, Any], tuned: dict[str, Any], gate: GateResult | None = None) -> str:
    lines = [f"{'suite':<13} {base['label']:>8}   {tuned['label']:>8}      Δ    95% CI (paired bootstrap)"]
    g = gate or compare_reports(base, tuned)
    for d in g.deltas:
        mark = {"improved": "▲", "regressed": "▼", "unchanged": "·"}[d.verdict]
        lines.append(f"{d.suite:<13} {d.base:8.3f}   {d.tuned:8.3f}  {d.delta:+.3f}  [{d.ci_low:+.3f}, {d.ci_high:+.3f}] {mark}")
    if g.ppl_ratio is not None:
        lines.append(f"{'heldout ppl':<13} ratio {g.ppl_ratio:.3f}x")
    lines.append(f"gate: {'PASS' if g.passed else 'FAIL'}" + ("".join(f"\n  ✗ {r}" for r in g.reasons)) + ("".join(f"\n  ! {w}" for w in g.warnings)))
    return "\n".join(lines)


def capability_of(suite: str) -> str:
    return CAPABILITY_OF[suite]
