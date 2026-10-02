"""Splitting and mixture building: deterministic prompt-hash splits, then a controllable task mixture."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from forgellm.data.dedup import normalize_key, prompt_key
from forgellm.data.schemas import Example
from forgellm.utils import stable_hash


def split_of(ex: Example, val_frac: float = 0.05, test_frac: float = 0.08) -> str:
    """Identical prompts always land in the same split (so a query cannot be both trained on and tested on),
    unless the record is flagged `force_split` (scraped copies that really did leak in from outside)."""
    forced = ex.meta.get("force_split")
    if forced:
        return forced
    b = int(stable_hash(ex.task_type, normalize_key(prompt_key(ex)), n=8), 16) % 10_000 / 10_000
    if b < test_frac:
        return "test"
    if b < test_frac + val_frac:
        return "val"
    return "train"


def build_mixture(pool: list[Example], weights: dict[str, float], total: int, seed: int = 0,
                  allow_upsample: bool = False) -> tuple[list[Example], dict[str, object]]:
    """Sample `total` examples with the requested task proportions.

    Tasks whose pool is smaller than their quota are capped (and the shortfall reported) unless `allow_upsample`.
    Within a task, sampling is stratified by difficulty to preserve the natural difficulty mix."""
    rng = random.Random(seed)
    by_task: dict[str, list[Example]] = defaultdict(list)
    for ex in pool:
        by_task[ex.task_type].append(ex)
    norm = sum(w for t, w in weights.items() if w > 0)
    chosen: list[Example] = []
    report: dict[str, dict[str, int]] = {}
    for task, w in weights.items():
        if w <= 0:
            continue
        quota = round(total * w / norm)
        avail = by_task.get(task, [])
        strata: dict[str, list[Example]] = defaultdict(list)
        for ex in avail:
            strata[ex.difficulty].append(ex)
        for s in strata.values():
            rng.shuffle(s)
        picked: list[Example] = []
        order = sorted(strata, key=lambda k: -len(strata[k]))
        i = 0
        while len(picked) < min(quota, len(avail)):  # round-robin by difficulty weighted by stratum size
            for d in order:
                take = max(1, round(len(strata[d]) / max(1, len(avail)) * 10))
                for _ in range(take):
                    if strata[d] and len(picked) < min(quota, len(avail)):
                        picked.append(strata[d].pop())
            i += 1
            if i > quota + 10:
                break
        if allow_upsample and avail and len(picked) < quota:
            picked += [rng.choice(avail) for _ in range(quota - len(picked))]
        report[task] = {"requested": quota, "available": len(avail), "used": len(picked), "short": max(0, quota - len(picked))}
        chosen.extend(picked)
    rng.shuffle(chosen)
    counts = Counter(e.task_type for e in chosen)
    n = max(1, len(chosen))
    return chosen, {"total": len(chosen), "realised": {t: round(c / n, 3) for t, c in counts.items()},
                    "requested": {t: round(w / norm, 3) for t, w in weights.items() if w > 0}, "per_task": report}
