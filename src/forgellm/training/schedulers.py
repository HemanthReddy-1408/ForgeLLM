"""Learning-rate schedules as pure functions of the step (so they are trivially testable and resume exactly)."""

from __future__ import annotations

import math

import torch


def lr_factor(step: int, total: int, warmup: int, kind: str = "cosine", min_ratio: float = 0.1,
              decay_frac: float = 0.2) -> float:
    """Multiplier in [min_ratio, 1] applied to the peak LR at optimizer step `step` (0-based)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup                                   # linear warmup from ~0 to peak
    if kind == "constant":
        return 1.0
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    if kind == "cosine":
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    if kind == "linear":
        return 1.0 - (1 - min_ratio) * prog
    if kind == "wsd":                                                # warmup - stable - decay
        start = 1.0 - decay_frac
        if prog < start:
            return 1.0
        return 1.0 - (1 - min_ratio) * (prog - start) / max(1e-9, decay_frac)
    raise ValueError(f"unknown scheduler {kind!r}")


class Scheduler:
    def __init__(self, opt: torch.optim.Optimizer, total: int, warmup: int, kind: str, min_ratio: float) -> None:
        self.opt, self.total, self.warmup, self.kind, self.min_ratio = opt, total, warmup, kind, min_ratio
        self.base = [g["lr"] for g in opt.param_groups]
        self.step_n = 0
        self.apply(0)

    def apply(self, step: int) -> float:
        f = lr_factor(step, self.total, self.warmup, self.kind, self.min_ratio)
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * f
        return self.opt.param_groups[0]["lr"]

    def step(self) -> float:
        self.step_n += 1
        return self.apply(self.step_n)

    def seek(self, step: int) -> None:
        self.step_n = step
        self.apply(step)
