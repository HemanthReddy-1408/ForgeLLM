"""AdamW from first principles (decoupled weight decay, bias-corrected moments) plus parameter-group construction
(no decay on norms/biases/adapters, LoRA+ learning-rate ratio for the B matrices)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from forgellm.config import TrainConfig


class AdamW(torch.optim.Optimizer):
    """θ ← θ(1 − lr·λ) − lr · m̂ / (√v̂ + ε),   m = β1 m + (1−β1) g,   v = β2 v + (1−β2) g²,   m̂ = m/(1−β1ᵗ), v̂ = v/(1−β2ᵗ)"""

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p, dtype=torch.float32)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                grad = p.grad.float()
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])        # decoupled: not inside the adaptive term
                st["m"].mul_(b1).add_(grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                m_hat = st["m"] / (1 - b1**t)
                denom = (st["v"] / (1 - b2**t)).sqrt_().add_(g["eps"])
                p.addcdiv_(m_hat.to(p.dtype), denom.to(p.dtype), value=-g["lr"])


def build_param_groups(model: nn.Module, cfg: TrainConfig, lora_plus_ratio: float = 1.0) -> list[dict[str, Any]]:
    """Groups: {decay | no-decay} x {base lr | B-matrix lr (LoRA+)}. Only trainable parameters are included."""
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_b = name.endswith(".B") and lora_plus_ratio != 1.0
        nodecay = p.ndim < 2 or name.endswith("bias") or "norm" in name or ".adapters." in name
        buckets.setdefault((nodecay, is_b), []).append(p)
    groups = []
    for (nodecay, is_b), params in sorted(buckets.items()):
        groups.append({"params": params, "weight_decay": 0.0 if nodecay else cfg.weight_decay,
                       "lr": cfg.learning_rate * (lora_plus_ratio if is_b else 1.0), "lr_mult": lora_plus_ratio if is_b else 1.0})
    if not groups:
        raise ValueError("no trainable parameters — did you inject/unfreeze an adapter?")
    return groups


def build_optimizer(model: nn.Module, cfg: TrainConfig, lora_plus_ratio: float = 1.0, impl: str = "forge") -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg, lora_plus_ratio)
    if impl == "torch":
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas)
    return AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas)


def global_grad_norm(params: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.float().pow(2).sum())
    return math.sqrt(total)
