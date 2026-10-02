"""LoRA / DoRA / rsLoRA implemented explicitly: module discovery, placement presets, a multi-adapter wrapper layer,
merge/unmerge, and parameter accounting.

    y = W x + (alpha/r) · B (A x)           ΔW = (alpha/r) · B A,   A ∈ R^{r×in},  B ∈ R^{out×r}  (B starts at 0)

One `AdapterLayer` can hold *several named adapters* on the same frozen base weight and apply any weighted subset of
them — that is what makes per-request adapter routing and adapter arithmetic possible on one shared base model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from forgellm.config import LoRAConfig
from forgellm.models.quantization import QuantLinear

ROLES = {"q_proj": "attn.q", "k_proj": "attn.k", "v_proj": "attn.v", "o_proj": "attn.o",
         "gate_proj": "mlp.gate", "up_proj": "mlp.up", "down_proj": "mlp.down", "lm_head": "lm_head"}
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
PLACEMENTS: dict[str, list[str]] = {
    "qv": ["q_proj", "v_proj"], "qkvo": ATTN, "attn": ATTN, "mlp": MLP, "attn_mlp": ATTN + MLP, "all_linear": ATTN + MLP,
}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass
class ModuleInfo:
    path: str
    role: str
    layer: int | None
    in_features: int
    out_features: int
    quantized: bool

    @property
    def leaf(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    def lora_params(self, r: int) -> int:
        return r * (self.in_features + self.out_features)


def is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, QuantLinear))


def find_candidate_modules(model: nn.Module) -> list[ModuleInfo]:
    """Every linear projection that an adapter could attach to (looks through already-installed adapter layers)."""
    out = []
    for path, m in model.named_modules():
        base = m.base if isinstance(m, AdapterLayer) else m
        if "." in path and isinstance(model.get_submodule(path.rsplit(".", 1)[0]), AdapterLayer):
            continue  # skip the inner .base of an AdapterLayer
        if not is_linear_like(base):
            continue
        leaf = path.rsplit(".", 1)[-1]
        lm = _LAYER_RE.search(path)
        out.append(ModuleInfo(path, ROLES.get(leaf, "other"), int(lm.group(1)) if lm else None,
                              base.in_features, base.out_features, isinstance(base, QuantLinear)))
    return out


def _layer_filter(spec: str, n_layers: int):
    spec = (spec or "all").strip()
    if spec == "all":
        return lambda i: True
    if spec.startswith("first:"):
        k = int(spec[6:])
        return lambda i: i < k
    if spec.startswith("last:"):
        k = int(spec[5:])
        return lambda i: i >= n_layers - k
    if re.fullmatch(r"\d+-\d+", spec):
        a, b = map(int, spec.split("-"))
        return lambda i: a <= i <= b
    raise ValueError(f"bad layers spec {spec!r} (use all | first:N | last:N | a-b)")


def target_names(cfg: LoRAConfig) -> list[str]:
    if cfg.placement:
        if cfg.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {cfg.placement!r}; choose from {sorted(PLACEMENTS)}")
        return PLACEMENTS[cfg.placement]
    return list(cfg.target_modules)


def resolve_targets(model: nn.Module, cfg: LoRAConfig, names: list[str] | None = None) -> list[ModuleInfo]:
    names = names or target_names(cfg)
    cands = find_candidate_modules(model)
    layers = [c.layer for c in cands if c.layer is not None]
    n_layers = (max(layers) + 1) if layers else 0
    keep = _layer_filter(cfg.layers, n_layers)
    sel = [c for c in cands if c.leaf in names and (c.layer is None or keep(c.layer))]
    if not sel:
        raise ValueError(f"no modules matched targets {names} (layers={cfg.layers}); candidates: "
                         f"{sorted({c.leaf for c in cands})}")
    return sel


def rank_for(info: ModuleInfo, cfg: LoRAConfig) -> int:
    for pat, r in cfg.rank_pattern.items():
        if re.search(pat, info.path):
            return r
    return cfg.rank


def describe_candidates(model: nn.Module, cfg: LoRAConfig | None = None) -> str:
    """The attention / MLP hierarchy with shapes, adapter parameter cost at the configured rank and whether selected."""
    cands = find_candidate_modules(model)
    chosen = {c.path for c in resolve_targets(model, cfg)} if cfg else set()
    by_leaf: dict[str, list[ModuleInfo]] = {}
    for c in cands:
        by_leaf.setdefault(c.leaf, []).append(c)
    r = cfg.rank if cfg else 16
    lines = [f"candidate modules ({len(cands)} linear layers); LoRA params shown at rank {r}"]
    for group, leaves in (("Attention", ATTN), ("MLP", MLP)):
        lines.append(f" {group}")
        for lf in leaves:
            ms = by_leaf.get(lf, [])
            if not ms:
                continue
            mark = "*" if any(m.path in chosen for m in ms) else " "
            lines.append(f"  {mark} {lf:<10} x{len(ms):<3} {ms[0].in_features}->{ms[0].out_features}"
                         f"  {ms[0].lora_params(r):>9,} params/layer{'  [quantized]' if ms[0].quantized else ''}")
    if cfg:
        total = sum(c.lora_params(rank_for(c, cfg)) for c in cands if c.path in chosen)
        lines.append(f" selected: {len(chosen)} modules -> {total:,} adapter parameters (* = adapted)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------
# adapters
# ------------------------------------------------------------------------------------------------------
class Adapter(nn.Module):
    """Interface: optionally rewrite the layer input, then rewrite the layer output."""
    kind = "adapter"

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def post(self, layer: AdapterLayer, x: torch.Tensor, out: torch.Tensor, weight: float) -> torch.Tensor:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        return {"kind": self.kind}


class LoRAAdapter(Adapter):
    kind = "lora"

    def __init__(self, in_f: int, out_f: int, r: int, alpha: float, dropout: float, rslora: bool = False,
                 dora_base: torch.Tensor | None = None, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.r, self.alpha, self.rslora = r, alpha, rslora
        self.scale = alpha / math.sqrt(r) if rslora else alpha / r
        self.A = nn.Parameter(torch.empty(r, in_f, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_f, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.magnitude: nn.Parameter | None = None
        if dora_base is not None:  # DoRA: learn a per-output-row magnitude, LoRA moves only the direction
            self.magnitude = nn.Parameter(dora_base.detach().float().norm(dim=1).to(dtype))

    @property
    def is_dora(self) -> bool:
        return self.magnitude is not None

    def delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        # compute in the activation dtype (bf16 on a bf16 base) with fp32 master weights: no fp32 copy of x is kept alive
        # for backward, which is what dominates activation memory when 100+ modules carry adapters
        xd = self.drop(x)
        return F.linear(F.linear(xd, self.A.to(xd.dtype)), self.B.to(xd.dtype)) * self.scale

    def post(self, layer: AdapterLayer, x: torch.Tensor, out: torch.Tensor, weight: float) -> torch.Tensor:
        if self.is_dora:
            w = layer.base.weight
            norm = (w.float() + self.delta_weight().float()).norm(dim=1).detach()  # norm treated as constant in backward
            mag = (self.magnitude.float() / norm).to(out.dtype)
            bias = layer.base.bias
            core = out if bias is None else out - bias.to(out.dtype)
            res = mag * (core + self.delta(x).to(out.dtype))
            return res if bias is None else res + bias.to(out.dtype)
        return out + weight * self.delta(x).to(out.dtype)

    def config(self) -> dict[str, Any]:
        return {"kind": "dora" if self.is_dora else "lora", "r": self.r, "alpha": self.alpha, "rslora": self.rslora}


class AdapterLayer(nn.Module):
    """Wraps one frozen linear layer; holds any number of named adapters and applies the active ones."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active: dict[str, float] = {}
        self.enabled = True
        self.merged: dict[str, float] = {}

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def add_adapter(self, name: str, adapter: Adapter) -> None:
        if name in self.adapters:
            raise ValueError(f"adapter '{name}' already exists on this layer")
        adapter.train(self.training)  # a new adapter inherits the model's train/eval mode (dropout!)
        self.adapters[name] = adapter

    def remove_adapter(self, name: str) -> None:
        if name in self.merged:
            self.unmerge(name)
        self.active.pop(name, None)
        del self.adapters[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        names = [n for n in self.active if n in self.adapters and n not in self.merged] if self.enabled else []
        if not names:
            return self.base(x)
        xin = x
        for n in names:
            xin = self.adapters[n].pre(xin)
        out = self.base(xin)
        for n in names:
            out = self.adapters[n].post(self, x, out, self.active[n])
        return out

    # -- merge / unmerge (plain nn.Linear only: merging into packed integer codes would destroy the quantization) ----
    @torch.no_grad()
    def merge(self, name: str, weight: float = 1.0) -> None:
        ad = self.adapters[name]
        if not isinstance(ad, LoRAAdapter):
            raise TypeError("only LoRA/DoRA adapters can be merged into the weight")
        if not isinstance(self.base, nn.Linear):
            raise TypeError("cannot merge into a quantized layer; dequantize first (merge_and_unload does this)")
        w = self.base.weight
        new = w.float() + weight * ad.delta_weight().float()
        if ad.is_dora:
            new = new * (ad.magnitude.float() / new.norm(dim=1))[:, None]
        self._backup = getattr(self, "_backup", {})
        self._backup[name] = w.detach().clone()
        w.copy_(new.to(w.dtype))
        self.merged[name] = weight

    @torch.no_grad()
    def unmerge(self, name: str) -> None:
        self.base.weight.copy_(self._backup.pop(name))
        self.merged.pop(name)


def _parent(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    head, _, leaf = path.rpartition(".")
    return (model.get_submodule(head) if head else model), leaf


def adapter_layers(model: nn.Module) -> Iterator[tuple[str, AdapterLayer]]:
    for n, m in model.named_modules():
        if isinstance(m, AdapterLayer):
            yield n, m


def inject_lora(model: nn.Module, name: str, cfg: LoRAConfig, dtype: torch.dtype = torch.float32) -> list[ModuleInfo]:
    """Attach a named LoRA/DoRA adapter to every module selected by the config's placement. Returns the modules."""
    dora = cfg.method == "dora"
    targets = resolve_targets(model, cfg)
    for info in targets:
        parent, leaf = _parent(model, info.path)
        layer = getattr(parent, leaf)
        if not isinstance(layer, AdapterLayer):
            layer = AdapterLayer(layer)
            setattr(parent, leaf, layer)
        base = layer.base
        if dora and not isinstance(base, nn.Linear):
            raise TypeError("DoRA needs an unquantized base layer")
        dev = (base.weight.device if isinstance(base, nn.Linear) else base.qweight.device)
        ad = LoRAAdapter(base.in_features, base.out_features, rank_for(info, cfg), cfg.alpha, cfg.dropout,
                         cfg.rslora, base.weight if dora else None, dtype).to(dev)
        layer.add_adapter(name, ad)
        layer.active[name] = 1.0
    return targets


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) — total counts quantized layers at their *logical* weight count."""
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += m.in_features * m.out_features
    return train, total


@contextmanager
def adapters_disabled(model: nn.Module):
    """Run the bare base model (used as the frozen reference policy in DPO)."""
    layers = [m for _, m in adapter_layers(model)]
    prev = [m.enabled for m in layers]
    for m in layers:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(layers, prev):
            m.enabled = p


@contextmanager
def adapters_active(model: nn.Module, weights: dict[str, float]):
    layers = [m for _, m in adapter_layers(model)]
    prev = [dict(m.active) for m in layers]
    set_active(model, weights)
    try:
        yield
    finally:
        for m, p in zip(layers, prev):
            m.active = p


def set_active(model: nn.Module, weights: dict[str, float] | None) -> None:
    for _, m in adapter_layers(model):
        m.active = {k: float(v) for k, v in (weights or {}).items() if k in m.adapters}


def loaded_adapters(model: nn.Module) -> list[str]:
    names: dict[str, None] = {}
    for _, m in adapter_layers(model):
        for k in m.adapters:
            names[k] = None
    return list(names)
