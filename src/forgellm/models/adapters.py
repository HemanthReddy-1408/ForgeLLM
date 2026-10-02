"""Adapters as first-class objects: other PEFT families (bottleneck, IA3), one injection entry point for every method,
safetensors persistence with a manifest, merge/unload, adapter arithmetic (SVD fusion) and a memory-bounded store.

    artifacts/adapters/<name>/adapter.safetensors     tensors (only the adapter, MBs not GBs)
    artifacts/adapters/<name>/adapter_config.json     method, targets, ranks, base-model id, data hash, metrics
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from forgellm.config import ARTIFACTS, LoRAConfig
from forgellm.models.lora import (
    Adapter,
    AdapterLayer,
    LoRAAdapter,
    ModuleInfo,
    _parent,
    adapter_layers,
    count_parameters,
    inject_lora,
    loaded_adapters,
    resolve_targets,
    set_active,
)
from forgellm.models.quantization import QuantLinear, dequantize_module


def adapters_dir() -> Path:
    return ARTIFACTS / "adapters"


class BottleneckAdapter(Adapter):
    """Houlsby-style serial adapter: out + up(act(down(out))), `up` initialised to zero (identity at the start)."""
    kind = "bottleneck"

    def __init__(self, dim: int, bottleneck: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.down = nn.Linear(dim, bottleneck, dtype=dtype)
        self.up = nn.Linear(bottleneck, dim, dtype=dtype)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def post(self, layer: AdapterLayer, x: torch.Tensor, out: torch.Tensor, weight: float) -> torch.Tensor:
        h = self.up(F.gelu(self.down(out.to(self.down.weight.dtype))))
        return out + weight * h.to(out.dtype)

    def config(self) -> dict[str, Any]:
        return {"kind": self.kind, "bottleneck": self.bottleneck}


class IA3Adapter(Adapter):
    """(IA)^3: learn a vector that rescales activations — the cheapest PEFT family (one number per feature)."""
    kind = "ia3"

    def __init__(self, dim: int, on_input: bool, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.on_input = on_input
        self.vec = nn.Parameter(torch.ones(dim, dtype=dtype))

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.vec.to(x.dtype) if self.on_input else x

    def post(self, layer: AdapterLayer, x: torch.Tensor, out: torch.Tensor, weight: float) -> torch.Tensor:
        return out if self.on_input else out * self.vec.to(out.dtype)

    def config(self) -> dict[str, Any]:
        return {"kind": self.kind, "on_input": self.on_input}


DEFAULT_TARGETS = {"bottleneck": ["o_proj", "down_proj"], "ia3": ["k_proj", "v_proj", "down_proj"]}


def inject_adapter(model: nn.Module, name: str, cfg: LoRAConfig, dtype: torch.dtype = torch.float32) -> list[ModuleInfo]:
    """One entry point for every PEFT method. `full` injects nothing (all weights train)."""
    m = cfg.method
    if m in ("lora", "dora"):
        return inject_lora(model, name, cfg, dtype)
    if m == "full":
        return []
    names = None if (cfg.placement or cfg.target_modules != ["q_proj", "v_proj"]) else DEFAULT_TARGETS[m]
    targets = resolve_targets(model, cfg, names)
    for info in targets:
        parent, leaf = _parent(model, info.path)
        layer = getattr(parent, leaf)
        if not isinstance(layer, AdapterLayer):
            layer = AdapterLayer(layer)
            setattr(parent, leaf, layer)
        base = layer.base
        dev = base.weight.device if isinstance(base, nn.Linear) else base.qweight.device
        if m == "bottleneck":
            ad: Adapter = BottleneckAdapter(base.out_features, cfg.bottleneck_dim, dtype).to(dev)
        elif m == "ia3":
            ad = IA3Adapter(base.in_features if info.leaf == "down_proj" else base.out_features, info.leaf == "down_proj", dtype).to(dev)
        else:
            raise ValueError(f"unknown PEFT method {m!r}")
        layer.add_adapter(name, ad)
        layer.active[name] = 1.0
    return targets


def freeze_base(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def adapter_parameters(model: nn.Module, name: str) -> list[tuple[str, nn.Parameter]]:
    out = []
    for path, layer in adapter_layers(model):
        if name in layer.adapters:
            out += [(f"{path}.adapters.{name}.{n}", p) for n, p in layer.adapters[name].named_parameters()]
    return out


def prepare_for_training(model: nn.Module, trainable: list[str] | None) -> tuple[int, int]:
    """Freeze everything, then unfreeze only the named adapters (or everything if `trainable` is None => full FT)."""
    if trainable is None:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        freeze_base(model)
        for n in trainable:
            for _, p in adapter_parameters(model, n):
                p.requires_grad_(True)
    return count_parameters(model)


# ------------------------------------------------------------------------------------------------------
# persistence
# ------------------------------------------------------------------------------------------------------
def adapter_state(model: nn.Module, name: str) -> dict[str, torch.Tensor]:
    return {k: p.detach().cpu().contiguous() for k, p in adapter_parameters(model, name)}


def save_adapter(model: nn.Module, name: str, out_dir: str | Path | None, cfg: LoRAConfig, meta: dict[str, Any] | None = None) -> Path:
    out = Path(out_dir) if out_dir else adapters_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    state = adapter_state(model, name)
    targets, ranks, kinds = [], {}, set()
    for path, layer in adapter_layers(model):
        if name in layer.adapters:
            targets.append(path)
            ad = layer.adapters[name]
            kinds.add(ad.kind)
            if isinstance(ad, LoRAAdapter):
                ranks[path] = ad.r
    n_params = sum(t.numel() for t in state.values())
    manifest = {"name": name, "method": cfg.method, "lora": cfg.__dict__ | {"target_modules": list(cfg.target_modules)},
                "targets": targets, "ranks": ranks, "num_parameters": n_params,
                "size_mb": round(sum(t.numel() * t.element_size() for t in state.values()) / 2**20, 3),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
    save_file(state, str(out / "adapter.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def read_adapter_config(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "adapter_config.json").read_text())


def load_adapter(model: nn.Module, path: str | Path, name: str | None = None, dtype: torch.dtype = torch.float32,
                 activate: bool = False) -> str:
    """Rebuild the adapter structure from its saved config, then load its tensors into the (shared, frozen) base."""
    path = Path(path)
    man = read_adapter_config(path)
    name = name or man["name"]
    if name not in loaded_adapters(model):
        lcfg = LoRAConfig(**{k: v for k, v in man["lora"].items() if k in LoRAConfig.__dataclass_fields__})
        paths = set(man["targets"])
        leaves = sorted({t.rsplit(".", 1)[-1] for t in paths})
        for info in resolve_targets(model, LoRAConfig(method=lcfg.method, target_modules=leaves), leaves):
            if info.path in paths:
                sub = LoRAConfig(**{**lcfg.__dict__, "rank": man["ranks"].get(info.path, lcfg.rank)})
                _inject_one(model, name, sub, info, dtype)
    state = load_file(str(path / "adapter.safetensors"))
    own = dict(adapter_parameters(model, name))
    missing = set(own) - set(state)
    extra = set(state) - set(own)
    if missing or extra:
        raise ValueError(f"adapter '{name}' does not fit this model: missing={sorted(missing)[:3]} unexpected={sorted(extra)[:3]}")
    with torch.no_grad():
        for k, p in own.items():
            p.copy_(state[k].to(p.device, p.dtype))
    for p in own.values():
        p.requires_grad_(False)
    if activate:
        set_active(model, {name: 1.0})
    return name


def _inject_one(model: nn.Module, name: str, cfg: LoRAConfig, info: ModuleInfo, dtype: torch.dtype) -> None:
    parent, leaf = _parent(model, info.path)
    layer = getattr(parent, leaf)
    if not isinstance(layer, AdapterLayer):
        layer = AdapterLayer(layer)
        setattr(parent, leaf, layer)
    base = layer.base
    dev = base.weight.device if isinstance(base, nn.Linear) else base.qweight.device
    if cfg.method in ("lora", "dora"):
        ad: Adapter = LoRAAdapter(base.in_features, base.out_features, cfg.rank, cfg.alpha, cfg.dropout, cfg.rslora,
                                  base.weight if cfg.method == "dora" else None, dtype).to(dev)
    elif cfg.method == "bottleneck":
        ad = BottleneckAdapter(base.out_features, cfg.bottleneck_dim, dtype).to(dev)
    else:
        ad = IA3Adapter(base.in_features if info.leaf == "down_proj" else base.out_features, info.leaf == "down_proj", dtype).to(dev)
    layer.add_adapter(name, ad)


def unload_adapter(model: nn.Module, name: str) -> None:
    """Remove an adapter everywhere; layers left with no adapters are unwrapped back to the bare base layer."""
    for path, layer in list(adapter_layers(model)):
        if name in layer.adapters:
            layer.remove_adapter(name)
        if not layer.adapters:
            parent, leaf = _parent(model, path)
            setattr(parent, leaf, layer.base)


# ------------------------------------------------------------------------------------------------------
# merge, fuse
# ------------------------------------------------------------------------------------------------------
@torch.no_grad()
def merge_and_unload(model: nn.Module, weights: dict[str, float]) -> nn.Module:
    """Fold LoRA/DoRA deltas into the base weights and delete the adapter layers: a standalone model with zero
    inference overhead. Quantized layers are dequantized first (merging into packed codes would lose the delta)."""
    for path, layer in list(adapter_layers(model)):
        base = layer.base
        if isinstance(base, QuantLinear):
            base = dequantize_module(base, torch.float32)
            layer.base = base
        for n, w in weights.items():
            if n in layer.adapters:
                layer.merge(n, w)
        parent, leaf = _parent(model, path)
        setattr(parent, leaf, base)
    return model


@torch.no_grad()
def fuse_adapters(model: nn.Module, weights: dict[str, float], out_name: str, rank: int | None = None) -> dict[str, float]:
    """Adapter arithmetic: ΔW_fused = Σ w_i ΔW_i, re-factorised to rank r by truncated SVD into a new LoRA adapter.
    Returns the mean relative reconstruction error per layer (how much the rank cap cost)."""
    errs: dict[str, float] = {}
    for path, layer in adapter_layers(model):
        parts = [(layer.adapters[n], w) for n, w in weights.items() if n in layer.adapters and isinstance(layer.adapters[n], LoRAAdapter)]
        if not parts:
            continue
        dev = parts[0][0].A.device
        delta = sum(w * ad.delta_weight().float() for ad, w in parts)
        r = rank or max(ad.r for ad, _ in parts)
        U, S, Vh = torch.linalg.svd(delta.cpu(), full_matrices=False)
        r = min(r, S.numel())
        new = LoRAAdapter(layer.in_features, layer.out_features, r, alpha=float(r), dropout=0.0, dtype=torch.float32).to(dev)
        sq = S[:r].sqrt()
        new.B.copy_((U[:, :r] * sq).to(dev))
        new.A.copy_((sq[:, None] * Vh[:r]).to(dev))
        layer.add_adapter(out_name, new)
        approx = new.delta_weight().cpu()
        errs[path] = float((delta.cpu() - approx).norm() / delta.cpu().norm().clamp_min(1e-12))
    return errs


# ------------------------------------------------------------------------------------------------------
# store
# ------------------------------------------------------------------------------------------------------
class AdapterStore:
    """Hot-swappable adapters on ONE shared base model, with a residency budget (LRU eviction)."""

    def __init__(self, model: nn.Module, root: str | Path | None = None, max_resident: int = 4) -> None:
        self.model = model
        self.root = Path(root) if root else adapters_dir()
        self.max_resident = max_resident
        self.resident: OrderedDict[str, float] = OrderedDict()   # name -> last used
        self.load_ms: dict[str, float] = {}
        self.evictions = 0

    def available(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/adapter_config.json")) if self.root.exists() else []

    def ensure(self, name: str, protect: tuple[str, ...] = ()) -> None:
        if name in self.resident:
            self.resident.move_to_end(name)
            return
        path = self.root / name
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter '{name}' not found under {self.root}")
        while len(self.resident) >= self.max_resident:
            victim = next((n for n in self.resident if n not in protect), None)
            if victim is None:
                raise RuntimeError(f"adapter budget ({self.max_resident}) too small for the requested stack")
            self.resident.pop(victim)
            unload_adapter(self.model, victim)
            self.evictions += 1
        t0 = time.perf_counter()
        load_adapter(self.model, path, name)
        self.load_ms[name] = (time.perf_counter() - t0) * 1000
        self.resident[name] = time.time()

    def activate(self, name: str | None, weight: float = 1.0) -> None:
        if name is None:
            set_active(self.model, {})
            return
        self.ensure(name)
        set_active(self.model, {name: weight})

    def chain(self, name: str) -> list[str]:
        """Adapters to apply for `name`, base-first (a DPO adapter stacks on the SFT adapter it was trained on)."""
        out, cur = [], name
        while cur and cur not in out:
            out.insert(0, cur)
            cur = self.info(cur).get("stack_on")
        return out

    def activate_chain(self, name: str) -> None:
        ch = self.chain(name)
        for n in ch:
            self.ensure(n, protect=tuple(ch))
        set_active(self.model, dict.fromkeys(ch, 1.0))

    def info(self, name: str) -> dict[str, Any]:
        return read_adapter_config(self.root / name)
