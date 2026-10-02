"""BaseModelLoader: one place that knows how to turn a ModelConfig into (model, tokenizer, spec) for any backbone —
a Hugging Face decoder (Qwen / Llama / Mistral / Gemma) or the from-scratch TinyGPT — including dtype, device,
quantization and context length. Everything downstream only sees the `LoadedModel` it returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from forgellm.config import ARTIFACTS, ModelConfig
from forgellm.models.quantization import (
    QuantConfig,
    QuantReport,
    bits_per_param,
    memory_bytes,
    quantize_model,
)
from forgellm.models.tinygpt import TinyGPT, TinyGPTConfig, load_tiny
from forgellm.models.tokenizer import ByteTokenizer, HFTokenizer
from forgellm.utils import pick_device, to_dtype


@dataclass
class ModelSpec:
    name: str
    family: str
    num_params: int
    hidden_size: int
    num_layers: int
    context_length: int
    dtype: str
    device: str
    quantization: str
    weight_bytes: int
    quant_report: QuantReport | None = None

    def summary(self) -> str:
        q = f", {self.quantization}" if self.quantization != "none" else ""
        return (f"{self.name} [{self.family}] {self.num_params / 1e6:.1f}M params, {self.num_layers}L x {self.hidden_size}d, "
                f"ctx {self.context_length}, {self.dtype}{q} on {self.device}, weights {self.weight_bytes / 2**20:.0f} MiB")


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: Any
    spec: ModelSpec
    device: torch.device

    @property
    def is_tiny(self) -> bool:
        return self.spec.family == "tinygpt"


def tiny_path(name: str) -> Path:
    return Path(name.split(":", 1)[1]) if ":" in name else ARTIFACTS / "tiny" / "base.pt"


class BaseModelLoader:
    """Usage: `BaseModelLoader(cfg.model).load()`. `name="tiny"` / `"tiny:<ckpt>"` -> TinyGPT, anything else -> HF id."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def load(self) -> LoadedModel:
        c = self.cfg
        device = pick_device(c.device)
        dtype = to_dtype(c.dtype)
        if c.name.startswith("tiny"):
            model, tok, family, ctx = self._load_tiny(device, dtype)
        else:
            model, tok, family, ctx = self._load_hf(device, dtype)
        report = None
        if c.quantization != "none":
            qcfg = QuantConfig(mode=c.quantization, block_size=c.quant_block_size, double_quant=c.double_quant)
            report = quantize_model(model, qcfg)
        model.to(device)
        model.requires_grad_(False)
        model.eval()
        if c.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) if family != "tinygpt" \
                else model.gradient_checkpointing_enable()
        cfg_obj = model.config
        spec = ModelSpec(c.name, family, sum(p.numel() for p in model.parameters()) + (
            sum(m.in_features * m.out_features for m in model.modules() if m.__class__.__name__ == "QuantLinear")),
            getattr(cfg_obj, "hidden_size", 0), getattr(cfg_obj, "num_hidden_layers", getattr(cfg_obj, "num_layers", 0)),
            ctx, c.dtype, str(device), c.quantization, memory_bytes(model), report)
        return LoadedModel(model, tok, spec, device)

    def _load_tiny(self, device: torch.device, dtype: torch.dtype):
        path = tiny_path(self.cfg.name)
        if path.exists():
            model = load_tiny(path)
        else:
            model = TinyGPT(TinyGPTConfig())
        model.to(dtype)
        return model, ByteTokenizer(), "tinygpt", model.config.max_position

    def _load_hf(self, device: torch.device, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.name, trust_remote_code=self.cfg.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(self.cfg.name, dtype=dtype, attn_implementation="sdpa",
                                                     trust_remote_code=self.cfg.trust_remote_code)
        model.config.use_cache = True
        ctx = getattr(model.config, "max_position_embeddings", self.cfg.max_seq_len)
        return model, HFTokenizer(tok), getattr(model.config, "model_type", "hf"), ctx


def estimate_weight_memory(num_params: float, dtype_bits: float = 16, quant: QuantConfig | None = None) -> float:
    """GB for the frozen weights — the arithmetic that motivates QLoRA."""
    bpp = bits_per_param(quant) if quant else dtype_bits
    return num_params * bpp / 8 / 1e9

