"""Typed configuration: YAML -> dataclasses, strict about unknown keys, dotted CLI overrides."""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CONFIGS = ROOT / "configs"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # HF id, or "tiny" for the from-scratch TinyGPT
    dtype: str = "float32"                    # float32 | float16 | bfloat16 (compute/storage dtype of frozen weights)
    device: str = "auto"                      # auto | cuda | mps | cpu
    quantization: str = "none"                # none | nf4 | int8
    quant_block_size: int = 64
    double_quant: bool = True
    max_seq_len: int = 512
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    method: str = "lora"                      # lora | dora | bottleneck | ia3 | full
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    placement: str | None = None              # preset name (qv|qkvo|attn_mlp|mlp|all_linear) overrides target_modules
    layers: str = "all"                       # all | first:N | last:N | a-b
    rank_pattern: dict[str, int] = field(default_factory=dict)  # regex on module path -> rank
    rslora: bool = False                      # alpha / sqrt(r) scaling
    bottleneck_dim: int = 32                  # for method=bottleneck
    lora_plus_ratio: float = 1.0              # LR multiplier for the B matrices (LoRA+)


@dataclass
class TrainConfig:
    task: str = "sft"                         # sft | dpo | orpo | simpo | pretrain
    run_name: str = "run"
    data_path: str = ""
    tasks: list[str] = field(default_factory=list)  # restrict to these task types ([] = all)
    max_examples: int = 0                     # 0 = all
    learning_rate: float = 2e-4
    min_lr_ratio: float = 0.1
    scheduler: str = "cosine"                 # cosine | linear | constant | wsd
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    nan_retries: int = 2                      # recompute a step whose gradients came back non-finite before skipping it
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: float = 1.0
    max_steps: int = 0                        # >0 overrides epochs
    amp: str = "none"                         # none | bf16 | fp16
    eval_every: int = 50
    save_every: int = 100
    keep_last: int = 2
    log_every: int = 5
    seed: int = 42
    num_workers: int = 0
    length_grouped: bool = True
    early_stopping_patience: int = 0
    # preference optimisation
    beta: float = 0.1
    label_smoothing: float = 0.0
    sft_weight: float = 0.0                   # add NLL on chosen (RPO-style) to DPO
    simpo_gamma: float = 1.0
    orpo_lambda: float = 0.1


@dataclass
class InferenceConfig:
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    router_threshold: float = 0.55
    rag_top_k: int = 3
    max_adapters_resident: int = 4
    adapters_dir: str = ""


@dataclass
class DataConfig:
    seed: int = 7
    per_task: dict[str, int] = field(default_factory=lambda: {
        "technical_qa": 3000, "reasoning": 3500, "coding": 3000,
        "extraction": 2800, "tool_use": 3500, "grounded_qa": 2000, "safety": 400,
    })
    defect_rate: float = 0.08                 # injected raw-data defects (dups, junk, malformed JSON ...)
    leak_rate: float = 0.02                   # fraction of eval items "scraped" back into the raw pool
    min_quality: float = 0.6
    near_dup_threshold: float = 0.85
    contamination_ngram: int = 8
    contamination_threshold: float = 0.9
    val_frac: float = 0.05
    test_frac: float = 0.08
    mixture: dict[str, float] = field(default_factory=lambda: {
        "technical_qa": 0.22, "reasoning": 0.22, "coding": 0.14, "extraction": 0.14,
        "tool_use": 0.14, "grounded_qa": 0.06, "safety": 0.05, "general": 0.03,
    })


@dataclass
class ForgeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dumps(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _coerce(tp: Any, value: Any) -> Any:
    origin = typing.get_origin(tp)
    if tp is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        return float(value)          # YAML 1.1 reads "5e-5" as a string; numeric fields are coerced here
    if tp == tuple[float, float] or origin is tuple:
        return tuple(value)
    return value


def _fill(cls: type, data: dict[str, Any], path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown config key(s) under '{path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        tp = hints[k]
        if dataclasses.is_dataclass(tp):
            kwargs[k] = _fill(tp, v or {}, f"{path}.{k}" if path else k)
        else:
            kwargs[k] = _coerce(tp, v)
    return cls(**kwargs)


def load_config(*paths: str | Path, overrides: list[str] | None = None) -> ForgeConfig:
    """Merge YAML files left-to-right, apply `section.key=value` overrides, build a ForgeConfig."""
    merged: dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.exists() and (CONFIGS / p).exists():
            p = CONFIGS / p
        with open(p) as fh:
            _deep_update(merged, yaml.safe_load(fh) or {})
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"override must look like section.key=value, got {ov!r}")
        node = merged
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return _fill(ForgeConfig, merged)


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def config_hash(cfg: ForgeConfig | dict) -> str:
    import hashlib

    d = cfg.to_dict() if isinstance(cfg, ForgeConfig) else cfg
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]
