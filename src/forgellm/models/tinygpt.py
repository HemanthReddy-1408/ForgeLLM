"""TinyGPT: a Llama-style decoder-only transformer written from scratch (RMSNorm, RoPE, grouped-query attention,
SwiGLU, KV cache). Module names (q_proj/k_proj/v_proj/o_proj, gate_proj/up_proj/down_proj, model.layers.N) match
Llama/Qwen so every adaptation tool in this repo works identically on it and on real Hugging Face models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class TinyGPTConfig:
    vocab_size: int = 260
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 688
    max_position: int = 2048
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class CausalOutput:
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    past_key_values: Any = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # normalise in fp32 even under 16-bit autocast
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq  # (B, T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]  # (B, 1, T, D)


class Attention(nn.Module):
    def __init__(self, c: TinyGPTConfig) -> None:
        super().__init__()
        self.h, self.kv, self.d = c.num_heads, c.num_kv_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.kv * self.d, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.kv * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor | None,
                past: tuple[torch.Tensor, torch.Tensor] | None, use_cache: bool):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        new_past = (k, v) if use_cache else None
        if self.kv != self.h:  # grouped-query attention: each KV head serves h/kv query heads
            rep = self.h // self.kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.h * self.d)), new_past


class MLP(nn.Module):
    def __init__(self, c: TinyGPTConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # SwiGLU


class Block(nn.Module):
    def __init__(self, c: TinyGPTConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, mask, past, use_cache):
        a, new_past = self.self_attn(self.input_layernorm(x), cos, sin, mask, past, use_cache)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), new_past


class TinyGPTModel(nn.Module):
    def __init__(self, c: TinyGPTConfig) -> None:
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_eps)
        self.rotary = Rotary(c.head_dim, c.rope_theta)
        self.gradient_checkpointing = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, past_key_values: list | None = None,
                use_cache: bool = False) -> CausalOutput:
        B, T = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        if position_ids is None:
            if attention_mask is not None and attention_mask.shape[1] == past_len + T:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, past_len:]
            else:
                position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)[None].expand(B, T)
        cos, sin = self.rotary(position_ids)
        mask = None
        if attention_mask is not None and past_len == 0 and bool(attention_mask.all()):
            attention_mask = None  # no padding: take the fused causal fast path
        if attention_mask is not None or past_len > 0:
            S = past_len + T
            q_pos = torch.arange(past_len, S, device=input_ids.device)[:, None]
            allow = torch.arange(S, device=input_ids.device)[None, :] <= q_pos  # causal
            allow = allow[None, None]
            if attention_mask is not None:
                keys = attention_mask[:, None, None, :S].bool()
                allow = allow & keys
                allow = allow | (~keys.any(-1, keepdim=True))  # fully padded query rows: avoid NaN softmax
            mask = allow
        x = self.embed_tokens(input_ids)
        new_past = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, p = checkpoint(layer, x, cos, sin, mask, past, use_cache, use_reentrant=False)
            else:
                x, p = layer(x, cos, sin, mask, past, use_cache)
            new_past.append(p)
        return CausalOutput(last_hidden_state=self.norm(x), past_key_values=new_past if use_cache else None)


class TinyGPT(nn.Module):
    def __init__(self, c: TinyGPTConfig) -> None:
        super().__init__()
        self.config = c
        self.model = TinyGPTModel(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2 style residual scaling
            if n.endswith("o_proj.weight") or n.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.num_layers))
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.model.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None, past_key_values: list | None = None,
                use_cache: bool = False, labels: torch.Tensor | None = None, logits_to_keep: int = 0) -> CausalOutput:
        out = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache)
        h = out.last_hidden_state[:, -logits_to_keep:] if logits_to_keep else out.last_hidden_state
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalOutput(logits=logits, loss=loss, last_hidden_state=out.last_hidden_state, past_key_values=out.past_key_values)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_tiny(model: TinyGPT, path: str | Path) -> None:
    from forgellm.utils import atomic_write

    path = Path(path)
    with atomic_write(path, "wb") as fh:
        torch.save({"config": asdict(model.config), "state": model.state_dict()}, fh)


def load_tiny(path: str | Path, map_location: str = "cpu") -> TinyGPT:
    blob = torch.load(path, map_location=map_location, weights_only=True)
    m = TinyGPT(TinyGPTConfig(**blob["config"]))
    m.load_state_dict(blob["state"])
    return m
