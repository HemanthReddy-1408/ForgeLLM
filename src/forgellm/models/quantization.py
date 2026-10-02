"""Weight quantization written from scratch in PyTorch (works on CPU, CUDA and Apple MPS — bitsandbytes does not).

    nf4      4-bit NormalFloat, blockwise absmax, optional *double quantization* of the scales  (QLoRA)
    uniform4 4-bit uniform levels, same blockwise machinery (baseline that shows why NF4 exists)
    int8     per-output-channel symmetric int8

A `QuantLinear` stores packed integer codes and dequantizes on the fly; its backward pass recomputes the
dequantized weight instead of saving it, so the full-precision matrix is never kept alive between forward and backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


def nf4_codebook() -> torch.Tensor:
    """The 16 NormalFloat levels: quantiles of N(0,1) (asymmetric so that 0 is exactly representable), scaled to [-1, 1]."""
    n = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    offset = 0.9677083
    pos = n.icdf(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])        # 8 positive levels
    neg = -n.icdf(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])       # 7 negative levels
    v = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (v / v.abs().max()).float()


def uniform4_codebook() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 16)


CODEBOOKS = {"nf4": nf4_codebook, "uniform4": uniform4_codebook}


@dataclass
class QuantConfig:
    mode: str = "nf4"            # nf4 | uniform4 | int8
    block_size: int = 64
    double_quant: bool = True
    scale_block: int = 256       # block size for the 8-bit quantization of the scales
    skip: tuple[str, ...] = ("lm_head", "embed_tokens")


def _quant_4bit(w: torch.Tensor, code: torch.Tensor, block: int):
    flat = w.detach().float().flatten()
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(1).clamp_min(1e-12)
    norm = blocks / absmax[:, None]
    mids = (code[1:] + code[:-1]) / 2
    idx = torch.bucketize(norm.flatten(), mids).to(torch.uint8)  # nearest code via midpoints
    packed = (idx[0::2] << 4) | idx[1::2]                        # two 4-bit codes per byte
    return packed, absmax


def _dequant_4bit(packed: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, block: int, shape: tuple[int, ...],
                  dtype: torch.dtype) -> torch.Tensor:
    idx = torch.stack([packed >> 4, packed & 0x0F], dim=-1).flatten().long()
    vals = code[idx].view(-1, block) * absmax[:, None]
    n = 1
    for s in shape:
        n *= s
    return vals.flatten()[:n].view(shape).to(dtype)


def _double_quant(absmax: torch.Tensor, block: int):
    mean = absmax.mean()
    c = absmax - mean
    pad = (-c.numel()) % block
    if pad:
        c = F.pad(c, (0, pad))
    cb = c.view(-1, block)
    scale = (cb.abs().amax(1) / 127.0).clamp_min(1e-12)
    q = torch.round(cb / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.flatten(), scale, mean


def _double_dequant(q: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor, block: int, n: int) -> torch.Tensor:
    return (q.view(-1, block).float() * scale[:, None]).flatten()[:n] + mean


class _DequantLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, layer: QuantLinear):  # type: ignore[override]
        w = layer.dequantize(layer.compute_dtype(x))
        ctx.layer = layer
        b = layer.bias.to(w.dtype) if layer.bias is not None else None
        return F.linear(x.to(w.dtype), w, b)

    @staticmethod
    def backward(ctx, g: torch.Tensor):  # type: ignore[override]
        w = ctx.layer.dequantize(g.dtype)  # recomputed: the dense weight was not kept alive
        return g @ w, None


class QuantLinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear."""

    def __init__(self, in_features: int, out_features: int, cfg: QuantConfig, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.in_features, self.out_features, self.cfg = in_features, out_features, cfg
        self.weight_shape = (out_features, in_features)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("code", CODEBOOKS[cfg.mode]() if cfg.mode in CODEBOOKS else torch.zeros(1), persistent=False)

    @classmethod
    def from_linear(cls, lin: nn.Linear, cfg: QuantConfig) -> QuantLinear:
        q = cls(lin.in_features, lin.out_features, cfg, lin.bias)
        w = lin.weight.detach().to("cpu")  # quantize on CPU: bucketize/ops are exact and device-independent
        if cfg.mode == "int8":
            scale = (w.float().abs().amax(1) / 127.0).clamp_min(1e-12)
            q.register_buffer("qweight", torch.round(w.float() / scale[:, None]).clamp(-127, 127).to(torch.int8))
            q.register_buffer("scale", scale)
        else:
            code = CODEBOOKS[cfg.mode]()
            packed, absmax = _quant_4bit(w, code, cfg.block_size)
            q.register_buffer("qweight", packed)
            if cfg.double_quant:
                dq, s2, mean = _double_quant(absmax, cfg.scale_block)
                q.register_buffer("absmax_q", dq)
                q.register_buffer("absmax_scale", s2)
                q.register_buffer("absmax_mean", mean)
                q.n_blocks = absmax.numel()
            else:
                q.register_buffer("absmax", absmax)
        return q.to(lin.weight.device)

    def compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        dev = x.device.type
        if dev in ("cuda", "cpu", "mps") and torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
        return x.dtype

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.cfg.mode == "int8":
            return (self.qweight.float() * self.scale[:, None]).to(dtype)
        absmax = (_double_dequant(self.absmax_q, self.absmax_scale, self.absmax_mean, self.cfg.scale_block, self.n_blocks)
                  if self.cfg.double_quant else self.absmax)
        return _dequant_4bit(self.qweight, absmax, self.code, self.cfg.block_size, self.weight_shape, dtype)

    @property
    def weight(self) -> torch.Tensor:  # read-only convenience for code that inspects .weight.shape/.dtype/.device
        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DequantLinearFn.apply(x, self)

    def storage_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for n, b in self.named_buffers() if n != "code")

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, {self.cfg.mode}, block={self.cfg.block_size}, dq={self.cfg.double_quant}"


@dataclass
class QuantReport:
    layers: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    mean_sqnr_db: float = 0.0
    worst_layer: str = ""
    worst_sqnr_db: float = 1e9
    per_layer: dict[str, float] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.bytes_before / max(1, self.bytes_after)


def sqnr_db(w: torch.Tensor, w_hat: torch.Tensor) -> float:
    err = (w.float() - w_hat.float()).pow(2).mean().clamp_min(1e-30)
    return float(10 * torch.log10(w.float().pow(2).mean() / err))


def quantize_model(model: nn.Module, cfg: QuantConfig, measure: bool = True) -> QuantReport:
    """Replace every eligible nn.Linear with a QuantLinear, in place."""
    rep = QuantReport()
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)
               and not any(s in n for s in cfg.skip)]
    sq = []
    for name, lin in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        before = lin.weight.numel() * lin.weight.element_size()
        q = QuantLinear.from_linear(lin, cfg)
        if measure:
            s = sqnr_db(lin.weight.detach().cpu(), q.dequantize().cpu())
            rep.per_layer[name] = s
            sq.append(s)
            if s < rep.worst_sqnr_db:
                rep.worst_sqnr_db, rep.worst_layer = s, name
        setattr(parent, attr, q)
        rep.layers += 1
        rep.bytes_before += before
        rep.bytes_after += q.storage_bytes()
    rep.mean_sqnr_db = sum(sq) / len(sq) if sq else 0.0
    return rep


def dequantize_module(q: QuantLinear, dtype: torch.dtype = torch.float32) -> nn.Linear:
    lin = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None, device=q.qweight.device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(q.dequantize(dtype))
        if q.bias is not None:
            lin.bias.copy_(q.bias.to(dtype))
    lin.requires_grad_(False)
    return lin


def bits_per_param(cfg: QuantConfig) -> float:
    if cfg.mode == "int8":
        return 8.0
    b = 4 + 32 / cfg.block_size if not cfg.double_quant else 4 + 8 / cfg.block_size + 32 / (cfg.block_size * cfg.scale_block)
    return b


def memory_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen or t is None:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total
