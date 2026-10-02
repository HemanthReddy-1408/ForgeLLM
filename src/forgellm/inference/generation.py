"""Batched autoregressive generation with a KV cache, left-padding, and temperature / top-k / top-p / repetition-penalty
sampling. Works with TinyGPT and with Hugging Face decoder models through the same call signature."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class GenConfig:
    max_new_tokens: int = 160
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_strings: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class GenResult:
    ids: list[int]
    text: str
    finish: str            # "stop" | "length" | "stop_string"
    prompt_tokens: int
    new_tokens: int


def sample_next(logits: torch.Tensor, cfg: GenConfig, history: torch.Tensor | None, gen: torch.Generator | None) -> torch.Tensor:
    """logits [B, V] -> next token ids [B]."""
    logits = logits.float()
    if cfg.repetition_penalty != 1.0 and history is not None:
        score = logits.gather(1, history)
        score = torch.where(score < 0, score * cfg.repetition_penalty, score / cfg.repetition_penalty)
        logits = logits.scatter(1, history, score)
    if cfg.temperature <= 0:
        return logits.argmax(-1)
    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        cum = torch.softmax(sl, -1).cumsum(-1)
        drop = cum - torch.softmax(sl, -1) > cfg.top_p
        sl = sl.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, si, sl)
    probs = torch.softmax(logits, -1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(1)


@torch.no_grad()
def generate_batch(model: nn.Module, tok, prompts: list[list[int]], cfg: GenConfig, device: torch.device,
                   stop_ids: set[int] | None = None) -> list[GenResult]:
    was_training = model.training
    model.eval()
    B = len(prompts)
    stop_ids = stop_ids if stop_ids is not None else {tok.eos_id}
    L = max(len(p) for p in prompts)
    ids = torch.full((B, L), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        mask[i, L - len(p):] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    if cfg.seed is not None and cfg.temperature > 0:
        torch.manual_seed(cfg.seed)
    out_ids: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B
    finish = ["length"] * B
    past = None
    cur, cur_pos = ids, pos
    hist = ids.clone() if cfg.repetition_penalty != 1.0 else None
    for _step in range(cfg.max_new_tokens):
        o = model(input_ids=cur, attention_mask=mask, position_ids=cur_pos, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = o.past_key_values
        nxt = sample_next(o.logits[:, -1], cfg, hist, None)
        nxt_cpu = nxt.tolist()
        for i, t in enumerate(nxt_cpu):
            if finished[i]:
                continue
            if t in stop_ids:
                finished[i], finish[i] = True, "stop"
            else:
                out_ids[i].append(t)
                if cfg.stop_strings:
                    txt = tok.decode(out_ids[i][-32:])
                    if any(s in txt for s in cfg.stop_strings):
                        finished[i], finish[i] = True, "stop_string"
        if all(finished):
            break
        cur = nxt.view(B, 1)
        cur_pos = cur_pos[:, -1:] + 1
        mask = torch.cat([mask, torch.ones((B, 1), dtype=mask.dtype, device=device)], dim=1)
        if hist is not None:
            hist = torch.cat([hist, cur], dim=1)
    if was_training:
        model.train()
    return [GenResult(o, tok.decode(o), finish[i], len(prompts[i]), len(o)) for i, o in enumerate(out_ids)]


def generate_many(model: nn.Module, tok, prompts: list[list[int]], cfg: GenConfig, device: torch.device,
                  batch_size: int = 16, stop_ids: set[int] | None = None, progress: bool = False) -> list[GenResult]:
    """Length-sorted batching (less padding), results returned in the original order."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[GenResult | None] = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        outs = generate_batch(model, tok, [prompts[i] for i in idx], cfg, device, stop_ids)
        for i, r in zip(idx, outs):
            results[i] = r
        if progress:
            print(f"[gen] {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    return results  # type: ignore[return-value]
