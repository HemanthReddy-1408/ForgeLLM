"""Chat template, label masking and batching. The same renderer builds training sequences and inference prompts,
so what the model sees at train time is exactly what it sees at serve time."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset, Sampler

from forgellm.data.schemas import Example

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IGNORE = -100

SYSTEM_PROMPTS = {
    "technical_qa": "You are ForgeLLM, a precise assistant for AI/ML engineering. Answer concisely and accurately.",
    "reasoning": "You are ForgeLLM. Solve quantitative ML-engineering problems step by step and finish with '#### <answer>'.",
    "coding": "You are ForgeLLM, a careful Python expert. Follow the instruction exactly.",
    "extraction": "You are ForgeLLM, an information-extraction engine. Output only valid JSON.",
    "grounded_qa": "You are ForgeLLM. Answer using only the passages provided and cite the passage number like [1]. If the answer is not in the passages, say so.",
    "tool_use": ("You are ForgeLLM, a tool-using assistant. Available tools (JSON signatures):\n{tools}\n"
                 "To call a tool, reply with <tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call>. "
                 "If no tool is needed, answer directly."),
    "safety": "You are ForgeLLM, a helpful and harmless assistant.",
    "general": "You are ForgeLLM, a helpful and harmless assistant.",
}


class TokenizerLike(Protocol):
    pad_id: int
    eos_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...


def build_messages(task_type: str, instruction: str, inp: str = "", history: list[dict[str, str]] | None = None,
                   system: str | None = None) -> list[dict[str, str]]:
    """The conversation up to (not including) the assistant turn that is trained / generated."""
    sys_t = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["general"])
    if task_type == "tool_use":
        user = instruction
        sys_t = sys_t.format(tools=inp)
    elif task_type == "grounded_qa":
        user = f"Passages:\n{inp}\n\nQuestion: {instruction}" if inp else instruction
    elif task_type == "extraction":
        user = f"{instruction}\n\nText:\n{inp}" if inp else instruction
    else:
        user = f"{instruction}\n\n{inp}" if inp else instruction
    msgs = [{"role": "system", "content": system or sys_t}, {"role": "user", "content": user}]
    return msgs + list(history or [])


def messages_for(ex: Example) -> list[dict[str, str]]:
    return build_messages(ex.task_type, ex.instruction, ex.input, ex.meta.get("history"))


def render_prompt(messages: list[dict[str, str]]) -> str:
    """ChatML text ending with the assistant header — the generation prompt."""
    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages) + f"{IM_START}assistant\n"


def encode_prompt(messages: list[dict[str, str]], tok: TokenizerLike) -> list[int]:
    return tok.encode(render_prompt(messages))


def encode_example(ex: Example, tok: TokenizerLike, max_len: int, target: str | None = None) -> dict[str, list[int]] | None:
    """prompt tokens (label -100) + target tokens + <|im_end|> (supervised). None if it does not fit max_len."""
    prompt = encode_prompt(messages_for(ex), tok)
    tgt = tok.encode((target if target is not None else ex.output) + IM_END)
    if len(prompt) + len(tgt) > max_len:
        return None
    return {"input_ids": prompt + tgt, "labels": [IGNORE] * len(prompt) + tgt, "prompt_len": [len(prompt)]}


class SFTDataset(Dataset):
    def __init__(self, examples: list[Example], tok: TokenizerLike, max_len: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.examples: list[Example] = []
        self.dropped = 0
        for ex in examples:
            enc = encode_example(ex, tok, max_len)
            if enc is None:
                self.dropped += 1
                continue
            self.items.append(enc)
            self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]

    def lengths(self) -> list[int]:
        return [len(x["input_ids"]) for x in self.items]

    def supervised_tokens(self) -> int:
        return sum(sum(1 for t in x["labels"] if t != IGNORE) for x in self.items)


@dataclass
class Collator:
    pad_id: int
    pad_to_multiple: int = 32   # bucket sequence lengths: fewer distinct tensor shapes => fewer GPU kernel recompiles

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        n = -(-n // self.pad_to_multiple) * self.pad_to_multiple
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


class PreferenceDataset(Dataset):
    """Each item holds a shared prompt plus chosen and rejected continuations."""

    def __init__(self, pairs: list[dict[str, Any]], tok: TokenizerLike, max_len: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.dropped = 0
        for p in pairs:
            ex = Example(p["instruction"], p["chosen"], p["task_type"], p.get("input", ""), meta=p.get("meta", {}))
            c = encode_example(ex, tok, max_len)
            r = encode_example(ex, tok, max_len, target=p["rejected"])
            if c is None or r is None:
                self.dropped += 1
                continue
            self.items.append({"chosen_ids": c["input_ids"], "chosen_labels": c["labels"],
                               "rejected_ids": r["input_ids"], "rejected_labels": r["labels"]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]

    def lengths(self) -> list[int]:
        return [max(len(x["chosen_ids"]), len(x["rejected_ids"])) for x in self.items]


@dataclass
class PreferenceCollator:
    """Stacks chosen rows first, then rejected rows: batch of B pairs -> 2B sequences, one forward pass."""
    pad_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        seqs = [(b["chosen_ids"], b["chosen_labels"]) for b in batch] + [(b["rejected_ids"], b["rejected_labels"]) for b in batch]
        out = Collator(self.pad_id)([{"input_ids": i, "labels": lab} for i, lab in seqs])
        out["num_pairs"] = torch.tensor(len(batch))
        return out


class LengthGroupedSampler(Sampler[int]):
    """Deterministic shuffled batches grouped by length (cuts padding). State is (seed, epoch) so a resumed run
    sees the same order and can fast-forward to the exact batch it stopped at."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0, group: bool = True, mega: int = 50) -> None:
        self.lengths, self.bs, self.seed, self.group, self.mega = lengths, batch_size, seed, group, mega
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        rng = random.Random(self.seed * 1000003 + self.epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        if self.group:
            chunk = self.bs * self.mega
            idx = [j for s in range(0, len(idx), chunk) for j in sorted(idx[s : s + chunk], key=lambda k: self.lengths[k])]
        bats = [idx[i : i + self.bs] for i in range(0, len(idx), self.bs)]
        rng.shuffle(bats)
        return [b for b in bats if len(b) == self.bs] or bats

    def __iter__(self):  # pragma: no cover - the Trainer consumes .batches()
        return iter([i for b in self.batches() for i in b])

    def __len__(self) -> int:
        return len(self.lengths)

    def num_batches(self) -> int:
        n = len(self.lengths) // self.bs
        return n if n else 1
