"""Tokenizers: a dependency-free byte-level tokenizer (for the from-scratch TinyGPT) and a thin wrapper that gives
Hugging Face tokenizers the same surface (`encode`, `decode`, `pad_id`, `eos_id`)."""

from __future__ import annotations

import re
from typing import Any

SPECIALS = ["<|pad|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class ByteTokenizer:
    """ids 0..255 are raw UTF-8 bytes; 256.. are special tokens. Lossless on any text, no training needed."""

    def __init__(self) -> None:
        self.special_to_id = {s: 256 + i for i, s in enumerate(SPECIALS)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = 256 + len(SPECIALS)
        self.pad_id = self.special_to_id["<|pad|>"]
        self.eos_id = self.special_to_id["<|im_end|>"]
        self.im_start_id = self.special_to_id["<|im_start|>"]
        self._split = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
        self.name = "byte"

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self._split.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(part.encode("utf-8"))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = bytearray()
        text: list[str] = []
        for i in ids:
            if i < 256:
                out.append(i)
            else:
                text.append(out.decode("utf-8", errors="replace"))
                out = bytearray()
                if not skip_special:
                    text.append(self.id_to_special.get(i, ""))
        text.append(out.decode("utf-8", errors="replace"))
        return "".join(text)

    def __len__(self) -> int:
        return self.vocab_size


class HFTokenizer:
    def __init__(self, tok: Any) -> None:
        self.tok = tok
        self.name = getattr(tok, "name_or_path", "hf")
        self.vocab_size = len(tok)
        end = tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_id = end if isinstance(end, int) and end >= 0 and end != tok.unk_token_id else (tok.eos_token_id or 0)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos_id
        tok.padding_side = "left"

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def __len__(self) -> int:
        return self.vocab_size
