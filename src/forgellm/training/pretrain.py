"""From-scratch pretraining of TinyGPT on a small technical + general corpus. This gives the PEFT study a real
(if tiny) base model to adapt, with measurable general-language ability that can be forgotten."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from forgellm.config import ARTIFACTS, TrainConfig
from forgellm.data import knowledge as kb
from forgellm.data.pipeline import default_data_dir, load_split
from forgellm.evaluation.bench_data import GENERAL_TEXT_TRAIN
from forgellm.models.tinygpt import TinyGPT, TinyGPTConfig, save_tiny
from forgellm.models.tokenizer import ByteTokenizer
from forgellm.training.losses import CausalLMLoss
from forgellm.training.trainer import Trainer
from forgellm.utils import pick_device, set_seed

SIZES = {
    "tiny": TinyGPTConfig(hidden_size=256, num_layers=6, num_heads=8, num_kv_heads=4, intermediate_size=688),
    "small": TinyGPTConfig(hidden_size=384, num_layers=8, num_heads=12, num_kv_heads=4, intermediate_size=1024),
}


def build_corpus(data_dir: Path | None = None) -> list[str]:
    docs = [d["text"] for d in kb.concept_docs()] + [d["text"] for d in kb.fact_docs(1)]
    docs += GENERAL_TEXT_TRAIN * 6
    code = []
    for ex in load_split("train", data_dir, ["coding"]):
        if ex.meta.get("subtype") == "write":
            code.append(ex.output.replace("```python\n", "").replace("\n```", ""))
    return docs * 4 + code[:600]


class WindowDataset(Dataset):
    """Random fixed-length windows over the concatenated byte stream (every token is supervised)."""

    def __init__(self, texts: list[str], tok: ByteTokenizer, window: int, n_windows: int, seed: int) -> None:
        rng = random.Random(seed)
        rng.shuffle(texts)
        ids: list[int] = []
        for t in texts:
            ids += [*tok.encode(t), tok.special_to_id["<|endoftext|>"]]
        self.ids = torch.tensor(ids)
        self.window, self.n = window, n_windows
        self.starts = [rng.randrange(0, max(1, len(ids) - window - 1)) for _ in range(n_windows)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        w = self.ids[self.starts[i] : self.starts[i] + self.window].tolist()
        return {"input_ids": w, "labels": list(w)}

    def lengths(self) -> list[int]:
        return [self.window] * self.n


def pretrain_tiny(size: str = "tiny", steps: int = 1500, batch_size: int = 32, window: int = 256, lr: float = 2e-3,
                  seed: int = 0, out: Path | None = None, data_dir: Path | None = None) -> dict:
    from forgellm.data.templates import Collator

    set_seed(seed)
    tok = ByteTokenizer()
    corpus = build_corpus(data_dir or default_data_dir())
    n_windows = steps * batch_size
    train_ds = WindowDataset(list(corpus), tok, window, n_windows, seed)
    val_ds = WindowDataset(list(corpus), tok, window, batch_size * 4, seed + 1)
    model = TinyGPT(SIZES[size])
    device = pick_device()
    model.to(device)
    print(f"[pretrain] TinyGPT-{size}: {model.num_parameters() / 1e6:.2f}M params, corpus {len(train_ds.ids):,} bytes, device {device}")
    cfg = TrainConfig(task="pretrain", run_name=f"pretrain-{size}", learning_rate=lr, batch_size=batch_size, max_steps=steps,
                      warmup_ratio=0.05, weight_decay=0.05, amp="none", eval_every=250, save_every=500, log_every=50,
                      length_grouped=False, seed=seed)
    trainer = Trainer(model, cfg, train_ds, Collator(tok.pad_id), CausalLMLoss(), ARTIFACTS / "runs" / cfg.run_name, device, val_ds)
    summary = trainer.fit(resume=True)
    path = out or ARTIFACTS / "tiny" / "base.pt"
    save_tiny(model.cpu(), path)
    summary["checkpoint"] = str(path)
    return summary
