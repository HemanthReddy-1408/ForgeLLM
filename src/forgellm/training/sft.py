"""Supervised fine-tuning orchestration: model -> PEFT injection -> datasets -> Trainer -> adapter on disk + registry."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from forgellm.config import ARTIFACTS, ForgeConfig, config_hash
from forgellm.data import mixing
from forgellm.data.pipeline import default_data_dir, load_split
from forgellm.data.schemas import TASK_TO_ADAPTER, Example
from forgellm.data.templates import Collator, SFTDataset
from forgellm.models.adapters import inject_adapter, prepare_for_training, save_adapter
from forgellm.models.base import BaseModelLoader, LoadedModel
from forgellm.models.lora import describe_candidates, set_active
from forgellm.store import get_store
from forgellm.training.losses import CausalLMLoss
from forgellm.training.trainer import Trainer
from forgellm.utils import file_hash, set_seed, write_json


@dataclass
class TrainResult:
    summary: dict[str, Any]
    adapter_dir: Path | None
    loaded: LoadedModel
    run_dir: Path
    version: int | None = None


def run_dir_for(name: str) -> Path:
    return ARTIFACTS / "runs" / name


def setup_peft(loaded: LoadedModel, cfg: ForgeConfig, adapter_name: str) -> dict[str, Any]:
    """Inject the configured PEFT method, freeze the base, report trainable fraction. method=full trains everything."""
    model, lc = loaded.model, cfg.lora
    info: dict[str, Any] = {"method": lc.method, "adapter": adapter_name}
    if lc.method == "full":
        trainable, total = prepare_for_training(model, None)
        info.update(targets=0, trainable=trainable, total=total)
        return info
    targets = inject_adapter(model, adapter_name, lc)
    trainable, total = prepare_for_training(model, [adapter_name])
    info.update(targets=len(targets), trainable=trainable, total=total, pct=100 * trainable / total,
                placement=lc.placement or ",".join(lc.target_modules), rank=lc.rank,
                quantization=cfg.model.quantization)
    return info


def select_examples(cfg: ForgeConfig, split: str = "train", data_dir: Path | None = None) -> list[Example]:
    """Training pool for a run. With `tasks` set (a per-skill adapter) the pool is those tasks plus a replay share of
    self-distilled general answers and safety examples (`data.mixture.general + .safety`) — regression protection.
    With no tasks, the configured task mixture is built over everything."""
    tc = cfg.training
    pool = load_split(split, data_dir, tc.tasks or None)
    if split != "train":
        return pool
    rng = random.Random(tc.seed)
    rng.shuffle(pool)
    n = tc.max_examples or len(pool)
    if not tc.tasks and cfg.data.mixture:
        mixed, _ = mixing.build_mixture(load_split("train", data_dir) + _replay(data_dir), cfg.data.mixture, n, tc.seed)
        return mixed
    frac = cfg.data.mixture.get("general", 0.0) + cfg.data.mixture.get("safety", 0.0)
    k = int(n * frac)
    extra = _replay(data_dir) + [e for e in load_split("train", data_dir) if e.task_type == "safety"]
    rng.shuffle(extra)
    out = pool[: max(0, n - k)] + (extra[:k] if k else [])
    rng.shuffle(out)
    return out


def _replay(data_dir: Path | None) -> list[Example]:
    """Self-distilled general replay (written by `forgellm replay`)."""
    f = (data_dir or default_data_dir()) / "replay_general.jsonl"
    if not f.exists():
        return []
    from forgellm.utils import read_jsonl

    return [Example.from_dict(r) for r in read_jsonl(f)]


def train_sft(cfg: ForgeConfig, adapter_name: str, examples: list[Example] | None = None,
              val_examples: list[Example] | None = None, loaded: LoadedModel | None = None,
              data_dir: Path | None = None, register: bool = True, resume: bool = True,
              task_for_registry: str | None = None, save: bool = True) -> TrainResult:
    tc = cfg.training
    set_seed(tc.seed)
    loaded = loaded or BaseModelLoader(cfg.model).load()
    run_dir = run_dir_for(tc.run_name)
    examples = examples if examples is not None else select_examples(cfg, "train", data_dir)
    val_examples = val_examples if val_examples is not None else load_split("val", data_dir, tc.tasks or None)
    tok = loaded.tokenizer
    train_ds = SFTDataset(examples, tok, cfg.model.max_seq_len)
    val_ds = SFTDataset(val_examples, tok, cfg.model.max_seq_len)
    print(f"[sft] {len(train_ds)} train / {len(val_ds)} val examples ({train_ds.dropped} dropped for length), "
          f"{train_ds.supervised_tokens():,} supervised tokens")
    set_active(loaded.model, {})          # adapters trained earlier in this process must not leak into this run
    peft = setup_peft(loaded, cfg, adapter_name)
    if cfg.lora.method != "full":
        print(describe_candidates(loaded.model, cfg.lora))
    print(f"[sft] trainable {peft['trainable']:,} / {peft['total']:,} ({peft.get('pct', 100):.3f}%)")
    store = get_store()
    run_id = f"{tc.run_name}"
    store.start_run(run_id, "sft", adapter_name, cfg.to_dict())

    def cb(event: str, payload: dict[str, Any]) -> None:
        store.log_metrics(run_id, payload.get("step", 0), {"event": event, **payload})

    trainer = Trainer(loaded.model, tc, train_ds, Collator(tok.pad_id), CausalLMLoss(), run_dir, loaded.device, val_ds,
                      lora_plus_ratio=cfg.lora.lora_plus_ratio, callbacks=[cb])
    t0 = time.time()
    try:
        summary = trainer.fit(resume=resume)
        if tc.early_stopping_patience:
            trainer.load_best()
    except BaseException as e:
        store.finish_run(run_id, "failed", {"error": repr(e)})
        raise
    summary.update(peft)
    adapter_dir, version = None, None
    meta = {"base_model": cfg.model.name, "train_summary": summary, "tasks": tc.tasks or list(cfg.data.mixture),
            "num_train_examples": len(train_ds), "data_hash": file_hash((data_dir or default_data_dir()) / "train.jsonl"),
            "config_hash": config_hash(cfg), "trained_seconds": round(time.time() - t0, 1),
            "quantization": cfg.model.quantization, "objective": "sft"}
    if cfg.lora.method != "full" and save:
        adapter_dir = save_adapter(loaded.model, adapter_name, ARTIFACTS / "adapters" / adapter_name, cfg.lora, meta)
        if register:
            version = store.register_adapter(adapter_name, cfg.model.name, cfg.lora.method,
                                             task_for_registry or (tc.tasks[0] if tc.tasks else "mixture"), str(adapter_dir),
                                             peft["trainable"], meta["data_hash"], meta["config_hash"], summary)
    store.finish_run(run_id, "done", summary)
    write_json(run_dir / "config.json", cfg.to_dict())
    return TrainResult(summary, adapter_dir, loaded, run_dir, version)


def adapter_for_tasks(tasks: list[str]) -> str:
    return TASK_TO_ADAPTER[tasks[0]] if len(tasks) == 1 else "multitask"


def device_info() -> str:
    return f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}, cuda={torch.cuda.is_available()}"
