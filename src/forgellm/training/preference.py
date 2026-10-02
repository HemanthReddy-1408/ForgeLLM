"""Preference optimisation stage (after SFT): DPO / IPO / ORPO / SimPO on chosen-vs-rejected pairs.

The reference policy for DPO is the SFT model — here simply "the SFT adapter active, the new preference adapter off",
so no second copy of the model is ever loaded."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from forgellm.config import ARTIFACTS, ForgeConfig, config_hash
from forgellm.data.pipeline import default_data_dir
from forgellm.data.templates import PreferenceCollator, PreferenceDataset
from forgellm.models.adapters import load_adapter, read_adapter_config, save_adapter
from forgellm.models.base import BaseModelLoader, LoadedModel
from forgellm.models.lora import set_active
from forgellm.store import get_store
from forgellm.training.losses import PreferenceLoss
from forgellm.training.sft import TrainResult, run_dir_for, setup_peft
from forgellm.training.trainer import Trainer
from forgellm.utils import file_hash, read_jsonl, set_seed, write_json


def train_preference(cfg: ForgeConfig, adapter_name: str, kind: str = "dpo", sft_adapter: str | None = None,
                     pairs: list[dict[str, Any]] | None = None, loaded: LoadedModel | None = None,
                     data_dir: Path | None = None, resume: bool = True) -> TrainResult:
    tc = cfg.training
    set_seed(tc.seed)
    loaded = loaded or BaseModelLoader(cfg.model).load()
    data_dir = data_dir or default_data_dir()
    pairs = pairs if pairs is not None else read_jsonl(data_dir / "dpo_pairs.jsonl")
    if tc.tasks:
        pairs = [p for p in pairs if p["task_type"] in tc.tasks]
    if tc.max_examples:
        pairs = pairs[: tc.max_examples]
    n_val = max(8, len(pairs) // 20)
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]
    tok = loaded.tokenizer
    train_ds = PreferenceDataset(train_pairs, tok, cfg.model.max_seq_len)
    val_ds = PreferenceDataset(val_pairs, tok, cfg.model.max_seq_len)
    print(f"[{kind}] {len(train_ds)} train / {len(val_ds)} val pairs ({train_ds.dropped} dropped for length)")
    stack: dict[str, float] = {}
    if sft_adapter:
        load_adapter(loaded.model, ARTIFACTS / "adapters" / sft_adapter, sft_adapter)
        stack[sft_adapter] = 1.0
    peft = setup_peft(loaded, cfg, adapter_name)           # new trainable adapter, base + SFT adapter stay frozen
    set_active(loaded.model, {**stack, adapter_name: 1.0})
    print(f"[{kind}] trainable {peft['trainable']:,} / {peft['total']:,}; reference = {'SFT adapter ' + sft_adapter if sft_adapter else 'base model'}")
    store = get_store()
    run_id = tc.run_name
    store.start_run(run_id, kind, adapter_name, cfg.to_dict())
    trainer = Trainer(loaded.model, tc, train_ds, PreferenceCollator(tok.pad_id),
                      PreferenceLoss(tc, kind, ref_adapters=stack or None), run_dir_for(tc.run_name), loaded.device, val_ds,
                      lora_plus_ratio=cfg.lora.lora_plus_ratio,
                      callbacks=[lambda e, p: store.log_metrics(run_id, p.get("step", 0), {"event": e, **p})])
    t0 = time.time()
    try:
        summary = trainer.fit(resume=resume)
    except BaseException as e:
        store.finish_run(run_id, "failed", {"error": repr(e)})
        raise
    summary.update(peft)
    meta = {"base_model": cfg.model.name, "train_summary": summary, "objective": kind, "stack_on": sft_adapter,
            "data_hash": file_hash(data_dir / "dpo_pairs.jsonl"), "config_hash": config_hash(cfg),
            "tasks": tc.tasks, "trained_seconds": round(time.time() - t0, 1), "quantization": cfg.model.quantization}
    if sft_adapter:
        meta["sft_manifest"] = {k: v for k, v in read_adapter_config(ARTIFACTS / "adapters" / sft_adapter).items() if k in ("name", "data_hash", "config_hash")}
    adapter_dir = save_adapter(loaded.model, adapter_name, ARTIFACTS / "adapters" / adapter_name, cfg.lora, meta)
    version = store.register_adapter(adapter_name, cfg.model.name, cfg.lora.method, "preference", str(adapter_dir),
                                     peft["trainable"], meta["data_hash"], meta["config_hash"], summary)
    store.finish_run(run_id, "done", summary)
    write_json(run_dir_for(tc.run_name) / "config.json", cfg.to_dict())
    return TrainResult(summary, adapter_dir, loaded, run_dir_for(tc.run_name), version)
