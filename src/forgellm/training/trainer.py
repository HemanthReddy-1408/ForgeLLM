"""The training engine: one Trainer for SFT, preference optimisation and pretraining.

    dataset -> length-grouped batches -> gradient accumulation (token-weighted) -> AMP autocast -> backward
            -> global-norm clipping -> AdamW -> LR schedule -> logging / eval / early stopping / checkpoint / resume

Resume is exact: the checkpoint holds the trainable weights, optimizer moments, step counter, data-order position
and RNG states, so an interrupted run continues on the very next batch it would have seen."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from forgellm.config import TrainConfig
from forgellm.data.templates import LengthGroupedSampler
from forgellm.training.losses import LossFn
from forgellm.training.optimizers import build_optimizer
from forgellm.training.schedulers import Scheduler
from forgellm.utils import device_memory_mb, free_device_cache, sync

Callback = Callable[[str, dict[str, Any]], None]


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) and v.ndim > 0 else v for k, v in batch.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, train_ds: Any, collate: Callable, loss_fn: LossFn,
                 out_dir: str | Path, device: torch.device, val_ds: Any = None, lora_plus_ratio: float = 1.0,
                 callbacks: list[Callback] | None = None, optimizer_impl: str = "forge") -> None:
        self.model, self.cfg, self.device = model, cfg, device
        self.train_ds, self.val_ds, self.collate, self.loss_fn = train_ds, val_ds, collate, loss_fn
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.callbacks = callbacks or []
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(model, cfg, lora_plus_ratio, optimizer_impl)
        self.sampler = LengthGroupedSampler(train_ds.lengths(), cfg.batch_size, cfg.seed, cfg.length_grouped)
        self.batches_per_epoch = max(1, len(self.sampler.batches()))
        self.total_steps = cfg.max_steps if cfg.max_steps > 0 else max(
            1, math.ceil(cfg.epochs * self.batches_per_epoch) // cfg.gradient_accumulation)
        self.warmup = int(self.total_steps * cfg.warmup_ratio)
        self.sched = Scheduler(self.optimizer, self.total_steps, self.warmup, cfg.scheduler, cfg.min_lr_ratio)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp)
        self.scaler = torch.amp.GradScaler("cuda") if (cfg.amp == "fp16" and device.type == "cuda") else None
        self.step = 0
        self.history: list[dict[str, Any]] = []
        self.best_val = float("inf")
        self.bad_evals = 0
        self.skipped = 0
        self.consecutive_bad = 0
        self.retries = 0
        self.tokens_seen = 0
        self._epoch_batches: list[list[int]] = []

    # ---- data ------------------------------------------------------------------------------------
    def _micro_batch(self, micro_idx: int) -> dict[str, torch.Tensor]:
        epoch, pos = divmod(micro_idx, self.batches_per_epoch)
        if not self._epoch_batches or self.sampler.epoch != epoch:
            self.sampler.set_epoch(epoch)
            self._epoch_batches = self.sampler.batches()
        idx = self._epoch_batches[pos % len(self._epoch_batches)]
        return self.collate([self.train_ds[i] for i in idx])

    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ---- one optimizer step ------------------------------------------------------------------------
    def _train_step(self) -> dict[str, float]:
        cfg = self.cfg
        micros = [_to_device(self._micro_batch(self.step * cfg.gradient_accumulation + i), self.device)
                  for i in range(cfg.gradient_accumulation)]
        total_units = max(1.0, sum(self.loss_fn.units(b) for b in micros))   # normalise by TOKENS in the whole window
        self.model.train()
        for attempt in range(self.cfg.nan_retries + 1):
            loss_total, agg = 0.0, {}
            for b in micros:
                with self._autocast():
                    loss_sum, met = self.loss_fn(self.model, b)
                loss = loss_sum / total_units
                (self.scaler.scale(loss) if self.scaler else loss).backward()
                loss_total += float(loss.detach())
                for k, v in met.items():
                    agg[k] = agg.get(k, 0.0) + v
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf")))
            if math.isfinite(gnorm) and math.isfinite(loss_total):
                break
            bad_params = [n for n, p in self.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print(f"[train] NON-FINITE step {self.step + 1} (attempt {attempt + 1}/{self.cfg.nan_retries + 1}): loss={loss_total} "
                  f"grad_norm={gnorm}, {len(bad_params)}/{len(self.params)} tensors non-finite after clipping; "
                  f"micro-batch shapes {[tuple(b['input_ids'].shape) for b in micros]}", flush=True)
            self.optimizer.zero_grad(set_to_none=True)
            free_device_cache(self.device)   # MPS: stale allocator state is what produces these NaNs; clearing it fixed 3/3 replays
            self.retries += 1
        else:
            self.skipped += 1                                                  # still non-finite: drop this update
            self.consecutive_bad += 1
            if self.consecutive_bad > 5:
                raise FloatingPointError("more than 5 consecutive non-finite steps — aborting (lower the LR or check the data)")
            return {"loss": float("nan"), "grad_norm": gnorm, "skipped": 1.0}
        self.consecutive_bad = 0
        self.tokens_seen += sum(int(b["attention_mask"].sum()) for b in micros if "attention_mask" in b)
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.sched.step()
        agg.update({"loss": loss_total, "grad_norm": gnorm, "lr": lr})
        return agg

    # ---- evaluation --------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, max_batches: int = 40) -> dict[str, float]:
        if self.val_ds is None or len(self.val_ds) == 0:
            return {}
        self.model.eval()
        bs = self.cfg.batch_size
        tot, units, agg = 0.0, 0.0, {}
        for s in range(0, min(len(self.val_ds), bs * max_batches), bs):
            b = _to_device(self.collate([self.val_ds[i] for i in range(s, min(s + bs, len(self.val_ds)))]), self.device)
            with self._autocast():
                loss_sum, met = self.loss_fn(self.model, b)
            tot += float(loss_sum)
            units += self.loss_fn.units(b)
            for k, v in met.items():
                agg[k] = agg.get(k, 0.0) + v
        out = {"val_loss": tot / max(1.0, units)}
        if "correct" in agg:
            out["val_acc"] = agg["correct"] / max(1.0, agg["tokens"])
            out["val_ppl"] = math.exp(min(20.0, out["val_loss"]))
        if "pairs" in agg:
            out["val_pref_acc"] = agg["acc"] / agg["pairs"]
            out["val_margin"] = agg["margin"] / agg["pairs"]
        return out

    # ---- checkpoints -------------------------------------------------------------------------------
    def _trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}

    def save_checkpoint(self, tag: str | None = None) -> Path:
        name = tag or f"step_{self.step:06d}"
        final = self.out / "ckpt" / name
        tmp = self.out / "ckpt" / f".{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save(self._trainable_state(), tmp / "trainable.pt")
        torch.save(self.optimizer.state_dict(), tmp / "optim.pt")
        state = {"step": self.step, "best_val": self.best_val, "bad_evals": self.bad_evals, "skipped": self.skipped,
                 "tokens_seen": self.tokens_seen, "history": self.history[-500:],
                 "torch_rng": torch.get_rng_state().tolist()}
        (tmp / "trainer.json").write_text(json.dumps(state))
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)                                                 # atomic publish
        if tag is None:
            ckpts = sorted((self.out / "ckpt").glob("step_*"))
            for old in ckpts[: -self.cfg.keep_last]:
                shutil.rmtree(old, ignore_errors=True)
        return final

    def latest_checkpoint(self) -> Path | None:
        ck = sorted((self.out / "ckpt").glob("step_*")) if (self.out / "ckpt").exists() else []
        return ck[-1] if ck else None

    def load_checkpoint(self, path: Path) -> None:
        sd = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, t in sd.items():
                own[n].copy_(t.to(own[n].device, own[n].dtype))
        self.optimizer.load_state_dict(torch.load(path / "optim.pt", map_location="cpu", weights_only=True))
        st = json.loads((path / "trainer.json").read_text())
        self.step, self.best_val, self.bad_evals = st["step"], st["best_val"], st["bad_evals"]
        self.skipped, self.tokens_seen, self.history = st["skipped"], st["tokens_seen"], st["history"]
        torch.set_rng_state(torch.tensor(st["torch_rng"], dtype=torch.uint8))
        self.sched.seek(self.step)

    # ---- the loop ----------------------------------------------------------------------------------
    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb(event, payload)

    def fit(self, resume: bool = True, best_hook: Callable[[], None] | None = None) -> dict[str, Any]:
        cfg = self.cfg
        if resume and (ck := self.latest_checkpoint()):
            self.load_checkpoint(ck)
            print(f"[train] resumed from {ck.name} (step {self.step}/{self.total_steps})")
        n_train = sum(p.numel() for p in self.params)
        print(f"[train] {cfg.task} '{cfg.run_name}': {len(self.train_ds)} examples, {self.total_steps} steps "
              f"(batch {cfg.batch_size} x accum {cfg.gradient_accumulation}), {n_train:,} trainable params, "
              f"lr {cfg.learning_rate:g} {cfg.scheduler}, amp={cfg.amp}, device={self.device}")
        log_f = open(self.out / "metrics.jsonl", "a")  # noqa: SIM115 - closed in the finally block below
        t0, t_last, tok_last = time.perf_counter(), time.perf_counter(), self.tokens_seen
        window: list[float] = []
        stopped_early = False
        try:
            while self.step < self.total_steps:
                m = self._train_step()
                self.step += 1
                if "skipped" in m:
                    continue
                window.append(m["loss"])
                if self.step % cfg.log_every == 0 or self.step == self.total_steps:
                    sync(self.device)
                    now = time.perf_counter()
                    rec = {"step": self.step, "loss": round(sum(window) / len(window), 5), "lr": m["lr"],
                           "grad_norm": round(m["grad_norm"], 4), "tok_s": round((self.tokens_seen - tok_last) / max(1e-9, now - t_last)),
                           "mem_mb": round(device_memory_mb(self.device)), "elapsed": round(now - t0, 1)}
                    if "correct" in m:
                        rec["acc"] = round(m["correct"] / max(1.0, m["tokens"]), 4)
                    if "pairs" in m:
                        rec["pref_acc"] = round(m["acc"] / m["pairs"], 4)
                        rec["margin"] = round(m["margin"] / m["pairs"], 4)
                    window, t_last, tok_last = [], now, self.tokens_seen
                    self.history.append(rec)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    self._emit("log", rec)
                    print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                if self.val_ds is not None and (self.step % cfg.eval_every == 0 or self.step == self.total_steps):
                    ev = self.evaluate()
                    ev["step"] = self.step
                    self.history.append(ev)
                    log_f.write(json.dumps(ev) + "\n")
                    log_f.flush()
                    self._emit("eval", ev)
                    print("[eval ] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in ev.items()))
                    if ev["val_loss"] < self.best_val - 1e-4:
                        self.best_val, self.bad_evals = ev["val_loss"], 0
                        self.save_checkpoint("best")
                    else:
                        self.bad_evals += 1
                        if cfg.early_stopping_patience and self.bad_evals >= cfg.early_stopping_patience:
                            print(f"[train] early stopping: no val improvement in {self.bad_evals} evals")
                            stopped_early = True
                if cfg.save_every and self.step % cfg.save_every == 0:
                    self.save_checkpoint()
                if stopped_early:
                    break
        finally:
            log_f.close()
        self.save_checkpoint()
        final_eval = self.evaluate() if self.val_ds is not None else {}
        train_losses = [h["loss"] for h in self.history if "loss" in h and "step" in h and "val_loss" not in h]
        summary = {"steps": self.step, "stopped_early": stopped_early, "wall_s": round(time.perf_counter() - t0, 1),
                   "final_train_loss": float(np.mean(train_losses[-5:])) if train_losses else None,
                   "first_train_loss": train_losses[0] if train_losses else None,
                   "best_val_loss": None if self.best_val == float("inf") else self.best_val, "skipped_steps": self.skipped, "nan_retries": self.retries,
                   "tokens_seen": self.tokens_seen, "trainable_params": sum(p.numel() for p in self.params),
                   "peak_mem_mb": round(device_memory_mb(self.device)), **{f"final_{k}": v for k, v in final_eval.items()}}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def load_best(self) -> bool:
        best = self.out / "ckpt" / "best"
        if not best.exists():
            return False
        sd = torch.load(best / "trainable.pt", map_location="cpu", weights_only=True)
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, t in sd.items():
                own[n].copy_(t.to(own[n].device, own[n].dtype))
        return True

