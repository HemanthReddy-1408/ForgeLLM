"""Training objectives written out explicitly:

    SFT   token cross-entropy on supervised (assistant) positions only
    DPO   -log σ(β[(π_c − π_r) − (ref_c − ref_r)])           (+ label smoothing => cDPO, + optional SFT term)
    IPO   ((π_c − π_r) − (ref_c − ref_r) − 1/2β)²
    ORPO  NLL(chosen) + λ · −log σ(log odds(chosen) − log odds(rejected))   — no reference model
    SimPO length-normalised, reference-free: −log σ(β(avg π_c − avg π_r) − γ)

Efficiency detail: the (hidden → vocab) projection is applied only at supervised positions, so a 150k-vocab model
never materialises a [batch, seq, vocab] logit tensor for prompt tokens it is not trained on."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from forgellm.config import TrainConfig
from forgellm.data.templates import IGNORE
from forgellm.models.lora import adapters_active, adapters_disabled

ROW_ALIGN = 32


def supervised_logits(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits at supervised positions [N, V], their target ids [N], and the batch row of each position [N].

    Backend workaround, measured on Apple MPS (Qwen2.5-0.5B, bf16, LoRA; share of trials with NaN gradients in every
    adapter tensor, 16-24 trials per variant): boolean-mask or advanced indexing of the hidden states gave 25-50% bad
    trials, and so did `index_select` or an fp32 projection (up to 79%) — but projecting an *unaligned* number of rows onto
    the 151k-wide vocabulary was the common factor: padding the gathered rows to a multiple of 32 before the projection and
    slicing the padding off afterwards gave 0/24, with and without gradient checkpointing, and logits at every position
    (always aligned) also gave 0%. Loss values and gradients are unchanged by the padding."""
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    tgt = batch["labels"][:, 1:]
    B, T = tgt.shape
    flat_tgt = tgt.reshape(-1)
    idx = (flat_tgt != IGNORE).nonzero(as_tuple=True)[0]
    hs = h[:, :-1].reshape(B * T, -1).index_select(0, idx)
    n = hs.size(0)
    pad = (-n) % ROW_ALIGN
    if pad:
        hs = torch.cat([hs, hs.new_zeros(pad, hs.size(1))])
    logits = model.lm_head(hs)[:n]
    return logits, flat_tgt.index_select(0, idx), torch.div(idx, T, rounding_mode="floor")


def sequence_logps(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Σ log p(token) over supervised tokens, and the token count, per sequence in the batch."""
    logits, tgt, rows = supervised_logits(model, batch)
    lp = -F.cross_entropy(logits.float(), tgt, reduction="none")
    n = batch["input_ids"].size(0)
    sums = torch.zeros(n, device=lp.device, dtype=torch.float32).index_add(0, rows, lp)
    cnt = torch.zeros(n, device=lp.device, dtype=torch.float32).index_add(0, rows, torch.ones_like(lp))
    return sums, cnt


class LossFn:
    """Interface used by the Trainer: `units(batch)` normalises across gradient accumulation; `__call__` returns
    (loss_sum, metric_sums)."""

    def units(self, batch: dict[str, torch.Tensor]) -> float:
        raise NotImplementedError

    def __call__(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError


@dataclass
class CausalLMLoss(LossFn):
    label_smoothing: float = 0.0

    def units(self, batch: dict[str, torch.Tensor]) -> float:
        return float((batch["labels"][:, 1:] != IGNORE).sum())

    def __call__(self, model: nn.Module, batch: dict[str, torch.Tensor]):
        logits, tgt, _ = supervised_logits(model, batch)
        logits = logits.float()
        loss = F.cross_entropy(logits, tgt, reduction="sum", label_smoothing=self.label_smoothing)
        correct = (logits.argmax(-1) == tgt).sum()
        return loss, {"correct": float(correct), "tokens": float(tgt.numel())}


def dpo_terms(pc: torch.Tensor, pr: torch.Tensor, rc: torch.Tensor, rr: torch.Tensor, beta: float, ls: float = 0.0) -> torch.Tensor:
    z = beta * ((pc - pr) - (rc - rr))
    return -(1 - ls) * F.logsigmoid(z) - ls * F.logsigmoid(-z)


def ipo_terms(pc, pr, rc, rr, beta: float) -> torch.Tensor:
    return ((pc - pr) - (rc - rr) - 1.0 / (2 * beta)) ** 2


def orpo_terms(lc: torch.Tensor, lr: torch.Tensor, lam: float) -> torch.Tensor:
    """lc/lr: AVERAGE per-token log-probs of chosen / rejected (both < 0)."""
    log_odds = (lc - lr) - (torch.log1p(-torch.exp(lc).clamp(max=1 - 1e-6)) - torch.log1p(-torch.exp(lr).clamp(max=1 - 1e-6)))
    return -lc + lam * -F.logsigmoid(log_odds)


def simpo_terms(lc, lr, beta: float, gamma: float) -> torch.Tensor:
    return -F.logsigmoid(beta * (lc - lr) - gamma)


@dataclass
class PreferenceLoss(LossFn):
    """Batch layout from PreferenceCollator: first B rows = chosen, last B rows = rejected (one forward pass)."""
    cfg: TrainConfig
    kind: str = "dpo"
    ref_adapters: dict[str, float] | None = None   # adapters active for the frozen reference ({} => bare base)

    def units(self, batch: dict[str, torch.Tensor]) -> float:
        return float(batch["num_pairs"])

    def _reference(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if self.ref_adapters:
                with adapters_active(model, self.ref_adapters):
                    s, _ = sequence_logps(model, batch)
            else:
                with adapters_disabled(model):
                    s, _ = sequence_logps(model, batch)
        return s

    def __call__(self, model: nn.Module, batch: dict[str, torch.Tensor]):
        B = int(batch["num_pairs"])
        sums, cnt = sequence_logps(model, batch)
        pc, pr = sums[:B], sums[B:]
        avg = sums / cnt.clamp_min(1)
        c = self.cfg
        if self.kind in ("dpo", "ipo"):
            ref = self._reference(model, batch)
            rc, rr = ref[:B], ref[B:]
            terms = dpo_terms(pc, pr, rc, rr, c.beta, c.label_smoothing) if self.kind == "dpo" else ipo_terms(pc, pr, rc, rr, c.beta)
            if c.sft_weight > 0:
                terms = terms + c.sft_weight * (-avg[:B])
            margin = c.beta * ((pc - pr) - (rc - rr))
        elif self.kind == "orpo":
            terms = orpo_terms(avg[:B], avg[B:], c.orpo_lambda)
            margin = avg[:B] - avg[B:]
        elif self.kind == "simpo":
            terms = simpo_terms(avg[:B], avg[B:], c.beta, c.simpo_gamma)
            margin = c.beta * (avg[:B] - avg[B:]) - c.simpo_gamma
        else:
            raise ValueError(f"unknown preference objective {self.kind!r}")
        margin, avg = margin.detach(), avg.detach()
        m = {"pairs": float(B), "acc": float((margin > 0).sum()), "margin": float(margin.sum()),
             "chosen_logp": float(avg[:B].sum()), "rejected_logp": float(avg[B:].sum())}
        return terms.sum(), m
