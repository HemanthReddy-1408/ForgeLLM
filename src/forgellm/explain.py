"""`forgellm explain-step`: run ONE real training step and print every stage of the pipeline with actual numbers —
prompt -> tokenizer -> input ids -> label mask -> embeddings -> transformer blocks -> logits -> cross-entropy ->
backprop -> optimizer update. The point is to show the mechanics, not to hide them behind a trainer."""

from __future__ import annotations

import math
from typing import Any

import torch

from forgellm.data.schemas import Example
from forgellm.data.templates import IGNORE, Collator, SFTDataset, messages_for, render_prompt
from forgellm.models.lora import adapter_layers
from forgellm.training.losses import CausalLMLoss, supervised_logits
from forgellm.training.optimizers import AdamW


def trace_training_step(loaded: Any, ex: Example, lr: float = 1e-3, max_len: int = 2048) -> dict[str, Any]:
    model, tok, dev = loaded.model, loaded.tokenizer, loaded.device
    ds = SFTDataset([ex], tok, max_len)
    if not len(ds):
        raise ValueError("example does not fit max_len")
    batch = {k: v.to(dev) for k, v in Collator(tok.pad_id)([ds[0]]).items()}
    out: dict[str, Any] = {}
    out["1_prompt_text"] = render_prompt(messages_for(ex)) + ex.output + "<|im_end|>"
    ids = batch["input_ids"][0].tolist()
    lab = batch["labels"][0]
    out["2_tokens"] = {"count": len(ids), "first_ids": ids[:12], "supervised": int((lab != IGNORE).sum()),
                       "masked_prompt_tokens": int((lab == IGNORE).sum())}
    out["3_label_mask"] = f"{'·' * 24}… ×{int((lab == IGNORE).sum())} masked prompt tokens | {'█' * 24}… ×{int((lab != IGNORE).sum())} supervised answer tokens"
    model.train()
    hooks, norms = [], []
    inner = model.model
    layers = getattr(inner, "layers", [])
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: norms.append((i, float((o[0] if isinstance(o, tuple) else o).float().norm(dim=-1).mean())))))
    emb = inner.embed_tokens(batch["input_ids"])
    out["4_embeddings"] = {"shape": list(emb.shape), "mean_norm": round(float(emb.float().norm(dim=-1).mean()), 3)}
    logits, tgt, _ = supervised_logits(model, batch)
    for h in hooks:
        h.remove()
    out["5_hidden_norm_per_layer"] = [(i, round(n, 2)) for i, n in norms][:: max(1, len(norms) // 6)]
    out["6_logits"] = {"shape": list(logits.shape), "note": "only supervised positions are projected to the vocabulary"}
    loss_sum, met = CausalLMLoss()(model, batch)
    loss = loss_sum / tgt.numel()
    out["7_loss"] = {"cross_entropy_nats": round(float(loss), 4), "perplexity": round(math.exp(min(20, float(loss))), 2),
                     "token_accuracy": round(met["correct"] / met["tokens"], 3)}
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    loss.backward()
    gnorm = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in params if p.grad is not None))
    frozen_with_grad = sum(1 for p in model.parameters() if not p.requires_grad and p.grad is not None)
    out["8_backward"] = {"trainable_tensors": len(params), "trainable_params": sum(p.numel() for p in params),
                         "grad_norm": round(gnorm, 4), "frozen_tensors_with_grad": frozen_with_grad}
    before = [p.detach().clone() for p in params]
    AdamW(params, lr=lr).step()
    delta = math.sqrt(sum(float((p.detach() - b).float().pow(2).sum()) for p, b in zip(params, before)))
    with torch.no_grad():
        for p, b in zip(params, before):
            p.copy_(b)                                                       # leave the model untouched
    out["9_optimizer"] = {"update_norm_after_one_AdamW_step": round(delta, 5), "lr": lr,
                          "note": "first Adam step moves every weight by ~lr (m̂/√v̂ ≈ ±1)"}
    for p in params:
        p.grad = None
    out["adapter_layers"] = sum(1 for _ in adapter_layers(model))
    model.eval()
    return out


def render(trace: dict[str, Any]) -> str:
    lines = []
    for k, v in trace.items():
        lines.append(f"{k.replace('_', ' ', 1):<28} {v}")
    return "\n".join(lines)
