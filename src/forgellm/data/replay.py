"""Self-distillation replay: the *base model's own* answers to generic prompts, mixed into fine-tuning so the adapted
model keeps behaving like the base outside the domain (a standard, cheap guard against catastrophic forgetting)."""

from __future__ import annotations

from pathlib import Path

from forgellm.data import synth
from forgellm.data.pipeline import default_data_dir
from forgellm.data.schemas import Example
from forgellm.data.templates import build_messages, encode_prompt
from forgellm.evaluation.bench_data import benchmark_prompt_texts
from forgellm.inference.generation import GenConfig, generate_many
from forgellm.models.base import LoadedModel
from forgellm.models.lora import set_active
from forgellm.utils import write_jsonl


def build_replay(loaded: LoadedModel, n: int = 160, max_new_tokens: int = 90, out: Path | None = None, seed: int = 99) -> Path:
    set_active(loaded.model, {})
    bench = {p.lower() for p in benchmark_prompt_texts()}
    prompts = [p for p in synth.general_prompts(n * 2, seed) if p.lower() not in bench][:n]
    tok = loaded.tokenizer
    ids = [encode_prompt(build_messages("general", p), tok) for p in prompts]
    outs = generate_many(loaded.model, tok, ids, GenConfig(max_new_tokens=max_new_tokens), loaded.device, batch_size=16, progress=True)
    rows = []
    for p, o in zip(prompts, outs):
        text = o.text.strip()
        if o.finish == "length":  # keep only complete answers
            cut = max(text.rfind(". "), text.rfind("\n"))
            text = text[: cut + 1].strip() if cut > 20 else ""
        if len(text.split()) >= 5:
            rows.append(Example(p, text, "general", difficulty="easy", source="self_distill", meta={"group": f"gen:{p}"}).to_dict())
    path = out or default_data_dir() / "replay_general.jsonl"
    write_jsonl(path, rows)
    return path
