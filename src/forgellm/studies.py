"""Engineering studies: each one answers a design question with measurements, not opinions.

    peft     which adaptation method / placement / rank, at what parameter & memory cost, with what forgetting? (TinyGPT)
    quant    what does 8-bit / NF4 / double-quant cost in memory, speed and quality? (any base model)
    rag-vs-ft  should changing knowledge go into the weights or into retrieval? (base model + adapters)
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch

from forgellm.config import ARTIFACTS, ForgeConfig, LoRAConfig, ModelConfig, load_config
from forgellm.data import knowledge as kb
from forgellm.data import synth
from forgellm.data.schemas import Example
from forgellm.data.templates import build_messages, encode_prompt
from forgellm.evaluation import benchmarks as bm
from forgellm.evaluation.harness import Arm, evaluate_arm
from forgellm.inference.generation import GenConfig, generate_many
from forgellm.models.adapters import AdapterStore
from forgellm.models.base import BaseModelLoader
from forgellm.models.lora import set_active
from forgellm.models.quantization import QuantConfig, bits_per_param, memory_bytes, quantize_model
from forgellm.rag.context import build_context
from forgellm.rag.retriever import Retriever
from forgellm.training.sft import train_sft
from forgellm.utils import set_seed, write_json

STUDIES_DIR = ARTIFACTS / "studies"

# label -> (method, placement, rank, quantization, lr)
PEFT_CONFIGS: dict[str, dict[str, Any]] = {
    "full-ft":            dict(method="full", lr=4e-4),
    "lora-qv-r8":         dict(method="lora", placement="qv", rank=8, lr=3e-3),
    "lora-qkvo-r8":       dict(method="lora", placement="qkvo", rank=8, lr=3e-3),
    "lora-attn+mlp-r4":   dict(method="lora", placement="attn_mlp", rank=4, lr=3e-3),
    "lora-attn+mlp-r8":   dict(method="lora", placement="attn_mlp", rank=8, lr=3e-3),
    "lora-attn+mlp-r32":  dict(method="lora", placement="attn_mlp", rank=32, lr=3e-3),
    "lora-last2-r8":      dict(method="lora", placement="attn_mlp", rank=8, layers="last:2", lr=3e-3),
    "dora-qkvo-r8":       dict(method="dora", placement="qkvo", rank=8, lr=3e-3),
    "qlora-nf4-attn+mlp-r8": dict(method="lora", placement="attn_mlp", rank=8, quant="nf4", lr=3e-3),
    "bottleneck-d16":     dict(method="bottleneck", bottleneck=16, lr=3e-3),
    "ia3":                dict(method="ia3", lr=1e-2),
}


def _cfg_for(label: str, spec: dict[str, Any], steps: int, tasks: list[str], batch: int, device: str = "auto") -> ForgeConfig:
    cfg = load_config("model.yaml", "lora.yaml", "training.yaml", "tiny.yaml")
    cfg.lora = LoRAConfig(method=spec["method"], rank=spec.get("rank", 8), alpha=2 * spec.get("rank", 8), dropout=0.0,
                          placement=spec.get("placement"), layers=spec.get("layers", "all"), bottleneck_dim=spec.get("bottleneck", 16),
                          target_modules=["q_proj", "v_proj"])
    cfg.model.quantization = spec.get("quant", "none")
    cfg.model.device = device
    t = cfg.training
    t.run_name, t.learning_rate, t.max_steps, t.batch_size, t.tasks = f"study-peft-{label}", spec["lr"], steps, batch, tasks
    t.eval_every, t.save_every, t.log_every, t.warmup_ratio, t.amp = steps, 0, 25, 0.06, "none"
    cfg.data.mixture = {}
    return cfg


def peft_study(steps: int = 150, tasks: list[str] | None = None, n_eval: int = 30, labels: list[str] | None = None,
               batch: int = 16, out: Path | None = None, device: str = "auto", threads: int = 0) -> dict[str, Any]:
    """Adapt the pretrained TinyGPT with every PEFT variant on the same data and budget; measure what each costs and keeps."""
    if threads:
        torch.set_num_threads(threads)
    tasks = tasks or ["technical_qa", "reasoning", "extraction"]
    suites = [*tasks, "general"]
    results: dict[str, Any] = {"tasks": tasks, "steps": steps, "batch": batch, "rows": {}}
    base_cfg = _cfg_for("base", PEFT_CONFIGS["lora-qv-r8"], steps, tasks, batch, device)
    loaded = BaseModelLoader(base_cfg.model).load()
    base_rep = evaluate_arm(loaded, Arm("base"), n_per_task=n_eval, suites=suites, batch_size=24, verbose=False)
    results["base"] = {"scores": {k: v["score"] for k, v in base_rep["suites"].items()}, "heldout_ppl": base_rep["suites"]["general"]["metrics"]["heldout_ppl"],
                       "weight_mib": round(memory_bytes(loaded.model) / 2**20, 2), "report": base_rep}
    print(f"[peft] base: {results['base']['scores']} ppl={results['base']['heldout_ppl']}")
    for label in labels or list(PEFT_CONFIGS):
        spec = PEFT_CONFIGS[label]
        set_seed(0)
        cfg = _cfg_for(label, spec, steps, tasks, batch, device)
        loaded = BaseModelLoader(cfg.model).load()
        base_mem = memory_bytes(loaded.model)
        t0 = time.time()
        res = train_sft(cfg, f"study-{label}", loaded=loaded, register=False, resume=False, save=False)
        wall = time.time() - t0
        name = f"study-{label}" if spec["method"] != "full" else None
        rep = evaluate_arm(loaded, Arm(label, adapter=name), n_per_task=n_eval, suites=suites, batch_size=24, verbose=False)
        scores = {k: v["score"] for k, v in rep["suites"].items()}
        row = {"trainable": res.summary["trainable"], "total": res.summary["total"],
               "trainable_pct": round(100 * res.summary["trainable"] / res.summary["total"], 3),
               "final_train_loss": res.summary.get("final_train_loss"), "val_loss": res.summary.get("final_val_loss"),
               "scores": scores, "heldout_ppl": rep["suites"]["general"]["metrics"]["heldout_ppl"],
               "forgetting_ppl_ratio": round(rep["suites"]["general"]["metrics"]["heldout_ppl"] / results["base"]["heldout_ppl"], 3),
               "domain_mean": round(sum(scores[t] for t in tasks) / len(tasks), 4),
               "base_weight_mib": round(base_mem / 2**20, 2), "wall_s": round(wall, 1), "s_per_step": round(wall / steps, 2),
               "adapter_mib": round(res.summary["trainable"] * 4 / 2**20, 3), "quant": spec.get("quant", "none")}
        results["rows"][label] = row
        print(f"[peft] {label:<24} trainable={row['trainable']:>9,} ({row['trainable_pct']:.2f}%) domain={row['domain_mean']:.3f} "
              f"val={row['val_loss'] and round(row['val_loss'], 3)} ppl_ratio={row['forgetting_ppl_ratio']} {row['wall_s']}s", flush=True)
        write_json(out or STUDIES_DIR / "peft.json", results)
        del loaded
    return results


def render_peft(results: dict[str, Any]) -> str:
    tasks = results["tasks"]
    head = "| method | trainable | % | " + " | ".join(tasks) + " | domain mean | val loss | general ppl x | s/step | base weights MiB |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    b = results["base"]
    lines.append("| base (no adaptation) | 0 | 0 | " + " | ".join(f"{b['scores'][t]:.3f}" for t in tasks)
                 + f" | {sum(b['scores'][t] for t in tasks) / len(tasks):.3f} | – | 1.00 | – | {b['weight_mib']} |")
    for label, r in results["rows"].items():
        lines.append(f"| {label} | {r['trainable']:,} | {r['trainable_pct']:.2f} | " + " | ".join(f"{r['scores'][t]:.3f}" for t in tasks)
                     + f" | {r['domain_mean']:.3f} | {r['val_loss']:.3f} | {r['forgetting_ppl_ratio']:.2f} | {r['s_per_step']} | {r['base_weight_mib']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
def quant_study(model_cfg: ModelConfig, n_new: int = 48, out: Path | None = None) -> dict[str, Any]:
    """Memory, decode speed and quality of the same model at different weight precisions."""
    variants = [("none", None), ("int8", QuantConfig("int8")), ("nf4", QuantConfig("nf4", 64, True)),
                ("nf4-noDQ", QuantConfig("nf4", 64, False)), ("uniform4", QuantConfig("uniform4", 64, True))]
    rows: dict[str, Any] = {}
    for label, q in variants:
        cfg = copy.deepcopy(model_cfg)
        cfg.quantization = "none"
        t0 = time.time()
        loaded = BaseModelLoader(cfg).load()
        rep = None
        if q is not None:
            loaded.model.to("cpu")
            rep = quantize_model(loaded.model, q)
            loaded.model.to(loaded.device)
        tok, dev, model = loaded.tokenizer, loaded.device, loaded.model
        mem = memory_bytes(model)
        ppl = bm.heldout_perplexity(model, tok, dev)["ppl"]
        acc = bm.general_mcq(model, tok, dev)["metrics"]["accuracy"]
        ids = encode_prompt(build_messages("technical_qa", "Explain gradient accumulation."), tok)
        from forgellm.utils import sync
        generate_many(model, tok, [ids], GenConfig(max_new_tokens=8), dev)
        sync(dev)
        t1 = time.perf_counter()
        r = generate_many(model, tok, [ids] * 4, GenConfig(max_new_tokens=n_new), dev, batch_size=4)
        sync(dev)
        dt = time.perf_counter() - t1
        toks = sum(x.new_tokens for x in r)
        rows[label] = {"weights_mib": round(mem / 2**20, 1), "bits_per_param": round(bits_per_param(q), 3) if q else 16.0,
                       "heldout_ppl": ppl, "mcq_acc": acc, "decode_tok_s": round(toks / dt, 1),
                       "mean_sqnr_db": round(rep.mean_sqnr_db, 2) if rep else None, "compress": round(rep.ratio, 2) if rep else 1.0,
                       "load_s": round(time.time() - t0, 1)}
        print(f"[quant] {label:<10} {rows[label]}", flush=True)
        del loaded
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    res = {"model": model_cfg.name, "rows": rows}
    write_json(out or STUDIES_DIR / "quant.json", res)
    return res


def render_quant(res: dict[str, Any]) -> str:
    lines = ["| precision | weights MiB | bits/param | held-out ppl | general MCQ | decode tok/s | SQNR dB |", "|---|---|---|---|---|---|---|"]
    for k, r in res["rows"].items():
        lines.append(f"| {k} | {r['weights_mib']} | {r['bits_per_param']} | {r['heldout_ppl']} | {r['mcq_acc']} | {r['decode_tok_s']} | {r['mean_sqnr_db'] or '–'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
def _fact_examples(n: int, seed: int = 5) -> list[Example]:
    """Questions about parameters whose value CHANGED between platform releases v1 and v2 (the 'stale trap')."""
    import random

    r = random.Random(seed)
    v1, v2 = kb.build_fact_sheet(1), kb.build_fact_sheet(2)
    changed = [k for k in v1 if v1[k] != v2[k]]
    r.shuffle(changed)
    out = []
    for s, p in changed[:n]:
        label = kb.PARAMS[p][0]
        out.append(Example(f"What is the {label} of the {s} service?", "", "grounded_qa", meta={"value": v2[(s, p)], "stale": v1[(s, p)], "service": s, "param": p}))
    return out


def _answer_stats(exs: list[Example], texts: list[str]) -> dict[str, float]:
    import re

    def has(num: str, t: str) -> bool:
        return re.search(rf"(?<![\d.]){re.escape(num.split()[0])}(?![\d]|\.\d)", t.lower().replace(",", "")) is not None

    cur = sum(has(e.meta["value"], t) and not has(e.meta["stale"], t) for e, t in zip(exs, texts)) / len(exs)
    stale = sum(has(e.meta["stale"], t) and not has(e.meta["value"], t) for e, t in zip(exs, texts)) / len(exs)
    return {"current_correct": round(cur, 4), "stale_answer": round(stale, 4)}


def rag_vs_ft_study(model_cfg: ModelConfig, grounded_adapter: str | None, baked_adapter: str | None, n: int = 40,
                    out: Path | None = None) -> dict[str, Any]:
    """Facts that changed in release v2. Arms: base+RAG, grounded-skill adapter+RAG, baked-v1-facts adapter (no RAG)."""
    loaded = BaseModelLoader(model_cfg).load()
    store = AdapterStore(loaded.model, max_resident=4)
    tok, dev, model = loaded.tokenizer, loaded.device, loaded.model
    exs = _fact_examples(n)
    retr = Retriever(kb.fact_docs(2))                 # the live corpus: updating it is a re-index, not a training run

    def run(adapter: str | None, rag: bool) -> dict[str, float]:
        store.activate_chain(adapter) if adapter else set_active(model, {})
        prompts = []
        for e in exs:
            if rag:
                passages, _ = build_context(retr.search(e.instruction, 3))
                msgs = build_messages("grounded_qa", e.instruction, passages)
            else:
                msgs = build_messages("technical_qa", e.instruction)
            prompts.append(encode_prompt(msgs, tok))
        outs = generate_many(model, tok, prompts, GenConfig(max_new_tokens=50), dev, batch_size=10)
        return _answer_stats(exs, [o.text for o in outs])

    rows: dict[str, Any] = {"base, closed-book": run(None, False), "base + RAG(v2)": run(None, True)}
    if grounded_adapter:
        rows[f"{grounded_adapter} + RAG(v2)"] = run(grounded_adapter, True)
    if baked_adapter:
        rows[f"{baked_adapter} (v1 facts baked in), closed-book"] = run(baked_adapter, False)
        rows[f"{baked_adapter} + RAG(v2)"] = run(baked_adapter, True)
    # recall of the retriever itself
    from forgellm.rag.retriever import recall_at_k

    rec = recall_at_k(retr, [(e.instruction, f"platform-v2-{e.meta['service'].lower()}") for e in exs], 3)
    res = {"n_changed_facts": len(exs), "retriever_recall@3": round(rec, 4), "rows": rows}
    write_json(out or STUDIES_DIR / "rag_vs_ft.json", res)
    return res


def bake_dataset(version: int = 1) -> list[Example]:
    return synth.closed_book_facts(version)


