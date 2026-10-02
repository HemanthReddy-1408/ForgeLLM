"""Evaluation harness: run one "arm" (base model, a specific adapter, or the fully routed system) over all capability
suites with identical prompts and decoding, and return a report that regression.py can compare."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from forgellm.data.pipeline import load_split
from forgellm.data.schemas import TASK_TYPES, Example
from forgellm.data.templates import build_messages, encode_prompt, messages_for
from forgellm.evaluation import benchmarks as bm
from forgellm.evaluation.suites import SCORERS
from forgellm.inference.generation import GenConfig, generate_many
from forgellm.models.adapters import AdapterStore
from forgellm.models.base import LoadedModel
from forgellm.models.lora import loaded_adapters, set_active

DEFAULT_SUITES = [*list(TASK_TYPES), "general", "safety"]
MAX_NEW = {"technical_qa": 110, "reasoning": 130, "coding": 170, "extraction": 130, "tool_use": 90, "grounded_qa": 60}
CAPABILITY_OF = {"technical_qa": "domain", "grounded_qa": "domain", "reasoning": "reasoning", "coding": "coding",
                 "extraction": "structured_output", "tool_use": "tool_calling", "general": "general", "safety": "safety"}


@dataclass
class Arm:
    """What to evaluate. `adapters` maps task -> adapter name (None = bare base). `adapter` applies one adapter to all."""
    label: str
    adapter: str | None = None
    adapters: dict[str, str | None] = field(default_factory=dict)
    closed_book: bool = False     # grounded_qa without passages (tests baked-in knowledge)
    router: Any = None            # if set, EVERY example picks its adapter through the router (end-to-end system eval)

    def adapter_for(self, suite: str) -> str | None:
        """Per-suite override first (the routed/oracle system), else the single adapter, else the bare base model."""
        return self.adapters.get(suite, self.adapter)


def sample_eval(examples: list[Example], n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    xs = list(examples)
    rng.shuffle(xs)
    return xs[:n]


def _build_prompts(exs: list[Example], tok, closed_book: bool) -> list[list[int]]:
    prompts = []
    for ex in exs:
        if closed_book and ex.task_type == "grounded_qa":
            msgs = build_messages("technical_qa", ex.instruction)
        else:
            msgs = messages_for(ex)
        prompts.append(encode_prompt(msgs, tok))
    return prompts


def evaluate_arm(loaded: LoadedModel, arm: Arm, test: list[Example] | None = None, n_per_task: int = 40,
                 suites: list[str] | None = None, store: AdapterStore | None = None, seed: int = 0,
                 batch_size: int = 12, verbose: bool = True) -> dict[str, Any]:
    suites = suites or DEFAULT_SUITES
    test = test if test is not None else load_split("test")
    store = store or AdapterStore(loaded.model)
    tok, dev, model = loaded.tokenizer, loaded.device, loaded.model
    by_task: dict[str, list[Example]] = {}
    for e in test:
        by_task.setdefault(e.task_type, []).append(e)
    report: dict[str, Any] = {"label": arm.label, "adapter": arm.adapter, "model": loaded.spec.name, "suites": {}, "n_per_task": n_per_task}
    t_all = time.time()
    for suite in suites:
        t0 = time.time()
        adapter = arm.adapter_for(suite)
        if adapter and adapter in loaded_adapters(model) and adapter not in store.resident:
            try:                                     # trained in-process: apply its stack (e.g. DPO on top of SFT)
                chain = store.chain(adapter)
            except FileNotFoundError:                # never saved (studies)
                chain = [adapter]
            set_active(model, dict.fromkeys(chain, 1.0))
        elif adapter:
            store.activate_chain(adapter)
        else:
            set_active(model, {})
        if suite == "general":
            r = bm.general_mcq(model, tok, dev)
            r["metrics"]["heldout_ppl"] = bm.heldout_perplexity(model, tok, dev)["ppl"]
        elif suite == "safety":
            r = bm.safety_suite(model, tok, dev)
        else:
            exs = sample_eval(by_task.get(suite, []), n_per_task, seed)
            if not exs:
                continue
            prompts = _build_prompts(exs, tok, arm.closed_book)
            if arm.router is not None:
                texts, used = _generate_routed(loaded, store, arm.router, exs, prompts, MAX_NEW[suite], batch_size)
                adapter = ",".join(sorted({u or "base" for u in used}))
                outs = [type("O", (), {"new_tokens": len(tok.encode(t))}) for t in texts]
            else:
                outs = generate_many(model, tok, prompts, GenConfig(max_new_tokens=MAX_NEW[suite]), dev, batch_size)
                texts = [o.text for o in outs]
            r = SCORERS[suite](exs, texts)
            r["samples"] = [{"prompt": e.instruction[:100], "gold": e.output[:120], "pred": t[:160]} for e, t in zip(exs[:3], texts[:3])]
            r["avg_new_tokens"] = round(sum(o.new_tokens for o in outs) / len(outs), 1)
        r["n"] = len(r["scores"])
        r["score"] = round(sum(r["scores"]) / max(1, len(r["scores"])), 4)
        r["seconds"] = round(time.time() - t0, 1)
        r["adapter_used"] = adapter
        report["suites"][suite] = r
        if verbose:
            print(f"[eval:{arm.label}] {suite:<13} n={r['n']:<3} score={r['score']:.3f}  {r['metrics']}  ({r['seconds']}s)", flush=True)
    report["capabilities"] = capabilities(report)
    report["seconds"] = round(time.time() - t_all, 1)
    set_active(model, {})
    return report


def _generate_routed(loaded: LoadedModel, store: AdapterStore, router: Any, exs: list[Example], prompts: list[list[int]],
                     max_new: int, batch_size: int) -> tuple[list[str], list[str | None]]:
    """Route each example from its instruction (+ pasted document for extraction/code), then generate adapter by adapter."""
    groups: dict[str | None, list[int]] = {}
    for i, e in enumerate(exs):
        ctx = e.input if e.task_type in ("extraction", "coding") else ""
        groups.setdefault(router.route(e.instruction, ctx).adapter, []).append(i)
    texts: list[str] = [""] * len(exs)
    used: list[str | None] = [None] * len(exs)
    for adapter, idx in groups.items():
        if adapter:
            store.activate_chain(adapter)
        else:
            set_active(loaded.model, {})
        outs = generate_many(loaded.model, loaded.tokenizer, [prompts[i] for i in idx], GenConfig(max_new_tokens=max_new), loaded.device, batch_size)
        for i, o in zip(idx, outs):
            texts[i], used[i] = o.text, adapter
    return texts, used


def capabilities(report: dict[str, Any]) -> dict[str, float]:
    """Roll suites up to the six capability axes from the design brief."""
    caps: dict[str, list[float]] = {}
    for name, s in report["suites"].items():
        caps.setdefault(CAPABILITY_OF[name], []).append(s["score"])
    out = {k: round(sum(v) / len(v), 4) for k, v in caps.items()}
    fmts = [sum(s["format"]) / len(s["format"]) for s in report["suites"].values() if s.get("format")]
    if fmts:
        out["instruction_following"] = round(sum(fmts) / len(fmts), 4)   # does the output obey the requested format?
    return out
