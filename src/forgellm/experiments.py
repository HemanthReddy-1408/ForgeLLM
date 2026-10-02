"""`forgellm experiment`: the full adaptation program as one resumable sequence. Every step writes a marker when it
finishes, so an interrupted run continues where it stopped.

    data -> router -> baseline eval -> per-skill SFT adapters (+ eval, regression gate, registry promotion)
         -> DPO on top of the extraction adapter -> no-replay forgetting ablation -> router-in-the-loop system eval
         -> quantization study -> RAG-vs-fine-tuning study -> TinyGPT PEFT study -> demo traces
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from forgellm import studies
from forgellm.config import ARTIFACTS, ForgeConfig, load_config
from forgellm.data import synth
from forgellm.data.pipeline import default_data_dir, load_split, run_pipeline
from forgellm.data.replay import build_replay
from forgellm.evaluation.harness import Arm, evaluate_arm
from forgellm.evaluation.regression import render_comparison
from forgellm.inference.router import TaskRouter, evaluate_router
from forgellm.models.adapters import AdapterStore
from forgellm.models.base import BaseModelLoader
from forgellm.registry import ADAPTER_TARGETS, promote
from forgellm.store import get_store
from forgellm.training.preference import train_preference
from forgellm.training.sft import train_sft
from forgellm.utils import read_json, write_json

EXP = ARTIFACTS / "experiment"
REPORTS = ARTIFACTS / "reports"
SKILLS: dict[str, list[str]] = {
    "technical_reasoning": ["technical_qa", "reasoning", "grounded_qa"],
    "coding": ["coding"], "extraction": ["extraction"], "tool_use": ["tool_use"],
}


def _done(name: str) -> bool:
    return (EXP / f"{name}.done").exists()


def _mark(name: str, payload: dict[str, Any] | None = None) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    write_json(EXP / f"{name}.done", payload or {"finished": time.strftime("%Y-%m-%d %H:%M:%S")})


def _step(name: str):
    def deco(fn):
        def run(*a, **k):
            if _done(name):
                print(f"[exp] skip {name} (done)")
                return None
            t0 = time.time()
            print(f"\n[exp] === {name} ===", flush=True)
            out = fn(*a, **k)
            _mark(name, {"seconds": round(time.time() - t0, 1)})
            print(f"[exp] {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
            return out
        return run
    return deco


def _cfg(steps: int, run: str, tasks: list[str], lr: float = 2e-4, overrides: list[str] | None = None) -> ForgeConfig:
    cfg = load_config("model.yaml", "lora.yaml", "training.yaml", "inference.yaml",
                      overrides=["lora.placement=attn_mlp", f"training.max_steps={steps}", f"training.run_name={run}",
                                 f"training.learning_rate={lr}", f"training.eval_every={max(10, steps // 2)}",
                                 "training.save_every=20", "training.keep_last=1", "training.log_every=5", "training.warmup_ratio=0.06", *(overrides or [])])
    cfg.training.tasks = tasks
    return cfg


def _report(label: str) -> dict[str, Any]:
    return read_json(REPORTS / f"{label}.json")


def run_experiment(steps: int = 80, eval_n: int = 20, only: list[str] | None = None) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    want = lambda n: not only or n in only  # noqa: E731
    base_cfg = _cfg(steps, "base", [])
    loaded = BaseModelLoader(base_cfg.model).load()
    print("[exp]", loaded.spec.summary())
    store = AdapterStore(loaded.model, max_resident=6)
    test = load_split("test")

    @_step("data")
    def data() -> None:
        if not (default_data_dir() / "train.jsonl").exists():
            run_pipeline(base_cfg.data)
    data()

    @_step("router")
    def router() -> None:
        r = TaskRouter(0.55)
        info = r.fit(load_split("train"))
        r.save(ARTIFACTS / "router.pt")
        rep = evaluate_router(r, test, synth.general_prompts(60, seed=1234))
        write_json(REPORTS / "router.json", {"fit": info, **rep})
        print("[exp] router:", {k: v for k, v in rep.items() if k != "confusion"})
    router()

    @_step("replay")
    def replay() -> None:
        if not (default_data_dir() / "replay_general.jsonl").exists():
            build_replay(loaded, 160)
    replay()

    @_step("eval-base")
    def eval_base() -> None:
        write_json(REPORTS / "base.json", evaluate_arm(loaded, Arm("base"), test, eval_n, store=store))
    if want("eval-base"):
        eval_base()

    # ---- per-skill SFT adapters ------------------------------------------------------------------------------
    for name, tasks in SKILLS.items():
        if not want(f"sft-{name}"):
            continue

        @_step(f"sft-{name}")
        def sft(name=name, tasks=tasks) -> None:
            n_steps = int(steps * 1.5) if len(tasks) > 1 else steps
            cfg = _cfg(n_steps, f"sft-{name}", tasks)
            res = train_sft(cfg, name, loaded=loaded, task_for_registry=tasks[0], resume=True)
            torch_free(loaded)
            rep = evaluate_arm(loaded, Arm(name, adapter=name), test, eval_n, store=store)
            write_json(REPORTS / f"{name}.json", rep)
            gate, status = promote(name, res.version, _report("base"), rep, ADAPTER_TARGETS[name])
            print(render_comparison(_report("base"), rep, gate))
            print(f"[exp] {name}: {status}")
        sft()

    # ---- preference optimisation on top of the SFT adapter -----------------------------------------------------
    @_step("dpo-extraction")
    def dpo() -> None:
        cfg = _cfg(max(40, steps * 3 // 4), "dpo-extraction", ["extraction", "tool_use"], lr=5e-5,
                   overrides=["training.beta=0.1", "training.batch_size=4", "lora.rank=8", "lora.alpha=16"])
        cfg.training.task = "dpo"
        cfg.training.max_examples = 600
        res = train_preference(cfg, "extraction-dpo", "dpo", sft_adapter="extraction", loaded=loaded, resume=True)
        torch_free(loaded)
        rep = evaluate_arm(loaded, Arm("extraction-dpo", adapter="extraction-dpo"), test, eval_n, ["extraction", "tool_use", "general", "safety"], store)
        write_json(REPORTS / "extraction-dpo.json", rep)
        print("[exp] dpo summary:", {k: v for k, v in res.summary.items() if isinstance(v, (int, float))})
    if want("dpo-extraction") and _done("sft-extraction"):
        dpo()

    # ---- DPO, corrected: same-skill pairs only, gentler update (the first attempt mixed skills and over-optimised) -------------
    @_step("dpo-extraction-only")
    def dpo_only() -> None:
        cfg = _cfg(40, "dpo-extraction-only", ["extraction"], lr=2e-5,
                   overrides=["training.beta=0.1", "training.batch_size=4", "lora.rank=8", "lora.alpha=16"])
        cfg.training.task = "dpo"
        cfg.training.max_examples = 400
        res = train_preference(cfg, "extraction-dpo-only", "dpo", sft_adapter="extraction", loaded=loaded, resume=True)
        torch_free(loaded)
        rep = evaluate_arm(loaded, Arm("extraction-dpo-only", adapter="extraction-dpo-only"), test, eval_n, ["extraction", "tool_use", "general", "safety"], store)
        write_json(REPORTS / "extraction-dpo-only.json", rep)
        print("[exp] dpo-only summary:", {k: v for k, v in res.summary.items() if isinstance(v, (int, float))})
    if want("dpo-extraction-only") and _done("sft-extraction"):
        dpo_only()

    # ---- forgetting ablation: the same adapter WITHOUT replay data -----------------------------------------------
    @_step("ablation-noreplay")
    def noreplay() -> None:
        cfg = _cfg(steps, "sft-extraction-noreplay", ["extraction"], overrides=["data.mixture={}"])
        res = train_sft(cfg, "extraction-noreplay", loaded=loaded, task_for_registry="extraction", resume=True)
        torch_free(loaded)
        rep = evaluate_arm(loaded, Arm("extraction-noreplay", adapter="extraction-noreplay"), test, eval_n, ["extraction", "general", "safety"], store)
        write_json(REPORTS / "extraction-noreplay.json", rep)
        print("[exp] no-replay trainable:", res.summary["trainable"])
    if want("ablation-noreplay") and _done("sft-extraction"):
        noreplay()

    # ---- remediation: if an adapter fails the gate on forgetting, retrain it more gently --------------------------------
    @_step("remediate-forgetting")
    def remediate() -> None:
        name = "technical_reasoning-lowforget"
        cfg = _cfg(int(steps * 1.5), f"sft-{name}", SKILLS["technical_reasoning"], lr=1e-4,
                   overrides=["lora.rank=8", "lora.alpha=16", "data.mixture={general: 0.15, safety: 0.05}"])
        res = train_sft(cfg, name, loaded=loaded, task_for_registry="technical_qa", resume=True)
        torch_free(loaded)
        rep = evaluate_arm(loaded, Arm(name, adapter=name), test, eval_n, store=store)
        write_json(REPORTS / f"{name}.json", rep)
        gate, status = promote(name, res.version, _report("base"), rep, ADAPTER_TARGETS["technical_reasoning"])
        print(render_comparison(_report("base"), rep, gate))
        print(f"[exp] {name}: {status}")
    if want("remediate-forgetting") and _done("sft-technical_reasoning"):
        remediate()

    # ---- the full system: router decides the adapter per request ---------------------------------------------------
    @_step("system-eval")
    def system() -> None:
        r = TaskRouter.load(ARTIFACTS / "router.pt")
        rep = evaluate_arm(loaded, Arm("routed-system", router=r), test, eval_n, suites=["technical_qa", "reasoning", "coding", "extraction", "tool_use", "grounded_qa"], store=store)
        write_json(REPORTS / "routed-system.json", rep)
    if want("system-eval") and all(_done(f"sft-{n}") for n in SKILLS):
        system()

    _free()

    @_step("study-quant")
    def quant() -> None:
        studies.quant_study(base_cfg.model)
    if want("study-quant"):
        quant()
        _free()

    @_step("study-rag-vs-ft")
    def ragft() -> None:
        cfg = _cfg(40, "sft-baked-v1-facts", ["grounded_qa"], overrides=["data.mixture={}", "training.batch_size=8"])
        bake = studies.bake_dataset(1)
        train_sft(cfg, "baked-v1", examples=bake * 3, val_examples=bake[:16], task_for_registry="baked_knowledge", resume=True)
        _free()
        res = studies.rag_vs_ft_study(base_cfg.model, "technical_reasoning", "baked-v1", n=40)
        print(json.dumps(res, indent=2))
    if want("study-rag-vs-ft") and _done("sft-technical_reasoning"):
        ragft()
        _free()

    @_step("study-peft")
    def peft() -> None:
        res = studies.peft_study(steps=120, n_eval=24, labels=["full-ft", "lora-qv-r8", "lora-qkvo-r8", "lora-attn+mlp-r8", "lora-attn+mlp-r32",
                                                               "lora-last2-r8", "dora-qkvo-r8", "qlora-nf4-attn+mlp-r8", "bottleneck-d16", "ia3"])
        print(studies.render_peft(res))
    if want("study-peft"):
        peft()

    @_step("repromote")
    def repromote() -> None:
        """Apply the final gate to every adapter report (the gate logic was refined while the run was in flight)."""
        for name in (*SKILLS, "technical_reasoning-lowforget"):
            if (REPORTS / f"{name}.json").exists():
                rows = get_store().adapters(name)
                if rows:
                    gate, status = promote(name, rows[-1]["version"], _report("base"), _report(name), ADAPTER_TARGETS.get(name, ADAPTER_TARGETS["technical_reasoning"]))
                    print(f"[exp] repromote {name}: {status} {gate.reasons}")
    if want("repromote"):
        repromote()

    @_step("demo")
    def demo() -> None:
        from forgellm.cli import _load_engine

        cfg = load_config("model.yaml", "lora.yaml", "training.yaml", "inference.yaml")
        eng = _load_engine(cfg)
        qs = ["What is the request timeout of the Cobalt service?", "How busy is GPU 3 right now?",
              "Write a Python function `evens(nums)` that returns the sum of the even numbers in nums.",
              "A 7B-parameter model is stored in 4-bit. How many GB do the weights take? Use decimal GB.",
              "Plan a relaxed weekend around birdwatching."]
        out = [eng.respond(q).to_dict() for q in qs]
        write_json(EXP / "demo.json", out)
    if want("demo"):
        demo()
    print("\n[exp] all steps complete")


def _free() -> None:
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def torch_free(loaded: Any) -> None:
    from forgellm.models.lora import set_active

    set_active(loaded.model, {})
    _free()


def summarize() -> str:
    """Markdown summary of everything the experiment produced (used for the README results section)."""
    lines: list[str] = []
    base = _report("base") if (REPORTS / "base.json").exists() else None
    if base:
        names = [n for n in (*SKILLS, "technical_reasoning-lowforget", "extraction-dpo", "extraction-dpo-only", "extraction-noreplay", "routed-system") if (REPORTS / f"{n}.json").exists()]
        suites = list(base["suites"])
        lines.append("| arm | " + " | ".join(suites) + " |")
        lines.append("|---|" + "---|" * len(suites))
        lines.append("| **base** | " + " | ".join(f"{base['suites'][s]['score']:.3f}" for s in suites) + " |")
        for n in names:
            r = _report(n)
            lines.append(f"| {n} | " + " | ".join(f"{r['suites'][s]['score']:.3f}" if s in r["suites"] else "–" for s in suites) + " |")
    return "\n".join(lines)


def path_of(name: str) -> Path:
    return REPORTS / f"{name}.json"
