"""`forgellm` command line. Every stage of the system is one subcommand; run `forgellm <cmd> -h` for options."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgellm.config import ARTIFACTS, CONFIGS, ForgeConfig, load_config

DEFAULT_CONFIGS = ["model.yaml", "lora.yaml", "training.yaml", "inference.yaml"]


def _cfg(args: argparse.Namespace, tiny: bool = False) -> ForgeConfig:
    files = list(args.config) if getattr(args, "config", None) else list(DEFAULT_CONFIGS)
    if tiny and "tiny.yaml" not in files:
        files.append("tiny.yaml")
    return load_config(*files, overrides=getattr(args, "set", None) or [])


def _add_cfg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", nargs="+", help=f"YAML files merged left-to-right (default {DEFAULT_CONFIGS}; searched in {CONFIGS})")
    p.add_argument("--set", nargs="*", metavar="k=v", help="dotted overrides, e.g. lora.rank=8 training.max_steps=100")
    p.add_argument("--tiny", action="store_true", help="use the from-scratch TinyGPT base (adds configs/tiny.yaml)")


def cmd_data(a: argparse.Namespace) -> None:
    from forgellm.data.pipeline import default_data_dir, load_split, run_pipeline

    if a.action == "build":
        cfg = _cfg(a).data
        if a.seed:
            cfg.seed = a.seed
        if a.scale != 1.0:
            cfg.per_task = {k: int(v * a.scale) for k, v in cfg.per_task.items()}
        run_pipeline(cfg)
    elif a.action == "report":
        print(json.dumps(json.loads((default_data_dir() / "report.json").read_text()), indent=2))
    elif a.action == "show":
        ex = load_split(a.split, tasks=[a.task] if a.task else None)[: a.n]
        for e in ex:
            print(json.dumps(e.to_dict(), indent=2, ensure_ascii=False))


def cmd_replay(a: argparse.Namespace) -> None:
    from forgellm.data.replay import build_replay
    from forgellm.models.base import BaseModelLoader

    cfg = _cfg(a, a.tiny)
    p = build_replay(BaseModelLoader(cfg.model).load(), a.n)
    print(f"wrote {p}")


def cmd_pretrain(a: argparse.Namespace) -> None:
    from forgellm.training.pretrain import pretrain_tiny

    s = pretrain_tiny(a.size, a.steps, a.batch, a.window, a.lr)
    print(json.dumps(s, indent=2))


def cmd_train(a: argparse.Namespace) -> None:
    cfg = _cfg(a, a.tiny)
    if a.tasks:
        cfg.training.tasks = a.tasks
    if a.run:
        cfg.training.run_name = a.run
    if a.objective == "sft":
        from forgellm.training.sft import train_sft

        res = train_sft(cfg, a.adapter, resume=not a.fresh)
    else:
        from forgellm.training.preference import train_preference

        cfg.training.task = a.objective
        res = train_preference(cfg, a.adapter, a.objective, a.sft, resume=not a.fresh)
    print(json.dumps({k: v for k, v in res.summary.items() if not isinstance(v, (dict, list))}, indent=2))
    print(f"adapter: {res.adapter_dir}  registered version: {res.version}")


def _load_engine(cfg: ForgeConfig, mode_router: bool = True, dense: bool = False):
    from forgellm.data.pipeline import default_data_dir
    from forgellm.inference.engine import InferenceEngine
    from forgellm.inference.router import TaskRouter
    from forgellm.models.base import BaseModelLoader
    from forgellm.rag.retriever import Retriever

    loaded = BaseModelLoader(cfg.model).load()
    rp = ARTIFACTS / "router.pt"
    router = TaskRouter.load(rp) if mode_router and rp.exists() else None
    if router:
        router.threshold = cfg.inference.router_threshold
    d = default_data_dir()
    files = [d / "knowledge_kb.jsonl", d / "knowledge_v2.jsonl"]
    retr = Retriever.from_jsonl(*[f for f in files if f.exists()], dense=dense) if files[0].exists() else None
    return InferenceEngine(loaded, cfg.inference, router, retr)


def cmd_eval(a: argparse.Namespace) -> None:
    from forgellm.data.pipeline import load_split
    from forgellm.evaluation.harness import Arm, evaluate_arm
    from forgellm.models.adapters import AdapterStore
    from forgellm.models.base import BaseModelLoader
    from forgellm.utils import write_json

    cfg = _cfg(a, a.tiny)
    loaded = BaseModelLoader(cfg.model).load()
    arm = Arm(a.label or (a.adapter or "base"), adapter=a.adapter)
    rep = evaluate_arm(loaded, arm, load_split("test"), a.n, a.suites or None, AdapterStore(loaded.model), batch_size=a.batch)
    out = Path(a.out) if a.out else ARTIFACTS / "reports" / f"{arm.label}.json"
    write_json(out, rep)
    print(json.dumps(rep["capabilities"], indent=2))
    print(f"report: {out}")


def cmd_compare(a: argparse.Namespace) -> None:
    from forgellm.evaluation.regression import compare_reports, render_comparison

    base, tuned = json.loads(Path(a.base).read_text()), json.loads(Path(a.tuned).read_text())
    g = compare_reports(base, tuned, a.targets or [], a.min_gain, a.max_drop)
    print(render_comparison(base, tuned, g))
    sys.exit(0 if g.passed else 1)


def cmd_promote(a: argparse.Namespace) -> None:
    from forgellm.registry import promote, render_comparison

    base, tuned = json.loads(Path(a.base).read_text()), json.loads(Path(a.tuned).read_text())
    gate, status = promote(a.adapter, a.version, base, tuned, a.targets or None, min_gain=a.min_gain, max_drop=a.max_drop)
    print(render_comparison(base, tuned, gate))
    print(f"-> {a.adapter}: {status}")


def cmd_registry(a: argparse.Namespace) -> None:
    from forgellm.registry import lineage, render_registry
    from forgellm.store import get_store

    print(f"store: {get_store().describe()}")
    if a.import_sqlite:
        print("imported from SQLite:", get_store().import_sqlite())
    if a.name:
        print(json.dumps(lineage(a.name), indent=2))
    else:
        print(render_registry())


def cmd_adapters(a: argparse.Namespace) -> None:
    from forgellm.models.adapters import AdapterStore, read_adapter_config

    root = ARTIFACTS / "adapters"
    if a.action == "list":
        print(f"{'adapter':<24} {'method':<10} {'params':>10} {'MiB':>7}  stack_on")
        for n in sorted(p.parent.name for p in root.glob("*/adapter_config.json")):
            m = read_adapter_config(root / n)
            print(f"{n:<24} {m['method']:<10} {m['num_parameters']:>10,} {m['size_mb']:>7}  {m.get('stack_on') or ''}")
    elif a.action == "info":
        print(json.dumps(read_adapter_config(root / a.name), indent=2))
    elif a.action == "describe":
        from forgellm.models.base import BaseModelLoader
        from forgellm.models.lora import describe_candidates

        cfg = _cfg(a, a.tiny)
        loaded = BaseModelLoader(cfg.model).load()
        print(loaded.spec.summary())
        print(describe_candidates(loaded.model, cfg.lora))
    elif a.action == "fuse":
        from forgellm.models.adapters import fuse_adapters, save_adapter
        from forgellm.models.base import BaseModelLoader

        cfg = _cfg(a, a.tiny)
        loaded = BaseModelLoader(cfg.model).load()
        store = AdapterStore(loaded.model, max_resident=8)
        weights = {}
        for item in a.items:
            n, _, w = item.partition("=")
            store.ensure(n)
            weights[n] = float(w or 1.0)
        errs = fuse_adapters(loaded.model, weights, a.out, a.rank)
        print(f"fused {weights} -> '{a.out}' (rank {a.rank}); mean relative SVD error {sum(errs.values()) / max(1, len(errs)):.4f}")
        save_adapter(loaded.model, a.out, root / a.out, cfg.lora, {"base_model": cfg.model.name, "fused_from": weights})


def cmd_route(a: argparse.Namespace) -> None:
    from forgellm.data.pipeline import load_split
    from forgellm.data.synth import general_prompts
    from forgellm.inference.router import TaskRouter, evaluate_router

    if a.action == "train":
        r = TaskRouter(0.55)
        info = r.fit(load_split("train"))
        r.save(ARTIFACTS / "router.pt")
        print(info)
    ood = general_prompts(60, seed=1234)
    r = TaskRouter.load(ARTIFACTS / "router.pt")
    r.threshold = a.threshold
    rep = evaluate_router(r, load_split("test"), ood)
    print(json.dumps({k: v for k, v in rep.items() if k != "confusion"}, indent=2))
    if a.action == "eval" and a.confusion:
        print(json.dumps(rep["confusion"], indent=2))
    if a.query:
        print(json.dumps(r.route(a.query).to_dict(), indent=2))


def cmd_ask(a: argparse.Namespace) -> None:
    cfg = _cfg(a, a.tiny)
    eng = _load_engine(cfg, dense=a.dense)
    r = eng.respond(a.query, a.context or "", a.mode, None, a.max_new)
    print("\n".join(f"  · {t}" for t in r.trace))
    print(f"\n{r.text}\n")
    if r.parsed is not None:
        print("parsed:", json.dumps(r.parsed))
    if r.tool_calls:
        print("tool calls:", json.dumps(r.tool_calls, indent=2))
    print(f"[{r.task} | adapter={r.adapter} | {r.prompt_tokens}+{r.new_tokens} tokens | {r.latency_ms:.0f} ms]")


def cmd_chat(a: argparse.Namespace) -> None:
    eng = _load_engine(_cfg(a, a.tiny))
    print("ForgeLLM chat — Ctrl-D to exit. Prefix with '/base ' to bypass adapters.")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        q = line.strip()
        if not q:
            continue
        mode = "auto"
        if q.startswith("/base "):
            mode, q = "base", q[6:]
        r = eng.respond(q, mode=mode)
        print("  " + " | ".join(r.trace[:2]))
        print(r.text)


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    from forgellm.serving.api import create_app

    eng = _load_engine(_cfg(a, a.tiny), dense=a.dense)
    uvicorn.run(create_app(eng), host=a.host, port=a.port, log_level="info")


def cmd_explain(a: argparse.Namespace) -> None:
    from forgellm.data.pipeline import load_split
    from forgellm.explain import render, trace_training_step
    from forgellm.models.base import BaseModelLoader
    from forgellm.training.sft import setup_peft

    cfg = _cfg(a, a.tiny)
    loaded = BaseModelLoader(cfg.model).load()
    setup_peft(loaded, cfg, "explain")
    ex = load_split("train", tasks=[a.task])[a.index]
    print(render(trace_training_step(loaded, ex)))


def cmd_study(a: argparse.Namespace) -> None:
    from forgellm import studies

    if a.name == "peft":
        res = studies.peft_study(a.steps, a.tasks, a.n, a.only, device=a.device, threads=a.threads)
        print(studies.render_peft(res))
    elif a.name == "quant":
        res = studies.quant_study(_cfg(a, a.tiny).model)
        print(studies.render_quant(res))
    elif a.name == "rag-vs-ft":
        res = studies.rag_vs_ft_study(_cfg(a, a.tiny).model, a.grounded, a.baked, a.n)
        print(json.dumps(res, indent=2))
    elif a.name == "render":
        for f, fn in (("peft.json", studies.render_peft), ("quant.json", studies.render_quant)):
            p = studies.STUDIES_DIR / f
            if p.exists():
                print(fn(json.loads(p.read_text())), "\n")


def cmd_experiment(a: argparse.Namespace) -> None:
    from forgellm import experiments

    if a.summary:
        print(experiments.summarize())
        return
    experiments.run_experiment(a.steps, a.eval_n, a.only or None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgellm", description="ForgeLLM — adaptive LLM training & personalization engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build / inspect the dataset")
    d.add_argument("action", choices=["build", "report", "show"])
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--scale", type=float, default=1.0, help="multiply per-task raw counts")
    d.add_argument("--split", default="train")
    d.add_argument("--task")
    d.add_argument("-n", type=int, default=3)
    _add_cfg(d)
    d.set_defaults(fn=cmd_data)

    r = sub.add_parser("replay", help="self-distilled general replay data from the base model")
    r.add_argument("-n", type=int, default=160)
    _add_cfg(r)
    r.set_defaults(fn=cmd_replay)

    pt = sub.add_parser("pretrain", help="pretrain the from-scratch TinyGPT base")
    pt.add_argument("--size", default="tiny", choices=["tiny", "small"])
    pt.add_argument("--steps", type=int, default=700)
    pt.add_argument("--batch", type=int, default=32)
    pt.add_argument("--window", type=int, default=256)
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.set_defaults(fn=cmd_pretrain)

    t = sub.add_parser("train", help="SFT / DPO / IPO / ORPO / SimPO with any PEFT method")
    t.add_argument("objective", choices=["sft", "dpo", "ipo", "orpo", "simpo"])
    t.add_argument("--adapter", required=True, help="name of the adapter to create")
    t.add_argument("--tasks", nargs="*", help="restrict to task types")
    t.add_argument("--sft", help="(preference) SFT adapter to stack on and use as the reference policy")
    t.add_argument("--run", help="run name")
    t.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    _add_cfg(t)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate the base model or an adapter on all capability suites")
    e.add_argument("--adapter")
    e.add_argument("--label")
    e.add_argument("-n", type=int, default=30, help="examples per task")
    e.add_argument("--suites", nargs="*")
    e.add_argument("--batch", type=int, default=12)
    e.add_argument("--out")
    _add_cfg(e)
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("compare", help="compare two eval reports and apply the regression gate")
    c.add_argument("base")
    c.add_argument("tuned")
    c.add_argument("--targets", nargs="*", help="suites the adapter is meant to improve")
    c.add_argument("--min-gain", type=float, default=0.03)
    c.add_argument("--max-drop", type=float, default=0.03)
    c.set_defaults(fn=cmd_compare)

    pr = sub.add_parser("promote", help="gate + promote an adapter in the registry")
    pr.add_argument("adapter")
    pr.add_argument("base")
    pr.add_argument("tuned")
    pr.add_argument("--version", type=int)
    pr.add_argument("--targets", nargs="*")
    pr.add_argument("--min-gain", type=float, default=0.03)
    pr.add_argument("--max-drop", type=float, default=0.03)
    pr.set_defaults(fn=cmd_promote)

    rg = sub.add_parser("registry", help="show the model registry")
    rg.add_argument("name", nargs="?")
    rg.add_argument("--import-sqlite", action="store_true", help="copy runs/adapters/reports from the local SQLite file into Postgres")
    rg.set_defaults(fn=cmd_registry)

    ad = sub.add_parser("adapters", help="list / inspect / fuse adapters; describe a model's adaptable modules")
    ad.add_argument("action", choices=["list", "info", "describe", "fuse"])
    ad.add_argument("name", nargs="?")
    ad.add_argument("--items", nargs="*", default=[], help="fuse: name=weight pairs")
    ad.add_argument("--out", default="fused")
    ad.add_argument("--rank", type=int, default=None)
    _add_cfg(ad)
    ad.set_defaults(fn=cmd_adapters)

    ro = sub.add_parser("route", help="train / evaluate the task router")
    ro.add_argument("action", choices=["train", "eval"])
    ro.add_argument("--threshold", type=float, default=0.55)
    ro.add_argument("--confusion", action="store_true")
    ro.add_argument("--query")
    ro.set_defaults(fn=cmd_route)

    k = sub.add_parser("ask", help="one request through the full system (router -> adapter -> RAG/tools -> parser)")
    k.add_argument("query")
    k.add_argument("--context", default="")
    k.add_argument("--mode", default="auto", help="auto | base | adapter:<name>")
    k.add_argument("--max-new", type=int, default=None)
    k.add_argument("--dense", action="store_true", help="hybrid BM25 + dense retrieval")
    _add_cfg(k)
    k.set_defaults(fn=cmd_ask)

    ch = sub.add_parser("chat", help="interactive loop")
    _add_cfg(ch)
    ch.set_defaults(fn=cmd_chat)

    s = sub.add_parser("serve", help="HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    s.add_argument("--dense", action="store_true")
    _add_cfg(s)
    s.set_defaults(fn=cmd_serve)

    x = sub.add_parser("explain-step", help="run one real training step and print every stage with numbers")
    x.add_argument("--task", default="extraction")
    x.add_argument("--index", type=int, default=0)
    _add_cfg(x)
    x.set_defaults(fn=cmd_explain)

    ex = sub.add_parser("experiment", help="run / resume the whole adaptation program (data -> adapters -> evals -> studies)")
    ex.add_argument("--steps", type=int, default=80, help="SFT steps per skill adapter")
    ex.add_argument("--eval-n", type=int, default=20, help="eval examples per task")
    ex.add_argument("--only", nargs="*", help="run only these steps (e.g. eval-base sft-coding)")
    ex.add_argument("--summary", action="store_true", help="print the results table and exit")
    ex.set_defaults(fn=cmd_experiment)

    st = sub.add_parser("study", help="engineering studies: peft | quant | rag-vs-ft | render")
    st.add_argument("name", choices=["peft", "quant", "rag-vs-ft", "render"])
    st.add_argument("--steps", type=int, default=150)
    st.add_argument("--tasks", nargs="*")
    st.add_argument("--only", nargs="*", help="peft: restrict to these labels")
    st.add_argument("-n", type=int, default=30)
    st.add_argument("--device", default="auto")
    st.add_argument("--threads", type=int, default=0)
    st.add_argument("--grounded")
    st.add_argument("--baked")
    _add_cfg(st)
    st.set_defaults(fn=cmd_study)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)

