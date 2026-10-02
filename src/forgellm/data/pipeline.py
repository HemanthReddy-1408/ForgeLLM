"""The data-engineering pipeline:

    raw generation (+ injected defects + leaked eval copies)
      -> schema validation -> cleaning -> quality scoring/filter -> prompt-hash split
      -> dedup (exact + MinHash) per split -> contamination removal (train vs val/test/benchmarks)
      -> preference pairs -> manifest + report
"""

from __future__ import annotations

import copy
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from forgellm.config import ARTIFACTS, DataConfig, config_hash
from forgellm.data import cleaning, dedup, mixing, synth
from forgellm.data import contamination as contam
from forgellm.data import knowledge as kb
from forgellm.data import quality as qual
from forgellm.data.preference import make_pairs
from forgellm.data.schemas import Example
from forgellm.evaluation.bench_data import benchmark_prompt_texts
from forgellm.utils import file_hash, read_jsonl, write_json, write_jsonl


def default_data_dir() -> Path:
    return ARTIFACTS / "data"


def _paraphrase(d: dict[str, Any], r: random.Random) -> dict[str, Any]:
    d = copy.deepcopy(d)
    d["instruction"] = r.choice(["Please ", "Quick question: ", "Hey, "]) + d["instruction"][0].lower() + d["instruction"][1:]
    return d


def build_raw(cfg: DataConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate the raw pool, then add realistic pollution: scraped copies of eval items and corrupt records."""
    rows: list[dict[str, Any]] = []
    gen_counts: dict[str, int] = {}
    tasks = dict(cfg.per_task)
    tasks.setdefault("safety", 300)
    for task, n in tasks.items():
        ex = synth.generate(task, n, cfg.seed)
        gen_counts[task] = len(ex)
        rows.extend(e.to_dict() for e in ex)
    r = random.Random(cfg.seed + 5)
    test_like = [d for d in rows if mixing.split_of(Example.from_dict(d), cfg.val_frac, cfg.test_frac) == "test"]
    leaks = []
    for d in r.sample(test_like, min(len(test_like), int(len(rows) * cfg.leak_rate))):
        c = _paraphrase(d, r) if r.random() < 0.5 else copy.deepcopy(d)
        c["source"] = "scraped"
        c.setdefault("meta", {})["force_split"] = "train"
        leaks.append(c)
    rows.extend(leaks)
    gen_counts["_leaked_eval_copies"] = len(leaks)
    rows = synth.inject_defects(rows, cfg.defect_rate, cfg.seed + 11)
    return rows, gen_counts


def run_pipeline(cfg: DataConfig | None = None, out_dir: Path | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or DataConfig()
    out = Path(out_dir or default_data_dir())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = (lambda *a: print("[data]", *a)) if verbose else (lambda *a: None)
    report: dict[str, Any] = {"config_hash": config_hash(cfg.__dict__), "seed": cfg.seed}

    raw, gen_counts = build_raw(cfg)
    write_jsonl(out / "raw.jsonl", raw)
    report["generated"] = gen_counts
    report["raw_records"] = len(raw)
    log(f"raw records: {len(raw)} (generated {sum(v for k, v in gen_counts.items() if not k.startswith('_'))}, "
        f"leaked eval copies {gen_counts['_leaked_eval_copies']}, defects injected at {cfg.defect_rate:.0%})")

    examples, clean_stats = cleaning.clean_all(raw)
    report["cleaning"] = dict(clean_stats)
    log(f"cleaning: kept {len(examples)}/{len(raw)}  drops={ {k: v for k, v in clean_stats.items() if k not in ('ok', 'redacted')} }  redacted={clean_stats.get('redacted', 0)}")

    q_reasons = qual.score_all(examples)
    scored = len(examples)
    examples = [e for e in examples if (e.quality_score or 0) >= cfg.min_quality]
    report["quality"] = {"scored": scored, "kept": len(examples), "below_threshold": scored - len(examples),
                         "failure_reasons": dict(q_reasons.most_common(12)), "threshold": cfg.min_quality}
    log(f"quality filter (>= {cfg.min_quality}): kept {len(examples)}/{scored}")

    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for e in examples:
        splits[mixing.split_of(e, cfg.val_frac, cfg.test_frac)].append(e)
    report["pre_dedup_split"] = {k: len(v) for k, v in splits.items()}

    dd_stats = {}
    for name in splits:
        splits[name], dd_stats[name] = dedup.dedup(splits[name], cfg.near_dup_threshold)
    report["dedup"] = dd_stats
    log("dedup: " + ", ".join(f"{k} -{v['exact']} exact/-{v['near']} near" for k, v in dd_stats.items()))

    eval_pool = splits["val"] + splits["test"]
    splits["train"], cinfo = contam.decontaminate(splits["train"], eval_pool, benchmark_prompt_texts(),
                                                  cfg.contamination_ngram, cfg.contamination_threshold)
    report["contamination"] = cinfo
    log(f"contamination: removed {cinfo['removed']} train examples overlapping val/test/benchmarks")

    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", (e.to_dict() for e in rows))
    pairs = make_pairs(splits["train"], n=2500, seed=cfg.seed)
    write_jsonl(out / "dpo_pairs.jsonl", (p.to_dict() for p in pairs))
    write_jsonl(out / "knowledge_kb.jsonl", kb.concept_docs())
    write_jsonl(out / "knowledge_v1.jsonl", kb.fact_docs(1))
    write_jsonl(out / "knowledge_v2.jsonl", kb.fact_docs(2))

    report["final"] = {k: len(v) for k, v in splits.items()}
    report["final_by_task"] = {k: dict(Counter(e.task_type for e in v)) for k, v in splits.items()}
    report["difficulty_train"] = dict(Counter(e.difficulty for e in splits["train"]))
    report["mean_quality_train"] = round(sum(e.quality_score or 0 for e in splits["train"]) / max(1, len(splits["train"])), 3)
    report["dpo_pairs"] = len(pairs)
    report["seconds"] = round(time.time() - t0, 1)
    manifest = {"files": {f.name: file_hash(f) for f in sorted(out.glob("*.jsonl"))}, "report": report}
    write_json(out / "report.json", report)
    write_json(out / "manifest.json", manifest)
    log(f"final: {report['final']}  dpo_pairs={len(pairs)}  ({report['seconds']}s)")
    return report


def load_split(name: str, data_dir: Path | None = None, tasks: list[str] | None = None) -> list[Example]:
    rows = read_jsonl((data_dir or default_data_dir()) / f"{name}.jsonl")
    ex = [Example.from_dict(r) for r in rows]
    return [e for e in ex if not tasks or e.task_type in tasks]
