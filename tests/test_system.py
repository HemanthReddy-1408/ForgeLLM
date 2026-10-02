from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import make_tiny
from forgellm.config import InferenceConfig, LoRAConfig
from forgellm.data import knowledge as kb
from forgellm.data.pipeline import load_split
from forgellm.data.schemas import Example
from forgellm.evaluation import benchmarks as bm
from forgellm.evaluation import sandbox, suites
from forgellm.evaluation.harness import Arm, capabilities, evaluate_arm
from forgellm.evaluation.regression import compare_reports, paired_bootstrap
from forgellm.inference.engine import InferenceEngine
from forgellm.inference.generation import GenConfig, generate_batch, generate_many, sample_next
from forgellm.inference.parser import (
    extract_citations,
    extract_code,
    extract_final_answer,
    extract_json,
    parse_number,
)
from forgellm.inference.router import TaskRouter, evaluate_router
from forgellm.models.adapters import adapter_parameters, inject_adapter, save_adapter
from forgellm.rag.context import build_context
from forgellm.rag.retriever import BM25, Retriever, recall_at_k, rrf, tokenize
from forgellm.registry import promote
from forgellm.serving.api import create_app
from forgellm.store import Store
from forgellm.tools.registry import ToolRegistry


# ---- generation ------------------------------------------------------------------------------------------------
def test_greedy_generation_matches_manual_argmax_loop(tiny):
    tok, m = tiny.tokenizer, tiny.model
    ids = tok.encode("hello world")
    out = generate_batch(m, tok, [ids], GenConfig(max_new_tokens=8), tiny.device, stop_ids=set())[0].ids
    cur = list(ids)
    for _ in range(8):
        cur.append(int(m(torch.tensor([cur])).logits[0, -1].argmax()))
    assert out == cur[len(ids):]


def test_batched_generation_equals_single_generation(tiny):
    tok = tiny.tokenizer
    ps = [tok.encode("a"), tok.encode("a much longer prompt here")]
    batched = generate_many(tiny.model, tok, ps, GenConfig(max_new_tokens=6), tiny.device, batch_size=2, stop_ids=set())
    for p, b in zip(ps, batched):
        assert generate_batch(tiny.model, tok, [p], GenConfig(max_new_tokens=6), tiny.device, stop_ids=set())[0].ids == b.ids


def test_sampling_filters():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -2.0]])
    assert sample_next(logits, GenConfig(temperature=0), None, None).item() == 0
    draws = {int(sample_next(logits, GenConfig(temperature=1.5, top_k=2), None, None)) for _ in range(200)}
    assert draws <= {0, 1}
    assert {int(sample_next(logits, GenConfig(temperature=1.0, top_p=0.5), None, None)) for _ in range(100)} == {0}
    pen = sample_next(torch.tensor([[2.0, 1.9]]), GenConfig(repetition_penalty=2.0), torch.tensor([[0]]), None)
    assert pen.item() == 1


# ---- parsing ----------------------------------------------------------------------------------------------------
def test_json_extraction_variants():
    assert extract_json('{"a": 1}') == ({"a": 1}, "strict")
    assert extract_json('Sure!\n```json\n{"a": [1, 2], "b": "x}"}\n```\nDone')[1] == "embedded"
    assert extract_json("{'a': 1, 'b': 2,}") == ({"a": 1, "b": 2}, "repaired")
    assert extract_json("no json here")[0] is None
    assert extract_json('prefix {"nested": {"k": "v"}} suffix')[0] == {"nested": {"k": "v"}}


def test_answer_and_code_extraction():
    assert extract_final_answer("Step 1...\n#### 1,024") == "1,024" and parse_number("1,024") == 1024
    assert extract_code("text\n```python\nx = 1\n```") == "x = 1"
    assert extract_citations("see [1] and [3].") == [1, 3]


# ---- RAG ---------------------------------------------------------------------------------------------------------
def test_bm25_ranks_the_right_service_document():
    r = Retriever(kb.fact_docs(2) + kb.concept_docs())
    top = r.search("What is the request timeout of the Cobalt service?", 3)
    assert top[0]["id"] == "platform-v2-cobalt"
    top = r.search("What is the default replica count of the Atlas-Gateway service?", 1)
    assert top[0]["id"] == "platform-v2-atlas-gateway"
    assert r.search("explain QLoRA double quantization", 3)[0]["id"].startswith("kb-")


def test_retrieval_recall_and_freshness():
    v2 = Retriever(kb.fact_docs(2))
    qs = [(f"What is the {kb.PARAMS[p][0]} of the {s} service?", f"platform-v2-{s.lower()}") for s in kb.SERVICES for p in kb.PARAMS]
    assert recall_at_k(v2, qs, 3) > 0.95
    s, p = "Orion", "default_lora_rank"
    sheet1, sheet2 = kb.build_fact_sheet(1), kb.build_fact_sheet(2)
    changed = [k for k in sheet1 if sheet1[k] != sheet2[k]][0]
    doc = v2.search(f"{kb.PARAMS[changed[1]][0]} of the {changed[0]} service", 1)[0]["text"]
    assert sheet2[changed] in doc and sheet1[changed] not in doc          # re-indexing alone updates the answer
    assert (s, p) in sheet1


def test_rrf_and_context_budget():
    assert max(rrf([[1, 2, 3], [3, 1, 4]]).items(), key=lambda kv: kv[1])[0] in (1, 3)
    hits = [{"id": str(i), "title": "t", "text": "x" * 800, "score": 1.0} for i in range(5)] + [{"id": "0", "title": "t", "text": "dup", "score": 1}]
    text, kept = build_context(hits, budget_chars=2000)
    assert len(kept) == 2 and text.startswith("[1] t") and "[2]" in text
    assert tokenize("Atlas-Gateway service")[:3] == ["atlas-gateway", "atlas", "gateway"]
    assert BM25(["a b c", "d e f"]).scores("zzz").sum() == 0


# ---- router --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_router(data_dir):
    r = TaskRouter(0.5)
    r.fit(load_split("train", data_dir))
    return r


def test_router_accuracy_and_fallback(trained_router, data_dir):
    rep = evaluate_router(trained_router, load_split("test", data_dir), ["Write a haiku about autumn leaves", "Tell me a joke about cats", "Plan a trip to Rome"])
    assert rep["adapter_accuracy"] > 0.85, rep["per_task_recall"]
    assert rep["ood_fallback_rate"] >= 0.3
    d = trained_router.route("Extract the following fields from the text and return only a JSON object: name (string).", "John is 30.")
    assert d.task == "extraction" and d.adapter == "extraction"
    d = trained_router.route("What is the request timeout of the Cobalt service?")
    assert d.task == "grounded_qa" and d.use_rag
    assert not trained_router.route("Write a Python function `f(nums)` that returns the sum.").use_rag
    assert trained_router.route("Is it raining in Pune?").adapter == "tool_use"


def test_router_save_load(trained_router, tmp_path):
    trained_router.save(tmp_path / "r.pt")
    r2 = TaskRouter.load(tmp_path / "r.pt")
    q = "Convert 5 GB to MB."
    assert r2.route(q).scores == trained_router.route(q).scores


# ---- engine --------------------------------------------------------------------------------------------------------
def _engine(tiny, tmp_path, router=None, retriever=None):
    donor = make_tiny()
    cfg = LoRAConfig(rank=2, alpha=4, dropout=0.0, placement="qv")
    for n in ("technical_reasoning", "tool_use", "extraction"):
        inject_adapter(donor.model, n, cfg)
        for _, p in adapter_parameters(donor.model, n):
            p.data.normal_(0, 0.05)
        save_adapter(donor.model, n, tmp_path / n, cfg, {})
    return InferenceEngine(tiny, InferenceConfig(max_new_tokens=12, adapters_dir=str(tmp_path)), router, retriever)


def test_engine_routes_swaps_adapters_and_retrieves(tiny, tmp_path, trained_router):
    eng = _engine(tiny, tmp_path, trained_router, Retriever(kb.fact_docs(2) + kb.concept_docs()))
    r = eng.respond("What is the request timeout of the Cobalt service?")
    assert r.task == "grounded_qa" and r.adapter == "technical_reasoning" and r.used_rag and r.retrieved[0]["id"] == "platform-v2-cobalt"
    r2 = eng.respond("Is it raining in Pune?")
    assert r2.adapter == "tool_use" and "tools offered" in " ".join(r2.trace) and "get_weather" in " ".join(r2.trace)
    assert eng.respond("anything", mode="base").adapter is None
    assert eng.respond("anything", mode="adapter:tool_use", task="tool_use").adapter == "tool_use"
    with pytest.raises(FileNotFoundError):
        eng.respond("x", mode="adapter:missing")


def test_engine_tool_loop_executes_and_feeds_observation(tiny, tmp_path):
    eng = _engine(tiny, tmp_path)
    scripted = iter(['<tool_call>\n{"name": "get_gpu_status", "arguments": {"gpu_id": 3}}\n</tool_call>', "GPU 3 is busy."])
    seen = []
    class R:  # minimal GenResult stand-in
        def __init__(self, t): self.text, self.finish, self.prompt_tokens, self.new_tokens = t, "stop", 5, 7
    def fake(messages, gcfg):
        seen.append(messages)
        return R(next(scripted))
    eng._gen = fake
    r = eng.respond("How busy is GPU 3?", task="tool_use")
    assert r.tool_calls[0]["ok"] and r.tool_calls[0]["observation"]["gpu_id"] == 3 and r.text == "GPU 3 is busy."
    assert "<tool_response>" in seen[1][-1]["content"] and seen[1][-2]["role"] == "assistant"


def test_engine_rejects_invalid_tool_arguments_without_crashing(tiny, tmp_path):
    eng = _engine(tiny, tmp_path)
    outs = iter(['<tool_call>\n{"name": "get_gpu_status", "arguments": {"gpu_id": 99}}\n</tool_call>', "Sorry, that GPU does not exist."])
    class R:
        def __init__(self, t): self.text, self.finish, self.prompt_tokens, self.new_tokens = t, "stop", 1, 1
    eng._gen = lambda m, g: R(next(outs))
    r = eng.respond("status of gpu 99", task="tool_use")
    assert r.tool_calls[0]["ok"] is False and "maximum" in r.tool_calls[0]["error"]


def test_engine_parses_structured_output(tiny, tmp_path):
    eng = _engine(tiny, tmp_path)
    class R:
        text, finish, prompt_tokens, new_tokens = '```json\n{"a": 1}\n```', "stop", 1, 1
    eng._gen = lambda m, g: R()
    r = eng.respond("extract", task="extraction")
    assert r.parsed == {"a": 1} and r.parse_status == "embedded"


# ---- evaluation ----------------------------------------------------------------------------------------------------
def test_sandbox_pass_fail_timeout_and_unsafe():
    t = [{"args": [[1, 2, 3]], "expected": 6}]
    assert sandbox.run_tests("def f(x):\n    return sum(x)", "f", t)["ok"]
    assert not sandbox.run_tests("def f(x):\n    return 0", "f", t)["ok"]
    assert sandbox.run_tests("def f(x):\n    while True: pass", "f", t, timeout=2)["error"] == "timeout"
    assert "unsafe" in sandbox.run_tests("import os\ndef f(x):\n    return 6", "f", t)["error"]
    assert not sandbox.run_tests("def f(x:\n", "f", t)["ok"]


def test_suite_scorers():
    ex = Example("q", json.dumps({"a": 1, "b": None, "c": "Hi There"}), "extraction", meta={"schema": {"a": "integer", "b": "string", "c": "string"}})
    s = suites.score_extraction([ex, ex, ex], ['{"a": 1, "b": null, "c": "hi there"}', '```json\n{"a": 2, "b": null, "c": "Hi There"}\n```', "nope"])
    assert s["scores"] == [1.0, pytest.approx(2 / 3), 0.0] and s["metrics"]["valid_json_strict"] == pytest.approx(1 / 3, abs=1e-3)
    r = Example("q", "x\n#### 42", "reasoning", meta={"answer": "42"})
    assert suites.score_reasoning([r, r, r], ["#### 42", "so 42.0", "#### 41"])["scores"] == [1.0, 1.0, 0.0]
    call = Example("Weather in Pune?", "", "tool_use", meta={"stage": "call", "tool": "get_weather", "arguments": {"city": "Pune"}})
    ok = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pune"}}\n</tool_call>'
    bad = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Delhi"}}\n</tool_call>'
    assert suites.score_tool_use([call, call, call], [ok, bad, "I cannot"])["scores"] == [1.0, 0.0, 0.0]
    ab = Example("hi", "hello", "tool_use", meta={"stage": "abstain", "tool": None})
    assert suites.score_tool_use([ab, ab], ["Hello there!", ok])["scores"] == [1.0, 0.0]
    g = Example("q", "", "grounded_qa", meta={"value": "30 seconds", "cite": 2})
    assert suites.score_grounded([g, g], ["It is 30 seconds [2].", "It is 130 [2]"])["scores"] == [1.0, 0.0]
    assert suites.token_f1("a b c", "a b c") == 1.0 and suites.keyterm_recall("gradient accumulation", "gradient accumulation sums gradients") > 0.4


def test_general_benchmarks_run_on_any_model(tiny):
    r = bm.general_mcq(tiny.model, tiny.tokenizer, tiny.device)
    assert len(r["scores"]) == 50 and 0.0 <= r["metrics"]["accuracy"] <= 1.0
    assert bm.heldout_perplexity(tiny.model, tiny.tokenizer, tiny.device)["ppl"] > 1
    assert bm.is_refusal("I'm sorry, but I can't help with that.") and not bm.is_refusal("Sure, here is how it works.")


def test_harness_end_to_end_on_tiny(tiny, data_dir):
    rep = evaluate_arm(tiny, Arm("tiny-base"), load_split("test", data_dir), n_per_task=4, suites=["extraction", "reasoning", "general"], batch_size=4, verbose=False)
    assert set(rep["suites"]) == {"extraction", "reasoning", "general"} and "structured_output" in rep["capabilities"]
    assert rep["suites"]["extraction"]["n"] == 4 and capabilities(rep)["general"] == rep["suites"]["general"]["score"]


def _rep(label, scores, ppl=10.0):
    return {"label": label, "suites": {k: {"scores": v, "score": sum(v) / len(v), "metrics": {"heldout_ppl": ppl} if k == "general" else {}} for k, v in scores.items()}}


def test_paired_bootstrap_detects_real_difference_but_not_noise():
    a = [0.0] * 60 + [1.0] * 40
    d, lo, hi = paired_bootstrap(a, [1.0] * 100)
    assert d == pytest.approx(0.6) and lo > 0.45
    d, lo, hi = paired_bootstrap(a, list(reversed(a)))
    assert lo <= 0 <= hi


def test_regression_gate_is_evidence_based():
    base = _rep("base", {"extraction": [0.0] * 50, "general": [1.0] * 40 + [0.0] * 10, "safety": [1.0] * 40})
    good = _rep("good", {"extraction": [1.0] * 50, "general": [1.0] * 40 + [0.0] * 10, "safety": [1.0] * 40})
    assert compare_reports(base, good, ["extraction"]).passed
    # significant forgetting of a protected capability fails; the "domain up, general down" warning fires
    forgetful = _rep("bad", {"extraction": [1.0] * 50, "general": [1.0] * 25 + [0.0] * 25, "safety": [1.0] * 40})
    g = compare_reports(base, forgetful, ["extraction"])
    assert not g.passed and any("protected 'general'" in r and "significant" in r for r in g.reasons)
    assert any("forgetting" in w for w in g.warnings)
    # a dip that is beyond the limit but statistically noise only warns
    noisy = _rep("noisy", {"extraction": [1.0] * 50, "general": [1.0] * 40 + [0.0] * 6 + [1.0] * 0 + [0.0] * 4, "safety": [1.0] * 38 + [0.0] * 2})
    g = compare_reports(base, noisy, ["extraction"])
    assert g.passed and any("not statistically significant" in w for w in g.warnings)
    # perplexity is deterministic: a hard limit
    ppl = _rep("ppl", {"extraction": [1.0] * 50, "general": [1.0] * 40 + [0.0] * 10, "safety": [1.0] * 40}, ppl=20.0)
    g = compare_reports(base, ppl, ["extraction"])
    assert not g.passed and g.ppl_ratio == 2.0
    # targets are judged on their mean gain, so one weak target suite does not sink a strong adapter
    multi_base = _rep("b", {"a": [0.0] * 40, "b": [0.5] * 40, "general": [1.0] * 40})
    multi = _rep("m", {"a": [1.0] * 40, "b": [0.5] * 40, "general": [1.0] * 40})
    assert compare_reports(multi_base, multi, ["a", "b"]).passed
    assert not compare_reports(base, _rep("useless", {"extraction": [0.0] * 50, "general": [1.0] * 40 + [0.0] * 10, "safety": [1.0] * 40}), ["extraction"]).passed
    # a target that significantly regresses fails even if another target gained
    mixed = _rep("x", {"a": [1.0] * 40, "b": [0.0] * 40, "general": [1.0] * 40})
    assert not compare_reports(multi_base, mixed, ["a", "b"]).passed


# ---- store / registry ----------------------------------------------------------------------------------------------
def _store_flow(s):
    s.start_run("r1", "sft", "x", {"lr": 1})
    s.log_metrics("r1", 5, {"loss": 1.5})
    s.finish_run("r1", "done", {"final": 1})
    assert s.runs()[0]["status"] == "done" and s.metrics("r1")[0]["loss"] == 1.5
    v1 = s.register_adapter("extraction", "m", "lora", "extraction", "/p", 10, "h", "c", {})
    v2 = s.register_adapter("extraction", "m", "lora", "extraction", "/p2", 10, "h", "c", {})
    assert (v1, v2) == (1, 2)
    base = _rep("base", {"extraction": [0.0] * 40, "general": [1.0] * 40, "safety": [1.0] * 40})
    tuned = _rep("t", {"extraction": [1.0] * 40, "general": [1.0] * 40, "safety": [1.0] * 40})
    gate, status = promote("extraction", 1, base, tuned, store=s)
    assert gate.passed and status == "production" and s.production_adapter("extraction")["version"] == 1
    gate, status = promote("extraction", 2, base, tuned, store=s)
    assert [a["status"] for a in s.adapters("extraction")] == ["archived", "production"]
    bad = _rep("bad", {"extraction": [1.0] * 40, "general": [0.0] * 40, "safety": [1.0] * 40})
    v3 = s.register_adapter("extraction", "m", "lora", "extraction", "/p3", 10, "h", "c", {})
    assert promote("extraction", v3, base, bad, store=s)[1] == "rejected"
    s.save_eval("e1", "other:v1", "full", {"x": 1})
    assert s.eval_reports("other:v1")[0]["report"] == {"x": 1}


def test_store_runs_registry_and_promotion_sqlite(tmp_path):
    s = Store(dsn="", sqlite_path=tmp_path / "t.db")
    assert s.backend == "sqlite"
    _store_flow(s)


def test_store_runs_registry_and_promotion_postgres_scratch_schema():
    """Same flow against a real Postgres (own throwaway schema, dropped afterwards). Skipped when no server is reachable."""
    import os
    s = Store(schema=f"forgellm_test_{os.getpid()}")
    if s.backend != "postgres":
        pytest.skip("Postgres not reachable")
    try:
        _store_flow(s)
    finally:
        s.drop_schema()


def test_store_falls_back_to_sqlite_when_postgres_unreachable(tmp_path):
    s = Store(dsn="postgresql://nobody:x@127.0.0.1:1/none", sqlite_path=tmp_path / "f.db")
    assert s.backend == "sqlite" and s.error and "postgres unavailable" in s.describe()


# ---- API --------------------------------------------------------------------------------------------------------------
def test_http_api(tiny, tmp_path, trained_router, monkeypatch):
    from forgellm import store as st
    monkeypatch.setattr(st, "_STORE", Store(dsn="", sqlite_path=tmp_path / "api.db"))
    eng = _engine(tiny, tmp_path, trained_router, Retriever(kb.fact_docs(2)))
    c = TestClient(create_app(eng))
    h = c.get("/health").json()
    assert h["status"] == "ok" and set(h["adapters_available"]) == {"technical_reasoning", "tool_use", "extraction"} and h["router_trained"]
    assert c.post("/v1/route", json={"query": "Is it raining in Pune?"}).json()["adapter"] == "tool_use"
    r = c.post("/v1/chat", json={"query": "How busy is GPU 3?", "max_new_tokens": 8})
    assert r.status_code == 200 and r.json()["adapter"] == "tool_use"
    assert c.post("/v1/chat", json={"query": ""}).status_code == 422
    assert c.post("/v1/chat", json={"query": "x", "mode": "adapter:nope"}).status_code == 404
    assert c.post("/v1/adapters/technical_reasoning/load").json()["resident"]
    assert any(a["resident"] for a in c.get("/v1/adapters").json())
    assert c.delete("/v1/adapters/technical_reasoning").status_code == 200 and c.delete("/v1/adapters/technical_reasoning").status_code == 404
    hits = c.get("/v1/rag/search", params={"q": "timeout of the Cobalt service", "k": 2}).json()
    assert hits[0]["id"] == "platform-v2-cobalt"
    m = c.get("/v1/metrics").json()
    assert m["requests"] >= 1 and m["by_adapter"]["tool_use"] >= 1
    assert isinstance(c.get("/v1/registry").json(), list)
    _ = ToolRegistry
