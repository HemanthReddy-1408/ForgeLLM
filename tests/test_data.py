from __future__ import annotations

import json

from forgellm.data import cleaning, contamination, dedup, mixing, quality, synth
from forgellm.data.pipeline import load_split
from forgellm.data.schemas import Example, validate_record
from forgellm.data.templates import (
    IGNORE,
    SFTDataset,
    build_messages,
    encode_example,
    render_prompt,
)
from forgellm.evaluation import sandbox
from forgellm.models.tokenizer import ByteTokenizer
from forgellm.tools.execution import ToolCall, execute, parse_tool_call
from forgellm.tools.registry import ToolRegistry, safe_eval
from forgellm.tools.schemas import validate_arguments, validate_json_schema


def test_schema_validation_rejects_bad_records():
    assert validate_record({"instruction": "a b c d", "output": "x", "task_type": "coding"}) == []
    assert validate_record({"instruction": "", "output": "x", "task_type": "coding"})
    assert validate_record({"instruction": "q", "output": "x", "task_type": "nope"})
    assert validate_record({"instruction": "q", "output": "x", "task_type": "coding", "quality_score": 3})


def test_every_generator_produces_verifiable_examples():
    for task in synth.GENERATORS:
        exs = synth.generate(task, 40, seed=3)
        assert exs, task
        for e in exs:
            assert e.instruction.strip() and e.output.strip()
            score, feats = quality.quality_score(e)
            assert score >= 0.6, (task, feats, e.output[:80])


def test_coding_references_pass_their_own_tests():
    exs = [e for e in synth.generate("coding", 120, seed=1) if e.meta.get("subtype") in ("write", "fix")]
    jobs = []
    for e in exs[:25]:
        code = e.output.split("```python\n")[1].split("```")[0]
        jobs.append((code, e.meta["entry"], e.meta["tests"]))
    assert all(r["ok"] for r in sandbox.run_many(jobs))


def test_fix_task_buggy_code_fails_its_tests():
    fixes = [e for e in synth.generate("coding", 200, seed=2) if e.meta.get("subtype") == "fix"][:15]
    bad = 0
    for e in fixes:
        code = e.input.split("```python\n")[1].split("```")[0]
        bad += not sandbox.run_tests(code, e.meta["entry"], e.meta["tests"])["ok"]
    assert bad >= len(fixes) - 1   # the injected bug must actually change behaviour


def test_reasoning_answers_are_numerically_consistent():
    for e in synth.generate("reasoning", 80, seed=4):
        assert e.output.rstrip().endswith(f"#### {e.meta['answer']}")


def test_extraction_targets_match_schema():
    for e in synth.generate("extraction", 60, seed=5):
        assert not validate_json_schema(json.loads(e.output), e.meta["schema"])


def test_tool_use_calls_are_valid_and_executable():
    reg = ToolRegistry()
    n = 0
    for e in synth.generate("tool_use", 80, seed=6):
        if e.meta["stage"] == "call":
            call = parse_tool_call(e.output)
            assert call and call.name == e.meta["tool"]
            assert execute(call, reg).ok
            n += 1
    assert n > 10


def test_tool_validation_and_safe_eval():
    reg = ToolRegistry()
    assert not validate_arguments(reg.spec("get_gpu_status"), {"gpu_id": 99}).ok
    assert not validate_arguments(reg.spec("get_gpu_status"), {"gpu_id": "3"}).ok
    assert not execute(ToolCall("calculate", {"expression": "__import__('os').system('x')"}), reg).ok
    assert safe_eval("2 + 3 * 4") == 14
    assert not execute(ToolCall("nope", {}), reg).ok


def test_cleaning_strips_markup_redacts_pii_and_drops_junk():
    ok, status = cleaning.clean_record({"instruction": "<p>What  is   LoRA?</p>", "output": "Low-rank adaptation. Call +1-415-555-0199.", "task_type": "technical_qa"})
    assert ok and ok.instruction == "What is LoRA?" and "[PHONE]" in ok.output and status == "redacted"
    assert cleaning.clean_record({"instruction": "What is x?", "output": "Ã©Ã©", "task_type": "technical_qa"})[1] == "mojibake"
    assert cleaning.clean_record({"instruction": "What is x?", "output": "As an AI language model, I cannot", "task_type": "technical_qa"})[1] == "boilerplate_output"
    keep = cleaning.clean_record({"instruction": "Who?", "output": "Contact a@example.com", "task_type": "technical_qa"})[0]
    assert keep is None or "a@example.com" in keep.output  # allow-listed synthetic domain survives


def test_dedup_exact_and_near():
    def mk(i, q, a="x y z w"):
        return Example(q, a, "technical_qa", quality_score=0.9 - i * 0.01)
    base = "Explain how gradient accumulation works when the batch does not fit in memory on a single GPU today"
    exs = [mk(0, base), mk(1, base.upper()), mk(2, base + "?"), mk(3, "What is a completely unrelated question about RoPE scaling?")]
    out, stats = dedup.dedup(exs)
    assert len(out) == 2 and stats["exact"] + stats["near"] == 2
    assert out[0].quality_score == 0.9   # highest-quality member of the cluster survives


def test_minhash_estimates_jaccard():
    a = dedup.shingles("the quick brown fox jumps over the lazy dog every single day of the week")
    b = dedup.shingles("the quick brown fox jumps over the lazy dog every single day of the year")
    mh = dedup.MinHasher(256)
    est = float((mh.signature(a) == mh.signature(b)).mean())
    assert abs(est - dedup.jaccard(a, b)) < 0.15


def test_contamination_flags_copies_not_templates():
    ev = [Example("Write a Python function `f(nums)` that takes a list of integers and returns the sum of the even numbers in nums.", "x", "coding")]
    idx = contamination.ContaminationIndex([dedup.prompt_key(e) for e in ev])
    assert idx.overlap(ev[0].instruction) == 1.0
    assert idx.overlap("Please " + ev[0].instruction) == 1.0                       # prefix edit
    assert idx.overlap("Write a Python function `f(nums)` that takes a list of integers and returns the sum of the odd numbers in nums.") < 0.9
    assert contamination.ContaminationIndex(["What is LoRA?"]).overlap("What is QLoRA?") == 0.0


def test_split_is_deterministic_and_prompt_stable():
    e = Example("What is LoRA?", "a b c d e", "technical_qa")
    assert mixing.split_of(e) == mixing.split_of(Example("what is lora?", "different answer here", "technical_qa"))
    assert mixing.split_of(Example("q", "a", "coding", meta={"force_split": "train"})) == "train"


def test_pipeline_outputs_are_clean_disjoint_and_leak_free(data_dir):
    train, test = load_split("train", data_dir), load_split("test", data_dir)
    assert len(train) > 500 and len(test) > 50
    tk = {dedup.normalize_key(dedup.prompt_key(e)) for e in train}
    assert not tk & {dedup.normalize_key(dedup.prompt_key(e)) for e in test}, "train/test prompt overlap"
    rep = json.loads((data_dir / "report.json").read_text())
    assert rep["contamination"]["removed"] > 0 and rep["cleaning"].get("mojibake", 0) > 0 and rep["dedup"]["train"]["exact"] > 0
    assert all((e.quality_score or 0) >= 0.6 for e in train)


def test_mixture_respects_proportions(data_dir):
    pool = load_split("train", data_dir)
    w = {"technical_qa": 0.4, "reasoning": 0.3, "extraction": 0.3}
    mix, rep = mixing.build_mixture(pool, w, 300, seed=1)
    assert abs(rep["realised"]["technical_qa"] - 0.4) < 0.02 and set(rep["realised"]) == set(w)


def test_chat_template_masks_prompt_and_supervises_only_the_answer():
    tok = ByteTokenizer()
    ex = Example("What is LoRA?", "Low-rank adaptation.", "technical_qa")
    enc = encode_example(ex, tok, 512)
    n_sup = sum(t != IGNORE for t in enc["labels"])
    assert tok.decode([t for t in enc["labels"] if t != IGNORE], skip_special=False) == "Low-rank adaptation.<|im_end|>"
    assert n_sup == len(tok.encode("Low-rank adaptation.<|im_end|>"))
    assert tok.decode(enc["input_ids"], skip_special=False).startswith("<|im_start|>system\n")
    assert render_prompt(build_messages("technical_qa", "q")).endswith("<|im_start|>assistant\n")
    assert encode_example(ex, tok, 10) is None


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    s = "naïve café — 你好 <|im_start|>user\nhi<|im_end|>"
    assert tok.decode(tok.encode(s), skip_special=False) == s


def test_sft_dataset_tool_final_stage_trains_on_final_turn_only():
    tok = ByteTokenizer()
    ex = next(e for e in synth.generate("tool_use", 100, seed=8) if e.meta["stage"] == "final")
    ds = SFTDataset([ex], tok, 4096)
    sup = tok.decode([t for t in ds[0]["labels"] if t != IGNORE], skip_special=False)
    assert sup == ex.output + "<|im_end|>" and "<tool_response>" in tok.decode(ds[0]["input_ids"], skip_special=False)
