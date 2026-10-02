# ForgeLLM — Adaptive LLM Training & Personalization Engine

A modular system that specialises open-source language models for **technical / AI-engineering intelligence** under
constrained compute — and measures, rather than assumes, that the specialisation helped.

It is not "a LoRA notebook". The repository contains the whole loop:

```
 DOMAIN DATA ─► DATA ENGINEERING ─► DATASET BUILDER ─► BASE LLM ─┬─► Full FT
 (6 task families,   schema · clean · MinHash dedup ·     (HF decoder │
  verifiable)        quality · decontamination ·          or TinyGPT) ├─► PEFT: LoRA · DoRA · rsLoRA · LoRA+
                     split · task mixture                             │        QLoRA (NF4) · bottleneck · (IA)³
                                                                      ▼
   SFT ─► DPO / IPO / ORPO / SimPO ─► ADAPTERS (versioned, stackable, hot-swappable)
                                              │
   request ─► TASK ROUTER ─► adapter select ──┤──► RAG grounding (changing knowledge)
                 │  (confidence-gated          ├──► tool selection → validate → execute → observe
                 │   fallback to base)         └──► output parser (JSON / code / answer / citations)
                 ▼
   EVALUATION: domain · reasoning · coding · tool calling · structured output · instruction following
               · general · safety  →  paired-bootstrap REGRESSION GATE  →  MODEL REGISTRY  →  HTTP API
```

**Principle it demonstrates:** *fine-tune behaviour and skills; retrieve knowledge that changes.*
The router sends "what is the request timeout of the Cobalt service?" to retrieval (re-index the docs, no training run),
while "extract these fields as JSON" goes to a LoRA adapter that learned the *skill*. `forgellm study rag-vs-ft` measures
what happens when you bake changing facts into weights instead.

Almost everything is written from scratch in PyTorch so the mechanics are visible and tested: LoRA/DoRA injection and
merge, NF4/int8 quantization with double quantization and a recompute-on-backward dequant kernel, AdamW, LR schedules,
the SFT/DPO/ORPO/SimPO losses, the trainer (accumulation, AMP, clipping, atomic checkpoints, exact resume), a Llama-style
decoder with RoPE + GQA + KV cache, BM25, MinHash-LSH, the router classifier, and the regression statistics.

> **Status.** The code, tests (80) and the experiment driver are complete. The measured results section is being filled in from
> a long-running experiment (`forgellm experiment`); until it lands, the commands below reproduce everything from scratch.

## Layout

```
configs/            model.yaml · lora.yaml · training.yaml · inference.yaml · tiny.yaml
src/forgellm/
  data/             schemas · synth (verifiable data generators) · knowledge · cleaning · quality · dedup ·
                    contamination · mixing · templates (chat format, label masking) · preference · replay · pipeline
  models/           base (BaseModelLoader) · tinygpt · tokenizer · lora (placement, DoRA, multi-adapter layer) ·
                    adapters (bottleneck, IA3, save/load, merge, SVD fusion, AdapterStore) · quantization (NF4/int8)
  training/         trainer · losses · optimizers · schedulers · sft · preference · pretrain
  inference/        engine · generation (KV-cache batching, sampling) · router · parser
  rag/              retriever (BM25 + optional dense + RRF) · context
  tools/            schemas (validator) · registry (8 tools) · execution (timeout, validation)
  evaluation/       suites · benchmarks · bench_data · sandbox · harness · regression
  serving/          api (FastAPI) · schemas
  registry.py · store.py · studies.py · explain.py · cli.py
tests/              76 tests (data · models · training · system)
```

## Quickstart

```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -e .        # torch + transformers already installed
.venv/bin/python -m pytest -q                                                    # 76 tests, ~15 s, no GPU/network
forgellm data build            # raw -> clean -> dedup -> decontaminate -> split -> DPO pairs  (artifacts/data)
forgellm route train           # task router
forgellm replay                # self-distilled general replay (needs the base model)
forgellm experiment            # the whole program below, resumable (hours on a laptop GPU)
```

Individual stages:

```bash
forgellm train sft --adapter extraction --tasks extraction --set lora.placement=attn_mlp lora.rank=16
forgellm train dpo --adapter extraction-dpo --sft extraction --tasks extraction tool_use
forgellm train sft --adapter qlora-demo --set model.quantization=nf4          # QLoRA: NF4 base + LoRA
forgellm eval --adapter extraction -n 30                                       # all capability suites -> artifacts/reports
forgellm compare artifacts/reports/base.json artifacts/reports/extraction.json --targets extraction
forgellm adapters describe                                                     # which modules can carry an adapter
forgellm adapters fuse --items coding=0.5 extraction=0.5 --out coding+extraction --rank 16
forgellm ask "How busy is GPU 3 right now?"                                    # router -> adapter -> tools -> answer
forgellm serve --port 8088                                                     # HTTP API (see /docs)
forgellm explain-step --task extraction                                        # one real training step, every stage printed
forgellm study quant | peft | rag-vs-ft
```

Training on a different backbone is a config change (`model.name`): any Hugging Face decoder with
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` (Qwen, Llama, Mistral, Gemma) works; `model.name: tiny`
uses the from-scratch TinyGPT. On CUDA hardware the same code runs unchanged; NF4/int8 are implemented in PyTorch, so
they also run on Apple-silicon and CPU where bitsandbytes does not.

## How the pieces work

### Data engineering (`forgellm data build`)
Six task families, every one with a *verifiable* ground truth — code is executed, arithmetic is computed, JSON is
parsed against its schema, tool calls are validated and run:

| task | what the model must do | how it is checked |
|---|---|---|
| `technical_qa` | answer concept questions from a 58-concept AI-engineering knowledge base | token-F1 + key-term recall |
| `reasoning` | ML-engineering word problems (LoRA param counts, memory, KV cache, batching, LR warm-up…), step-by-step + `#### answer` | numeric match |
| `coding` | write / trace / fix small Python functions | unit tests in a sandbox subprocess |
| `extraction` | free text → schema-constrained JSON, `null` for absent fields | JSON validity, schema, field accuracy |
| `tool_use` | pick a tool, emit arguments, read the observation, answer — or abstain | name, args, abstention, observation use |
| `grounded_qa` | answer *only* from supplied passages and cite `[n]` (the RAG-grounding skill) | value + citation |

The raw pool is deliberately polluted so each stage has real work: 8% injected defects (duplicates, near-duplicates,
HTML, mojibake, truncated outputs, malformed JSON, boilerplate refusals, phone numbers) and ~2% of eval items *leaked back
into the training pool* (verbatim and lightly paraphrased). Pipeline: schema validation → cleaning/PII redaction → task-aware
quality score (parse the JSON, compile the code, validate the tool call…) → split by *prompt hash* (identical prompts can
never straddle splits) → exact + MinHash-LSH near-dup removal → decontamination of train against val/test **and** the
general/safety benchmark prompts → preference pairs (corrupted-gold rejects: wrong arithmetic, malformed JSON, wrong
tool, buggy code, ungrounded answer). `artifacts/data/report.json` records every count.

Training sequences use one ChatML renderer for both training and serving; loss is masked to the answer tokens (tested).

### PEFT (`models/lora.py`, `models/adapters.py`)
`find_candidate_modules` walks any model and lists the adaptable projections; placements are explicit
(`qv`, `qkvo`, `attn_mlp`, layer ranges such as `last:2`, per-module ranks via `rank_pattern`). An `AdapterLayer` wraps one
frozen linear and holds *several named adapters* at once, so one base model in memory serves every skill:
`W x + Σᵢ wᵢ · Δᵢ(x)`. Supported: LoRA, rsLoRA, DoRA (magnitude/direction split), LoRA+ (B-matrix LR ratio), Houlsby
bottleneck adapters, (IA)³. Adapters save to safetensors + a manifest (targets, ranks, base model, data hash, config hash,
training summary), can be **stacked** (a DPO adapter on top of its SFT adapter), **merged** into the base
(`merge_and_unload`, dequantizing first if the base is quantized) or **fused** with weights via truncated SVD.
`AdapterStore` hot-swaps adapters on a shared base with an LRU residency budget.

### QLoRA (`models/quantization.py`)
NF4 codebook built from normal quantiles (matches the paper's values — tested), blockwise absmax with 64-element blocks,
optional double quantization of the scales (4.127 bits/param, matches the formula — tested), int8 per-channel, and a
uniform-4-bit baseline that shows why NF4 exists. `QuantLinear.backward` recomputes the dequantized weight instead of
saving it. Adapters stay in fp32 over the frozen quantized weights.

### Training (`training/`)
One `Trainer` for SFT, DPO/IPO/ORPO/SimPO and pretraining: token-weighted gradient accumulation (equivalent to the big
batch — tested), AMP, global-norm clipping, NaN/Inf step guard, own AdamW (matches `torch.optim.AdamW` — tested), warm-up +
cosine/linear/WSD schedules, validation + early stopping, atomic checkpoints with exact resume (bit-identical — tested),
gradient checkpointing, length-grouped batching. The (hidden→vocab) projection runs only at supervised positions. For DPO the
reference policy is "SFT adapter on, preference adapter off", so no second model is loaded.

### Routing, RAG, tools (`inference/`, `rag/`, `tools/`)
* **Router** — softmax regression on hashed 1–3-gram features trained from scratch, plus structural rules and a
  domain-vocabulary-coverage feature; an explicit `general` class and a confidence threshold send off-domain requests to the
  bare base model. A wrong skill is worse than no skill.
* **RAG** — BM25 (from scratch) with optional dense embeddings fused by reciprocal rank; the live platform docs are a
  separate index, so changing a fact is a re-index, not a training run.
* **Tools** — 8 tools with JSON-schema-style validation, a safe calculator (AST whitelist), timeouts; the engine selects
  the 4 most relevant tools per request, runs the call → validate → execute → observe loop, and returns a full trace.

### Evaluation & regression protection (`evaluation/`)
Eight suites roll up to the capability axes in the design (domain, reasoning, coding, tool calling, structured output,
instruction following, general, safety). *General* = 50 four-way multiple-choice questions scored by log-likelihood +
held-out perplexity; *safety* = 16 harmful requests that should be refused and 10 sensitive-but-legitimate ones that
should not (over-refusal). `compare_reports` pairs per-item scores and bootstraps the 95% CI of the difference; the **gate**
fails an adapter that does not improve its target skill, or that drops a protected capability (general, safety, other
skills) by more than 3 points or inflates held-out perplexity — and warns on the "domain ↑ / general ↓" pattern.
`promote` records the verdict in the registry (`candidate → production | rejected`).

### Store & registry (`store.py`)
Runs, metrics, adapters (versioned, with lineage hashes) and eval reports go to **Postgres** when
`FORGELLM_POSTGRES_DSN` is set and reachable (own `forgellm` schema, so a shared database is untouched), otherwise to a
local SQLite file — same API either way.
