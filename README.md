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
tests/              80 tests (data · models · training · system · real-Postgres registry)
```

## Quickstart

```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -e .        # torch + transformers already installed
.venv/bin/python -m pytest -q                                                    # 80 tests, ~15 s, no GPU/network (the Postgres test skips itself if no server is up)
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

## Results

Everything below was produced by `forgellm experiment` on one laptop (Apple-silicon GPU via MPS, 16 GB, ~1 TFLOP/s effective)
with **Qwen2.5-0.5B-Instruct** in bf16 as the shared frozen base. Honest scale notes: each task suite has **20** test
examples (general = 50 multiple-choice, safety = 26 prompts), one seed, one run — differences of a few points are noise
(the gate and tables below carry bootstrap CIs where it matters). All numbers are re-derivable from `artifacts/reports/*.json`.

### Data pipeline
20,049 raw records → cleaning dropped 561 (215 schema-invalid, 162 boilerplate, 145 mojibake, 39 empty; 161 PII-redacted) →
quality filter dropped 170 → split by prompt hash → dedup removed 5,045 exact + 1,261 near-duplicates from train
(+ 255/37 val, 455/53 test) → decontamination removed 446 training prompts that overlapped val/test/benchmarks (this
includes the 364 eval items deliberately leaked into the pool) → **10,124 train / 614 val / 1,028 test**, 2,500 DPO pairs.

### Router
Held-out test prompts (1,022): **99.4%** task accuracy (per-task recall 1.00 everywhere except `tool_use` 0.96).
Off-domain requests (60 prompts about cooking, travel, science explainers… never seen in training): **77%** fall back to the
base model; the rest are routed to a skill adapter — the router's weakest spot (science "explain X" prompts look like technical Q&A).

### Four skill adapters (LoRA r16 on attention + MLP = 8.8M trainable params = 1.75% of the model, 80–120 steps)

| arm | technical_qa | reasoning | coding | extraction | tool_use | grounded_qa | general | safety |
|---|---|---|---|---|---|---|---|---|
| **base model** | 0.138 | 0.000 | 0.150 | 0.788 | 0.750 | 0.000 | 0.800 | 0.923 |
| `technical_reasoning` | 0.139 | **0.450** | 0.200 | 0.812 | 0.700 | **0.950** | 0.740 | 0.885 |
| `coding` | 0.125 | 0.000 | **0.650** | 0.828 | 0.800 | 0.600 | 0.800 | 1.000 |
| `extraction` | 0.125 | 0.000 | 0.150 | **1.000** | 0.500 | 0.000 | 0.740 | 0.885 |
| `tool_use` | 0.141 | 0.050 | 0.100 | 0.870 | **1.000** | 0.000 | 0.760 | 0.885 |
| `technical_reasoning-lowforget` | 0.117 | **0.600** | 0.200 | 0.725 | 0.650 | **0.950** | 0.760 | 0.923 |
| **routed system** (one base + router) | 0.139 | 0.450 | 0.650 | 1.000 | 1.000 | 0.950 | – | – |

Each adapter lifts its own skill and the routed system reproduces the per-adapter numbers exactly (no routing mistakes on
this sample): reasoning 0 → 0.45 (answer format followed 100% vs 0%), coding pass@1 0.15 → 0.65 (every output compiles),
extraction → 1.000 (strict-JSON validity 0% → 100%), tool calls → 1.000, grounded QA 0 → 0.95 (the base model answers
"[2]" and nothing else). `technical_qa` does **not** improve: 120 steps cover about a quarter of an epoch of that data — enough
to change style (token-F1 0.14 → 0.18) but not to memorise 58 concepts of facts. The off-diagonal cells show why routing
matters: the extraction adapter applied to tool-use requests drops that suite 0.75 → 0.50.

### Regression gate and remediation
The gate compares each adapter to the base with a paired bootstrap, fails on a *significant* drop of a protected capability
(general, safety), on a target skill that does not improve on average, and on held-out perplexity > 1.15×.

| adapter | held-out perplexity ratio | verdict |
|---|---|---|
| `technical_reasoning` | 1.21× | **rejected** |
| `coding` | 1.25× | **rejected** |
| `extraction` | 1.16× | **rejected** (by 0.01) |
| `tool_use` | 1.09× | **production** |
| `technical_reasoning-lowforget` (lr 1e-4, rank 8, 20% replay, 120 steps) | **1.03×** | **production** |

Remediation worked as intended: the gentler retrain of the failing adapter passes the gate *and* is better on its target
(reasoning 0.45 → 0.60). In every case general multiple-choice accuracy moved by at most 3 questions of 50 and none of the
protected-capability drops were statistically significant at this sample size (the gate reports them as warnings with CIs).

### Findings that did not go the expected way
* **Replay did not measurably help.** The same extraction adapter trained with 8% self-distilled replay + safety data vs none:
  perplexity 1.16× vs 1.09×, safety 0.885 vs 0.962, general 0.76 vs 0.76 — within noise, if anything slightly worse with replay
  (120 replay examples at this scale are too few; the lower-LR/lower-rank recipe was what actually reduced forgetting).
* **DPO on synthetic negatives hurt or did nothing.** SFT had already saturated extraction, and the rejected answers (malformed
  JSON, wrong tool, wrong number) are trivially separable — preference accuracy was 100% by step 10. The first run (extraction
  *and* tool-use pairs stacked on the extraction adapter, lr 5e-5) dropped extraction 1.000 → 0.890, tool use to 0.200 and
  safety to 0.769. The corrected run (same-skill pairs only, lr 2e-5, 40 steps, margin 3.7 instead of 7.2) was harmless
  (extraction 0.988, safety 0.885) but gave no gain over SFT. Preference optimisation is only worth it where SFT leaves headroom.
* **Skipped training steps.** Because of the MPS bug below, 2 of 9 runs lost a single step (the other 7 lost none).

### Knowledge vs skills: RAG against fine-tuning (40 facts that changed between platform release v1 and v2)

| setup | current (v2) value correct | stale (v1) value given |
|---|---|---|
| base model, no retrieval | 2.5% | 10% |
| base model + RAG | 2.5% | 2.5% |
| grounding adapter + RAG | **97.5%** | 0% |
| adapter with v1 facts *baked into the weights*, no retrieval | 20% | **30%** |
| baked-in adapter + RAG | 100% | 0% |

Retrieval found the right document for all 40 (recall@3 = 1.0). Baking facts in makes the model confidently wrong when they
change; re-indexing documents fixes it with no training. The base model cannot use retrieved passages by itself — *that*
reading-and-citing behaviour is the skill the adapter teaches.

### Quantization (same base model, weights only)

| precision | weights (MiB) | bits/param | held-out ppl | general MCQ | decode tok/s |
|---|---|---|---|---|---|
| 16-bit | 942 | 16 | 10.97 | 0.80 | 41.6 |
| int8 | 602 | 8 | 11.05 | 0.80 | 11.4 |
| NF4 + double quant | 436 | 4.127 | 12.95 | 0.70 | 6.6 |
| NF4, no double quant | 452 | 4.5 | 12.95 | 0.72 | 6.9 |
| uniform 4-bit | 436 | 4.127 | **15.38** | **0.60** | 7.0 |

NF4 beats uniform 4-bit at identical memory (perplexity 12.95 vs 15.38, SQNR 20.4 vs 19.2 dB); double quantization saves 16 MiB
for no measurable quality change. Decoding is 4–6× slower than 16-bit because dequantization runs in unfused PyTorch each
forward — memory is what is being bought here, not speed.

### PEFT design space (from-scratch 4.4M-parameter TinyGPT, 120 steps, same data; validation loss is the informative column)

| method | trainable (% of model) | val loss | s/step | base weights |
|---|---|---|---|---|
| full fine-tuning | 4,419,840 (100%) | **0.817** | 1.83 | 16.9 MiB |
| LoRA q,v · r8 | 43,008 (0.96%) | 1.961 | 1.72 | 16.9 |
| LoRA q,k,v,o · r8 | 86,016 (1.91%) | 1.533 | 1.54 | 16.9 |
| LoRA attn+MLP · r8 | 221,952 (4.78%) | 1.134 | 1.79 | 16.9 |
| LoRA attn+MLP · r32 | 887,808 (16.7%) | 0.863 | 1.65 | 16.9 |
| LoRA attn+MLP · r8, last 2 layers only | 73,984 (1.65%) | 2.050 | 0.85 | 16.9 |
| DoRA q,k,v,o · r8 | 90,624 (2.01%) | 1.532 | 1.64 | 16.9 |
| **QLoRA** (NF4 base) attn+MLP · r8 | 221,952 (4.78%) | 1.143 | 8.2 | **2.4** |
| Houlsby bottleneck d16 | 101,568 (2.25%) | 2.148 | 1.16 | 16.9 |
| (IA)³ | 5,664 (0.13%) | 3.562 | 1.24 | 16.9 |

More adapted modules and higher rank both lower the loss (placement matters more than rank at a fixed budget: last-2-layers is the
worst per parameter); DoRA ties LoRA at this scale; QLoRA matches LoRA's quality on a base 7× smaller in memory but is ~4.6×
slower per step. The task scores of this tiny model are near zero for every method after 120 steps, and the "general ppl ×"
column in the raw study output is *below* 1 for all rows because the pretrained tiny base had overfit its small corpus
(held-out perplexity 96) — it is not evidence of reduced forgetting, which is why the table shows validation loss.

### End to end (`forgellm ask`, traces from `artifacts/experiment/demo.json`)
* *"What is the request timeout of the Cobalt service?"* → router: grounded_qa (conf 1.00) → `technical_reasoning` adapter loaded in 228 ms
  → BM25 retrieved `platform-v2-cobalt` → "The request timeout of the Cobalt service is 15 seconds [1]." (4.2 s)
* *"How busy is GPU 3 right now?"* → `tool_use` adapter → tools offered by retrieval over tool descriptions → `get_gpu_status({"gpu_id": 3})`
  validated and executed → "GPU 3 is idle: 74% utilization, 21.5 GB memory used, 63°C."
* A coding request → `coding` adapter → a correct function. An off-domain request (birdwatching weekend) → router falls back to the **base model**.
* **A failure, kept on purpose:** a freely-phrased memory-arithmetic question (not shaped like the training templates) sent the reasoning
  adapter into a repetition loop. A 0.5B model fine-tuned on templated word problems generalises to the *templates*, not to arithmetic.

### Engineering notes worth knowing
* **An Apple-MPS backend bug, found and worked around.** Training intermittently returned NaN gradients in *every* adapter tensor
  while the loss was finite. Isolating it took four rounds of measured trials (12–24 backward passes per variant): the cause is
  the allocator state / an unaligned number of rows in the (hidden→151,936-vocab) projection. Fixes now in the code and tests:
  gather supervised rows with `index_select`, pad them to a multiple of 32 before the vocab matmul and slice after (0/24 bad
  trials), and — because it was not fully eliminated — recompute a bad step after clearing the device cache, skipping it only if
  it fails 3 times. Documented in `training/losses.py` and `training/trainer.py`.
* **Memory on a 16 GB machine:** LoRA computes in the activation dtype with fp32 master weights (an fp32 copy of every layer input
  had been kept alive for backward and ran out of memory at 18 GB), gradient checkpointing is on, micro-batch 4 × accumulation 2.
  amp is off: autocast on top of a bf16 base measured 2× slower on MPS.
* `forgellm experiment` is resumable (periodic exact-resume checkpoints; finished steps are skipped after a restart).

### Limitations (read before quoting numbers)
Small test sets (20 per suite) and one seed; the data is procedurally generated, so scores measure skill acquisition on a
well-defined distribution, not open-ended ability; "general" is a 50-question multiple-choice probe plus perplexity on six
passages; `technical_qa` is scored by token overlap, not by a judge; the adapters were trained for 80–120 steps (minutes, not
hours); the TinyGPT study's task scores are uninformative (hence validation loss); quantized decoding is unfused PyTorch.
The tracking store uses Postgres when `FORGELLM_POSTGRES_DSN` is reachable and SQLite otherwise — this run's registry is in
SQLite because the Postgres server was not up when the final steps ran (`forgellm registry --import-sqlite` copies it across).
