"""Domain knowledge base: AI/ML-engineering concepts (stable knowledge) and an internal-platform fact sheet
(volatile knowledge, versioned v1 -> v2) used to demonstrate *skills vs. knowledge* — fine-tune the former,
retrieve the latter."""

from __future__ import annotations

import random
from dataclasses import dataclass

# key: (name, definition, why, when, pitfall)
_C = [
    ("lora", "LoRA (Low-Rank Adaptation)",
     "LoRA freezes the pretrained weight W and learns a low-rank update ΔW = B·A, where A is r×d_in and B is d_out×r, so only a tiny fraction of parameters is trained.",
     "It cuts trainable parameters and optimizer memory by orders of magnitude while matching full fine-tuning quality on many tasks, and adapters are small enough to store per task.",
     "Use it when you must specialise a large model on limited GPU memory or maintain many task-specific variants of one base model.",
     "A rank that is too low underfits hard tasks, and forgetting to scale by alpha/r or to merge the adapter before export gives silently wrong outputs."),
    ("qlora", "QLoRA",
     "QLoRA keeps the base model frozen in 4-bit NormalFloat (NF4) precision with double quantization and trains higher-precision LoRA adapters on top.",
     "It reduces the memory needed for the frozen weights by about 4x versus fp16, so multi-billion-parameter models fit on a single consumer GPU.",
     "Use it when the model in fp16 does not fit in memory for fine-tuning, accepting slower steps from on-the-fly dequantization.",
     "Merging a QLoRA adapter into the quantized weights loses precision, so merge into a dequantized copy or keep the adapter separate."),
    ("dora", "DoRA (Weight-Decomposed Low-Rank Adaptation)",
     "DoRA splits each weight into a magnitude vector and a direction, and applies LoRA only to the direction while learning the magnitude separately.",
     "Separating magnitude from direction makes the learning pattern closer to full fine-tuning and often improves quality at the same rank.",
     "Use it when plain LoRA at a given rank underperforms and you can afford slightly more compute per step.",
     "It adds a weight-norm computation to every forward pass, which costs extra time and memory compared with plain LoRA."),
    ("full_ft", "full fine-tuning",
     "Full fine-tuning updates every parameter of the pretrained model on the new data.",
     "It offers the most capacity to change model behaviour because no weights are frozen.",
     "Use it when you have ample memory, a large high-quality dataset and need the maximum possible adaptation.",
     "It needs gradients and optimizer states for all parameters, which is roughly 16 bytes per parameter with mixed-precision Adam, and it forgets more easily."),
    ("sft", "supervised fine-tuning (SFT)",
     "SFT continues training a pretrained language model on prompt-response pairs with cross-entropy loss computed only on the response tokens.",
     "It teaches the model a task format and behaviour using demonstrations of the desired output.",
     "Use it as the first adaptation stage before any preference optimisation.",
     "Computing loss on the prompt tokens as well wastes capacity and can teach the model to imitate user text."),
    ("dpo", "Direct Preference Optimization (DPO)",
     "DPO trains a policy directly on preferred and rejected response pairs by increasing the log-probability margin of the chosen response relative to a frozen reference model, without a separate reward model.",
     "It gives preference alignment with a simple supervised-style loss instead of the unstable reinforcement-learning loop of RLHF.",
     "Use it after SFT when you have pairs of better and worse responses for the same prompts.",
     "If the reference model differs from the SFT starting point, or beta is too small, the policy drifts and quality collapses."),
    ("orpo", "ORPO (Odds Ratio Preference Optimization)",
     "ORPO adds an odds-ratio penalty to the normal SFT loss so a single stage both learns the task and prefers chosen over rejected responses, with no reference model.",
     "It removes the reference model and the separate SFT stage, saving memory and time.",
     "Use it when you want preference tuning on a tight memory budget.",
     "The penalty weight lambda must be tuned, because too large a value suppresses likelihood of both responses."),
    ("rlhf", "RLHF (reinforcement learning from human feedback)",
     "RLHF trains a reward model on human preference comparisons and then optimises the language model against it with reinforcement learning such as PPO, with a KL penalty to the SFT model.",
     "It aligns outputs with human preferences that are hard to express as demonstrations.",
     "Use it when you need fine-grained alignment and can afford the reward model and the RL infrastructure.",
     "The policy can exploit weaknesses in the reward model, called reward hacking, if the KL penalty is too weak."),
    ("catastrophic_forgetting", "catastrophic forgetting",
     "Catastrophic forgetting is the loss of previously learned abilities when a model is fine-tuned on a narrow new distribution.",
     "It is the main risk of specialising a general model, so it must be measured explicitly.",
     "Check for it whenever domain accuracy rises, by re-running general and safety benchmarks on the fine-tuned model.",
     "Mitigations such as replaying general data, lowering the learning rate or using low-rank adapters reduce but do not remove it."),
    ("grad_accum", "gradient accumulation",
     "Gradient accumulation sums gradients over several micro-batches before one optimizer step, simulating a larger batch.",
     "It lets a small GPU train with a large effective batch size without extra memory.",
     "Use it when the batch you want does not fit in memory.",
     "The loss must be normalised by the total number of tokens in the whole window, otherwise sequences of different lengths are weighted incorrectly."),
    ("grad_clip", "gradient clipping",
     "Gradient clipping rescales the gradient when its global norm exceeds a threshold, usually 1.0.",
     "It prevents a single bad batch from producing a huge update that destabilises training.",
     "Use it for virtually every transformer training run.",
     "Clipping per parameter instead of by the global norm changes the gradient direction."),
    ("mixed_precision", "mixed-precision training",
     "Mixed precision runs forward and backward computation in a 16-bit format while keeping master weights and optimizer states in 32-bit.",
     "It roughly halves activation memory and speeds up matrix multiplications on modern hardware.",
     "Use it by default on GPUs with tensor cores.",
     "With fp16 the gradients can underflow, which is why loss scaling is required; bf16 avoids this because it has the range of fp32."),
    ("bf16_fp16", "bf16 versus fp16",
     "bf16 has 8 exponent bits and 7 mantissa bits, matching fp32's range, while fp16 has 5 exponent bits and 10 mantissa bits.",
     "bf16 trades precision for range so training rarely overflows, while fp16 has finer precision but a narrow range.",
     "Prefer bf16 when the hardware supports it, and use fp16 with loss scaling otherwise.",
     "Casting a bf16 model to fp16 for inference can overflow activations."),
    ("grad_ckpt", "gradient checkpointing",
     "Gradient checkpointing discards intermediate activations in the forward pass and recomputes them during backward.",
     "It trades about one extra forward pass of compute for a large reduction in activation memory.",
     "Use it when activations, not weights, are what overflow memory, for example with long sequences.",
     "It slows training by roughly 20 to 30 percent and does nothing for weight or optimizer memory."),
    ("adamw", "AdamW",
     "AdamW is Adam with decoupled weight decay: it keeps running averages of the gradient and squared gradient, bias-corrects them, and applies weight decay directly to the weights.",
     "Decoupling weight decay from the adaptive update gives proper regularisation, which L2 regularisation inside Adam does not.",
     "Use it as the default optimizer for transformers.",
     "Applying weight decay to biases and normalisation parameters usually hurts, so exclude them."),
    ("lr_warmup", "learning-rate warmup",
     "Warmup increases the learning rate linearly from near zero to its peak over the first steps of training.",
     "Adaptive optimizers have unreliable second-moment estimates at the start, and warmup avoids large destructive early updates.",
     "Use a warmup of a few percent of total steps, followed by a cosine or linear decay.",
     "Skipping warmup with a high peak learning rate is a common cause of early divergence."),
    ("cosine_schedule", "cosine learning-rate schedule",
     "A cosine schedule decays the learning rate from its peak to a minimum following half a cosine curve.",
     "It decays smoothly, spending time at both high and low learning rates, which tends to give good final loss.",
     "Use it when the total number of steps is known in advance.",
     "If training is cut short, the learning rate has not yet decayed and the checkpoint is worse than expected."),
    ("kv_cache", "the KV cache",
     "The KV cache stores the key and value tensors of all previous tokens so each decoding step only computes attention for the new token.",
     "It turns the cost of generating each token from quadratic in the sequence length to linear.",
     "It is used in all autoregressive inference, and its size is 2 times layers times KV heads times head dimension times sequence length times bytes per element.",
     "At long contexts and large batches the cache can exceed the model weights in memory."),
    ("flash_attention", "FlashAttention",
     "FlashAttention computes exact attention in tiles that stay in fast on-chip SRAM, never materialising the full attention matrix in GPU memory.",
     "It makes attention memory linear in sequence length and much faster because it is bound by memory traffic rather than arithmetic.",
     "Use it for training or inference with long sequences.",
     "It is exact, not an approximation, but it requires supported hardware and dtypes."),
    ("rope", "rotary position embeddings (RoPE)",
     "RoPE encodes position by rotating query and key vectors by an angle proportional to token position, so attention scores depend on relative distance.",
     "It encodes relative position without learned position tables and extends more gracefully to longer contexts.",
     "It is the position scheme used by Llama, Qwen and Mistral models.",
     "Running a model far beyond its trained context without rescaling the rotation frequencies degrades quality."),
    ("gqa", "grouped-query attention (GQA)",
     "GQA shares each key and value head across a group of query heads, so there are fewer KV heads than query heads.",
     "It shrinks the KV cache and memory bandwidth at inference with little loss of quality compared with full multi-head attention.",
     "Use it when serving long contexts or large batches.",
     "Converting a multi-head model to GQA requires some additional training to recover quality."),
    ("rmsnorm", "RMSNorm",
     "RMSNorm normalises activations by their root mean square and scales them with a learned gain, without subtracting the mean.",
     "It is cheaper than LayerNorm and works as well in transformers.",
     "It is the normalisation used in Llama-style decoder models.",
     "The computation should be done in float32, because computing the mean of squares in fp16 can overflow."),
    ("swiglu", "SwiGLU",
     "SwiGLU is a feed-forward block that multiplies a SiLU-gated projection with a linear projection before the down-projection, using gate, up and down matrices.",
     "Gated activations improve quality over a plain ReLU or GELU feed-forward at equal parameter count.",
     "It is the MLP design used in Llama and Qwen models, which is why those models have gate_proj, up_proj and down_proj layers.",
     "Because there are three matrices instead of two, the hidden size is reduced to about two thirds to keep parameters comparable."),
    ("bpe", "byte-pair encoding (BPE)",
     "BPE builds a subword vocabulary by repeatedly merging the most frequent adjacent symbol pair in the training corpus.",
     "It represents any text with a bounded vocabulary while keeping common words as single tokens.",
     "It is the tokenisation used by almost all modern language models.",
     "Tokenisation differences, such as how whitespace or numbers split, can silently change model behaviour on code and arithmetic."),
    ("perplexity", "perplexity",
     "Perplexity is the exponential of the average per-token cross-entropy loss, measured in nats.",
     "It summarises how surprised a language model is by held-out text, with lower being better.",
     "Use it to compare models that share a tokenizer on the same text.",
     "It cannot be compared across different tokenizers and does not measure instruction-following or factuality."),
    ("nf4", "NormalFloat 4-bit (NF4)",
     "NF4 is a 4-bit data type whose 16 levels are placed at quantiles of a normal distribution, so it matches how pretrained weights are distributed.",
     "Weights are roughly normal, so quantile-spaced levels waste fewer codes than uniform 4-bit levels and lose less accuracy.",
     "Use it for quantizing frozen weights in QLoRA.",
     "Each block of weights needs its own scale constant, so the effective storage is a little above 4 bits per weight."),
    ("double_quant", "double quantization",
     "Double quantization quantizes the per-block scaling constants themselves to 8 bits.",
     "It saves about 0.37 bits per parameter, which matters for models with billions of weights.",
     "Use it together with blockwise 4-bit quantization when memory is the constraint.",
     "It adds a small dequantization overhead and another place for rounding error."),
    ("quantization", "weight quantization",
     "Quantization stores weights with fewer bits, such as 8-bit or 4-bit integers, and dequantizes them to higher precision for computation.",
     "It reduces memory and bandwidth, which are the bottleneck of LLM inference.",
     "Use it for deployment when a small quality loss is acceptable for a large saving in memory and latency.",
     "Outlier weights and activations cause the largest errors, and quality should always be re-measured after quantizing."),
    ("rag", "retrieval-augmented generation (RAG)",
     "RAG retrieves relevant documents at query time and places them in the prompt so the model grounds its answer in them.",
     "It gives the model access to knowledge that is private, large or changing without retraining.",
     "Use it for facts that change or must be cited, and fine-tune only for behaviour and skills.",
     "If retrieval returns irrelevant passages the model may still answer confidently, so retrieval quality needs its own evaluation."),
    ("bm25", "BM25",
     "BM25 is a sparse lexical ranking function that scores documents by term frequency with saturation, inverse document frequency and length normalisation.",
     "It is fast, needs no training and is strong on exact keywords, identifiers and rare terms.",
     "Use it as a baseline retriever or combine it with dense retrieval.",
     "It cannot match synonyms or paraphrases with no shared words."),
    ("dense_retrieval", "dense retrieval",
     "Dense retrieval embeds queries and documents into vectors and ranks documents by similarity such as cosine distance.",
     "It captures meaning, so it finds relevant text even without shared keywords.",
     "Use it for natural-language questions over paraphrased content, ideally fused with BM25.",
     "It can miss exact identifiers, error codes and rare names that a lexical retriever finds easily."),
    ("hybrid_search", "hybrid retrieval with rank fusion",
     "Hybrid retrieval runs lexical and dense retrievers and merges their ranked lists, commonly with reciprocal rank fusion.",
     "The two retrievers fail on different queries, so fusing them is more robust than either alone.",
     "Use it when queries mix natural language with exact identifiers.",
     "Fusing raw scores from different scales is unreliable, so fuse ranks instead."),
    ("hallucination", "hallucination",
     "Hallucination is when a language model generates fluent content that is unsupported by its input or by facts.",
     "It limits trust in model outputs and must be measured and reduced.",
     "Reduce it with retrieval grounding, citations, verification and training the model to abstain.",
     "Fine-tuning on facts the base model does not know can increase hallucination rather than reduce it."),
    ("prompt_injection", "prompt injection",
     "Prompt injection is when untrusted text in the input, such as a web page or tool result, contains instructions that hijack the model's behaviour.",
     "It is the main security risk of models that read external content or call tools.",
     "Defend against it by separating trusted instructions from data, validating tool calls and limiting tool permissions.",
     "Filtering for known phrases is easy to bypass, so permissions and validation must not depend on the model's judgement alone."),
    ("speculative_decoding", "speculative decoding",
     "Speculative decoding has a small draft model propose several tokens that the large model then verifies in a single forward pass.",
     "It speeds up generation without changing the output distribution because rejected tokens are resampled.",
     "Use it when latency matters and a well-matched small model is available.",
     "A poorly matched draft model has a low acceptance rate and can make decoding slower."),
    ("continuous_batching", "continuous batching",
     "Continuous batching lets new requests join and finished requests leave a running batch at every decoding step.",
     "It keeps the GPU busy and raises throughput compared with waiting for a whole batch to finish.",
     "Use it in any multi-user serving system.",
     "Long and short requests share the batch, so scheduling must prevent a long request from starving others."),
    ("fsdp", "FSDP and ZeRO",
     "FSDP and ZeRO shard parameters, gradients and optimizer states across data-parallel workers and gather them only when needed.",
     "They cut per-GPU memory roughly in proportion to the number of GPUs, enabling models too large for one device.",
     "Use them when full fine-tuning does not fit on a single GPU.",
     "The extra all-gather communication slows training on slow interconnects."),
    ("tensor_parallel", "tensor parallelism",
     "Tensor parallelism splits individual weight matrices across GPUs so each computes part of every layer.",
     "It lets a single layer's weights and compute exceed one GPU and reduces latency.",
     "Use it inside a fast node, typically with NVLink.",
     "It requires communication at every layer, so it scales poorly across slow links."),
    ("dedup", "data deduplication",
     "Deduplication removes exact and near-duplicate examples from a training set, for example with MinHash and locality-sensitive hashing.",
     "Duplicates waste compute, bias the model towards repeated text and increase memorisation and test leakage.",
     "Run it before splitting data and again against evaluation sets.",
     "A similarity threshold that is too low removes legitimately distinct examples."),
    ("contamination", "benchmark contamination",
     "Contamination is when evaluation examples, or near copies of them, appear in the training data.",
     "It inflates scores and makes fine-tuning appear more effective than it is.",
     "Check for it with n-gram overlap between every training example and every evaluation item.",
     "Paraphrased copies evade exact matching, so fuzzy matching is required."),
    ("minhash", "MinHash",
     "MinHash estimates the Jaccard similarity of two sets by comparing the minimum hash values of each set under many hash functions.",
     "It compresses each document into a short signature so near-duplicates can be found without comparing every pair.",
     "Use it with banding (LSH) to find near-duplicate text at scale.",
     "The number of permutations and bands controls the trade-off between false positives and missed duplicates."),
    ("label_masking", "label masking",
     "Label masking sets the target of prompt and padding tokens to an ignore index so only response tokens contribute to the loss.",
     "It makes the model learn to produce answers rather than to imitate the instructions.",
     "Use it in every instruction-tuning run.",
     "A misaligned mask, for example off by one token around the assistant header, silently trains on the wrong tokens."),
    ("packing", "sequence packing",
     "Packing concatenates several short examples into one long sequence to remove padding.",
     "It improves token throughput when example lengths vary widely.",
     "Use it when padding wastes a large share of each batch.",
     "Without a block-diagonal attention mask, examples attend to each other and contaminate one another."),
    ("scaling_laws", "scaling laws",
     "Scaling laws describe how loss falls predictably as a power law in model size, data and compute.",
     "They let you plan the compute budget and choose model and data size before running expensive training.",
     "Use them to decide whether to spend extra compute on a bigger model or more tokens.",
     "They predict pretraining loss, not downstream task quality after fine-tuning."),
    ("moe", "mixture of experts (MoE)",
     "A mixture-of-experts layer routes each token to a few of many expert feed-forward networks chosen by a learned gate.",
     "It increases total parameters, and hence capacity, without increasing compute per token.",
     "Use it when you want capacity beyond what dense compute budgets allow.",
     "All experts must be held in memory, and unbalanced routing leaves some experts under-trained."),
    ("distillation", "knowledge distillation",
     "Distillation trains a small student model to match the output distribution or generated responses of a larger teacher.",
     "It transfers much of a large model's capability into something cheaper to serve.",
     "Use it when serving cost or latency of the large model is prohibitive.",
     "The student inherits the teacher's errors and cannot exceed it on the distilled distribution."),
    ("embeddings", "text embeddings",
     "Text embeddings are dense vectors produced by an encoder so that semantically similar texts lie close together.",
     "They enable semantic search, clustering and routing with simple vector similarity.",
     "Use them for retrieval, deduplication and classifying queries.",
     "Embeddings from different models are not comparable, so the same model must embed queries and documents."),
    ("cross_entropy", "cross-entropy loss",
     "Cross-entropy loss for language modelling is the negative log-probability the model assigns to the correct next token, averaged over tokens.",
     "Minimising it maximises the likelihood of the training text.",
     "It is the loss used for pretraining and supervised fine-tuning.",
     "Averaging per sequence instead of per token changes how much long examples count."),
    ("overfitting", "overfitting",
     "Overfitting is when a model fits the training data, including noise, so that its validation performance gets worse while training loss keeps falling.",
     "Detecting it is the reason a validation set exists.",
     "Watch the gap between training and validation loss and stop early when it widens.",
     "Fine-tuning a large model for many epochs on a small dataset overfits quickly."),
    ("early_stopping", "early stopping",
     "Early stopping halts training when validation loss has not improved for a set number of evaluations.",
     "It prevents overfitting and saves compute.",
     "Use it together with checkpointing so the best checkpoint is kept.",
     "A patience that is too small stops training at a noisy plateau."),
    ("eval_harness", "an evaluation harness",
     "An evaluation harness runs fixed benchmarks on a model with fixed prompts and scoring so that results are comparable across runs.",
     "It turns claims about improvement into measurements.",
     "Run it on the base and fine-tuned model with identical settings, for domain, general and safety capabilities.",
     "Changing prompt format or decoding settings between runs makes the comparison meaningless."),
    ("llm_judge", "LLM-as-judge",
     "LLM-as-judge uses a strong language model to grade or compare the outputs of other models.",
     "It scales evaluation of open-ended answers that have no single correct string.",
     "Use it to supplement, not replace, programmatic checks such as exact match and unit tests.",
     "Judges prefer longer answers and their own style, so randomise order and calibrate against human labels."),
    ("constrained_decoding", "constrained decoding",
     "Constrained decoding masks the model's next-token logits so that only tokens consistent with a grammar or JSON schema can be generated.",
     "It guarantees syntactically valid structured output regardless of model quality.",
     "Use it when downstream code requires valid JSON.",
     "It guarantees syntax, not correctness, and a very restrictive grammar can force wrong values."),
    ("function_calling", "function calling",
     "Function calling is when a model emits a structured request naming a tool and its arguments, which the application executes and returns as an observation.",
     "It lets a model act on external systems and fetch live data.",
     "Fine-tune for it when you need reliable tool selection and argument formatting from a small model.",
     "Executing model-generated arguments without validation is a security hole."),
    ("adapter_routing", "adapter routing",
     "Adapter routing selects a task-specific adapter for each request and applies it to one shared frozen base model.",
     "It provides several specialisations without keeping a separate full model per task.",
     "Use it when one deployment must serve several distinct skills such as coding, extraction and tool use.",
     "A wrong routing decision applies the wrong skill, so the router needs a confidence threshold and a base-model fallback."),
    ("model_registry", "a model registry",
     "A model registry records each model or adapter version with its training data, configuration, metrics and deployment status.",
     "It makes models reproducible and lets you promote or roll back versions safely.",
     "Gate promotion on evaluation results, including regression checks.",
     "Without data and config lineage, a good model cannot be reproduced."),
    ("weight_decay", "weight decay",
     "Weight decay shrinks weights towards zero a little on every step.",
     "It regularises the model and discourages large weights.",
     "Use a small value, such as 0.01 to 0.1, for full fine-tuning.",
     "For LoRA adapters it is often set to zero because the adapters are already small."),
    ("rank_choice", "choosing the LoRA rank",
     "The LoRA rank r sets how many directions the weight update can span, and the adapter has r times (d_in plus d_out) parameters per adapted matrix.",
     "A higher rank has more capacity but costs more memory and risks overfitting.",
     "Start with 8 to 16 for most tasks and raise it only if validation loss stays high.",
     "Raising rank without scaling alpha changes the effective update scale alpha over r."),
]


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    definition: str
    why: str
    when: str
    pitfall: str

    def doc(self) -> str:
        return f"{self.name}. {self.definition} {self.why} {self.when} {self.when and ''}Pitfall: {self.pitfall}".replace("  ", " ")


CONCEPTS: list[Concept] = [Concept(*c) for c in _C]
CONCEPT_BY_KEY = {c.key: c for c in CONCEPTS}

COMPARISONS = [
    ("lora", "full_ft", "LoRA trains a small low-rank update on a frozen model, which needs far less memory and gives small per-task adapters, while full fine-tuning updates every weight, which has the most capacity but needs gradients and optimizer states for all parameters."),
    ("lora", "qlora", "LoRA keeps the frozen base in 16-bit precision, while QLoRA stores the frozen base in 4-bit NF4, cutting weight memory by about 4x at the cost of slower dequantizing steps."),
    ("lora", "dora", "LoRA learns a low-rank update to the whole weight, while DoRA separates magnitude from direction and applies the low-rank update only to the direction, which often gives better quality at the same rank."),
    ("dpo", "rlhf", "DPO optimises preferences directly with a supervised-style loss against a frozen reference model, while RLHF trains a separate reward model and runs a reinforcement-learning loop, which is more complex and less stable."),
    ("dpo", "orpo", "DPO needs a reference model and an earlier SFT stage, while ORPO folds an odds-ratio preference penalty into the SFT loss and needs no reference model."),
    ("sft", "dpo", "SFT imitates demonstrations of good responses, while DPO uses pairs of better and worse responses to push the model away from bad behaviour."),
    ("rag", "full_ft", "RAG supplies knowledge at query time and is easy to update, while fine-tuning bakes behaviour into the weights; use RAG for changing facts and fine-tuning for skills and format."),
    ("bf16_fp16", "mixed_precision", "bf16 and fp16 are the 16-bit formats, whereas mixed precision is the training technique that combines a 16-bit compute format with 32-bit master weights."),
    ("bm25", "dense_retrieval", "BM25 matches exact terms and rare identifiers without training, while dense retrieval matches meaning through embeddings and handles paraphrases but can miss exact identifiers."),
    ("flash_attention", "gqa", "FlashAttention changes how exact attention is computed to save memory traffic, while grouped-query attention changes the architecture by sharing KV heads to shrink the KV cache."),
    ("gradient_placeholder", "", ""),
]
COMPARISONS = [c for c in COMPARISONS if c[1]]
COMPARISONS += [
    ("grad_accum", "grad_ckpt", "Gradient accumulation fakes a large batch by summing gradients over micro-batches, while gradient checkpointing saves activation memory by recomputing activations during backward; they address different memory limits."),
    ("quantization", "distillation", "Quantization shrinks the same model by storing weights in fewer bits, while distillation trains a different, smaller model to imitate a larger one."),
    ("rope", "gqa", "RoPE is a way to encode token positions inside attention, while GQA reduces the number of key and value heads; both appear in Llama-style models but solve different problems."),
    ("dedup", "contamination", "Deduplication removes repeated examples within the training data, while contamination checks look for overlap between training data and evaluation sets."),
]


def concept_doc(c: Concept) -> str:
    return f"{c.name}: {c.definition} {c.why} {c.when} Common pitfall: {c.pitfall}"


# ----------------------------------------------------------------------------------------------------
# Volatile platform knowledge: facts that change between releases. v2 changes ~40% of v1's values.
# ----------------------------------------------------------------------------------------------------
SERVICES = ["Orion", "Helios", "Atlas-Gateway", "Nimbus", "Quark", "Vega", "Lumen", "Cobalt", "Tessera",
            "Aurora", "Basalt", "Cascade", "Drift", "Ember", "Flux", "Granite", "Harbor", "Ion"]
PARAMS = {
    "default_lora_rank": ("default LoRA rank", lambda r: r.choice([4, 8, 16, 32, 64]), ""),
    "max_context_tokens": ("maximum context length", lambda r: r.choice([2048, 4096, 8192, 16384, 32768]), " tokens"),
    "request_timeout": ("request timeout", lambda r: r.choice([10, 15, 30, 45, 60, 120]), " seconds"),
    "max_batch_size": ("maximum batch size", lambda r: r.choice([8, 16, 32, 64, 128]), ""),
    "retention_days": ("log retention", lambda r: r.choice([7, 14, 30, 60, 90]), " days"),
    "replica_count": ("default replica count", lambda r: r.choice([1, 2, 3, 4, 6, 8]), ""),
}


def build_fact_sheet(version: int, seed: int = 1234) -> dict[tuple[str, str], str]:
    """(service, param) -> value-string. v2 differs from v1 for a deterministic ~40% of entries."""
    base = random.Random(seed)
    v1 = {(s, p): spec[1](base) for s in SERVICES for p, spec in PARAMS.items()}
    if version == 1:
        return {k: f"{v}{PARAMS[k[1]][2]}" for k, v in v1.items()}
    ch = random.Random(seed + 99)
    out = {}
    for k, v in v1.items():
        nv = v
        if ch.random() < 0.4:
            while nv == v:
                nv = PARAMS[k[1]][1](ch)
        out[k] = f"{nv}{PARAMS[k[1]][2]}"
    return out


def fact_sentence(service: str, param: str, value: str, version: int) -> str:
    label = PARAMS[param][0]
    return f"In platform release v{version}, the {label} of the {service} service is {value}."


def fact_docs(version: int) -> list[dict[str, str]]:
    """One short document per (service) listing all its parameters — the volatile knowledge corpus for RAG."""
    sheet = build_fact_sheet(version)
    docs = []
    for s in SERVICES:
        lines = [fact_sentence(s, p, sheet[(s, p)], version) for p in PARAMS]
        docs.append({"id": f"platform-v{version}-{s.lower()}", "title": f"{s} service configuration (v{version})",
                     "text": f"{s} service configuration, platform release v{version}. " + " ".join(lines)})
    return docs


def concept_docs() -> list[dict[str, str]]:
    return [{"id": f"kb-{c.key}", "title": c.name, "text": concept_doc(c)} for c in CONCEPTS]
