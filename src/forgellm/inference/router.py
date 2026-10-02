"""Task router: query -> (task, adapter, retrieve?) with a confidence-gated fallback to the bare base model.

Two signals are combined:
  * rules   — high-precision structural cues (passages present, code fences, "extract ... JSON", tool verbs)
  * learned — a softmax-regression classifier over hashed word 1-3-grams, trained from scratch on the instruction data
Low confidence => no adapter (the base model answers), because a wrong skill is worse than no skill."""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forgellm.data import knowledge as kb
from forgellm.data.schemas import TASK_TO_ADAPTER, TASK_TYPES, Example
from forgellm.data.synth import ood_prompts

N_FEATURES = 1 << 15
_WORD = re.compile(r"[a-z0-9_]+")


def featurize(text: str) -> dict[int, float]:
    toks = _WORD.findall(text.lower())
    feats: dict[int, float] = {}
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            h = zlib.crc32((f"{n}|" + " ".join(toks[i : i + n])).encode()) % N_FEATURES
            feats[h] = feats.get(h, 0.0) + 1.0
    norm = sum(v * v for v in feats.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in feats.items()}


_GENERIC = set(["a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "was", "were", "it", "its", "this", "that", "these", "those", "with", "as", "at", "by", "from", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does", "did", "can", "could", "should", "would", "will", "may", "might", "i", "me", "my", "you", "your", "we", "our", "they", "their", "he", "she", "give", "gave", "tell", "write", "explain", "define", "describe", "briefly", "short", "please", "help", "need", "want", "like", "get", "make", "some", "any", "all", "about", "into", "than", "then", "so", "if", "not", "no", "yes", "more", "most", "very", "just", "also", "one", "two", "three", "first", "second", "quick", "question", "hey", "hi", "hello", "thanks", "thank", "good", "morning", "use", "used", "using", "used", "example", "examples", "number", "numbers", "value", "values", "name", "names", "result", "results", "find", "show", "list", "check", "status", "current", "latest", "top", "step", "steps", "json", "object", "field", "fields", "text", "return", "returns", "only", "string", "integer", "following"])


def _structure_tokens(query: str, context: str, domain_vocab: frozenset[str] = frozenset()) -> str:
    """Cheap structural markers appended to the text so the classifier can use them."""
    marks = []
    if "```" in query or "```" in context:
        marks.append("__codefence__")
    if re.search(r"\bdef \w+\(", query + context):
        marks.append("__hasdef__")
    if "passages:" in (query + context).lower() or re.search(r"^\[\d+\] ", context, re.M):
        marks.append("__passages__")
    if context and not marks:
        marks.append("__hascontext__")
    if domain_vocab:   # how much of the query is domain vocabulary? zero hits is the strongest out-of-domain signal
        hits = len({t for t in _WORD.findall(query.lower()) if t in domain_vocab})
        marks.append(f"__dom{min(hits, 3)}__")
    return " ".join(marks)


@dataclass
class RouteDecision:
    task: str
    adapter: str | None
    confidence: float
    use_rag: bool
    reason: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "adapter": self.adapter, "confidence": round(self.confidence, 3), "use_rag": self.use_rag,
                "reason": self.reason, "scores": {k: round(v, 3) for k, v in self.scores.items()}}


RULES: list[tuple[str, str, float]] = [  # (task, regex, weight added to that task's probability mass)
    ("grounded_qa", r"^\[1\] |passages:", 0.7),                                  # the caller already supplied passages
    ("grounded_qa", r"\b(service|release v?\d|configuration|config)\b.*\b(default|maximum|timeout|retention|replica|batch size|context length|lora rank)\b", 0.5),
    ("grounded_qa", r"\b(default|maximum|timeout|retention|replica|context length|lora rank|batch size)\b.*\bservice\b", 0.5),
    ("tool_use", r"\b(weather|gpu \d|gpu number|schedule|queue a job|latest (loss|accuracy|f1|perplexity)|convert \d|search (our )?doc|find papers|search for papers|status of gpu)\b", 0.45),
    ("tool_use", r"^(hello|hi|thanks|thank you|good morning)\b", 0.3),
    ("extraction", r"\b(extract|return only a json|json object)\b", 0.6),
    ("coding", r"(\bpython function\b|\bbug\b|```|\bdef \w+\(|returns? for this function)", 0.5),
    ("reasoning", r"\b(how many|what is the (effective|learning rate|perplexity)|how much|compute|gb|mib|per parameter)\b.*\d", 0.3),
]


class SoftmaxClassifier:
    """Multinomial logistic regression on sparse hashed features, trained with Adam (all from scratch)."""

    def __init__(self, classes: list[str]) -> None:
        self.classes = classes
        self.W = torch.zeros(N_FEATURES, len(classes))
        self.b = torch.zeros(len(classes))

    @staticmethod
    def _sparse(rows: list[dict[int, float]]) -> torch.Tensor:
        idx_r, idx_c, vals = [], [], []
        for r, f in enumerate(rows):
            for c, v in f.items():
                idx_r.append(r)
                idx_c.append(c)
                vals.append(v)
        return torch.sparse_coo_tensor([idx_r, idx_c], vals, (len(rows), N_FEATURES), check_invariants=False).coalesce()

    def fit(self, rows: list[dict[int, float]], labels: list[int], epochs: int = 60, lr: float = 0.5, l2: float = 1e-5) -> list[float]:
        X = self._sparse(rows)
        y = torch.tensor(labels)
        W = self.W.clone().requires_grad_(True)
        b = self.b.clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=lr)
        losses = []
        for _ in range(epochs):
            opt.zero_grad()
            logits = torch.sparse.mm(X, W) + b
            loss = torch.nn.functional.cross_entropy(logits, y) + l2 * W.pow(2).sum()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.W, self.b = W.detach(), b.detach()
        return losses

    def proba(self, feats: dict[int, float]) -> np.ndarray:
        if not feats:
            return np.full(len(self.classes), 1 / len(self.classes))
        idx = torch.tensor(list(feats))
        v = torch.tensor(list(feats.values()))
        logits = (self.W[idx] * v[:, None]).sum(0) + self.b
        return torch.softmax(logits, -1).numpy()


class TaskRouter:
    def __init__(self, threshold: float = 0.55, classes: list[str] | None = None) -> None:
        self.classes = classes or [*TASK_TYPES, "general"]
        self.clf = SoftmaxClassifier(self.classes)
        self.threshold = threshold
        self.trained = False
        self.domain_vocab: frozenset[str] = frozenset()

    # -- training ---------------------------------------------------------------------------------------
    def text_of(self, query: str, context: str = "") -> str:
        return f"{query} {_structure_tokens(query, context, self.domain_vocab)} {context[:160] if context else ''}"

    def fit(self, examples: list[Example], max_per_task: int = 1500) -> dict[str, Any]:
        per: dict[str, int] = {}
        rows, labels = [], []
        df: dict[str, int] = {}
        for ex in examples:
            if ex.task_type in self.classes:
                for t in set(_WORD.findall(ex.instruction.lower())):
                    df[t] = df.get(t, 0) + 1
        ood_tokens = {t for q in ood_prompts(500) for t in _WORD.findall(q.lower())}
        self.domain_vocab = frozenset(t for t, c in df.items() if c >= 3 and t not in _GENERIC and t not in ood_tokens and not t.isdigit())
        for ex in examples:
            if ex.task_type not in self.classes or per.get(ex.task_type, 0) >= max_per_task:
                continue
            if ex.meta.get("stage") == "abstain" and ":knowledge:" in str(ex.meta.get("group", "")):
                continue  # "What is X?" with tools on offer is a technical question; labelling it tool_use would be noise
            per[ex.task_type] = per.get(ex.task_type, 0) + 1
            # at serve time the router sees the query only for grounded QA (retrieval supplies the passages later)
            ctx = ex.input if ex.task_type in ("extraction", "coding") else ""
            rows.append(featurize(self.text_of(ex.instruction, ctx)))
            labels.append(self.classes.index(ex.task_type))
        if "general" in self.classes:   # explicit out-of-domain class: unrelated requests go to the bare base model
            for q in ood_prompts(500):
                rows.append(featurize(self.text_of(q)))
                labels.append(self.classes.index("general"))
            per["general"] = 500
        losses = self.clf.fit(rows, labels)
        self.trained = True
        return {"examples": len(rows), "final_loss": round(losses[-1], 4), "per_task": per}

    # -- inference --------------------------------------------------------------------------------------
    def route(self, query: str, context: str = "") -> RouteDecision:
        p = self.clf.proba(featurize(self.text_of(query, context))) if self.trained else np.full(len(self.classes), 1 / len(self.classes))
        scores = {c: float(p[i]) for i, c in enumerate(self.classes)}
        reason = "classifier"
        low = (query + " " + context).lower()
        for task, rx, w in RULES:
            if task in scores and re.search(rx, low, re.S | re.M):
                scores[task] += w
                reason = "classifier+rules"
        z = sum(scores.values())
        scores = {k: v / z for k, v in scores.items()}
        task = max(scores, key=lambda k: scores[k])
        conf = scores[task]
        if task == "general":
            return RouteDecision("general", None, conf, False, "out-of-domain -> base model", scores)
        if conf < self.threshold:
            return RouteDecision("general", None, conf, False, f"low confidence ({conf:.2f} < {self.threshold}) -> base model", scores)
        return RouteDecision(task, TASK_TO_ADAPTER[task], conf, self.wants_retrieval(task, query, context), reason, scores)

    @staticmethod
    def wants_retrieval(task: str, query: str, context: str) -> bool:
        """Volatile platform knowledge is retrieved; stable skills are not. Never retrieve when passages were supplied."""
        if context and "[1]" in context:
            return False
        if task == "grounded_qa":
            return True
        q = query.lower()
        return any(s.lower() in q for s in kb.SERVICES) and "service" in q

    # -- persistence ------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"W": self.clf.W, "b": self.clf.b, "classes": self.classes, "threshold": self.threshold,
                    "domain_vocab": sorted(self.domain_vocab)}, path)

    @classmethod
    def load(cls, path: str | Path) -> TaskRouter:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        r = cls(blob["threshold"], blob["classes"])
        r.clf.W, r.clf.b, r.trained = blob["W"], blob["b"], True
        r.domain_vocab = frozenset(blob.get("domain_vocab", []))
        return r


def evaluate_router(router: TaskRouter, examples: list[Example], ood_queries: list[str] | None = None) -> dict[str, Any]:
    """Accuracy at the adapter level (what actually matters), per-task recall, confusion matrix and OOD fallback rate."""
    tasks = [c for c in router.classes if c != "general"]
    conf = {t: dict.fromkeys([*tasks, "general"], 0) for t in tasks}
    correct = adapter_ok = total = 0
    for ex in examples:
        if ex.task_type not in tasks:
            continue
        ctx = ex.input if ex.task_type in ("extraction", "grounded_qa", "coding") else ""
        if ex.task_type == "grounded_qa":
            ctx = ""   # at serve time users do not paste passages; retrieval supplies them
        d = router.route(ex.instruction, ctx)
        conf[ex.task_type][d.task] += 1
        total += 1
        correct += d.task == ex.task_type
        adapter_ok += d.adapter == TASK_TO_ADAPTER[ex.task_type]
    per_task = {t: round(conf[t][t] / max(1, sum(conf[t].values())), 3) for t in tasks}
    out: dict[str, Any] = {"n": total, "task_accuracy": round(correct / max(1, total), 4),
                           "adapter_accuracy": round(adapter_ok / max(1, total), 4), "per_task_recall": per_task, "confusion": conf}
    if ood_queries:
        fb = sum(router.route(q).adapter is None for q in ood_queries)
        out["ood_fallback_rate"] = round(fb / len(ood_queries), 3)
        out["ood_n"] = len(ood_queries)
    return out


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2)
