"""The inference engine — everything between a user request and a final answer:

    request -> TaskRouter -> adapter selection (hot-swap on ONE shared base) -> [RAG retrieval | tool selection]
            -> prompt build -> generation -> [tool loop: parse call -> validate -> execute -> observation -> generate]
            -> output parser -> Response (text + parsed object + full trace)"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forgellm.config import InferenceConfig
from forgellm.data.templates import build_messages, encode_prompt
from forgellm.inference.generation import GenConfig, GenResult, generate_batch
from forgellm.inference.parser import extract_json
from forgellm.inference.router import RouteDecision, TaskRouter
from forgellm.models.adapters import AdapterStore
from forgellm.models.base import LoadedModel
from forgellm.models.lora import loaded_adapters, set_active
from forgellm.rag.context import build_context, cited_documents
from forgellm.rag.retriever import BM25, Retriever
from forgellm.tools.execution import ToolResult, execute, parse_tool_call
from forgellm.tools.registry import ToolRegistry

STOP = ["</tool_call>"]


@dataclass
class Response:
    text: str
    task: str
    adapter: str | None
    parsed: Any = None
    parse_status: str | None = None
    used_rag: bool = False
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    route: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    new_tokens: int = 0
    latency_ms: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {}


class InferenceEngine:
    def __init__(self, loaded: LoadedModel, cfg: InferenceConfig | None = None, router: TaskRouter | None = None,
                 retriever: Retriever | None = None, tools: ToolRegistry | None = None,
                 store: AdapterStore | None = None) -> None:
        self.loaded = loaded
        self.cfg = cfg or InferenceConfig()
        self.router = router
        self.retriever = retriever
        self.tools = tools or ToolRegistry()
        self.store = store or AdapterStore(loaded.model, self.cfg.adapters_dir or None, self.cfg.max_adapters_resident)
        self._tool_index = BM25([f"{t.name.replace('_', ' ')} {t.description} {t.keywords}" for t in self.tools.tools.values()])
        self._tool_names = list(self.tools.tools)
        if retriever is not None and "search_docs" in self.tools.backends:
            self.tools.backends["search_docs"] = lambda query, top_k=3: {
                "query": query, "results": [{"id": h["id"], "title": h["title"]} for h in retriever.search(query, top_k)]}

    # -- helpers ---------------------------------------------------------------------------------------------
    def select_tools(self, query: str, k: int = 4) -> list[str]:
        s = self._tool_index.scores(query)
        order = sorted(range(len(self._tool_names)), key=lambda i: -s[i])
        return [self._tool_names[i] for i in order[:k]]

    def _gen(self, messages: list[dict[str, str]], gcfg: GenConfig) -> GenResult:
        tok = self.loaded.tokenizer
        ids = encode_prompt(messages, tok)
        limit = self.loaded.spec.context_length - gcfg.max_new_tokens
        ids = ids[-limit:] if len(ids) > limit else ids
        return generate_batch(self.loaded.model, tok, [ids], gcfg, self.loaded.device)[0]

    def _activate(self, adapter: str | None, trace: list[str]) -> None:
        if adapter is None:
            set_active(self.loaded.model, {})
            trace.append("adapter: none (base model)")
            return
        was = adapter in loaded_adapters(self.loaded.model)
        t0 = time.perf_counter()
        self.store.activate_chain(adapter)
        trace.append(f"adapter: {adapter} ({'resident' if was else f'loaded in {(time.perf_counter() - t0) * 1000:.0f} ms'})")

    # -- main entry ------------------------------------------------------------------------------------------
    def respond(self, query: str, context: str = "", mode: str = "auto", use_rag: bool | None = None,
                max_new_tokens: int | None = None, max_tool_rounds: int = 2, task: str | None = None) -> Response:
        """mode: 'auto' (router decides) | 'base' (never an adapter) | 'adapter:<name>' (forced adapter)."""
        t0 = time.perf_counter()
        trace: list[str] = []
        if task is not None:
            from forgellm.data.schemas import TASK_TO_ADAPTER
            decision = RouteDecision(task, TASK_TO_ADAPTER.get(task), 1.0, False, "forced task")
        elif self.router is not None:
            decision = self.router.route(query, context)
        else:
            decision = RouteDecision("general", None, 0.0, False, "no router")
        trace.append(f"route: task={decision.task} conf={decision.confidence:.2f} ({decision.reason})")
        adapter = decision.adapter
        if mode == "base":
            adapter = None
        elif mode.startswith("adapter:"):
            adapter = mode.split(":", 1)[1]
        rag = decision.use_rag if use_rag is None else use_rag
        self._activate(adapter, trace)
        g = GenConfig(max_new_tokens=max_new_tokens or self.cfg.max_new_tokens, temperature=self.cfg.temperature,
                      top_k=self.cfg.top_k, top_p=self.cfg.top_p, repetition_penalty=self.cfg.repetition_penalty)
        resp = Response("", decision.task, adapter, route=decision.to_dict())
        eff_task = decision.task if decision.task != "general" else "general"

        retrieved: list[dict[str, Any]] = []
        passages = context
        if rag and self.retriever is not None and not context:
            hits = self.retriever.search(query, self.cfg.rag_top_k)
            passages, retrieved = build_context(hits)
            trace.append(f"rag: retrieved {[h['id'] for h in retrieved]}")
            resp.used_rag, resp.retrieved = True, retrieved
            eff_task = "grounded_qa"

        history: list[dict[str, str]] = []
        tool_names: list[str] = []
        sys_input = passages
        if eff_task == "tool_use":
            tool_names = self.select_tools(query)
            sys_input = self.tools.prompt(tool_names)
            trace.append(f"tools offered: {tool_names}")
        total_new, prompt_tokens = 0, 0
        text = ""
        for _rnd in range(max_tool_rounds + 1):
            msgs = build_messages(eff_task, query, sys_input, history)
            r = self._gen(msgs, GenConfig(**{**g.__dict__, "stop_strings": STOP if eff_task == "tool_use" else []}))
            prompt_tokens = r.prompt_tokens
            total_new += r.new_tokens
            text = r.text.strip()
            call = parse_tool_call(text + ("</tool_call>" if r.finish == "stop_string" and "</tool_call>" not in text else "")) \
                if eff_task == "tool_use" else None
            if call is None:
                break
            res: ToolResult = execute(call, self.tools)
            resp.tool_calls.append({"name": call.name, "arguments": call.arguments, "ok": res.ok, "observation": res.observation,
                                    "error": res.error, "latency_ms": round(res.latency_ms, 2)})
            trace.append(f"tool: {call.name}({call.arguments}) -> {'ok' if res.ok else 'ERROR ' + str(res.error)}")
            history += [{"role": "assistant", "content": f"<tool_call>\n{call.raw}\n</tool_call>"},
                        {"role": "user", "content": f"<tool_response>\n{res.as_text()}\n</tool_response>"}]
        resp.text = text
        resp.prompt_tokens, resp.new_tokens = prompt_tokens, total_new
        if eff_task == "extraction":
            resp.parsed, resp.parse_status = extract_json(text)
        elif eff_task == "grounded_qa" and retrieved:
            resp.citations = cited_documents(text, retrieved)
        resp.task = eff_task if eff_task != decision.task else decision.task
        resp.trace = trace
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp
