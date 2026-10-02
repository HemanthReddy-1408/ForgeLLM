"""FastAPI service: one shared frozen base model, adapters hot-swapped per request, router + RAG + tools in front."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from forgellm.inference.engine import InferenceEngine
from forgellm.models.adapters import unload_adapter
from forgellm.models.lora import loaded_adapters
from forgellm.serving.schemas import (
    AdapterInfo,
    ChatRequest,
    ChatResponse,
    Health,
    RouteRequest,
    SearchHit,
)
from forgellm.store import get_store


class Metrics:
    def __init__(self) -> None:
        self.n = 0
        self.errors = 0
        self.by_adapter: dict[str, int] = defaultdict(int)
        self.lat: deque[float] = deque(maxlen=500)
        self.tokens = 0

    def snapshot(self) -> dict[str, Any]:
        lat = sorted(self.lat)
        pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None  # noqa: E731
        return {"requests": self.n, "errors": self.errors, "by_adapter": dict(self.by_adapter), "generated_tokens": self.tokens,
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95)}}


def create_app(engine: InferenceEngine) -> FastAPI:
    app = FastAPI(title="ForgeLLM", version="0.1.0", description="Adapter-routed LLM inference with RAG and tool use")
    lock = threading.Lock()          # one model, one generation at a time (adapters are swapped on the shared base)
    metrics = Metrics()

    @app.get("/health", response_model=Health)
    def health() -> Health:
        spec = engine.loaded.spec
        return Health(status="ok", model=spec.name, device=spec.device, quantization=spec.quantization,
                      adapters_resident=loaded_adapters(engine.loaded.model), adapters_available=engine.store.available(),
                      router_trained=bool(engine.router and engine.router.trained),
                      retriever_docs=len(engine.retriever) if engine.retriever else 0, store=get_store().describe())

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        try:
            with lock:
                r = engine.respond(req.query, req.context, req.mode, req.use_rag, req.max_new_tokens, task=req.task)
        except FileNotFoundError as e:
            metrics.errors += 1
            raise HTTPException(404, str(e)) from e
        except Exception as e:
            metrics.errors += 1
            raise HTTPException(500, f"{type(e).__name__}: {e}") from e
        metrics.n += 1
        metrics.by_adapter[r.adapter or "base"] += 1
        metrics.lat.append((time.perf_counter() - t0) * 1000)
        metrics.tokens += r.new_tokens
        return ChatResponse(**r.to_dict())

    @app.post("/v1/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        if engine.router is None:
            raise HTTPException(503, "no router loaded")
        return engine.router.route(req.query, req.context).to_dict()

    @app.get("/v1/adapters", response_model=list[AdapterInfo])
    def adapters() -> list[AdapterInfo]:
        res = set(loaded_adapters(engine.loaded.model))
        out = []
        for n in engine.store.available():
            man = engine.store.info(n)
            reg = get_store().production_adapter(n)
            out.append(AdapterInfo(name=n, resident=n in res, method=man.get("method"), num_parameters=man.get("num_parameters"),
                                   size_mb=man.get("size_mb"), stack_on=man.get("stack_on"), status="production" if reg else None))
        return out

    @app.post("/v1/adapters/{name}/load", response_model=AdapterInfo)
    def load(name: str) -> AdapterInfo:
        try:
            with lock:
                engine.store.ensure(name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        man = engine.store.info(name)
        return AdapterInfo(name=name, resident=True, method=man.get("method"), num_parameters=man.get("num_parameters"),
                           size_mb=man.get("size_mb"), stack_on=man.get("stack_on"))

    @app.delete("/v1/adapters/{name}")
    def unload(name: str) -> dict[str, str]:
        with lock:
            if name not in loaded_adapters(engine.loaded.model):
                raise HTTPException(404, f"adapter '{name}' is not resident")
            unload_adapter(engine.loaded.model, name)
            engine.store.resident.pop(name, None)
        return {"unloaded": name}

    @app.get("/v1/rag/search", response_model=list[SearchHit])
    def search(q: str = Query(min_length=1), k: int = Query(3, ge=1, le=10)) -> list[SearchHit]:
        if engine.retriever is None:
            raise HTTPException(503, "no retriever loaded")
        return [SearchHit(**h) for h in engine.retriever.search(q, k)]

    @app.get("/v1/registry")
    def registry() -> list[dict[str, Any]]:
        return [{k: v for k, v in a.items() if k != "metrics"} for a in get_store().adapters()]

    @app.get("/v1/metrics")
    def get_metrics() -> dict[str, Any]:
        return metrics.snapshot()

    return app
