"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    context: str = Field("", max_length=8000, description="Optional passages / document the query refers to")
    mode: str = Field("auto", description="auto | base | adapter:<name>")
    use_rag: bool | None = Field(None, description="Force retrieval on/off; default lets the router decide")
    max_new_tokens: int | None = Field(None, ge=1, le=512)
    task: Literal["technical_qa", "reasoning", "coding", "extraction", "tool_use", "grounded_qa"] | None = None


class ChatResponse(BaseModel):
    text: str
    task: str
    adapter: str | None
    parsed: Any = None
    parse_status: str | None = None
    used_rag: bool = False
    retrieved: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    route: dict[str, Any] = {}
    prompt_tokens: int = 0
    new_tokens: int = 0
    latency_ms: float = 0.0
    trace: list[str] = []


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    context: str = ""


class AdapterInfo(BaseModel):
    name: str
    resident: bool
    method: str | None = None
    num_parameters: int | None = None
    size_mb: float | None = None
    stack_on: str | None = None
    status: str | None = None


class SearchHit(BaseModel):
    id: str
    title: str
    text: str
    score: float


class Health(BaseModel):
    status: str
    model: str
    device: str
    quantization: str
    adapters_resident: list[str]
    adapters_available: list[str]
    router_trained: bool
    retriever_docs: int
    store: str
