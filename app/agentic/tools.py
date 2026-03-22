from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from app.llm import LLMGateway
from app.rag import RAGEngine
from app.schemas import PaperRecord
from app.tooling import search_via_mcp


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    """Simple async-aware tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, handler=handler)

    def has(self, name: str) -> bool:
        return name in self._tools

    def descriptions(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    async def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(f"Unknown tool '{name}'. Available: {available}")
        out = self._tools[name].handler(**kwargs)
        if inspect.isawaitable(out):
            out = await out
        return out


@dataclass
class ToolContext:
    llm: LLMGateway
    rag: RAGEngine


def build_paperrank_tool_registry(ctx: ToolContext) -> ToolRegistry:
    registry = ToolRegistry()

    async def search_batch(
        question: str,
        sub_queries: list[str],
        source: str,
        per_query_limit: int,
        locked_concepts: list[str],
        max_papers: int,
    ) -> tuple[list[PaperRecord], list[dict]]:
        return await search_via_mcp(
            question=question,
            sub_queries=sub_queries,
            source=source,
            per_query_limit=per_query_limit,
            locked_concepts=locked_concepts,
            max_papers=max_papers,
        )

    def ingest_pdf(paper: PaperRecord) -> int:
        return ctx.rag.ingest_paper_pdf(paper)

    def retrieve_evidence(question: str, paper: PaperRecord, top_k: int = 3):
        return ctx.rag.retrieve_evidence(question=question, paper=paper, top_k=top_k)

    def score_single(
        question: str,
        paper: dict,
        evidence_snippets: list[dict],
        years: list[int],
        complementarity: float,
        quality_signal: float,
    ):
        return ctx.llm.score_paper(
            question=question,
            paper=paper,
            evidence_snippets=evidence_snippets,
            years=years,
            complementarity=complementarity,
            quality_signal=quality_signal,
        )

    def synthesize(question: str, papers_payload: list[dict]) -> str:
        return ctx.llm.synthesize(question=question, papers_payload=papers_payload)

    registry.register(
        name="search_batch",
        description="Search papers for all sub-queries via MCP tooling layer and return deduped candidates.",
        handler=search_batch,
    )
    registry.register(
        name="ingest_pdf",
        description="Download and chunk one paper PDF into local vector store.",
        handler=ingest_pdf,
    )
    registry.register(
        name="retrieve_evidence",
        description="Retrieve top evidence chunks for one paper given the research question.",
        handler=retrieve_evidence,
    )
    registry.register(
        name="score_single",
        description="Compute one paper's five-dimension score.",
        handler=score_single,
    )
    registry.register(
        name="synthesize",
        description="Generate final Chinese synthesis with citations from scored papers.",
        handler=synthesize,
    )

    return registry
