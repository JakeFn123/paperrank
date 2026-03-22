from __future__ import annotations

from dataclasses import dataclass

from app.agentic.loop import LoopOptions, PaperRankAgentLoop
from app.llm import LLMGateway
from app.schemas import AgentOutput


@dataclass
class RunOptions:
    source: str = "all"
    per_query_limit: int = 6
    ingest_top_n: int = 6
    locked_concepts: list[str] | None = None


class PaperEvaluationAgent:
    def __init__(self, llm: LLMGateway | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.loop = PaperRankAgentLoop(self.llm)

    async def run(self, question: str, options: RunOptions | None = None) -> AgentOutput:
        options = options or RunOptions()
        return await self.loop.run(
            question=question,
            options=LoopOptions(
                source=options.source,
                per_query_limit=options.per_query_limit,
                ingest_top_n=options.ingest_top_n,
                locked_concepts=options.locked_concepts or [],
            ),
        )
