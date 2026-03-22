from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agentic.skills import SkillRegistry
from app.agentic.subagents import PlannerSubAgent, RetrievalSubAgent, ScoringSubAgent, SynthesisSubAgent
from app.agentic.tasks import TaskBoard
from app.agentic.tools import ToolContext, build_paperrank_tool_registry
from app.llm import LLMGateway
from app.rag import RAGEngine
from app.schemas import AgentOutput


@dataclass
class LoopOptions:
    source: str = "all"
    per_query_limit: int = 6
    ingest_top_n: int = 6
    max_papers: int = 30
    locked_concepts: list[str] | None = None
    intent_slots_override: dict[str, list[str]] | None = None


class PaperRankAgentLoop:
    """Structured agent loop with explicit tools, skills, subagents and task system.

    Design references:
    - learn-claude-code: agent loop (s01)
    - subagent isolation (s04)
    - skill loading (s05)
    - persistent task board (s07)
    """

    def __init__(
        self,
        llm: LLMGateway | None = None,
        *,
        clear_task_board_on_run: bool = True,
    ) -> None:
        self.llm = llm or LLMGateway()
        self.rag = RAGEngine(self.llm)
        self.skills = SkillRegistry()
        self.task_board = TaskBoard()
        self.tools = build_paperrank_tool_registry(ToolContext(llm=self.llm, rag=self.rag))
        self.clear_task_board_on_run = clear_task_board_on_run

        self.planner = PlannerSubAgent(
            name="planner",
            purpose="理解研究问题、拆解子查询、识别隐含前提与澄清问题",
            required_skills=["query-decomposition"],
        )
        self.retriever = RetrievalSubAgent(
            name="retriever",
            purpose="通过 MCP 学术检索工具召回候选论文",
            required_skills=["academic-retrieval"],
        )
        self.scorer = ScoringSubAgent(
            name="evaluator",
            purpose="基于证据与多维指标逐篇评分",
            required_skills=["evidence-grading"],
        )
        self.synthesizer = SynthesisSubAgent(
            name="synthesizer",
            purpose="生成中文综合结论并标注引用",
            required_skills=["synthesis"],
        )

    def _trace(self, traces: list[dict], stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        traces.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "message": message,
                "details": details or {},
            }
        )

    async def run(self, question: str, options: LoopOptions | None = None) -> AgentOutput:
        options = options or LoopOptions()
        locked = options.locked_concepts or []
        traces: list[dict] = []

        if self.clear_task_board_on_run:
            self.task_board.clear()

        root = self.task_board.create(
            title="PaperRank run",
            assignee="orchestrator",
            payload={
                "question": question,
                "options": {
                    "source": options.source,
                    "per_query_limit": options.per_query_limit,
                    "ingest_top_n": options.ingest_top_n,
                    "max_papers": options.max_papers,
                    "locked_concepts": locked,
                    "intent_slots_override": options.intent_slots_override or {},
                },
            },
        )
        self.task_board.update(root.id, status="in_progress", result_summary="workflow started")
        self._trace(traces, "loop", "root task started", {"task_id": root.id})

        plan_task = self.task_board.create(
            title="Question decomposition",
            assignee=self.planner.name,
            depends_on=[root.id],
        )
        retrieval_task = self.task_board.create(
            title="Academic retrieval",
            assignee=self.retriever.name,
            depends_on=[plan_task.id],
        )
        score_task = self.task_board.create(
            title="Evidence scoring",
            assignee=self.scorer.name,
            depends_on=[retrieval_task.id],
        )
        synth_task = self.task_board.create(
            title="Final synthesis",
            assignee=self.synthesizer.name,
            depends_on=[score_task.id],
        )

        self._trace(
            traces,
            "config",
            "agent framework initialized",
            {
                "tools": self.tools.descriptions(),
                "skills": [
                    self.planner.describe(self.skills),
                    self.retriever.describe(self.skills),
                    self.scorer.describe(self.skills),
                    self.synthesizer.describe(self.skills),
                ],
            },
        )

        # 1) Planner
        self.task_board.update(plan_task.id, status="in_progress")
        plan_result = self.planner.run(
            question=question,
            locked_concepts=locked,
            llm=self.llm,
            intent_slots_override=options.intent_slots_override or {},
        )
        plan = plan_result.payload
        self.task_board.update(plan_task.id, status="completed", result_summary=plan_result.notes)
        self._trace(
            traces,
            "planner",
            "question decomposition completed",
            {
                "sub_queries": plan.sub_queries,
                "research_intent": plan.research_intent,
                "intent_slots": plan.intent_slots,
            },
        )

        # 2) Retriever
        self.task_board.update(retrieval_task.id, status="in_progress")
        retrieval_result = await self.retriever.run(
            tools=self.tools,
            question=question,
            plan=plan,
            source=options.source,
            per_query_limit=options.per_query_limit,
            locked_concepts=locked,
            max_papers=options.max_papers,
        )
        papers = retrieval_result.payload["papers"]
        search_log = retrieval_result.payload["search_log"]
        self.task_board.update(retrieval_task.id, status="completed", result_summary=retrieval_result.notes)
        self._trace(
            traces,
            "retrieval",
            "retrieval finished",
            {"paper_count": len(papers), "search_log": search_log},
        )

        # 3) Scorer
        self.task_board.update(score_task.id, status="in_progress")
        scoring_result = await self.scorer.run(
            tools=self.tools,
            question=question,
            papers=papers,
            ingest_top_n=options.ingest_top_n,
        )
        scored_papers = scoring_result.payload
        self.task_board.update(score_task.id, status="completed", result_summary=scoring_result.notes)
        self._trace(
            traces,
            "scoring",
            "paper scoring finished",
            {
                "scored_count": len(scored_papers),
                "top_scores": [sp.score.total for sp in scored_papers[:5]],
            },
        )

        # 4) Synthesis
        self.task_board.update(synth_task.id, status="in_progress")
        synthesis_result = await self.synthesizer.run(
            tools=self.tools,
            question=question,
            plan=plan,
            scored_papers=scored_papers,
        )
        final_answer = synthesis_result.payload
        evidence_audit = synthesis_result.extra.get("evidence_audit", {})
        self.task_board.update(synth_task.id, status="completed", result_summary=synthesis_result.notes)
        self._trace(traces, "synthesis", "final synthesis generated", {"evidence_audit": evidence_audit})

        self.task_board.update(root.id, status="completed", result_summary="workflow completed")
        self._trace(traces, "loop", "root task completed", {"task_id": root.id})

        return AgentOutput(
            question=question,
            plan=plan,
            search_log=search_log,
            scored_papers=scored_papers,
            final_answer_markdown=final_answer,
            evidence_audit=evidence_audit,
            task_board_snapshot=self.task_board.snapshot(),
            loop_trace=traces,
        )
