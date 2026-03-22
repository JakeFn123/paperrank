from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agentic.skills import SkillRegistry
from app.agentic.tools import ToolRegistry
from app.schemas import EvidenceSpan, PaperRecord, ResearchPlan, ScoredPaper
from app.tooling import compute_complementarity, compute_quality_signal


@dataclass
class SubAgentResult:
    payload: Any
    notes: str = ""


@dataclass
class BaseSubAgent:
    name: str
    purpose: str
    required_skills: list[str] = field(default_factory=list)

    def describe(self, skills: SkillRegistry) -> dict[str, Any]:
        loaded = []
        for name in self.required_skills:
            skill = skills.get(name)
            if skill:
                loaded.append({"name": skill.name, "description": skill.description, "path": skill.path})
        return {
            "name": self.name,
            "purpose": self.purpose,
            "required_skills": self.required_skills,
            "loaded_skills": loaded,
        }


@dataclass
class PlannerSubAgent(BaseSubAgent):
    def run(self, question: str, locked_concepts: list[str], llm) -> SubAgentResult:
        plan = llm.plan_question(question, locked_concepts=locked_concepts)
        return SubAgentResult(payload=plan, notes=f"generated {len(plan.sub_queries)} sub-queries")


@dataclass
class RetrievalSubAgent(BaseSubAgent):
    async def run(
        self,
        tools: ToolRegistry,
        question: str,
        plan: ResearchPlan,
        source: str,
        per_query_limit: int,
        locked_concepts: list[str],
    ) -> SubAgentResult:
        papers, search_log = await tools.call(
            "search_batch",
            question=question,
            sub_queries=plan.sub_queries,
            source=source,
            per_query_limit=per_query_limit,
            locked_concepts=locked_concepts,
        )
        payload = {"papers": papers, "search_log": search_log}
        return SubAgentResult(payload=payload, notes=f"retrieved {len(papers)} papers")


@dataclass
class ScoringSubAgent(BaseSubAgent):
    async def run(
        self,
        tools: ToolRegistry,
        question: str,
        papers: list[PaperRecord],
        ingest_top_n: int,
    ) -> SubAgentResult:
        if not papers:
            return SubAgentResult(payload=[], notes="no papers to score")

        complementarity_map = compute_complementarity(papers)
        year_list = [p.year for p in papers if p.year]

        scored: list[ScoredPaper] = []
        for idx, paper in enumerate(papers):
            chunk_count = 0
            if idx < ingest_top_n:
                chunk_count = await tools.call("ingest_pdf", paper=paper)

            evidence = []
            if chunk_count > 0:
                evidence = await tools.call("retrieve_evidence", question=question, paper=paper, top_k=3)

            if not evidence:
                evidence = [
                    EvidenceSpan(
                        paper_id=paper.paper_id,
                        ref_id="",
                        page=1,
                        text=(paper.abstract or "No abstract available.")[:700],
                    )
                ]

            score = await tools.call(
                "score_single",
                question=question,
                paper=paper.model_dump(),
                evidence_snippets=[{"page": e.page, "text": e.text[:350]} for e in evidence],
                years=year_list,
                complementarity=complementarity_map.get(paper.paper_id, 50.0),
                quality_signal=compute_quality_signal(paper),
            )

            scored.append(ScoredPaper(paper=paper, score=score, evidence=evidence))

        scored.sort(key=lambda x: x.score.total, reverse=True)
        return SubAgentResult(payload=scored, notes=f"scored {len(scored)} papers")


@dataclass
class SynthesisSubAgent(BaseSubAgent):
    async def run(
        self,
        tools: ToolRegistry,
        question: str,
        plan: ResearchPlan,
        scored_papers: list[ScoredPaper],
    ) -> SubAgentResult:
        if not scored_papers:
            empty_md = "## 直接回答\n未检索到可用论文。\n\n## 当前不确定性\n- 当前没有可用证据。"
            return SubAgentResult(payload=empty_md, notes="empty synthesis")

        payload = []
        for i, sp in enumerate(scored_papers, start=1):
            ref_id = f"P{i}"
            for e in sp.evidence:
                e.ref_id = ref_id
            payload.append(
                {
                    "ref_id": ref_id,
                    "title": sp.paper.title,
                    "year": sp.paper.year,
                    "venue": sp.paper.venue,
                    "source": sp.paper.source,
                    "score": sp.score.model_dump(),
                    "evidence": [{"page": e.page, "text": e.text[:480]} for e in sp.evidence],
                }
            )

        answer = await tools.call("synthesize", question=question, papers_payload=payload)
        if plan.clarification_questions:
            clarify = ["## 需你确认的问题"] + [f"- {q}" for q in plan.clarification_questions]
            answer = "\n".join(clarify) + "\n\n" + answer

        return SubAgentResult(payload=answer, notes="synthesis completed")
