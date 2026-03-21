from __future__ import annotations

from dataclasses import dataclass

from app.llm import LLMGateway
from app.rag import RAGEngine
from app.schemas import AgentOutput, EvidenceSpan, ScoredPaper
from app.tooling import compute_complementarity, compute_quality_signal, search_via_mcp


@dataclass
class RunOptions:
    source: str = "all"
    per_query_limit: int = 6
    ingest_top_n: int = 6
    locked_concepts: list[str] | None = None


class PaperEvaluationAgent:
    def __init__(self, llm: LLMGateway | None = None) -> None:
        self.llm = llm or LLMGateway()
        self.rag = RAGEngine(self.llm)

    async def run(self, question: str, options: RunOptions | None = None) -> AgentOutput:
        options = options or RunOptions()

        locked = options.locked_concepts or []
        plan = self.llm.plan_question(question, locked_concepts=locked)
        papers, search_log = await search_via_mcp(
            question=question,
            sub_queries=plan.sub_queries,
            source=options.source,
            per_query_limit=options.per_query_limit,
            locked_concepts=locked,
        )

        if not papers:
            empty_md = "## 直接回答\n未检索到可用论文。\n\n## 当前不确定性\n- 当前没有可用证据。"
            return AgentOutput(
                question=question,
                plan=plan,
                search_log=search_log,
                scored_papers=[],
                final_answer_markdown=empty_md,
            )

        complementarity_map = compute_complementarity(papers)
        year_list = [p.year for p in papers if p.year]

        scored: list[ScoredPaper] = []
        for idx, paper in enumerate(papers):
            chunk_count = 0
            if idx < options.ingest_top_n:
                chunk_count = self.rag.ingest_paper_pdf(paper)

            evidence = self.rag.retrieve_evidence(question, paper, top_k=3) if chunk_count > 0 else []
            if not evidence:
                evidence = [
                    EvidenceSpan(
                        paper_id=paper.paper_id,
                        ref_id="",
                        page=1,
                        text=(paper.abstract or "No abstract available.")[:700],
                    )
                ]

            score = self.llm.score_paper(
                question=question,
                paper=paper.model_dump(),
                evidence_snippets=[{"page": e.page, "text": e.text[:350]} for e in evidence],
                years=year_list,
                complementarity=complementarity_map.get(paper.paper_id, 50.0),
                quality_signal=compute_quality_signal(paper),
            )

            scored.append(ScoredPaper(paper=paper, score=score, evidence=evidence))

        scored.sort(key=lambda x: x.score.total, reverse=True)

        payload = []
        for i, sp in enumerate(scored, start=1):
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
                    "evidence": [
                        {"page": e.page, "text": e.text[:480]} for e in sp.evidence
                    ],
                }
            )

        answer = self.llm.synthesize(question, payload)
        if plan.clarification_questions:
            clarify = ["## 需你确认的问题"] + [f"- {q}" for q in plan.clarification_questions]
            answer = "\n".join(clarify) + "\n\n" + answer

        return AgentOutput(
            question=question,
            plan=plan,
            search_log=search_log,
            scored_papers=scored,
            final_answer_markdown=answer,
        )
