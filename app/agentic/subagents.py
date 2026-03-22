from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.agentic.skills import SkillRegistry
from app.agentic.tools import ToolRegistry
from app.schemas import EvidenceSpan, PaperRecord, ResearchPlan, ScoredPaper
from app.tooling import compute_complementarity, compute_quality_signal


@dataclass
class SubAgentResult:
    payload: Any
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


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
    def run(
        self,
        question: str,
        locked_concepts: list[str],
        llm,
        intent_slots_override: dict[str, list[str]] | None = None,
    ) -> SubAgentResult:
        plan = llm.plan_question(
            question,
            locked_concepts=locked_concepts,
            forced_intent_slots=intent_slots_override or {},
        )
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
        max_papers: int,
    ) -> SubAgentResult:
        papers, search_log = await tools.call(
            "search_batch",
            question=question,
            sub_queries=plan.sub_queries,
            source=source,
            per_query_limit=per_query_limit,
            locked_concepts=locked_concepts,
            max_papers=max_papers,
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
    def _escape_md_cell(self, text: str, max_len: int = 120) -> str:
        value = (text or "").replace("\n", " ").replace("|", "\\|").strip()
        if len(value) > max_len:
            return value[: max_len - 3] + "..."
        return value

    def _evidence_by_ref(self, payload: list[dict]) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in payload:
            ref = str(row.get("ref_id", "")).strip()
            if not ref:
                continue
            evidence = row.get("evidence", []) or []
            if evidence:
                first = evidence[0] or {}
                page = first.get("page", 1)
                text = str(first.get("text", "") or "").strip()
                if text:
                    out[ref] = f"{text} [{ref} p.{page}]"
                    continue
            out[ref] = "无可用证据片段"
        return out

    def _collect_alignment_rows(self, answer: str, payload: list[dict]) -> list[tuple[str, str, str]]:
        lines = (answer or "").splitlines()
        in_key_section = False
        bullets: list[str] = []
        for line in lines:
            striped = line.strip()
            if striped.startswith("## "):
                in_key_section = striped.startswith("## 证据支撑的关键结论")
                continue
            if in_key_section and striped.startswith("-"):
                bullets.append(striped)

        evidence_map = self._evidence_by_ref(payload)
        rows: list[tuple[str, str, str]] = []
        for bullet in bullets:
            refs = sorted(set(re.findall(r"(P\d+)", bullet)))
            if refs:
                for ref in refs:
                    rows.append((bullet.lstrip("- ").strip(), ref, evidence_map.get(ref, "无可用证据片段")))
            else:
                rows.append((bullet.lstrip("- ").strip(), "未标注引用", "无可用证据片段"))

        if rows:
            return rows[:12]

        # Fallback: if model omitted key-conclusion bullets, build minimal rows from top papers.
        for row in payload[:3]:
            ref = str(row.get("ref_id", "")).strip() or "P?"
            title = str(row.get("title", "")).strip() or "Untitled paper"
            conclusion = f"论文《{title}》被纳入综合结论参考。"
            rows.append((conclusion, ref, evidence_map.get(ref, "无可用证据片段")))
        return rows

    def _append_alignment_table(self, answer: str, payload: list[dict]) -> str:
        if "## 结论-证据对齐表" in (answer or ""):
            return answer
        rows = self._collect_alignment_rows(answer, payload)
        if not rows:
            return answer

        table_lines = [
            "## 结论-证据对齐表",
            "| 结论 | 引用 | 证据片段 |",
            "|---|---|---|",
        ]
        for conclusion, ref, evidence in rows:
            table_lines.append(
                f"| {self._escape_md_cell(conclusion)} | {self._escape_md_cell(ref, 24)} | {self._escape_md_cell(evidence, 180)} |"
            )
        return (answer or "").rstrip() + "\n\n" + "\n".join(table_lines)

    def _build_evidence_audit(self, answer: str, payload: list[dict]) -> dict:
        ref_to_has_evidence = {}
        valid_refs = set()
        for row in payload:
            ref = str(row.get("ref_id", "")).strip()
            if not ref:
                continue
            valid_refs.add(ref)
            evidence = row.get("evidence", []) or []
            ref_to_has_evidence[ref] = any((e.get("text") or "").strip() for e in evidence)

        citation_matches = re.findall(r"\[(P\d+)(?:\s+p\.\d+)?\]", answer or "")
        cited_refs = sorted(set(citation_matches))
        invalid_refs = sorted([ref for ref in cited_refs if ref not in valid_refs])
        refs_without_evidence = sorted([ref for ref in cited_refs if ref_to_has_evidence.get(ref) is False])

        lines = (answer or "").splitlines()
        in_key_section = False
        key_lines: list[str] = []
        for line in lines:
            striped = line.strip()
            if striped.startswith("## "):
                in_key_section = striped.startswith("## 证据支撑的关键结论")
                continue
            if in_key_section and striped.startswith("-"):
                key_lines.append(striped)

        missing_citation_lines = [ln for ln in key_lines if re.search(r"\[P\d+(?:\s+p\.\d+)?\]", ln) is None]
        has_any_citation = len(cited_refs) > 0
        passed = has_any_citation and not invalid_refs and not missing_citation_lines and not refs_without_evidence

        if passed:
            confidence = "high"
        elif has_any_citation and not invalid_refs:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "passed": passed,
            "confidence": confidence,
            "has_any_citation": has_any_citation,
            "cited_refs": cited_refs,
            "invalid_refs": invalid_refs,
            "missing_citation_lines": missing_citation_lines,
            "refs_without_evidence": refs_without_evidence,
        }

    def _append_audit_section(self, answer: str, audit: dict) -> str:
        if not answer:
            answer = ""
        lines = ["## 证据对齐审查", f"- 审查结果：{'通过' if audit.get('passed') else '未完全通过'}"]
        lines.append(f"- 置信等级：{audit.get('confidence', 'low')}")
        if audit.get("invalid_refs"):
            lines.append(f"- 无效引用：{', '.join(audit['invalid_refs'])}")
        if audit.get("refs_without_evidence"):
            lines.append(f"- 无证据片段引用：{', '.join(audit['refs_without_evidence'])}")
        if audit.get("missing_citation_lines"):
            lines.append("- 未带引用的关键结论：")
            for ln in audit["missing_citation_lines"][:5]:
                lines.append(f"  - {ln}")
        return answer.rstrip() + "\n\n" + "\n".join(lines)

    async def run(
        self,
        tools: ToolRegistry,
        question: str,
        plan: ResearchPlan,
        scored_papers: list[ScoredPaper],
    ) -> SubAgentResult:
        if not scored_papers:
            empty_md = "## 直接回答\n未检索到可用论文。\n\n## 当前不确定性\n- 当前没有可用证据。"
            audit = {
                "passed": False,
                "confidence": "low",
                "has_any_citation": False,
                "cited_refs": [],
                "invalid_refs": [],
                "missing_citation_lines": [],
                "refs_without_evidence": [],
            }
            return SubAgentResult(payload=empty_md, notes="empty synthesis", extra={"evidence_audit": audit})

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

        answer = self._append_alignment_table(answer, payload)
        audit = self._build_evidence_audit(answer, payload)
        if not audit.get("passed", False):
            answer = self._append_audit_section(answer, audit)

        return SubAgentResult(payload=answer, notes="synthesis completed", extra={"evidence_audit": audit})
