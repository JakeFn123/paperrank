from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class ResearchPlan(BaseModel):
    research_intent: str
    intent_slots: dict[str, List[str]] = Field(default_factory=dict)
    sub_queries: List[str] = Field(default_factory=list)
    hidden_assumptions: List[str] = Field(default_factory=list)
    clarification_questions: List[str] = Field(default_factory=list)


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    citation_count: int = 0
    authors: List[str] = Field(default_factory=list)
    source: str = ""
    paper_url: str = ""
    pdf_url: str = ""
    query_match_score: float = 0.0
    rerank_score: float = 0.0
    concept_hit_count: int = 0
    matched_concepts: List[str] = Field(default_factory=list)

    @field_validator(
        "paper_id",
        "title",
        "abstract",
        "venue",
        "source",
        "paper_url",
        "pdf_url",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value):
        if value is None:
            return ""
        return str(value)

    @field_validator("authors", mode="before")
    @classmethod
    def _normalize_authors(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    @field_validator("citation_count", mode="before")
    @classmethod
    def _normalize_citation_count(cls, value):
        if value in (None, "", "null"):
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    @field_validator("query_match_score", mode="before")
    @classmethod
    def _normalize_query_match_score(cls, value):
        if value in (None, "", "null"):
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    @field_validator("rerank_score", mode="before")
    @classmethod
    def _normalize_rerank_score(cls, value):
        if value in (None, "", "null"):
            return 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    @field_validator("concept_hit_count", mode="before")
    @classmethod
    def _normalize_concept_hit_count(cls, value):
        if value in (None, "", "null"):
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    @field_validator("matched_concepts", mode="before")
    @classmethod
    def _normalize_matched_concepts(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return [str(value)]

    @field_validator("year", mode="before")
    @classmethod
    def _normalize_year(cls, value):
        if value in (None, "", "null"):
            return None
        try:
            year = int(value)
            if year < 1000 or year > 3000:
                return None
            return year
        except Exception:
            return None


class EvidenceSpan(BaseModel):
    paper_id: str
    ref_id: str
    page: int
    text: str


class PaperScore(BaseModel):
    content_relevance: float
    method_relevance: float
    timeliness: float
    quality_signal: float
    complementarity: float
    total: float
    rationale: str


class ScoredPaper(BaseModel):
    paper: PaperRecord
    score: PaperScore
    evidence: List[EvidenceSpan] = Field(default_factory=list)


class AgentOutput(BaseModel):
    question: str
    plan: ResearchPlan
    search_log: List[dict]
    scored_papers: List[ScoredPaper]
    final_answer_markdown: str
    evidence_audit: dict = Field(default_factory=dict)
    task_board_snapshot: List[dict] = Field(default_factory=list)
    loop_trace: List[dict] = Field(default_factory=list)
