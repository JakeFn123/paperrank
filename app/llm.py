from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI

from app import prompts
from app.config import EMBEDDING_MODEL, LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL, SUBQUERY_COUNT
from app.schemas import PaperScore, ResearchPlan

_EN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "does",
    "will",
    "can",
    "are",
    "was",
    "were",
    "is",
    "to",
    "of",
    "in",
    "on",
    "by",
    "how",
    "why",
}

_CONCEPT_HINTS: list[tuple[list[str], str]] = [
    (["rag", "retrieval augmented generation", "检索增强", "检索增强生成"], "retrieval augmented generation"),
    (["reranker", "rerank", "重排序", "重排"], "reranking"),
    (["cross-encoder", "交叉编码器"], "cross-encoder reranker"),
    (["企业", "enterprise", "知识库", "knowledge base"], "enterprise knowledge base question answering"),
    (["稳定", "鲁棒", "reliability", "robustness", "stability"], "robustness and reliability"),
    (["hallucination", "幻觉"], "hallucination mitigation"),
    (["准确", "accuracy", "faithfulness", "factuality"], "answer faithfulness and accuracy"),
    (["延迟", "latency", "效率", "cost"], "latency and cost trade-off"),
    (["benchmark", "评测", "实验"], "benchmark evaluation"),
]

_CONCEPT_LEXICON = [
    {
        "category": "subject",
        "en": "retrieval augmented generation",
        "aliases": ["rag", "retrieval augmented generation", "retrieval-augmented generation", "检索增强", "检索增强生成"],
    },
    {
        "category": "subject",
        "en": "question answering",
        "aliases": ["question answering", "qa", "问答", "知识问答"],
    },
    {
        "category": "subject",
        "en": "multi-agent systems",
        "aliases": ["multi-agent", "multi agent", "agentic", "多智能体"],
    },
    {
        "category": "subject",
        "en": "llm agents",
        "aliases": ["agent", "agents", "智能体", "llm agent", "llm agents"],
    },
    {
        "category": "subject",
        "en": "code generation",
        "aliases": ["code generation", "代码生成", "program synthesis"],
    },
    {
        "category": "intervention",
        "en": "reranking",
        "aliases": ["rerank", "reranker", "re-ranking", "重排", "重排序", "重排序器"],
    },
    {
        "category": "intervention",
        "en": "cross-encoder reranker",
        "aliases": ["cross-encoder", "cross encoder", "交叉编码器"],
    },
    {
        "category": "intervention",
        "en": "dense retrieval",
        "aliases": ["dense retrieval", "dual-encoder", "双塔检索"],
    },
    {
        "category": "intervention",
        "en": "self-reflection mechanism",
        "aliases": ["reflection", "self-reflection", "reflect", "反思机制", "反思", "自反思"],
    },
    {
        "category": "intervention",
        "en": "markov decision process",
        "aliases": ["mdp", "markov decision process", "马尔可夫决策过程"],
    },
    {
        "category": "intervention",
        "en": "reinforcement learning policy",
        "aliases": ["reinforcement learning", "rl", "强化学习", "策略学习"],
    },
    {
        "category": "context",
        "en": "enterprise knowledge base",
        "aliases": ["enterprise", "企业", "knowledge base", "知识库"],
    },
    {
        "category": "context",
        "en": "long-context documents",
        "aliases": ["long context", "长文本", "长上下文"],
    },
    {
        "category": "outcome",
        "en": "robustness",
        "aliases": ["robustness", "鲁棒", "稳定性", "stability", "reliability"],
    },
    {
        "category": "outcome",
        "en": "faithfulness",
        "aliases": ["faithfulness", "factuality", "事实性", "忠实性", "hallucination", "幻觉"],
    },
    {
        "category": "outcome",
        "en": "accuracy",
        "aliases": ["accuracy", "准确率", "正确率", "precision", "recall", "f1"],
    },
    {
        "category": "outcome",
        "en": "task completion rate",
        "aliases": ["task completion", "任务完成率", "completion rate", "成功率"],
    },
    {
        "category": "outcome",
        "en": "task execution efficiency",
        "aliases": ["task execution", "执行效率", "执行进度", "执行过程", "efficiency"],
    },
    {
        "category": "outcome",
        "en": "error rate",
        "aliases": ["error rate", "错误率", "错误", "failure rate"],
    },
    {
        "category": "outcome",
        "en": "latency and cost",
        "aliases": ["latency", "时延", "延迟", "cost", "成本", "效率"],
    },
    {
        "category": "evaluation",
        "en": "benchmark",
        "aliases": ["benchmark", "基准", "评测", "ablation", "消融"],
    },
    {
        "category": "evaluation",
        "en": "systematic review",
        "aliases": ["systematic review", "survey", "综述", "系统综述"],
    },
]


@dataclass
class LLMGateway:
    backend: str = LLM_BACKEND

    def __post_init__(self) -> None:
        self.client = None
        if self.backend == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when LLM_BACKEND=openai")
            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _json_from_text(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        return json.loads(text)

    def _openai_text(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return response.output_text

    def plan_question(
        self,
        question: str,
        locked_concepts: list[str] | None = None,
        forced_intent_slots: dict[str, list[str]] | None = None,
    ) -> ResearchPlan:
        locked = self._normalize_locked_concepts(locked_concepts or [])
        frame = self._build_intent_frame(question, locked)
        frame = self._merge_forced_intent_slots(frame, forced_intent_slots or {})
        fallback_plan = self._rule_based_plan(question, frame, locked)

        if self.backend == "mock":
            return fallback_plan

        try:
            raw = self._openai_text(prompts.build_decompose_prompt(question, SUBQUERY_COUNT))
            data = self._json_from_text(raw)
            llm_plan = ResearchPlan.model_validate(data)
        except Exception:
            return fallback_plan

        llm_plan.sub_queries = self._normalize_subqueries(llm_plan.sub_queries, question, frame, locked)
        llm_plan.sub_queries = self._enforce_locked_concepts(llm_plan.sub_queries, locked)
        if not self._is_plan_relevant(question, llm_plan.sub_queries, frame, locked):
            return fallback_plan

        # Use the better plan between LLM and rule-based according to intent coverage.
        llm_score = self._plan_coverage_score(llm_plan.sub_queries, frame)
        fallback_score = self._plan_coverage_score(fallback_plan.sub_queries, frame)
        plan = llm_plan if llm_score >= fallback_score else fallback_plan

        # Force intent text to stay aligned with extracted frame.
        plan.research_intent = self._build_research_intent(question, frame, locked)
        plan.intent_slots = frame
        plan.sub_queries = self._enforce_locked_concepts(plan.sub_queries, locked)
        if len(plan.hidden_assumptions) < 2:
            plan.hidden_assumptions = self._build_assumptions(frame)
        if not plan.clarification_questions:
            plan.clarification_questions = self._build_clarifications(frame)
        if locked:
            plan.clarification_questions = [
                f"你锁定的关键概念：{', '.join(locked)}。是否还要加入额外约束（数据集/行业/年份）？"
            ] + plan.clarification_questions
            plan.clarification_questions = plan.clarification_questions[:3]
        return plan

    def score_paper(
        self,
        question: str,
        paper: dict,
        evidence_snippets: Iterable[dict],
        years: list[int],
        complementarity: float,
        quality_signal: float,
    ) -> PaperScore:
        if self.backend == "mock":
            return self._mock_score(question, paper, complementarity, quality_signal)

        raw = self._openai_text(
            prompts.build_llm_scoring_prompt(question, paper, evidence_snippets, years)
        )
        parsed = self._json_from_text(raw)
        content = float(parsed.get("content_relevance", 50))
        method = float(parsed.get("method_relevance", 50))
        timeliness = float(parsed.get("timeliness", 50))

        total = (
            content * 0.35
            + method * 0.25
            + timeliness * 0.15
            + quality_signal * 0.15
            + complementarity * 0.10
        )
        return PaperScore(
            content_relevance=round(content, 2),
            method_relevance=round(method, 2),
            timeliness=round(timeliness, 2),
            quality_signal=round(quality_signal, 2),
            complementarity=round(complementarity, 2),
            total=round(total, 2),
            rationale=str(parsed.get("rationale", "由大模型完成评分")),
        )

    def synthesize(self, question: str, papers_payload: list[dict]) -> str:
        if self.backend == "mock":
            return self._mock_synthesis(question, papers_payload)
        return self._openai_text(prompts.build_synthesis_prompt(question, papers_payload))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "openai":
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [row.embedding for row in response.data]
        return [self._mock_embedding(t) for t in texts]

    def _mock_plan(self, question: str) -> ResearchPlan:
        frame = self._build_intent_frame(question, [])
        return self._rule_based_plan(question, frame, [])

    def _mock_overlap(self, a: str, b: str) -> float:
        set_a = {x for x in re.split(r"[^a-zA-Z0-9]+", a.lower()) if len(x) > 2}
        set_b = {x for x in re.split(r"[^a-zA-Z0-9]+", b.lower()) if len(x) > 2}
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union

    def _extract_en_terms(self, text: str) -> list[str]:
        terms = []
        for t in re.split(r"[^a-zA-Z0-9+.-]+", text.lower()):
            if len(t) < 3:
                continue
            if t in _EN_STOPWORDS:
                continue
            terms.append(t)
        uniq = []
        for t in terms:
            if t not in uniq:
                uniq.append(t)
        return uniq

    def _extract_hint_terms(self, question: str) -> list[str]:
        q_lower = question.lower()
        selected: list[str] = []
        for keys, term in _CONCEPT_HINTS:
            if any(k in q_lower or k in question for k in keys):
                selected.append(term)
        uniq = []
        for t in selected:
            if t not in uniq:
                uniq.append(t)
        return uniq

    def _extract_acronyms(self, text: str) -> list[str]:
        acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text or "")
        uniq = []
        for a in acronyms:
            low = a.lower()
            if low not in uniq:
                uniq.append(low)
        return uniq

    def _normalize_locked_concepts(self, locked_concepts: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in locked_concepts:
            value = str(raw or "").strip().lower()
            if not value:
                continue
            value = " ".join(value.split())
            # Canonicalize known Chinese/English aliases into searchable English concepts.
            mapped = None
            for row in _CONCEPT_LEXICON:
                aliases = [str(a).lower() for a in row.get("aliases", [])]
                if value in aliases:
                    mapped = str(row.get("en", "")).lower().strip()
                    break
            if mapped:
                value = mapped
            if value not in normalized:
                normalized.append(value)
        return normalized

    def _normalize_slot_values(self, values) -> list[str]:
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        out: list[str] = []
        for raw in values:
            val = str(raw or "").strip().lower()
            if not val:
                continue
            val = " ".join(val.split())
            if val not in out:
                out.append(val)
        return out

    def _merge_forced_intent_slots(
        self,
        frame: dict[str, list[str]],
        forced_intent_slots: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {
            "subject": list(frame.get("subject", [])),
            "intervention": list(frame.get("intervention", [])),
            "outcome": list(frame.get("outcome", [])),
            "context": list(frame.get("context", [])),
            "evaluation": list(frame.get("evaluation", [])),
        }
        allowed = {"subject", "intervention", "outcome", "context", "evaluation"}
        for key, values in (forced_intent_slots or {}).items():
            if key not in allowed:
                continue
            normalized_values = self._normalize_slot_values(values)
            if normalized_values:
                merged[key] = normalized_values
        return merged

    def _build_intent_frame(self, question: str, locked_concepts: list[str] | None = None) -> dict[str, list[str]]:
        q_lower = (question or "").lower()
        frame: dict[str, list[str]] = {
            "subject": [],
            "intervention": [],
            "outcome": [],
            "context": [],
            "evaluation": [],
        }

        for row in _CONCEPT_LEXICON:
            aliases = [str(a).lower() for a in row["aliases"]]
            if any(alias in q_lower for alias in aliases):
                category = str(row["category"])
                value = str(row["en"])
                if value not in frame[category]:
                    frame[category].append(value)

        # Keep acronyms (e.g., MCP, RLHF, LoRA) as subject hints when not already covered.
        for token in self._extract_acronyms(question):
            if token in {"rag", "qa"}:
                continue
            if token not in frame["subject"]:
                frame["subject"].append(token)

        # Fill minimal frame using generic hints to avoid empty or off-topic queries.
        if not frame["subject"]:
            hints = self._extract_hint_terms(question)
            if hints:
                frame["subject"].append(hints[0])
        if not frame["subject"]:
            frame["subject"].append("machine learning")

        if not frame["intervention"] and any(x in q_lower for x in ["提升", "improve", "优化", "优化", "增强"]):
            frame["intervention"].append("method design")

        if not frame["outcome"]:
            if any(x in q_lower for x in ["效果", "performance", "提升", "improve"]):
                frame["outcome"].append("accuracy")
            if any(x in q_lower for x in ["稳定", "鲁棒", "reliability", "robust"]):
                frame["outcome"].append("robustness")
        if not frame["outcome"]:
            frame["outcome"].append("effectiveness")

        for lc in locked_concepts or []:
            if lc not in frame["subject"]:
                frame["subject"].append(lc)

        return frame

    def _normalize_subqueries(
        self,
        sub_queries: list[str],
        question: str,
        frame: dict[str, list[str]],
        locked_concepts: list[str],
    ) -> list[str]:
        cleaned = [self._clean_query(q) for q in sub_queries if q and q.strip()]
        cleaned = [q for q in cleaned if q]

        core_terms = self._core_terms(question, frame)
        valid = [q for q in cleaned if self._query_has_core(q, core_terms)]

        fallback = self._rule_based_subqueries(question, frame, locked_concepts)
        queries = []
        for q in valid + fallback:
            if q not in queries:
                queries.append(q)
            if len(queries) >= SUBQUERY_COUNT:
                break
        return self._enforce_locked_concepts(queries[:SUBQUERY_COUNT], locked_concepts)

    def _is_plan_relevant(
        self,
        question: str,
        sub_queries: list[str],
        frame: dict[str, list[str]],
        locked_concepts: list[str],
    ) -> bool:
        if not sub_queries:
            return False
        joined = " ".join(sub_queries).lower()
        if locked_concepts and not all(lc in joined for lc in locked_concepts):
            return False
        core_terms = self._core_terms(question, frame)
        if core_terms and any(t in joined for t in core_terms):
            # Need minimum intent coverage, not just token overlap.
            return self._plan_coverage_score(sub_queries, frame) >= 0.5
        en_terms = self._extract_en_terms(question)
        if not en_terms:
            # Chinese-only question: require at least one technical token in query.
            return any(
                token in joined
                for token in ["retrieval", "generation", "rerank", "benchmark", "question answering"]
            )
        overlap = sum(1 for t in en_terms if t in joined)
        return overlap >= max(1, min(2, len(en_terms)))

    def _clean_query(self, query: str, max_tokens: int = 10) -> str:
        tokens = [t for t in re.split(r"[^a-zA-Z0-9+.-]+", query.lower()) if t]
        uniq = []
        for t in tokens:
            if t in _EN_STOPWORDS:
                continue
            if t not in uniq:
                uniq.append(t)
        return " ".join(uniq[:max_tokens]).strip()

    def _core_terms(self, question: str, frame: dict[str, list[str]] | None = None) -> list[str]:
        if frame is None:
            frame = self._build_intent_frame(question)
        hints = [h.lower() for h in self._extract_hint_terms(question)]
        mapped = []
        for h in hints:
            mapped.extend([t for t in re.split(r"[^a-zA-Z0-9+.-]+", h) if t])
        for category in ["subject", "intervention", "outcome", "context"]:
            for phrase in frame.get(category, []):
                mapped.extend([t for t in re.split(r"[^a-zA-Z0-9+.-]+", phrase.lower()) if len(t) > 2])
        en = self._extract_en_terms(question)
        core = mapped + en
        uniq = []
        for t in core:
            if t not in uniq:
                uniq.append(t)
        return uniq

    def _query_has_core(self, query: str, core_terms: list[str]) -> bool:
        if not core_terms:
            return True
        q = query.lower()
        return any(t in q for t in core_terms)

    def _enforce_locked_concepts(self, sub_queries: list[str], locked_concepts: list[str]) -> list[str]:
        if not locked_concepts:
            return sub_queries
        enforced: list[str] = []
        for q in sub_queries:
            base = q.lower().strip()
            missing = [lc for lc in locked_concepts if lc not in base]
            if missing:
                q = f"{q} {' '.join(missing)}"
            q = self._clean_query(q, max_tokens=24)
            enforced.append(q)
        return enforced

    def _query_from_parts(self, parts: list[str]) -> str:
        return self._clean_query(" ".join([p for p in parts if p]))

    def _rule_based_subqueries(self, question: str, frame: dict[str, list[str]], locked_concepts: list[str]) -> list[str]:
        subject = frame.get("subject", [])[:2]
        intervention = frame.get("intervention", [])[:2]
        outcome = frame.get("outcome", [])[:2]
        context = frame.get("context", [])[:1]
        evaluation = frame.get("evaluation", [])[:1]

        core = (subject + intervention)[:3]
        if not core:
            core = ["machine learning"]

        candidates = [
            self._query_from_parts(core + context + outcome[:1] + ["empirical study"]),
            self._query_from_parts(core + evaluation + ["benchmark", "ablation"]),
            self._query_from_parts(core + outcome + ["failure analysis"]),
            self._query_from_parts(core + ["latency", "cost", "trade-off"]),
            self._query_from_parts(core + ["systematic review"]),
        ]

        uniq = []
        for q in candidates:
            if not q:
                continue
            if q not in uniq:
                uniq.append(q)

        return self._enforce_locked_concepts(uniq[: max(SUBQUERY_COUNT, 3)], locked_concepts)

    def _build_research_intent(self, question: str, frame: dict[str, list[str]], locked_concepts: list[str]) -> str:
        subject = ", ".join(frame.get("subject", [])[:2]) or "目标技术"
        intervention = ", ".join(frame.get("intervention", [])[:2]) or "候选方法"
        outcome = ", ".join(frame.get("outcome", [])[:2]) or "关键指标"
        context = ", ".join(frame.get("context", [])[:1]) or "目标场景"
        q_low = (question or "").lower()
        if any(k in q_low for k in ["是否", "whether", "does", "do", "显著", "improve", "提升", "降低"]):
            mode = "效果验证"
            mode_desc = "验证该方法是否带来可观测改进"
        elif any(k in q_low for k in ["如何", "how", "策略", "方案", "design"]):
            mode = "方法设计"
            mode_desc = "寻找可执行的方法设计与优化路径"
        elif any(k in q_low for k in ["对比", "比较", "vs", "versus", "compare"]):
            mode = "对比评估"
            mode_desc = "比较不同方案在同一评测口径下的差异"
        elif any(k in q_low for k in ["为什么", "why", "机理", "原因", "机制"]):
            mode = "机理解释"
            mode_desc = "解释关键现象背后的原因与边界条件"
        else:
            mode = "证据综合"
            mode_desc = "综合文献证据并形成可落地判断"

        suffix = ""
        if locked_concepts:
            suffix = f"；并强制围绕锁定概念：{', '.join(locked_concepts)}"
        return (
            f"研究目标（{mode}）：围绕用户问题“{question}”，在{context}中评估“{intervention}”对“{subject}”的影响，"
            f"重点关注{outcome}，目标是{mode_desc}，并识别适用边界与代价{suffix}。"
        )

    def _build_assumptions(self, frame: dict[str, list[str]]) -> list[str]:
        assumptions = ["默认不同论文实验设置具备可比性，但真实场景可能存在口径差异。"]
        if frame.get("context"):
            assumptions.append("默认公开论文场景可迁移到你的业务场景，实际可能需要领域适配。")
        if frame.get("outcome"):
            assumptions.append("默认论文报告指标与真实目标一致，可能存在指标-业务目标错位。")
        return assumptions[:3]

    def _build_clarifications(self, frame: dict[str, list[str]]) -> list[str]:
        questions = []
        if "robustness" in frame.get("outcome", []) and "accuracy" in frame.get("outcome", []):
            questions.append("你希望优先优化准确率，还是优先降低幻觉并提升鲁棒性？")
        elif "robustness" in frame.get("outcome", []):
            questions.append("你对鲁棒性的定义更偏向抗噪性、跨域稳定性，还是低幻觉率？")
        else:
            questions.append("你最关心的评估指标是什么（如准确率、召回率、延迟、成本）？")
        if frame.get("context"):
            questions.append("你的目标场景是否有行业约束（如金融/医疗/法律）需要单独筛选？")
        questions.append("你希望优先近两年论文，还是同时纳入经典基线论文？")
        return questions[:3]

    def _plan_coverage_score(self, sub_queries: list[str], frame: dict[str, list[str]]) -> float:
        if not sub_queries:
            return 0.0

        category_terms = {
            "subject": frame.get("subject", []),
            "intervention": frame.get("intervention", []),
            "outcome": frame.get("outcome", []),
            "context": frame.get("context", []),
        }
        active = [k for k, v in category_terms.items() if v]
        if not active:
            return 0.0

        scores = []
        for q in sub_queries:
            q_low = q.lower()
            hit = 0
            for cat in active:
                phrases = category_terms[cat]
                cat_hit = False
                for ph in phrases:
                    ph_low = ph.lower()
                    if ph_low in q_low:
                        cat_hit = True
                        break
                    tokens = [t for t in re.split(r"[^a-zA-Z0-9+.-]+", ph_low) if len(t) > 2]
                    if tokens and any(t in q_low for t in tokens):
                        cat_hit = True
                        break
                if cat_hit:
                    hit += 1
            scores.append(hit / len(active))
        return sum(scores) / len(scores)

    def _rule_based_plan(self, question: str, frame: dict[str, list[str]], locked_concepts: list[str]) -> ResearchPlan:
        queries = self._rule_based_subqueries(question, frame, locked_concepts)[:SUBQUERY_COUNT]
        if len(queries) < SUBQUERY_COUNT:
            queries.extend(["related work survey"] * (SUBQUERY_COUNT - len(queries)))
        queries = self._enforce_locked_concepts(queries[:SUBQUERY_COUNT], locked_concepts)

        return ResearchPlan(
            research_intent=self._build_research_intent(question, frame, locked_concepts),
            intent_slots=frame,
            sub_queries=queries,
            hidden_assumptions=self._build_assumptions(frame),
            clarification_questions=self._build_clarifications(frame),
        )

    def _mock_score(self, question: str, paper: dict, complementarity: float, quality_signal: float) -> PaperScore:
        # Improve multilingual robustness in mock mode by appending inferred English concepts.
        q_aug = (question or "").strip()
        hints = self._extract_hint_terms(question)
        if hints:
            q_aug = f"{q_aug} {' '.join(hints)}"

        content = 30 + 70 * self._mock_overlap(q_aug, paper.get("title", "") + " " + paper.get("abstract", ""))
        method = 45 + 40 * self._mock_overlap(q_aug, paper.get("abstract", ""))
        year = paper.get("year") or 2018
        timeliness = max(20.0, min(100.0, 60 + (year - 2018) * 4))
        total = content * 0.35 + method * 0.25 + timeliness * 0.15 + quality_signal * 0.15 + complementarity * 0.10

        return PaperScore(
            content_relevance=round(content, 2),
            method_relevance=round(method, 2),
            timeliness=round(timeliness, 2),
            quality_signal=round(quality_signal, 2),
            complementarity=round(complementarity, 2),
            total=round(total, 2),
            rationale="基于词项重叠与元数据先验的本地模拟评分。",
        )

    def _mock_synthesis(self, question: str, papers_payload: list[dict]) -> str:
        top = papers_payload[:3]
        support_refs = []
        caution_refs = []
        for row in papers_payload:
            title = (row.get("title") or "").lower()
            ref = row.get("ref_id", "P?")
            if any(k in title for k in ["improve", "gain", "outperform", "better"]):
                support_refs.append(ref)
            if any(k in title for k in ["review", "limitation", "challenge", "risk", "failure"]):
                caution_refs.append(ref)

        lines = [
            "## 直接回答",
            f"针对问题“{question}”，当前检索结果显示：部分论文支持相关结论，但结论仍需结合场景与证据强度谨慎解读。",
            "",
            "## 证据支撑的关键结论",
        ]
        for row in top:
            rid = row["ref_id"]
            lines.append(f"- 论文《{row['title']}》提供了与问题相关的证据 [{rid}]。")
            for ev in row.get("evidence", [])[:1]:
                lines.append(f"  证据片段可定位到对应页码 [{rid} p.{ev.get('page', 1)}]。")
        lines.extend(
            [
                "",
                "## 文献分歧与不同立场",
                "- 部分论文强调性能提升，另一些论文强调鲁棒性、边界条件和限制，因此结论并不完全一致。",
                f"- 倾向支持的参考：{', '.join(support_refs) if support_refs else '未识别'}",
                f"- 倾向谨慎的参考：{', '.join(caution_refs) if caution_refs else '未识别'}",
                "",
                "## 论文入选原因（多维评分）",
                "- 入选依据为多维评分：内容相关性、方法相关性、时效性、质量信号与互补性。",
                "",
                "## 当前不确定性",
                "- 受限于可获取全文、实验可比性与数据集差异，现阶段结论仍存在不确定性。",
            ]
        )
        return "\n".join(lines)

    def _mock_embedding(self, text: str, dim: int = 256) -> list[float]:
        values = [0.0] * dim
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]
        if not tokens:
            return values
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = digest[0] % dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            values[idx] += sign * (1 + (digest[2] / 255.0))
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
