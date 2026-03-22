from __future__ import annotations

import json
from typing import Iterable


def build_decompose_prompt(question: str, subquery_count: int) -> str:
    return f"""
你是一个科研问题分析助手。

任务：
1) 理解用户真正的研究意图。
2) 将问题拆解为 {subquery_count} 个可独立检索的子查询。
3) 识别隐含前提，并给出需要用户澄清的问题。

只返回 JSON，格式如下：
{{
  "research_intent": "...",
  "intent_slots": {{
    "subject": ["..."],
    "intervention": ["..."],
    "outcome": ["..."],
    "context": ["..."],
    "evaluation": ["..."]
  }},
  "sub_queries": ["..."],
  "hidden_assumptions": ["..."],
  "clarification_questions": ["..."]
}}

约束：
- sub_queries 必须使用英文技术短语，便于论文召回。
- intent_slots 里每个槽位都尽量给出 1-3 个词组。
- JSON 里除 sub_queries 外，其余字段默认用中文。
- 不要输出 markdown。

用户问题：
{question}
""".strip()


def build_llm_scoring_prompt(question: str, paper: dict, evidence_snippets: Iterable[dict], years: list[int]) -> str:
    sample = {
        "question": question,
        "paper": {
            "title": paper.get("title"),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "citation_count": paper.get("citation_count", 0),
        },
        "evidence": list(evidence_snippets),
        "year_distribution": years,
    }

    return f"""
你在做单篇论文相关性评分。

只返回 JSON：
{{
  "content_relevance": 0-100,
  "method_relevance": 0-100,
  "timeliness": 0-100,
  "rationale": "简短中文解释"
}}

评分标准：
- content_relevance：论文是否直接回答用户问题。
- method_relevance：方法/实验设定是否可迁移到用户场景。
- timeliness：相对候选论文年份分布是否仍有时效价值。

输入数据：
{json.dumps(sample, ensure_ascii=False, indent=2)}
""".strip()


def build_synthesis_prompt(question: str, papers_payload: list[dict]) -> str:
    return f"""
你是“论文搜索与综合评估智能体”。

目标：
- 对用户问题给出直接回答。
- 以论文证据支撑结论。
- 存在冲突证据时必须呈现分歧。

输出必须是中文 Markdown，且按以下顺序：
1. ## 直接回答
2. ## 证据支撑的关键结论
3. ## 文献分歧与不同立场
4. ## 论文入选原因（多维评分）
5. ## 当前不确定性

引用格式要求：
- 每条核心观点都要带引用，如 [P1]。
- 有页码证据时尽量使用 [P1 p.3]。
- 论文标题、作者、摘要必须保持原文语言（通常是英文），不要翻译。

禁止编造任何未在证据中出现的事实。

用户问题：
{question}

论文卡片：
{json.dumps(papers_payload, ensure_ascii=False, indent=2)}
""".strip()
