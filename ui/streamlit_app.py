from __future__ import annotations

import asyncio
import inspect
import re
import sys
from pathlib import Path

import streamlit as st

# Allow running from any current working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline import PaperEvaluationAgent, RunOptions

st.set_page_config(page_title="论文搜索与综合评估智能体", layout="wide")


def _quality_components(paper):
    citation_signal = min(100.0, (paper.citation_count / 300.0) * 100.0)
    venue_signal = 70.0 if paper.venue and paper.venue.lower() != "arxiv" else 50.0
    author_signal = min(100.0, 40.0 + len(paper.authors) * 8.0)
    weighted = (
        citation_signal * 0.45
        + venue_signal * 0.35
        + author_signal * 0.20
    )
    return {
        "citation_signal": round(citation_signal, 2),
        "venue_signal": round(venue_signal, 2),
        "author_signal": round(author_signal, 2),
        "quality_signal_weighted": round(weighted, 2),
    }


def _total_contributions(score):
    c_content = round(score.content_relevance * 0.35, 2)
    c_method = round(score.method_relevance * 0.25, 2)
    c_time = round(score.timeliness * 0.15, 2)
    c_quality = round(score.quality_signal * 0.15, 2)
    c_comp = round(score.complementarity * 0.10, 2)
    calc_total = round(c_content + c_method + c_time + c_quality + c_comp, 2)
    return {
        "content": c_content,
        "method": c_method,
        "timeliness": c_time,
        "quality": c_quality,
        "complementarity": c_comp,
        "calc_total": calc_total,
    }


def _paper_match_fields(paper):
    # Backward-compatible for old session objects that were created before
    # query_match_score/concept_hit_count/matched_concepts were introduced.
    query_match_score = float(getattr(paper, "query_match_score", 0.0) or 0.0)
    concept_hit_count = int(getattr(paper, "concept_hit_count", 0) or 0)
    matched_concepts = getattr(paper, "matched_concepts", [])
    if matched_concepts is None:
        matched_concepts = []
    if not isinstance(matched_concepts, list):
        matched_concepts = [str(matched_concepts)]
    matched_concepts = [str(x) for x in matched_concepts if x is not None]
    return query_match_score, concept_hit_count, matched_concepts


def _parse_locked_concepts(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [x.strip() for x in re.split(r"[,，;；\n]+", raw) if x.strip()]
    uniq = []
    for p in parts:
        low = p.lower()
        if low not in uniq:
            uniq.append(low)
    return uniq


def _build_run_options(source: str, per_query_limit: int, ingest_top_n: int, locked_concepts: list[str]):
    kwargs = {
        "source": source,
        "per_query_limit": per_query_limit,
        "ingest_top_n": ingest_top_n,
        "locked_concepts": locked_concepts,
    }
    try:
        params = inspect.signature(RunOptions).parameters
        if "locked_concepts" not in params:
            kwargs.pop("locked_concepts", None)
    except Exception:
        pass

    try:
        return RunOptions(**kwargs)
    except TypeError:
        kwargs.pop("locked_concepts", None)
        return RunOptions(**kwargs)


def _extract_key_concepts(
    sub_queries: list[str],
    locked_concepts: list[str] | None = None,
    top_k: int = 8,
) -> list[str]:
    locked_concepts = [c.strip().lower() for c in (locked_concepts or []) if c and c.strip()]
    if locked_concepts:
        # When user locks concepts, treat them as the authoritative key concepts.
        return locked_concepts[:top_k]
    stopwords = {
        "with",
        "from",
        "that",
        "this",
        "into",
        "for",
        "and",
        "the",
        "are",
        "using",
        "based",
        "over",
        "under",
        "across",
        "than",
        "does",
        "what",
        "when",
        "study",
        "empirical",
        "benchmark",
        "ablation",
        "failure",
        "analysis",
        "systematic",
        "review",
        "literature",
        "comparison",
        "evaluation",
        "method",
        "methods",
        "trade",
        "off",
        "cost",
        "latency",
    }
    freq: dict[str, int] = {}
    for q in sub_queries:
        tokens = [t.lower() for t in re.split(r"[^a-zA-Z0-9+.-]+", q) if len(t) > 2]
        for t in tokens:
            if t in stopwords:
                continue
            if t.isdigit():
                continue
            freq[t] = freq.get(t, 0) + 1
    ranked = [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]

    # Locked concepts are always first-class key concepts.
    result: list[str] = []
    for c in locked_concepts:
        if c not in result:
            result.append(c)
    for w in ranked:
        if w not in result:
            result.append(w)
        if len(result) >= top_k:
            break
    return result[:top_k]


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "agent" not in st.session_state:
    st.session_state.agent = PaperEvaluationAgent()
if "last_locked_concepts" not in st.session_state:
    st.session_state.last_locked_concepts = []

st.title("论文搜索与综合评估智能体")
st.caption(
    "输入研究问题后，系统会自动拆解子查询、检索论文、进行多维评分，并输出带证据引用的综合结论。"
)

with st.sidebar:
    st.header("运行设置")
    source = st.selectbox("检索来源", ["all", "semantic_scholar", "arxiv"], index=0)
    per_query_limit = st.slider("每个子查询返回上限", min_value=2, max_value=12, value=6)
    ingest_top_n = st.slider("尝试解析 PDF 的论文数量", min_value=0, max_value=10, value=6)
    locked_concepts_text = st.text_input(
        "人工锁定关键概念（逗号分隔）",
        value="",
        placeholder="例如: retrieval augmented generation, reranker, robustness",
        help="锁定后，Agent 生成的每条子查询都会强制包含这些概念。",
    )
    locked_concepts = _parse_locked_concepts(locked_concepts_text)
    if locked_concepts:
        st.caption(f"当前锁定概念: {', '.join(locked_concepts)}")

    st.header("检索日志")
    if st.session_state.last_result:
        for item in st.session_state.last_result.search_log:
            status = item.get("status", "ok")
            if status == "ok":
                st.write(f"{item['query']} -> {item['hits']} ({item['source']})")
            else:
                st.warning(
                    f"{item['query']} -> 错误 ({item['source']}): {item.get('error', 'unknown')}"
                )
    else:
        st.info("先提一个问题，即可看到检索轨迹。")

    st.header("评分概览")
    if st.session_state.last_result and st.session_state.last_result.scored_papers:
        for idx, sp in enumerate(st.session_state.last_result.scored_papers[:6], start=1):
            st.markdown(f"**P{idx}** {sp.paper.title[:72]}")
            st.write(f"总分: {sp.score.total}")
            st.write(
                f"内容:{sp.score.content_relevance} 方法:{sp.score.method_relevance} 时效:{sp.score.timeliness}"
            )
            st.write(
                f"质量:{sp.score.quality_signal} 互补:{sp.score.complementarity}"
            )
            st.divider()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("请输入研究问题...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索论文、抽取证据并生成综合结论..."):
            result = asyncio.run(
                st.session_state.agent.run(
                    question,
                    options=_build_run_options(
                        source=source,
                        per_query_limit=per_query_limit,
                        ingest_top_n=ingest_top_n,
                        locked_concepts=locked_concepts,
                    ),
                )
            )
            st.session_state.last_result = result
            st.session_state.last_locked_concepts = locked_concepts
            st.markdown(result.final_answer_markdown)

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.last_result.final_answer_markdown}
    )

if st.session_state.last_result:
    plan = st.session_state.last_result.plan
    used_locked_concepts = st.session_state.last_locked_concepts or []

    st.subheader("问题理解与拆解（Agent Planner）")
    st.markdown(f"**核心研究意图**：{plan.research_intent}")
    if used_locked_concepts:
        st.markdown(f"**人工锁定关键概念（本次运行）**：{', '.join(used_locked_concepts)}")

    concepts = _extract_key_concepts(plan.sub_queries, locked_concepts=used_locked_concepts)
    if concepts:
        st.markdown("**关键概念（锁定概念优先 + 子查询归纳）**")
        st.write(", ".join(concepts))

    st.markdown("**可独立搜索的子查询（英文）**")
    for i, q in enumerate(plan.sub_queries, start=1):
        st.write(f"{i}. {q}")

    st.markdown("**识别到的隐含前提**")
    if plan.hidden_assumptions:
        for item in plan.hidden_assumptions:
            st.write(f"- {item}")
    else:
        st.write("- 暂未识别到明显隐含前提。")

    st.markdown("**需用户确认的问题**")
    if plan.clarification_questions:
        for i, item in enumerate(plan.clarification_questions, start=1):
            st.write(f"{i}. {item}")

        with st.form("clarification_form"):
            st.caption("可选：补充你的约束或偏好后，一键重跑分析。")
            answer_keys = []
            for i, cq in enumerate(plan.clarification_questions, start=1):
                key = f"clarify_answer_{i}"
                answer_keys.append((cq, key))
                st.text_input(f"Q{i}: {cq}", key=key)
            submitted = st.form_submit_button("基于澄清重跑")

        if submitted:
            answer_lines = []
            for cq, key in answer_keys:
                ans = (st.session_state.get(key) or "").strip()
                if ans:
                    answer_lines.append(f"- {cq}: {ans}")
            if not answer_lines:
                st.warning("你还没有填写澄清信息，已取消重跑。")
            else:
                clarification_text = "\n".join(answer_lines)
                augmented_question = (
                    f"{st.session_state.last_result.question}\n\n"
                    f"用户澄清补充：\n{clarification_text}"
                )
                st.session_state.messages.append(
                    {"role": "user", "content": f"补充澄清信息：\n{clarification_text}"}
                )
                with st.spinner("已接收澄清信息，正在重新检索与评分..."):
                    rerun_result = asyncio.run(
                        st.session_state.agent.run(
                            augmented_question,
                            options=_build_run_options(
                                source=source,
                                per_query_limit=per_query_limit,
                                ingest_top_n=ingest_top_n,
                                locked_concepts=locked_concepts,
                            ),
                        )
                    )
                st.session_state.last_result = rerun_result
                st.session_state.last_locked_concepts = locked_concepts
                st.session_state.messages.append(
                    {"role": "assistant", "content": rerun_result.final_answer_markdown}
                )
                st.rerun()
    else:
        st.write("- 当前无需额外确认，系统已直接执行。")

    st.divider()

    st.subheader("检索论文列表（英文原文）")
    paper_rows = []
    for idx, sp in enumerate(st.session_state.last_result.scored_papers, start=1):
        query_match_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
        paper_rows.append(
            {
                "ref": f"P{idx}",
                "title": sp.paper.title,
                "year": sp.paper.year,
                "venue": sp.paper.venue,
                "source": sp.paper.source,
                "citation_count": sp.paper.citation_count,
                "query_match_score": query_match_score,
                "concept_hit_count": concept_hit_count,
                "matched_concepts": ", ".join(matched_concepts),
                "total_score": sp.score.total,
                "content": sp.score.content_relevance,
                "method": sp.score.method_relevance,
                "timeliness": sp.score.timeliness,
                "quality": sp.score.quality_signal,
                "complementarity": sp.score.complementarity,
                "paper_url": sp.paper.paper_url,
            }
        )
    if paper_rows:
        st.dataframe(paper_rows, use_container_width=True)
    else:
        st.info("当前问题未检索到论文。")

    st.subheader("查询-论文匹配分（命中概念）")
    st.caption("匹配分 = 概念命中数 + 词项相关性分，用于展示检索阶段的相关性，不等同于最终综合评分。")
    match_rows = []
    for idx, sp in enumerate(st.session_state.last_result.scored_papers, start=1):
        query_match_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
        match_rows.append(
            {
                "ref": f"P{idx}",
                "title": sp.paper.title,
                "query_match_score": query_match_score,
                "concept_hit_count": concept_hit_count,
                "matched_concepts": ", ".join(matched_concepts) if matched_concepts else "N/A",
            }
        )
    st.dataframe(match_rows, use_container_width=True)

    with st.expander("问题拆解结果"):
        st.json(st.session_state.last_result.plan.model_dump())

    with st.expander("证据表（论文保持英文原文）"):
        rows = []
        for idx, sp in enumerate(st.session_state.last_result.scored_papers, start=1):
            for ev in sp.evidence[:2]:
                rows.append(
                    {
                        "ref": f"P{idx}",
                        "title": sp.paper.title,
                        "page": ev.page,
                        "snippet": ev.text[:220],
                    }
                )
        st.dataframe(rows, use_container_width=True)

    st.subheader("评分过程明细（逐篇可解释）")
    st.caption("展示五维分数、质量信号拆解、总分加权贡献，便于核对每篇论文是如何得到最终排名。")
    for idx, sp in enumerate(st.session_state.last_result.scored_papers, start=1):
        query_match_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
        quality_parts = _quality_components(sp.paper)
        contrib = _total_contributions(sp.score)

        with st.expander(f"P{idx} | {sp.paper.title[:120]}"):
            st.markdown("**0) 查询匹配信息（检索阶段）**")
            st.write(
                {
                    "query_match_score": query_match_score,
                    "concept_hit_count": concept_hit_count,
                    "matched_concepts": matched_concepts,
                }
            )

            st.markdown("**1) 原始五维分数**")
            st.write(
                {
                    "content_relevance": sp.score.content_relevance,
                    "method_relevance": sp.score.method_relevance,
                    "timeliness": sp.score.timeliness,
                    "quality_signal": sp.score.quality_signal,
                    "complementarity": sp.score.complementarity,
                }
            )

            st.markdown("**2) quality_signal 计算过程**")
            st.code(
                (
                    "quality_signal = 0.45*citation_signal + 0.35*venue_signal + 0.20*author_signal\n"
                    f"citation_signal = min(100, citation_count/300*100) = {quality_parts['citation_signal']}\n"
                    f"venue_signal = 70 (non-arXiv) else 50 = {quality_parts['venue_signal']}\n"
                    f"author_signal = min(100, 40 + 8*author_count) = {quality_parts['author_signal']}\n"
                    f"quality_signal (calc) = {quality_parts['quality_signal_weighted']}"
                ),
                language="text",
            )
            st.caption(
                f"论文元数据：citation_count={sp.paper.citation_count}, "
                f"venue={sp.paper.venue or 'N/A'}, author_count={len(sp.paper.authors)}"
            )

            st.markdown("**3) total 总分加权过程**")
            st.code(
                (
                    "total = 0.35*content + 0.25*method + 0.15*timeliness + 0.15*quality + 0.10*complementarity\n"
                    f"= 0.35*{sp.score.content_relevance} ({contrib['content']})\n"
                    f"+ 0.25*{sp.score.method_relevance} ({contrib['method']})\n"
                    f"+ 0.15*{sp.score.timeliness} ({contrib['timeliness']})\n"
                    f"+ 0.15*{sp.score.quality_signal} ({contrib['quality']})\n"
                    f"+ 0.10*{sp.score.complementarity} ({contrib['complementarity']})\n"
                    f"= {contrib['calc_total']} (stored: {sp.score.total})"
                ),
                language="text",
            )

            st.markdown("**4) 评分理由（LLM 输出）**")
            st.write(sp.score.rationale)
