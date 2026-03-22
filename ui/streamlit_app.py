from __future__ import annotations

import asyncio
import html
import inspect
import re
import statistics
import sys
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

# Allow running from any current working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline import PaperEvaluationAgent, RunOptions

st.set_page_config(page_title="PaperRank", layout="wide")


# ---------- Style ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
  --bg: #f7f9fc;
  --card: #ffffff;
  --line: #e5e9f0;
  --text: #1f2937;
  --muted: #6b7280;
  --brand: #1a73e8;
  --brand-dark: #174ea6;
  --ok: #0f766e;
  --chip: #eef4ff;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', 'Noto Sans SC', sans-serif;
}

.stApp {
  background: radial-gradient(circle at 20% -10%, #eef4ff 0%, var(--bg) 42%, #f8fafc 100%);
}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

.pr-hero {
  background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 60%, #f9fafb 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 22px;
  margin-bottom: 14px;
}

.pr-title {
  font-size: 1.6rem;
  font-weight: 700;
  color: #111827;
  margin: 0 0 4px 0;
}

.pr-subtitle {
  color: var(--muted);
  font-size: 0.96rem;
  margin: 0;
}

.pr-kpi-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--card);
  padding: 12px 14px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

.pr-kpi-label {
  color: var(--muted);
  font-size: 0.78rem;
  margin-bottom: 3px;
}

.pr-kpi-value {
  color: var(--text);
  font-weight: 700;
  font-size: 1.24rem;
  line-height: 1.15;
}

.pr-kpi-sub {
  color: #4b5563;
  font-size: 0.75rem;
  margin-top: 2px;
}

.pr-scholar-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  box-shadow: 0 1px 8px rgba(30, 41, 59, 0.04);
}

.pr-scholar-title a {
  color: #1a0dab;
  text-decoration: none;
  font-size: 1.03rem;
  font-weight: 600;
}

.pr-scholar-title a:hover {
  color: #174ea6;
  text-decoration: underline;
}

.pr-scholar-title span {
  color: #1a0dab;
  font-size: 1.03rem;
  font-weight: 600;
}

.pr-scholar-meta {
  margin-top: 4px;
  color: #4b5563;
  font-size: 0.84rem;
}

.pr-scholar-snippet {
  margin-top: 8px;
  color: #1f2937;
  font-size: 0.9rem;
  line-height: 1.45;
}

.pr-badges {
  margin-top: 10px;
}

.pr-badge {
  display: inline-block;
  padding: 2px 8px;
  margin: 0 6px 6px 0;
  border-radius: 999px;
  border: 1px solid #dbe7ff;
  background: var(--chip);
  color: #174ea6;
  font-size: 0.75rem;
}

.pr-side-box {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
}

.pr-side-title {
  font-weight: 700;
  font-size: 0.92rem;
  margin-bottom: 6px;
  color: #111827;
}

small, .stCaption {
  color: var(--muted) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def _quality_components(paper):
    citation_signal = min(100.0, (paper.citation_count / 300.0) * 100.0)
    venue_signal = 70.0 if paper.venue and paper.venue.lower() != "arxiv" else 50.0
    author_signal = min(100.0, 40.0 + len(paper.authors) * 8.0)
    weighted = citation_signal * 0.45 + venue_signal * 0.35 + author_signal * 0.20
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
    query_match_score = float(getattr(paper, "query_match_score", 0.0) or 0.0)
    rerank_score = float(getattr(paper, "rerank_score", 0.0) or 0.0)
    concept_hit_count = int(getattr(paper, "concept_hit_count", 0) or 0)
    matched_concepts = getattr(paper, "matched_concepts", [])
    if matched_concepts is None:
        matched_concepts = []
    if not isinstance(matched_concepts, list):
        matched_concepts = [str(matched_concepts)]
    matched_concepts = [str(x) for x in matched_concepts if x is not None]
    return query_match_score, rerank_score, concept_hit_count, matched_concepts


def _slot_values_to_text(values: list[str]) -> str:
    if not values:
        return ""
    return ", ".join([str(v) for v in values if str(v).strip()])


def _parse_slot_values(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [x.strip().lower() for x in re.split(r"[,，;；\n]+", raw) if x.strip()]
    uniq: list[str] = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return uniq


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


def _build_run_options(
    source: str,
    per_query_limit: int,
    ingest_top_n: int,
    max_papers: int,
    locked_concepts: list[str],
    intent_slots_override: dict[str, list[str]] | None = None,
):
    kwargs = {
        "source": source,
        "per_query_limit": per_query_limit,
        "ingest_top_n": ingest_top_n,
        "max_papers": max_papers,
        "locked_concepts": locked_concepts,
        "intent_slots_override": intent_slots_override or {},
    }
    try:
        params = inspect.signature(RunOptions).parameters
        if "max_papers" not in params:
            kwargs.pop("max_papers", None)
        if "locked_concepts" not in params:
            kwargs.pop("locked_concepts", None)
        if "intent_slots_override" not in params:
            kwargs.pop("intent_slots_override", None)
    except Exception:
        pass

    try:
        return RunOptions(**kwargs)
    except TypeError:
        kwargs.pop("locked_concepts", None)
        kwargs.pop("intent_slots_override", None)
        return RunOptions(**kwargs)


def _extract_key_concepts(sub_queries: list[str], locked_concepts: list[str] | None = None, top_k: int = 8) -> list[str]:
    locked_concepts = [c.strip().lower() for c in (locked_concepts or []) if c and c.strip()]
    if locked_concepts:
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
            if t in stopwords or t.isdigit():
                continue
            freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_k]]


def _kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
<div class="pr-kpi-card">
  <div class="pr-kpi-label">{html.escape(label)}</div>
  <div class="pr-kpi-value">{html.escape(value)}</div>
  <div class="pr-kpi-sub">{html.escape(sub)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _safe_external_url(raw_url: str) -> str | None:
    url = (raw_url or "").strip()
    if not url:
        return None

    if url.startswith("www."):
        url = f"https://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return url


def _compute_dashboard_metrics(result, locked_concepts: list[str], per_query_limit: int) -> dict:
    papers = result.scored_papers
    n = len(papers)

    match_scores = []
    rerank_scores = []
    concept_hits = []
    locked_hit_papers = 0
    for sp in papers:
        query_match_score, rerank_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
        match_scores.append(query_match_score)
        rerank_scores.append(rerank_score)
        concept_hits.append(concept_hit_count)
        if any(str(x).lower().startswith("locked:") for x in matched_concepts):
            locked_hit_papers += 1

    totals = [sp.score.total for sp in papers]
    citations = [sp.paper.citation_count for sp in papers]
    years = [sp.paper.year for sp in papers if sp.paper.year]

    hits_total = sum(int(x.get("hits", 0) or 0) for x in result.search_log)
    capacity = max(1, len(result.plan.sub_queries) * max(1, per_query_limit))

    source_dist: dict[str, int] = {}
    for sp in papers:
        source_dist[sp.paper.source] = source_dist.get(sp.paper.source, 0) + 1

    recent_count = sum(1 for y in years if y >= 2023)

    return {
        "paper_count": n,
        "avg_total": round(sum(totals) / n, 2) if n else 0.0,
        "avg_match": round(sum(match_scores) / n, 2) if n else 0.0,
        "avg_rerank": round(sum(rerank_scores) / n, 2) if n else 0.0,
        "hit_utilization": round(hits_total / capacity * 100, 1),
        "high_rel_count": sum(1 for x in match_scores if x >= 2.5),
        "avg_citations": round(sum(citations) / n, 1) if n else 0.0,
        "median_year": int(statistics.median(years)) if years else None,
        "latest_year": max(years) if years else None,
        "recent_ratio": round((recent_count / len(years) * 100), 1) if years else 0.0,
        "avg_concept_hits": round(sum(concept_hits) / n, 2) if n else 0.0,
        "locked_coverage": round((locked_hit_papers / n * 100), 1) if n and locked_concepts else None,
        "source_dist": source_dist,
    }


def _run_pipeline(
    question: str,
    source: str,
    per_query_limit: int,
    ingest_top_n: int,
    max_papers: int,
    locked_concepts: list[str],
    intent_slots_override: dict[str, list[str]] | None = None,
):
    result = asyncio.run(
        st.session_state.agent.run(
            question,
            options=_build_run_options(
                source=source,
                per_query_limit=per_query_limit,
                ingest_top_n=ingest_top_n,
                max_papers=max_papers,
                locked_concepts=locked_concepts,
                intent_slots_override=intent_slots_override or {},
            ),
        )
    )
    st.session_state.last_result = result
    st.session_state.last_locked_concepts = locked_concepts
    st.session_state.last_question = question


# ---------- Session ----------
if "agent" not in st.session_state:
    st.session_state.agent = PaperEvaluationAgent()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_locked_concepts" not in st.session_state:
    st.session_state.last_locked_concepts = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# Apply pending sample query before query_input widget is instantiated.
if st.session_state.pending_query:
    st.session_state.query_input = st.session_state.pending_query
    st.session_state.pending_query = ""


# ---------- Sidebar ----------
with st.sidebar:
    st.header("检索设置")
    source = st.selectbox("检索来源", ["all", "semantic_scholar", "arxiv"], index=0)
    per_query_limit = st.slider("每个子查询返回上限", min_value=2, max_value=12, value=10)
    max_papers = st.slider("候选论文上限", min_value=8, max_value=30, value=30)
    ingest_top_n = st.slider("尝试解析 PDF 的论文数量", min_value=0, max_value=10, value=6)

    locked_concepts_text = st.text_area(
        "人工锁定关键概念",
        key="locked_concepts_text",
        placeholder="例如: retrieval augmented generation, reranker, robustness",
        help="锁定后，系统会在子查询与检索排序中优先满足这些概念。",
        height=80,
    )
    locked_concepts = _parse_locked_concepts(locked_concepts_text)
    if locked_concepts:
        st.caption("当前锁定概念：" + ", ".join(locked_concepts))

    st.divider()
    st.subheader("检索日志")
    if st.session_state.last_result:
        st.dataframe(st.session_state.last_result.search_log, use_container_width=True)
    else:
        st.info("提交问题后显示检索日志")


# ---------- Main Header ----------
st.markdown(
    """
<div class="pr-hero">
  <p class="pr-title">PaperRank</p>
  <p class="pr-subtitle">按“检索结果优先、证据与评分可解释”的方式展示论文分析。默认中文分析，论文元数据保持英文。</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- Search Form ----------
with st.form("search_form", clear_on_submit=False):
    st.text_input(
        "研究问题",
        key="query_input",
        label_visibility="collapsed",
        placeholder="输入研究问题，例如：如何用 MDP 建模智能体任务执行并提升 LLM agent 执行效率？",
    )
    c1, c2 = st.columns([1.1, 6])
    with c1:
        run_btn = st.form_submit_button("搜索并评估", use_container_width=True)
    with c2:
        st.caption("提示：可在左侧锁定关键概念，提升检索对齐度。")

if run_btn and st.session_state.query_input.strip():
    with st.spinner("正在执行 Agent Loop：拆解 -> 检索 -> 评分 -> 综合..."):
        _run_pipeline(
            question=st.session_state.query_input.strip(),
            source=source,
            per_query_limit=per_query_limit,
            ingest_top_n=ingest_top_n,
            max_papers=max_papers,
            locked_concepts=locked_concepts,
        )


# ---------- Empty State ----------
if not st.session_state.last_result:
    st.info("输入问题并点击“搜索并评估”后查看结果。")
    sample_cols = st.columns(2)
    samples = [
        "How to model LLM agent task execution with MDP and deep learning?",
        "在企业知识库问答中，cross-encoder reranker 是否能提升 RAG 稳定性？",
    ]
    for col, text in zip(sample_cols, samples):
        with col:
            if st.button(text, use_container_width=True):
                st.session_state.pending_query = text
                st.rerun()
    st.stop()


# ---------- Dashboard Metrics ----------
result = st.session_state.last_result
used_locked_concepts = st.session_state.last_locked_concepts or []
metrics = _compute_dashboard_metrics(result, used_locked_concepts, per_query_limit)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    _kpi("候选论文", str(metrics["paper_count"]), "去重后进入评分")
with k2:
    _kpi("平均总分", f"{metrics['avg_total']}", "最终排序分")
with k3:
    _kpi("平均匹配分", f"{metrics['avg_match']}", "检索阶段相关性")
with k4:
    _kpi("平均重排分", f"{metrics['avg_rerank']}", "两阶段重排得分")
with k5:
    _kpi("高相关论文", str(metrics["high_rel_count"]), "query_match_score ≥ 2.5")
with k6:
    _kpi("平均引文", f"{metrics['avg_citations']}", "citation_count")

k7, k8, k9, k10 = st.columns(4)
with k7:
    _kpi("平均概念命中", f"{metrics['avg_concept_hits']}", "concept_hit_count")
with k8:
    if metrics["locked_coverage"] is None:
        _kpi("锁定概念覆盖", "N/A", "未设置锁定概念")
    else:
        _kpi("锁定概念覆盖", f"{metrics['locked_coverage']}%", "命中 Locked:* 标签论文占比")
with k9:
    if metrics["median_year"] is None:
        _kpi("年份中位数", "N/A", "无年份数据")
    else:
        _kpi("年份中位数", str(metrics["median_year"]), f"最新：{metrics['latest_year']}")
with k10:
    _kpi("命中利用率", f"{metrics['hit_utilization']}%", "hits / 理论上限")


# ---------- Tabs ----------
tab_results, tab_answer, tab_planner, tab_scoring, tab_loop = st.tabs(
    ["检索结果", "综合结论", "问题理解与拆解", "评分过程", "Agent Loop"]
)


with tab_results:
    left, right = st.columns([3.2, 1.2], gap="large")

    with left:
        st.subheader("论文结果流")
        st.caption("标题为链接（若可用），下方展示元信息、摘要片段与核心指标。")

        for idx, sp in enumerate(result.scored_papers, start=1):
            query_match_score, rerank_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
            title = html.escape(sp.paper.title or "Untitled")
            snippet = html.escape((sp.paper.abstract or "No abstract available.")[:420])
            venue = html.escape(sp.paper.venue or "Unknown venue")
            source_name = html.escape(sp.paper.source or "unknown")
            year = sp.paper.year if sp.paper.year else "N/A"
            cits = sp.paper.citation_count
            authors = ", ".join(sp.paper.authors[:4]) if sp.paper.authors else "Unknown authors"
            if len(sp.paper.authors) > 4:
                authors += " ..."
            authors = html.escape(authors)

            raw_url = sp.paper.paper_url or sp.paper.pdf_url or ""
            safe_url = _safe_external_url(raw_url)
            if safe_url:
                title_html = f'<a href="{html.escape(safe_url)}" target="_blank">[{idx}] {title}</a>'
            else:
                title_html = f"<span>[{idx}] {title}</span>"

            badges = [
                f"Total {sp.score.total}",
                f"Rerank {rerank_score}",
                f"Match {query_match_score}",
                f"Concept Hits {concept_hit_count}",
                f"Citations {cits}",
                f"Year {year}",
            ]
            badge_html = "".join([f'<span class="pr-badge">{html.escape(str(b))}</span>' for b in badges])

            st.markdown(
                f"""
<div class="pr-scholar-card">
  <div class="pr-scholar-title">{title_html}</div>
  <div class="pr-scholar-meta">{authors} · {venue} · {year} · {source_name}</div>
  <div class="pr-scholar-snippet">{snippet}</div>
  <div class="pr-badges">{badge_html}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            if matched_concepts:
                st.caption("匹配概念：" + ", ".join(matched_concepts))
            if raw_url and not safe_url:
                st.caption("链接格式异常，已禁用跳转。")

    with right:
        st.markdown('<div class="pr-side-box">', unsafe_allow_html=True)
        st.markdown('<div class="pr-side-title">检索分布</div>', unsafe_allow_html=True)
        st.write(metrics["source_dist"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pr-side-box">', unsafe_allow_html=True)
        st.markdown('<div class="pr-side-title">关键统计</div>', unsafe_allow_html=True)
        st.write(
            {
                "paper_count": metrics["paper_count"],
                "avg_total": metrics["avg_total"],
                "avg_match": metrics["avg_match"],
                "avg_rerank": metrics["avg_rerank"],
                "hit_utilization": f"{metrics['hit_utilization']}%",
                "median_year": metrics["median_year"],
                "latest_year": metrics["latest_year"],
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pr-side-box">', unsafe_allow_html=True)
        st.markdown('<div class="pr-side-title">检索日志</div>', unsafe_allow_html=True)
        st.dataframe(result.search_log, use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)


with tab_answer:
    st.subheader("综合评估结论")
    st.markdown(result.final_answer_markdown)
    audit = getattr(result, "evidence_audit", {}) or {}
    if audit:
        st.markdown("### 证据对齐审查")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("审查结果", "通过" if audit.get("passed") else "未完全通过")
        with c2:
            st.metric("置信等级", str(audit.get("confidence", "low")).upper())
        with c3:
            st.metric("引用论文数", str(len(audit.get("cited_refs", []) or [])))
        with st.expander("查看审查细节"):
            st.json(audit)


with tab_planner:
    plan = result.plan
    st.subheader("Planner 输出")
    st.markdown(f"**核心研究意图**：{plan.research_intent}")
    intent_slots = getattr(plan, "intent_slots", {}) or {}
    st.markdown("**结构化意图槽位（可编辑后重跑）**")
    with st.form("intent_slots_override_form"):
        subject_text = st.text_input("subject", value=_slot_values_to_text(intent_slots.get("subject", [])))
        intervention_text = st.text_input("intervention", value=_slot_values_to_text(intent_slots.get("intervention", [])))
        outcome_text = st.text_input("outcome", value=_slot_values_to_text(intent_slots.get("outcome", [])))
        context_text = st.text_input("context", value=_slot_values_to_text(intent_slots.get("context", [])))
        evaluation_text = st.text_input("evaluation", value=_slot_values_to_text(intent_slots.get("evaluation", [])))
        slot_submit = st.form_submit_button("按槽位重跑")

    if slot_submit:
        slot_override = {
            "subject": _parse_slot_values(subject_text),
            "intervention": _parse_slot_values(intervention_text),
            "outcome": _parse_slot_values(outcome_text),
            "context": _parse_slot_values(context_text),
            "evaluation": _parse_slot_values(evaluation_text),
        }
        with st.spinner("按手工槽位重跑中：拆解 -> 检索 -> 评分 -> 综合..."):
            _run_pipeline(
                question=result.question,
                source=source,
                per_query_limit=per_query_limit,
                ingest_top_n=ingest_top_n,
                max_papers=max_papers,
                locked_concepts=locked_concepts,
                intent_slots_override=slot_override,
            )
        st.rerun()

    if used_locked_concepts:
        st.markdown("**人工锁定关键概念（本次运行）**：" + ", ".join(used_locked_concepts))

    concepts = _extract_key_concepts(plan.sub_queries, locked_concepts=used_locked_concepts)
    if concepts:
        st.markdown("**关键概念（锁定优先 + 子查询归纳）**")
        st.write(", ".join(concepts))

    st.markdown("**可独立检索的子查询（英文）**")
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
            st.caption("补充约束后，一键重跑分析。")
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
                    f"{result.question}\n\n"
                    f"用户澄清补充：\n{clarification_text}"
                )
                with st.spinner("已接收澄清信息，正在重新检索与评分..."):
                    _run_pipeline(
                        question=augmented_question,
                        source=source,
                        per_query_limit=per_query_limit,
                        ingest_top_n=ingest_top_n,
                        max_papers=max_papers,
                        locked_concepts=locked_concepts,
                        intent_slots_override=intent_slots,
                    )
                st.rerun()


with tab_scoring:
    st.subheader("检索列表与评分细节")

    paper_rows = []
    for idx, sp in enumerate(result.scored_papers, start=1):
        query_match_score, rerank_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
        paper_rows.append(
            {
                "ref": f"P{idx}",
                "title": sp.paper.title,
                "year": sp.paper.year,
                "venue": sp.paper.venue,
                "source": sp.paper.source,
                "citation_count": sp.paper.citation_count,
                "rerank_score": rerank_score,
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

    st.dataframe(paper_rows, use_container_width=True)

    st.caption("两阶段检索：先召回再重排（rerank_score）；匹配分用于解释召回原因，总分用于最终排序。")

    with st.expander("评分过程明细（逐篇可解释）"):
        for idx, sp in enumerate(result.scored_papers, start=1):
            query_match_score, rerank_score, concept_hit_count, matched_concepts = _paper_match_fields(sp.paper)
            quality_parts = _quality_components(sp.paper)
            contrib = _total_contributions(sp.score)

            with st.expander(f"P{idx} | {sp.paper.title[:120]}"):
                st.markdown("**0) 查询匹配信息（检索阶段）**")
                st.write(
                    {
                        "query_match_score": query_match_score,
                        "rerank_score": rerank_score,
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
                        f"citation_signal = {quality_parts['citation_signal']}\n"
                        f"venue_signal = {quality_parts['venue_signal']}\n"
                        f"author_signal = {quality_parts['author_signal']}\n"
                        f"quality_signal (calc) = {quality_parts['quality_signal_weighted']}"
                    ),
                    language="text",
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


with tab_loop:
    st.subheader("Agent Loop / Task System")
    task_board_snapshot = getattr(result, "task_board_snapshot", []) or []
    loop_trace = getattr(result, "loop_trace", []) or []
    audit = getattr(result, "evidence_audit", {}) or {}

    if task_board_snapshot:
        st.markdown("**任务看板（持久化任务系统）**")
        st.dataframe(task_board_snapshot, use_container_width=True)
    else:
        st.info("当前运行未返回任务看板信息。")

    if loop_trace:
        st.markdown("**执行轨迹（Agent Loop）**")
        st.dataframe(loop_trace, use_container_width=True)

        config_rows = [x for x in loop_trace if x.get("stage") == "config"]
        if config_rows:
            details = config_rows[-1].get("details", {})
            with st.expander("Tools / Skills 定义（本次运行）"):
                st.markdown("**Tools**")
                st.json(details.get("tools", []))
                st.markdown("**Subagents + Skills**")
                st.json(details.get("skills", []))

    if audit:
        st.markdown("**证据审查结果（Synthesis 后置）**")
        st.json(audit)
