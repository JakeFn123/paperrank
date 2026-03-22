from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pipeline import PaperEvaluationAgent, RunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="论文搜索与综合评估智能体")
    parser.add_argument("question", type=str, nargs="?", default="在企业知识库问答中，reranker 是否显著提升 RAG 稳定性？")
    parser.add_argument("--source", type=str, default="all", choices=["all", "semantic_scholar", "arxiv"])
    parser.add_argument("--per-query-limit", type=int, default=6)
    parser.add_argument("--ingest-top-n", type=int, default=6)
    parser.add_argument("--max-papers", type=int, default=30, help="Upper bound of candidate papers (hard-capped at 30).")
    parser.add_argument(
        "--locked-concepts",
        type=str,
        default="",
        help="Comma-separated concepts that must appear in generated sub-queries.",
    )
    parser.add_argument(
        "--intent-slots-json",
        type=str,
        default="",
        help='JSON string, e.g. {"subject":["llm agent"],"outcome":["efficiency"]}',
    )
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    agent = PaperEvaluationAgent()
    slots_override = {}
    if args.intent_slots_json.strip():
        try:
            raw = json.loads(args.intent_slots_json)
            if isinstance(raw, dict):
                slots_override = {str(k): v for k, v in raw.items()}
        except Exception:
            slots_override = {}
    result = await agent.run(
        args.question,
        options=RunOptions(
            source=args.source,
            per_query_limit=args.per_query_limit,
            ingest_top_n=args.ingest_top_n,
            max_papers=args.max_papers,
            locked_concepts=[x.strip() for x in args.locked_concepts.split(",") if x.strip()],
            intent_slots_override=slots_override,
        ),
    )

    print("\n=== 问题拆解 ===")
    print("研究意图:", result.plan.research_intent)
    print("子查询（英文检索词）:")
    for q in result.plan.sub_queries:
        print("-", q)
    if result.plan.intent_slots:
        print("意图槽位:", result.plan.intent_slots)
    if result.plan.hidden_assumptions:
        print("隐含前提:")
        for h in result.plan.hidden_assumptions:
            print("-", h)

    print("\n=== 检索日志 ===")
    for row in result.search_log:
        print(f"- query={row['query']} hits={row['hits']} source={row['source']}")

    print("\n=== 评分前五 ===")
    for i, sp in enumerate(result.scored_papers[:5], start=1):
        s = sp.score
        print(f"{i}. {sp.paper.title[:100]}")
        print(
            f"   total={s.total} | rerank={sp.paper.rerank_score} | content={s.content_relevance} | method={s.method_relevance} "
            f"| timeliness={s.timeliness} | quality={s.quality_signal} | complementarity={s.complementarity}"
        )

    print("\n=== 综合结论 ===")
    print(result.final_answer_markdown)
    if result.evidence_audit:
        print("\n=== 证据审查 ===")
        print(result.evidence_audit)

    if result.task_board_snapshot:
        print("\n=== Task System ===")
        for t in result.task_board_snapshot:
            print(
                f"- #{t['id']} [{t['status']}] {t['title']} "
                f"(assignee={t['assignee']}) summary={t.get('result_summary', '')}"
            )

    if result.loop_trace:
        print("\n=== Agent Loop Trace ===")
        for row in result.loop_trace:
            print(f"- {row.get('stage')}: {row.get('message')}")


if __name__ == "__main__":
    asyncio.run(main())
