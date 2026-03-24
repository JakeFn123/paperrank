---
name: academic-retrieval
description: 学术检索执行技能，负责多源召回、去重、概念匹配、两阶段重排与检索日志审计。
tags: retrieval,mcp,arxiv,semantic-scholar,rerank
---

# Academic Retrieval Skill

## 技能说明

该技能将 `sub_queries` 转化为高质量候选论文集，核心是“多源召回 + 后处理重排 + 日志审计”。

## 触发条件（When to use）

- 需要从 arXiv / Semantic Scholar 召回候选论文。
- 需要结果可解释（为何被召回、为何排序靠前）。
- 需要在 `<=30` 候选约束下保持召回率。

## 不适用边界（When NOT to use）

- 用户已指定固定论文集合，仅做评分与总结。
- 任务目标是全文事实核对（应走 evidence 流程而非仅检索）。
- 用户要求离线无外部 API 调用。

## 必要输入

- `question`
- `sub_queries`
- `source`
- `per_query_limit`
- `max_papers`（<=30）
- `locked_concepts`

## 工作模式决策

1. `Broad Recall Mode`
- 适用于探索类问题；强调召回池覆盖（默认 recall pool=100）。

2. `Precision Mode`
- 适用于强锁定概念问题；强调 lock hit 与 rerank 精排。

3. `Fallback Mode`
- API 429/0 命中时自动降级（简化查询、单源保底、lexical 重排）。

## 核心执行步骤

### 阶段 1：多源召回

- MCP 调用 `search_papers`；
- 计算 `effective_limit`，动态拉高每个子查询召回量。

### 阶段 2：0 命中重试

- 子查询无命中时，自动进行“简化查询重试”（前 10 个核心词）。

### 阶段 3：标准化与去重

- 统一字段类型；
- `paper_id` 或稳定哈希去重；
- 合并摘要与 URL。

### 阶段 4：召回解释打分

- 计算 `concept_hit_count`、`matched_concepts`、`query_match_score`。

### 阶段 5：两阶段重排

- recall pool -> Cross-Encoder（失败回退 lexical）-> top_k。
- 写入 `rerank_score`。

### 阶段 6：日志审计

- 输出逐 query 日志 + `__postprocess__` 汇总日志。

## 质量门禁（量化）

- 输出论文数 `<= max_papers <= 30`。
- 每篇论文必须有 `query_match_score` 与 `rerank_score`。
- `search_log` 必须包含 `__postprocess__`。
- 若非全失败：`search_log` 至少 1 条 `status=ok`。

## 失败回退策略

- 单源失败：保留可用源结果。
- Cross-Encoder 不可用：降级 lexical。
- 概念过滤过严导致空集：回退部分匹配。

## 标准输出模板（Output Contract）

```json
{
  "papers": [
    {
      "paper_id": "...",
      "title": "...",
      "query_match_score": 0.0,
      "rerank_score": 0.0,
      "concept_hit_count": 0,
      "matched_concepts": ["..."]
    }
  ],
  "search_log": [
    {"query": "...", "hits": 0, "source": "all", "status": "ok"},
    {"query": "__postprocess__", "hits": 0, "source": "local", "status": "ok"}
  ]
}
```

## 最小样例

输入：
- `source=all`, `max_papers=30`, `locked_concepts=["reranker", "robustness"]`

输出特征：
- `papers` 中多数论文 `matched_concepts` 含 `Locked:*` 或相关概念；
- `search_log` 末尾含 `__postprocess__`。
