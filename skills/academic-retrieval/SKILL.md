---
name: academic-retrieval
description: 学术检索执行技能，负责多源召回、去重、概念匹配、两阶段重排与检索日志审计。
tags: retrieval,mcp,arxiv,semantic-scholar,rerank
---

# Academic Retrieval Skill

## 技能说明

该技能负责把 Planner 产出的 `sub_queries` 转化为高质量候选论文集，并保证检索过程可追踪、可解释。  
技能核心是：**多源召回 + 后处理重排 + 日志审计**。

## 适用场景

- 需要同时接入 arXiv 与 Semantic Scholar 进行论文召回。  
- 需要将候选论文控制在 `<=30` 的可评估范围。  
- 需要展示“为什么这篇论文被召回”。  
- 需要应对 API 限流、0 命中、重复论文等常见问题。

## 输入与输出契约

### 输入

- `question`
- `sub_queries`
- `source`（`all/semantic_scholar/arxiv`）
- `per_query_limit`
- `max_papers`
- `locked_concepts`

### 输出

- `papers: PaperRecord[]`（已去重、已打 `query_match_score` 与 `rerank_score`）
- `search_log: list[dict]`（包含每个子查询 hits/status/source）

## 核心执行步骤

### 阶段 1：子查询召回（MCP）

1. 对每条子查询调用 `search_papers`。  
2. 计算动态 `effective_limit`，目标形成 recall pool（默认约 100）。  
3. 记录每次调用日志（requested/effective/hits/status）。

### 阶段 2：0 命中回退

1. 若子查询 0 命中，触发“简化查询重试”（保留前 10 个核心术语）。  
2. 若重试成功，合并重试结果。  
3. 将重试信息写入 `search_log`，避免黑盒检索。

### 阶段 3：标准化与去重

1. 统一字段类型（`None -> ""`, 数值转型）。  
2. `paper_id` 去重，缺失时使用 `title+year+source` 稳定哈希。  
3. 合并冲突字段（更长摘要、可用 URL 优先）。

### 阶段 4：匹配打分（召回解释分）

1. 计算 `concept_hit_count` 与 `matched_concepts`。  
2. 计算 `lock_match_count`（锁定概念命中数）。  
3. 计算 `query_match_score` 并排序。

### 阶段 5：两阶段重排

1. 对 recall pool 执行 rerank（Cross-Encoder 优先，失败回退 lexical）。  
2. 写入 `rerank_score`。  
3. 按 `max_papers` 截断（硬限制 `<=30`）。

### 阶段 6：后处理审计日志

增加 `query=__postprocess__` 日志项，至少包含：
- `recall_target`
- `recall_after_dedupe`
- `rerank_backend`
- `rerank_model`
- `rerank_fallback`

## 质量门禁（必须通过）

1. 返回论文条数 `<= max_papers` 且 `max_papers <= 30`。  
2. 每篇论文必须带 `query_match_score`。  
3. 重排阶段后每篇论文必须带 `rerank_score`。  
4. `search_log` 必须有后处理记录（`__postprocess__`）。

## 失败回退策略

1. 单源失败时保留另一数据源结果。  
2. Cross-Encoder 不可用时自动降级 lexical 重排。  
3. 概念过滤导致空集时，允许部分匹配回退，避免零召回。

## 最佳实践

1. `source=all` 作为默认，以提升召回鲁棒性。  
2. `max_papers` 建议 20-30，兼顾覆盖与评分开销。  
3. 对高精度问题配合 `locked_concepts` 与 `intent_slots` 使用。
