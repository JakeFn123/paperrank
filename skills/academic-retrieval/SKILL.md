---
name: academic-retrieval
description: 使用多源学术检索工具进行高质量论文召回与去重。
tags: retrieval,mcp,arxiv,semantic-scholar
---

# Academic Retrieval Skill

## Objective
通过 MCP 工具层统一访问 Semantic Scholar 与 arXiv，召回候选论文。

## Checklist
1. 对每个子查询执行检索。
2. 记录检索日志（query/hits/status/source）。
3. 执行 paper_id/title/year 级别去重。
4. 对锁定概念执行后排序加权，而不是过早硬过滤导致零召回。

## Output Contract
- 返回 `papers`（去重后的候选集）。
- 返回 `search_log`（可追踪检索过程）。
