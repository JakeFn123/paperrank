---
name: evidence-grading
description: 基于全文证据和多维指标对论文进行可解释评分。
tags: rag,scoring,evidence,evaluation
---

# Evidence Grading Skill

## Objective
对每篇论文计算可解释总分，并保留支撑证据。

## Checklist
1. 优先解析 Top-N 论文全文（PDF -> chunks -> embedding）。
2. 从向量库检索证据片段（含页码）。
3. 计算维度分：content/method/timeliness/quality/complementarity。
4. 记录评分理由与总分贡献，支持 UI 展示。

## Output Contract
- 返回 `ScoredPaper[]`（paper + score + evidence）。
- score.total 用于最终排序。
