---
name: evidence-grading
description: 证据驱动评分技能，负责全文证据构建、五维评分计算与评分可解释性输出。
tags: rag,scoring,evidence,evaluation,explainability
---

# Evidence Grading Skill

## 技能说明

该技能将候选论文转为可解释排序结果，核心原则是“评分必须有证据”。

## 触发条件（When to use）

- 需要对候选论文进行排序。
- 需要输出评分依据和证据片段。
- 需要在 UI 展示逐篇评分过程。

## 不适用边界（When NOT to use）

- 仅做检索展示，不需要评分。
- 用户只需快速摘要，不关心证据链。
- 没有任何候选论文输入。

## 必要输入

- `question`
- `papers`
- `ingest_top_n`

## 工作模式决策

1. `Deep Evidence Mode`
- `ingest_top_n > 0`：优先 PDF 解析 + 向量证据检索。

2. `Abstract Fallback Mode`
- PDF 不可用时：摘要兜底证据，流程不中断。

3. `Fast Scoring Mode`
- 高并发或资源受限：降低 `ingest_top_n`，保留规则维度评分稳定性。

## 核心执行步骤

### 阶段 1：证据构建

- 对前 N 篇 ingest PDF 并检索 evidence。

### 阶段 2：证据兜底

- 无全文证据时，使用摘要生成 fallback evidence。

### 阶段 3：五维评分

- `content_relevance`
- `method_relevance`
- `timeliness`
- `quality_signal`
- `complementarity`

### 阶段 4：总分聚合与排序

- 按固定权重计算 `total`，按 `total` 降序输出。

### 阶段 5：可解释性输出

- 保留 `rationale` + evidence + 维度分。

## 质量门禁（量化）

- 每篇论文至少 1 条 evidence（全文或摘要兜底）。
- 每篇 `score.total` 可由五维分复算。
- `scored_papers` 按 `total` 单调降序。

## 失败回退策略

- PDF 下载/解析失败：摘要兜底。
- 单篇打分失败：保守默认分 + 标注原因。
- 向量检索异常：直接走摘要证据。

## 标准输出模板（Output Contract）

```json
{
  "scored_papers": [
    {
      "paper": {"paper_id": "...", "title": "..."},
      "score": {
        "content_relevance": 0,
        "method_relevance": 0,
        "timeliness": 0,
        "quality_signal": 0,
        "complementarity": 0,
        "total": 0,
        "rationale": "..."
      },
      "evidence": [{"page": 1, "text": "..."}]
    }
  ]
}
```

## 最小样例

输入：
- `papers=12`, `ingest_top_n=6`

输出特征：
- 前 6 篇多数含页码证据；
- 其余论文至少含摘要兜底 evidence。
