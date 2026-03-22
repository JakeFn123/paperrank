---
name: evidence-grading
description: 证据驱动评分技能，负责全文证据构建、五维评分计算与评分可解释性输出。
tags: rag,scoring,evidence,evaluation,explainability
---

# Evidence Grading Skill

## 技能说明

该技能负责将候选论文从“检索结果”转为“可解释评分结果”。  
核心要求是：评分必须由证据支撑，而不是只看标题摘要。

## 适用场景

- 需要对候选论文进行定量排序。  
- 需要展示评分由哪些维度构成。  
- 需要在 UI 中给出逐篇评分过程。  
- 需要明确“无全文时如何降级但不中断流程”。

## 输入与输出契约

### 输入

- `question`
- `papers: PaperRecord[]`
- `ingest_top_n`（尝试解析全文的论文数）

### 输出

- `ScoredPaper[]`
  - `paper`
  - `score`
  - `evidence[]`

## 核心执行步骤

### 阶段 1：证据构建

1. 对前 `ingest_top_n` 篇执行 PDF ingest。  
2. 解析文本并切块写入本地向量库。  
3. 依据 `question` 检索每篇论文的 top-k 证据片段（含页码）。

### 阶段 2：证据兜底

1. 若 PDF 不可获取或解析失败，使用摘要构造兜底证据。  
2. 兜底证据必须标记页码与来源，保证下游一致性。

### 阶段 3：维度评分

计算五个维度：
- `content_relevance`
- `method_relevance`
- `timeliness`
- `quality_signal`
- `complementarity`

其中：
- 前三维可由 LLM 结合证据打分。  
- 后两维由规则函数计算（引文、venue、作者、互补性）。

### 阶段 4：总分聚合

按固定权重计算 `total`，并按 `total` 降序排序输出。

### 阶段 5：可解释性输出

每篇论文必须保留：
1. 评分理由（`rationale`）。  
2. 可追溯证据片段。  
3. 各维度分数，便于 UI 展示公式分解。

## 质量门禁（必须通过）

1. 每篇论文必须有至少 1 条证据（全文或摘要兜底）。  
2. `score.total` 必须是五维分数可复算结果。  
3. 输出结果必须可排序且稳定。

## 失败回退策略

1. PDF 失败 -> 摘要兜底证据。  
2. 单篇打分失败 -> 使用保守默认分并标记理由。  
3. 向量检索异常 -> 降级为摘要证据，不中断整体流程。

## 最佳实践

1. `ingest_top_n` 推荐 4-8，平衡效果与速度。  
2. 当问题高度专业时，提高 `ingest_top_n` 获得更强证据。  
3. 检查高分论文是否也具备高质量证据，防止“高分低证据”。
