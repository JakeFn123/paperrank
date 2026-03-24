---
name: synthesis
description: 综合写作技能，基于证据与评分输出中文结论，并执行引用一致性审查。
tags: synthesis,citation,report,audit
---

# Synthesis Skill

## 技能说明

该技能把评分结果转为结构化结论，核心是“结论必须能追溯到证据”。

## 触发条件（When to use）

- 需要最终中文综合报告。
- 需要结论含引用并可审计。
- 需要输出不确定性与分歧立场。

## 不适用边界（When NOT to use）

- 用户仅要原始论文列表。
- 尚未完成评分或无 evidence。
- 用户要求纯英文报告（当前默认中文模板）。

## 必要输入

- `question`
- `plan`
- `scored_papers`

## 工作模式决策

1. `Evidence-First Synthesis`
- 正常路径：先生成结论，再执行引用审查。

2. `Empty-Set Fallback`
- 无论文时输出最小结论模板，明确不可下结论。

3. `Audit-Corrective`
- 审查不通过时，强制追加“证据对齐审查”。

## 核心执行步骤

### 阶段 1：参考编号构建

- 生成 `P1/P2/...` 并绑定 evidence。

### 阶段 2：结构化结论生成

- 固定章节：直接回答、证据结论、文献分歧、入选原因、不确定性。

### 阶段 3：澄清问题前置

- 若存在澄清问题，置于报告开头。

### 阶段 4：结论-证据对齐表生成

- 输出 `## 结论-证据对齐表`：结论句 -> 引用 -> 证据片段。

### 阶段 5：证据审查

- 检查关键结论引用覆盖、引用有效性、证据可用性。

## 质量门禁（量化）

- 关键结论引用覆盖率 = 100%。
- `invalid_refs` 必须为 0。
- 报告必须包含“不确定性”章节。
- 输出必须含 `evidence_audit`。

## 失败回退策略

- 无候选论文：输出最小可解释结论。
- 模型写作失败：回退保守模板。
- 审查失败：不隐藏错误，显式输出审查段。

## 标准输出模板（Output Contract）

```json
{
  "final_answer_markdown": "## 直接回答\n...",
  "evidence_audit": {
    "passed": true,
    "confidence": "high",
    "cited_refs": ["P1"],
    "invalid_refs": [],
    "missing_citation_lines": [],
    "refs_without_evidence": []
  }
}
```

Markdown 必含章节：
- `## 直接回答`
- `## 证据支撑的关键结论`
- `## 文献分歧与不同立场`
- `## 论文入选原因（多维评分）`
- `## 当前不确定性`
- `## 结论-证据对齐表`

## 最小样例

输入：
- `scored_papers=10`

输出特征：
- 每条关键结论至少一个 `[P#]`；
- 对齐表能定位到对应 evidence 文本。
