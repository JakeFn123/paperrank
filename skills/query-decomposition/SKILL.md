---
name: query-decomposition
description: 面向科研问题的结构化意图拆解技能，输出高相关子查询并支持人工槽位覆盖。
tags: planner,query,intent,decomposition,slots
---

# Query Decomposition Skill

## 技能说明

该技能将用户研究问题从自然语言转为可检索、可验证、可约束的 `ResearchPlan`，是检索质量的第一责任层。

## 触发条件（When to use）

- 用户输入开放式研究问题，需要自动拆解检索路径。
- 用户提供锁定概念，希望系统围绕指定概念执行检索。
- 需要在 UI 里编辑槽位后重跑，保持问题-检索一致性。

## 不适用边界（When NOT to use）

- 用户只要求对已有论文列表做评分或总结（应跳过拆解）。
- 用户问题是单一精确查询（例如“找某篇已知论文”）。
- 用户明确禁止改写原始查询。

## 必要输入

- `question: str`
- `locked_concepts: list[str]`
- `forced_intent_slots: dict[str, list[str]]`

## 工作模式决策

1. `Hypothesis Validation`（是否/能否提升）
- 优先输出“对象 + 方法 + 指标”结构，强调验证口径。

2. `Method Design`（如何设计/优化）
- 优先输出“机制 + 约束 + 代价”结构，强调可执行策略。

3. `Comparative Evaluation`（A vs B）
- 优先输出“对比对象 + 统一评测标准 + 场景边界”。

4. `Mechanism Explanation`（为什么有效）
- 优先输出“机制假设 + 失败模式 + 边界条件”。

## 核心执行步骤

### 阶段 1：问题归一化

- 识别语言、缩写、核心技术词；
- 归一化锁定概念（中英文别名映射到检索友好形式）。

### 阶段 2：构建意图槽位

- 生成 `subject/intervention/outcome/context/evaluation`；
- 若存在 `forced_intent_slots`，同名槽位强制覆盖。

### 阶段 3：子查询生成

- 根据槽位组合英文技术子查询；
- 强制注入锁定概念；
- 清理空泛词和重复词。

### 阶段 4：相关性自检

- 计算槽位覆盖度；
- 覆盖不足则回退规则模板重建子查询。

### 阶段 5：补全解释字段

- 输出 `research_intent`、`hidden_assumptions`、`clarification_questions`。

## 质量门禁（量化）

- 每条子查询必须是英文技术短语。
- `sub_queries` 至少 1 条，默认目标 3 条。
- 若 `locked_concepts` 非空：子查询锁定概念覆盖率 = 100%。
- `intent_slots` 必须包含 5 个标准槽位键。

## 失败回退策略

- LLM 拆解失败：回退规则模板。
- 槽位覆盖不足：以 `forced_intent_slots` + 规则模板重建。
- 锁定概念冲突：保留锁定概念，降级非核心术语。

## 标准输出模板（Output Contract）

```json
{
  "research_intent": "...",
  "intent_slots": {
    "subject": ["..."],
    "intervention": ["..."],
    "outcome": ["..."],
    "context": ["..."],
    "evaluation": ["..."]
  },
  "sub_queries": ["...", "...", "..."],
  "hidden_assumptions": ["..."],
  "clarification_questions": ["..."]
}
```

## 最小样例

输入：
- `question`: 在企业知识库问答中，cross-encoder reranker 是否提升 RAG 稳定性？
- `locked_concepts`: ["retrieval augmented generation", "reranker", "robustness"]

输出特征：
- `sub_queries` 每条都包含 `retrieval augmented generation / reranker / robustness`。
- `intent_slots.outcome` 至少包含 `robustness`。
