---
name: query-decomposition
description: 面向科研问题的结构化意图拆解技能，输出高相关子查询并支持人工槽位覆盖。
tags: planner,query,intent,decomposition,slots
---

# Query Decomposition Skill

## 技能说明

该技能用于把用户研究问题从“自然语言描述”转成“可检索、可验证、可约束”的结构化计划。  
它是检索质量的第一责任层，重点解决：

1. 研究意图识别偏差。  
2. 关键概念抽取不准。  
3. 子查询与目标论文不相关。  
4. 中英文输入导致的术语漂移。

## 适用场景

- 用户输入一个复杂研究问题，希望自动拆解成检索子查询。  
- 用户给出“人工锁定关键概念”，要求检索必须围绕这些概念。  
- 需要在 UI 中手工编辑 `intent_slots` 后重跑。  
- 需要输出“可审计”的问题理解结果。

## 输入与输出契约

### 输入

- `question`：用户研究问题（中英文均可）
- `locked_concepts`：人工锁定概念列表（可空）
- `forced_intent_slots`：人工覆盖槽位（可空）

### 输出（ResearchPlan）

- `research_intent`：中文研究目标定义
- `intent_slots`：
  - `subject`
  - `intervention`
  - `outcome`
  - `context`
  - `evaluation`
- `sub_queries`：英文检索短语列表
- `hidden_assumptions`：隐含前提
- `clarification_questions`：需用户澄清问题

## 核心执行步骤

### 阶段 1：问题归一化

1. 清洗问题文本，去除无效噪声词。  
2. 识别语言和关键术语（含大写缩写如 MCP、RAG、MDP）。  
3. 归一化锁定概念（中英文别名映射到检索友好形式）。

### 阶段 2：构建意图槽位（Intent Slots）

1. 从词典与规则提取 `subject/intervention/outcome/context/evaluation`。  
2. 如果用户传入 `forced_intent_slots`，覆盖对应槽位。  
3. 若槽位为空，使用保守默认值避免空计划。

### 阶段 3：生成子查询

1. 基于槽位组合生成候选子查询。  
2. 过滤空泛模板词，优先保留技术词与场景词。  
3. 对每条子查询强制注入 `locked_concepts`（如有）。

### 阶段 4：相关性自检

1. 检查子查询是否覆盖核心槽位。  
2. 若覆盖不足，回退规则计划并重建。  
3. 子查询数量按系统参数截断（默认 3）。

### 阶段 5：补全解释字段

1. 生成 `research_intent`，明确目标、场景、指标、边界。  
2. 生成 `hidden_assumptions` 与 `clarification_questions`。  
3. 保证输出结构完整。

## 质量门禁（必须通过）

1. 每条子查询必须是英文技术短语。  
2. 若存在 `locked_concepts`，每条子查询都必须包含。  
3. `intent_slots` 不能为空字典。  
4. 子查询不能仅由泛词构成（如 review/method/systematic）。

## 失败回退策略

1. LLM 拆解异常时，使用规则拆解方案。  
2. 子查询相关性不足时，按槽位模板重建。  
3. 锁定概念冲突时，优先保留用户锁定概念并降级其他词。

## 最佳实践

1. 优先用“对象 + 方法 + 指标 + 场景”提问。  
2. 锁定概念建议 2-5 个，避免过窄导致零召回。  
3. 对高约束场景，配合 `intent_slots` 手工编辑再重跑。
