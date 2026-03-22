---
name: synthesis
description: 基于评分结果生成带引用的中文综合评估结论。
tags: synthesis,citation,report
---

# Synthesis Skill

## Objective
将多篇论文证据整合为可执行、可追溯的最终结论。

## Checklist
1. 先给出直接回答，再给证据链。
2. 显示文献分歧，不做单边结论。
3. 引用格式统一：`[P1]` 或 `[P1 p.3]`。
4. 对不确定性进行显式说明。

## Output Contract
- 输出中文 Markdown。
- 论文题目、作者、摘要保持原文（英文）。
