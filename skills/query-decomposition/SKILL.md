---
name: query-decomposition
description: 将研究问题拆解为高召回且高精度的英文子查询。
tags: planner,query,intent,decomposition
---

# Query Decomposition Skill

## Objective
将用户输入的问题转换为可检索的子查询集合，避免无关泛化词。

## Checklist
1. 明确研究对象（subject）。
2. 明确方法或机制（intervention）。
3. 明确指标（outcome）。
4. 明确场景（context）。
5. 若有用户锁定概念，必须在每条子查询中覆盖。

## Output Contract
- 子查询使用英文技术短语。
- 每条子查询尽量 6-18 个词，避免 stopword 堆叠。
- 避免空泛词：review、benchmark、method、systematic（除非用户显式要求）。
