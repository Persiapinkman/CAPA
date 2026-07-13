# Unified Benchmark 与评测方法说明

最后更新：2026-04-24

本文档说明 `single_step_rag_100_v1` 题集的设计目标、字段语义、评测流程、指标定义和结果解读方法，适用于统一检索问答接口：

- `POST /rag/chat_engine/unified_query`

---

## 1. Benchmark 目标与范围

### 1.1 评测目标

该 benchmark 用于评估 unified 网关在以下能力上的表现：

1. 路由能力：能否选择正确的数据源（document / table / adela）。
2. 检索能力：融合后证据是否命中目标来源和目标实体。
3. 回答能力：最终答案是否覆盖标准答案关键字段。
4. 性能能力：端到端耗时和分阶段耗时是否稳定。

### 1.2 题集范围

当前题集目录：`benchmark/single_step_rag_100_v1`

- 主题集：`benchmark_100.jsonl`
- 兼容 CSV：`benchmark_100_for_eval.csv`
- 问题列表：`questions.txt`

题目覆盖三类来源：

- `document`：模型发版文档正文检索问答
- `table`：结构化表格字段查询
- `adela`：部署记录查询

并包含部分跨源核对题（例如同时需要 `table + adela`）。

---

## 2. 样本字段定义

每条样本（JSONL 一行）常见字段如下：

- `id`：题目标识
- `question`：问题文本
- `reference_answer`：标准答案
- `source_doc` / `source_page`：来源文件与页码/行号
- `evidence`：证据片段（人工核对用）
- `question_type` / `difficulty`：题型与难度
- `expected_keywords`：期望关键词（回答评测）
- `source_type`：来源类型（document / table / adela）
- `source_id`：来源主键（文档名或 row_id）
- `retrieval_source_types`：建议检索源（可多源）

---

## 3. 本次评测结果展示（100 题）

本节直接展示当前仓库中的一轮完整评测结果（100 题）：

- 结果目录：`results/unified_benchmark/unified_benchmark_20260423_221740`
- 结果时间：2026-04-23
- 评测脚本：`scripts/unified_benchmark_eval.py`
- 可视化脚本：`scripts/visualize_unified_benchmark_results.py`

### 3.1 本次评测配置

| 配置项 | 取值 |
| --- | --- |
| benchmark | `benchmark/single_step_rag_100_v1/benchmark_100.jsonl` |
| 样本数 | `100` |
| repeat | `1` |
| route_with_llm | `true` |
| fused_top_k | `12` |
| rrf_k | `60` |
| 数据源开关 | `document/table/adela = true/true/true` |
| timeout | `180s` |

### 3.2 总体指标

| 指标 | 数值 |
| --- | --- |
| 总请求数 | `100` |
| 成功请求数 | `99` |
| 失败请求数 | `1` |
| 成功率 | `99.00%` |
| 平均客户端耗时 | `5318.175 ms` |
| 客户端耗时 P50 / P95 | `4306.438 / 10125.580 ms` |
| 平均检索耗时 | `520.095 ms` |
| 平均回答耗时 | `3897.213 ms` |

### 3.3 检索与回答指标

| 指标组 | 指标 | 数值 |
| --- | --- | --- |
| 路由 | `avg_route_precision` | `0.885` |
| 路由 | `avg_route_recall` | `0.920` |
| 路由 | `route_hit_rate` | `0.920` |
| 融合证据 | `avg_evidence_recall_at_k` | `0.880` |
| 融合证据 | `avg_evidence_mrr` | `0.697088` |
| 回答 | `exact_match_rate` | `0.000` |
| 回答 | `reference_containment_rate` | `0.300` |
| 回答 | `avg_keyword_recall` | `0.750833` |
| 回答 | `avg_answer_score` | `0.777910` |
| 回答 | `answer_correct_rate` | `0.700` |

### 3.4 分来源表现

| 期望来源 | 样本数 | evidence_recall@k | answer_correct_rate |
| --- | ---: | ---: | ---: |
| `document` | `8` | `0.8750` | `0.6250` |
| `table` | `51` | `0.8039` | `0.7647` |
| `adela` | `59` | `0.9492` | `0.5593` |

### 3.5 路由分布（selected_sources）

| 路由结果 | 次数 |
| --- | ---: |
| `adela` | `40` |
| `adela+table` | `25` |
| `table` | `19` |
| `document` | `15` |
| `none` | `1` |

---

## 4. 指标定义（评测方法）

### 4.1 可用性与耗时指标

- `success_rate`：接口成功率
- 分阶段耗时（毫秒）：
  - `client_total_ms`
  - `server_total_ms`
  - `route_ms`
  - `retrieve_ms`
  - `fuse_ms`
  - `answer_ms`
- 统计项：`avg/min/p50/p90/p95/max`

### 4.2 检索指标

### A. Source-level 指标

1. 期望来源集合：
   - 优先取 `retrieval_source_types`
   - 若缺失，则回退 `source_type`
2. 预测来源集合：
   - `route_selected_sources`（路由）
   - `fused_source_types`（融合证据）
3. 计算：
   - `precision / recall / f1 / hit`

分别统计两组指标：

- `route_*`：路由选择质量
- `fused_source_*`：融合结果来源质量

### B. Evidence-level 指标

1. 期望证据键：
   - 来自样本 `source_id` 与 `source_doc`
2. 实际证据键：
   - 来自接口返回 `fused_evidences` 的 `evidence_id/title/payload` 关键字段
3. 匹配规则：
   - 规范化后完全相等，或长度 >= 8 的子串匹配
4. 计算：
   - `evidence_precision_at_k`
   - `evidence_recall_at_k`（当前为命中即 1，否则 0）
   - `evidence_mrr`
   - `evidence_first_hit_rank`

### 4.3 回答指标

### A. `exact_match`

将参考答案和模型答案都做规范化（去空白、去标点、统一大小写）后，完全一致记为 `True`。

### B. `reference_containment`

规范化后，若参考答案是模型答案连续子串，或反过来（长度阈值 >= 6），记为 `True`。

### C. `keyword_recall`

`expected_keywords` 中命中的比例；若无 `expected_keywords`，脚本会从 `reference_answer` 自动回退生成关键词。

### D. `char_f1`

基于字符级重叠的 F1，用于衡量“非严格一致但内容相近”的情况。

### E. `answer_score` 与 `answer_correct`

```text
answer_score = max(
  exact_match ? 1.0 : 0.0,
  reference_containment ? 0.95 : 0.0,
  keyword_recall,
  char_f1
)
answer_correct = answer_score >= 0.8
```

---

## 5. 本次结果文件说明

本次展示结果位于：`results/unified_benchmark/unified_benchmark_20260423_221740`

- `config.json`：运行配置
- `requests.jsonl`：逐请求明细（含检索与回答指标）
- `requests.csv`：明细表格
- `summary.json`：聚合指标
- `failures.jsonl`：bad case 子集（失败或未命中或回答错误）
- `viz/report.html`：可视化报告
- `viz/charts/*.svg`：图像文件

---

## 6. Bad Case 统计与示例

### 6.1 Bad Case 统计

基于本次 100 条请求：

- API 失败：`1`
- 检索未完全命中（`evidence_recall_at_k < 1`）：`12`
- 回答不达标（`answer_correct = false`）：`30`
- 同时“检索未完全命中 + 回答不达标”：`3`

### 6.2 典型 Bad Case

1. 接口失败（上下文超长）
   - 样本：`ssr100-043`
   - 现象：HTTP 500（内部透传 400），提示输入 token 超过模型上下文上限。
   - 含义：该类问题不是检索质量问题，而是 prompt 长度控制问题。

2. 期望来源为 table，但实际融合来源偏到 document
   - 样本：`ssr100-042`、`ssr100-063`、`ssr100-067`、`ssr100-074`
   - 现象：`expected_source_types = ['table']`，但 `fused_source_types` 出现 `document`，`evidence_recall_at_k = 0`。
   - 含义：路由或检索源配置导致结构化题未稳定命中表格来源。

3. 检索命中但回答字段偏差
   - 样本：`ssr100-076`、`ssr100-077`、`ssr100-052`
   - 现象：`evidence_recall_at_k = 1.0`，但 `keyword_hit = 0/1`，`answer_score` 很低（约 `0.03~0.04`）。
   - 含义：证据已找到，但回答阶段在关键字段抽取/归纳上有偏差。

---

## 7. 评测注意事项

1. `exact_match` 是严格指标，开放式回答场景中通常偏低，建议结合 `keyword_recall + char_f1` 看。
2. 跨源题建议开启 `--route-with-llm`，并关注 `route_fallback_used`。
3. 对比不同策略时，建议固定：
   - 相同 `benchmark-jsonl`
   - 相同 `fused_top_k / rrf_k`
   - 相同 `repeat`
   - 相同服务负载时段
4. 产线前请至少同时看：
   - `p95 client_total_ms`
   - `avg_evidence_recall_at_k`
   - `answer_correct_rate`
