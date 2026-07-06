# adela-cli 使用说明

本文档说明 `scripts/run_pipeline.py` 的用途、回流输出格式和示例命令。

## 相关文件

- 流程入口：`scripts/run_pipeline.py`
- adela 基础调用：`scripts/run_cli.py`
- 输出解析：`scripts/result_parser.py`
- 评测配置模板：`references/accuracy_eval.json`、`references/speed_eval.json`

## 命令行参数

```bash
python scripts/run_pipeline.py --rawmodel_id <ID> --platform <PLATFORM> --eval_type <0|1>
```

- `rawmodel_id`：原模型 ID
- `platform`：目标部署平台（例如 `cuda11.0-trt7.1-int8-T4`）
- `eval_type`：
  - `0` 精度评测（`normal precision`）
  - `1` 性能评测（`normal performance`）

## 回流输出格式

脚本按 SSE 风格输出，单条格式为：

```text
data: {"event":"<event_name>", ...}
```

结束时固定输出：

```text
data: [DONE]
```

### 细粒度命令回流

每执行一次底层 `adela` 子命令（与 `run_cli.py` 中一次调用对应），会立即输出一条：

- `adela_api_result`
  - `command`：`deployment_list`（由 `deployment_list_result` 覆盖，不重复发 `adela_api_result`）、`benchmark_list`、`deployment_info`、`benchmark_info`、`deployment_add`、`benchmark_add` 等
  - 另含与命令相关的字段，例如 `deployment_id`、`benchmark_id`、`record_count`、`poll_iteration`、`status`、`phase` 等

### Step1 数据查询（语义事件）

- `deployment_list_result`
  - 在第一次 `deployment_list` 完成后立刻发出
  - 常用字段：`rawmodel_id`、`platform`、`is_quant_platform`、`deployment_count`、`target_deployment_id`
- `benchmark_probe_result`
  - 在完成「目标部署上的评测列表探测」后发出（无目标部署时也会发一条表示未探测到目标）
  - 字段：`deployment_id`（可为 `null`）、`record_count`、`matched`、`benchmark_id`（命中时）
- `quant_dataset_missing` / `quant_dataset_result`
  - 量化平台且需要复用量化数据集时：失败或成功；`quant_dataset_result` 含 `dataset_id`
  - 兼容旧名：`quant_dataset_ready` 已不再由脚本发出，前端仍可兼容解析
- `eval_dataset_missing` / `eval_dataset_result`
  - 评测数据集解析失败或成功；`eval_dataset_result` 含 `dataset_id`
  - 兼容旧名：`eval_dataset_ready` 已不再由脚本发出

### 历史命中与部署 / 评测

- `adela_existing_result`：第一步即命中历史成功评测，不再提交新部署与评测
- `submit_model_deployment`：发起新部署后立即回流，`result` 为部署接口返回
- `model_deployment_result`：部署达到终态 `SUCCESS` / `FAILURE` 时回流；`FAILURE` 时流程结束
- `submit_model_evaluation`：发起评测后立即回流
- `model_evluation_result`：评测达到终态 `SUCCESS` / `FAILURE` 时回流

### 结束与错误

- `adela_final_result`：脚本主流程正常结束后由 `__main__` 汇总发出（含整段 `result`）
- `adela_pipeline_error`：未捕获异常时的用户可读错误

## 示例命令

### 1) 成功查询

```bash
python scripts/run_pipeline.py --rawmodel_id 51476 --platform cuda11.0-trt7.1-int8-T4 --eval_type 0
```

### 2) 缺少量化数据集

```bash
python scripts/run_pipeline.py --rawmodel_id 49258 --platform cuda11.0-trt7.1-int8-T4 --eval_type 0
```

### 3) 缺少评测数据集

```bash
python scripts/run_pipeline.py --rawmodel_id 49258 --platform cuda11.0-trt7.1-fp16-T4 --eval_type 0
```
