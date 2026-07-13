# 表格语料说明

该目录存放结构化表格检索与问答所使用的标准化数据文件。

## 当前文件

- `model_release_records.jsonl`：由 `data_source/模型发版记录汇总.xlsx` 导出的规范化 JSONL 数据。

## 生成方式

请在 `rag-api` 环境中执行：

```bash
python scripts/export_model_release_table_to_jsonl.py
```

如需覆盖默认输入/输出路径，可显式传参：

```bash
python scripts/export_model_release_table_to_jsonl.py \
  --input data_source/模型发版记录汇总.xlsx \
  --output data_source/tables/model_release_records.jsonl
```
