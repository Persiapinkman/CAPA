# Skill Agent Demo

`demo/demo_server.py` 是一个本地可视化 Demo：

- 用户在浏览器输入文本，并可上传参考图片（最多 10 张）；可多次点击「＋」逐批添加，或在一次文件选择框里多选
- **仅自动路由**：文本含「检测 / 标注 / 评测 / detect / …」且已上传图片 → 跑完整 **pipeline**（与 `run_pipeline.py` 逐步一致，**固定生成 3 张** Flux 图）；否则只跑 **意图理解**
- `POST /run` 返回 **NDJSON 流**（逐行 JSON），每跑完一个子进程即推送事件；页面按步展示意图摘要、每张生成图、标注图、评估报告
- 生成图 / 标注图通过 `GET /demo-run/<run_stamp>/...` 拉取

## 启动

在仓库根目录运行：

```bash
python3 demo/demo_server.py --port 18080
```

默认绑定 **`0.0.0.0`**（监听所有网卡），本机与其他电脑都能访问。

- 在本机浏览器打开：`http://127.0.0.1:18080`
- 在局域网其他电脑打开：`http://<服务器内网IP>:18080`（例如 `http://10.111.32.254:18080`）

若你希望 **只允许本机** 访问（更安全），显式指定：

```bash
  
```

## 路由逻辑（自动）

- 文本含检测类关键词 **且** 有参考图 → 按步调用：意图理解 → 提示词扩展 → Flux×3 → Qwen/Rex 检测与合并 → 画框 → 评估报告
- 否则 → 仅调用 `user-intent-understanding`（可选带图）

## 注意

- **API Key**：从仓库根目录 `api_key.txt` 读取。`DEMO_API_BASE` 覆盖 LLM/Flux 的 OpenAI 兼容地址（默认 `https://api.apiyi.com/v1`）。检测服务可用 `DEMO_QWEN_BASE_URL`、`DEMO_REX_BASE_URL`（默认与 `run_pipeline.py` 一致）。
- **知识库 RAG**：默认 `RAG_BASE_URL=http://127.0.0.1:6062/api/v1/playbook/query`（ace-rag Playbook 问答）。需先在本机启动 gbrain-rag（6061）与 ace-rag（6062），可用 `curl http://127.0.0.1:6062/api/v1/playbook/health` 探活。若报错 `RAG HTTP 500: {"detail":"Connection error."}`，多半是地址仍指向 `6061/api/v1/rag/query`（旧默认），请改为上述 playbook 地址后重启 demo。
- **RAG 回答后置判定开关**：默认关闭。设置 `DEMO_RAG_RESOLUTION_JUDGE_ENABLED=1` 时，RAG 命中后的答案和 3 轮 miss 后的 answerer 兜底答案都会再经过一轮“是否真正解决问题”的 judger；不设置或设为 `0`/`false`/`off`/`no` 时跳过这轮判定，不再因此进入 clarify。
- **完整流水线**：必须上传参考图片。
- **程序调用**：`POST /run` 为 **multipart** 或 **json**；响应为 **application/x-ndjson** 流，非单条 JSON。
- 每次运行产物会保存在 `demo/runs/<timestamp>/` 下（含本次上传的 `uploaded_image_00.*` … `uploaded_image_09.*`）。子进程命令与输出追加写入 `demo/runs/run_log.txt`，页面不展示。

## 后端批量测试迁移顾问

新增 `POST /test/migration-advisor`，仅供后端自动化测试使用，直接触发迁移顾问链路，不需要前端点击确认。

请求体为 `application/json`，支持两种方式：

1. 单条 query

```json
{
  "query": "香港项目要求识别头盔颜色，现有模型是否可迁移？",
  "session_id_prefix": "ma_test",
  "force_migration_advisor": true
}
```

2. CSV 批量

```json
{
  "csv_path": "/abs/path/to/queries.csv",
  "index": 3,
  "session_id_prefix": "ma_batch",
  "force_migration_advisor": true
}
```

或：

```json
{
  "csv_path": "/abs/path/to/queries.csv",
  "indices": [0, 2, 5],
  "session_id_prefix": "ma_batch",
  "force_migration_advisor": true
}
```

CSV 默认读取列名 `query / text / question / prompt` 之一；若都不存在，则读取第一列。

每条测试会：

- 直接运行迁移顾问，不走手动确认分支
- 在 `demo/runs/<run_stamp>/` 下落盘 `migration_advisor_report.json`
- 同时落盘 `migration_advisor_report.md`
