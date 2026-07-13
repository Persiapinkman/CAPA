# GBrain RAG 图谱可视化

这个目录用于把 SQLite 索引里的实体和共现关系导出成可交互 HTML。生成的 HTML 不依赖外网资源，直接用浏览器打开即可查看。

## 数据来源

默认读取项目索引：

```text
data/index/gbrain.sqlite3
```

使用的表：

- `entities`：实体节点，包含 `kind`、`name`、`normalized_name`
- `entity_links`：实体关系边，当前主要是 `co_mentions`
- `chunk_entities` / `chunks`：用于统计实体出现次数，并给节点/边详情内嵌少量来源片段

## 生成默认图

在项目根目录执行：

```bash
python visualization/export_graph_html.py
```

默认输出：

```text
visualization/output/entity_graph.html
```

然后在浏览器中打开该文件。

## 常用参数

只看模型、平台、场景之间的强关系：

```bash
python visualization/export_graph_html.py \
  --include-kinds model,platform,scene \
  --min-evidence 3 \
  --limit-nodes 260 \
  --limit-edges 520 \
  --output visualization/output/model_platform_scene.html
```

排除版本号和 OID，降低图噪声：

```bash
python visualization/export_graph_html.py \
  --exclude-kinds version,oid \
  --min-evidence 2 \
  --output visualization/output/no_version_oid.html
```

围绕某个实体名或关键词导出子图：

```bash
python visualization/export_graph_html.py \
  --query safety_rope \
  --min-evidence 1 \
  --limit-nodes 160 \
  --limit-edges 320 \
  --output visualization/output/safety_rope.html
```

## 参数说明

- `--db`：SQLite 索引路径，默认 `data/index/gbrain.sqlite3`
- `--output`：生成的 HTML 路径
- `--query`：按实体名、归一化实体名、实体类型匹配子图
- `--include-kinds`：只保留这些实体类型，逗号分隔，如 `model,platform,scene`
- `--exclude-kinds`：排除这些实体类型，逗号分隔，如 `version,oid`
- `--relations`：只保留指定关系类型，当前常用 `co_mentions`
- `--min-evidence`：边的最小 `evidence_count`
- `--limit-nodes` / `--limit-edges`：限制导出的节点和边数量
- `--node-samples`：每个节点内嵌的来源片段数量
- `--edge-samples`：每条强边内嵌的共现片段数量
- `--edge-sample-edges`：只有权重最高的前 N 条边会内嵌共现片段

## HTML 交互

- 鼠标滚轮缩放，拖拽空白区域平移
- 拖拽节点可以临时调整布局
- 左侧可按实体类型、实体文本、边权过滤
- 点击节点查看实体统计和来源片段
- 点击边查看关系类型、共现次数和共现片段
- `适配` 会把当前筛选后的图居中到画布
