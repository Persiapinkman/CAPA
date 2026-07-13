# data_source 语料说明

最后更新：2026-04-15

## 当前包含的语料

- `PDFs/`：主要语料目录，当前共 `75` 份 PDF 文档。
- `tables/`：结构化表格语料目录，当前包含模型发版记录的标准化 `jsonl` 文件。
- `adela/`：adela 部署记录语料目录，`adela/data/` 下为原始 JSON，`adela/adela_release_records.jsonl` 为标准化导出结果。
- `pdf_reference_links.csv`：PDF 文件与原始来源链接（ONES wiki）的对应关系表。
- `ones语料汇总.xlsx`：语料汇总表（Excel）。
- `模型发版记录汇总.xlsx`：模型发版记录原始 Excel，可转换为 `tables/model_release_records.jsonl` 供表格问答接口使用。

## 语料主题概览（按文件名归纳）

当前语料以模型发布记录和算法能力说明为主，覆盖方向包括：

- 人脸与人体相关：人脸质量、模糊、姿态、人体二分类、行人属性等。
- 车辆与车牌相关：车牌检测/识别、车辆属性、车辆细分类、车辆跟踪、非机动车属性等。
- 安防与场景任务：城市结构化、异常行为（如打架）、实名举报场景等。
- 通用能力与特征：ReID、Embedding、视频指纹、特征模型更新等。
- 行业项目与专项任务：如中车项目、横幅标语、旗帜/特殊文字/二维码检测等。

## 备注

- 部分语料文件名包含版本号，可用于追踪模型迭代历史。
- `pdf_reference_links.csv` 中个别条目的链接可能为空（例如仅本地 PDF）。
- 表格语料可通过 `python scripts/export_model_release_table_to_jsonl.py` 重新从 Excel 导出。
