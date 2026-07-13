# ONES 差异化 Case 清单

用于从现有语料中挑选适合与开源算法、公开 benchmark 或开源榜单做对比的 case。

说明：
- 下面的“可对照开源方向”是基于任务类型做的映射建议，不是 ONES 原文直接声明。
- 优先保留同时满足这三点的 case：差异化明显、指标提升清晰、容易映射到公开任务。

## 主推 Case

| 优先级 | Case | 任务类型 | 差异化亮点 | 可对照开源方向 | ONES 链接 |
| --- | --- | --- | --- | --- | --- |
| P0 | `Shikra-Embedding-V3.7.0` | 多模态检索 | 行人品牌 `mAP +15.02%`、车辆品牌 `+9.68%`、否定测试集 `+9.25%`、重点检索词 `+22.77%`；150 万底库整体 `mAP` 相对提升约 `10.8%` | 中文多模态检索、中文 CLIP、文搜图/图搜图 benchmark；可重点对 `COCO_CN`、`Flickr30K-CNA` 语境 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/7PXBL1jP |
| P0 | `SmallFace_544x960_3.0.0` | 小目标人脸检测 | 相对 `smallface_4.15.3`，`struct_test_new@FPPI0.01` 提升 `11.2%`；`GZ` 提升 `19.8%`；`WC` 提升 `17.8%`；明确写出监控强、网图弱，trade-off 清晰 | Tiny face / small object face detection；适合映射 WIDER Face hard、监控小脸检测 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/4Y9LtFw5 |
| P0 | `KM_textrecognition_CarPlate_large 5.0.1` | 车牌 OCR | 模糊 `66.84% -> 72.70% (+5.86)`、新能源 `95.04% -> 95.46% (+0.42)`、夜间 `94.82% -> 95.46% (+0.64)`；端到端 `wuchang`、结构化难例、`nanjing` 都有提升 | 车牌识别、OCR、低质车牌识别；可映射 CCPD 一类公开任务 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/Ug9emfzJ |
| P0 | `车牌特征 1.5.0` | 车牌检索 / ReID | `Blur5w` 上 `CMC1 59.2% -> 79.3% (+20.1)`；`256 -> 64` 维后百万底库 `CMC1` 只下降 `0.1%`；文档明确提到车牌特征与车辆特征联合可超过车辆 ReID | Vehicle ReID、plate retrieval、大底库检索 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/6gRcGJCv |
| P0 | `Hermes_3.1.1 & Horae_1.6.1` | 车辆跟踪 / MOT | 单目标跟踪 `precision` 平均提升 `3.2%`；多目标跟踪 `MOTA` 平均提升 `3.9%`；高速车辆 1/2 上 `MOTA +1.6% / +6.3%`；业务端到端“违法压线”召回 `82.09% -> 100%`，24 小时误报 `15 -> 14` | 多目标跟踪、交通场景 MOT、业务端到端事件跟踪 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/9ur3CirU |
| P1 | `KM_senu_Vehicle_ReID 1.12.0` | 车辆 ReID | 有牌场景 `mAP/cmc1 +1.5/+1.0`；无牌场景 `+4.9/+3.1`；新增 `FJ_FR-10w` 也有提升；对 `badly_blur`、`no_plate`、`NIGHT/CUT/POSE` 等困难场景更强 | Vehicle ReID、跨摄像头车辆检索、弱牌/无牌鲁棒性对比 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/TpFjFnvv |

## 备选 Case

| 优先级 | Case | 任务类型 | 差异化亮点 | 可对照开源方向 | ONES 链接 |
| --- | --- | --- | --- | --- | --- |
| P1 | `KM_obb_carplate_detect_1.0.0` | 旋转框车牌检测 | 香港通用集识别准确率略降 `0.008`，但三车牌评测集 `0.7846 -> 0.9413`，提升约 `+15.7pt`；适合讲“极难场景专用优化” | 旋转目标检测、香港/多车牌检测、端到端车牌识别链路 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/EpMK9JeR |
| P1 | `KM_classifier_Face_None_Security 3.0.2 / 3.0.1` | 人脸误检过滤 | `CLIP` 蒸馏 hardcase 平均 `acc 38.6% -> 72.5%`；`common` 测试集人脸召回 `+26%`、误报召回 `+43%`；大角度、遮挡、后脑、全遮挡场景提升明显 | 人脸误检过滤、CLIP 蒸馏、小样本 hardcase 分类 | https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/GebSt74Y/page/RxxiLWb8 |

## 使用建议

- 如果目标是做“公开 benchmark / 开源榜单”风格的对外对比，优先从 `Shikra-Embedding-V3.7.0`、`SmallFace_544x960_3.0.0`、`车牌特征 1.5.0` 这 3 个 case 开始。
- 如果目标是做“业务难场景优势”表达，优先从 `车牌 OCR`、`车辆跟踪`、`车辆 ReID` 开始。
- 如果需要强调“极端或定制场景能力”，可以补充 `香港车牌-旋转框检测` 和 `人脸误检过滤` 两个备选 case。
