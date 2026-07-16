# Dataset Card: planner_multistep_grpo_hard_v2

## 目的

V2 是 V1 family-level calibration 失败后的独立改版。V1 不允许逐 case 筛选；其
完整 family gate 只确认了两个有效机制：

- Qwen uncertain observation 后转 `migration_advisor`：35B 95.8%，raw 7B 0%；
- 有图且要求迁移顾问内部视觉探针：35B 100%，raw 7B 0%。

V2 使用 V1 未暴露的新实体，把这两个机制展开为可解释、可验证的业务/observation
子族，并保留高置信 stop 反事实作为 calibration guardrail。

## Calibration

| 项目 | 数值 |
|---|---:|
| cases | 384 |
| Planner decisions | 640 |
| entity bundles | 32 |
| scenario families | 12 |
| primary challenge families | 10 |
| counterfactual guardrail families | 2 |

六个两步 primary family 为 Qwen/Rex × 候选框波动、零结果、场景域偏移。它们
共享同一初始 query；observation 不同但都明确禁止再次检测，并要求转迁移顾问、
使用图片、内部视觉复核、最终停止。Qwen/Rex 的高置信 observation 分别导向 end，
只作为防止模型学成“探针后必迁移”的 guardrail，不要求 base 表现差。

四个单步 primary family 分别要求迁移顾问输出能力边界、低成本验证、风险约束和
补数需求；它们都严格要求 `use_image=true`、`use_visual_probe=true`、
`finish_after_tool=true`。

## Family gate

Primary family 必须同时满足：35B >=95%、raw 7B <=70%、gap >=25pp。至少准入
6 个 family，且至少 3 个为多步 family。Guardrail 单独报告，不用于满足“base
表现不佳”的准入数量。

只有 V2 calibration 通过后，才按冻结 allowlist 使用第三组实体和不同模板生成
600-case confirmation。Confirmation 不得逐题删选。

## 完整性

- V2 calibration 与 V1 calibration 的 entity ID、template ID、case ID、规范化
  query 均不重叠；
- 与仓库其它 case 文件的 case ID/规范化 query 自动审计为零交叉；
- strict action、strict JSON types、wrong-action cap 和过程奖励沿用 V1；
- 文件 hash 与自动审计见 `build_report.json`。

## 当前状态

- 384-case calibration 已生成并通过自动检查；
- raw 7B calibration strict pass 为 148/384（38.54%），35B 为
  383/384（99.74%），均无 API error 或空 decision；
- family gate 以完整场景族为单位通过，8 个两步 family 被准入，4 个 base 已
  100% 的单步 direct-migration family 被剔除；
- 已按冻结 allowlist 生成 600-case confirmation：75 个独立 entity group、
  8 个场景族、1,200 个 Planner decision；未做逐题筛选；
- confirmation 与 calibration 的 case ID、entity、template、规范化 query 和
  fixture family 均零交叉；
- raw 7B 的 128-step / 1,024-completion GRPO sampling-support audit 已完成：
  全部 step 与 guardrail 混合的严格全局门失败；六个 primary family 的待学习
  step2 角色门通过（reward variance 91.67%、usable support 83.33%、gold action
  support 85.42%、平均 distinct valid action 1.8125、饱和率 0%）；
- primary step2 的完整 exact-argument support 仅 2/48（4.17%），因此这些场景
  适合 dense-reward GRPO，但不批准 raw-base 直接 GRPO；先做短 SFT initializer，
  再按同一门复验 strict full-decision support；
- confirmation 的首个 384-token 尝试因 35B 出现 4 次首 completion 截断和
  3 个 timeout 被整批作废；在未查看 correctness 分数前已冻结协议修订为
  768 tokens / 300 秒，并要求两模型从 case 1 完整重跑 600 x 3；
- 768-token 重跑仍出现 1 次首 completion 截断，亦整批作废且未查看 correctness；
  最终运行协议改为 2,048 tokens / 300 秒，并再次要求两模型完整重跑；
- 最终修订后的 confirmation 3x 已完成并通过整版 gate：35B pass-all
  583/600（97.17%），raw 7B 136/600（22.67%），差 74.5pp；
- 运行有效性按实际 Planner 合同判定：首调用达到上限但 retry 成功时计入
  `recovered_first_length_truncations` 稳定性指标；retry 仍截断、最终 error 或空
  decision 才使整轮无效；
- 三轮均无最终 error、空 decision 或 retry 截断；35B 共 5 次 recovered
  first-call truncation，7B 为 0；
- 未生成训练集，未启动训练。

## GRPO 适用性边界

全局 128 个 step 中，step1 是已经饱和的点名检测前缀，stop family 是用于防止
“探测后固定迁移”的 evaluation guardrail；把它们和主要学习 step 混合会得到
54.69% reward-variance、44.53% usable-support，并触发全局 fail。该结果保留在
`grpo_support_gate.json`，不能删除或改写。

角色分层后，六个 detection-observation -> migration family 的 48 个 step2 在
8-sample 下通过原设计的可探索性门，结果记录在
`grpo_support_gate_primary_step2.json`。不过 exact full-task support 仍低，后续
训练方案必须把 step1/stop guardrail 放入 SFT/评测配额，把 primary step2 作为
GRPO 主采样单元；SFT 后若 strict support 不提升则停止，不启动 GRPO。

Qwen2.5 tokenizer 审计显示：calibration/confirmation 的最大 prompt 分别为
3,976/3,938 tokens，均低于 6,144；canonical completion 最大 73/72 tokens。
1,024 个随机 support completion 最大 211 tokens，174 个超过 128、无一个超过
384。因此未来训练方案把 completion 上限由旧计划的 128 调整为 256；这只影响
训练资源规划，不改变本次 2,048-token 外部评测修订。
