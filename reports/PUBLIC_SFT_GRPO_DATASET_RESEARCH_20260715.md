# CAPA 公开 SFT / GRPO 数据集调研与快速跑通方案

日期：2026-07-15
目标模型：`/raid/zkq/models/Qwen3.5-4B`
目标：先用成熟公开数据把 SFT、GRPO、评测和反 reward-hacking 闭环跑通，再替换为 CAPA
多步工具路由数据。

## 1. 结论

最推荐的路径不是选一个“大而全”的 Agent 数据集，而是按验证目标分三层：

1. **GSM8K 做最小 plumbing smoke**：验证数据转换、assistant loss mask、EOS、LoRA、保存和
   评测脚本。它很小、MIT、最终答案可精确验证，但对 Qwen3.5-4B 可能过于简单，只能证明
   工程链路能跑，不能证明 GRPO 有价值。
2. **MATH-lighteval 做第一套正式 SFT -> GRPO 因果实验**：7,500 train / 5,000 test，MIT，
   有完整解题过程、难度 1-5 和学科标签，可以通过难度分桶控制 base policy 的正样本支持率，
   并用 Math-Verify 做无需 LLM judge 的奖励。这是本机最快获得可信 GRPO 曲线的首选。
3. **API-Bank 做第一套工具路由迁移实验**：MIT、规模适中、带本地可执行 API 和独立测试
   对话。先用 Level 1/2 做工具调用 SFT，再在 Level 2/3 做逐步或环境结果 GRPO；它比
   ToolBench/tau3-bench 更容易接入现有 CAPA verifier。

工具调用 SFT 可增加 **Hermes Function-Calling V1** 作为格式和多轮暖身；BFCL 和
tau3-bench 应保留为后续评测，不应混入首轮训练。

## 2. 候选数据集比较

| 数据集 | 公开性与规模 | 最适合的用途 | 奖励可验证性 | 本项目结论 |
|---|---|---|---|---|
| GSM8K | MIT；7,473 train / 1,319 test | SFT/GRPO 最小 smoke | 最终数字精确匹配 | **先跑，但不用于证明 GRPO 价值** |
| MATH-lighteval | MIT；7,500 train / 5,000 test；Level 1-5 | 正式 SFT + GRPO | Math-Verify 符号等价 | **首选** |
| OpenR1-Math-220k | Apache-2.0；default 93,733；长推理轨迹 | 大规模 reasoning SFT/GRPO | Math-Verify | 当前 V100 首轮过重，不选 |
| Countdown 3-to-4 | 490,364；仅 `nums/target` | verifier 单元测试、玩具 GRPO | 表达式执行 | 无 license/test/解题轨迹，不作正式基准 |
| Hermes FC V1 | Apache-2.0；五配置约 11.6k | 单轮/多轮工具 SFT | 无可靠执行环境 | **工具 SFT 快速暖身** |
| ToolACE | Apache-2.0；11,300 对话；API pool 26,507 | 复杂工具 SFT | 发布数据无统一可执行环境 | 第二工具 SFT 备选 |
| xLAM FC 60k | CC-BY-4.0；60k；auto-gated | 大规模 function-call SFT | 结构化 gold | 有访问权限后可替代 Hermes |
| API-Bank | MIT；源论文 1,888 train dialogues；314 eval dialogues | 工具 SFT + 小规模 tool GRPO | 73 个本地可执行测试 API | **工具迁移首选** |
| APIGen-MT-5k | CC-BY-NC-4.0；5,000 多轮轨迹 | 高质量多轮 SFT | 三层验证生成数据 | 研究可用，部署许可不合适 |
| ToolBench | Apache-2.0；126,486 instances / 16,464 APIs | 大规模开放域 tool learning | 依赖 live/simulated API | 变量太多，不用于首轮 |
| BFCL | Apache-2.0 代码；V1-V4 function-call benchmark | 函数调用回归与多步评测 | AST/可执行检查 | **只评测，不训练** |
| tau3-bench | MIT；Gym；多领域有状态环境 | Agent RL、最终状态和 policy 评测 | DB/communication/action 等 | 后续阶段；单独 Python 3.12 环境 |

数据卡与官方仓库链接见文末及
`sources/research_public_sft_grpo_datasets_20260715.md`。

## 3. 为什么首选 MATH，而不是直接上公开 Agent 数据

MATH 同时满足四个关键条件：

- 有 SFT gold trace，也有 GRPO 可用的最终答案；
- 有正式 train/test split，且可按 Level 和 type 分层；
- prompt 短于当前 CAPA 4.3k-token schema，能先隔离显存、模板和 reward 问题；
- verifier 是确定性代码，不需要昂贵且漂移的 LLM judge。

这使每种失败的归因比较清楚：SFT 不收敛主要查 label/mask/学习率；GRPO 不学习主要查
support、group variance 和 reward；准确率上涨但真实能力不涨则查 parser/reward hacking。

OpenR1-Math-220k 虽然是 Open R1 官方配方使用的数据，但其 default 版本物化后约 5 GB，
官方数据卡说明生成 trace 使用最高 16k token。它适合规模化复现，不适合当前 V100 上的首轮
闭环。TRL 当前文档使用的 DeepMath-103K 在数据卡中没有声明 license，也不作为首选。

## 4. 建议的实验阶梯

### P0：CPU verifier 与数据契约

不加载模型，先冻结统一数据接口：

```text
sample_id
split
messages
tools                 # 数学任务为空
verifier_type         # gsm8k_numeric / math_verify / api_bank
ground_truth
reward_metadata
source_family
```

要求：

- 按 source problem、API family、domain/entity 去重后再切分；
- test 永不进入 SFT、GRPO、support filtering 或 checkpoint selection；
- gold 可解析率必须为 100%，不能把 gold parse failure 当成 reward 0、None 或 1；
- verifier 先跑至少 30 个 adversarial unit cases。

### P1：GSM8K 32-case SFT overfit

目的只是暴露 SFT 工程错误：

- 32 train + 32 held-out；
- Qwen3.5 native non-thinking contract；
- LoRA，`assistant_only_loss=true`；
- `packing=false`，最大长度由 token audit 决定，首轮建议不超过 1,024；
- 检查 train loss、mean token accuracy、label-token count、EOS label、生成准确率；
- 成功条件不是“loss 在下降”，而是能过拟合训练小集且 held-out 生成仍正常终止。

### P2：MATH 小型 SFT

建议先从正式 train split 中构建：

- 1,024 条 SFT train；
- 128 条 disjoint development；
- 按 `level/type` 分层；
- 仅保留 prompt + completion 在首轮长度门内的样本，同时报告被过滤比例；
- 官方 test 只抽固定分层 500 条作 sealed quick eval，完整 5,000 条留到 promotion。

先跑 32-case overfit，再跑 100-300 optimizer-step screen。对照必须保留同一 initializer 的
未训练 adapter/control。

### P3：MATH GRPO

在 **SFT 未使用** 的 MATH train rows 上做 base/SFT-initializer support audit：每题 8 次采样，
按 Level 逐桶统计。首轮只使用能同时采到正确与错误答案的桶，不根据 sealed test 选题。

建议初始 reward：

```text
math_accuracy      0.95
strict_format      0.05
tag_count          0.00  # 只记录
reasoning_steps    0.00  # 只记录
length             0.00  # 只记录
```

GRPO 仍使用当前已验证的 `G=4` 跨 rank 分发、local generation batch 1 和 Dr. GRPO。长度上限
必须以完整 support pool 的 p99/max 决定，不能再用单题探针推断。

顺序：G1/G2 -> support gate -> G3/G4 -> 5-step canary -> 50/100-step seed42 screen。

### P4：工具调用 SFT

两种选择：

1. 最快：Hermes `func_calling_singleturn` 1,893 + `func_calling` 1,893；
2. 更复杂：ToolACE 11,300 对话。

都要先转换成 Transformers/TRL 原生 `messages + tools`，不能直接继承数据集自带的
`<tool_call>` 或 `[Func(args)]` 字符串模板，否则学到的是旧格式而不是 Qwen3.5 tool-call contract。
SFT 评测至少拆为 route、required args、type、JSON、stop 五项。

### P5：API-Bank 工具 GRPO

API-Bank 的 Hugging Face 训练文件已拆成 6,184 Level-1、9,279 Level-2、1,245 Level-3
prompt-completion entries；源论文口径是 1,888 个训练对话。两种数量不能混用。

建议：

- Level 1 做 known-tool route/args SFT；
- Level 2 做 tool retrieval + call SFT/GRPO；
- Level 3 做多步执行评测和小规模环境 GRPO；
- 不安装旧 demo 的完整 requirements，只移植 API schema、实现和 test fixtures；
- 把旧 `[ApiName(k='v')]` 格式统一转换为严格 JSON tool calls；
- split 按 API/domain family，而不是随机拆 prompt，防止同一 API 模板泄漏。

## 5. SFT loss 最容易踩的坑

### 5.1 assistant mask 可以全错，但 loss 仍可能“正常”

本机真实测试：Qwen3.5 原生 tokenizer template 不包含 Jinja `{% generation %}` marker。
直接请求 assistant mask 时，五轮工具对话得到 `0/77` 个 assistant label token。

冻结的 TRL 1.8 `SFTTrainer` 会识别 Qwen3.5，并在
`assistant_only_loss=true` 时切换到 Qwen3.5 training template；同一输入得到 `34/73` 个
assistant token，两个 assistant turn 均被覆盖，且 `<|im_end|>` 位于监督区。

因此数据预处理门必须逐样本断言：

- supervised token count > 0；
- user/system/tool observation token 的 label 全为 `-100`；
- 每个 assistant turn 都有 label；
- `<|im_end|>` 被监督；
- 截断后仍至少保留一个完整 assistant turn。

如果把对话提前 render 成普通 `text`，`assistant_only_loss` 的结构信息可能丢失。首轮应把
结构化 messages 交给 SFTTrainer，或自己生成并审计 labels。

### 5.2 loss 的绝对值不是能力指标

SFT loss 是非 masked token 上的 token cross entropy。不同的 assistant mask、序列长度、
packing 或 tokenizer 会改变其口径，因此不能把两个 run 的绝对 loss 直接比较。必须同时记录：

- supervised token fraction；
- mean token accuracy；
- train/dev loss；
- route/answer exact generation accuracy；
- EOS/截断率；
- 每个长度桶的 loss。

### 5.3 EOS、packing 和 truncation

- Qwen3.5 的 assistant stop 应保持 `<|im_end|>`，不能用 PAD 替代监督 EOS；
- 当前 V100 使用 SDPA，首轮 `packing=false`，避免 packed sample 的跨样本 attention 污染；
- token audit 必须在 chat template 之后做；
- 工具 observation 很长时，不能只保留序列开头而把目标 assistant call 截掉。

## 6. GRPO reward 与 reward hacking 实验

### 6.1 先测 support，不把 reward std 当成 task signal

每个 prompt 的 G 个 completion 必须真的存在任务差异。建议 support gate 至少报告：

- nonzero task-reward std group rate；
- at-least-one-correct support rate；
- fully-correct 和 fully-wrong group rate；
- distinct parsed answers/actions；
- parser failure、EOS 和 clipped rate；
- **task reward std 与 format reward std 分开**。

当前 CAPA 数据已经证明：总 reward 有方差可能只来自格式尾部或截断，并不代表路由可学习。

### 6.2 第一版不要奖励“看起来像推理”

Open R1 示例同时使用 accuracy、format、tag-count；这适合展示功能，不应原样当作安全配方。
空 `<think></think>`、重复标签或堆砌 `Step 1/2/3` 都能获得代理奖励。第一版将 tag-count、
reasoning-step count 和 length 设为零权重诊断项，只让可验证的最终任务结果主导更新。

Open R1 当前 reward 源码还展示了一个值得专门测试的边界：部分 length/cosine reward 在 gold
无法解析时会把样本当作 correct 或直接给 1。我们的数据必须在训练前剔除不可解析 gold，
而不是在 reward 内静默补偿。

### 6.3 Math-Verify 也需要对抗测试

Math-Verify 官方说明 verifier 是非对称的，正是为了避免模型原样返回题目中的不等式而得分。
至少测试：

- 多个 `\boxed{}` 时取哪个；
- 题目回显、gold 泄漏和 solution chain；
- 等式/不等式方向；
- 单位、百分号、浮点精度；
- NaN/Inf/超长 SymPy 表达式；
- malformed LaTeX、空答案和截断答案。

verifier 应设超时，且 gold/prediction 的参数顺序固定。

### 6.4 工具 reward 的典型漏洞

API-Bank 原 evaluator 用正则在任意文本中搜索第一个 `[Api(...)]`，因此“额外解释 + 正确
substring”也可能通过。新 verifier 必须 full-string parse、严格 schema、禁止 `eval`，并对
重复调用、未知字段、缺参和错误 side effect 单独计数。

不要把唯一参考 trajectory 当成唯一正确路径。tau3-bench 的官方 reward 文档明确区分：
`actions` 通常只是一个参考路径，airline/retail/telecom 的正式奖励主要由最终 DB state 和
communication 决定。反过来，纯 end-state reward 也可能被“什么都不做”利用：官方文档给出
了 DB 初态即目标且 communication 为空时，不调用任何查询工具也能拿满分的例子。

工具 GRPO 应同时记录但不一定同时优化：

- final state / execution success；
- route、required args、types；
- policy constraints 和 forbidden side effects；
- call count、重复调用、stop；
- reference-trajectory similarity（仅诊断）。

### 6.5 建议做一个受控 hacking lab

在不用于 promotion 的 20-step throwaway run 中比较：

| Arm | Reward | 预期用途 |
|---|---|---|
| R0 | accuracy only | verifier/优化基线 |
| R1 | 0.95 accuracy + 0.05 format | 推荐安全起点 |
| R2 | 0.10 accuracy + 0.90 format | 故意脆弱；验证 composite reward 上涨但 held-out accuracy 不涨 |

工具数据再比较 route-only 与 route+args+execution。若监控系统不能识别 R2 或 route-only 的
虚假进步，则不应进入正式训练。

## 7. 首轮硬门建议

### SFT

- label contract 100% 通过；
- gold/EOS 不被截断；
- 32-case overfit 的 train generation 明显提升；
- dev exact 没有在 loss 下降时反向崩溃；
- 无 NaN/Inf，GradScaler 和显存门沿用当前 Qwen3.5 方案。

### GRPO

- support audit 先通过；
- clipped <= 1%；
- format/tag/length 不能贡献主要 reward variance；
- composite reward 与 shadow task accuracy 同向；
- `frac_reward_zero_std` 不持续接近 1；
- 5-step canary 后权重确实更新、所有梯度 finite；
- 50/100-step screen 后 sealed dev/test 才用于一次性比较。

## 8. 推荐执行顺序

1. 下载并冻结 GSM8K、MATH-lighteval 的 commit/hash；
2. 实现统一 dataset adapter 和 verifier adversarial tests；
3. 跑 GSM8K 32-case SFT overfit；
4. 跑 MATH 1,024-case SFT screen；
5. 对 MATH 各 Level 做 80-case x 8 support audit；
6. 跑 R0/R1/R2 受控 hacking lab；
7. 固定安全 reward，跑 5-step + 50/100-step GRPO；
8. 再接 Hermes SFT 和 API-Bank tool GRPO；
9. 用 BFCL 做 function-call sealed regression；
10. 最后另建 Python 3.12 环境接 tau3-bench Gym，不污染当前训练环境。

## 9. 最终选择

若只选一个数据集：**MATH-lighteval**。
若选一套完整迁移路径：**GSM8K -> MATH-lighteval -> Hermes FC -> API-Bank -> BFCL/tau3**。

这套顺序能依次暴露 assistant mask、EOS、loss 口径、verifier、稀疏奖励、group zero-std、
format hacking、route-only hacking、执行状态和多步环境问题，同时不会一开始就把所有变量混在
一个大型 Agent 系统里。

## 10. 主要官方来源

- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [MATH-lighteval dataset card](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
- [Math-Verify](https://github.com/huggingface/Math-Verify)
- [Open R1](https://github.com/huggingface/open-r1)
- [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
- [Hermes Function-Calling V1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [tau3-bench](https://github.com/sierra-research/tau2-bench)
