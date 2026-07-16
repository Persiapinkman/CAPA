# Qwen3.5-4B 多步工具路由 V2 评测与 GRPO 兼容性审计

日期：2026-07-14
范围：`planner_multistep_grpo_hard_v2` confirmation 600 cases，Qwen3.5-4B 三轮正式评测，以及现有 TRL GRPO 路径的无更新兼容性检查。未启动 SFT 或 GRPO。

## 结论

Qwen3.5-4B 的 strict 结果为三轮均 `380/600 = 63.3333%`，3/3 pass-all 也是
63.3333%，mean dense score 为 0.935241。它显著优于 raw Qwen2.5-7B，但离
Qwen3.5-35B-A3B 仍有明显距离：

| Arm | run1 | run2 | run3 | pass-all | mean score |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 593/600 | 594/600 | 591/600 | 583/600 = 97.1667% | 0.999496 |
| **Qwen3.5-4B** | **380/600** | **380/600** | **380/600** | **380/600 = 63.3333%** | **0.935241** |
| Raw Qwen2.5-7B-Instruct | 146/600 | 147/600 | 147/600 | 136/600 = 22.6667% | 0.724724 |

以 75 个 `entity_id` 做 20,000 次 paired cluster bootstrap：

- 4B pass-all 95% CI：52.8333%–73.5000%；
- `4B - raw 7B`：+40.6667pp，95% CI 为 +33.3333–+48.0000pp；
- `35B - 4B`：+33.8333pp，95% CI 为 +23.6667–+44.5000pp。

现有 GRPO 代码的核心思路可以迁移到 Qwen3.5-4B，但**不能只替换模型路径直接训练**。
当前至少有依赖环境、chat template、hybrid LoRA targets、fp16 scaler 和完整 GRPO
显存契约五项适配工作。单序列 4,177-token 的无 optimizer forward/backward 在正确
设置下已通过，但当前环境的 `GRPOTrainer` 导入仍失败，`num_generations=8` 的完整
GRPO 内存路径也尚未验证。

模型已从 `/mnt/zkq/models/Qwen3.5-4B` 复制到
`/raid/zkq/models/Qwen3.5-4B`。目标路径原先不存在，未覆盖任何模型；32 个普通文件、
9,343,257,586 字节经 `rsync --checksum --dry-run` 复核为零差异。

## 正式评测有效性

- 数据：600 cases / 75 entity clusters / 8 families，每轮 1,200 decisions；
- 数据 SHA256：`42201f057366411fdfab77621f6b54e3f4c16d7fcdc8109616065e6212cc3008`；
- `temperature=0`, `top_p=1`, `do_sample=false`, `seed=42`；
- `max_steps=3`, `max_tokens=2048`，两个 timeout 均为 300 秒；
- fp16，本地 Transformers 服务，Qwen3.5 non-thinking 模式；
- 三轮共 1,800 case-runs / 3,600 decisions；
- API/rollout error、空 decision、首次截断、retry 截断和 retry 均为 0；
- 600/600 cases 的 strict pass/fail verdict 三轮一致；591/600 的结构化 decision 字段
  三轮完全一致，另 9 题虽有结构变化但未发生 verdict flip；
- dataset、rollout config、strict reward、分片合并和 cluster bootstrap 共 16 项相关测试通过。

## Family 结果

下表为 3/3 strict pass-all：

| Family | raw 7B | Qwen3.5-4B | 35B |
|---|---:|---:|---:|
| `qwen_box_variance_to_migration` | 30.67% | 61.33% | 100.00% |
| `qwen_empty_result_to_migration` | 6.67% | 58.67% | 97.33% |
| `qwen_domain_shift_to_migration` | 56.00% | 62.67% | 98.67% |
| `rex_box_variance_to_migration` | 12.00% | 66.67% | 94.67% |
| `rex_empty_result_to_migration` | 2.67% | 66.67% | 94.67% |
| `rex_domain_shift_to_migration` | 4.00% | 66.67% | 93.33% |
| `qwen_confident_stop_guardrail` | 56.00% | 57.33% | 98.67% |
| `rex_confident_stop_guardrail` | 13.33% | 66.67% | 100.00% |

## Bad-case 结构与数据集判断

4B 有 220 个 strict failed cases，但失败并不均匀：

- 185/220（84.09%）只有一个 typed-argument 词面错误：gold 要求 `可见烟雾`，模型输出
  `烟雾`；action、步骤和其他参数均通过；
- 其余 35/220 才是真正的路由/停止错误，根因高度一致：跳过 detector/probe，直接调用
  `migration_advisor`；
- 仅作诊断、忽略上述一个 label alias 时为 565/600 = 94.1667%。该数值不是正式指标，
  frozen strict 结果仍是 63.3333%。

因此 V2 对 raw Qwen2.5-7B 是有效的能力差分集，但对 Qwen3.5-4B 不再是理想的 GRPO
训练场景来源：4B 超过原 preregistration 的 base 上限 60%，且大部分 reward 差异来自
精确词面复制，更像 SFT/格式服从问题，而非复杂软边界探索。不能读取这 600 题后直接用
其 gold 训练；应保持其 sealed evaluation 身份，用全新实体、模板和 fixture 重建 4B
专属 calibration/train 数据。

真正适合继续扩增的模式是那 35 个 bad case 所代表的状态转移：在证据不足、空结果、
domain shift 或 stop guardrail 下，先做指定 detector/probe，读取 observation 后再迁移或
停止。扩增必须复用模式而非复用 sealed case 文本、ID 或 gold 参数。

## 为什么不能直接切换 GRPO 基模

### 1. 依赖环境不闭合

现有 `.venv-trl-grpo-cu124` 为 Transformers 4.57.6 / TRL 1.8.0 / Torch 2.6 cu124，
无法识别 `model_type=qwen3_5`。用于本次模型加载的 `.venv-train` 能解析 Qwen3.5，
但其 TRL 1.7-dev 会加载已安装的 vLLM 0.10.2，而该版本缺少
`StructuredOutputsParams`，导致 `GRPOTrainer` 在构造前就导入失败。

应新建独立、冻结版本的 Qwen3.5 GRPO 环境，保留现有 Qwen2.5 环境不动。若仍采用
`use_vllm=false`，训练环境可不安装 vLLM；若要使用 vLLM，则必须同时满足 TRL 和
Qwen3.5 的版本要求。

### 2. Prompt contract 不同

Qwen3.5 默认 thinking；non-thinking 的 generation prompt 结尾是：

```text
<|im_start|>assistant
<think>

</think>

```

现有 128 条 frozen step prompt 全部停在旧的
`<|im_start|>assistant\n`，没有空 think block。必须用 Qwen3.5 tokenizer 的
`apply_chat_template(..., enable_thinking=false)` 重建 step data，使训练和正式推理一致。
重建只增加 4 tokens，最长 prompt 从 4,104 变为 4,108，不构成长度问题。

### 3. 原 LoRA targets 漏掉 24/32 层

原 `q_proj,k_proj,v_proj,o_proj` 只命中 8 个 full-attention 层：3,145,728 个可训练
参数，按真实加载模型计占约 0.0747%。建议的最小 hybrid targets 为：

```text
q_proj,k_proj,v_proj,o_proj,
in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj
```

它同时覆盖 8 个 full-attention 和 24 个 Gated DeltaNet 层，共 14,376,960 个可训练
参数，按真实加载模型计占约 0.3407%。最终训练前仍应对该 target set 做小规模消融，
而不是把它视为已
优化超参。

### 4. V100 精度与 kernel 需要专用配置

V100 无原生 bf16，本机只能以 fp16 为主。当前环境没有 Qwen3.5 Gated DeltaNet 的
可选 fast kernel，模型回退到纯 PyTorch 实现。Torch 2.8 cu128 / cuDNN 9.19 在 V100
上的 depthwise fp16 Conv1d 会报 `CUDNN_STATUS_NOT_INITIALIZED`；关闭 cuDNN 后正常。
现有公共训练模块恰好会关闭 cuDNN，但这一行为应在新入口中显式固定并测试。

fp16 还必须走 autocast：不使用 autocast 时梯度会出现非有限值。即使使用 autocast，
默认高 loss scale 也不安全：512 tokens 下 65,536 和 4,096 均溢出；4,177 tokens 下
scale=256 仍有 64/304 个梯度张量非有限，scale=1 时 304/304 全部有限。因此需要给
Accelerate/GradScaler 显式配置低初始 scale，并把 finite-gradient check 设为训练门。

### 5. 单序列通过不等于完整 GRPO 通过

最终无更新契约使用 4,108-token prompt + 69-token completion、gradient checkpointing、
fp16 autocast、scale=1、关闭 cuDNN 和 hybrid LoRA：

- loss：0.815358；
- 304/304 gradient tensors finite；
- 152 个梯度张量初始非零，符合 LoRA-B 零初始化时 LoRA-A 首步为零的预期；
- peak allocated：22.191 GiB；
- 未创建 optimizer、未执行 step、未保存权重。

但当前正式配置为 `num_generations=8` / `generation_batch_size=8`，还会增加 rollout、
completion logprob 和 GRPO loss 缓冲。22.191 GiB 的单序列结果不能证明完整 GRPO 能在
32 GiB V100 上运行；训练前还需一次不更新权重的 GRPO generation-and-loss memory
microbenchmark。

## 建议的后续门槛（本轮未执行）

1. 保持当前 600-case confirmation sealed，不把逐题 gold 加入 SFT/GRPO。
2. 用新实体/模板/fixture 构建 Qwen3.5-4B 专属 calibration/train：重点扩增“先 probe，
   observation 后迁移/停止”的真实路由差异；避免主要靠一个 label 字面量制造 gap。
3. 先用 disjoint SFT 数据建立 exact typed-argument 和 non-thinking 输出契约。
4. 新建 Qwen3.5 专用依赖环境，加入 GRPOTrainer import、4k finite-gradient、显存和多卡
   smoke gate；固定 hybrid LoRA、关闭 cuDNN、fp16 autocast 和低初始 GradScaler scale。
5. 对 SFT initializer 在新 calibration prompts 上每题采样 8 次；只有 full-decision
   support、组内 reward 方差和非饱和度达标后才启动 GRPO。

## 产物

- 评测 preregistration：
  `experiments/studies/planner_multistep_tool_routing_grpo_qwen25_7b_v1/qwen35_4b_evaluation_preregistration_20260714.json`
- 结构化结果：
  `experiments/studies/planner_multistep_tool_routing_grpo_qwen25_7b_v1/qwen35_4b_evaluation_result_20260714.json`
- GRPO 兼容性审计：
  `experiments/studies/planner_multistep_tool_routing_grpo_qwen25_7b_v1/qwen35_4b_grpo_compatibility_20260714.json`
- 正式 aggregate：
  `/raid/zkq/artifacts/CAPA/evals/20260714_planner_multistep_grpo_hard_v2/confirmation/qwen35_4b_combined/qwen35_4b_v2_confirmation_t0_3x_aggregate.json`
- case audit / failed cases：同目录下
  `qwen35_4b_v2_confirmation_t0_3x_case_audit.csv` 与
  `qwen35_4b_v2_confirmation_t0_3x_failed_cases.csv`
- paired bootstrap：同目录下
  `qwen35_4b_vs_qwen35_a3b_cluster_bootstrap.json` 与
  `qwen25_7b_vs_qwen35_4b_cluster_bootstrap.json`

本轮没有启动训练，没有 optimizer step，也没有生成或保存任何 adapter 权重。
