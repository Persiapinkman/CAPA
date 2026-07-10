# CAPA Planner GRPO 技术记录

更新时间：2026-07-10

## 当前结论

当前基座模型切换为 `/raid/zkq/models/Qwen2.5-7B-Instruct`。本机是 V100 32GB，训练默认路线采用 `verl==0.4.1 + HF rollout + fp16 + FSDP`，不再使用旧模型类补丁。vLLM 在这台机器上保留为可选项，但不作为默认训练 rollout 路径。后续正式训练建议使用 LoRA，而不是 7B 全参训练。

Qwen2.5-7B-Instruct 的未训练基座在 CAPA Planner 多步工具路由评测上表现为：基础单步视觉探测和 RAG 路由很稳，多步 `probe -> migration` 和带图迁移顾问路由能走到正确大方向，但严格字段对齐不足，尤其是 `finish_after_tool`、`use_visual_probe`、`use_image`、`user_query` 等 action_input 字段。

## 多步路由评测

评测命令：

```bash
PYTHON_BIN=.venv-train-cu124/bin/python \
MODEL=Qwen2.5-7B-Instruct \
API_BASE=http://127.0.0.1:8004/v1 \
REPORT_PREFIX=qwen25_7b_instruct_compound245_t0_run1 \
CASES=training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl \
RUNS=1 \
TIMEOUT_SECONDS=180 \
OPENAI_TIMEOUT_SECONDS=180 \
MAX_STEPS=3 \
TEMPERATURE=0 \
TOP_P=1 \
DO_SAMPLE=false \
bash scripts/run_grpo_repro_eval_3x.sh
```

临时推理服务使用 `demo/deploy_qwen_server.py` 在 GPU4 上启动，评测结束后已停止。

评测集：

- 文件：`training/planner_grpo_seed_v1/cases/planner_grpo_compound245_eval_cases.jsonl`
- 总 case：245
- 多步 case：61
- 最大规划步数：2
- 生成 decision 总数：306
- 空决策：0
- rollout error/timeout：0
- 耗时：约 18.85 分钟

核心结果：

| 指标 | 数值 |
| --- | ---: |
| mean_score | 0.903151 |
| passed | 112 / 245 |
| pass_rate | 45.7143% |
| empty_decisions | 0 |
| output_tokens_mean | 63.72 |
| input_tokens_mean | 4072.49 |

分类表现：

| category | count | mean_score | pass_rate |
| --- | ---: | ---: | ---: |
| single_image_probe | 74 | 1.000000 | 100.00% |
| historical_asset_qa | 7 | 1.000000 | 100.00% |
| intent_ambiguity | 1 | 1.000000 | 100.00% |
| migration_feasibility | 38 | 0.973684 | 73.68% |
| adela_eval | 2 | 0.950000 | 50.00% |
| full_detection_eval | 5 | 0.890000 | 20.00% |
| probe_then_migration | 61 | 0.851452 | 0.00% |
| general_answer | 6 | 0.800000 | 0.00% |
| migration_feasibility_with_image | 51 | 0.768301 | 0.00% |

主要失败原因：

| failure | count |
| --- | ---: |
| step1 `finish_after_tool` expected True, got False | 65 |
| step2 `finish_after_tool` expected True, got False | 44 |
| step2 `use_visual_probe` expected True, got False | 35 |
| step1 `finish_after_tool` expected False, got string `"false"` | 25 |
| step2 `use_image` / `use_visual_probe` missing | 10 / 10 |
| step1 expected `migration_advisor`, got `qwen_detection` | 9 |
| step2 expected `migration_advisor`, got `qwen_detection` | 9 |

产物：

- aggregate：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_aggregate.json`
- predictions：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_run1_predictions.jsonl`
- reward：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_run1_reward.json`
- failed cases：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_failed_cases.csv`
- case audit：`training/planner_grpo_seed_v1/reports/repro_eval/qwen25_7b_instruct_compound245_t0_run1_case_audit.csv`

解释：当前 7B 基座不是完全不会路由，平均分已经高，但严格通过率低。失败集中在工具参数 schema 和多步状态字段，而不是超时、空输出或 JSON 完全不可解析。因此 GRPO 的第一阶段应优先训练“字段精确性”和“二步转迁移”的结构化决策，而不是继续扩大普通单步检测样本。

## GRPO 训练阶段安排

当前阶段：

1. 环境和模型准备：已完成。Qwen2.5-7B-Instruct 已下载，`verl==0.4.1` 环境可用，V100 兼容脚本已固化。
2. 基座路由评测：已完成本次 `compound245` 评测，作为训练前 baseline。
3. verl 数据准备：已完成 focused step-level parquet，当前为 245 step samples，train/val = 221/24。
4. 训练 smoke：已完成 1 step GRPO smoke，完整跑通 generation、reward、old/ref logprob、advantage、actor update。
5. LoRA smoke：下一步先用 `scripts/run_qwen25_7b_verl_grpo_lora.sh trainer.total_training_steps=1` 验证 LoRA 路径。
6. 正式 GRPO LoRA：smoke 通过后执行完整 epoch。
7. 训练后评测：用同一 `compound245` 路由集复测，并对比 baseline 的 failed_cases。

推荐执行顺序：

```bash
# 1. 环境检查/安装
PYTHON_BIN=.venv-train-cu124/bin/python bash scripts/setup_verl_env.sh

# 2. 如需重建 verl 数据
.venv-train-cu124/bin/python training/planner_grpo_seed_v1/scripts/prepare_verl_grpo_data.py

# 3. LoRA smoke
PYTHON_BIN=.venv-train-cu124/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=4 \
bash scripts/run_qwen25_7b_verl_grpo_lora.sh trainer.total_training_steps=1

# 4. 正式 GRPO LoRA
PYTHON_BIN=.venv-train-cu124/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=4 \
bash scripts/run_qwen25_7b_verl_grpo_lora.sh
```

## 当前训练框架参数

全参入口脚本：`scripts/run_qwen25_7b_verl_grpo.sh`

LoRA 推荐入口脚本：`scripts/run_qwen25_7b_verl_grpo_lora.sh`

模型：

- `MODEL_PATH=/raid/zkq/models/Qwen2.5-7B-Instruct`
- `model_type=qwen2`
- `architectures=Qwen2ForCausalLM`
- 约 7.62B parameters
- 28 layers, hidden size 3584, 28 attention heads, 4 KV heads
- tokenizer/model context max position 32768

数据：

- 默认 cases：`training/planner_grpo_seed_v1/cases/planner_grpo_focused_4b_cases.jsonl`
- verl train：`training/planner_grpo_seed_v1/verl_data/train.parquet`
- verl val：`training/planner_grpo_seed_v1/verl_data/val.parquet`
- step samples：245
- train/val split：221 / 24
- val ratio：0.1
- seed：42

硬件和并行：

- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- `NNODES=1`
- `NGPUS_PER_NODE=4`
- actor/ref 使用 FSDP
- V100 默认 `float16`，不用 bf16

核心 verl 覆盖参数：

| 参数 | 当前值 |
| --- | --- |
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.use_kl_in_reward` | `False` |
| `data.train_batch_size` | `4` |
| `data.max_prompt_length` | `4608` |
| `data.max_response_length` | `512` |
| `actor_rollout_ref.model.override_config._attn_implementation` | `eager` |
| `actor_rollout_ref.model.use_remove_padding` | `True` |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` |
| `actor_rollout_ref.actor.optim.lr` | full-param: `1e-6`; LoRA: `5e-6` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `4` |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `True` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `8192` |
| `actor_rollout_ref.actor.use_kl_loss` | `True` |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` |
| `actor_rollout_ref.actor.entropy_coeff` | `0` |
| actor mixed precision | param fp16, reduce fp32, buffer fp32 |
| actor param/optimizer offload | `False` / `False` |
| `actor_rollout_ref.rollout.name` | `hf` |
| `actor_rollout_ref.rollout.dtype` | `float16` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` |
| `actor_rollout_ref.rollout.n` | `2` |
| `actor_rollout_ref.rollout.micro_batch_size` | `1` |
| `actor_rollout_ref.rollout.temperature` | `1.0` |
| `actor_rollout_ref.rollout.top_p` | `1.0` |
| `actor_rollout_ref.rollout.top_k` | `0` for HF |
| rollout logprob dynamic batch | `True` |
| ref logprob dynamic batch | `True` |
| ref param offload | `True` |
| `trainer.total_epochs` | `1` |
| `trainer.save_freq` | `20` |
| `trainer.test_freq` | `-1` |
| `trainer.val_before_train` | `False` |

LoRA 推荐参数：

| 参数 | 当前推荐 |
| --- | --- |
| `actor_rollout_ref.model.lora_rank` | `16` |
| `actor_rollout_ref.model.lora_alpha` | `32` |
| `actor_rollout_ref.model.target_modules` | `all-linear` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` 起步 |
| `TRAIN_BATCH_SIZE` | `8` |
| `PPO_MINI_BATCH_SIZE` | `8` |
| `ROLLOUT_N` | `2` |
| `MAX_RESPONSE_LENGTH` | `512` |

自定义 reward：

- 路径：`training/planner_grpo_seed_v1/scripts/verl_reward_planner_grpo.py`
- 函数：`compute_score`
- reward 目标：结构化 JSON 决策、action 命中、关键 action_input 字段、禁止工具、二步 probe/migration 转换、`finish_after_tool`。

V100/verl 兼容：

- `training/verl_flash_attn_shim` 提供 verl 0.4.1 需要的 `flash_attn.bert_padding` API。
- `scripts/patch_verl041_v100_compat.py` 在环境安装后重放通用兼容补丁：
  - HF rollout autocast 使用 fp16，避免 V100 bf16。
  - HF generation 打开 `remove_invalid_values` 和 `renormalize_logits`。
  - HF rollout 保留 FSDP auto wrap，避免 7B 单卡完整展开 OOM。
  - 给 HF rollout sharding manager 补 timing 字段。

## 已验证训练 smoke

命令：

```bash
PYTHON_BIN=.venv-train-cu124/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NGPUS_PER_NODE=4 \
OUTPUT_DIR=outputs/planner-grpo-qwen25-7b-verl-smoke \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
VAL_BEFORE_TRAIN=False \
bash scripts/run_qwen25_7b_verl_grpo.sh trainer.total_training_steps=1
```

结果：

- 1/1 step 成功完成。
- 单 step 约 203 秒。
- generation 约 83 秒。
- old_log_prob 约 20 秒。
- ref logprob 约 30 秒。
- actor update 约 69 秒。
- peak reserved 显存约 30.9GB。

结论：7B 全参 GRPO 能在 4 张 V100 32GB 上跑通，但显存余量很紧，速度也慢。后续不建议继续全参训练，建议转 LoRA：冻结 7B base，只训练 adapter，保留 GRPO 的在线采样和规则 reward。

## 下一步建议

1. 先跑 LoRA 1-step smoke。
2. smoke 通过后用当前 focused 数据跑完整 1 epoch。
3. 训练后用本文件中的 `compound245` 命令复测。
4. 对比 failed_cases，重点看以下指标是否改善：
   - `probe_then_migration` pass_rate 是否从 0% 上升。
   - `migration_feasibility_with_image` pass_rate 是否从 0% 上升。
   - `finish_after_tool` 错误是否明显下降。
   - `use_visual_probe/use_image/user_query` 缺失是否明显下降。
5. 如果单 epoch 后仍主要是字段缺失，不急着扩大模型或数据，优先加强 schema 约束样本和 reward 权重。
