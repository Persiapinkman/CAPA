# Qwen3.5-4B GSM8K 32-case SFT smoke 结果

日期：2026-07-15
模型：`/raid/zkq/models/Qwen3.5-4B`
结论：**P1 SFT smoke gate 通过；GSM8K 本轮推荐 checkpoint-50，不推荐 final-100。**

## 1. 本轮范围

本轮执行公开数据流程的前三步：冻结 GSM8K/MATH-lighteval、实现严格 adapter/verifier，
并完成 GSM8K 32-case assistant-only LoRA SFT overfit。只评测 train 和 development；
128-case `sealed_test` 未生成、未查看结果，也未参与 checkpoint 选择。

数据 manifest SHA256：
`de52206d4692eb41932992c308066747276c823b760e82d5b27a363ff3a27dd4`。
train/development 均为从 GSM8K official train 按 seed 42 确定性选出的互斥 32 条；
sealed_test 来自 official test。上游 train/test 归一化交叉数和派生 split 交叉数均为 0。

## 2. 冻结训练配置

- 4×V100 32 GiB：物理 GPU `0,1,3,5`；没有占用他人的 GPU 2/4；
- Qwen3.5 native non-thinking prompt，TRL Qwen3.5 training template；
- `assistant_only_loss=true`，`completion_only_loss=false`；
- assistant mask 非空率 100%，监督 `<|im_end|>` 率 100%；
- `max_length=1024`，`packing=false`，无 gold/EOS 截断；
- LoRA r=16、alpha=32、dropout=0，152 个 LoRA module，14,376,960 个可训练参数；
- fp16、SDPA、gradient checkpointing；
- per-device batch 1、gradient accumulation 2、global batch 8；
- 100 optimizer steps，constant-with-warmup，warmup 5，LR `2e-5`；
- greedy generation，`max_new_tokens=384`，seed 42。

冻结环境为 Python 3.10.12、PyTorch 2.6.0+cu124、Transformers 5.12.0、TRL 1.8.0、
PEFT 0.19.1、Accelerate 1.14.0、Datasets 5.0.0。

## 3. 真实生成结果

### Train 32

| 模型点 | Exact | Strict format | Loose exact | EOS | Clipped | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| Base | 19/32 = 59.38% | 20/32 = 62.50% | 22/32 = 68.75% | 19/32 = 59.38% | 13/32 = 40.63% | 320.59 |
| Checkpoint-50 | 26/32 = 81.25% | 29/32 = 90.63% | 26/32 = 81.25% | 31/32 = 96.88% | 1/32 = 3.13% | 143.44 |
| Final-100 | 28/32 = 87.50% | 31/32 = 96.88% | 28/32 = 87.50% | 32/32 = 100% | 0/32 = 0% | 128.91 |

Checkpoint-50 相对 base 修复 7 个 exact case、没有破坏原本正确 case；final-100 修复 11 个，
但破坏 2 个原本正确 case。两者均显著学会最终答案 marker 和主动停止。

### Development 32

| 模型点 | Exact | Strict format | Loose exact | EOS | Clipped | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| Base | 22/32 = 68.75% | 22/32 = 68.75% | 22/32 = 68.75% | 22/32 = 68.75% | 10/32 = 31.25% | 310.47 |
| Checkpoint-50 | **26/32 = 81.25%** | **31/32 = 96.88%** | **26/32 = 81.25%** | **32/32 = 100%** | **0/32 = 0%** | 123.69 |
| Final-100 | 21/32 = 65.63% | 30/32 = 93.75% | 22/32 = 68.75% | 32/32 = 100% | 0/32 = 0% | 106.75 |

Checkpoint-50 相对 base 新增 5 个正确 case、损失 1 个，净增 4/32（+12.5pp）；
final-100 新增 4 个但损失 5 个，净损失 1/32。final-100 仍保持更好的格式和停止行为，
但后半程已经把“输出更确定”推成对未见题的过拟合，不能只看 train loss 或 EOS 选模型。
即使 final teacher-forced loss 已接近 0，train generation exact 也只有 87.5% 而不是 100%，
再次说明 SFT loss/token accuracy 不能替代真实自回归生成评测。

## 4. Loss、数值与资源审计

- step 1 train loss：0.74828；final：0.08585；最低单批：0.05548；
- train mean token accuracy：0.84835 -> 0.97308，最高 0.97854；
- dev loss：step 10 为 0.6835，最低点 step 60 为 0.4225，final 回升到 0.5521；
- 100/100 optimizer step 完成，训练主循环 666.8 秒；
- loss/eval loss/grad norm 非有限值计数为 0；
- 各 rank 峰值 allocated 8.89--9.08 GiB，结束时仍有 22.07--22.38 GiB device free。

这证明当前 4 卡 DDP SFT 的模型、label、LoRA、fp16 scaler、反向传播、保存和重载链路均可用。
SFT 的显存远低于 32 GiB，因此之前 GRPO OOM 更符合 generation placement/并发配置问题，
而不是 Qwen3.5-4B 文本模型本身无法放入 V100。

## 5. 本轮实际踩到并修复的坑

1. 基线 v1 把 batched generation 在 EOS 后的 PAD 计入 completion length。修复为第一个 EOS
   截断，并加入回归测试；base v2 是正式基线，v1 仅保留审计。
2. Transformers 5.12 的 `SFTConfig` 已移除 `overwrite_output_dir`。首次启动在模型加载前、
   optimizer step 0 失败；现已做完整 kwargs/signature 审计，并让 dry-run 实际构造 SFTConfig。
3. Qwen3.5 目录的 `architectures` 声明为 `Qwen3_5ForConditionalGeneration`，TRL 按字符串路径
   会自动加载包含视觉封装的 723 组权重。第二次启动在 optimizer step 0 被模型审计拒绝。
   入口现显式使用 `AutoModelForCausalLM`，加载 426 组文本权重；成功运行的 base class 为
   `Qwen3_5ForCausalLM`，visual module 数为 0。
4. 成功训练结束时出现一次未显式销毁 NCCL process group 的 warning；训练产物不受影响，入口已
   在 `finally` 中加入 `destroy_process_group()`，后续运行会正常清理。

前两个失败目录分别是 `...overfit_v1` 和 `...overfit_v2`，均没有 optimizer step；成功训练是
`...overfit_v3`。失败产物未删除，便于审计版本兼容路径。

## 6. Gate 判定

| SFT hard gate | 结果 |
|---|---|
| Label contract 100% | PASS |
| Gold/EOS 无截断 | PASS |
| 32-case train generation 明显提升 | PASS：checkpoint-50 +21.88pp，final-100 +28.13pp |
| loss 下降时 dev exact 不反向崩溃 | PASS at checkpoint-50：+12.5pp；final-100 已过拟合 |
| 无 NaN/Inf | PASS |
| GradScaler/显存门 | PASS |

因此 GSM8K SFT smoke 整体通过，checkpoint-50 是本轮 promotion point。它只用于证明流程和
checkpoint 选择机制，不应直接当作后续 MATH 正式模型。下一阶段应从原始 base 重新开始
MATH-lighteval 1,024-case SFT screen，以隔离数据效应；继续沿用 strict generation、EOS、
clipped、train/dev loss 和逐长度桶审计。
本轮每个 split 只有 32 条，checkpoint-50 的 +12.5pp 只是 smoke evidence，不作统计显著性或
广泛泛化声明；是否稳定提升必须由下一阶段更大的 development 和最终 sealed evaluation 验证。

## 7. 主要产物

- 数据与 manifest：`training/public_sft_grpo_v1/data/gsm8k_sft32_v1/`
- 正式 base：`experiments/runs/20260715_qwen35_4b_gsm8k_base_eval_sft32_v2/`
- 成功训练：`experiments/runs/20260715_qwen35_4b_gsm8k_sft32_overfit_v3/`
- checkpoint-50 评测：`experiments/runs/20260715_qwen35_4b_gsm8k_sft32_checkpoint50_eval_v1/`
- final-100 评测：`experiments/runs/20260715_qwen35_4b_gsm8k_sft32_final100_eval_v1/`
- 训练入口：`training/public_sft_grpo_v1/scripts/train_qwen35_gsm8k_sft.py`
- 评测入口：`training/public_sft_grpo_v1/scripts/eval_qwen35_gsm8k.py`
- 启动器：`scripts/run_qwen35_gsm8k_sft32.sh`

成功 final adapter SHA256：
`b35c5d9cd7677ecb0b78107861d02921e778992755621deac49a0629c88b6758`。
最终代码回归测试为 12/12 通过。
