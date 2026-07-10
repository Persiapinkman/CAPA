# Qwen2.5-7B Post-Training Lessons and Active Path

Date: 2026-07-10

## Active Path

后续主链路固定为：

```text
SFT warmup -> TRL GRPO -> held-out eval -> PPO candidate
```

当前硬件约束是 V100-32G，因此默认使用：

- Qwen2.5-7B-Instruct
- fp16
- LoRA, not full-parameter training
- TRL + Transformers generation
- `attn_implementation=sdpa`
- no vLLM for the first stable route

## Files Kept

- `training/planner_grpo_seed_v1/scripts/train_planner_grpo.py`
  - 保留作为 prompt/reward 公共函数来源。
  - 不再作为主训练入口。
- `training/planner_grpo_seed_v1/scripts/train_planner_grpo_trl.py`
  - 当前 GRPO 主入口。
- `scripts/run_qwen25_7b_trl_grpo_lora.sh`
  - 当前 GRPO 复用启动脚本。
- `outputs/planner-grpo-qwen25-7b-trl-lora-v1`
  - 已跑通的 v1 adapter 和 checkpoint。

## Files Removed

删除范围限于可再生临时产物和不再采用的 verl 入口：

- `outputs/planner-grpo-qwen25-7b-trl-lora-dryrun`
- `outputs/planner-grpo-qwen25-7b-trl-lora-dryrun2`
- `outputs/planner-grpo-qwen25-7b-trl-lora-smoke`
- `scripts/setup_verl_env.sh`
- `scripts/run_qwen25_7b_verl_grpo.sh`
- `scripts/run_qwen25_7b_verl_grpo_lora.sh`
- `scripts/patch_verl041_v100_compat.py`
- `training/planner_grpo_seed_v1/scripts/prepare_verl_grpo_data.py`
- `training/planner_grpo_seed_v1/scripts/verl_reward_planner_grpo.py`
- `training/planner_grpo_seed_v1/verl_data`
- `training/planner_grpo_seed_v1/verl_para.sh`
- `training/verl_flash_attn_shim`

## Pitfalls Captured

1. **Do not use eager attention for Qwen2.5-7B on this host.**
   `attn_implementation=eager` produced visibly corrupted output even on short
   JSON prompts. `sdpa` and the default attention path generated normal text.

2. **V100 fp16 sampling needs invalid-logit protection.**
   Without `remove_invalid_values=true` and `renormalize_logits=true`,
   long planner prompts can hit CUDA assertions in multinomial sampling:
   probability tensor contains inf, nan, or negative values.

3. **Do not start with vLLM/verl on this machine.**
   vLLM/verl adds another compatibility layer on V100. The stable first route is
   TRL GRPO with regular Transformers generation.

4. **Do not mask truncated completions until stopping is fixed.**
   The model often emits the correct first JSON and then keeps generating.
   With `mask_truncated_completions=true`, all clipped completions can be masked
   away and produce no useful update.

5. **GRPO without SFT warmup wastes samples.**
   Some groups have zero reward variance and several outputs do not stop after a
   single JSON. SFT warmup should teach exact JSON shape and EOS behavior before
   longer GRPO.

6. **Existing eval files are not clean generalization eval.**
   `planner_grpo_compound245_eval_cases.jsonl` can be used for regression, but
   not for held-out claims because prior train/eval assets overlap.

## SFT Warmup Definition

SFT warmup trains the model to emit exactly one planner JSON decision for the
current step and then stop.

Input:

- Prompt from the same planner context builder used by GRPO.

Target:

- Canonical JSON converted from `expected_decisions[step]`.
- No markdown.
- No extra text.
- EOS after the JSON.

Primary metrics:

- JSON valid rate
- action match
- argument match
- `finish_after_tool` match
- first-JSON reward
- extra text after JSON ratio
- exact stop / EOS success
- category-level score

The immediate SFT goal is not business reward maximization. It is to make GRPO
cheaper and cleaner by reducing malformed or never-ending completions.
