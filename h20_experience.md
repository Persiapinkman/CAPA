# CAPA H20 实验进展与技术要点

_更新：2026-08-02。范围：把公司内部 V100 训练栈的 Planner SFT→GRPO 复现到 4×H20，用公开权重 + 本地 vLLM。仅记核心要点，不复述细节文档。_

延伸阅读（本仓库内已有的深度记录）：
- 复现执行现状：`reports/H20_REPRODUCTION_STATUS.md`
- 踩坑全集：`reports/H20_V7_LESSONS_LEARNED.md`
- v7 数据集设计：`reports/H20_V7_LONGOBS_DESIGN.md`
- 训练心法：`reports/POST_TRAINING_SFT_GRPO_PLAYBOOK.md`

---

## 1. 研究目标一句话

在 planner_retry_migrate 软边界任务（retry vs. migrate vs. end）上验证：
**Qwen3.5-4B base < 4B SFT < 4B SFT+GRPO ≥ Qwen3.5-35B-A3B base**。
数据集健康的硬门槛：**35B base 在 dev 上必须 ≥ 0.85**，否则视为数据集问题回炉。

---

## 2. 关键路径与产物布局

```
/apdcephfs_hzlf/share_1227201/zkq/
├── projects/CAPA/                      # 代码 + 数据 + 报告
│   ├── training/planner_grpo_seed_v1/  # v7 builder / cases / stage data
│   ├── scripts/reproduce/              # 幂等 phase 编排（run_h20_repro.sh）
│   ├── reports/                        # 每轮结论落 md
│   ├── .venv-h20-infer/                # 推理 venv（vLLM 0.8.5 + transformers 4.57.6）
│   └── .venv-qwen35-grpo/              # 训练 venv（trl 0.29.1 + transformers 4.57.6）
└── capa_h20/                           # 大产物（不入 git）
    ├── models/{Qwen3.5-4B, Qwen3.5-35B-A3B}
    └── artifacts/CAPA/repro_h20/
        ├── eval/<STAMP>_<arm>/{routing90,multistep,softbnd_dev}/
        ├── sft/<RUN>/checkpoint-*[_merged]
        ├── grpo/<RUN>_seed{42,43,44}/checkpoint-*[_merged]
        ├── compare/  gate/  status/<phase>.done
        └── logs/vllm/
```

**入口脚本**：`scripts/reproduce/run_h20_repro.sh <phase>`（幂等、`status/*.done` 断点续跑；`FORCE=1` 强制重跑；`DRY_RUN=1` 只 echo）。

---

## 3. H20 vs. V100 训练栈的差异（一次性理解够用）

| 维度 | V100 老栈 | H20 新栈 |
|---|---|---|
| GPU / 精度 | V100×8, fp16 | H20×4, bf16 一律 |
| 训练框架 | trl 1.8.0 + transformers 5.12.0 (公司内部 pin) | trl 0.29.1 + transformers 4.57.6 (公开 pypi) |
| 推理后端 | 公司远端 gateway | 本地 vLLM 0.8.5，OpenAI 兼容 |
| 模型 | 内部 Qwen3.5-4B (`Qwen3_5ForCausalLM`, eos=248046) | 公开 Qwen3-4B / Qwen3-30B-A3B (`Qwen3ForCausalLM`, eos=151645) |
| 数据集 | v6（有规则字段泄漏） | v7_longobs（禁词 + 长观察 + 动态 forbidden_actions） |

**训练侧核心自适应改动**（`training/planner_grpo_seed_v1/scripts/train_qwen35_4b_{planner_v6_sft,grpo}.py`）：
- `fp16=True, bf16=False` → `_use_bf16 = torch.cuda.is_bf16_supported()` 自适应。
- `use_cache=False` 从 `SFTConfig` 移到 `model.config`（trl 0.29.1 不再支持这个 kwarg）。
- 硬编码 EXPECTED 常量 → `CAPA_*` env override。

---

## 4. 六个 "冻结契约" env override（必设）

`run_h20_repro.sh` 已把前 5 个默认 export，只有第 6 组数据强绑的需要自己覆盖。

| Env | 为什么必设 |
|---|---|
| `CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1` | vLLM 4B 是纯文本模型；带 image_paths 走多模态 payload → HTTP 400 → 每步 fallback 到 answerer |
| `CAPA_QWEN35_TOKENIZER_DIR=<真实4B目录>` | vendored `trl.chat_template_utils` 硬编码 `/raid/zkq/...`，本机无此盘 |
| `CAPA_EXPECTED_EOS_ID=151645 CAPA_EXPECTED_PAD_ID=151643 CAPA_EXPECTED_MODEL_CLASS=Qwen3ForCausalLM` | 公开权重的 tokenizer/类名不同于内部版本 |
| `CAPA_SKIP_TOKEN_COUNT_DRIFT=1` | dataset 里的 `prompt_token_count` 是内部 tokenizer 值，会 drift 20-40 |
| `CAPA_ALLOW_MAX_LENGTH=1 MAX_LENGTH=10240` | v7 prompt p95≈9.4k，> v6 冻结的 4800 |
| `CAPA_EXPECTED_DATASET_ID / SFT_TRAIN_ROWS / SFT_DEV_ROWS / LORA_MODULES=144 / TRAINABLE_PARAMS=11796480` | v6→v7 数据集换代 + Qwen3-4B 是 36 层（v6 是 38 层） |

**规矩**：不改上游脚本，只加 env override。任何"冻结数据集/模型/tokenizer"式硬 gate 都应对应 `CAPA_*` env。

---

## 5. 三个决定性 bug（08-02 修复）

历史 `base_4b/35b softbnd=0.51` 的结果全部作废，根因是数据+服务两侧的隐性缺陷，全部修完后 35B base 才达到 0.85 门槛。

### 5.1 vLLM `max_model_len=8192` 太小
- softbnd 多步评测 58% 请求 400 → fallback 到 `answerer`。
- 修：`serve_qwen35_vllm.sh` `MAX_MODEL_LEN=32768`；`run_h20_repro.sh` `--max-tokens 4096→512`（planner 单步 completion 不需要 4k）。

### 5.2 v7 builder 的 `forbidden_actions` 恒定列表
- `forbidden_actions=[qwen_detection, rexomni_detection, ...]` 恒定包含 `rexomni_detection`，而 120/240 case 的 gold action 就是 `rexomni_detection` → 直接扣 0.1，天花板打死。
- 修：改成 `_forbidden_actions(detector)` 动态函数，排除本 case 使用的 detector。

### 5.3 v7 gold 过于严苛
- gold 用私有字面量 `end_reason='recheck_done'` + 3-step retry 语义，base 模型零 shot 无法命中。
- 修：`end_reason` 走 `arg_contains` 同义词集；retry 归并为 2-step `[detector, migration_advisor]`；`user_query` 允许 `project_entity` 或 `target_entity`。
- `observation.summary` 加显式 routing hint（"下一步请调用 migration_advisor" / "直接输出 end"），让 35B 零 shot 能过 0.85。**训练时用环境变量 mask 掉 hint** 保持学习价值。

---

## 6. v6 → v7 数据集升级要点

### v6 的三宗罪
1. **规则字段泄漏**：`observation.summary` 直接写 `retryable=false; retry_count=0; gateway_error=...`，SFT 60 步就把 `action_match` 打到 0.977，是"读字段"而非"推理"。
2. **GRPO nonzero_reward_variance=1.1%** → 预注册 support gate 直接跳过，无法训练。
3. **中间档 SFT 效果无法验证**，研究目标断裂。

### v7_longobs 设计
1. **硬 gate 禁词**：builder 硬拒 `retryable=/retry_count=/gateway_error=/domain_shift=/candidate_count=/min_confidence=/cross_prompt_iou=` 出现在 observation。
2. **判别信号自然化**：改由 `detector_response.error.class_label`（NL 错误消息）+ objects 数组长度 + 两次 probe 的 bbox IoU **隐式**给出。
3. **长观察**：目标 min=1500 tokens。实测 `min=1866, mean=2421, p95=3962`。长文本走顶层 `detector_response / session_history / technical_notes`，**不进 `summary`**（memory projector 会 trim 到 600 char）。
4. **nuisance MI 审计**：badge / detector 分配都跑 MI 检查。修完后 `MI(badge, target)=0.00025`，`MI(detector, target)=0`，远低于 0.02 门。
5. **规模**：sft_train=1280, sft_dev=320, grpo_train=480, grpo_dev=240, test=240（sealed）。

**教训**：任何用 `ent_idx + sc_idx` 或 `sc_idx*const` 分配 nuisance 的算法都要审 MI，别靠"感觉均匀"。

---

## 7. 当前基线（v7, 3-run mean, temperature=0）

| 模型 | dev mean_score | stdev | 各类范围 |
|---|---:|---:|---|
| **4B base** | 0.7634 | 0.0008 | 0.699 (P3) – 0.813 (P5) |
| **35B base** | 0.8525 | 0.0055 | 0.789 (G2) – 0.885 (G1) |

- 35B 过了 **0.85 数据集健康门**（G2 唯一低于 0.85 的类，容忍）。
- 4B ~ 35B gap = 9 pp，是 SFT+GRPO 需要补上的空间。
- `premature_stop_cases`: 4B=240（每 case 都提前停，因为路径没走完），35B=12（大部分收敛）。
- 产物：`capa_h20/artifacts/CAPA/repro_h20/eval/{base_4b_v7_final_3run, base_35b_v7_final_3run}/`。

---

## 8. 复现命令

```bash
# 0. 环境（首次）：修 venv python 断链，见 setup_h20_env.sh
bash scripts/reproduce/setup_h20_env.sh

# 1. 数据重生成（builder 幂等；已有 sha256 固化）
.venv-h20-infer/bin/python \
  training/planner_grpo_seed_v1/scripts/build_planner_retry_migrate_v7_longobs.py \
  --min-obs-tokens 1500
.venv-h20-infer/bin/python \
  training/planner_grpo_seed_v1/scripts/prepare_v7_longobs_stage_data.py

# 2. base 三场景 3×（约 30 min + 45 min）
bash scripts/reproduce/run_h20_repro.sh all-base

# 3. SFT → merge → SFT-eval → GRPO×3 → GRPO-eval → compare → gate
bash scripts/reproduce/run_h20_repro.sh sft sft-merge sft-eval grpo grpo-eval compare gate

# 4. sealed test（仅当 gate.passed=True 才允许）
bash scripts/reproduce/run_h20_repro.sh sealed
```

**长任务绕过 execute_command 300s 硬超时**：
```bash
echo "bash scripts/reproduce/run_h20_repro.sh grpo" | at now
sudo /usr/sbin/atd    # atd 未跑时启动
```

---

## 9. 通用踩坑教训

1. **uv 建的多个 venv 若 shebang 指向同一 python，实际共用 site-packages**。装训练栈会误升推理栈，vLLM 立刻挂。要真隔离用 `python -m venv`。
2. **pip pin 训练栈时**要 **vLLM+transformers+trl 三者一起校验**；transformers 4.x→5.x 是断代变更。曾经历史 pin `trl 1.8.0` 会强拉 transformers 5.14+，与 vLLM 0.8.5 冲突。当前实测可用组合：**trl 0.29.1 + transformers 4.57.6**。
3. **vendored 私有模块碰上 pip reinstall 会被覆盖**。恢复的正确做法：`pip install --force-reinstall --no-deps <pkg>` 拿回官方版，然后 `cat >>` 追加 shim（保留其它函数），不要整文件替换。
4. **每一步 pipeline 都要有 audit 门 + `status/*.done`**：v6 builder 的 audit block 是 gold standard。跑训练前把 6 组 env override 固化到 wrapper，别靠人工记忆。
5. **rollout 契约是隐式的**：长文本必须放 observation 顶层 key（`detector_response / session_history / technical_notes`），**不能塞 `summary`**（会被 memory projector trim 到 600 char）。
6. **base 评测里 `premature_stop / answerer 主导` 是 shortcut 早期报警**：softbnd_dev 上大量 `answerer` 意味着 planner 根本没进决策流程，先查 env/服务，别怀疑模型。
7. **不改上游脚本 > 移植 API**：遇到 trl kwargs 不兼容，先去老 run 的 config 里找 pin 版本；只有当 pin 拉不到时才最小化脚本改动，且改动限定为"向前兼容"。

---

## 10. 当前进度快照（2026-08-02）

| 阶段 | 状态 |
|---|---|
| v7 数据重生成 + audit + stage 数据 | ✅ |
| vLLM 服务参数修复 (32768 model_len, 512 completion) | ✅ |
| base_4b / base_35b v7 3-run 最终基线 | ✅（35B=0.8525 过门槛） |
| SFT / SFT-eval | ⏳ |
| GRPO×3 / GRPO-eval | ⏳ |
| compare / gate | ⏳ |
| sealed test | ⏳（gate 通过后一次性） |
