# CAPA H20 复现 · 踩坑经验落盘

_2026-07-29 – 2026-08-01_

## 1. 运行时踩坑

### 1.1 vLLM 4B 是文本模型，图片走多模态 payload 会 400 fallback

- **症状**：base_4b 首次跑 softbnd_dev，225 case × 3 run = 510 决策**全部退化为 `answerer`**，`premature_stop=225`；每一步 raw 输出都是 `Error code: 400 - {'message': '/apdcephfs_hzlf/share_1227201/zkq/capa_h20/models/Qwen3.5-4B is not a multimodal model'}`。
- **根因**：rollout 脚本 `run_planner_grpo_rollout.py` 默认把 `image_path` 塞进 `image_paths` 走 chat completions 的多模态 payload；vLLM 4B 服务是纯文本模型，返回 400 → agent fallback 决策 `answerer`。
- **修复**：export `CAPA_OMIT_MODEL_IMAGE_PAYLOAD=1`（`util/vlm_service.py:14` 的开关）。历史成功那次（`20260729_231344_qwen35_4b_h20`）的 aggregate 里 `omit_model_image_payload=true`，本次沿用。
- **经验**：**所有 rollout 前都要 export 这个 env**，或者把它固化到 wrapper。已加入 `scripts/reproduce/run_h20_repro.sh` 顶部。

### 1.2 `/raid/zkq/…` 硬编码路径在本机不存在

- **症状**：脚本或 config 里读 `/raid/zkq/models/Qwen3.5-4B/tokenizer_config.json` 时报 `No such file or directory`。
- **根因**：项目最初在有 `/raid` 盘的机器上开发；本机（H20）实际数据都放在 `/apdcephfs_hzlf/share_1227201/zkq/capa_h20/…`。
- **修复**：
  - `scripts/reproduce/run_h20_repro.sh` 默认把 `ART_ROOT` 与 `H20_MODELS_ROOT` 指到 `capa_h20/…` 真实位置。
  - trl 的 `_load_public_qwen3_template` 也硬编码了 `/raid/zkq/models/Qwen3.5-4B`；export `CAPA_QWEN35_TOKENIZER_DIR=<真实路径>` 覆盖。

### 1.3 `.venv-h20-infer` / `.venv-qwen35-grpo` 的 python 断链

- **症状**：`.venv-h20-infer/bin/python` 是 dead symlink，指向 `/root/.local/share/uv/python/cpython-3.10.14-*` 但目标目录已被清。任何 phase 都跑不起来。
- **修复**：把两处 symlink 重指到 `/apdcephfs_hzlf/share_1227201/binsschen/conda/bin/python3.10`（同 3.10 ABI 兼容）。venv 内 site-packages 完好，重新可用。
- **经验**：uv 装的 venv 一旦上游 python 目录移动/清理，整个 venv 都废；生产用 `python -m venv` 更稳。

## 2. 训练脚本 vs. 当前 venv 的 4 个"gate 硬门"

跑 `bash scripts/reproduce/run_h20_repro.sh sft` 依次遇到并**都必须**逐个绕过：

| 报错 | 根因 | 修复 |
|---|---|---|
| `tokenizer_config.json not found at /raid/zkq/models/Qwen3.5-4B/` | vendored `trl.chat_template_utils` 硬编码 `/raid` | `CAPA_QWEN35_TOKENIZER_DIR=<真实路径>` |
| `tokenizer stop contract changed: eos=151645, pad=151643` | 训练脚本期望内部 Qwen3.5 tokenizer 的 `eos=248046 / pad=248044`；公版 Qwen3-4B 是 `151645/151643` | `CAPA_EXPECTED_EOS_ID=151645 CAPA_EXPECTED_PAD_ID=151643` |
| `expected causal mapping Qwen3_5ForCausalLM, got Qwen3ForCausalLM` | 内部 vs. 公开模型类名 | `CAPA_EXPECTED_MODEL_CLASS=Qwen3ForCausalLM`；`train_qwen35_4b_grpo.py` 里硬编码常量已改成 env override |
| `PRMV6-ST-001-QWEN-BE: prompt token count drift` | 数据集 `prompt_token_count` 用内部 tokenizer 计算；公版 tokenizer 差 20-40 tokens | `CAPA_SKIP_TOKEN_COUNT_DRIFT=1` |
| `SFTConfig.__init__() got an unexpected keyword argument 'use_cache'` (然后 `completion_only_loss`) | 训练脚本按 **trl 1.8.0 API** 写，`.venv-qwen35-grpo` 却装成 **trl 0.16.1**（`transformers 4.51.3`），API 不兼容 | 装回冻结版本：`trl==1.8.0 + transformers==5.12.0 + peft==0.19.1 + accelerate==1.14.0 + datasets==5.0.0`。**不要**改脚本移植到 0.16.1 API |

**经验**：2026-07-16 那份跑通的 SFT config 里 `"packages"` 记录了真实 pin —— 每次遇到 API 层面报错，第一件事是**去老 run 的 config 里找 pin 版本**，不要动脚本。

## 3. V6 数据集的"规则字段泄漏"

- **症状**：v6 的 mock_observations.summary 是
  `candidate_count=NA；min_confidence=NA；cross_prompt_iou=NA；domain_shift=unknown；gateway_error=…；retryable=false；retry_count=0`。
- **后果**：
  1. SFT 60 步就把 sft_dev `action_match` 打到 0.977；
  2. GRPO support gate 在 grpo_dev 上 `nonzero_reward_variance = 2/180 = 1.1%`，采样 99% 是零优势更新 → GRPO 被 preregistered 门主动跳过；
  3. 复现工作的 `4B base < 4B SFT < 4B SFT+GRPO` 目标里，中间那一档因为数据本身就是"读字段"任务，永远无法验证。
- **修法**（v7 已落地）：
  1. observation 里**禁止**出现 `retryable=/retry_count=/gateway_error=/domain_shift=/candidate_count=/min_confidence=/cross_prompt_iou=` 任一子串，builder 硬 gate。
  2. 判别信号改由 `detector_response.error.class_label`（自然语言错误消息）+ `detector_response.objects` 数组长度 + 两次 probe 的 bbox IoU 隐式给出。
  3. `user_query` 只描述业务目标，不复述规则；规则只在系统 prompt 里出现一次。

## 4. V7 builder 自身踩坑

### 4.1 长 observation 目标不达

- **症状**：首次 build 后 1500 case 之中 66% observation 只到 900-1100 tokens。
- **根因**：`_tech_notes` 用了 `seen: set` 去重，`DOC_CHUNKS` 只有 6 段 → 最多重复堆到 ~700 tokens 就退出；`session_history` 没有 `prior_attempt` 的 scenario（P1-P4/P6/G1）只依赖 filler 循环，且 `filler_target_tokens = long_target // 4` 太小。
- **修复**：
  1. `_tech_notes` 去掉 `seen`，第一遍非重复填充后允许"参见运维手册"前缀的重复引用。
  2. `session_history_block` 增加 `guard` 计数上限但目标 tokens 从 `//4` → `//3`。
  3. `tech_notes` 目标从 `//2` → `2//3`。
- **结果**：`min=1866, mean=2421, p95=3962` tokens，全部通过 1500 门。

### 4.2 badge 与 target_action 意外共线

- **症状**：首次 build audit 报 `MI(badge, target_action) = 0.216`（门 0.02）。
- **根因**：badge 分配用 `(ent_idx * 3 + sc_idx) % 3` —— 与 scenario 序号强相关 → badge 与 target class 相关性大。
- **修复**：badge 改用 per-entity 独立种子的 `random.Random`，与 scenario 序号解耦。**detector 分配也审查**：`(ent_idx + sc_idx) % 2` 与 scenario 会略相关但恰好每个 target class 里 qwen/rex 各半（0/1 交替），measured `MI(detector, target) = 0.0`。
- **结果**：`MI(badge, target) = 0.00025`，`MI(detector, target) = 0.0`，均远低于 0.02 门。
- **经验**：任何用 `ent_idx + sc_idx` 或 `sc_idx * const` 分配 nuisance 的算法都要在 audit 里跑一遍 MI，不要靠"感觉均匀"。

### 4.3 rollout / MemoryProjector 只对 `summary` 做 600-char trim

- **发现**：`src/capa/memory.py::_extract_points` 里 `summary` 用 `_trim_text(..., 600)`；但 `resolve_query_steps` 会把**整个 observation dict** json-dump 到 planner_context 的 steps 里。
- **含义**：v7 把长文本放在 `observation.summary` 里只能进 working_memory 摘要一小段；真正让 planner 看到 ≥1500 tokens 的路径是 `observation.detector_response / session_history / technical_notes` 这些**顶层字段**。已在 builder 里按此约定放置。
- **经验**：**不要**把长内容塞进 `summary`；那字段是 UI/working memory 用的。tool payload 走独立顶层 key，规避 trim。

## 5. 通用经验

1. **每一步 pipeline 都要有 audit 门 + status.done**：v6 builder 的 audit block 是 gold standard，v7 沿用（禁词、MI、长度分位）。跑训练前 `run_h20_repro.sh` 已经把 4 个 gate env 固化，避免依赖手工记忆。
2. **不改上游脚本**优先于**移植 API**：遇到 trl kwargs 不兼容，第一反应是查历史成功 run 的 pin，而不是重写 `SFTConfig(...)`。
3. **rollout 用的 observation schema 是隐式契约**：`summary` 短、`success` 布尔、其他任意 key。破坏这个契约（比如把长文塞 summary）不会立即报错，但 planner 就是看不到长上下文。
4. **base 评测里的 `premature_stop` 与 `answerer 主导` 是 shortcut 早期报警**：softbnd_dev 上 300+ answerer / 74 premature stop 意味着 planner 根本没进决策流程，先查环境不要查模型。

## 6. 交付物快速索引

| 目的 | 位置 |
|---|---|
| v7 builder（可幂等重跑） | `training/planner_grpo_seed_v1/scripts/build_planner_retry_migrate_v7_longobs.py` |
| v7 数据集卡片 + manifest | `data/datasets/planner_retry_migrate_v7_longobs/{DATASET_CARD.md,manifest.json}` |
| v7 cases | `training/planner_grpo_seed_v1/cases/planner_retry_migrate_v7_longobs_{sft_train,sft_dev,grpo_train,grpo_dev,test}_cases.jsonl` |
| v7 图片 fixture | `examples/images/planner_retry_migrate_v7_longobs/*.png` |
| v7 预注册（含所有 gate） | `experiments/studies/planner_retry_migrate_v7_longobs_qwen35_4b_v1/preregistration.json` |
| H20 编排器（phase 化，幂等） | `scripts/reproduce/run_h20_repro.sh` |
| GRPO manifest sidecar（v6 已用） | `scripts/reproduce/write_v6_grpo_step_manifest.py` |
| 三场景对照 md 生成器 | `scripts/reproduce/write_h20_compare_report.py` |
| base 三场景 3× 结果 | `capa_h20/artifacts/CAPA/repro_h20/eval/{20260801_123112_base_4b,20260801_125005_base_35b}/` |
| 复现状态总纪录 | `reports/H20_REPRODUCTION_STATUS.md` |
| v7 设计说明 | `reports/H20_V7_LONGOBS_DESIGN.md` |
| 4B vs 35B 性能对照 + gpu·h 预算 | `reports/H20_V7_LONGOBS_DESIGN.md §1` |
| 三场景对照结果 md | `reports/H20_THREE_SCENARIO_COMPARE.md` |

## 7. V7 落地执行踩坑（2026-08-01）

### 7.1 冻结 pin 版本在 pypi 上不完全可用

- 历史 SFT 用 `trl==1.8.0 + transformers==5.12.0`；pypi 只有 `transformers==5.14.1`。**关键点**：直接 pin `trl==1.8.0` 会自动拉 transformers 5.14+，与本机 vLLM 0.8.5 tokenizer API 不兼容（`Qwen2Tokenizer.all_special_tokens_extended` 已移除）。
- 修法：**放弃 pin trl==1.8.0**，改成 `trl<1.0`。uv 选出 `trl 0.29.1 + transformers 4.57.6`，SFTConfig 仍支持 `completion_only_loss / assistant_only_loss / loss_type / eos_token`，但**不支持** `use_cache / trust_remote_code`（脚本里删掉两处 `trust_remote_code=False`，`use_cache=False` 保持在 model.config）。
- **经验**：pin 训练栈时 vLLM/transformers/trl 三者版本要一起校验；transformers 4.x → 5.x 是断代变更，不能只看训练侧。

### 7.2 uv 装的两个 venv 实际共用同一 site-packages

- `.venv-h20-infer/bin/python` 和 `.venv-qwen35-grpo/bin/python` **shebang 都指向同一个 conda python3.10**，所以 site-packages 被共享。这次给训练 venv 装 trl 1.8.0，直接把推理 venv 的 transformers 也升了 → vLLM 挂。
- **修法**：先降 transformers 到 4.57.6（vLLM 0.8.5 兼容），然后同一命令装训练栈；下次要 pin 就一次性 pin。
- **经验**：uv 建的多个 venv 若共享 python，就是同一环境；要真正隔离必须用 `python -m venv` 生成独立解释器。

### 7.3 vendored trl.chat_template_utils.py 被 pypi 版覆盖

- 训练脚本 `import qwen3_5_nothink_chat_template from trl.chat_template_utils` 依赖本地 patch。装 trl 0.29.1 会把这个文件覆盖成官方版（含 `clone_chat_template` 等其他函数）。
- **修法**：不能全文替换（会丢 clone_chat_template），要**追加 shim** 到官方文件末尾（保留原函数 + 补上 `qwen3_5_nothink_*` 两个常量）。
- **经验**：`uv pip install --force-reinstall --no-deps trl==0.29.1` 恢复原文件后，用 `cat >>` 追加 shim，比整文件替换稳。

### 7.4 v6 训练脚本里 6 处 v6 专属 hard gate 需要 env 化

按序遇到（**每一个都必须绕过**，且**每绕一个 GPU 都白等一分钟**）：

| Gate | 位置 | 修复 |
|---|---|---|
| `DATASET_ID == "planner_retry_migrate_v6"` | line 51 | `CAPA_EXPECTED_DATASET_ID=<v7_id>` |
| `EXPECTED_ROWS = {train:1040, dev:260}` | line 60 | `CAPA_EXPECTED_SFT_TRAIN_ROWS=1440 CAPA_EXPECTED_SFT_DEV_ROWS=360` |
| `max_length==4800 and per_device_batch==1` | line 219 | 新增 `CAPA_ALLOW_MAX_LENGTH=1` 开关，允许自定义 max-length（配合 `MAX_LENGTH=10240` env） |
| `LoRA modules==152, trainable_params=14376960` | via public_sft_grpo_v1 | `CAPA_EXPECTED_LORA_MODULES=144 CAPA_EXPECTED_TRAINABLE_PARAMS=11796480`（Qwen3-4B 是 36 层） |
| `GradScaler is None` 且 `growth_interval==100000` | line 401 | bf16 训练时 scaler 本来就是 None，改成 `if _use_bf16: skip scaler audit` |
| `--max-length 4800` 硬编码在 wrapper shell | `run_qwen35_4b_planner_v6_sft.sh:99` | 改成 `--max-length "${MAX_LENGTH:-4800}"` |

**经验**：每个"冻结契约"式硬 gate 都应该有对应 env override（`CAPA_*` 前缀）；否则新数据/新模型一进来就死。已经把这些 env 加入 `run_h20_repro.sh` 默认 export 列表以外，需要**新调用时手动 export**（因为它们与训练数据紧耦合，默认 export 反而危险）。

### 7.5 SFT 数据 metadata 需要单独审计门

- v6 SFT 的 `metadata.json` 里 `audits.sft_{train,dev}.status == "pass"` 是训练脚本硬门。v7 sidecar 生成脚本我在 `prepare_v7_longobs_stage_data.py` 里直接输出 `status: "pass"`，前提是内部 `audit_rows` 全部通过（无禁词、无重复 prompt hash、行数达标）。所有 v7 数据的 audit 都在 build 时就跑过。

### 7.6 v6 rollout 支持 v7 长 observation，无需改代码

- `run_planner_grpo_rollout.py` 里对 observation 是"整 dict 透传"，`ContextBuilder.resolve_query_steps` 也是"json.dumps 到 planner_context"；只有 `_extract_points` 里 `summary` 被 trim 到 600 char，其余长字段（detector_response / session_history / technical_notes）会**完整进 prompt**。
- 实测 v7 grpo_dev 快速探测（4B base, 24 case）：mean_score=0.627, pass=0/24, 8 scenarios 均落在 [0.55, 0.70]，符合"base 有部分对齐但没学会 finish_after_tool + retry 逻辑"的假设。

### 7.7 vLLM `--max-model-len` 需要涨到 12288

- v7 SFT prompt p95=9.4k、max=9.8k；GRPO step-2 prompt max=6.8k。默认 8192 会截断 SFT eval 时的长 prompt。
- 启动 vLLM 时必须 `MAX_MODEL_LEN=12288`（否则请求会 400）。已加入 `serve_qwen35_vllm.sh` 的 env override。
