# Public SFT/GRPO dataset research sources

Access date: 2026-07-15 UTC

The `research-lookup` skill was selected for this task. Its default `parallel-cli`
backend and API keys were unavailable in this environment. The web search endpoint
also returned a network error, so primary sources were retrieved directly over HTTPS.
This file preserves the queries, URLs, and the facts used in the associated report.

## Research questions

1. Which public, mature datasets support a short SFT -> GRPO -> evaluation loop on a
   4B model and V100 GPUs?
2. Which datasets offer deterministic or executable verification instead of an LLM
   judge?
3. Which public tool-use datasets are close to CAPA's multi-step routing task?
4. What reward/parser behaviors are known to create false progress or reward hacking?
5. Which datasets or benchmarks should remain evaluation-only?

## Primary sources

### Training frameworks and reward implementation

- TRL SFTTrainer documentation:
  https://huggingface.co/docs/trl/main/en/sft_trainer
- TRL SFTTrainer source documentation:
  https://raw.githubusercontent.com/huggingface/trl/main/docs/source/sft_trainer.md
- TRL GRPOTrainer documentation:
  https://huggingface.co/docs/trl/main/en/grpo_trainer
- TRL GRPOTrainer source documentation:
  https://raw.githubusercontent.com/huggingface/trl/main/docs/source/grpo_trainer.md
- Open R1 repository and reference recipes:
  https://github.com/huggingface/open-r1
- Open R1 reward implementation:
  https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
- Math-Verify repository and verifier guidance:
  https://github.com/huggingface/Math-Verify

### Math datasets

- MATH lighteval dataset card (MIT; 7,500 train / 5,000 test; difficulty and
  subject labels; full solutions):
  https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval
- Original MATH repository:
  https://github.com/hendrycks/math
- GSM8K dataset card (MIT; 7,473 train / 1,319 test):
  https://huggingface.co/datasets/openai/gsm8k
- Original GSM8K repository:
  https://github.com/openai/grade-school-math
- OpenR1-Math-220k dataset card (Apache-2.0; 93,733 default / 225,129 all):
  https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- Countdown Tasks 3-to-4 dataset card (490,364 rows; no declared license or test
  split in the card):
  https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
- TRL DeepMath-103K dataset used in the current GRPO documentation (no license
  declared in its dataset metadata at access time):
  https://huggingface.co/datasets/trl-lib/DeepMath-103K

### Function calling and tool-use training data

- API-Bank paper/code (EMNLP 2023; 73 executable evaluation tools, 314 evaluation
  dialogues, 753 calls, 1,888 source training dialogues):
  https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank
- API-Bank Hugging Face data mirror (MIT):
  https://huggingface.co/datasets/liminghao1630/API-Bank
- Hermes Function-Calling V1 (Apache-2.0; five configs):
  https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- ToolACE (Apache-2.0; 11,300 released dialogues; 26,507-API synthesis pool):
  https://huggingface.co/datasets/Team-ACE/ToolACE
- xLAM function-calling 60k (CC-BY-4.0; auto-gated):
  https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- xLAM/APIGen project repository:
  https://github.com/SalesforceAIResearch/xLAM
- APIGen-MT-5k (5,000 multi-turn trajectories; CC-BY-NC-4.0 and additional
  restriction stated in the data card):
  https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- ToolBench (Apache-2.0; 126,486 instances, 16,464 APIs, 469,585 real calls):
  https://github.com/OpenBMB/ToolBench

### Evaluation and stateful agent environments

- Berkeley Function-Calling Leaderboard (BFCL) implementation:
  https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- BFCL overview and V1-V4 categories:
  https://gorilla.cs.berkeley.edu/leaderboard.html
- Current tau3-bench repository (MIT; Gymnasium interface; airline, retail,
  telecom and banking-knowledge domains):
  https://github.com/sierra-research/tau2-bench
- tau3-bench reward-basis documentation, including the distinction between a
  reference action trace and end-state correctness:
  https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
- tau3-bench Gym interface:
  https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/gym/README.md

## Metadata snapshot used in the report

- MATH-lighteval: MIT, 7,500 train, 5,000 test, `problem/level/solution/type`.
- GSM8K main: MIT, 7,473 train, 1,319 test, `question/answer`.
- OpenR1-Math-220k: Apache-2.0; default 93,733 rows and about 4.96 GB
  materialized; all 225,129 rows and about 9.73 GB materialized.
- Countdown 3-to-4: 490,364 training rows with only `target` and `nums`; no
  declared license, solution trace, or official test split in the card.
- Hermes Function-Calling V1: 1,893 `func_calling_singleturn`, 1,893
  `func_calling`, 5,209 cleaned Glaive, 1,342 agentic JSON, and 1,241
  single-turn JSON rows (11,578 total files at access time).
- ToolACE: 11,300 released JSON dialogues; Apache-2.0.
- API-Bank transformed Hugging Face files: 6,184 level-1, 9,279 level-2, and
  1,245 level-3 prompt-completion training entries. These are decomposed entries,
  not the source-paper dialogue count.
- APIGen-MT: 5,000 multi-turn trajectories; retail and airline domains inherited
  from tau-bench; CC-BY-NC-4.0.

## Local compatibility observations

Environment:
`/raid/zkq/artifacts/CAPA/runtime/venv-qwen35-grpo-cu124-v1`

- Qwen3.5 native tokenizer template has no Jinja `{% generation %}` markers.
- Calling `apply_chat_template(..., return_assistant_tokens_mask=True)` directly
  on a five-message tool conversation produced 0 assistant-mask tokens out of 77.
- TRL 1.8 contains Qwen3.5 thinking and non-thinking *training* templates. Its
  `SFTTrainer` selects a training template when `assistant_only_loss=True`.
- Applying that training template to the same conversation produced 34 supervised
  assistant tokens out of 73, split over both assistant turns, and included
  `<|im_end|>` in each supervised span.
- `math_verify` and `latex2sympy2_extended` are not installed in the frozen CAPA
  Qwen3.5 environment; adding them must be version-locked and smoke-tested before
  modifying the environment.
