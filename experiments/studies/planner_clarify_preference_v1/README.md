# Planned Clarify Preference Study

The completed ChatML/SFT/GRPO control found no supported post-training gain and all four arms scored `0.1` on the two clarify development cases. Online GRPO sampling therefore lacks positive support for the desired `decision_type="clarify"` action.

The next training study will use diverse preference pairs where the chosen response asks one decisive clarification question and rejected responses represent each plausible wrong route, including generation, detection, RAG, pipeline evaluation, and migration advice.

Training must not start until a new template/entity-separated dataset and sealed test split pass the dataset audit. DPO is the primary candidate; PPO and another GRPO extension are out of scope for this study.
