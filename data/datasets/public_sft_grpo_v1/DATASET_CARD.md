# Public SFT/GRPO Smoke Dataset V1

This dataset registration freezes the public sources used to validate CAPA's
SFT -> GRPO engineering loop before returning to planner tool-routing data.

## Frozen sources

| Source | Hub revision | License | Local artifact |
|---|---|---|---|
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | MIT | `/raid/zkq/artifacts/CAPA/datasets/public_sft_grpo_v1/raw/hf/openai__gsm8k/740312add88f781978c0658806c59bc2815b9866` |
| `DigitalLearningGmbH/MATH-lighteval` | `0530c78699ea5e8eb5530600900e1f328b48acad` | MIT | `/raid/zkq/artifacts/CAPA/datasets/public_sft_grpo_v1/raw/hf/DigitalLearningGmbH__MATH-lighteval/0530c78699ea5e8eb5530600900e1f328b48acad` |

The upstream test splits are excluded from SFT and GRPO training. GSM8K's
official train split is deterministically partitioned for the 32-case overfit
smoke. The official test subset is materialized only as a sealed evaluation
artifact and is not used for checkpoint selection.

## Initial experiment

- Model: `/raid/zkq/models/Qwen3.5-4B`
- Template: native Qwen3.5 non-thinking behavior
- SFT loss: assistant tokens only, including `<|im_end|>`
- Packing: disabled
- First gate: 32 train / 32 development GSM8K rows
- Verifier: strict final `#### integer` plus a separate loose-numeric diagnostic

Generated files and their hashes are recorded under
`training/public_sft_grpo_v1/data/gsm8k_sft32_v1/manifest.json`.

## MATH SFT screen

The second gate uses a deterministic, proportional `level x type` stratification:

- 1,024 training and 256 development rows from the official train split;
- 512 sealed rows from the official test split;
- all 35 `Level 1-5 x subject` strata represented in every derived split;
- the one known upstream train/test problem overlap and one normalized train duplicate removed;
- multi-box, empty-box, unparseable strict-gold, problem-box leakage, and sequences over 2,048 tokens removed;
- every assistant target normalized to exactly one terminal `\\boxed{answer}` line and supervised EOS.

Math equivalence uses `math-verify==0.9.0`, `latex2sympy2-extended==1.11.0`, and
`antlr4-python3-runtime==4.13.2` in the isolated
`/raid/zkq/artifacts/CAPA/runtime/venv-qwen35-math-cu124-v1` environment.
Generated files and hashes are recorded under
`training/public_sft_grpo_v1/data/math_sft1024_v1/manifest.json`.
