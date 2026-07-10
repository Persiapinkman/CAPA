# Qwen2.5 Baseline vs SFT v1 on SFT Val

| Model | Mean Score | JSON Valid | Extra Text Rate | Mean Extra Chars |
|---|---:|---:|---:|---:|
| Baseline | 0.6407 | 1.000 | 1.000 | 160.96 |
| SFT v1 | 0.6173 | 1.000 | 1.000 | 157.14 |

Mean score delta: -0.0234.

SFT v1 is not an improvement overall; it keeps JSON validity at 1.0 but still emits extra text after the first JSON for every sample.
