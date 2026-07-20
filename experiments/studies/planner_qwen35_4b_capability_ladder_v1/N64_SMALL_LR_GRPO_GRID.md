# N64: small-learning-rate genuine GRPO grid

Starting from the stable n58 targeted-SFT checkpoint 6, three fresh-optimizer
GRPO runs changed only the learning rate (`5e-9`, `1e-8`, or `2e-8`). Every
run used the identical n59-supported 32-row optimizer set, seed 42, temperature
1.3, top-p 0.95, 32 sampled completions, and exactly one optimizer step.

All three runs were healthy: completion clipping was zero, reward standard
deviation was `0.196158`, advantage standard deviation was `0.432900`, and
gradient norm was `0.136726`. Each post-GRPO adapter was byte-distinct from the
initializer SHA-256 `088f1ef331e9e3787b5e4246cbed3750bd87182239d972b9b8898f5a88461123`.

| LR | Post-adapter SHA-256 | V14 metric | V14 current | V14 weighted (%) |
|---:|---|---:|---:|---:|
| 5e-9 | `7cf89112ce1ccc0ac39237717bfd6050bba274bbcf39057386f81b2ea530cea1` | 8/12 | 12/12 | 70.4000 |
| 1e-8 | `0c7446a35aba26131e76235780ad90a659b65e241513550491b33ed34537491a` | 11/12 | 12/12 | 92.6000 |
| 2e-8 | `a962c637746f21ab063677a8da0d1fe0db83fc4cbe5b305b896997e276f3b0c2` | 12/12 | 12/12 | 100.0000 |

The preregistered complete-cohort rule selected `2e-8`. Its one complete V13
retention run also passed metric 12/12 and current 12/12 (100.0000%). Its V13
prediction SHA-256 is
`29684c7a9a6a264ff118d7f6712155904db2162941be40626a15728750e088e2`;
its V14 prediction SHA-256 is
`c3d065ed83b7049e504ced475d29ba078097fa4e51a60c41f213dc63d7622d28`.

The candidate is not yet confirmation-ready: both 100% results are single
runs. It must next pass two additional complete runs on each opened development
cohort before a new entity- and lexicon-disjoint confirmation is generated.

Artifacts: `/raid/zkq/artifacts/CAPA/arbor/ladder_n64/grpo_lr_grid_20260720T0745Z`.
