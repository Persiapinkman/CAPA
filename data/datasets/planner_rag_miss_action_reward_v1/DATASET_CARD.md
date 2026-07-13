# Dataset Card: planner_rag_miss_action_reward_v1

## Purpose

This training view keeps the RAG-miss state-machine prompts and sampling weights fixed, but replaces a dense verifier that rewarded wrong actions with an action-dominant reward.

## Why A New Reward

Under the previous reward, sampled verifier mean increased from `0.8325` to `0.8400` while strict actions decreased from `147/192` to `146/192`; greedy output was unchanged. GRPO optimized argument and formatting partial credit instead of the route.

The new task reward is:

| Component | Weight |
|---|---:|
| Action match | 0.75 |
| Argument match | 0.10 |
| Finish flag | 0.05 |
| Forbidden-action avoidance | 0.05 |
| Decision type | 0.03 |
| Valid JSON | 0.02 |

Any wrong action is capped at task reward `0.20`. The outer reward uses task `0.95` and format `0.05`, so a wrong action cannot exceed total reward `0.24`.

## Counterfactual Validation

Rescoring the existing SFT development samples produced:

- 147 correct actions: mean task reward `0.9789`, minimum `0.925`.
- 45 wrong actions: mean task reward `0.1571`, maximum `0.20`.
- Minimum worst-case total-reward separation: `0.6388`.

## Composition

- 240 weighted rows from 168 unique source steps and 24 training entities.
- Three RAG-miss states each have 48 rows.
- Coreference, direct RAG, memory end, and general answer each have 24 contrast rows.
- Evaluation uses the original unweighted development and sealed-test splits.

## Human Review

Read `HUMAN_REVIEW.md`. Review prompt labels using the source state-machine data card, then separately review the reward contract. Sampling replicas are weights, not independent cases.

## Claim Boundary

A positive result would demonstrate that action-aligned GRPO can improve this mocked RAG-miss state machine. It would not show that arbitrary dense GRPO rewards or production RAG improve.
