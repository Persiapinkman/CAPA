import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from reward_planner_grpo import load_jsonl, score_case

cases = {c["case_id"]: c for c in load_jsonl(
    Path("training/planner_grpo_seed_v1/cases/planner_grpo_train_cases.jsonl"))}


def pick(p):
    return next(cid for cid in cases if cid.startswith(p))


D = "decision_type"
tests = [
    ("A组 探针后过早收口", pick("GRPO-EXP-PROBE-MIG-STRICT"),
     [{D: "tool", "action": "qwen_detection", "action_input": {"label": "钓鱼", "finish_after_tool": True}},
      {D: "end", "end_reason": "memory_hit"}]),
    ("B组 纯探针却误转migration", pick("GRPO-EXP-PROBE-ONLY"),
     [{D: "tool", "action": "qwen_detection", "action_input": {"label": "钓鱼", "finish_after_tool": False}},
      {D: "tool", "action": "migration_advisor", "action_input": {"finish_after_tool": True}}]),
    ("clarify 该问不问直接检测", pick("GRPO-EXP-CLARIFY"),
     [{D: "tool", "action": "qwen_detection", "action_input": {"label": "黑猫", "finish_after_tool": True}}]),
]
for desc, cid, dec in tests:
    r = score_case(cases[cid], {"case_id": cid, "decisions": dec})
    print(f"{desc:28s} score={r['score']:.3f} passed={r['passed']} "
          f"premature={r['premature_stop_hit']} forbidden={r['forbidden_hit']}")
