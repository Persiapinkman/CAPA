#!/usr/bin/env python3
"""Apply the preregistered V10 primary-plus-control support gate."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.planner_grpo_seed_v1.scripts.gate_planner_retry_safe_end_residual_v8_support import (  # noqa: F401,E402
    aggregate,
    apply_gate,
    check,
    load_json,
    load_jsonl,
    main,
)


if __name__ == "__main__":
    main()
