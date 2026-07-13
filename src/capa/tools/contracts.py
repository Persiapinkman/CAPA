from __future__ import annotations

"""
Typed contracts for tool orchestration.

主要功能：
- 定义工具层跨模块的数据契约，提升可读性与类型安全。
- 统一 Planner -> Executor -> Orchestrator 的输入输出结构。

主要模块：
- `ToolCall`：Planner 产出的动作对象（含 action/action_input/thought/final_answer）。
- `ToolExecutionContext`：工具执行所需上下文（文本、图像、运行目录、凭证等）。
- `ToolResult`：执行结果对象（action、observation、ok、error_message）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolCall:
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    final_answer: str = ""

    @classmethod
    def from_step(cls, step: dict) -> "ToolCall":
        data = step if isinstance(step, dict) else {}
        action_input = data.get("action_input")
        if not isinstance(action_input, dict):
            action_input = {}
        return cls(
            action=str(data.get("action") or "").strip(),
            action_input=action_input,
            thought=str(data.get("thought") or "").strip(),
            final_answer=str(data.get("final_answer") or "").strip(),
        )


@dataclass(slots=True)
class ToolExecutionContext:
    text: str
    image_path: str
    api_key: str
    api_base: str
    run_dir: Path
    run_stamp: str
    image_paths: list[str] = field(default_factory=list)
    session_id: str = ""
    session: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    action: str
    observation: dict[str, Any]
    ok: bool
    error_message: str = ""
    requires_clarification: bool = False
