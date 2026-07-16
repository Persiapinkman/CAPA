"""Machine-readable capability inventory for the CAPA demo agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tools import registry as tool_registry


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    tool_name: str
    runtime_owner: str
    components: tuple[str, ...]
    services: tuple[str, ...]
    side_effecting: bool = False


COMPONENT_ASSETS: dict[str, tuple[str, ...]] = {
    "planner": ("src/capa/agent.py", "src/capa/prompts.py"),
    "answerer": ("src/capa/agent.py",),
    "query-rewrite": ("src/capa/agent.py",),
    "rag-retrieve-answer": (
        "skills/rag-retrieve-answer/SKILL.md",
        "skills/rag-retrieve-answer/scripts/run_rag.py",
        "demo/gbrain_rag_client.py",
    ),
    "user-intent-understanding": (
        "skills/user-intent-understanding/SKILL.md",
        "skills/user-intent-understanding/scripts/run_intent.py",
    ),
    "llm-prompts-generation": (
        "skills/llm-prompts-generation/SKILL.md",
        "skills/llm-prompts-generation/scripts/run_prompt_generation.py",
    ),
    "flux-image-generation": (
        "skills/flux-image-generation/SKILL.md",
        "skills/flux-image-generation/scripts/run_generation.py",
    ),
    "qwen-vlm-open-set-delection": (
        "skills/qwen-vlm-open-set-delection/SKILL.md",
        "skills/qwen-vlm-open-set-delection/scripts/run_detection.py",
    ),
    "rexomni-open-set-detection": (
        "skills/rexomni-open-set-detection/SKILL.md",
        "skills/rexomni-open-set-detection/scripts/run_detection.py",
    ),
    "eval-reports-generation": (
        "skills/eval-reports-generation/SKILL.md",
        "skills/eval-reports-generation/scripts/run_eval_report_generation.py",
    ),
    "target-detection-evaluation": (
        "skills/target-detection-evaluation/SKILL.md",
        "skills/target-detection-evaluation/scripts/run_pipeline.py",
    ),
    "migration-advisor": ("demo/migration_advisor.py",),
    "adela-cli": (
        "skills/adela-cli/README.md",
        "skills/adela-cli/scripts/run_pipeline.py",
    ),
}


CAPABILITY_SPECS: dict[str, CapabilitySpec] = {
    tool_registry.TOOL_RAG_ANSWER: CapabilitySpec(
        tool_name=tool_registry.TOOL_RAG_ANSWER,
        runtime_owner="executor",
        components=("rag-retrieve-answer",),
        services=("rag",),
    ),
    tool_registry.TOOL_RE_QUESTION: CapabilitySpec(
        tool_name=tool_registry.TOOL_RE_QUESTION,
        runtime_owner="executor",
        components=("query-rewrite",),
        services=("model_gateway",),
    ),
    tool_registry.TOOL_ANSWERER: CapabilitySpec(
        tool_name=tool_registry.TOOL_ANSWERER,
        runtime_owner="orchestrator",
        components=("answerer",),
        services=("model_gateway",),
    ),
    tool_registry.TOOL_FLUX_IMAGE_GENERATION: CapabilitySpec(
        tool_name=tool_registry.TOOL_FLUX_IMAGE_GENERATION,
        runtime_owner="executor",
        components=(
            "user-intent-understanding",
            "llm-prompts-generation",
            "flux-image-generation",
        ),
        services=("model_gateway", "flux_api"),
        side_effecting=True,
    ),
    tool_registry.TOOL_QWEN_DETECTION: CapabilitySpec(
        tool_name=tool_registry.TOOL_QWEN_DETECTION,
        runtime_owner="executor",
        components=("qwen-vlm-open-set-delection",),
        services=("qwen_detection",),
    ),
    tool_registry.TOOL_REXOMNI_DETECTION: CapabilitySpec(
        tool_name=tool_registry.TOOL_REXOMNI_DETECTION,
        runtime_owner="executor",
        components=("rexomni-open-set-detection",),
        services=("model_gateway",),
    ),
    tool_registry.TOOL_PIPELINE_EVAL: CapabilitySpec(
        tool_name=tool_registry.TOOL_PIPELINE_EVAL,
        runtime_owner="executor",
        components=(
            "target-detection-evaluation",
            "user-intent-understanding",
            "llm-prompts-generation",
            "flux-image-generation",
            "qwen-vlm-open-set-delection",
            "rexomni-open-set-detection",
            "eval-reports-generation",
        ),
        services=("model_gateway", "flux_api", "qwen_detection"),
        side_effecting=True,
    ),
    tool_registry.TOOL_MIGRATION_ADVISOR: CapabilitySpec(
        tool_name=tool_registry.TOOL_MIGRATION_ADVISOR,
        runtime_owner="executor",
        components=(
            "migration-advisor",
            "rag-retrieve-answer",
            "rexomni-open-set-detection",
        ),
        services=("rag", "model_gateway"),
    ),
    tool_registry.TOOL_ADELA_CLI_EVAL: CapabilitySpec(
        tool_name=tool_registry.TOOL_ADELA_CLI_EVAL,
        runtime_owner="executor",
        components=("adela-cli", "rag-retrieve-answer"),
        services=("rag", "adela_cli"),
        side_effecting=True,
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_capability_inventory(root: Path | None = None) -> list[dict[str, Any]]:
    workspace = Path(root or repository_root()).resolve()
    alias_map = tool_registry.get_action_aliases()
    inventory: list[dict[str, Any]] = []
    for declaration in tool_registry.get_active_tool_declarations():
        tool_name = declaration["name"]
        spec = CAPABILITY_SPECS[tool_name]
        assets = {
            component: [
                {
                    "path": relative_path,
                    "exists": (workspace / relative_path).is_file(),
                }
                for relative_path in COMPONENT_ASSETS[component]
            ]
            for component in spec.components
        }
        item = asdict(spec)
        item.update(
            {
                "executor_branch": declaration["executor_branch"],
                "flow": declaration["flow"],
                "requires_image": bool(declaration["requires_image"]),
                "aliases": sorted(
                    alias
                    for alias, canonical in alias_map.items()
                    if canonical == tool_name and alias != tool_name
                ),
                "assets": assets,
            }
        )
        inventory.append(item)
    return inventory


def validate_capability_inventory(root: Path | None = None) -> list[str]:
    declared = set(tool_registry.get_all_tool_names())
    specified = set(CAPABILITY_SPECS)
    errors: list[str] = []
    for missing in sorted(declared - specified):
        errors.append(f"missing capability spec: {missing}")
    for extra in sorted(specified - declared):
        errors.append(f"capability spec has no declared tool: {extra}")
    if errors:
        return errors
    for item in build_capability_inventory(root):
        for component, assets in item["assets"].items():
            for asset in assets:
                if not asset["exists"]:
                    errors.append(
                        f"missing asset for {item['tool_name']}/{component}: {asset['path']}"
                    )
    return errors
