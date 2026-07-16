#!/usr/bin/env python3
"""Create the versioned CAPA post-training W&B workspace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/wandb/post_training_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--workspace-name", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the workspace without credentials or network writes.",
    )
    return parser.parse_args()


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "1.0":
        raise ValueError(f"unsupported dashboard schema: {spec.get('schema_version')}")
    sections = spec.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("dashboard requires at least one section")
    for section in sections:
        if not section.get("name") or not section.get("panels"):
            raise ValueError(f"invalid dashboard section: {section}")
        for panel in section["panels"]:
            if panel.get("type") != "line" or not panel.get("title") or not panel.get("y"):
                raise ValueError(f"invalid dashboard panel: {panel}")
    return spec


def create_workspace(spec: dict[str, Any], *, entity: str, project: str, name: str) -> Any:
    if not entity:
        raise ValueError("--entity or WANDB_ENTITY is required for a live workspace")
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise RuntimeError(
            "Install the observability extra first: uv sync --extra observability"
        ) from exc

    x_axis = str(spec["x_axis"])
    sections = []
    for section in spec["sections"]:
        panels = [
            wr.LinePlot(title=panel["title"], x=x_axis, y=list(panel["y"]))
            for panel in section["panels"]
        ]
        sections.append(
            ws.Section(
                name=section["name"],
                panels=panels,
                is_open=bool(section.get("is_open", True)),
            )
        )
    return ws.Workspace(name=name, entity=entity, project=project, sections=sections).save()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    spec = load_spec(config_path)
    project = args.project or str(spec["project"])
    name = args.workspace_name or str(spec["workspace_name"])
    resolved = {**spec, "entity": args.entity, "project": project, "workspace_name": name}
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return
    workspace = create_workspace(resolved, entity=args.entity, project=project, name=name)
    print(
        json.dumps(
            {
                "status": "saved",
                "entity": args.entity,
                "project": project,
                "workspace_name": name,
                "url": str(getattr(workspace, "url", "")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
