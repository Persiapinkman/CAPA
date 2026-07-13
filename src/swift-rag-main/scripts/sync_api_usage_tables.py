#!/usr/bin/env python3
"""Sync parameter tables in docs/API_USAGE.md from OpenAPI schema.

Usage:
  python scripts/sync_api_usage_tables.py \
    --openapi-url http://127.0.0.1:6060/openapi.json \
    --doc docs/API_USAGE.md

You can also use --openapi-file /path/to/openapi.json.

The script updates content between markers:
  <!-- AUTO_TABLES: <path> START -->
  ...
  <!-- AUTO_TABLES: <path> END -->
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

TARGET_PATHS = [
    "/api/v1/rag/doc_engine/chunking_embedding",
    "/api/v1/rag/chat_engine/query",
    "/api/v1/rag/chat_engine/table_query",
    "/api/v1/rag/chat_engine/adela_query",
    "/api/v1/rag/chat_engine/unified_retrieve",
    "/api/v1/rag/chat_engine/unified_query",
    "/api/v1/rag/embedding",
]


def load_openapi(url: str | None, file_path: str | None) -> Dict[str, Any]:
    if file_path:
        return json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not url:
        raise ValueError("Either --openapi-url or --openapi-file must be provided")
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_ref(schema: Dict[str, Any], openapi: Dict[str, Any]) -> Dict[str, Any]:
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/components/schemas/"):
            return schema
        name = ref.split("/")[-1]
        schema = openapi.get("components", {}).get("schemas", {}).get(name, {})
    return schema


def format_default(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, str):
        # Compress long path-like defaults to avoid oversized columns.
        shown = value
        if len(shown) > 60 and "/" in shown:
            parts = [p for p in shown.split("/") if p]
            if len(parts) >= 2:
                shown = f".../{parts[-2]}/{parts[-1]}"
            else:
                shown = f".../{shown[-40:]}"
        elif len(shown) > 80:
            shown = shown[:77] + "..."
        return f"`{shown}`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (list, dict)):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    return f"`{value}`"


def schema_type(schema: Dict[str, Any], openapi: Dict[str, Any]) -> str:
    schema = resolve_ref(schema, openapi)

    if "anyOf" in schema:
        parts = []
        for part in schema["anyOf"]:
            part = resolve_ref(part, openapi)
            if part.get("type") == "null":
                parts.append("null")
            else:
                parts.append(schema_type(part, openapi))
        dedup = []
        for p in parts:
            if p not in dedup:
                dedup.append(p)
        return " | ".join(dedup) if dedup else "any"

    if "oneOf" in schema:
        parts = [schema_type(resolve_ref(part, openapi), openapi) for part in schema["oneOf"]]
        dedup = []
        for p in parts:
            if p not in dedup:
                dedup.append(p)
        return " | ".join(dedup) if dedup else "any"

    t = schema.get("type")
    if t == "array":
        item = resolve_ref(schema.get("items", {}), openapi)
        return f"{schema_type(item, openapi)}[]"
    if t == "object":
        return "object"
    if t:
        return t
    if "properties" in schema:
        return "object"
    return "any"


def build_desc(schema: Dict[str, Any], openapi: Dict[str, Any]) -> str:
    schema = resolve_ref(schema, openapi)
    desc = (schema.get("description") or "").strip()

    def collect_enum_values(s: Dict[str, Any]) -> List[Any]:
        s = resolve_ref(s, openapi)
        vals: List[Any] = []
        if s.get("enum"):
            vals.extend(s.get("enum") or [])
        if s.get("type") == "array":
            items = resolve_ref(s.get("items", {}), openapi)
            if items.get("enum"):
                vals.extend(items.get("enum") or [])
        for k in ("anyOf", "oneOf", "allOf"):
            for part in s.get(k, []) or []:
                vals.extend(collect_enum_values(part))
        return vals

    enum_vals = collect_enum_values(schema)
    # de-duplicate while preserving order
    seen = set()
    enum_vals = [v for v in enum_vals if not (v in seen or seen.add(v))]
    enum_text = ""
    if enum_vals:
        enum_text = "枚举: " + " / ".join(f"`{v}`" for v in enum_vals)
    if desc and enum_text:
        return f"{desc}；{enum_text}"
    if enum_text:
        return enum_text
    return desc or "-"


def md_cell(text: str) -> str:
    """Escape markdown table-breaking characters and normalize whitespace."""
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def flatten_object_for_input(
    schema: Dict[str, Any],
    openapi: Dict[str, Any],
    prefix: str = "",
    parent_required: List[str] | None = None,
) -> List[Tuple[str, str, str, str, str]]:
    schema = resolve_ref(schema, openapi)
    if parent_required is None:
        parent_required = schema.get("required", []) or []

    rows: List[Tuple[str, str, str, str, str]] = []
    props = schema.get("properties", {})

    for name, prop_schema_raw in props.items():
        prop_schema = resolve_ref(prop_schema_raw, openapi)
        key = f"{prefix}.{name}" if prefix else name
        t = schema_type(prop_schema, openapi)
        required = "是" if name in parent_required else "否"
        default = format_default(prop_schema["default"]) if "default" in prop_schema else "-"
        desc = build_desc(prop_schema, openapi)
        rows.append((f"`{key}`", f"`{t}`", required, default, desc))

        if prop_schema.get("type") == "object" or "properties" in prop_schema or "$ref" in prop_schema_raw:
            sub = resolve_ref(prop_schema_raw, openapi)
            if sub.get("properties"):
                rows.extend(
                    flatten_object_for_input(
                        sub,
                        openapi,
                        prefix=key,
                        parent_required=sub.get("required", []) or [],
                    )
                )

        if prop_schema.get("type") == "array":
            items = resolve_ref(prop_schema.get("items", {}), openapi)
            if items.get("type") == "object" or items.get("properties"):
                rows.extend(
                    flatten_object_for_input(
                        items,
                        openapi,
                        prefix=f"{key}[]",
                        parent_required=items.get("required", []) or [],
                    )
                )

    return rows


def flatten_object_for_output(
    schema: Dict[str, Any],
    openapi: Dict[str, Any],
    prefix: str = "",
) -> List[Tuple[str, str, str]]:
    schema = resolve_ref(schema, openapi)
    rows: List[Tuple[str, str, str]] = []
    props = schema.get("properties", {})

    for name, prop_schema_raw in props.items():
        prop_schema = resolve_ref(prop_schema_raw, openapi)
        key = f"{prefix}.{name}" if prefix else name
        t = schema_type(prop_schema, openapi)
        desc = build_desc(prop_schema, openapi)
        rows.append((f"`{key}`", f"`{t}`", desc))

        if prop_schema.get("type") == "object" or "properties" in prop_schema or "$ref" in prop_schema_raw:
            sub = resolve_ref(prop_schema_raw, openapi)
            if sub.get("properties"):
                rows.extend(flatten_object_for_output(sub, openapi, prefix=key))

        if prop_schema.get("type") == "array":
            items = resolve_ref(prop_schema.get("items", {}), openapi)
            if items.get("type") == "object" or items.get("properties"):
                rows.extend(flatten_object_for_output(items, openapi, prefix=f"{key}[]"))

    return rows


def render_input_table(rows: List[Tuple[str, str, str, str, str]]) -> str:
    dedup_rows: List[Tuple[str, str, str, str, str]] = []
    seen = set()
    for row in rows:
        if row in seen:
            continue
        dedup_rows.append(row)
        seen.add(row)

    lines = [
        "**输入参数**",
        "",
        "| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if not dedup_rows:
        lines.append("| - | - | - | - | OpenAPI 未定义请求体 |")
    else:
        for row in dedup_rows:
            lines.append(
                f"| {md_cell(row[0])} | {md_cell(row[1])} | {md_cell(row[2])} | {md_cell(row[3])} | {md_cell(row[4])} |"
            )
    return "\n".join(lines)


def render_output_table(rows: List[Tuple[str, str, str]], stream_note: bool = False) -> str:
    dedup_rows: List[Tuple[str, str, str]] = []
    seen = set()
    for row in rows:
        if row in seen:
            continue
        dedup_rows.append(row)
        seen.add(row)

    lines = [
        "**输出参数**",
        "",
        "| 参数 | 类型 | 说明 |",
        "| --- | --- | --- |",
    ]
    if stream_note:
        lines.append("| - | - | OpenAPI 未定义结构化响应体（该接口为 `StreamingResponse`，SSE 文本流） |")
    elif not dedup_rows:
        lines.append("| - | - | OpenAPI 未定义响应体 |")
    else:
        for row in dedup_rows:
            lines.append(f"| {md_cell(row[0])} | {md_cell(row[1])} | {md_cell(row[2])} |")
    return "\n".join(lines)


def generate_tables_for_path(openapi: Dict[str, Any], path: str) -> str:
    op = openapi.get("paths", {}).get(path, {}).get("post")
    if not op:
        return "**输入参数**\n\n| 参数 | 类型 | 必填 | 默认值 | 说明/限制 |\n| --- | --- | --- | --- | --- |\n| - | - | - | - | OpenAPI 中未找到该接口 |\n\n**输出参数**\n\n| 参数 | 类型 | 说明 |\n| --- | --- | --- |\n| - | - | OpenAPI 中未找到该接口 |"

    req_schema = (
        op.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema")
    )
    resp_schema = (
        op.get("responses", {})
        .get("200", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema")
    )

    input_rows: List[Tuple[str, str, str, str, str]] = []
    if req_schema:
        resolved_req = resolve_ref(req_schema, openapi)
        input_rows = flatten_object_for_input(
            resolved_req,
            openapi,
            prefix="",
            parent_required=resolved_req.get("required", []) or [],
        )

    output_rows: List[Tuple[str, str, str]] = []
    stream_note = False
    if resp_schema:
        resolved_resp = resolve_ref(resp_schema, openapi)
        output_rows = flatten_object_for_output(resolved_resp, openapi)
    else:
        stream_note = False

    return (
        render_input_table(input_rows)
        + "\n\n"
        + render_output_table(output_rows, stream_note=stream_note)
    )


def replace_block(doc_text: str, path: str, new_block: str) -> str:
    start = f"<!-- AUTO_TABLES: {path} START -->"
    end = f"<!-- AUTO_TABLES: {path} END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    replacement = f"{start}\n{new_block}\n{end}"
    if not pattern.search(doc_text):
        raise ValueError(f"Marker not found for path: {path}")
    return pattern.sub(replacement, doc_text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openapi-url", default=None)
    parser.add_argument("--openapi-file", default=None)
    parser.add_argument("--doc", default="docs/API_USAGE.md")
    args = parser.parse_args()

    openapi = load_openapi(args.openapi_url, args.openapi_file)
    doc_path = Path(args.doc)
    text = doc_path.read_text(encoding="utf-8")

    for path in TARGET_PATHS:
        block = generate_tables_for_path(openapi, path)
        text = replace_block(text, path, block)

    doc_path.write_text(text, encoding="utf-8")
    print(f"Synced tables in {doc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
