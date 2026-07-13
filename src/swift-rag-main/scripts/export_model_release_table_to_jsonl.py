import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data_source" / "模型发版记录汇总.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_source" / "tables" / "model_release_records.jsonl"

SOURCE_COLUMN_MAPPING = {
    "目标名称": "target_name",
    "算法类型": "algorithm_type",
    "算法名称": "algorithm_name",
    "ones发版链接": "ones_release_link",
    "应用场景": "application_scene",
    "负责人(人员)": "owner",
    "模型名称": "model_name",
    "OID": "oid",
    "支持设备": "supported_device",
    "推荐配置": "recommended_config",
    "最近更新时间": "last_updated",
    "最近更新时间-提取年月": "last_updated_month",
    "计数": "count",
}

SEARCHABLE_KEYS = [
    "target_name",
    "algorithm_type",
    "algorithm_name",
    "application_scene",
    "owner",
    "model_name",
    "supported_device",
    "recommended_config",
    "last_updated",
    "last_updated_month",
]


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass

    text = str(value).strip()
    return text or None


def _normalize_last_updated(raw_value: Any) -> str | None:
    if pd.isna(raw_value):
        return None
    parsed = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(parsed):
        text = str(raw_value).strip()
        return text or None
    return parsed.date().isoformat()


def _normalize_last_updated_month(raw_value: Any) -> str | None:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        text = text.replace("年", "-").replace("月", "").replace("/", "-")
        parts = [part for part in text.split("-") if part]
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1].zfill(2)}"
        return text
    return parsed.strftime("%Y-%m")


def _build_search_text(record: dict[str, Any]) -> str:
    parts = []
    for key in SEARCHABLE_KEYS:
        value = record.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def export_excel_to_jsonl(input_path: Path, output_path: Path) -> int:
    workbook = pd.ExcelFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df = df.rename(columns=SOURCE_COLUMN_MAPPING)

            for row_offset, row in enumerate(df.to_dict(orient="records"), start=2):
                record = {key: _normalize_value(value) for key, value in row.items()}
                record["last_updated"] = _normalize_last_updated(row.get("last_updated"))
                record["last_updated_month"] = _normalize_last_updated_month(
                    row.get("last_updated_month")
                )
                record["row_id"] = f"{sheet_name}-{row_offset:04d}"
                record["source_file"] = str(input_path.relative_to(PROJECT_ROOT))
                record["sheet_name"] = sheet_name
                record["source_row_number"] = row_offset
                record["search_text"] = _build_search_text(record)

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_rows += 1

    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="将模型发版记录 Excel 导出为规范化 JSONL")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 Excel 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSONL 路径")
    args = parser.parse_args()

    row_count = export_excel_to_jsonl(args.input.resolve(), args.output.resolve())
    print(f"exported {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
