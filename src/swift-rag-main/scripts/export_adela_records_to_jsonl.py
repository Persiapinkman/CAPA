import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.adela_dataset import export_adela_records_to_jsonl


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data_source" / "adela" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_source" / "adela" / "adela_release_records.jsonl"
DEFAULT_RECORDS_CSV = PROJECT_ROOT / "data_source" / "adela" / "adela_release_records.csv"
DEFAULT_SEARCHABLE_FIELDS = [
    "model_name",
    "name",
    "label_list",
    "labels",
    "type",
    "platform",
    "status",
    "did",
    "rid",
    "version",
    "version_train_date",
    "source_file",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="将 adela 部署记录 JSON 导出为规范化 JSONL")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="adela 原始 JSON 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=DEFAULT_RECORDS_CSV,
        help="adela 发布记录 CSV 路径（包含 did/rid/model_name/platform/label_list）",
    )
    args = parser.parse_args()

    row_count = export_adela_records_to_jsonl(
        input_dir=args.input_dir,
        output_path=args.output,
        searchable_fields=DEFAULT_SEARCHABLE_FIELDS,
        records_csv_path=args.records_csv,
    )
    print(f"exported {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
