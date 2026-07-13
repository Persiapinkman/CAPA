from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from webbrowser import open_new_tab

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ace_rag.playbook.inspect import render_database_snapshot, write_html_report


def default_db_path() -> Path:
    configured = os.getenv("ACE_RAG_PLAYBOOK_DB_PATH")
    if not configured:
        return PROJECT_ROOT / "data" / "playbook.sqlite3"
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ACE Playbook SQLite HTML report.")
    parser.add_argument("--db", type=Path, default=default_db_path())
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "playbook_report.html")
    parser.add_argument(
        "--table",
        choices=["all", "playbook_items", "qa_runs", "qa_feedback", "playbook_operations", "playbook_state"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--full", action="store_true", help="Show all decoded columns for each row.")
    parser.add_argument("--open", action="store_true", help="Open the generated report in a browser.")
    parser.add_argument("--text", action="store_true", help="Print a terminal snapshot instead of writing HTML.")
    args = parser.parse_args()

    if args.text:
        print(render_database_snapshot(args.db, table=args.table, limit=args.limit, full=args.full))
        return

    snapshot = write_html_report(
        db_path=args.db,
        output_path=args.output,
        table=args.table,
        limit=args.limit,
        full=args.full,
    )
    output_path = args.output.expanduser().resolve()
    print(f"Wrote {output_path}")
    print(
        "Included "
        f"{len(snapshot['tables'])} tables, "
        f"{sum(table['displayed_count'] for table in snapshot['tables'])} displayed rows."
    )
    if args.open:
        open_new_tab(output_path.as_uri())


if __name__ == "__main__":
    main()
