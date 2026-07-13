from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

DEFAULT_TABLE_ORDER = [
    "playbook_items",
    "qa_runs",
    "qa_feedback",
    "playbook_operations",
    "playbook_state",
]

SUMMARY_COLUMNS: dict[str, list[str]] = {
    "playbook_items": [
        "item_id",
        "section",
        "status",
        "confidence",
        "helpful_count",
        "harmful_count",
        "tags_json",
        "source_hints_json",
        "query_intents_json",
        "expansion_terms_json",
        "content",
        "created_at",
        "updated_at",
    ],
    "qa_runs": [
        "run_id",
        "query",
        "answer",
        "playbook_item_ids_json",
        "timings_json",
        "created_at",
    ],
    "qa_feedback": [
        "feedback_id",
        "run_id",
        "feedback_type",
        "rating",
        "status",
        "comment",
        "expected_evidence_ids_json",
        "created_at",
    ],
    "playbook_operations": [
        "op_id",
        "feedback_id",
        "operation_type",
        "target_item_id",
        "status",
        "payload_json",
        "created_at",
        "applied_at",
    ],
    "playbook_state": [
        "key",
        "value_json",
        "updated_at",
    ],
}

SUMMARY_MAX_STRING = 240
TIMESTAMP_COLUMNS = {"created_at", "updated_at", "applied_at"}
REPORT_MAX_CELL_STRING = 180


def open_database(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    names = [row["name"] if isinstance(row, sqlite3.Row) else row[0] for row in rows]
    ordered = [name for name in DEFAULT_TABLE_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in DEFAULT_TABLE_ORDER))
    return ordered


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) AS count FROM {quote_identifier(table)}"
    ).fetchone()
    return int(row["count"] if row else 0)


def _decode_value(column: str, value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value

    if isinstance(value, str):
        stripped = value.strip()
        if column.endswith("_json") or (stripped[:1] in {"{", "["} and stripped[-1:] in {"}", "]"}):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value


def decode_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: _decode_value(key, row[key]) for key in row.keys()}


def fetch_rows(
    conn: sqlite3.Connection,
    table: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    order_by = {
        "playbook_items": "section, item_id",
        "qa_runs": "created_at DESC",
        "qa_feedback": "created_at DESC",
        "playbook_operations": "created_at DESC",
        "playbook_state": "updated_at DESC",
    }.get(table)

    sql = f"SELECT * FROM {quote_identifier(table)}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    sql += " LIMIT ? OFFSET ?"

    rows = conn.execute(sql, (limit, offset)).fetchall()
    return [decode_row(row) for row in rows]


def _format_timestamp(value: Any) -> Any:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return value
    return value


def _summarize_value(column: str, value: Any) -> Any:
    if column in TIMESTAMP_COLUMNS:
        return _format_timestamp(value)
    if isinstance(value, str) and len(value) > SUMMARY_MAX_STRING:
        return value[:SUMMARY_MAX_STRING] + "..."
    return value


def select_row_columns(table: str, row: dict[str, Any], full: bool = False) -> dict[str, Any]:
    if full:
        columns = list(row.keys())
    else:
        columns = SUMMARY_COLUMNS.get(table, list(row.keys()))

    selected: dict[str, Any] = {}
    for column in columns:
        if column in row:
            selected[column] = _summarize_value(column, row[column])
    return selected


def render_row(table: str, row: dict[str, Any], full: bool = False) -> str:
    selected = row if full else select_row_columns(table, row, full=False)
    return json.dumps(selected, ensure_ascii=False, indent=2)


def render_table_section(
    conn: sqlite3.Connection,
    table: str,
    limit: int = 20,
    full: bool = False,
) -> str:
    count = count_rows(conn, table)
    rows = fetch_rows(conn, table, limit=limit)

    lines = [f"{table} ({count} rows)"]
    if count > limit:
        lines.append(f"  showing first {limit} rows")

    if not rows:
        lines.append("  <empty>")
        return "\n".join(lines)

    for index, row in enumerate(rows, start=1):
        lines.append(f"  [{index}]")
        rendered = render_row(table, row, full=full)
        for line in rendered.splitlines():
            lines.append(f"    {line}")
    return "\n".join(lines)


def render_database_snapshot(
    db_path: Path,
    table: str = "all",
    limit: int = 20,
    full: bool = False,
) -> str:
    conn = open_database(db_path)
    try:
        tables = list_tables(conn)
        if table != "all":
            if table not in tables:
                raise ValueError(f"Table not found in database: {table}")
            tables = [table]

        lines = [f"database: {Path(db_path).expanduser()}"]
        lines.append("tables: " + ", ".join(f"{name}={count_rows(conn, name)}" for name in tables))
        for name in tables:
            lines.append("")
            lines.append(render_table_section(conn, name, limit=limit, full=full))
        return "\n".join(lines)
    finally:
        conn.close()


def build_database_snapshot(
    db_path: Path,
    table: str = "all",
    limit: int = 100,
    full: bool = False,
) -> dict[str, Any]:
    conn = open_database(db_path)
    try:
        tables = list_tables(conn)
        if table != "all":
            if table not in tables:
                raise ValueError(f"Table not found in database: {table}")
            tables = [table]

        table_snapshots: list[dict[str, Any]] = []
        for name in tables:
            rows = fetch_rows(conn, name, limit=limit)
            display_rows = [select_row_columns(name, row, full=full) for row in rows]
            columns = list(display_rows[0].keys()) if display_rows else SUMMARY_COLUMNS.get(name, [])
            table_snapshots.append(
                {
                    "name": name,
                    "count": count_rows(conn, name),
                    "displayed_count": len(rows),
                    "columns": columns,
                    "rows": rows,
                    "display_rows": display_rows,
                }
            )

        return {
            "database": str(Path(db_path).expanduser()),
            "generated_at": _format_timestamp(time.time()),
            "limit": limit,
            "full": full,
            "tables": table_snapshots,
        }
    finally:
        conn.close()


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _short_text(value: Any, limit: int = REPORT_MAX_CELL_STRING) -> str:
    text = _value_to_text(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _render_cell(value: Any) -> str:
    if value is None:
        return '<span class="muted">null</span>'
    if isinstance(value, list):
        if not value:
            return '<span class="muted">[]</span>'
        chips = []
        for item in value[:12]:
            chips.append(f'<span class="chip">{escape(_short_text(item, 40))}</span>')
        if len(value) > 12:
            chips.append(f'<span class="chip muted">+{len(value) - 12}</span>')
        return '<div class="chips">' + "".join(chips) + "</div>"
    if isinstance(value, dict):
        return f'<code class="json-inline">{escape(_short_text(value))}</code>'
    return escape(_short_text(value))


def _render_details(row: dict[str, Any]) -> str:
    raw = json.dumps(row, ensure_ascii=False, indent=2)
    return (
        '<details class="row-details">'
        "<summary>JSON</summary>"
        f"<pre>{escape(raw)}</pre>"
        "</details>"
    )


def render_html_report(snapshot: dict[str, Any]) -> str:
    total_rows = sum(int(table["count"]) for table in snapshot["tables"])
    displayed_rows = sum(int(table["displayed_count"]) for table in snapshot["tables"])
    playbook_count = next(
        (int(table["count"]) for table in snapshot["tables"] if table["name"] == "playbook_items"),
        0,
    )
    feedback_count = next(
        (int(table["count"]) for table in snapshot["tables"] if table["name"] == "qa_feedback"),
        0,
    )

    nav_buttons = ['<button class="tab is-active" type="button" data-table="all">All</button>']
    for table in snapshot["tables"]:
        label = f'{table["name"]} ({table["count"]})'
        nav_buttons.append(
            f'<button class="tab" type="button" data-table="{escape(table["name"])}">'
            f"{escape(label)}</button>"
        )

    table_sections: list[str] = []
    for table in snapshot["tables"]:
        headers = "".join(f"<th>{escape(column)}</th>" for column in table["columns"])
        rows_html: list[str] = []
        for display_row, raw_row in zip(table["display_rows"], table["rows"]):
            cells = "".join(f"<td>{_render_cell(display_row.get(column))}</td>" for column in table["columns"])
            rows_html.append(
                f'<tr class="data-row"><td class="details-cell">{_render_details(raw_row)}</td>{cells}</tr>'
            )

        if rows_html:
            body = "\n".join(rows_html)
            table_html = (
                '<div class="table-wrap">'
                "<table>"
                f"<thead><tr><th>Details</th>{headers}</tr></thead>"
                f"<tbody>{body}</tbody>"
                "</table>"
                "</div>"
            )
        else:
            table_html = '<div class="empty">No rows in this table.</div>'

        limit_note = ""
        if table["count"] > table["displayed_count"]:
            limit_note = f'<span class="pill">Showing {table["displayed_count"]} of {table["count"]}</span>'

        table_sections.append(
            f"""
            <section class="table-section" data-table="{escape(table["name"])}">
              <div class="section-head">
                <div>
                  <h2>{escape(table["name"])}</h2>
                  <p>{table["count"]} rows</p>
                </div>
                {limit_note}
              </div>
              {table_html}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ACE Playbook SQLite Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dde5;
      --accent: #0f766e;
      --accent-ink: #ffffff;
      --chip: #e8f4f2;
      --chip-ink: #0f5f59;
      --warn: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 26px; font-weight: 700; }}
    h2 {{ font-size: 18px; font-weight: 700; }}
    .subhead {{
      margin-top: 8px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .stat strong {{
      display: block;
      margin-top: 6px;
      font-size: 24px;
    }}
    .controls {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .control-row {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    input[type="search"] {{
      flex: 1 1 260px;
      min-width: 0;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      color: var(--ink);
      background: var(--panel);
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    .tab {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      height: 36px;
      padding: 0 12px;
      white-space: nowrap;
      cursor: pointer;
    }}
    .tab.is-active {{
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
    }}
    main .shell {{
      display: grid;
      gap: 18px;
    }}
    .table-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .section-head p {{
      color: var(--muted);
      margin-top: 2px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #fff4d6;
      color: var(--warn);
      white-space: nowrap;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 72vh;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      max-width: 420px;
      overflow-wrap: anywhere;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef2f6;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    tr[hidden] {{ display: none; }}
    .details-cell {{
      width: 96px;
      max-width: 96px;
    }}
    .row-details summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    pre {{
      margin: 10px 0 0;
      padding: 12px;
      max-width: min(900px, 78vw);
      max-height: 460px;
      overflow: auto;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #101828;
      color: #f8fafc;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      display: inline-flex;
      max-width: 240px;
      min-height: 24px;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--chip);
      color: var(--chip-ink);
      overflow-wrap: anywhere;
    }}
    .muted {{ color: var(--muted); }}
    .json-inline {{
      display: block;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #344054;
    }}
    .empty {{
      padding: 28px 16px;
      color: var(--muted);
    }}
    .hidden-count {{
      color: var(--muted);
      min-height: 20px;
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 16px; }}
      h1 {{ font-size: 22px; }}
      .section-head {{
        align-items: flex-start;
        flex-direction: column;
      }}
      th, td {{ max-width: 280px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="shell">
      <h1>ACE Playbook SQLite Report</h1>
      <p class="subhead">{escape(snapshot["database"])} · generated {escape(str(snapshot["generated_at"]))} · limit {snapshot["limit"]} · {'full' if snapshot["full"] else 'summary'}</p>
      <div class="stats">
        <div class="stat"><span>Tables</span><strong>{len(snapshot["tables"])}</strong></div>
        <div class="stat"><span>Total Rows</span><strong>{total_rows}</strong></div>
        <div class="stat"><span>Displayed Rows</span><strong>{displayed_rows}</strong></div>
        <div class="stat"><span>Playbook Items</span><strong>{playbook_count}</strong></div>
        <div class="stat"><span>Feedback</span><strong>{feedback_count}</strong></div>
      </div>
    </div>
  </header>
  <div class="controls">
    <div class="shell control-row">
      <input id="search" type="search" placeholder="Search rows">
      <div class="tabs" role="tablist">
        {''.join(nav_buttons)}
      </div>
      <div id="match-count" class="hidden-count"></div>
    </div>
  </div>
  <main>
    <div class="shell">
      {''.join(table_sections)}
    </div>
  </main>
  <script>
    const search = document.querySelector("#search");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const sections = Array.from(document.querySelectorAll(".table-section"));
    const matchCount = document.querySelector("#match-count");
    let activeTable = "all";

    function updateVisibleTable(tableName) {{
      activeTable = tableName;
      tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.table === tableName));
      sections.forEach((section) => {{
        section.hidden = tableName !== "all" && section.dataset.table !== tableName;
      }});
      applySearch();
    }}

    function applySearch() {{
      const query = search.value.trim().toLowerCase();
      let shown = 0;
      let hidden = 0;
      sections.forEach((section) => {{
        if (section.hidden) return;
        section.querySelectorAll(".data-row").forEach((row) => {{
          const matched = !query || row.textContent.toLowerCase().includes(query);
          row.hidden = !matched;
          if (matched) shown += 1;
          else hidden += 1;
        }});
      }});
      matchCount.textContent = query ? `${{shown}} shown · ${{hidden}} hidden` : "";
    }}

    tabs.forEach((tab) => tab.addEventListener("click", () => updateVisibleTable(tab.dataset.table)));
    search.addEventListener("input", applySearch);
  </script>
</body>
</html>
"""


def write_html_report(
    db_path: Path,
    output_path: Path,
    table: str = "all",
    limit: int = 100,
    full: bool = False,
) -> dict[str, Any]:
    snapshot = build_database_snapshot(db_path=db_path, table=table, limit=limit, full=full)
    html = render_html_report(snapshot)
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return snapshot
