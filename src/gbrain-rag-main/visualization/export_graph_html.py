#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZATION_DIR = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "data" / "index" / "gbrain.sqlite3"
DEFAULT_TEMPLATE = VISUALIZATION_DIR / "templates" / "entity_graph.html"
DEFAULT_OUTPUT = VISUALIZATION_DIR / "output" / "entity_graph.html"


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite index not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    required = {"chunks", "entities", "chunk_entities", "entity_links"}
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    existing = {row["name"] for row in rows}
    missing = sorted(required - existing)
    if missing:
        raise RuntimeError(f"SQLite index is missing required tables: {', '.join(missing)}")


def database_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    def count(table: str) -> int:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    kind_rows = conn.execute(
        "SELECT kind, COUNT(*) AS count FROM entities GROUP BY kind ORDER BY count DESC, kind"
    ).fetchall()
    relation_rows = conn.execute(
        """
        SELECT relation, COUNT(*) AS edges, SUM(evidence_count) AS evidence
        FROM entity_links
        GROUP BY relation
        ORDER BY edges DESC, relation
        """
    ).fetchall()
    source_rows = conn.execute(
        "SELECT source_type, COUNT(*) AS count FROM chunks GROUP BY source_type ORDER BY count DESC"
    ).fetchall()
    return {
        "chunks": count("chunks"),
        "entities": count("entities"),
        "chunk_entities": count("chunk_entities"),
        "entity_links": count("entity_links"),
        "kinds": {row["kind"]: int(row["count"]) for row in kind_rows},
        "relations": {
            row["relation"]: {
                "edges": int(row["edges"]),
                "evidence": int(row["evidence"] or 0),
            }
            for row in relation_rows
        },
        "sources": {row["source_type"]: int(row["count"]) for row in source_rows},
    }


def build_edge_where(args: argparse.Namespace) -> tuple[str, list[Any]]:
    clauses = ["l.evidence_count >= ?"]
    params: list[Any] = [args.min_evidence]

    include_kinds = parse_csv(args.include_kinds)
    exclude_kinds = parse_csv(args.exclude_kinds)
    relations = parse_csv(args.relations)

    if include_kinds:
        placeholders = ",".join("?" for _ in include_kinds)
        clauses.append(f"se.kind IN ({placeholders})")
        params.extend(include_kinds)
        clauses.append(f"te.kind IN ({placeholders})")
        params.extend(include_kinds)

    if exclude_kinds:
        placeholders = ",".join("?" for _ in exclude_kinds)
        clauses.append(f"se.kind NOT IN ({placeholders})")
        params.extend(exclude_kinds)
        clauses.append(f"te.kind NOT IN ({placeholders})")
        params.extend(exclude_kinds)

    if relations:
        placeholders = ",".join("?" for _ in relations)
        clauses.append(f"l.relation IN ({placeholders})")
        params.extend(relations)

    terms = [term.lower() for term in parse_csv(args.query.replace(" ", ","))] if args.query else []
    if terms:
        term_clauses = []
        for term in terms:
            pattern = f"%{term}%"
            term_clauses.append(
                """
                (
                    LOWER(se.name) LIKE ?
                    OR LOWER(se.normalized_name) LIKE ?
                    OR LOWER(se.kind) LIKE ?
                    OR LOWER(te.name) LIKE ?
                    OR LOWER(te.normalized_name) LIKE ?
                    OR LOWER(te.kind) LIKE ?
                )
                """
            )
            params.extend([pattern, pattern, pattern, pattern, pattern, pattern])
        clauses.append("(" + " OR ".join(term_clauses) + ")")

    return " AND ".join(clauses), params


def fetch_candidate_edges(conn: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    where_sql, params = build_edge_where(args)
    candidate_limit = args.candidate_edges or max(args.limit_edges * 8, args.limit_nodes * 20, 4000)
    params.append(candidate_limit)
    return conn.execute(
        f"""
        WITH node_stats AS (
            SELECT entity_id, COUNT(*) AS chunk_count
            FROM chunk_entities
            GROUP BY entity_id
        )
        SELECT
            l.source_entity_id,
            l.target_entity_id,
            l.relation,
            l.evidence_count,
            se.kind AS source_kind,
            se.name AS source_name,
            se.normalized_name AS source_normalized_name,
            COALESCE(sns.chunk_count, 0) AS source_chunk_count,
            te.kind AS target_kind,
            te.name AS target_name,
            te.normalized_name AS target_normalized_name,
            COALESCE(tns.chunk_count, 0) AS target_chunk_count
        FROM entity_links l
        JOIN entities se ON se.entity_id = l.source_entity_id
        JOIN entities te ON te.entity_id = l.target_entity_id
        LEFT JOIN node_stats sns ON sns.entity_id = se.entity_id
        LEFT JOIN node_stats tns ON tns.entity_id = te.entity_id
        WHERE {where_sql}
        ORDER BY
            l.evidence_count DESC,
            (COALESCE(sns.chunk_count, 0) + COALESCE(tns.chunk_count, 0)) DESC,
            LOWER(se.name) ASC,
            LOWER(te.name) ASC
        LIMIT ?
        """,
        params,
    ).fetchall()


def node_from_row(row: sqlite3.Row, prefix: str) -> dict[str, Any]:
    return {
        "id": row[f"{prefix}_entity_id"],
        "label": row[f"{prefix}_name"],
        "kind": row[f"{prefix}_kind"],
        "normalized_name": row[f"{prefix}_normalized_name"],
        "chunk_count": int(row[f"{prefix}_chunk_count"]),
        "degree": 0,
        "weighted_degree": 0,
    }


def select_graph(rows: list[sqlite3.Row], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for row in rows:
        edge_key = (row["source_entity_id"], row["target_entity_id"], row["relation"])
        if edge_key in seen_edges:
            continue
        if len(edges) >= args.limit_edges:
            break

        new_nodes = [
            entity_id
            for entity_id in (row["source_entity_id"], row["target_entity_id"])
            if entity_id not in nodes
        ]
        if len(nodes) + len(new_nodes) > args.limit_nodes:
            if row["source_entity_id"] not in nodes or row["target_entity_id"] not in nodes:
                continue

        if row["source_entity_id"] not in nodes:
            nodes[row["source_entity_id"]] = node_from_row(row, "source")
        if row["target_entity_id"] not in nodes:
            nodes[row["target_entity_id"]] = node_from_row(row, "target")

        evidence_count = int(row["evidence_count"])
        nodes[row["source_entity_id"]]["degree"] += 1
        nodes[row["target_entity_id"]]["degree"] += 1
        nodes[row["source_entity_id"]]["weighted_degree"] += evidence_count
        nodes[row["target_entity_id"]]["weighted_degree"] += evidence_count

        edges.append(
            {
                "id": f"{row['source_entity_id']}::{row['target_entity_id']}::{row['relation']}",
                "source": row["source_entity_id"],
                "target": row["target_entity_id"],
                "relation": row["relation"],
                "evidence_count": evidence_count,
            }
        )
        seen_edges.add(edge_key)

    ordered_nodes = sorted(
        nodes.values(),
        key=lambda item: (
            -int(item["weighted_degree"]),
            -int(item["degree"]),
            -int(item["chunk_count"]),
            item["kind"],
            item["label"],
        ),
    )
    for rank, node in enumerate(ordered_nodes, start=1):
        node["rank"] = rank
    return ordered_nodes, edges


def fetch_fallback_nodes(conn: sqlite3.Connection, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.query:
        return []
    include_kinds = parse_csv(args.include_kinds)
    exclude_kinds = parse_csv(args.exclude_kinds)
    terms = [term.lower() for term in parse_csv(args.query.replace(" ", ","))]
    clauses = []
    params: list[Any] = []
    if include_kinds:
        placeholders = ",".join("?" for _ in include_kinds)
        clauses.append(f"e.kind IN ({placeholders})")
        params.extend(include_kinds)
    if exclude_kinds:
        placeholders = ",".join("?" for _ in exclude_kinds)
        clauses.append(f"e.kind NOT IN ({placeholders})")
        params.extend(exclude_kinds)
    if terms:
        term_clauses = []
        for term in terms:
            pattern = f"%{term}%"
            term_clauses.append(
                "(LOWER(e.name) LIKE ? OR LOWER(e.normalized_name) LIKE ? OR LOWER(e.kind) LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])
        clauses.append("(" + " OR ".join(term_clauses) + ")")
    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(args.limit_nodes)
    rows = conn.execute(
        f"""
        WITH node_stats AS (
            SELECT entity_id, COUNT(*) AS chunk_count
            FROM chunk_entities
            GROUP BY entity_id
        )
        SELECT
            e.entity_id,
            e.kind,
            e.name,
            e.normalized_name,
            COALESCE(ns.chunk_count, 0) AS chunk_count
        FROM entities e
        LEFT JOIN node_stats ns ON ns.entity_id = e.entity_id
        {where_sql}
        ORDER BY COALESCE(ns.chunk_count, 0) DESC, LOWER(e.name) ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    nodes = [
        {
            "id": row["entity_id"],
            "label": row["name"],
            "kind": row["kind"],
            "normalized_name": row["normalized_name"],
            "chunk_count": int(row["chunk_count"]),
            "degree": 0,
            "weighted_degree": 0,
            "rank": index + 1,
        }
        for index, row in enumerate(rows)
    ]
    return nodes


def fetch_node_samples(conn: sqlite3.Connection, entity_id: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = conn.execute(
        """
        SELECT
            c.chunk_id,
            c.doc_name,
            c.source_type,
            c.title,
            c.page_label,
            SUBSTR(REPLACE(REPLACE(c.text, CHAR(10), ' '), CHAR(13), ' '), 1, 260) AS snippet
        FROM chunk_entities ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
        WHERE ce.entity_id = ?
        ORDER BY c.updated_at DESC, c.doc_name ASC
        LIMIT ?
        """,
        (entity_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_edge_samples(
    conn: sqlite3.Connection,
    source_entity_id: str,
    target_entity_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows = conn.execute(
        """
        SELECT
            c.chunk_id,
            c.doc_name,
            c.source_type,
            c.title,
            c.page_label,
            SUBSTR(REPLACE(REPLACE(c.text, CHAR(10), ' '), CHAR(13), ' '), 1, 260) AS snippet
        FROM chunk_entities source_ce
        JOIN chunk_entities target_ce
          ON target_ce.chunk_id = source_ce.chunk_id
         AND target_ce.entity_id = ?
        JOIN chunks c ON c.chunk_id = source_ce.chunk_id
        WHERE source_ce.entity_id = ?
        ORDER BY c.updated_at DESC, c.doc_name ASC
        LIMIT ?
        """,
        (target_entity_id, source_entity_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def hydrate_samples(
    conn: sqlite3.Connection,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.node_samples > 0:
        for node in nodes:
            node["samples"] = fetch_node_samples(conn, node["id"], args.node_samples)

    if args.edge_sample_edges > 0 and args.edge_samples > 0:
        edges_by_weight = sorted(edges, key=lambda item: int(item["evidence_count"]), reverse=True)
        for edge in edges_by_weight[: args.edge_sample_edges]:
            edge["samples"] = fetch_edge_samples(
                conn,
                edge["source"],
                edge["target"],
                args.edge_samples,
            )


def safe_json_for_html(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def format_count(value: Any) -> str:
    return f"{int(value or 0):,}"


def render_html(template_path: Path, output_path: Path, graph_data: dict[str, Any]) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"HTML template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    meta = graph_data["meta"]
    html = template
    html = html.replace("__STAT_NODES__", format_count(meta["exported"]["nodes"]))
    html = html.replace("__STAT_EDGES__", format_count(meta["exported"]["edges"]))
    html = html.replace("__STAT_ALL_NODES__", format_count(meta["stats"]["entities"]))
    html = html.replace("__STAT_ALL_EDGES__", format_count(meta["stats"]["entity_links"]))
    html = html.replace("__GRAPH_JSON__", safe_json_for_html(graph_data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def build_graph_data(conn: sqlite3.Connection, db_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    stats = database_stats(conn)
    rows = fetch_candidate_edges(conn, args)
    nodes, edges = select_graph(rows, args)
    if not nodes and args.query:
        nodes = fetch_fallback_nodes(conn, args)
        edges = []
    hydrate_samples(conn, nodes, edges, args)

    exported_kind_counts: dict[str, int] = {}
    for node in nodes:
        exported_kind_counts[node["kind"]] = exported_kind_counts.get(node["kind"], 0) + 1

    max_evidence = max((int(edge["evidence_count"]) for edge in edges), default=0)
    return {
        "meta": {
            "title": args.title,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "db_path": str(db_path),
            "stats": stats,
            "filters": {
                "query": args.query,
                "include_kinds": parse_csv(args.include_kinds),
                "exclude_kinds": parse_csv(args.exclude_kinds),
                "relations": parse_csv(args.relations),
                "min_evidence": args.min_evidence,
                "limit_nodes": args.limit_nodes,
                "limit_edges": args.limit_edges,
            },
            "exported": {
                "nodes": len(nodes),
                "edges": len(edges),
                "kinds": exported_kind_counts,
                "max_evidence": max_evidence,
            },
        },
        "nodes": nodes,
        "edges": edges,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the GBrain RAG entity co-mention graph to an interactive HTML file."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to gbrain.sqlite3.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated HTML output path.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="HTML template path.")
    parser.add_argument("--title", default="GBrain RAG Entity Graph", help="Title shown in the HTML.")
    parser.add_argument("--query", default="", help="Only export edges touching entities matching this text.")
    parser.add_argument(
        "--include-kinds",
        default="",
        help="Comma-separated entity kinds to include, for example: model,platform,scene.",
    )
    parser.add_argument(
        "--exclude-kinds",
        default="",
        help="Comma-separated entity kinds to exclude, for example: version,oid.",
    )
    parser.add_argument(
        "--relations",
        default="",
        help="Comma-separated relation types to include. Empty means all relations.",
    )
    parser.add_argument("--min-evidence", type=int, default=2, help="Minimum edge evidence_count.")
    parser.add_argument("--limit-nodes", type=int, default=360, help="Maximum exported nodes.")
    parser.add_argument("--limit-edges", type=int, default=720, help="Maximum exported edges.")
    parser.add_argument(
        "--candidate-edges",
        type=int,
        default=0,
        help="Internal SQL candidate edge limit. Defaults to a value based on graph limits.",
    )
    parser.add_argument("--node-samples", type=int, default=3, help="Sample chunks embedded per node.")
    parser.add_argument(
        "--edge-sample-edges",
        type=int,
        default=80,
        help="Only this many strongest edges embed sample chunks.",
    )
    parser.add_argument("--edge-samples", type=int, default=2, help="Sample chunks embedded per sampled edge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_nodes <= 0 or args.limit_edges < 0:
        raise SystemExit("--limit-nodes must be > 0 and --limit-edges must be >= 0")
    if args.min_evidence <= 0:
        raise SystemExit("--min-evidence must be > 0")

    with connect_readonly(args.db) as conn:
        ensure_schema(conn)
        graph_data = build_graph_data(conn, args.db, args)
    render_html(args.template, args.output, graph_data)
    print(
        f"Wrote {args.output} "
        f"({graph_data['meta']['exported']['nodes']} nodes, "
        f"{graph_data['meta']['exported']['edges']} edges)."
    )


if __name__ == "__main__":
    main()
