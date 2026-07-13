import sqlite3
import tempfile
import unittest
from pathlib import Path

from ace_rag.playbook.inspect import list_tables, render_database_snapshot, write_html_report


class PlaybookInspectTests(unittest.TestCase):
    def make_db(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db_path = Path(tmp.name) / "playbook.sqlite3"

        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE playbook_items (
              item_id TEXT PRIMARY KEY,
              section TEXT NOT NULL,
              content TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'active',
              tags_json TEXT NOT NULL DEFAULT '[]',
              source_hints_json TEXT NOT NULL DEFAULT '[]',
              query_intents_json TEXT NOT NULL DEFAULT '[]',
              expansion_terms_json TEXT NOT NULL DEFAULT '[]',
              helpful_count INTEGER NOT NULL DEFAULT 0,
              harmful_count INTEGER NOT NULL DEFAULT 0,
              confidence REAL NOT NULL DEFAULT 0.5,
              provenance_json TEXT NOT NULL DEFAULT '{}',
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE TABLE qa_runs (
              run_id TEXT PRIMARY KEY,
              query TEXT NOT NULL,
              request_json TEXT NOT NULL DEFAULT '{}',
              v2_request_json TEXT NOT NULL DEFAULT '{}',
              v2_response_json TEXT NOT NULL DEFAULT '{}',
              playbook_item_ids_json TEXT NOT NULL DEFAULT '[]',
              answer TEXT NOT NULL,
              timings_json TEXT NOT NULL DEFAULT '{}',
              created_at REAL NOT NULL
            );

            CREATE TABLE qa_feedback (
              feedback_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              feedback_type TEXT NOT NULL,
              rating INTEGER,
              corrected_answer TEXT,
              expected_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
              comment TEXT,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at REAL NOT NULL
            );

            CREATE TABLE playbook_operations (
              op_id TEXT PRIMARY KEY,
              feedback_id TEXT,
              operation_type TEXT NOT NULL,
              target_item_id TEXT,
              payload_json TEXT NOT NULL DEFAULT '{}',
              status TEXT NOT NULL DEFAULT 'pending',
              created_at REAL NOT NULL,
              applied_at REAL
            );

            CREATE TABLE playbook_state (
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL DEFAULT '{}',
              updated_at REAL NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO playbook_items (
              item_id, section, content, status, tags_json, source_hints_json,
              query_intents_json, expansion_terms_json, helpful_count,
              harmful_count, confidence, provenance_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "pb-1",
                "source_routing",
                "查询 did/rid 时优先看 adela",
                "active",
                '["deployment"]',
                '["adela"]',
                '["deployment"]',
                '["did","rid"]',
                1,
                0,
                0.9,
                '{"source":"seed"}',
                10.0,
                11.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO qa_runs (
              run_id, query, request_json, v2_request_json, v2_response_json,
              playbook_item_ids_json, answer, timings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "did 是多少",
                '{"query":"did 是多少"}',
                '{"query":"did 是多少"}',
                '{"ok":true}',
                '["pb-1"]',
                "answer text",
                '{"retrieve_ms":1.2}',
                12.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO qa_feedback (
              feedback_id, run_id, feedback_type, rating, corrected_answer,
              expected_evidence_ids_json, comment, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fb-1",
                "run-1",
                "helpful",
                5,
                None,
                '["e1","e2"]',
                "looks good",
                "pending",
                13.0,
            ),
        )
        conn.commit()
        conn.close()
        return db_path

    def test_list_tables(self):
        db_path = self.make_db()
        with sqlite3.connect(db_path) as conn:
            self.assertEqual(
                list_tables(conn),
                ["playbook_items", "qa_runs", "qa_feedback", "playbook_operations", "playbook_state"],
            )

    def test_render_database_snapshot(self):
        db_path = self.make_db()
        output = render_database_snapshot(db_path, limit=10)
        self.assertIn("database:", output)
        self.assertIn("playbook_items (1 rows)", output)
        self.assertIn("qa_runs (1 rows)", output)
        self.assertIn("pb-1", output)
        self.assertIn("source_routing", output)
        self.assertIn("did/rid", output)
        self.assertIn("looks good", output)

    def test_write_html_report(self):
        db_path = self.make_db()
        output_path = db_path.parent / "report.html"
        snapshot = write_html_report(db_path, output_path, limit=10)

        self.assertTrue(output_path.exists())
        html = output_path.read_text(encoding="utf-8")
        self.assertIn("ACE Playbook SQLite Report", html)
        self.assertIn("playbook_items", html)
        self.assertIn("qa_feedback", html)
        self.assertIn("pb-1", html)
        self.assertIn("Search rows", html)
        self.assertEqual(snapshot["tables"][0]["name"], "playbook_items")


if __name__ == "__main__":
    unittest.main()
