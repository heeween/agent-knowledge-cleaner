from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from incremental_kb.analyzers import DeterministicAnalyzer
from incremental_kb.core import (
    bootstrap_baseline, connect, decide_review, ingest, list_reviews, publish,
    rollback, sha256_file, validate_release,
)


PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT / "output" / "kb_entries_official_v3.jsonl"
BASELINE_SHA = "f57b19feb7f6a6e118e7ab48737d54ce2f59d56a70c1f2fea9e274076561c51f"


def chat(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.incoming = self.root / "incoming"
        self.incoming.mkdir()
        self.db = connect(self.root / "state.sqlite3")
        bootstrap_baseline(self.db, BASELINE)
        self.analyzer = DeterministicAnalyzer()

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def write(self, name: str, content: str) -> Path:
        path = self.incoming / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_frozen_baseline_is_unchanged(self):
        self.assertEqual(sha256_file(BASELINE), BASELINE_SHA)
        rows = [json.loads(x) for x in BASELINE.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 645)
        self.assertEqual([rows[0]["kb_id"], rows[-1]["kb_id"]], ["KB-0001", "KB-0645"])

    def test_new_file_and_idempotent_repeat(self):
        path = self.write("a.md", chat("【2026-09-15 10:00】客户：怎么设置新的业务标签？", "【2026-09-15 10:01】客服：进入标签设置页面后新建标签并保存。"))
        first = ingest(self.db, path, self.analyzer)
        second = ingest(self.db, path, self.analyzer)
        self.assertEqual(first["summary"]["files_new"], 1)
        self.assertEqual(first["summary"]["candidate_issues"], 1)
        self.assertEqual(second["summary"]["files_unchanged"], 1)
        self.assertEqual(len(list_reviews(self.db)), 1)

    def test_dry_run_has_no_side_effects(self):
        path = self.write("dry.md", chat("【2026-09-15 10:00】客户：如何导出客户？", "【2026-09-15 10:01】客服：点击导出并选择字段。"))
        result = ingest(self.db, path, self.analyzer, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0], 0)

    def test_cli_dry_run_does_not_create_missing_state(self):
        path = self.write("cli.md", chat("【2026-09-15 10:00】客户：如何导出客户？", "【2026-09-15 10:01】客服：点击导出并选择字段。"))
        state = self.root / "missing.sqlite3"
        command = [str(PROJECT / ".venv/bin/python"), str(PROJECT / "pipeline.py"), "--state", str(state), "ingest", str(path), "--dry-run"]
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(state.exists())

    def test_append_recomputes_overlap_and_cross_boundary_issue(self):
        path = self.write("append.md", chat("【2026-09-15 10:00】客户：如何配置短信签名？"))
        ingest(self.db, path, self.analyzer, overlap=30)
        path.write_text(path.read_text(encoding="utf-8") + "【2026-09-15 10:01】客服：先完成签名备案，再在短信设置中选择该签名。\n", encoding="utf-8")
        result = ingest(self.db, path, self.analyzer, overlap=30)
        self.assertEqual(result["summary"]["files_appended"], 1)
        active = self.db.execute("SELECT question,answer FROM issues WHERE status='active'").fetchall()
        self.assertTrue(any("签名备案" in row["answer"] for row in active))

    def test_historical_modification_and_deletion(self):
        path = self.write("change.md", chat("【2026-09-15 10:00】客户：为什么报表没有数据？", "【2026-09-15 10:01】客服：请先选择统计日期。"))
        ingest(self.db, self.incoming, self.analyzer)
        path.write_text(chat("【2026-09-15 10:00】客户：为什么报表没有数据？", "【2026-09-15 10:01】客服：请先选择门店和统计日期。"), encoding="utf-8")
        modified = ingest(self.db, self.incoming, self.analyzer)
        self.assertEqual(modified["summary"]["files_modified"], 1)
        path.unlink()
        deleted = ingest(self.db, self.incoming, self.analyzer)
        self.assertEqual(deleted["summary"]["files_deleted"], 1)
        self.assertEqual(self.db.execute("SELECT status FROM sources").fetchone()[0], "deleted")

    def test_source_change_invalidates_evidence_and_requires_revalidation(self):
        self.db.execute("INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)", (
            "KB-0646", 1, "active", "如何新增测试标签？", "在标签设置中新增并保存。", None, "fixture", "2026-09-15T00:00:00Z",
        ))
        self.db.commit()
        path = self.write("evidence.md", chat("【2026-09-15 10:00】客户：如何新增测试标签？", "【2026-09-15 10:01】客服：在标签设置中新增并保存。"))
        ingest(self.db, path, self.analyzer)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM evidence_links").fetchone()[0], 1)
        path.write_text(chat("【2026-09-15 10:00】客户：如何新增测试标签？", "【2026-09-15 10:01】客服：在其他页面新增并保存。"), encoding="utf-8")
        ingest(self.db, path, self.analyzer)
        self.assertTrue(any(row["relation"] == "revalidation_required" for row in list_reviews(self.db)))
        self.assertEqual(self.db.execute("SELECT question,answer FROM kb_revisions WHERE kb_id='KB-0646'").fetchone()["answer"], "在标签设置中新增并保存。")

    def test_exact_move_keeps_source_id(self):
        path = self.write("old.md", chat("【2026-09-15 10:00】客户：如何新增客户？", "【2026-09-15 10:01】客服：点击新增客户并保存。"))
        ingest(self.db, self.incoming, self.analyzer)
        source_id = self.db.execute("SELECT source_id FROM sources").fetchone()[0]
        path.rename(self.incoming / "new.md")
        result = ingest(self.db, self.incoming, self.analyzer)
        self.assertEqual(result["summary"]["files_moved"], 1)
        self.assertEqual(self.db.execute("SELECT source_id FROM sources").fetchone()[0], source_id)

    def test_conflict_requires_review(self):
        self.db.execute("INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)", (
            "KB-0646", 1, "active", "系统是否支持删除客户？", "系统支持删除客户。", None, "test", "2026-09-15T00:00:00Z",
        ))
        self.db.commit()
        path = self.write("conflict.md", chat("【2026-09-15 10:00】客户：系统是否支持删除客户？", "【2026-09-15 10:01】客服：系统不支持删除客户。"))
        result = ingest(self.db, path, self.analyzer)
        self.assertEqual(result["summary"]["knowledge_conflict"], 1)
        self.assertEqual(list_reviews(self.db)[0]["relation"], "conflict")

    def test_invalidated_candidate_cannot_be_approved(self):
        path = self.write("obsolete.md", chat("【2026-09-15 10:00】客户：怎么设置新的业务标签？", "【2026-09-15 10:01】客服：进入标签设置后新增并保存。"))
        ingest(self.db, path, self.analyzer)
        old_review = list_reviews(self.db)[0]["review_id"]
        path.write_text(chat("【2026-09-15 10:00】客户：怎么设置新的业务标签？", "【2026-09-15 10:01】客服：进入其他设置后新增并保存。"), encoding="utf-8")
        ingest(self.db, path, self.analyzer)
        with self.assertRaises(ValueError):
            decide_review(self.db, old_review, "approved")

    def test_stable_kb_id_allocation(self):
        for number in (1, 2):
            path = self.write(f"new-{number}.md", chat(f"【2026-09-15 10:00】客户：如何配置测试功能{number}？", f"【2026-09-15 10:01】客服：进入功能{number}页面完成配置并保存。"))
            ingest(self.db, path, self.analyzer)
            review = list_reviews(self.db)[0]
            result = decide_review(self.db, review["review_id"], "approved", "fixture approval")
            self.assertEqual(result["kb_id"], f"KB-{645 + number:04d}")

    def test_publish_manifest_immutable_and_rollback(self):
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
        first = publish(self.db, repo, "1.0.0", embedding_model="mock", embedding_dimension=3)
        manifest = validate_release(first)
        self.assertEqual(manifest["chunk_count"], 645)
        self.assertEqual(manifest["embedding"]["dimension"], 3)
        with self.assertRaises(FileExistsError):
            publish(self.db, repo, "1.0.0", embedding_model="mock", embedding_dimension=3)
        second = publish(self.db, repo, "1.0.1", embedding_model="mock", embedding_dimension=3)
        self.assertEqual((repo / "releases" / "current").resolve(), second.resolve())
        self.assertEqual(rollback(repo), "1.0.0")
    def test_manifest_detects_tampering(self):
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        (repo / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
        release = publish(self.db, repo, "1.0.0", embedding_model="mock", embedding_dimension=3)
        (release / "chunks.jsonl").write_text((release / "chunks.jsonl").read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_release(release)


if __name__ == "__main__":
    unittest.main()
