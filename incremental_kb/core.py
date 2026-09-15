from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from . import SCHEMA_VERSION
from .analyzers import Analyzer


MESSAGE_PATTERN = re.compile(r"^【(?P<timestamp>[^】]+)】(?P<speaker>[^:：]+)[:：]\s*(?P<content>.*)$")
QUESTION_PATTERN = re.compile(r"[?？]|为什么|怎么|怎样|如何|能不能|可不可以|有没有|是不是|什么|哪里|多少|失败|报错|不行|看不到")
SENSITIVE_PATTERNS = [
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("api_key", re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*\S+")),
    ("env_path", re.compile(r"(?:^|/)\.env(?:\.|$)")),
]
RELATIONS = {
    "new", "duplicate_evidence", "answer_supplement", "multi_cause",
    "conflict", "temporal_update", "low_value",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def connect(path: Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sources(
      source_id TEXT PRIMARY KEY, managed_root TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
      content_sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
      last_message_time TEXT, message_watermark INTEGER NOT NULL DEFAULT 0,
      processed_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active'
    );
    CREATE TABLE IF NOT EXISTS messages(
      source_id TEXT NOT NULL, message_index INTEGER NOT NULL, line_number INTEGER NOT NULL,
      timestamp TEXT, speaker TEXT, content TEXT NOT NULL, message_sha256 TEXT NOT NULL,
      PRIMARY KEY(source_id,message_index), FOREIGN KEY(source_id) REFERENCES sources(source_id)
    );
    CREATE TABLE IF NOT EXISTS issues(
      issue_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, start_index INTEGER NOT NULL,
      end_index INTEGER NOT NULL, fingerprint TEXT NOT NULL, question TEXT NOT NULL,
      answer TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
      invalidated_at TEXT, UNIQUE(source_id,fingerprint)
    );
    CREATE TABLE IF NOT EXISTS reviews(
      review_id TEXT PRIMARY KEY, issue_id TEXT NOT NULL, relation TEXT NOT NULL,
      target_kb_id TEXT, proposed_question TEXT NOT NULL, proposed_answer TEXT NOT NULL,
      confidence REAL NOT NULL, reason TEXT, status TEXT NOT NULL,
      created_at TEXT NOT NULL, decided_at TEXT, decision_note TEXT,
      UNIQUE(issue_id,relation,target_kb_id)
    );
    CREATE TABLE IF NOT EXISTS kb_revisions(
      kb_id TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL,
      question TEXT NOT NULL, answer TEXT NOT NULL, supersedes TEXT,
      provenance TEXT NOT NULL, created_at TEXT NOT NULL,
      PRIMARY KEY(kb_id,revision)
    );
    CREATE TABLE IF NOT EXISTS evidence_links(
      kb_id TEXT NOT NULL, revision INTEGER NOT NULL, issue_id TEXT NOT NULL,
      source_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
      PRIMARY KEY(kb_id,revision,issue_id)
    );
    CREATE TABLE IF NOT EXISTS audit_events(
      event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, entity_id TEXT NOT NULL,
      payload_json TEXT NOT NULL, created_at TEXT NOT NULL
    );
    """)
    return db


def audit(db: sqlite3.Connection, event_type: str, entity_id: str, payload: dict) -> None:
    db.execute(
        "INSERT INTO audit_events VALUES(?,?,?,?,?)",
        (f"AUD-{uuid.uuid4().hex}", event_type, entity_id, canonical_json(payload), utc_now()),
    )


def parse_messages(path: Path) -> list[dict]:
    messages = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        match = MESSAGE_PATTERN.match(line.strip())
        if not match:
            continue
        item = {
            "line_number": line_number,
            "timestamp": match.group("timestamp").strip(),
            "speaker": match.group("speaker").strip(),
            "content": match.group("content").strip(),
        }
        item["message_sha256"] = sha256_bytes(canonical_json(item).encode())
        messages.append(item)
    return messages


def segment_messages(source_id: str, messages: list[dict], start: int, max_messages: int = 30) -> list[dict]:
    if not messages:
        return []
    start = max(0, start)
    boundaries = [start]
    for index in range(start + 1, len(messages)):
        if index - boundaries[-1] >= max_messages:
            boundaries.append(index)
    boundaries.append(len(messages))
    issues = []
    for left, right in zip(boundaries, boundaries[1:]):
        block = messages[left:right]
        qpos = next((i for i, m in enumerate(block) if QUESTION_PATTERN.search(m["content"])), None)
        if qpos is None:
            continue
        question = block[qpos]["content"]
        answer_parts = [m["content"] for m in block[qpos + 1:] if m["content"]]
        answer = "\n".join(answer_parts)
        fingerprint = sha256_bytes((source_id + "\0" + "\0".join(m["message_sha256"] for m in block)).encode())
        issues.append({
            "issue_id": f"ISS-{fingerprint[:20]}", "source_id": source_id,
            "start_index": left, "end_index": right - 1, "fingerprint": fingerprint,
            "question": question, "answer": answer,
        })
    return issues


def changed_interval(old: list[str], new: list[str]) -> tuple[int, int]:
    matcher = SequenceMatcher(a=old, b=new, autojunk=False)
    spans = [(j, j + size) for tag, _, _, j, size in matcher.get_opcodes() if tag != "equal"]
    if not spans:
        return len(new), len(new)
    return min(x[0] for x in spans), max(x[1] for x in spans)


def list_input_files(incoming: Path) -> tuple[list[Path], bool, Path]:
    incoming = incoming.resolve()
    if incoming.is_dir():
        return sorted(p.resolve() for p in incoming.rglob("*.md") if p.is_file()), True, incoming
    if incoming.is_file():
        return [incoming], False, incoming.parent
    raise FileNotFoundError(incoming)


def active_kb(db: sqlite3.Connection) -> list[dict]:
    return [dict(row) for row in db.execute("""
      SELECT k.* FROM kb_revisions k JOIN (
        SELECT kb_id, MAX(revision) revision FROM kb_revisions GROUP BY kb_id
      ) latest USING(kb_id,revision) WHERE k.status='active' ORDER BY k.kb_id
    """)]


def bootstrap_baseline(db: sqlite3.Connection, baseline: Path) -> int:
    if db.execute("SELECT COUNT(*) FROM kb_revisions").fetchone()[0]:
        return 0
    records = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 645 or records[0]["kb_id"] != "KB-0001" or records[-1]["kb_id"] != "KB-0645":
        raise ValueError("frozen baseline must contain KB-0001..KB-0645")
    if sha256_file(baseline) != "f57b19feb7f6a6e118e7ab48737d54ce2f59d56a70c1f2fea9e274076561c51f":
        raise ValueError("frozen baseline SHA-256 mismatch")
    now = utc_now()
    for record in records:
        db.execute(
            "INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)",
            (record["kb_id"], 1, "active", record["question"], record["answer"], None,
             "frozen:kb_entries_official_v3.jsonl", now),
        )
    audit(db, "baseline_imported", "KB-0001..KB-0645", {"count": 645, "sha256": sha256_file(baseline)})
    db.commit()
    return len(records)


def _source_plan(db: sqlite3.Connection, path: Path, root: Path, overlap: int, explicit_source_id: str | None) -> dict:
    raw = path.read_bytes()
    digest, messages = sha256_bytes(raw), parse_messages(path)
    existing = db.execute("SELECT * FROM sources WHERE path=?", (str(path),)).fetchone()
    moved = None
    if not existing:
        moved = db.execute(
            "SELECT * FROM sources WHERE content_sha256=? AND status='active' AND path<>? ORDER BY processed_at LIMIT 1",
            (digest, str(path)),
        ).fetchone()
    if explicit_source_id:
        mapped = db.execute("SELECT * FROM sources WHERE source_id=?", (explicit_source_id,)).fetchone()
        if mapped is None:
            raise ValueError(f"unknown --source-id: {explicit_source_id}")
        if existing and existing["source_id"] != explicit_source_id:
            raise ValueError("--source-id conflicts with the registered path")
        existing = existing or mapped
    source_id = explicit_source_id or (existing["source_id"] if existing else moved["source_id"] if moved else f"SRC-{uuid.uuid4().hex}")
    old_messages = []
    if existing or moved:
        old_messages = [dict(row) for row in db.execute(
            "SELECT * FROM messages WHERE source_id=? ORDER BY message_index", (source_id,)
        )]
    old_hashes = [m["message_sha256"] for m in old_messages]
    new_hashes = [m["message_sha256"] for m in messages]
    if existing and existing["content_sha256"] == digest:
        change, affected_start = "unchanged", len(messages)
    elif moved and moved["content_sha256"] == digest:
        change, affected_start = "moved", len(messages)
    elif existing and len(new_hashes) >= len(old_hashes) and new_hashes[:len(old_hashes)] == old_hashes:
        change, affected_start = "appended", max(0, len(old_hashes) - overlap)
    elif existing:
        change = "modified"
        interval_start, _ = changed_interval(old_hashes, new_hashes)
        affected_start = max(0, interval_start - overlap)
    else:
        change, affected_start = "new", 0
    if existing and change in {"appended", "modified"}:
        boundary = db.execute(
            "SELECT MIN(start_index) FROM issues WHERE source_id=? AND status='active' AND end_index>=?",
            (source_id, affected_start),
        ).fetchone()[0]
        if boundary is not None:
            affected_start = min(affected_start, boundary)
    return {
        "path": str(path), "managed_root": str(root), "source_id": source_id,
        "content_sha256": digest, "size_bytes": len(raw), "messages": messages,
        "change": change, "affected_start": affected_start,
        "old_message_count": len(old_messages), "moved_from": moved["path"] if moved else None,
    }


def ingest(db: sqlite3.Connection, incoming: Path, analyzer: Analyzer, *, dry_run: bool = False,
           overlap: int = 30, explicit_source_id: str | None = None) -> dict:
    files, authoritative, root = list_input_files(incoming)
    plans = [_source_plan(db, path, root, overlap, explicit_source_id if len(files) == 1 else None) for path in files]
    seen = {p["path"] for p in plans}
    deleted = []
    if authoritative:
        deleted = [dict(row) for row in db.execute(
            "SELECT * FROM sources WHERE managed_root=? AND status='active'", (str(root),)
        ) if row["path"] not in seen and not any(p["moved_from"] == row["path"] for p in plans)]

    counts = Counter(f"files_{p['change']}" for p in plans)
    counts["files_deleted"] = len(deleted)
    candidate_issues = []
    for plan in plans:
        if plan["change"] in {"unchanged", "moved"}:
            continue
        candidate_issues.extend(segment_messages(plan["source_id"], plan["messages"], plan["affected_start"]))
    analyses = [(issue, analyzer.analyze(issue, active_kb(db))) for issue in candidate_issues]
    counts["new_messages"] = sum(max(0, len(p["messages"]) - p["old_message_count"]) for p in plans)
    counts["candidate_issues"] = len(candidate_issues)
    for _, analysis in analyses:
        counts[f"knowledge_{analysis.relation}"] += 1
    summary = dict(sorted(counts.items()))
    if dry_run:
        return {"dry_run": True, "summary": summary, "sources": [{k: v for k, v in p.items() if k != "messages"} for p in plans]}

    now = utc_now()
    with db:
        for deleted_source in deleted:
            db.execute("UPDATE sources SET status='deleted', processed_at=? WHERE source_id=?", (now, deleted_source["source_id"]))
            _invalidate_issues(db, deleted_source["source_id"], 0, now, "source_deleted")
        for plan in plans:
            if plan["change"] == "unchanged":
                continue
            if plan["change"] == "moved":
                db.execute("UPDATE sources SET path=?, managed_root=?, processed_at=? WHERE source_id=?",
                           (plan["path"], plan["managed_root"], now, plan["source_id"]))
                audit(db, "source_moved", plan["source_id"], {"to_sha256": plan["content_sha256"]})
                continue
            _invalidate_issues(db, plan["source_id"], plan["affected_start"], now, "source_changed")
            db.execute("DELETE FROM messages WHERE source_id=?", (plan["source_id"],))
            db.execute("""INSERT INTO sources VALUES(?,?,?,?,?,?,?,?, 'active')
              ON CONFLICT(source_id) DO UPDATE SET managed_root=excluded.managed_root,path=excluded.path,
              content_sha256=excluded.content_sha256,size_bytes=excluded.size_bytes,
              last_message_time=excluded.last_message_time,message_watermark=excluded.message_watermark,
              processed_at=excluded.processed_at,status='active'""",
              (plan["source_id"], plan["managed_root"], plan["path"], plan["content_sha256"],
               plan["size_bytes"], plan["messages"][-1]["timestamp"] if plan["messages"] else None,
               len(plan["messages"]), now))
            for index, msg in enumerate(plan["messages"]):
                db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?)", (
                    plan["source_id"], index, msg["line_number"], msg["timestamp"], msg["speaker"],
                    msg["content"], msg["message_sha256"],
                ))
            audit(db, f"source_{plan['change']}", plan["source_id"], {
                "content_sha256": plan["content_sha256"], "messages": len(plan["messages"]),
                "affected_start": plan["affected_start"],
            })
        for issue, analysis in analyses:
            db.execute("""INSERT INTO issues VALUES(?,?,?,?,?,?,?,?,?,NULL)
              ON CONFLICT(issue_id) DO UPDATE SET status='active',invalidated_at=NULL""",
              (issue["issue_id"], issue["source_id"], issue["start_index"], issue["end_index"],
               issue["fingerprint"], analysis.question, analysis.answer, "active", now))
            if analysis.relation == "duplicate_evidence" and analysis.target_kb_id:
                revision = db.execute("SELECT MAX(revision) FROM kb_revisions WHERE kb_id=?", (analysis.target_kb_id,)).fetchone()[0]
                db.execute("INSERT OR IGNORE INTO evidence_links VALUES(?,?,?,?,?,?)",
                           (analysis.target_kb_id, revision, issue["issue_id"], issue["source_id"], "active", now))
            elif analysis.relation == "low_value":
                audit(db, "issue_archived_low_value", issue["issue_id"], {"reason": analysis.reason})
            else:
                review_id = f"REV-{sha256_bytes((issue['issue_id'] + analysis.relation + str(analysis.target_kb_id)).encode())[:20]}"
                db.execute("INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?,NULL,NULL)", (
                    review_id, issue["issue_id"], analysis.relation, analysis.target_kb_id,
                    analysis.question, analysis.answer, analysis.confidence, analysis.reason,
                    "pending", now,
                ))
    return {"dry_run": False, "summary": summary}


def _invalidate_issues(db: sqlite3.Connection, source_id: str, start: int, now: str, reason: str) -> None:
    rows = list(db.execute(
        "SELECT issue_id FROM issues WHERE source_id=? AND status='active' AND end_index>=?", (source_id, start)
    ))
    for row in rows:
        issue_id = row["issue_id"]
        db.execute("UPDATE issues SET status='invalidated',invalidated_at=? WHERE issue_id=?", (now, issue_id))
        db.execute("UPDATE reviews SET status='obsolete',decided_at=?,decision_note=? WHERE issue_id=? AND status='pending'",
                   (now, reason, issue_id))
        links = list(db.execute("SELECT * FROM evidence_links WHERE issue_id=? AND status='active'", (issue_id,)))
        for link in links:
            db.execute("UPDATE evidence_links SET status='invalidated' WHERE kb_id=? AND revision=? AND issue_id=?",
                       (link["kb_id"], link["revision"], issue_id))
            review_id = f"REV-{sha256_bytes((issue_id + link['kb_id'] + reason).encode())[:20]}"
            entry = db.execute("SELECT * FROM kb_revisions WHERE kb_id=? AND revision=?",
                               (link["kb_id"], link["revision"])).fetchone()
            db.execute("INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?,NULL,NULL)", (
                review_id, issue_id, "revalidation_required", link["kb_id"], entry["question"],
                entry["answer"], 1.0, reason, "pending", now,
            ))


def list_reviews(db: sqlite3.Connection, status: str = "pending") -> list[dict]:
    return [dict(row) for row in db.execute("SELECT * FROM reviews WHERE status=? ORDER BY created_at,review_id", (status,))]


def decide_review(db: sqlite3.Connection, review_id: str, decision: str, note: str = "") -> dict:
    review = db.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
    if not review:
        raise KeyError(review_id)
    if review["status"] != "pending":
        raise ValueError("review already decided")
    issue_status = db.execute("SELECT status FROM issues WHERE issue_id=?", (review["issue_id"],)).fetchone()
    if review["relation"] != "revalidation_required" and (not issue_status or issue_status["status"] != "active"):
        raise ValueError("cannot approve a missing or invalidated issue")
    now = utc_now()
    with db:
        db.execute("UPDATE reviews SET status=?,decided_at=?,decision_note=? WHERE review_id=?",
                   (decision, now, note, review_id))
        if decision != "approved":
            audit(db, "review_rejected", review_id, {"note": note})
            return dict(review)
        if review["relation"] == "revalidation_required":
            audit(db, "revalidation_acknowledged", review_id, {"note": note})
            return dict(review)
        kb_id = review["target_kb_id"]
        if not kb_id:
            maximum = db.execute("SELECT MAX(CAST(SUBSTR(kb_id,4) AS INTEGER)) FROM kb_revisions").fetchone()[0] or 0
            kb_id = f"KB-{maximum + 1:04d}"
            revision, supersedes = 1, None
        else:
            previous = db.execute("SELECT MAX(revision) FROM kb_revisions WHERE kb_id=?", (kb_id,)).fetchone()[0]
            revision, supersedes = previous + 1, f"{kb_id}@r{previous}"
            db.execute("UPDATE kb_revisions SET status='superseded' WHERE kb_id=? AND revision=?", (kb_id, previous))
        db.execute("INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)", (
            kb_id, revision, "active", review["proposed_question"], review["proposed_answer"],
            supersedes, f"review:{review_id}", now,
        ))
        issue = db.execute("SELECT source_id FROM issues WHERE issue_id=?", (review["issue_id"],)).fetchone()
        if issue:
            db.execute("INSERT INTO evidence_links VALUES(?,?,?,?,?,?)",
                       (kb_id, revision, review["issue_id"], issue["source_id"], "active", now))
        audit(db, "review_approved", review_id, {"kb_id": kb_id, "revision": revision, "note": note})
    return {**dict(review), "kb_id": kb_id, "revision": revision}


def export_kb_registry(db: sqlite3.Connection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in db.execute("SELECT * FROM kb_revisions ORDER BY kb_id,revision"):
            record = {key: row[key] for key in ("kb_id", "revision", "status", "supersedes", "provenance")}
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def scan_sensitive(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [name for name, pattern in SENSITIVE_PATTERNS if pattern.search(text)]


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("publish requires a Git commit (git rev-parse HEAD failed)")
    return result.stdout.strip()


def git_tracked_tree_clean(root: Path) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root)
    return result.returncode == 0


def publish(db: sqlite3.Connection, root: Path, version: str, *, embedding_model: str,
            embedding_dimension: int) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("release version must be SemVer, e.g. 1.0.0")
    baseline = root / "output" / "kb_entries_official_v3.jsonl"
    if baseline.exists() and sha256_file(baseline) != "f57b19feb7f6a6e118e7ab48737d54ce2f59d56a70c1f2fea9e274076561c51f":
        raise ValueError("frozen baseline changed; publish refused")
    releases = root / "releases"
    target = releases / version
    if target.exists():
        raise FileExistsError(f"immutable release already exists: {target}")
    pending = db.execute("SELECT COUNT(*) FROM reviews WHERE status='pending' AND relation IN ('conflict','temporal_update','revalidation_required')").fetchone()[0]
    if pending:
        raise ValueError(f"publishability gate blocked by {pending} high-risk pending review(s)")
    active = active_kb(db)
    if not active or active[0]["kb_id"] != "KB-0001":
        raise ValueError("active KB snapshot is invalid")
    frozen = [row for row in active if int(row["kb_id"][3:]) <= 645]
    if len(frozen) != 645 or any(row["kb_id"] != f"KB-{index:04d}" or row["revision"] != 1 for index, row in enumerate(frozen, 1)):
        raise ValueError("frozen KB identity/revision guard failed")
    source_commit = git_commit(root)
    if not git_tracked_tree_clean(root):
        raise RuntimeError("publish requires committed tracked source changes")
    releases.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=releases))
    try:
        chunks = temp / "chunks.jsonl"
        with chunks.open("w", encoding="utf-8") as output:
            for row in active:
                chunk_id = f"{row['kb_id']}@r{row['revision']}"
                output.write(json.dumps({
                    "chunk_id": chunk_id, "kb_id": row["kb_id"], "revision": row["revision"],
                    "status": row["status"], "question": row["question"], "answer": row["answer"],
                    "text": f"问题：{row['question']}\n答案：{row['answer']}",
                    "supersedes": row["supersedes"], "provenance": row["provenance"],
                }, ensure_ascii=False, sort_keys=True) + "\n")
        previous = _current_release(releases)
        previous_rows = {}
        if previous:
            for line in (releases / previous / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
                item = json.loads(line); previous_rows[item["kb_id"]] = item
        current_rows = {row["kb_id"]: row for row in active}
        changes = {
            "added": sorted(set(current_rows) - set(previous_rows)),
            "revised": sorted(k for k in set(current_rows) & set(previous_rows) if current_rows[k]["revision"] != previous_rows[k]["revision"]),
            "removed": sorted(set(previous_rows) - set(current_rows)),
        }
        (temp / "changelog.json").write_text(json.dumps({
            "release_version": version, "previous_release": previous, **changes,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        file_hashes = {name: sha256_file(temp / name) for name in ("chunks.jsonl", "changelog.json")}
        manifest = {
            "schema_version": SCHEMA_VERSION, "release_version": version,
            "generated_at": utc_now(), "chunk_count": len(active),
            "embedding": {"model": embedding_model, "dimension": embedding_dimension, "artifact": "external_regenerable"},
            "source_commit": source_commit, "files": {name: {"sha256": digest} for name, digest in file_hashes.items()},
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_hashes = {**file_hashes, "manifest.json": sha256_file(temp / "manifest.json")}
        (temp / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())), encoding="utf-8")
        validate_release(temp)
        os.replace(temp, target)
        _switch_current(releases, version)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def validate_release(release: Path) -> dict:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    chunks = [json.loads(line) for line in (release / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(chunks) != manifest["chunk_count"]:
        raise ValueError("chunk_count mismatch")
    if len({x["chunk_id"] for x in chunks}) != len(chunks):
        raise ValueError("duplicate chunk_id")
    for name, descriptor in manifest["files"].items():
        if sha256_file(release / name) != descriptor["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
    sensitive = []
    for name in ("chunks.jsonl", "changelog.json", "manifest.json"):
        sensitive.extend(f"{name}:{item}" for item in scan_sensitive(release / name))
    if sensitive:
        raise ValueError("sensitive release content: " + ", ".join(sensitive))
    return manifest


def _current_release(releases: Path) -> str | None:
    current = releases / "current"
    return current.resolve().name if current.is_symlink() and current.exists() else None


def _switch_current(releases: Path, version: str) -> None:
    link = releases / ".current.tmp"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(version, target_is_directory=True)
    os.replace(link, releases / "current")


def rollback(root: Path, version: str | None = None) -> str:
    releases = root / "releases"
    current = _current_release(releases)
    versions = sorted([p.name for p in releases.iterdir() if p.is_dir() and re.fullmatch(r"\d+\.\d+\.\d+", p.name)], key=lambda s: tuple(map(int, s.split("."))))
    if version is None:
        if current not in versions or versions.index(current) == 0:
            raise ValueError("no previous release available")
        version = versions[versions.index(current) - 1]
    target = releases / version
    validate_release(target)
    _switch_current(releases, version)
    return version
