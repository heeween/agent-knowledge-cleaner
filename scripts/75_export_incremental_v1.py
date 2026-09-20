#!/usr/bin/env python3
"""
Step 18.7 - 导出增量正式条目并导入 registry
============================================

输入:

    output/funnel_v2_publishability.jsonl  (gate 全量审计)
    output/funnel_v2_dedup_survivors.jsonl (QA 原文)

守卫 (与 57 口径一致):

- 只收 verdict=publish
- 无 pre_flags (PRIVACY_PHONE / ANSWER_TOO_SHORT 不得出现)
- 答案非空且 >= 20 字
- 手机号全量复扫 (0 允许)
- kb_id 从 KB-0646 起连续 (跳过退役段, 与
  decide_review 的分配规则一致)

输出 / 动作:

    output/kb_entries_incremental_v1.jsonl
    registry 导入: kb_revisions r1 active,
    provenance=funnel_v2:publish, 幂等 (重复运行跳过已导入)
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
STATE_DIR = ROOT_DIR / ".state"
REGISTRY = STATE_DIR / "registry.sqlite3"

GATE_FILE = OUTPUT_DIR / "funnel_v2_publishability.jsonl"
SURVIVORS_FILE = OUTPUT_DIR / "funnel_v2_dedup_survivors.jsonl"
EXPORT_FILE = OUTPUT_DIR / "kb_entries_incremental_v1.jsonl"

PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
FIRST_NEW_ID = 646
PROVENANCE = "funnel_v2:publish"


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main():
    gate = {row["issue_key"]: row for row in load_jsonl(GATE_FILE)}
    source = {row["issue_key"]: row for row in load_jsonl(SURVIVORS_FILE)}

    publishable = []
    for issue_key, verdict in gate.items():
        row = source.get(issue_key)
        if row is None:
            raise SystemExit(f"守卫失败: {issue_key} 在 QA 原文缺失")
        if verdict["verdict"] != "publish":
            continue
        if verdict["pre_flags"]:
            raise SystemExit(f"守卫失败: {issue_key} verdict=publish 但带前置标记 {verdict['pre_flags']}")
        if len(row["answer"]) < 20 or not row["answer"].strip():
            raise SystemExit(f"守卫失败: {issue_key} 答案过短")
        if PHONE_PATTERN.search(row["question"] + "\n" + row["answer"]):
            raise SystemExit(f"守卫失败: {issue_key} 含手机号")
        publishable.append((issue_key, row, verdict))

    publishable.sort(key=lambda item: item[0])
    existing = load_jsonl(EXPORT_FILE)
    if existing:
        last_id = int(existing[-1]["kb_id"][3:])
        if last_id < FIRST_NEW_ID:
            raise SystemExit(f"守卫失败: 已有导出文件首 id 异常: {existing[-1]['kb_id']}")
        known = {row["source_issue_key"] for row in existing}
        new_rows = [
            (issue_key, row, verdict)
            for issue_key, row, verdict in publishable
            if issue_key not in known
        ]
        next_id = last_id + 1
    else:
        new_rows = publishable
        next_id = FIRST_NEW_ID

    exported = []
    with open(EXPORT_FILE, "a", encoding="utf-8") as f:
        for issue_key, row, verdict in new_rows:
            record = {
                "kb_id": f"KB-{next_id:04d}",
                "question": row["question"],
                "answer": row["answer"],
                "source_issue_key": issue_key,
                "source_filename": row["source_filename"],
                "issue_date": row["issue_date"],
                "gate_reason": verdict["reason"],
                "provenance": PROVENANCE,
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            exported.append(record)
            next_id += 1
    print(f"导出 {len(exported)} 条新条目 -> {EXPORT_FILE.name} (累计 {len(existing) + len(exported)})")

    if not exported:
        return
    db = sqlite3.connect(str(REGISTRY))
    db.row_factory = sqlite3.Row
    imported = 0
    with db:
        for record in exported:
            exists = db.execute(
                "SELECT 1 FROM kb_revisions WHERE kb_id=? AND revision=1", (record["kb_id"],)
            ).fetchone()
            if exists:
                continue
            current = db.execute("SELECT question, answer FROM kb_revisions WHERE kb_id=?", (record["kb_id"],)).fetchone()
            if current is not None:
                raise SystemExit(f"守卫失败: {record['kb_id']} 已被占用")
            db.execute(
                "INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)",
                (record["kb_id"], 1, "active", record["question"], record["answer"], None,
                 PROVENANCE, utc_now()),
            )
            db.execute(
                "INSERT INTO audit_events VALUES(?,?,?,?,?)",
                (f"AUD-INC-{record['kb_id']}", "incremental_entry_imported", record["kb_id"],
                 json.dumps({"source_issue_key": record["source_issue_key"],
                             "source_filename": record["source_filename"]},
                            ensure_ascii=False, sort_keys=True),
                 utc_now()),
            )
            imported += 1
    db.close()
    print(f"registry 导入 {imported} 条 (KB-{FIRST_NEW_ID} 起)")


if __name__ == "__main__":
    main()
