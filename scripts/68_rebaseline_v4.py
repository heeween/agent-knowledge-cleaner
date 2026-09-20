#!/usr/bin/env python3
"""
Step 18 - 正式 KB v4 重立基线（用户决策: 2026-09-20）
====================================================

背景:

- 服务器端 kb-admin 快速修订通道已产出 1.0.5..1.0.8,
  累计修订 26 条、删除 4 条; 这些 r2+ 修订只存在于
  服务器 release, 不在本仓库 registry 中。
- 用户决策: 放弃"645 条全量保留"的增量路线, 改为重立基线:
  - 保留: 服务器端修改过的 26 条 (文本以 1.0.8 为准)
  - 退役: 未修改的 615 条 (用户决策删除)
  - 退役: 服务器端已删除的 4 条 (维持删除)
- 新知识条目从 KB-0646 起, 来源限定 2026-03-01 之后的聊天切片。

本脚本 (确定性, 无 LLM):

1. 对账 服务器 1.0.4 vs 1.0.8 chunks -> revised / removed 集合
2. 生成 output/kb_entries_official_v4.jsonl (26 条)
   与 output/kb_official_v4_contract.json (survivor/retired 契约)
3. 升级本地 registry (先在线备份):
   - survivor: 追加 r+1 (文本=1.0.8, provenance=remote:yj-kb@1.0.8)
   - 其余 619 条: 最新 revision 置为 retired (审计链保留)
4. 过滤视频联动 -> output/video_linkage_v2.jsonl (只留 survivor 引用)

守卫:

- 1.0.4/1.0.8 chunks kb_id 唯一且与 v3 基线 645 ID 完全一致
- revised 条目在 1.0.8 中 revision>1 且 provenance 为 kb-admin
- 本地 registry r1 文本与服务器 1.0.4 逐字一致 (干净起点)
- 幂等: 重复运行不产生重复 revision / 重复 retirement
- 不修改任何冻结文件 (v3 基线/漏斗产物只读)

输出 (全部新文件):

    output/kb_entries_official_v4.jsonl
    output/kb_official_v4_contract.json
    output/video_linkage_v2.jsonl
    .state/registry.backup-<ts>.sqlite3
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
STATE_DIR = ROOT_DIR / ".state"

BASELINE_V3 = OUTPUT_DIR / "kb_entries_official_v3.jsonl"
REMOTE_104 = OUTPUT_DIR / "remote_release_1.0.4_chunks.jsonl"
REMOTE_108 = OUTPUT_DIR / "remote_release_1.0.8_chunks.jsonl"
BASELINE_V4 = OUTPUT_DIR / "kb_entries_official_v4.jsonl"
CONTRACT_V4 = OUTPUT_DIR / "kb_official_v4_contract.json"
LINKAGE_V1 = OUTPUT_DIR / "video_linkage.jsonl"
LINKAGE_V2 = OUTPUT_DIR / "video_linkage_v2.jsonl"
REGISTRY = STATE_DIR / "registry.sqlite3"

PROVENANCE_REMOTE = "remote:yj-kb@1.0.8"


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def chunk_text(chunk):
    return (chunk["question"], chunk["answer"])


def load_chunk_map(path, label):
    records = load_jsonl(path)
    mapping = {}
    for record in records:
        kb_id = record["kb_id"]
        if kb_id in mapping:
            raise SystemExit(f"守卫失败: {label} 中 kb_id 重复: {kb_id}")
        mapping[kb_id] = record
    return mapping


def diff_releases(old_map, new_map):
    revised = {
        kb_id: new_map[kb_id]
        for kb_id in sorted(set(old_map) & set(new_map))
        if chunk_text(old_map[kb_id]) != chunk_text(new_map[kb_id])
    }
    removed = sorted(set(old_map) - set(new_map))
    added = sorted(set(new_map) - set(old_map))
    return revised, removed, added


def registry_backup():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = STATE_DIR / f"registry.backup-{timestamp}.sqlite3"
    source = sqlite3.connect(str(REGISTRY))
    target_db = sqlite3.connect(str(target))
    with target_db:
        source.backup(target_db)
    target_db.close()
    source.close()
    return target


def upgrade_registry(registry, revised, retire_ids):
    """survivor 追加 r+1; 其余 619 条最新 revision 置 retired。幂等。"""
    db = sqlite3.connect(str(registry))
    db.row_factory = sqlite3.Row
    imported, retired, skipped = [], [], []

    with db:
        for kb_id, chunk in revised.items():
            latest = db.execute(
                "SELECT * FROM kb_revisions WHERE kb_id=? ORDER BY revision DESC LIMIT 1",
                (kb_id,),
            ).fetchone()
            if latest is None:
                raise SystemExit(f"守卫失败: registry 缺少 {kb_id}")
            if latest["question"] == chunk["question"] and latest["answer"] == chunk["answer"]:
                skipped.append(kb_id)
                continue
            if latest["status"] != "active":
                raise SystemExit(f"守卫失败: {kb_id}@r{latest['revision']} 状态为 {latest['status']}")
            new_revision = latest["revision"] + 1
            db.execute(
                "UPDATE kb_revisions SET status='superseded' WHERE kb_id=? AND revision=?",
                (kb_id, latest["revision"]),
            )
            db.execute(
                "INSERT INTO kb_revisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    kb_id, new_revision, "active", chunk["question"], chunk["answer"],
                    f"{kb_id}@r{latest['revision']}", PROVENANCE_REMOTE, utc_now(),
                ),
            )
            db.execute(
                "INSERT INTO audit_events VALUES(?,?,?,?,?)",
                (
                    f"AUD-{kb_id}-{new_revision}", "remote_revision_imported", f"{kb_id}@r{new_revision}",
                    json.dumps({
                        "source_release": "1.0.8", "provenance": PROVENANCE_REMOTE,
                        "superseded_revision": latest["revision"],
                    }, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
            imported.append(f"{kb_id}@r{new_revision}")

        for kb_id in retire_ids:
            latest = db.execute(
                "SELECT * FROM kb_revisions WHERE kb_id=? ORDER BY revision DESC LIMIT 1",
                (kb_id,),
            ).fetchone()
            if latest is None:
                raise SystemExit(f"守卫失败: registry 缺少 {kb_id}")
            if latest["status"] == "retired":
                continue
            if latest["status"] != "active":
                raise SystemExit(f"守卫失败: {kb_id}@r{latest['revision']} 状态为 {latest['status']}")
            db.execute(
                "UPDATE kb_revisions SET status='retired' WHERE kb_id=? AND revision=?",
                (kb_id, latest["revision"]),
            )
            db.execute(
                "INSERT INTO audit_events VALUES(?,?,?,?,?)",
                (
                    f"AUD-RETIRE-{kb_id}", "entry_retired_rebaseline", kb_id,
                    json.dumps({"reason": "rebaseline_v4_user_decision"}, ensure_ascii=False),
                    utc_now(),
                ),
            )
            retired.append(kb_id)
    db.close()
    return imported, retired, skipped


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    old_map = load_chunk_map(REMOTE_104, "1.0.4")
    new_map = load_chunk_map(REMOTE_108, "1.0.8")
    baseline_v3 = load_jsonl(BASELINE_V3)
    v3_ids = [record["kb_id"] for record in baseline_v3]

    if sorted(old_map) != v3_ids:
        raise SystemExit("守卫失败: 1.0.4 kb_id 集合与 v3 基线不一致")
    unknown = sorted(set(new_map) - set(v3_ids))
    if unknown:
        raise SystemExit(f"守卫失败: 1.0.8 出现 v3 基线之外的条目: {unknown}")

    revised, removed, added = diff_releases(old_map, new_map)
    if added:
        raise SystemExit(f"守卫失败: 1.0.8 出现未知新增条目: {added}")
    for kb_id, chunk in revised.items():
        if chunk["revision"] <= 1 or "kb-admin" not in chunk["provenance"]:
            raise SystemExit(
                f"守卫失败: {kb_id} 文本变化但 revision/provenance 异常: "
                f"r{chunk['revision']} {chunk['provenance']}"
            )
    unmodified = sorted(set(v3_ids) - set(revised) - set(removed))
    retire_ids = unmodified + removed
    print(f"对账结果: revised={len(revised)} removed={len(removed)} unmodified={len(unmodified)}")

    # 干净起点校验: 本地 registry r1 必须与服务器 1.0.4 逐字一致
    db = sqlite3.connect(str(REGISTRY))
    db.row_factory = sqlite3.Row
    for kb_id, chunk in old_map.items():
        row = db.execute(
            "SELECT question, answer FROM kb_revisions WHERE kb_id=? AND revision=1", (kb_id,)
        ).fetchone()
        if row is None or (row["question"], row["answer"]) != chunk_text(chunk):
            raise SystemExit(f"守卫失败: 本地 registry {kb_id}@r1 与服务器 1.0.4 不一致")
    db.close()
    print("干净起点校验通过: 本地 r1 == 服务器 1.0.4 (645 条逐字一致)")

    # 1. v4 基线
    with open(BASELINE_V4, "w", encoding="utf-8") as f:
        for kb_id in sorted(revised):
            chunk = revised[kb_id]
            f.write(json.dumps({
                "kb_id": kb_id,
                "question": chunk["question"],
                "answer": chunk["answer"],
                "provenance": PROVENANCE_REMOTE,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    baseline_sha = sha256_file(BASELINE_V4)
    print(f"写出 {BASELINE_V4.name}: {len(revised)} 条, sha256={baseline_sha}")

    # 2. 契约
    contract = {
        "baseline": BASELINE_V4.name,
        "baseline_sha256": baseline_sha,
        "source_release": "1.0.8",
        "survivor_ids": sorted(revised),
        "removed_ids": removed,
        "retired_ids": retire_ids,
        "generated_at": utc_now(),
        "decision": "用户决策: 未被 yj-kb 修改的 615 条退役, 服务器已删的 4 条维持删除, "
                    "新条目从 KB-0646 起, 来源限定 2026-03-01 之后聊天",
    }
    with open(CONTRACT_V4, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"写出 {CONTRACT_V4.name}: survivor={len(revised)} retired={len(retire_ids)}")

    # 3. registry 升级
    backup = registry_backup()
    print(f"registry 已备份: {backup.name}")
    imported, retired_now, skipped = upgrade_registry(REGISTRY, revised, retire_ids)
    print(f"registry 升级: 导入 r+1 {len(imported)} 条, 退役 {len(retired_now)} 条, 幂等跳过 {len(skipped)} 条")

    # 4. 视频联动过滤
    if LINKAGE_V1.exists():
        rows = load_jsonl(LINKAGE_V1)
        kept = [row for row in rows if row["kb_id"] in revised]
        dropped = len(rows) - len(kept)
        with open(LINKAGE_V2, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"写出 {LINKAGE_V2.name}: 保留 {len(kept)} 条, 剔除指向退役条目的 {dropped} 条")

    print("\n后续步骤: core.py 的 v4 sha 常量更新为上面输出的 sha256 后再运行测试")


if __name__ == "__main__":
    sys.exit(main())
