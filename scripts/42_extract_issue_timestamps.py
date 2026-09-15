#!/usr/bin/env python3
"""
Step 12.1 - 真实 timestamp 提取
================================

为全部 4095 个 extracted issue 建立
issue_key → 真实聊天时间 的 deterministic 映射。

映射路径 (PROJECT_HANDOFF 第 18 节):

    issue_key
      → source_candidate_id
        (来自 question_embeddings.jsonl,
         与 09 的 make_issue_key 同源)
      → extracted_issues.jsonl
        (source_message_indexes,
         1-based messages.jsonl 行号,
         与 06 的 enumerate(f, start=1) 同构)
      → messages.jsonl
        (timestamp)

索引空间已验证:

    source_message_index = messages.jsonl 物理行号 (1-based)。
    验证用例 ISSUE-CAND-000585#ROW-00638:
    6966/6967/6968/6972 精确命中
    "为什么还会在24小时回访" 的问答原话。

不调用 LLM。
不修改任何 frozen 输入。

输出:

    output/issue_timestamps.jsonl
    output/issue_timestamps.xlsx

    sheets:
      summary             总量 / 覆盖率 / 校验结果
      all_issues          4095 条完整时间映射
      temporal_blockers   4 个 material temporal cluster
      candidate_sources   v3 candidate 的全部 source keys
      mismatches          解析失败 / 文档不一致
      collisions          同 candidate 内规范化问题重复
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT_DIR / "output"

MESSAGES_FILE = OUTPUT_DIR / "messages.jsonl"

CANDIDATES_FILE = OUTPUT_DIR / "issue_candidates.jsonl"

EXTRACTED_FILE = OUTPUT_DIR / "extracted_issues.jsonl"

EMBEDDINGS_FILE = OUTPUT_DIR / "question_embeddings.jsonl"

STRUCTURE_FILE = (
    OUTPUT_DIR
    / "mixed_structure_classifications_v3_1.xlsx"
)

CANDIDATE_V3 = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v3.jsonl"
)

OUTPUT_JSONL = OUTPUT_DIR / "issue_timestamps.jsonl"

OUTPUT_XLSX = OUTPUT_DIR / "issue_timestamps.xlsx"

DRY_RUN = "--dry-run" in sys.argv

# 4 个 material temporal cluster (frozen v3.1)
TEMPORAL_CLUSTER_IDS = [
    "QCLUSTER-0008",
    "QCLUSTER-0074",
    "QCLUSTER-0139",
    "QCLUSTER-0194",
]


# ============================================================
# 工具
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    return str(value).strip()


def load_jsonl(path):

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def parse_ts(value):

    text = clean_text(value)

    if not text:
        return None

    try:
        return datetime.strptime(
            text, "%Y-%m-%d %H:%M:%S"
        )

    except ValueError:
        return None


def month_key(value):

    ts = parse_ts(value)

    if ts is None:
        return ""

    return ts.strftime("%Y-%m")


# ============================================================
# 输入加载
# ============================================================

def load_messages_by_line():

    """
    复现 script 06 的枚举:
    enumerate(f, start=1)，
    物理行号 (1-based) → message。
    空行跳过但计数器继续。
    """

    by_line = {}

    blank = 0

    with MESSAGES_FILE.open(
        "r", encoding="utf-8"
    ) as f:

        for source_index, line in enumerate(
            f, start=1
        ):

            if not line.strip():
                blank += 1
                continue

            by_line[source_index] = json.loads(line)

    return by_line, blank


def load_candidates_by_id():

    return {
        clean_text(row.get("issue_id")): row
        for row in load_jsonl(CANDIDATES_FILE)
    }


def load_extracted_records():

    """
    (source_candidate_id, question_normalized)
    → 按文件顺序的记录列表。
    """

    groups = {}

    for order, row in enumerate(
        load_jsonl(EXTRACTED_FILE)
    ):

        key = (
            clean_text(row.get("source_candidate_id")),
            clean_text(row.get("question_normalized")),
        )

        row["_file_order"] = order

        groups.setdefault(key, []).append(row)

    return groups


def load_issue_keys():

    return load_jsonl(EMBEDDINGS_FILE)


def load_temporal_focus():

    df = pd.read_excel(
        STRUCTURE_FILE,
        sheet_name="evidence_roles",
    )

    focus = df[
        df["cluster_id"].isin(TEMPORAL_CLUSTER_IDS)
    ][["cluster_id", "issue_key"]].copy()

    return {
        clean_text(row["issue_key"]): clean_text(
            row["cluster_id"]
        )
        for row in focus.to_dict(orient="records")
    }


def load_candidate_source_keys():

    keys = {}

    for entry in load_jsonl(CANDIDATE_V3):

        cluster_id = clean_text(
            entry.get("cluster_id")
        )

        for key in entry.get(
            "source_issue_keys"
        ) or []:
            keys.setdefault(
                clean_text(key), []
            ).append(cluster_id)

    return keys


# ============================================================
# 主提取
# ============================================================

def extract():

    messages_by_line, blank_lines = (
        load_messages_by_line()
    )

    candidates = load_candidates_by_id()

    extracted_groups = load_extracted_records()

    issue_keys = load_issue_keys()

    temporal_focus = load_temporal_focus()

    candidate_sources = (
        load_candidate_source_keys()
    )

    used_records = set()

    results = []

    mismatches = []

    collisions = []

    unmapped = []

    for emb in issue_keys:

        issue_key = clean_text(
            emb.get("issue_key")
        )

        candidate_id = clean_text(
            emb.get("source_candidate_id")
        )

        question = clean_text(
            emb.get("question_normalized")
        )

        record_key = (candidate_id, question)

        group = extracted_groups.get(record_key, [])

        if not group:

            unmapped.append(
                {
                    "issue_key": issue_key,
                    "reason": "extracted_issues 中无匹配记录",
                }
            )

            continue

        record = None

        for item in group:

            if id(item) not in used_records:
                record = item
                break

        if record is None:

            # 同 candidate 同规范化问题重复:
            # 时间戳取并集, 並记录 collision
            record = group[0]

            collisions.append(
                {
                    "issue_key": issue_key,
                    "source_candidate_id": candidate_id,
                    "question_normalized": question,
                    "duplicate_count": len(group),
                }
            )

        used_records.add(id(record))

        block = candidates.get(candidate_id)

        if block is None:

            mismatches.append(
                {
                    "issue_key": issue_key,
                    "type": "CANDIDATE_NOT_FOUND",
                    "detail": candidate_id,
                }
            )

            continue

        document_id = clean_text(
            block.get("document_id")
        )

        indexes = [
            int(i)
            for i in record.get(
                "source_message_indexes"
            ) or []
        ]

        message_rows = []

        for index in indexes:

            message = messages_by_line.get(index)

            if message is None:

                mismatches.append(
                    {
                        "issue_key": issue_key,
                        "type": "MESSAGE_INDEX_NOT_FOUND",
                        "detail": str(index),
                    }
                )

                continue

            message_doc = clean_text(
                message.get("document_id")
            )

            if message_doc != document_id:

                mismatches.append(
                    {
                        "issue_key": issue_key,
                        "type": "DOCUMENT_MISMATCH",
                        "detail": (
                            f"index={index} "
                            f"message_doc={message_doc} "
                            f"candidate_doc={document_id}"
                        ),
                    }
                )

                continue

            message_rows.append(
                {
                    "index": index,
                    "timestamp": clean_text(
                        message.get("timestamp")
                    ),
                    "line_number": message.get(
                        "line_number"
                    ),
                    "speaker": clean_text(
                        message.get("speaker")
                    ),
                }
            )

        times = [
            parse_ts(row["timestamp"])
            for row in message_rows
        ]

        times = [t for t in times if t]

        block_start = clean_text(
            block.get("start_time")
        )

        block_end = clean_text(
            block.get("end_time")
        )

        fallback = False

        if times:

            issue_time_min = min(times).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            issue_time_max = max(times).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        else:

            fallback = True

            issue_time_min = block_start

            issue_time_max = block_end

        results.append(
            {
                "issue_key": issue_key,
                "source_candidate_id": candidate_id,
                "question_normalized": question,
                "document_id": document_id,
                "source_filename": clean_text(
                    block.get("source_filename")
                ),
                "block_start_time": block_start,
                "block_end_time": block_end,
                "issue_message_count": len(
                    message_rows
                ),
                "issue_time_min": issue_time_min,
                "issue_time_max": issue_time_max,
                "time_fallback_to_block": fallback,
                "issue_date": (
                    issue_time_min[:10]
                    if issue_time_min
                    else ""
                ),
                "issue_month": month_key(
                    issue_time_min
                ),
                "resolution": clean_text(
                    record.get("resolution")
                ),
                "temporal_status": clean_text(
                    record.get("temporal_status")
                ),
                "knowledge_value": clean_text(
                    record.get("knowledge_value")
                ),
                "useful_for_knowledge_base": bool(
                    record.get(
                        "useful_for_knowledge_base"
                    )
                ),
                "temporal_cluster": (
                    temporal_focus.get(issue_key, "")
                ),
                "candidate_entry": ",".join(
                    candidate_sources.get(issue_key, [])
                ),
                "messages": message_rows,
            }
        )

    stats = {
        "issue_keys_total": len(issue_keys),
        "issue_keys_mapped": len(results),
        "issue_keys_unmapped": len(unmapped),
        "messages_indexed": len(messages_by_line),
        "messages_blank_lines": blank_lines,
        "mismatches": len(mismatches),
        "collisions": len(collisions),
        "fallback_to_block": sum(
            1
            for row in results
            if row["time_fallback_to_block"]
        ),
    }

    return (
        results,
        stats,
        mismatches,
        collisions,
        unmapped,
    )


# ============================================================
# 输出
# ============================================================

def export_excel(
    results,
    stats,
    mismatches,
    collisions,
    unmapped,
):

    summary_df = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in stats.items()
        ]
    )

    all_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "messages"
            }
            for row in results
        ]
    )

    focus_df = all_df[
        all_df["temporal_cluster"] != ""
    ].copy()

    entry_df = all_df[
        all_df["candidate_entry"] != ""
    ].copy()

    mismatch_df = pd.DataFrame(mismatches)

    collision_df = pd.DataFrame(collisions)

    unmapped_df = pd.DataFrame(unmapped)

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        summary_df.to_excel(
            writer, sheet_name="summary", index=False
        )

        all_df.to_excel(
            writer, sheet_name="all_issues", index=False
        )

        focus_df.to_excel(
            writer,
            sheet_name="temporal_blockers",
            index=False,
        )

        entry_df.to_excel(
            writer,
            sheet_name="candidate_sources",
            index=False,
        )

        mismatch_df.to_excel(
            writer, sheet_name="mismatches", index=False
        )

        collision_df.to_excel(
            writer, sheet_name="collisions", index=False
        )

        unmapped_df.to_excel(
            writer, sheet_name="unmapped", index=False
        )


def print_focus(results):

    by_key = {
        row["issue_key"]: row for row in results
    }

    print()
    print("4 个 material temporal cluster 的时间分布:")
    print("-" * 70)

    focus = [
        row
        for row in results
        if row["temporal_cluster"]
    ]

    for row in sorted(
        focus,
        key=lambda r: (
            r["temporal_cluster"],
            r["issue_time_min"],
        ),
    ):

        print(
            f"  {row['temporal_cluster']:15} "
            f"{row['issue_key']:32} "
            f"{row['issue_time_min']} → "
            f"{row['issue_time_max']} "
            f"({row['temporal_status']}/"
            f"{row['resolution']})"
        )

    print()
    print("v3 candidate sources 的时间分布:")
    print("-" * 70)

    entry = [
        row
        for row in results
        if row["candidate_entry"]
    ]

    for row in sorted(
        entry,
        key=lambda r: (
            r["candidate_entry"],
            r["issue_time_min"],
        ),
    ):

        print(
            f"  {row['candidate_entry']:15} "
            f"{row['issue_key']:32} "
            f"{row['issue_time_min']} → "
            f"{row['issue_time_max']}"
        )


def main():

    print("=" * 70)
    print("Step 12.1 - Issue Timestamp Extraction")
    print("=" * 70)

    results, stats, mismatches, collisions, unmapped = (
        extract()
    )

    print()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if stats["issue_keys_mapped"] != (
        stats["issue_keys_total"]
        - stats["issue_keys_unmapped"]
    ):
        raise RuntimeError(
            "mapped 数与 total-unmapped 不一致"
        )

    if DRY_RUN:

        print()
        print("DRY RUN 结束，未写任何输出。")
        print_focus(results)
        return

    OUTPUT_JSONL.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in results
        )
        + "\n",
        encoding="utf-8",
    )

    export_excel(
        results,
        stats,
        mismatches,
        collisions,
        unmapped,
    )

    print()
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"Excel: {OUTPUT_XLSX}")

    print_focus(results)

    print()
    print("下一步 (Step 12.2): temporal validity 处理")
    print("基于本步骤时间戳判断 4 个 cluster 的")
    print("current vs historical。")


if __name__ == "__main__":
    main()
