#!/usr/bin/env python3
"""
Step 14.2 - 正式 KB 导出 v2
===========================

把正式 KB 的全部 76 条导出为统一文件
（74 条既有冻结 + Step 14 新 publish 的 2 条 mixed）:

    safe         69  (kb_source_quality.xlsx approved)
    regenerated   2  (regenerated_publishability.xlsx approved)
    multicause    3  (kb_entries_multicause_publishable.xlsx)
    mixed         2  (mixed_final_publishability_v1 verdict=publish:
                      QCLUSTER-0004 / QCLUSTER-0033)

规则:

- 全部内容逐字来自冻结输出, 不改写任何字段
- 每条带 provenance (来源文件 + 判定依据)
- mixed 条目额外携带 gate 溯源
  (verdict / reason / grounding_status / model)
- 只新建文件, 不覆盖任何既有导出
- 0008 (reject) 与 7 条 manual_review 不进入

输出:

    output/kb_entries_official_v2.jsonl
    output/kb_entries_official_v2.xlsx

用法:

    .venv/bin/python scripts/51_export_official_kb_v2.py
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

SAFE_FILE = OUTPUT_DIR / "kb_entries_safe.jsonl"
QUALITY_XLSX = OUTPUT_DIR / "kb_source_quality.xlsx"
REGEN_FILE = OUTPUT_DIR / "kb_entries_regenerated.jsonl"
REGEN_PUB_XLSX = (
    OUTPUT_DIR / "regenerated_publishability.xlsx"
)
MULTI_XLSX = (
    OUTPUT_DIR / "kb_entries_multicause_publishable.xlsx"
)
CANDIDATE_FILE = (
    OUTPUT_DIR / "kb_entries_mixed_candidate_v5.jsonl"
)
GATE_FILE = (
    OUTPUT_DIR / "mixed_final_publishability_v1.jsonl"
)

OUTPUT_JSONL = OUTPUT_DIR / "kb_entries_official_v2.jsonl"
OUTPUT_XLSX = OUTPUT_DIR / "kb_entries_official_v2.xlsx"


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    # ---------- Part 1: safe 69 ----------

    approved_safe = pd.read_excel(
        QUALITY_XLSX, sheet_name="approved"
    )
    approved_safe_ids = set(
        approved_safe["cluster_id"].astype(str)
    )

    safe_all = load_jsonl(SAFE_FILE)
    safe_entries = [
        record for record in safe_all
        if record["cluster_id"] in approved_safe_ids
    ]

    # ---------- Part 2: regenerated 2 ----------

    approved_regen = pd.read_excel(
        REGEN_PUB_XLSX, sheet_name="approved"
    )
    approved_regen_ids = set(
        approved_regen["cluster_id"].astype(str)
    )

    regen_all = load_jsonl(REGEN_FILE)
    regen_entries = [
        record for record in regen_all
        if record["cluster_id"] in approved_regen_ids
    ]

    # ---------- Part 3: multicause 3 ----------

    multi_df = pd.read_excel(
        MULTI_XLSX, sheet_name="kb_entries"
    )
    multi_records = multi_df.to_dict(
        orient="records"
    )

    # ---------- Part 4: mixed 2 ----------

    gate_records = {
        record["cluster_id"]: record
        for record in load_jsonl(GATE_FILE)
    }
    publish_ids = {
        cluster_id
        for cluster_id, record in gate_records.items()
        if record["verdict_final"] == "publish"
    }

    candidates = {
        record["cluster_id"]: record
        for record in load_jsonl(CANDIDATE_FILE)
    }

    mixed_entries = []

    for cluster_id in sorted(publish_ids):
        entry = candidates[cluster_id]
        gate = gate_records[cluster_id]

        mixed_entries.append({
            "cluster_id": cluster_id,
            "canonical_question": entry.get(
                "canonical_question"
            ),
            "summary_answer": entry.get(
                "summary_answer"
            ),
            "units": entry.get("units") or [],
            "limitations": entry.get("limitations")
            or [],
            "knowledge_structure": entry.get(
                "knowledge_structure"
            ),
            "generation_mode": entry.get(
                "generation_mode"
            ),
            "source_issue_keys": entry.get(
                "source_issue_keys"
            ),
            "merged_from_clusters": entry.get(
                "merged_from_clusters"
            ),
            "temporal_validity": entry.get(
                "temporal_validity"
            ),
            "gate_verdict": gate["verdict_final"],
            "gate_reason": gate["reason"],
            "gate_model": gate["model"],
            "grounding_status": gate[
                "grounding_status"
            ],
        })

    # ---------- 守卫 ----------

    errors = []

    expected = {
        "safe": 69,
        "regenerated": 2,
        "multicause": 3,
        "mixed": 2,
    }

    actual = {
        "safe": len(safe_entries),
        "regenerated": len(regen_entries),
        "multicause": len(multi_records),
        "mixed": len(mixed_entries),
    }

    if actual != expected:
        errors.append(
            f"数量不符: expected={expected} "
            f"actual={actual}"
        )

    seen = {}

    for part, entries in (
        ("safe", safe_entries),
        ("regenerated", regen_entries),
        ("multicause", multi_records),
        ("mixed", mixed_entries),
    ):
        for entry in entries:
            cluster_id = str(entry["cluster_id"])

            if cluster_id in seen:
                errors.append(
                    f"cluster_id 重复: {cluster_id} "
                    f"在 {seen[cluster_id]} 与 {part}"
                )

            seen[cluster_id] = part

    if "QCLUSTER-0008" in seen:
        errors.append("0008 (reject) 不应进入正式 KB")

    total = sum(actual.values())

    if total != 76:
        errors.append(
            f"总数 {total} != 76"
        )

    if errors:
        for error in errors:
            print(f"导出守卫失败: {error}")

        raise SystemExit(1)

    # ---------- 统一信封 ----------

    envelope = []
    kb_index = 0

    part_order = [
        ("safe", safe_entries,
         "kb_source_quality.xlsx#approved"),
        ("regenerated", regen_entries,
         "regenerated_publishability.xlsx#approved"),
        ("multicause", multi_records,
         "kb_entries_multicause_publishable.xlsx#kb_entries"),
        ("mixed", mixed_entries,
         "mixed_final_publishability_v1.jsonl#publish"),
    ]

    for part, entries, provenance in part_order:
        for entry in entries:
            kb_index += 1

            cluster_id = str(entry["cluster_id"])

            if part == "multicause":
                question = entry.get("question")
                answer = entry.get("answer")
                sources = entry.get(
                    "source_issue_keys"
                )
            elif part == "mixed":
                question = entry.get(
                    "canonical_question"
                )
                answer = entry.get(
                    "summary_answer"
                )
                sources = entry.get(
                    "source_issue_keys"
                )
            else:
                question = entry.get(
                    "canonical_question"
                )
                answer = entry.get("answer")
                sources = entry.get(
                    "source_issue_keys"
                )

            envelope.append({
                "kb_id": f"KB-{kb_index:04d}",
                "part": part,
                "cluster_id": cluster_id,
                "question": question,
                "answer": answer,
                "source_issue_keys": sources,
                "provenance": provenance,
                "payload": entry,
            })

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in envelope:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    # ---------- xlsx ----------

    flat_rows = []

    for record in envelope:
        flat_rows.append({
            "kb_id": record["kb_id"],
            "part": record["part"],
            "cluster_id": record["cluster_id"],
            "question": record["question"],
            "answer": record["answer"],
            "provenance": record["provenance"],
        })

    summary = {
        "official_kb_version": "v2",
        "total": total,
        "safe": actual["safe"],
        "regenerated": actual["regenerated"],
        "multicause": actual["multicause"],
        "mixed_new": actual["mixed"],
        "previous_official_count": 74,
        "mixed_publish_ids": sorted(publish_ids),
        "excluded": (
            "0008 reject; "
            "7 manual_review 保留 candidate 身份"
        ),
    }

    mixed_flat = []

    for entry in mixed_entries:
        mixed_flat.append({
            "cluster_id": entry["cluster_id"],
            "canonical_question": entry[
                "canonical_question"
            ],
            "summary_answer": entry[
                "summary_answer"
            ],
            "units": json.dumps(
                entry["units"], ensure_ascii=False
            ),
            "limitations": json.dumps(
                entry["limitations"],
                ensure_ascii=False,
            ),
            "merged_from_clusters": entry[
                "merged_from_clusters"
            ],
            "gate_verdict": entry["gate_verdict"],
            "gate_reason": entry["gate_reason"],
            "gate_model": entry["gate_model"],
            "grounding_status": entry[
                "grounding_status"
            ],
        })

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(flat_rows).to_excel(
            writer, sheet_name="all_entries", index=False
        )
        pd.DataFrame(safe_entries).to_excel(
            writer, sheet_name="safe_entries", index=False
        )
        pd.DataFrame(regen_entries).to_excel(
            writer,
            sheet_name="regenerated_entries",
            index=False,
        )
        pd.DataFrame(multi_records).to_excel(
            writer,
            sheet_name="multicause_entries",
            index=False,
        )
        pd.DataFrame(mixed_flat).to_excel(
            writer,
            sheet_name="mixed_entries",
            index=False,
        )

    print("=" * 60)
    print("正式 KB v2 导出完成（Step 14.2）")
    print("=" * 60)
    print(f"safe:        {actual['safe']}")
    print(f"regenerated: {actual['regenerated']}")
    print(f"multicause:  {actual['multicause']}")
    print(f"mixed(新):   {actual['mixed']} "
          f"→ {sorted(publish_ids)}")
    print(f"总计:        {total}")
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
