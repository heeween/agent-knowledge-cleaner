#!/usr/bin/env python3
"""
Step 15.2 - Singleton funnel Layer A + B
========================================

对 3598 条 singleton issues 做 deterministic 硬过滤
（PROJECT_HANDOFF 19 / Step 15 设计）:

Layer A (质量硬过滤):

    A1  confidence < 0.7
    A2  knowledge_value == low
    A3  answer < 20 字
    A4  temporal_status == future_plan
    A5  temporal_status == historical

Layer B (时效, 沿用 12.2 先例):

    B1  last evidence date < 2025-08-06
        -> STALE_NO_RECENT_CONFIRMATION
        (未来可凭正式文档佐证救回,
         全部留审计名单)

红线:

- 不要求 resolution == resolved
  (unresolved / partial 全部进入后续 LLM 层)
- feature_request 不在 A/B 硬排,
  送 LLM 层识别"当前不支持X"型稳定知识

singleton 名单重建方式:

- 与 scripts/12 完全同源的 make_issue_key
  (extracted_issues.xlsx 行序 + source_candidate_id
   + issue_seq)
- 减去 question_clusters.xlsx#cluster_members 的 497 条
- 重建结果必须 = 3598, 与冻结 summary 一致

输出 (不修改任何冻结输入):

    output/singleton_funnel_v1.jsonl     (全部 3598 条审计)
    output/singleton_funnel_v1.xlsx
    output/singleton_survivors_ab_v1.jsonl (A+B 幸存者)
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

ISSUE_XLSX = OUTPUT_DIR / "extracted_issues.xlsx"
CLUSTER_MEMBERS_XLSX = (
    OUTPUT_DIR / "question_clusters.xlsx"
)
TIMESTAMPS_FILE = OUTPUT_DIR / "issue_timestamps.jsonl"

FUNNEL_JSONL = OUTPUT_DIR / "singleton_funnel_v1.jsonl"
FUNNEL_XLSX = OUTPUT_DIR / "singleton_funnel_v1.xlsx"
SURVIVORS_JSONL = (
    OUTPUT_DIR / "singleton_survivors_ab_v1.jsonl"
)

CONFIDENCE_FLOOR = 0.7
ANSWER_MIN_CHARS = 20
STALE_CUTOFF = "2025-08-06"

EXPECTED_SINGLETONS = 3598


def clean_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def make_issue_key(row, row_index):
    """与 scripts/12_build_question_clusters.py 完全同源"""

    source_candidate_id = clean_text(
        row.get("source_candidate_id")
    )

    issue_seq = clean_text(row.get("issue_seq"))

    if source_candidate_id and issue_seq:
        return (
            f"{source_candidate_id}"
            f"#{issue_seq}"
        )

    if source_candidate_id:
        return (
            f"{source_candidate_id}"
            f"#ROW-{row_index + 1:05d}"
        )

    return (
        f"ROW-{row_index + 1:05d}"
    )


def main():

    issue_df = pd.read_excel(
        ISSUE_XLSX,
        sheet_name="extracted_issues",
    )

    members = pd.read_excel(
        CLUSTER_MEMBERS_XLSX,
        sheet_name="cluster_members",
    )
    clustered = set(
        members["issue_key"].astype(str)
    )

    timestamps = {}

    with open(
        TIMESTAMPS_FILE, encoding="utf-8"
    ) as f:
        for line in f:
            record = json.loads(line)
            timestamps[record["issue_key"]] = record

    records = []

    for index, row in issue_df.iterrows():
        issue_key = make_issue_key(row, index)

        if issue_key in clustered:
            continue

        ts = timestamps.get(issue_key)

        if ts is None:
            raise KeyError(
                f"{issue_key} 无 timestamp 记录"
            )

        issue_date = clean_text(ts.get("issue_date"))

        confidence = float(row["confidence"])
        knowledge_value = clean_text(
            row.get("knowledge_value")
        )
        resolution = clean_text(
            row.get("resolution")
        )
        temporal_status = clean_text(
            row.get("temporal_status")
        )
        answer = clean_text(row.get("answer"))

        exclusion_reasons = []

        if confidence < CONFIDENCE_FLOOR:
            exclusion_reasons.append(
                "A1_CONFIDENCE_BELOW_0.7"
            )

        if knowledge_value == "low":
            exclusion_reasons.append(
                "A2_KNOWLEDGE_VALUE_LOW"
            )

        if len(answer) < ANSWER_MIN_CHARS:
            exclusion_reasons.append(
                "A3_ANSWER_TOO_SHORT"
            )

        if temporal_status == "future_plan":
            exclusion_reasons.append(
                "A4_FUTURE_PLAN"
            )

        if temporal_status == "historical":
            exclusion_reasons.append(
                "A5_HISTORICAL"
            )

        stale = bool(issue_date) and (
            issue_date < STALE_CUTOFF
        )

        if stale:
            exclusion_reasons.append(
                "B1_STALE_NO_RECENT_CONFIRMATION"
            )

        records.append({
            "issue_key": issue_key,
            "question": clean_text(
                row.get("question")
            ),
            "question_normalized": clean_text(
                row.get("question_normalized")
            ),
            "answer": answer,
            "answer_chars": len(answer),
            "confidence": confidence,
            "knowledge_value": knowledge_value,
            "resolution": resolution,
            "temporal_status": temporal_status,
            "problem_type": clean_text(
                row.get("problem_type")
            ),
            "crm_module": clean_text(
                row.get("crm_module")
            ),
            "crm_feature": clean_text(
                row.get("crm_feature")
            ),
            "document_id": clean_text(
                row.get("document_id")
            ),
            "source_filename": clean_text(
                row.get("source_filename")
            ),
            "issue_date": issue_date,
            "stale": stale,
            "excluded": bool(exclusion_reasons),
            "exclusion_reasons": exclusion_reasons,
        })

    if len(records) != EXPECTED_SINGLETONS:
        raise RuntimeError(
            f"singleton 重建数 {len(records)} != "
            f"{EXPECTED_SINGLETONS}"
        )

    survivors = [
        record for record in records
        if not record["excluded"]
    ]

    unresolved_survivors = [
        record for record in survivors
        if record["resolution"] in (
            "unresolved", "partial"
        )
    ]

    # 红线守卫:
    # unresolved/partial 必须仍在幸存者中
    if len(unresolved_survivors) == 0:
        raise RuntimeError(
            "红线守卫失败: A+B 后无 unresolved/partial"
        )

    with open(
        FUNNEL_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in records:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    with open(
        SURVIVORS_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in survivors:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    df = pd.DataFrame(records)

    reason_rows = []

    for reason in [
        "A1_CONFIDENCE_BELOW_0.7",
        "A2_KNOWLEDGE_VALUE_LOW",
        "A3_ANSWER_TOO_SHORT",
        "A4_FUTURE_PLAN",
        "A5_HISTORICAL",
        "B1_STALE_NO_RECENT_CONFIRMATION",
    ]:
        reason_rows.append({
            "reason": reason,
            "count": int(
                df["exclusion_reasons"].apply(
                    lambda values: reason in values
                ).sum()
            ),
        })

    summary = {
        "singletons_total": len(records),
        "excluded_total": int(df["excluded"].sum()),
        "survivors_ab": len(survivors),
        "survivors_unresolved_partial": len(
            unresolved_survivors
        ),
        "survivors_feature_request": sum(
            1 for record in survivors
            if record["resolution"]
            == "feature_request"
        ),
        "stale_rescue_backlog": int(
            df["stale"].sum()
        ),
        "confidence_floor": CONFIDENCE_FLOOR,
        "answer_min_chars": ANSWER_MIN_CHARS,
        "stale_cutoff": STALE_CUTOFF,
    }

    survivor_df = pd.DataFrame(survivors)

    with pd.ExcelWriter(
        FUNNEL_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        reason_df = pd.DataFrame(reason_rows)
        reason_df.to_excel(
            writer, sheet_name="exclusion_reasons",
            index=False,
        )
        survivor_df.to_excel(
            writer, sheet_name="survivors", index=False
        )
        df[df["excluded"]].drop(
            columns=["excluded"]
        ).to_excel(
            writer, sheet_name="excluded", index=False
        )

    print("=" * 60)
    print("Singleton funnel A+B 完成（Step 15.2）")
    print("=" * 60)
    print(f"singletons: {len(records)}")
    print(f"A+B 排除: {summary['excluded_total']}")
    print(f"幸存: {len(survivors)}")
    print(
        f"  其中 unresolved/partial: "
        f"{len(unresolved_survivors)}"
    )
    print(
        f"  其中 feature_request(送LLM层): "
        f"{summary['survivors_feature_request']}"
    )
    print()
    print("排除原因分布:")

    for row in reason_rows:
        print(f"  {row['reason']}: {row['count']}")

    print()
    print(f"stale 救回候补(B 层全部留档): "
          f"{summary['stale_rescue_backlog']}")
    print()
    print(f"输出: {FUNNEL_JSONL}")
    print(f"输出: {FUNNEL_XLSX}")
    print(f"输出: {SURVIVORS_JSONL}")


if __name__ == "__main__":
    main()
