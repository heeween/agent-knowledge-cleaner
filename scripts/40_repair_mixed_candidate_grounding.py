"""
Step 11.5.1 - Deterministic Candidate KB Grounding Repairs

本脚本只处理 Step 11.5 中已经确认的 summary 级缺陷。

原则：

1. 不覆盖 Step 11.4 冻结输出
2. 不修改任何 unit 内容
3. 不调用 LLM
4. 每条修改都记录 before / after / reason
5. QCLUSTER-0021 / QCLUSTER-0066 的 same_intent 冲突
   不在本步骤处理，留到 Step 11.6 dedup / merge decision

输入：

output/kb_entries_mixed_candidate.jsonl
output/kb_entry_grounding_validations_mixed.jsonl

输出：

output/kb_entries_mixed_candidate_v2.jsonl
output/kb_entries_mixed_candidate_v2.xlsx
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_JSONL = (
    OUTPUT_DIR / "kb_entries_mixed_candidate.jsonl"
)

VALIDATION_JSONL = (
    OUTPUT_DIR
    / "kb_entry_grounding_validations_mixed.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v2.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_mixed_candidate_v2.xlsx"
)


SUMMARY_REPAIRS = {
    "QCLUSTER-0004": {
        "summary_answer": (
            "CRM 账号密码重置方式取决于账号类型：客户账号可联系客服重置，"
            "默认密码为 123456，登录后可自行修改；"
            "员工账号按“员工账号密码重置”场景处理。"
        ),
        "reason": (
            "移除 source 未支撑的执行者“管理员”，"
            "并将员工账号的具体菜单和按钮细节保留在对应 scenario unit 中，"
            "避免 summary 混入单一场景的 UI 细节。"
        ),
    },
    "QCLUSTER-0024": {
        "summary_answer": (
            "邀约进店数据按邀约时间统计：车辆进店前有人邀约即计为邀约进店；"
            "多人跟进取最近一次记录；当前默认有效期为一年。"
            "门店自定义有效期属于未来规划，尚未上线，详见 limitations。"
        ),
        "reason": (
            "明确区分当前统计规则与未来规划，不在 summary 中把"
            "“预计 1 个月内上线”写成当前规则。"
        ),
    },
    "QCLUSTER-0014": {
        "summary_answer": (
            "更正错误的里程数据时，可先按标准流程直接使用修改功能；"
            "如果遇到特定录入错误场景（如多填 0），"
            "可按临时处理方案先刷新界面再修改。"
        ),
        "reason": (
            "把直接修改与刷新后修改拆成明确适用场景，"
            "避免把刷新界面泛化为通用操作。"
        ),
    },
    "QCLUSTER-0033": {
        "summary_answer": (
            "批量导入潜客数据有两种路径：需要客服协助时，"
            "可先购买潜客解析套餐包，并将整理后的 Excel 数据交给客服导入；"
            "导入无消费客户并分配跟进人时，"
            "可使用系统管理--导入设置自行导入，"
            "并注意表格内容（例如避免表情符号等特殊字符）、"
            "跳过已存在客户和手动分配跟进人。"
        ),
        "reason": (
            "保留两条路径的 source-grounded 条件，"
            "并把过度概括的“表格格式规范”改回具体来源表述。"
        ),
    },
}


FALSE_POSITIVE_NOTES = {
    "QCLUSTER-0094": (
        "Step 11.5 将“财务人员”判为 invented actor，"
        "但 grounded source 的 supported_answer 与 supported_solution "
        "均明确包含“财务人员”。该 hard flag 属于 validator false positive，"
        "不修改 Candidate 内容。"
    ),
}


def clean_text(value):

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value).strip()


def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(f"缺少输入文件: {path}")

    records = []

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(path, records):

    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def apply_repairs(candidates, validations):

    validation_by_cluster = {
        clean_text(item.get("cluster_id")): item
        for item in validations
    }

    candidate_ids = {
        clean_text(item.get("cluster_id"))
        for item in candidates
    }

    if set(validation_by_cluster) != candidate_ids:
        raise RuntimeError(
            "Candidate 与 validation cluster 集合不一致。"
            f" candidate={sorted(candidate_ids)} "
            f"validation={sorted(validation_by_cluster)}"
        )

    repaired = []
    repair_rows = []

    for original in candidates:

        cluster_id = clean_text(original.get("cluster_id"))

        record = json.loads(json.dumps(original))

        validation = validation_by_cluster[cluster_id]

        old_summary = clean_text(record.get("summary_answer"))

        repair = SUMMARY_REPAIRS.get(cluster_id)

        if repair:

            expected_present = any(
                flag in clean_text(validation.get("validation_flags"))
                for flag in [
                    "INVENTED_ACTOR",
                    "INVENTED_TIME",
                    "INVENTED_PRODUCT_RULE",
                    "INVENTED_OTHER",
                    "SUMMARY_PARTIALLY_GROUNDED",
                ]
            )

            if not expected_present:
                raise RuntimeError(
                    f"{cluster_id}: validation 未确认 summary 缺陷，"
                    f"拒绝应用 repair。"
                )

            record["summary_answer"] = repair["summary_answer"]

            record["repair_applied"] = True

            record["repair_fields"] = ["summary_answer"]

            record["repair_reason"] = repair["reason"]

            record["repair_source_status"] = validation.get(
                "grounding_status"
            )

            repair_rows.append(
                {
                    "cluster_id": cluster_id,
                    "action": "replace_summary",
                    "field": "summary_answer",
                    "before": old_summary,
                    "after": repair["summary_answer"],
                    "reason": repair["reason"],
                }
            )

        else:

            record["repair_applied"] = False

            record["repair_fields"] = []

            record["repair_reason"] = ""

            record["repair_source_status"] = validation.get(
                "grounding_status"
            )

        if cluster_id in FALSE_POSITIVE_NOTES:

            notes = list(record.get("notes") or [])

            note = FALSE_POSITIVE_NOTES[cluster_id]

            if note not in notes:
                notes.append(note)

            record["notes"] = notes

            record["validator_false_positive_reviewed"] = True

            record["validator_false_positive_reason"] = note

            repair_rows.append(
                {
                    "cluster_id": cluster_id,
                    "action": "record_false_positive",
                    "field": "notes / validator_false_positive_reason",
                    "before": "",
                    "after": note,
                    "reason": (
                        "被指控的 actor 原文同时出现在 "
                        "supported_answer 和 supported_solution。"
                    ),
                }
            )

        else:

            record["validator_false_positive_reviewed"] = False

            record["validator_false_positive_reason"] = ""

        repaired.append(record)

    return repaired, repair_rows


def export_excel(records, repair_rows):

    frame = pd.DataFrame(records)

    entry_columns = [
        "cluster_id",
        "canonical_question",
        "knowledge_structure",
        "generation_mode",
        "summary_answer",
        "unit_count",
        "usable_source_count",
        "cited_source_count",
        "candidate_status",
        "repair_applied",
        "repair_fields",
        "repair_reason",
        "repair_source_status",
        "validator_false_positive_reviewed",
        "validator_false_positive_reason",
        "notes",
        "limitations",
        "source_issue_keys",
        "soft_flags",
    ]

    entries = frame[
        [column for column in entry_columns if column in frame.columns]
    ].copy()

    unit_rows = []

    for record in records:

        for index, unit in enumerate(
            record.get("units") or [], start=1
        ):

            unit_rows.append(
                {
                    "cluster_id": record["cluster_id"],
                    "canonical_question": record[
                        "canonical_question"
                    ],
                    "unit_index": index,
                    "unit_type": unit.get("unit_type"),
                    "title": unit.get("title"),
                    "condition": unit.get("condition"),
                    "content": unit.get("content"),
                    "steps": " | ".join(
                        clean_text(step)
                        for step in unit.get("steps") or []
                    ),
                    "source_issue_keys": " | ".join(
                        clean_text(key)
                        for key in unit.get("source_issue_keys") or []
                    ),
                }
            )

    summary = pd.DataFrame(
        [
            {
                "metric": "entry_count",
                "value": len(records),
            },
            {
                "metric": "summary_repaired_count",
                "value": sum(
                    1 for item in records if item["repair_applied"]
                ),
            },
            {
                "metric": "unchanged_count",
                "value": sum(
                    1 for item in records
                    if not item["repair_applied"]
                ),
            },
            {
                "metric": "false_positive_reviewed_count",
                "value": sum(
                    1 for item in records
                    if item["validator_false_positive_reviewed"]
                ),
            },
            {
                "metric": "unit_count",
                "value": sum(
                    len(item.get("units") or [])
                    for item in records
                ),
            },
        ]
    )

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:

        summary.to_excel(
            writer,
            index=False,
            sheet_name="summary",
        )

        entries.to_excel(
            writer,
            index=False,
            sheet_name="kb_entries",
        )

        pd.DataFrame(unit_rows).to_excel(
            writer,
            index=False,
            sheet_name="units",
        )

        pd.DataFrame(repair_rows).to_excel(
            writer,
            index=False,
            sheet_name="repairs",
        )


def main():

    print("=" * 70)
    print("Step 11.5.1 - Deterministic Grounding Repairs")
    print("=" * 70)

    print("LLM 调用: 无")

    candidates = load_jsonl(INPUT_JSONL)

    validations = load_jsonl(VALIDATION_JSONL)

    records, repair_rows = apply_repairs(
        candidates,
        validations,
    )

    write_jsonl(OUTPUT_JSONL, records)

    export_excel(records, repair_rows)

    print()
    print(f"输入 Candidate: {len(candidates)}")

    print(f"修复 summary: {len(SUMMARY_REPAIRS)}")

    print(
        "false positive 复核: "
        f"{len(FALSE_POSITIVE_NOTES)}"
    )

    print()
    print("修复记录:")

    for row in repair_rows:

        print(
            f"{row['cluster_id']} "
            f"| {row['action']} "
            f"| {row['field']}"
        )

    print()
    print(f"JSONL: {OUTPUT_JSONL}")

    print(f"Excel: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
