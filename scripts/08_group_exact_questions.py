from pathlib import Path
import json

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "extracted_issues.xlsx"

OUTPUT_FILE = OUTPUT_DIR / "exact_question_groups.xlsx"


# ============================================================
# 工具函数
# ============================================================

def normalize_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def unique_nonempty(values):
    result = []

    for value in values:
        text = normalize_value(value)

        if not text:
            continue

        if text not in result:
            result.append(text)

    return result


def join_unique(values):
    items = unique_nonempty(values)

    return " | ".join(items)


def count_unique_nonempty(values):
    return len(
        unique_nonempty(values)
    )


def safe_json(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
        )
    except Exception:
        return str(value)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("Step 8.1 - Exact question_normalized grouping")
    print("=" * 70)

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="extracted_issues",
    )

    print(f"总 issue 数: {len(df)}")

    required_columns = [
        "source_candidate_id",
        "question",
        "question_normalized",
        "description",
        "answer",
        "solution",
        "problem_type",
        "crm_module",
        "crm_feature",
        "resolution",
        "confidence",
        "knowledge_value",
        "needs_followup",
        "temporal_status",
        "source_message_indexes",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # --------------------------------------------------------
    # 清理 question_normalized
    # --------------------------------------------------------

    df["question_normalized_clean"] = (
        df["question_normalized"]
        .apply(normalize_value)
    )

    empty_count = (
        df["question_normalized_clean"]
        .eq("")
        .sum()
    )

    print(
        f"空 question_normalized: "
        f"{empty_count}"
    )

    df_valid = df[
        df["question_normalized_clean"]
        != ""
    ].copy()

    # --------------------------------------------------------
    # 统计完全相同问题
    # --------------------------------------------------------

    counts = (
        df_valid[
            "question_normalized_clean"
        ]
        .value_counts()
    )

    duplicate_questions = counts[
        counts > 1
    ]

    print(
        f"重复 question_normalized 数: "
        f"{len(duplicate_questions)}"
    )

    print(
        f"重复组成员总数: "
        f"{duplicate_questions.sum()}"
    )

    # --------------------------------------------------------
    # 构造 group id
    # --------------------------------------------------------

    duplicate_question_set = set(
        duplicate_questions.index.tolist()
    )

    duplicate_df = df_valid[
        df_valid[
            "question_normalized_clean"
        ].isin(
            duplicate_question_set
        )
    ].copy()

    group_map = {}

    for idx, question in enumerate(
        sorted(
            duplicate_question_set
        ),
        start=1,
    ):
        group_map[question] = (
            f"GROUP-EXACT-{idx:04d}"
        )

    duplicate_df["exact_group_id"] = (
        duplicate_df[
            "question_normalized_clean"
        ].map(group_map)
    )

    # --------------------------------------------------------
    # 每组 summary
    # --------------------------------------------------------

    group_rows = []

    grouped = duplicate_df.groupby(
        "exact_group_id",
        sort=True,
    )

    for group_id, group in grouped:

        question_normalized = (
            group[
                "question_normalized_clean"
            ].iloc[0]
        )

        answers = unique_nonempty(
            group["answer"]
        )

        solutions = unique_nonempty(
            group["solution"]
        )

        resolutions = unique_nonempty(
            group["resolution"]
        )

        temporal_statuses = unique_nonempty(
            group["temporal_status"]
        )

        knowledge_values = unique_nonempty(
            group["knowledge_value"]
        )

        crm_modules = unique_nonempty(
            group["crm_module"]
        )

        crm_features = unique_nonempty(
            group["crm_feature"]
        )

        problem_types = unique_nonempty(
            group["problem_type"]
        )

        needs_followups = unique_nonempty(
            group["needs_followup"]
        )

        confidences = pd.to_numeric(
            group["confidence"],
            errors="coerce",
        )

        answer_unique_count = len(
            answers
        )

        solution_unique_count = len(
            solutions
        )

        # ----------------------------------------------------
        # 初步组类型
        # ----------------------------------------------------

        if (
            answer_unique_count == 1
            and solution_unique_count <= 1
        ):
            group_type = "same_question_same_answer"

        elif answer_unique_count > 1:
            group_type = "same_question_different_answer"

        else:
            group_type = "same_question_mixed"

        # ----------------------------------------------------
        # 是否值得人工复核
        # ----------------------------------------------------

        review_reasons = []

        if answer_unique_count > 1:
            review_reasons.append(
                "multiple_answers"
            )

        if len(resolutions) > 1:
            review_reasons.append(
                "resolution_diff"
            )

        if len(temporal_statuses) > 1:
            review_reasons.append(
                "temporal_status_diff"
            )

        if len(knowledge_values) > 1:
            review_reasons.append(
                "knowledge_value_diff"
            )

        if len(crm_modules) > 1:
            review_reasons.append(
                "crm_module_diff"
            )

        if len(needs_followups) > 1:
            review_reasons.append(
                "needs_followup_diff"
            )

        needs_review = (
            len(review_reasons) > 0
        )

        group_rows.append(
            {
                "exact_group_id":
                    group_id,

                "question_normalized":
                    question_normalized,

                "member_count":
                    len(group),

                "answer_unique_count":
                    answer_unique_count,

                "solution_unique_count":
                    solution_unique_count,

                "group_type":
                    group_type,

                "needs_review":
                    needs_review,

                "review_reasons":
                    ", ".join(
                        review_reasons
                    ),

                "avg_confidence":
                    confidences.mean(),

                "min_confidence":
                    confidences.min(),

                "max_confidence":
                    confidences.max(),

                "problem_type_set":
                    join_unique(
                        group["problem_type"]
                    ),

                "crm_module_set":
                    join_unique(
                        group["crm_module"]
                    ),

                "crm_feature_set":
                    join_unique(
                        group["crm_feature"]
                    ),

                "resolution_set":
                    join_unique(
                        group["resolution"]
                    ),

                "knowledge_value_set":
                    join_unique(
                        group["knowledge_value"]
                    ),

                "needs_followup_set":
                    join_unique(
                        group["needs_followup"]
                    ),

                "temporal_status_set":
                    join_unique(
                        group["temporal_status"]
                    ),

                "answer_preview":
                    " || ".join(
                        answers[:5]
                    ),

                "solution_preview":
                    " || ".join(
                        solutions[:5]
                    ),

                "source_candidate_ids":
                    join_unique(
                        group[
                            "source_candidate_id"
                        ]
                    ),
            }
        )

    groups_df = pd.DataFrame(
        group_rows
    )

    # --------------------------------------------------------
    # member 明细
    # --------------------------------------------------------

    member_columns = [
        "exact_group_id",
        "source_candidate_id",
        "question",
        "question_normalized",
        "description",
        "answer",
        "solution",
        "problem_type",
        "crm_module",
        "crm_feature",
        "resolution",
        "confidence",
        "knowledge_value",
        "needs_followup",
        "temporal_status",
        "source_message_indexes",
    ]

    members_df = duplicate_df[
        member_columns
    ].copy()

    # --------------------------------------------------------
    # review 组
    # --------------------------------------------------------

    if not groups_df.empty:
        review_groups_df = groups_df[
            groups_df[
                "needs_review"
            ] == True
        ].copy()

        review_group_ids = set(
            review_groups_df[
                "exact_group_id"
            ].tolist()
        )

        review_members_df = members_df[
            members_df[
                "exact_group_id"
            ].isin(
                review_group_ids
            )
        ].copy()

    else:
        review_groups_df = pd.DataFrame()
        review_members_df = pd.DataFrame()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary_rows = [
        {
            "metric":
                "total_issue_count",
            "value":
                len(df),
        },
        {
            "metric":
                "valid_question_normalized_count",
            "value":
                len(df_valid),
        },
        {
            "metric":
                "exact_duplicate_group_count",
            "value":
                len(groups_df),
        },
        {
            "metric":
                "exact_duplicate_member_count",
            "value":
                len(members_df),
        },
        {
            "metric":
                "review_group_count",
            "value":
                len(review_groups_df),
        },
        {
            "metric":
                "same_question_same_answer_group_count",
            "value":
                (
                    groups_df[
                        "group_type"
                    ]
                    .eq(
                        "same_question_same_answer"
                    )
                    .sum()
                    if not groups_df.empty
                    else 0
                ),
        },
        {
            "metric":
                "same_question_different_answer_group_count",
            "value":
                (
                    groups_df[
                        "group_type"
                    ]
                    .eq(
                        "same_question_different_answer"
                    )
                    .sum()
                    if not groups_df.empty
                    else 0
                ),
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # 导出
    # --------------------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        groups_df.to_excel(
            writer,
            sheet_name="exact_groups",
            index=False,
        )

        members_df.to_excel(
            writer,
            sheet_name="exact_members",
            index=False,
        )

        review_groups_df.to_excel(
            writer,
            sheet_name="review_groups",
            index=False,
        )

        review_members_df.to_excel(
            writer,
            sheet_name="review_members",
            index=False,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        print(
            f"{row['metric']:<45}"
            f"{row['value']}"
        )

    print()
    print(
        f"输出文件: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()