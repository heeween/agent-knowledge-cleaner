from pathlib import Path

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

KB_FILE = (
    OUTPUT_DIR
    / "kb_entries_safe.xlsx"
)

ALIGNMENT_FILE = (
    OUTPUT_DIR
    / "issue_qa_alignments.xlsx"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "kb_entry_validations.xlsx"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "kb_source_quality.xlsx"
)


# ============================================================
# 工具
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def parse_issue_keys(value):

    text = clean_text(value)

    if not text:
        return []

    # Step 9.1 Excel 中是用 " | " 拼接
    parts = [
        x.strip()
        for x in text.split("|")
    ]

    return [
        x
        for x in parts
        if x
    ]


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 9.4 - Check KB source quality"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 读取
    # --------------------------------------------------------

    kb_df = pd.read_excel(
        KB_FILE,
        sheet_name="kb_entries",
    )

    alignment_df = pd.read_excel(
        ALIGNMENT_FILE,
        sheet_name="all_issues",
    )

    audit_df = pd.read_excel(
        AUDIT_FILE,
        sheet_name="all_entries",
    )

    print(
        f"KB entries: "
        f"{len(kb_df)}"
    )

    print(
        f"Alignment issues: "
        f"{len(alignment_df)}"
    )

    # --------------------------------------------------------
    # alignment lookup
    # --------------------------------------------------------

    alignment_lookup = {}

    for _, row in (
        alignment_df.iterrows()
    ):

        issue_key = clean_text(
            row.get("issue_key")
        )

        if not issue_key:
            continue

        alignment_lookup[
            issue_key
        ] = {
            "alignment":
                clean_text(
                    row.get(
                        "alignment"
                    )
                ),

            "alignment_confidence":
                row.get(
                    "alignment_confidence"
                ),

            "usable_for_kb":
                row.get(
                    "usable_for_kb"
                ),

            "mismatch_reason":
                clean_text(
                    row.get(
                        "mismatch_reason"
                    )
                ),

            "question":
                clean_text(
                    row.get(
                        "question"
                    )
                ),

            "answer":
                clean_text(
                    row.get(
                        "answer"
                    )
                ),
        }

    # --------------------------------------------------------
    # 检查每条 KB 的实际 source_issue_keys
    # --------------------------------------------------------

    result_rows = []
    source_detail_rows = []

    for _, kb_row in (
        kb_df.iterrows()
    ):

        cluster_id = clean_text(
            kb_row.get(
                "cluster_id"
            )
        )

        canonical_question = clean_text(
            kb_row.get(
                "canonical_question"
            )
        )

        source_keys = parse_issue_keys(
            kb_row.get(
                "source_issue_keys"
            )
        )

        aligned_keys = []
        partial_keys = []
        misaligned_keys = []
        no_answer_keys = []
        uncertain_keys = []
        missing_keys = []

        unusable_keys = []

        for issue_key in source_keys:

            info = alignment_lookup.get(
                issue_key
            )

            if info is None:

                missing_keys.append(
                    issue_key
                )

                continue

            alignment = info[
                "alignment"
            ]

            usable = info[
                "usable_for_kb"
            ]

            if alignment == "aligned":

                aligned_keys.append(
                    issue_key
                )

            elif (
                alignment
                == "partially_aligned"
            ):

                partial_keys.append(
                    issue_key
                )

            elif alignment == "misaligned":

                misaligned_keys.append(
                    issue_key
                )

            elif alignment == "no_answer":

                no_answer_keys.append(
                    issue_key
                )

            else:

                uncertain_keys.append(
                    issue_key
                )

            if usable is not True:

                unusable_keys.append(
                    issue_key
                )

            # source 明细
            source_detail_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        canonical_question,

                    "issue_key":
                        issue_key,

                    "alignment":
                        alignment,

                    "alignment_confidence":
                        info[
                            "alignment_confidence"
                        ],

                    "usable_for_kb":
                        usable,

                    "source_question":
                        info[
                            "question"
                        ],

                    "source_answer":
                        info[
                            "answer"
                        ],

                    "mismatch_reason":
                        info[
                            "mismatch_reason"
                        ],
                }
            )

        # ----------------------------------------------------
        # Source quality verdict
        # ----------------------------------------------------

        if not source_keys:

            source_quality = (
                "missing_source_keys"
            )

            recommended_action = (
                "manual_review"
            )

        elif misaligned_keys:

            # 实际引用了答非所问 source
            source_quality = (
                "contaminated"
            )

            recommended_action = (
                "regenerate"
            )

        elif (
            no_answer_keys
            or uncertain_keys
        ):

            # 模型居然把没有答案的 source
            # 当作最终答案依据
            source_quality = (
                "weak_source_used"
            )

            recommended_action = (
                "regenerate"
            )

        elif missing_keys:

            source_quality = (
                "source_not_found"
            )

            recommended_action = (
                "manual_review"
            )

        elif partial_keys:

            # partially_aligned 可能仍可用，
            # 但建议重新审计
            source_quality = (
                "usable_with_caution"
            )

            recommended_action = (
                "reaudit"
            )

        else:

            source_quality = (
                "clean"
            )

            recommended_action = (
                "keep"
            )

        result_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    canonical_question,

                "answer_relation":
                    clean_text(
                        kb_row.get(
                            "answer_relation"
                        )
                    ),

                "knowledge_confidence":
                    kb_row.get(
                        "knowledge_confidence"
                    ),

                "source_count":
                    len(source_keys),

                "aligned_source_count":
                    len(aligned_keys),

                "partial_source_count":
                    len(partial_keys),

                "misaligned_source_count":
                    len(misaligned_keys),

                "no_answer_source_count":
                    len(no_answer_keys),

                "uncertain_source_count":
                    len(uncertain_keys),

                "missing_source_count":
                    len(missing_keys),

                "unusable_source_count":
                    len(unusable_keys),

                "source_quality":
                    source_quality,

                "recommended_action":
                    recommended_action,

                "source_issue_keys":
                    " | ".join(
                        source_keys
                    ),

                "bad_source_issue_keys":
                    " | ".join(
                        unusable_keys
                    ),

                "misaligned_issue_keys":
                    " | ".join(
                        misaligned_keys
                    ),

                "no_answer_issue_keys":
                    " | ".join(
                        no_answer_keys
                    ),

                "partial_issue_keys":
                    " | ".join(
                        partial_keys
                    ),

                "missing_issue_keys":
                    " | ".join(
                        missing_keys
                    ),
            }
        )

    result_df = pd.DataFrame(
        result_rows
    )

    source_detail_df = pd.DataFrame(
        source_detail_rows
    )

    # --------------------------------------------------------
    # 合并 Step 9.2 审计结果
    # --------------------------------------------------------

    audit_cols = [
        "cluster_id",
        "verdict",
        "audit_confidence",
        "question_alignment",
        "grounding",
        "completeness",
        "temporal_risk",
        "unsupported_claims",
        "missing_key_information",
        "contradictions",
        "reason",
    ]

    available_audit_cols = [
        col
        for col in audit_cols
        if col in audit_df.columns
    ]

    audit_small_df = (
        audit_df[
            available_audit_cols
        ]
        .copy()
    )

    # canonical_question 可能重复列，
    # 所以只按 cluster_id 合并
    final_df = result_df.merge(
        audit_small_df,
        on="cluster_id",
        how="left",
    )

    # --------------------------------------------------------
    # 最终放行状态
    # --------------------------------------------------------

    def decide_final_status(row):

        source_quality = (
            clean_text(
                row.get(
                    "source_quality"
                )
            )
        )

        audit_verdict = (
            clean_text(
                row.get(
                    "verdict"
                )
            )
        )

        grounding = (
            clean_text(
                row.get(
                    "grounding"
                )
            )
        )

        temporal_risk = (
            clean_text(
                row.get(
                    "temporal_risk"
                )
            )
        )

        # ----------------------------------------------------
        # Source 本身已经污染
        # ----------------------------------------------------

        if source_quality in {
            "contaminated",
            "weak_source_used",
        }:

            return (
                "REGENERATE"
            )

        # ----------------------------------------------------
        # Source key 异常
        # ----------------------------------------------------

        if source_quality in {
            "missing_source_keys",
            "source_not_found",
        }:

            return (
                "MANUAL_REVIEW"
            )

        # ----------------------------------------------------
        # Grounding 审计没通过
        # ----------------------------------------------------

        if audit_verdict != "pass":

            return (
                "MANUAL_REVIEW"
            )

        if grounding not in {
            "fully_supported",
            "mostly_supported",
        }:

            return (
                "MANUAL_REVIEW"
            )

        if temporal_risk == "high":

            return (
                "MANUAL_REVIEW"
            )

        # ----------------------------------------------------
        # partial source
        # ----------------------------------------------------

        if (
            source_quality
            == "usable_with_caution"
        ):

            return (
                "MANUAL_REVIEW"
            )

        return "APPROVED"

    final_df[
        "final_status"
    ] = final_df.apply(
        decide_final_status,
        axis=1,
    )

    # --------------------------------------------------------
    # 分表
    # --------------------------------------------------------

    approved_df = final_df[
        final_df[
            "final_status"
        ]
        == "APPROVED"
    ].copy()

    regenerate_df = final_df[
        final_df[
            "final_status"
        ]
        == "REGENERATE"
    ].copy()

    manual_review_df = final_df[
        final_df[
            "final_status"
        ]
        == "MANUAL_REVIEW"
    ].copy()

    # --------------------------------------------------------
    # 被污染 source 的明细
    # --------------------------------------------------------

    bad_detail_df = source_detail_df[
        (
            source_detail_df[
                "usable_for_kb"
            ]
            != True
        )
        |
        (
            source_detail_df[
                "alignment"
            ]
            .isin(
                [
                    "misaligned",
                    "no_answer",
                    "uncertain",
                ]
            )
        )
    ].copy()

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    source_quality_stats = (
        final_df[
            "source_quality"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "source_quality"
        )
        .reset_index(
            name="count"
        )
    )

    final_status_stats = (
        final_df[
            "final_status"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "final_status"
        )
        .reset_index(
            name="count"
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "kb_entry_count",
                "value":
                    len(final_df),
            },
            {
                "metric":
                    "approved_count",
                "value":
                    len(approved_df),
            },
            {
                "metric":
                    "regenerate_count",
                "value":
                    len(regenerate_df),
            },
            {
                "metric":
                    "manual_review_count",
                "value":
                    len(
                        manual_review_df
                    ),
            },
            {
                "metric":
                    "clean_source_count",
                "value":
                    int(
                        (
                            final_df[
                                "source_quality"
                            ]
                            == "clean"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "contaminated_count",
                "value":
                    int(
                        (
                            final_df[
                                "source_quality"
                            ]
                            == "contaminated"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "weak_source_used_count",
                "value":
                    int(
                        (
                            final_df[
                                "source_quality"
                            ]
                            == "weak_source_used"
                        ).sum()
                    ),
            },
        ]
    )

    # --------------------------------------------------------
    # Excel
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

        source_quality_stats.to_excel(
            writer,
            sheet_name="source_quality_stats",
            index=False,
        )

        final_status_stats.to_excel(
            writer,
            sheet_name="final_status_stats",
            index=False,
        )

        final_df.to_excel(
            writer,
            sheet_name="all_entries",
            index=False,
        )

        approved_df.to_excel(
            writer,
            sheet_name="approved",
            index=False,
        )

        regenerate_df.to_excel(
            writer,
            sheet_name="regenerate",
            index=False,
        )

        manual_review_df.to_excel(
            writer,
            sheet_name="manual_review",
            index=False,
        )

        source_detail_df.to_excel(
            writer,
            sheet_name="source_details",
            index=False,
        )

        bad_detail_df.to_excel(
            writer,
            sheet_name="bad_source_details",
            index=False,
        )

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        final_status_stats
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()