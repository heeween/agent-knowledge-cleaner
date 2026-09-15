from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CLEAN_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_structurally_clean.xlsx"
)

PUBLISHABILITY_FILE = (
    OUTPUT_DIR
    / "multicause_publishability.xlsx"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_publishable.xlsx"
)


def clean_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def main():

    print("=" * 70)
    print(
        "Step 10.7 - Export publishable multicause KB"
    )
    print("=" * 70)

    entries_df = pd.read_excel(
        CLEAN_FILE,
        sheet_name="entries",
    )

    causes_df = pd.read_excel(
        CLEAN_FILE,
        sheet_name="causes",
    )

    audit_df = pd.read_excel(
        PUBLISHABILITY_FILE,
        sheet_name="all_clusters",
    )

    # --------------------------------------------------------
    # 只取最终 publish
    # --------------------------------------------------------

    publish_df = audit_df[
        audit_df["verdict"]
        == "publish"
    ].copy()

    publish_ids = set(
        publish_df[
            "cluster_id"
        ]
        .astype(str)
        .tolist()
    )

    final_entries = entries_df[
        entries_df[
            "cluster_id"
        ]
        .astype(str)
        .isin(
            publish_ids
        )
    ].copy()

    final_causes = causes_df[
        causes_df[
            "cluster_id"
        ]
        .astype(str)
        .isin(
            publish_ids
        )
    ].copy()

    # --------------------------------------------------------
    # 加最终审核信息
    # --------------------------------------------------------

    audit_columns = [
        "cluster_id",
        "verdict",
        "reason_category",
        "audit_confidence",
        "reason",
        "publish_notes",
    ]

    audit_small = publish_df[
        audit_columns
    ].copy()

    final_entries = (
        final_entries.merge(
            audit_small,
            on="cluster_id",
            how="left",
        )
    )

    # --------------------------------------------------------
    # 构造可直接用于 KB 的完整回答文本
    # --------------------------------------------------------

    kb_rows = []

    for _, entry in (
        final_entries.iterrows()
    ):

        cluster_id = str(
            entry["cluster_id"]
        )

        canonical_question = (
            clean_text(
                entry[
                    "canonical_question"
                ]
            )
        )

        cluster_causes = (
            final_causes[
                final_causes[
                    "cluster_id"
                ].astype(str)
                == cluster_id
            ]
            .sort_values(
                "cause_index"
            )
        )

        answer_lines = []

        for _, cause_row in (
            cluster_causes.iterrows()
        ):

            idx = int(
                cause_row[
                    "cause_index"
                ]
            )

            cause = clean_text(
                cause_row[
                    "cause"
                ]
            )

            action = clean_text(
                cause_row[
                    "check_or_action"
                ]
            )

            line = (
                f"{idx}. {cause}"
            )

            if action:

                line += (
                    f"\n   建议处理：{action}"
                )

            answer_lines.append(
                line
            )

        final_answer = (
            "\n".join(
                answer_lines
            )
        )

        source_keys = []

        for value in (
            cluster_causes[
                "source_issue_keys"
            ]
        ):

            for key in clean_text(
                value
            ).split("|"):

                key = key.strip()

                if (
                    key
                    and
                    key not in source_keys
                ):
                    source_keys.append(
                        key
                    )

        kb_rows.append(
            {
                "kb_id":
                    (
                        "KB-MULTI-"
                        + cluster_id.replace(
                            "QCLUSTER-",
                            ""
                        )
                    ),

                "cluster_id":
                    cluster_id,

                "question":
                    canonical_question,

                "answer":
                    final_answer,

                "cause_count":
                    len(
                        cluster_causes
                    ),

                "source_issue_keys":
                    " | ".join(
                        source_keys
                    ),

                "knowledge_type":
                    "multiple_causes",

                "publish_status":
                    "PUBLISHED",

                "audit_confidence":
                    entry[
                        "audit_confidence"
                    ],

                "publish_reason":
                    clean_text(
                        entry[
                            "reason"
                        ]
                    ),
            }
        )

    kb_df = pd.DataFrame(
        kb_rows
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "publishable_multicause_entries",
                "value":
                    len(
                        kb_df
                    ),
            },
            {
                "metric":
                    "publishable_causes",
                "value":
                    len(
                        final_causes
                    ),
            },
        ]
    )

    # --------------------------------------------------------
    # 安全检查
    # --------------------------------------------------------

    expected_ids = {
        "QCLUSTER-0030",
        "QCLUSTER-0067",
        "QCLUSTER-0150",
    }

    actual_ids = set(
        kb_df[
            "cluster_id"
        ].astype(str)
    )

    if actual_ids != expected_ids:

        print(
            "实际 publish cluster 与预期不一致！"
        )

        print(
            "Actual:",
            sorted(actual_ids),
        )

        print(
            "Expected:",
            sorted(expected_ids),
        )

        raise RuntimeError(
            "Publish cluster changed."
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

        kb_df.to_excel(
            writer,
            sheet_name="kb_entries",
            index=False,
        )

        final_entries.to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        final_causes.to_excel(
            writer,
            sheet_name="causes",
            index=False,
        )

    print()
    print(
        f"Published KB entries: "
        f"{len(kb_df)}"
    )

    print(
        f"Published causes: "
        f"{len(final_causes)}"
    )

    print()
    print(
        f"Excel: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()