from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

REBUILT_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_rebuilt.xlsx"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "multicause_rebuild_validations_v2.xlsx"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_structurally_clean.xlsx"
)


# ============================================================
# 手工确定性修复
#
# 注意：
# 这里只处理 Step 10.4 v2 明确抓到的 5 条。
#
# 不调用 LLM。
# 不新增知识。
# 所有内容都来自前面已经审核过的 Cause。
# ============================================================

FIXES = {

    # --------------------------------------------------------
    # 0017
    #
    # 两个 source 实质都是：
    # “短信签名尚未完成报备”
    #
    # 去掉被错误塞进 Cause 的处理动作：
    # “需等待次日重新提交报备尝试”
    # --------------------------------------------------------

    "QCLUSTER-0017#REBUILT-02": [
        {
            "cause":
                "短信签名尚未完成报备或审核",

            "check_or_action":
                None,

            "source_issue_keys":
                (
                    "ISSUE-CAND-000246#ROW-00234"
                    " | "
                    "ISSUE-CAND-000393#ROW-00416"
                ),
        }
    ],


    # --------------------------------------------------------
    # 0001
    #
    # 原 rebuilt 错误合并：
    #
    # 1. 短信签名未完成运营商报备
    # 2. 尚未开通短信签名功能
    #
    # 拆成两个独立 Cause。
    # --------------------------------------------------------

    "QCLUSTER-0001#REBUILT-03": [
        {
            "cause":
                "短信签名未完成运营商报备",

            "check_or_action":
                None,

            "source_issue_keys":
                "ISSUE-CAND-000051#ROW-00040",
        },
        {
            "cause":
                "尚未开通短信签名功能",

            "check_or_action":
                None,

            "source_issue_keys":
                "ISSUE-CAND-001482#ROW-01679",
        },
    ],


    # --------------------------------------------------------
    # 0022 rebuilt 01
    #
    # 物理音频连接
    # 和
    # 耳机模式
    #
    # 是两个独立 Cause。
    # --------------------------------------------------------

    "QCLUSTER-0022#REBUILT-01": [
        {
            "cause":
                (
                    "电脑输入麦克风或输出音频"
                    "未正确连接到电话机"
                ),

            "check_or_action":
                (
                    "检查并确保电脑的输入麦克风"
                    "和输出音频均连接到电话机"
                ),

            "source_issue_keys":
                "ISSUE-CAND-000624#ROW-00682",
        },
        {
            "cause":
                "未切换到耳机模式",

            "check_or_action":
                "点击界面右下角按钮切换至耳机模式",

            "source_issue_keys":
                "ISSUE-CAND-000624#ROW-00682",
        },
    ],


    # --------------------------------------------------------
    # 0022 rebuilt 03
    #
    # 静音/音量配置
    # 和
    # 物理连接/声音设备
    #
    # 独立保留。
    # --------------------------------------------------------

    "QCLUSTER-0022#REBUILT-03": [
        {
            "cause":
                (
                    "电脑音频设置异常"
                    "（如静音或音量过低）"
                ),

            "check_or_action":
                (
                    "检查电脑音频设置，"
                    "确认音量未静音且数值正常"
                ),

            "source_issue_keys":
                "ISSUE-CAND-001655#ROW-01853",
        },
        {
            "cause":
                (
                    "电脑与耳机的物理连接"
                    "或声音设备设置存在问题"
                ),

            "check_or_action":
                (
                    "检查电脑与耳机的物理连接，"
                    "并检查电脑的声音设备设置"
                ),

            "source_issue_keys":
                "ISSUE-CAND-001902#ROW-02152",
        },
    ],


    # --------------------------------------------------------
    # 0170
    #
    # 未录工单
    # 与
    # 数据同步问题
    #
    # 是两个不同的数据链路原因。
    # --------------------------------------------------------

    "QCLUSTER-0170#REBUILT-02": [
        {
            "cause":
                (
                    "未录入工单导致系统"
                    "未显示最新保养记录"
                ),

            "check_or_action":
                (
                    "检查后台工单录入情况，"
                    "确认是否漏录"
                ),

            "source_issue_keys":
                "ISSUE-CAND-003266#ROW-03673",
        },
        {
            "cause":
                (
                    "数据同步问题"
                    "（如跨店数据未同步）"
                    "导致记录缺失"
                ),

            "check_or_action":
                (
                    "核实数据同步状态，"
                    "确认是否为跨店数据"
                    "未同步导致的显示缺失"
                ),

            "source_issue_keys":
                "ISSUE-CAND-003266#ROW-03673",
        },
    ],
}


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
        "Step 10.5 - Apply deterministic multicause fixes"
    )
    print("=" * 70)

    rebuilt_df = pd.read_excel(
        REBUILT_FILE,
        sheet_name="causes",
    )

    entries_df = pd.read_excel(
        REBUILT_FILE,
        sheet_name="entries",
    )

    audit_df = pd.read_excel(
        AUDIT_FILE,
        sheet_name="all_causes",
    )

    # --------------------------------------------------------
    # 确认实际需要修复的 item
    # --------------------------------------------------------

    bad_df = audit_df[
        audit_df["verdict"]
        != "pass"
    ].copy()

    actual_bad_ids = set(
        bad_df["item_id"]
        .astype(str)
        .tolist()
    )

    expected_bad_ids = set(
        FIXES.keys()
    )

    if (
        actual_bad_ids
        != expected_bad_ids
    ):

        print()
        print(
            "实际审核失败项与 FIXES 不一致，停止。"
        )

        print(
            "Actual:"
        )

        for item_id in sorted(
            actual_bad_ids
        ):
            print(
                "  ",
                item_id,
            )

        print(
            "Expected:"
        )

        for item_id in sorted(
            expected_bad_ids
        ):
            print(
                "  ",
                item_id,
            )

        raise RuntimeError(
            "审核失败项发生变化，禁止自动修复。"
        )

    # --------------------------------------------------------
    # 为 rebuilt cause 生成 item_id
    # --------------------------------------------------------

    rebuilt_df[
        "item_id"
    ] = rebuilt_df.apply(
        lambda row:
            (
                f"{clean_text(row['cluster_id'])}"
                f"#REBUILT-"
                f"{int(row['cause_index']):02d}"
            ),
        axis=1,
    )

    final_rows = []

    # --------------------------------------------------------
    # Pass 原样保留
    # --------------------------------------------------------

    for _, row in (
        rebuilt_df.iterrows()
    ):

        item_id = row["item_id"]

        if item_id in FIXES:

            # 当前错误 rebuilt cause
            # 不保留
            continue

        final_rows.append(
            {
                "cluster_id":
                    clean_text(
                        row["cluster_id"]
                    ),

                "canonical_question":
                    clean_text(
                        row[
                            "canonical_question"
                        ]
                    ),

                "cause":
                    clean_text(
                        row["cause"]
                    ),

                "check_or_action":
                    (
                        clean_text(
                            row[
                                "check_or_action"
                            ]
                        )
                        or None
                    ),

                "source_issue_keys":
                    clean_text(
                        row[
                            "source_issue_keys"
                        ]
                    ),

                "repair_source":
                    "PASS_REBUILT",
            }
        )

    # --------------------------------------------------------
    # 加入确定性修复
    # --------------------------------------------------------

    question_map = dict(
        zip(
            entries_df[
                "cluster_id"
            ].astype(str),

            entries_df[
                "canonical_question"
            ].astype(str),
        )
    )

    for item_id, replacements in (
        FIXES.items()
    ):

        cluster_id = (
            item_id.split(
                "#REBUILT-"
            )[0]
        )

        canonical_question = (
            question_map[
                cluster_id
            ]
        )

        for replacement in replacements:

            final_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        canonical_question,

                    "cause":
                        replacement[
                            "cause"
                        ],

                    "check_or_action":
                        replacement[
                            "check_or_action"
                        ],

                    "source_issue_keys":
                        replacement[
                            "source_issue_keys"
                        ],

                    "repair_source":
                        item_id,
                }
            )

    final_df = pd.DataFrame(
        final_rows
    )

    # --------------------------------------------------------
    # 排序 + 重新编号
    # --------------------------------------------------------

    cluster_order = {
        cluster_id: idx

        for idx, cluster_id
        in enumerate(
            entries_df[
                "cluster_id"
            ].astype(str)
        )
    }

    final_df[
        "_cluster_order"
    ] = final_df[
        "cluster_id"
    ].map(
        cluster_order
    )

    final_df = (
        final_df
        .sort_values(
            [
                "_cluster_order",
                "repair_source",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_cluster_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    final_df[
        "cause_index"
    ] = (
        final_df
        .groupby(
            "cluster_id"
        )
        .cumcount()
        + 1
    )

    # 调整列顺序
    final_df = final_df[
        [
            "cluster_id",
            "canonical_question",
            "cause_index",
            "cause",
            "check_or_action",
            "source_issue_keys",
            "repair_source",
        ]
    ]

    # --------------------------------------------------------
    # Cluster 统计
    # --------------------------------------------------------

    cluster_stats = (
        final_df
        .groupby(
            [
                "cluster_id",
                "canonical_question",
            ],
            dropna=False,
        )
        .agg(
            cause_count=(
                "cause",
                "count",
            ),
        )
        .reset_index()
    )

    cluster_stats[
        "status"
    ] = cluster_stats[
        "cause_count"
    ].apply(
        lambda x:
            (
                "MULTICAUSE_OK"
                if x >= 2
                else
                "INSUFFICIENT_MULTICAUSE"
            )
    )

    # --------------------------------------------------------
    # entries
    # --------------------------------------------------------

    final_entries = (
        cluster_stats[
            [
                "cluster_id",
                "canonical_question",
                "cause_count",
                "status",
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    repaired_count = sum(
        len(v)
        for v in FIXES.values()
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "cluster_count",
                "value":
                    len(
                        final_entries
                    ),
            },
            {
                "metric":
                    "final_cause_count",
                "value":
                    len(
                        final_df
                    ),
            },
            {
                "metric":
                    "fixed_rebuilt_item_count",
                "value":
                    len(FIXES),
            },
            {
                "metric":
                    "replacement_cause_count",
                "value":
                    repaired_count,
            },
            {
                "metric":
                    "multicause_ok_count",
                "value":
                    int(
                        (
                            final_entries[
                                "status"
                            ]
                            == "MULTICAUSE_OK"
                        ).sum()
                    ),
            },
        ]
    )

    # --------------------------------------------------------
    # 修复明细
    # --------------------------------------------------------

    repair_rows = []

    for item_id, replacements in (
        FIXES.items()
    ):

        for idx, item in enumerate(
            replacements,
            start=1,
        ):

            repair_rows.append(
                {
                    "replaced_item_id":
                        item_id,

                    "replacement_index":
                        idx,

                    "cause":
                        item["cause"],

                    "check_or_action":
                        item[
                            "check_or_action"
                        ],

                    "source_issue_keys":
                        item[
                            "source_issue_keys"
                        ],
                }
            )

    repairs_df = pd.DataFrame(
        repair_rows
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

        final_entries.to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        final_df.to_excel(
            writer,
            sheet_name="causes",
            index=False,
        )

        repairs_df.to_excel(
            writer,
            sheet_name="repairs",
            index=False,
        )

        cluster_stats.to_excel(
            writer,
            sheet_name="cluster_stats",
            index=False,
        )

    print()
    print(
        f"Cluster count: "
        f"{len(final_entries)}"
    )

    print(
        f"Final cause count: "
        f"{len(final_df)}"
    )

    print(
        f"Fixed rebuilt items: "
        f"{len(FIXES)}"
    )

    print()
    print(
        f"Excel: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()