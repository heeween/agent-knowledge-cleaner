from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

PAIR_FILE = (
    OUTPUT_DIR
    / "pair_classifications.xlsx"
)

ISSUE_FILE = (
    OUTPUT_DIR
    / "extracted_issues.xlsx"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)


# 必须是 same_intent
ALLOWED_RELATION = "same_intent"

# LLM 置信度最低要求
MIN_CONFIDENCE = 0.90

# embedding 已经在上一步限制 >= 0.88
MIN_SIMILARITY = 0.88


# ============================================================
# 工具函数
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


def make_issue_key(row, row_index):

    source_candidate_id = clean_text(
        row.get("source_candidate_id")
    )

    issue_seq = clean_text(
        row.get("issue_seq")
    )

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


def pair_key(a, b):

    if a <= b:
        return (a, b)

    return (b, a)


def join_unique(values):

    result = []

    for value in values:

        text = clean_text(value)

        if not text:
            continue

        if text not in result:
            result.append(text)

    return " | ".join(result)


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 8.5 - Conservative question clustering"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 读取 issue
    # --------------------------------------------------------

    issue_df = pd.read_excel(
        ISSUE_FILE,
        sheet_name="extracted_issues",
    )

    print(
        f"Issue count: {len(issue_df)}"
    )

    issue_meta = {}

    issue_order = []

    for idx, row in issue_df.iterrows():

        key = make_issue_key(
            row,
            idx,
        )

        issue_order.append(key)

        issue_meta[key] = {
            "question":
                clean_text(
                    row.get("question")
                ),

            "question_normalized":
                clean_text(
                    row.get(
                        "question_normalized"
                    )
                ),

            "answer":
                clean_text(
                    row.get("answer")
                ),

            "solution":
                clean_text(
                    row.get("solution")
                ),

            "crm_module":
                clean_text(
                    row.get("crm_module")
                ),

            "crm_feature":
                clean_text(
                    row.get("crm_feature")
                ),

            "problem_type":
                clean_text(
                    row.get("problem_type")
                ),

            "resolution":
                clean_text(
                    row.get("resolution")
                ),

            "confidence":
                row.get("confidence"),

            "knowledge_value":
                clean_text(
                    row.get(
                        "knowledge_value"
                    )
                ),

            "needs_followup":
                row.get(
                    "needs_followup"
                ),

            "temporal_status":
                clean_text(
                    row.get(
                        "temporal_status"
                    )
                ),

            "source_candidate_id":
                clean_text(
                    row.get(
                        "source_candidate_id"
                    )
                ),
        }

    # --------------------------------------------------------
    # 读取 pair classification
    # --------------------------------------------------------

    pair_df = pd.read_excel(
        PAIR_FILE,
        sheet_name="all_pairs",
    )

    print(
        f"Classified pair count: "
        f"{len(pair_df)}"
    )

    # --------------------------------------------------------
    # 保存所有已分类 pair
    #
    # 注意：
    # 如果两个成员之间没有 pair，
    # 就不能假设它们是 same_intent。
    # --------------------------------------------------------

    pair_lookup = {}

    for _, row in pair_df.iterrows():

        a = clean_text(
            row["issue_key_a"]
        )

        b = clean_text(
            row["issue_key_b"]
        )

        key = pair_key(
            a,
            b,
        )

        pair_lookup[key] = {
            "relation":
                clean_text(
                    row["relation"]
                ),

            "confidence":
                float(
                    row["confidence"]
                ),

            "similarity":
                float(
                    row["similarity"]
                ),

            "canonical_question":
                clean_text(
                    row.get(
                        "canonical_question"
                    )
                ),
        }

    # --------------------------------------------------------
    # 合格 edge
    # --------------------------------------------------------

    eligible_edges = []

    for _, row in pair_df.iterrows():

        relation = clean_text(
            row["relation"]
        )

        confidence = float(
            row["confidence"]
        )

        similarity = float(
            row["similarity"]
        )

        if relation != ALLOWED_RELATION:
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        if similarity < MIN_SIMILARITY:
            continue

        eligible_edges.append(
            {
                "a":
                    clean_text(
                        row["issue_key_a"]
                    ),

                "b":
                    clean_text(
                        row["issue_key_b"]
                    ),

                "similarity":
                    similarity,

                "confidence":
                    confidence,

                "canonical_question":
                    clean_text(
                        row.get(
                            "canonical_question"
                        )
                    ),
            }
        )

    # 优先处理最可靠的 edge
    eligible_edges.sort(
        key=lambda x: (
            x["confidence"],
            x["similarity"],
        ),
        reverse=True,
    )

    print(
        f"Eligible same_intent edges: "
        f"{len(eligible_edges)}"
    )

    # --------------------------------------------------------
    # 初始 cluster
    # --------------------------------------------------------

    clusters = {}

    issue_to_cluster = {}

    next_cluster_number = 1

    for key in issue_order:

        cluster_id = (
            f"TMP-{next_cluster_number:05d}"
        )

        next_cluster_number += 1

        clusters[cluster_id] = {
            key
        }

        issue_to_cluster[key] = (
            cluster_id
        )

    # --------------------------------------------------------
    # Clique 检查
    # --------------------------------------------------------

    def can_merge(
        members_a,
        members_b,
    ):
        """
        两个 cluster 合并的必要条件：

        两个 cluster 的所有跨组 member pair
        都必须满足：
        relation = same_intent
        confidence >= MIN_CONFIDENCE
        similarity >= MIN_SIMILARITY
        """

        for a in members_a:

            for b in members_b:

                if a == b:
                    continue

                info = pair_lookup.get(
                    pair_key(a, b)
                )

                if info is None:
                    return (
                        False,
                        "missing_pair",
                        a,
                        b,
                    )

                if (
                    info["relation"]
                    != ALLOWED_RELATION
                ):
                    return (
                        False,
                        "relation_not_same",
                        a,
                        b,
                    )

                if (
                    info["confidence"]
                    < MIN_CONFIDENCE
                ):
                    return (
                        False,
                        "low_confidence",
                        a,
                        b,
                    )

                if (
                    info["similarity"]
                    < MIN_SIMILARITY
                ):
                    return (
                        False,
                        "low_similarity",
                        a,
                        b,
                    )

        return (
            True,
            "",
            "",
            "",
        )

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    accepted_edge_rows = []
    blocked_edge_rows = []

    for edge in eligible_edges:

        a = edge["a"]
        b = edge["b"]

        cluster_a = (
            issue_to_cluster[a]
        )

        cluster_b = (
            issue_to_cluster[b]
        )

        # 已经同组
        if cluster_a == cluster_b:

            accepted_edge_rows.append(
                {
                    **edge,
                    "action":
                        "already_same_cluster",
                }
            )

            continue

        members_a = clusters[
            cluster_a
        ]

        members_b = clusters[
            cluster_b
        ]

        (
            allowed,
            reason,
            conflict_a,
            conflict_b,
        ) = can_merge(
            members_a,
            members_b,
        )

        if not allowed:

            blocked_edge_rows.append(
                {
                    **edge,

                    "cluster_a":
                        cluster_a,

                    "cluster_b":
                        cluster_b,

                    "cluster_a_size":
                        len(members_a),

                    "cluster_b_size":
                        len(members_b),

                    "blocked_reason":
                        reason,

                    "conflict_issue_a":
                        conflict_a,

                    "conflict_issue_b":
                        conflict_b,
                }
            )

            continue

        # ----------------------------------------------------
        # merge B -> A
        # ----------------------------------------------------

        merged_members = (
            members_a
            | members_b
        )

        clusters[
            cluster_a
        ] = merged_members

        del clusters[
            cluster_b
        ]

        for member in merged_members:

            issue_to_cluster[
                member
            ] = cluster_a

        accepted_edge_rows.append(
            {
                **edge,
                "action":
                    "merged",
            }
        )

    # --------------------------------------------------------
    # 正式 cluster id
    #
    # 单成员不算重复 cluster。
    # 多成员才分配 QUESTION-CLUSTER ID。
    # --------------------------------------------------------

    multi_clusters = [
        members
        for members in clusters.values()
        if len(members) >= 2
    ]

    # 大 cluster 优先
    multi_clusters.sort(
        key=lambda members: (
            -len(members),
            sorted(members)[0],
        )
    )

    final_cluster_map = {}

    for idx, members in enumerate(
        multi_clusters,
        start=1,
    ):

        cluster_id = (
            f"QCLUSTER-{idx:04d}"
        )

        final_cluster_map[
            cluster_id
        ] = members

    # --------------------------------------------------------
    # cluster summary
    # --------------------------------------------------------

    cluster_rows = []

    member_rows = []

    for cluster_id, members in (
        final_cluster_map.items()
    ):

        members = sorted(
            members
        )

        metas = [
            issue_meta[m]
            for m in members
        ]

        questions = [
            m[
                "question_normalized"
            ]
            for m in metas
        ]

        answers = [
            m["answer"]
            for m in metas
        ]

        # ----------------------------------------------------
        # cluster 内所有 pair 的最低 similarity/confidence
        # ----------------------------------------------------

        internal_similarities = []
        internal_confidences = []

        canonical_candidates = []

        for i in range(
            len(members)
        ):

            for j in range(
                i + 1,
                len(members),
            ):

                info = pair_lookup.get(
                    pair_key(
                        members[i],
                        members[j],
                    )
                )

                if not info:
                    continue

                internal_similarities.append(
                    info[
                        "similarity"
                    ]
                )

                internal_confidences.append(
                    info[
                        "confidence"
                    ]
                )

                canonical = (
                    info[
                        "canonical_question"
                    ]
                )

                if canonical:
                    canonical_candidates.append(
                        canonical
                    )

        # ----------------------------------------------------
        # canonical 先用 LLM pair 中出现频率最高的
        # 这里只做候选，不代表最终知识标题
        # ----------------------------------------------------

        if canonical_candidates:

            canonical_question = (
                Counter(
                    canonical_candidates
                )
                .most_common(1)[0][0]
            )

        else:

            canonical_question = (
                questions[0]
                if questions
                else ""
            )

        cluster_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "member_count":
                    len(members),

                "canonical_question_candidate":
                    canonical_question,

                "question_unique_count":
                    len(
                        set(questions)
                    ),

                "answer_unique_count":
                    len(
                        set(
                            x
                            for x in answers
                            if x
                        )
                    ),

                "min_internal_similarity":
                    (
                        min(
                            internal_similarities
                        )
                        if internal_similarities
                        else None
                    ),

                "avg_internal_similarity":
                    (
                        sum(
                            internal_similarities
                        )
                        /
                        len(
                            internal_similarities
                        )
                        if internal_similarities
                        else None
                    ),

                "min_llm_confidence":
                    (
                        min(
                            internal_confidences
                        )
                        if internal_confidences
                        else None
                    ),

                "crm_module_set":
                    join_unique(
                        [
                            m[
                                "crm_module"
                            ]
                            for m in metas
                        ]
                    ),

                "crm_feature_set":
                    join_unique(
                        [
                            m[
                                "crm_feature"
                            ]
                            for m in metas
                        ]
                    ),

                "resolution_set":
                    join_unique(
                        [
                            m[
                                "resolution"
                            ]
                            for m in metas
                        ]
                    ),

                "knowledge_value_set":
                    join_unique(
                        [
                            m[
                                "knowledge_value"
                            ]
                            for m in metas
                        ]
                    ),

                "temporal_status_set":
                    join_unique(
                        [
                            m[
                                "temporal_status"
                            ]
                            for m in metas
                        ]
                    ),

                "questions_preview":
                    " || ".join(
                        list(
                            dict.fromkeys(
                                questions
                            )
                        )[:8]
                    ),
            }
        )

        for member in members:

            meta = issue_meta[
                member
            ]

            member_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "issue_key":
                        member,

                    **meta,
                }
            )

    cluster_df = pd.DataFrame(
        cluster_rows
    )

    member_df = pd.DataFrame(
        member_rows
    )

    accepted_df = pd.DataFrame(
        accepted_edge_rows
    )

    blocked_df = pd.DataFrame(
        blocked_edge_rows
    )

    # --------------------------------------------------------
    # cluster size distribution
    # --------------------------------------------------------

    if not cluster_df.empty:

        size_stats = (
            cluster_df[
                "member_count"
            ]
            .value_counts()
            .sort_index()
            .rename_axis(
                "cluster_size"
            )
            .reset_index(
                name="cluster_count"
            )
        )

    else:

        size_stats = pd.DataFrame(
            columns=[
                "cluster_size",
                "cluster_count",
            ]
        )

    clustered_member_count = (
        len(member_df)
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "total_issue_count",
                "value":
                    len(issue_df),
            },
            {
                "metric":
                    "classified_pair_count",
                "value":
                    len(pair_df),
            },
            {
                "metric":
                    "eligible_same_intent_edge_count",
                "value":
                    len(
                        eligible_edges
                    ),
            },
            {
                "metric":
                    "question_cluster_count",
                "value":
                    len(
                        cluster_df
                    ),
            },
            {
                "metric":
                    "clustered_issue_count",
                "value":
                    clustered_member_count,
            },
            {
                "metric":
                    "singleton_issue_count",
                "value":
                    (
                        len(issue_df)
                        -
                        clustered_member_count
                    ),
            },
            {
                "metric":
                    "largest_cluster_size",
                "value":
                    (
                        int(
                            cluster_df[
                                "member_count"
                            ].max()
                        )
                        if not cluster_df.empty
                        else 0
                    ),
            },
            {
                "metric":
                    "blocked_edge_count",
                "value":
                    len(
                        blocked_df
                    ),
            },
            {
                "metric":
                    "min_similarity",
                "value":
                    MIN_SIMILARITY,
            },
            {
                "metric":
                    "min_confidence",
                "value":
                    MIN_CONFIDENCE,
            },
        ]
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

        size_stats.to_excel(
            writer,
            sheet_name="cluster_size_stats",
            index=False,
        )

        cluster_df.to_excel(
            writer,
            sheet_name="clusters",
            index=False,
        )

        member_df.to_excel(
            writer,
            sheet_name="cluster_members",
            index=False,
        )

        accepted_df.to_excel(
            writer,
            sheet_name="accepted_edges",
            index=False,
        )

        blocked_df.to_excel(
            writer,
            sheet_name="blocked_edges",
            index=False,
        )

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Cluster size distribution:"
    )

    print(
        size_stats.to_string(
            index=False
        )
    )

    print()
    print(
        f"输出文件: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()