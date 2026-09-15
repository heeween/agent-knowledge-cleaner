from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

EMBEDDING_FILE = OUTPUT_DIR / "question_embeddings.jsonl"
ISSUE_FILE = OUTPUT_DIR / "extracted_issues.xlsx"

OUTPUT_FILE = OUTPUT_DIR / "similar_question_pairs.xlsx"

# 只输出 >= 这个阈值的 pair
MIN_SIMILARITY = 0.75

# 每个 issue 最多保留多少个近邻
TOP_K_PER_ISSUE = 20


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
        return f"{source_candidate_id}#{issue_seq}"

    if source_candidate_id:
        return (
            f"{source_candidate_id}"
            f"#ROW-{row_index + 1:05d}"
        )

    return f"ROW-{row_index + 1:05d}"


# ============================================================
# 读取 embedding
# ============================================================

def load_embeddings():
    records = []

    with EMBEDDING_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            records.append(obj)

    return records


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Step 8.3 - Find similar question pairs")
    print("=" * 70)

    embedding_records = load_embeddings()

    print(
        f"Embedding record count: "
        f"{len(embedding_records)}"
    )

    if not embedding_records:
        raise RuntimeError(
            "没有读取到 embedding 数据"
        )

    # --------------------------------------------------------
    # 检查向量维度
    # --------------------------------------------------------

    dimensions = {
        len(x["embedding"])
        for x in embedding_records
    }

    print(
        f"Embedding dimensions: "
        f"{sorted(dimensions)}"
    )

    if len(dimensions) != 1:
        raise RuntimeError(
            f"存在不同维度 embedding: "
            f"{dimensions}"
        )

    # --------------------------------------------------------
    # issue 元数据
    # --------------------------------------------------------

    issue_df = pd.read_excel(
        ISSUE_FILE,
        sheet_name="extracted_issues",
    )

    issue_meta = {}

    for idx, row in issue_df.iterrows():

        issue_key = make_issue_key(
            row,
            idx,
        )

        issue_meta[issue_key] = {
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

            "knowledge_value":
                clean_text(
                    row.get("knowledge_value")
                ),

            "temporal_status":
                clean_text(
                    row.get("temporal_status")
                ),

            "confidence":
                row.get("confidence"),
        }

    # --------------------------------------------------------
    # 构造矩阵
    # --------------------------------------------------------

    issue_keys = [
        x["issue_key"]
        for x in embedding_records
    ]

    questions = [
        x["question_normalized"]
        for x in embedding_records
    ]

    matrix = np.array(
        [
            x["embedding"]
            for x in embedding_records
        ],
        dtype=np.float32,
    )

    print(
        f"Matrix shape: {matrix.shape}"
    )

    # --------------------------------------------------------
    # L2 normalize
    # cosine similarity = dot product
    # --------------------------------------------------------

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    matrix = matrix / norms

    # 4095 x 4095，大约 67MB float32
    similarity = matrix @ matrix.T

    print(
        "Cosine similarity matrix calculated."
    )

    # --------------------------------------------------------
    # 找近邻 pair
    # --------------------------------------------------------

    pair_rows = []

    n = len(issue_keys)

    for i in range(n):

        scores = similarity[i]

        # 自己不要
        scores[i] = -1.0

        # 先取 top-k，避免输出量爆炸
        top_indices = np.argpartition(
            scores,
            -TOP_K_PER_ISSUE,
        )[-TOP_K_PER_ISSUE:]

        top_indices = top_indices[
            np.argsort(
                scores[top_indices]
            )[::-1]
        ]

        for j in top_indices:

            score = float(
                scores[j]
            )

            if score < MIN_SIMILARITY:
                continue

            # 只记录一次 pair
            if i >= j:
                continue

            key_a = issue_keys[i]
            key_b = issue_keys[j]

            meta_a = issue_meta.get(
                key_a,
                {},
            )

            meta_b = issue_meta.get(
                key_b,
                {},
            )

            question_a = questions[i]
            question_b = questions[j]

            exact_same_question = (
                question_a == question_b
            )

            same_module = (
                meta_a.get("crm_module")
                ==
                meta_b.get("crm_module")
            )

            same_feature = (
                meta_a.get("crm_feature")
                ==
                meta_b.get("crm_feature")
            )

            same_problem_type = (
                meta_a.get("problem_type")
                ==
                meta_b.get("problem_type")
            )

            pair_rows.append(
                {
                    "issue_key_a":
                        key_a,

                    "issue_key_b":
                        key_b,

                    "similarity":
                        score,

                    "question_a":
                        meta_a.get(
                            "question",
                            "",
                        ),

                    "question_b":
                        meta_b.get(
                            "question",
                            "",
                        ),

                    "question_normalized_a":
                        question_a,

                    "question_normalized_b":
                        question_b,

                    "exact_same_question":
                        exact_same_question,

                    "crm_module_a":
                        meta_a.get(
                            "crm_module",
                            "",
                        ),

                    "crm_module_b":
                        meta_b.get(
                            "crm_module",
                            "",
                        ),

                    "same_module":
                        same_module,

                    "crm_feature_a":
                        meta_a.get(
                            "crm_feature",
                            "",
                        ),

                    "crm_feature_b":
                        meta_b.get(
                            "crm_feature",
                            "",
                        ),

                    "same_feature":
                        same_feature,

                    "problem_type_a":
                        meta_a.get(
                            "problem_type",
                            "",
                        ),

                    "problem_type_b":
                        meta_b.get(
                            "problem_type",
                            "",
                        ),

                    "same_problem_type":
                        same_problem_type,

                    "resolution_a":
                        meta_a.get(
                            "resolution",
                            "",
                        ),

                    "resolution_b":
                        meta_b.get(
                            "resolution",
                            "",
                        ),

                    "knowledge_value_a":
                        meta_a.get(
                            "knowledge_value",
                            "",
                        ),

                    "knowledge_value_b":
                        meta_b.get(
                            "knowledge_value",
                            "",
                        ),

                    "temporal_status_a":
                        meta_a.get(
                            "temporal_status",
                            "",
                        ),

                    "temporal_status_b":
                        meta_b.get(
                            "temporal_status",
                            "",
                        ),

                    "confidence_a":
                        meta_a.get(
                            "confidence",
                            None,
                        ),

                    "confidence_b":
                        meta_b.get(
                            "confidence",
                            None,
                        ),

                    "answer_a":
                        meta_a.get(
                            "answer",
                            "",
                        ),

                    "answer_b":
                        meta_b.get(
                            "answer",
                            "",
                        ),

                    "solution_a":
                        meta_a.get(
                            "solution",
                            "",
                        ),

                    "solution_b":
                        meta_b.get(
                            "solution",
                            "",
                        ),
                }
            )

    pairs_df = pd.DataFrame(
        pair_rows
    )

    if not pairs_df.empty:
        pairs_df = (
            pairs_df
            .sort_values(
                "similarity",
                ascending=False,
            )
            .drop_duplicates(
                subset=[
                    "issue_key_a",
                    "issue_key_b",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # 阈值统计
    # --------------------------------------------------------

    thresholds = [
        0.75,
        0.80,
        0.82,
        0.85,
        0.88,
        0.90,
        0.92,
        0.95,
        0.97,
        0.99,
    ]

    threshold_rows = []

    for threshold in thresholds:

        if pairs_df.empty:
            count = 0
            non_exact_count = 0
            same_module_count = 0

        else:
            mask = (
                pairs_df["similarity"]
                >= threshold
            )

            subset = pairs_df[
                mask
            ]

            count = len(
                subset
            )

            non_exact_count = (
                (~subset[
                    "exact_same_question"
                ])
                .sum()
            )

            same_module_count = (
                subset[
                    "same_module"
                ]
                .sum()
            )

        threshold_rows.append(
            {
                "threshold":
                    threshold,

                "pair_count":
                    count,

                "non_exact_pair_count":
                    int(
                        non_exact_count
                    ),

                "same_module_pair_count":
                    int(
                        same_module_count
                    ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # --------------------------------------------------------
    # 相似度区间分布
    # --------------------------------------------------------

    bins = [
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.97,
        0.99,
        1.000001,
    ]

    labels = [
        "0.75-0.80",
        "0.80-0.85",
        "0.85-0.90",
        "0.90-0.95",
        "0.95-0.97",
        "0.97-0.99",
        "0.99-1.00",
    ]

    if not pairs_df.empty:

        pairs_df[
            "similarity_bucket"
        ] = pd.cut(
            pairs_df["similarity"],
            bins=bins,
            labels=labels,
            right=False,
        )

        bucket_df = (
            pairs_df[
                "similarity_bucket"
            ]
            .value_counts(
                sort=False
            )
            .rename_axis(
                "similarity_bucket"
            )
            .reset_index(
                name="pair_count"
            )
        )

    else:
        bucket_df = pd.DataFrame(
            {
                "similarity_bucket":
                    labels,
                "pair_count":
                    [0] * len(labels),
            }
        )

    # --------------------------------------------------------
    # 抽取高相似 review 样本
    # --------------------------------------------------------

    if not pairs_df.empty:

        review_df = pairs_df[
            pairs_df["similarity"]
            >= 0.85
        ].copy()

    else:
        review_df = pd.DataFrame()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "embedding_count",
                "value":
                    len(issue_keys),
            },
            {
                "metric":
                    "min_similarity_output",
                "value":
                    MIN_SIMILARITY,
            },
            {
                "metric":
                    "top_k_per_issue",
                "value":
                    TOP_K_PER_ISSUE,
            },
            {
                "metric":
                    "output_pair_count",
                "value":
                    len(pairs_df),
            },
            {
                "metric":
                    "exact_same_question_pair_count",
                "value":
                    (
                        int(
                            pairs_df[
                                "exact_same_question"
                            ].sum()
                        )
                        if not pairs_df.empty
                        else 0
                    ),
            },
        ]
    )

    # --------------------------------------------------------
    # 导出 Excel
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

        threshold_df.to_excel(
            writer,
            sheet_name="threshold_stats",
            index=False,
        )

        bucket_df.to_excel(
            writer,
            sheet_name="similarity_buckets",
            index=False,
        )

        pairs_df.to_excel(
            writer,
            sheet_name="all_pairs",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review_085",
            index=False,
        )

    print()
    print("=" * 70)
    print("THRESHOLD STATS")
    print("=" * 70)

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"输出 pair 数: "
        f"{len(pairs_df)}"
    )

    print(
        f"输出文件: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()