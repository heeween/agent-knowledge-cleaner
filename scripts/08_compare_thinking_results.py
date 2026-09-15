from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache
import re

import pandas as pd


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

ON_FILE = OUTPUT_DIR / "thinking_on_test.xlsx"
OFF_FILE = OUTPUT_DIR / "thinking_off_test.xlsx"

OUT_FILE = OUTPUT_DIR / "thinking_comparison_v2.xlsx"


# issue 匹配最低相似度。
# 低于这个值，认为 ON/OFF 没有对应 issue。
MATCH_THRESHOLD = 0.35


# ============================================================
# 基础工具
# ============================================================

def normalize_value(value):
    """
    Excel 中的 NaN / None 转为空字符串。
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_compare_text(value):
    """
    用于字符串近似匹配：
    - 去空格
    - 去常见标点
    - 小写
    """
    text = normalize_value(value).lower()

    text = re.sub(
        r"[\s，。！？、；：,.!?;:"
        r"（）()\[\]【】<>《》"
        r"“”\"'‘’\-_/]+",
        "",
        text,
    )

    return text


def sequence_similarity(a, b):
    """
    字符序列相似度。
    """
    a = normalize_compare_text(a)
    b = normalize_compare_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def char_bigram_similarity(a, b):
    """
    中文文本使用字符 bigram Jaccard，
    对“表达不同但关键词相近”的问题比纯字符比较更稳。
    """
    a = normalize_compare_text(a)
    b = normalize_compare_text(b)

    if not a or not b:
        return 0.0

    if len(a) == 1 or len(b) == 1:
        return 1.0 if a == b else 0.0

    a_set = {
        a[i:i + 2]
        for i in range(len(a) - 1)
    }

    b_set = {
        b[i:i + 2]
        for i in range(len(b) - 1)
    }

    if not a_set or not b_set:
        return 0.0

    intersection = len(
        a_set & b_set
    )

    union = len(
        a_set | b_set
    )

    if union == 0:
        return 0.0

    return intersection / union


def text_similarity(a, b):
    """
    最终文本相似度。
    """
    seq = sequence_similarity(a, b)
    bigram = char_bigram_similarity(a, b)

    return (
        seq * 0.65
        + bigram * 0.35
    )


def same_value(a, b):
    return (
        normalize_value(a)
        == normalize_value(b)
    )


def bool_value(value):
    """
    将 Excel 中的各种 bool 表示转换为 bool。
    """
    if isinstance(value, bool):
        return value

    text = normalize_value(value).lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "",
    }:
        return False

    return False


# ============================================================
# Workbook 读取
# ============================================================

def read_sheet_if_exists(
    path,
    sheet_name,
):
    excel = pd.ExcelFile(path)

    if sheet_name not in excel.sheet_names:
        return pd.DataFrame()

    return pd.read_excel(
        path,
        sheet_name=sheet_name,
    )


def detect_candidate_column(df):
    """
    兼容不同版本字段名。
    """
    candidates = [
        "source_candidate_id",
        "candidate_id",
        "issue_id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_workbook(path):
    """
    同时读取：
    - extracted_issues
    - discarded_candidates

    这样 useful=False / issues=0 的 candidate
    也不会消失。
    """

    extracted = read_sheet_if_exists(
        path,
        "extracted_issues",
    )

    discarded = read_sheet_if_exists(
        path,
        "discarded_candidates",
    )

    # --------------------------------------------------------
    # extracted issues
    # --------------------------------------------------------

    if not extracted.empty:

        candidate_col = detect_candidate_column(
            extracted
        )

        if candidate_col is None:
            raise RuntimeError(
                f"{path.name} 的 extracted_issues "
                "找不到 candidate ID 字段"
            )

        if (
            candidate_col
            != "source_candidate_id"
        ):
            extracted = extracted.rename(
                columns={
                    candidate_col:
                    "source_candidate_id"
                }
            )

    # --------------------------------------------------------
    # discarded candidates
    # --------------------------------------------------------

    if not discarded.empty:

        candidate_col = detect_candidate_column(
            discarded
        )

        if candidate_col is None:
            raise RuntimeError(
                f"{path.name} 的 discarded_candidates "
                "找不到 candidate ID 字段"
            )

        if (
            candidate_col
            != "source_candidate_id"
        ):
            discarded = discarded.rename(
                columns={
                    candidate_col:
                    "source_candidate_id"
                }
            )

    # --------------------------------------------------------
    # 所有 candidate
    # --------------------------------------------------------

    candidate_ids = set()

    if (
        not extracted.empty
        and "source_candidate_id"
        in extracted.columns
    ):
        candidate_ids.update(
            extracted[
                "source_candidate_id"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    if (
        not discarded.empty
        and "source_candidate_id"
        in discarded.columns
    ):
        candidate_ids.update(
            discarded[
                "source_candidate_id"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    return {
        "extracted": extracted,
        "discarded": discarded,
        "candidate_ids": candidate_ids,
    }


# ============================================================
# Issue 内容
# ============================================================

ISSUE_FIELDS = [
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


def ensure_issue_columns(df):
    df = df.copy()

    for col in ISSUE_FIELDS:
        if col not in df.columns:
            df[col] = None

    return df


def get_candidate_issues(
    extracted_df,
    candidate_id,
):
    if extracted_df.empty:
        return []

    subset = extracted_df[
        extracted_df[
            "source_candidate_id"
        ].astype(str)
        == str(candidate_id)
    ].copy()

    subset = ensure_issue_columns(
        subset
    )

    return subset.to_dict(
        orient="records"
    )


# ============================================================
# Issue 匹配
# ============================================================

def issue_similarity(
    on_issue,
    off_issue,
):
    """
    只主要依据问题文本匹配。

    不使用 crm_module / resolution 等字段作为
    主要匹配条件，否则会掩盖我们真正想发现的分类差异。
    """

    on_normalized = normalize_value(
        on_issue.get(
            "question_normalized"
        )
    )

    off_normalized = normalize_value(
        off_issue.get(
            "question_normalized"
        )
    )

    normalized_score = text_similarity(
        on_normalized,
        off_normalized,
    )

    question_score = text_similarity(
        on_issue.get("question"),
        off_issue.get("question"),
    )

    description_score = text_similarity(
        on_issue.get("description"),
        off_issue.get("description"),
    )

    # question_normalized 权重最高
    score = (
        normalized_score * 0.60
        + question_score * 0.30
        + description_score * 0.10
    )

    return score


def best_issue_matching(
    on_issues,
    off_issues,
):
    """
    使用动态规划寻找一对一的最大总相似度匹配。

    Candidate 内通常只有 1~3 个 issue，
    所以这里不需要 scipy，也不需要额外依赖。
    """

    n = len(on_issues)
    m = len(off_issues)

    if n == 0 or m == 0:
        return []

    scores = [
        [
            issue_similarity(
                on_issues[i],
                off_issues[j],
            )
            for j in range(m)
        ]
        for i in range(n)
    ]

    @lru_cache(maxsize=None)
    def solve(
        on_index,
        used_mask,
    ):
        if on_index >= n:
            return (
                0.0,
                (),
            )

        # 方案 1：
        # 当前 ON issue 不匹配
        best_score, best_pairs = solve(
            on_index + 1,
            used_mask,
        )

        # 方案 2：
        # 与某个 OFF issue 匹配
        for off_index in range(m):

            if (
                used_mask
                & (1 << off_index)
            ):
                continue

            similarity = scores[
                on_index
            ][
                off_index
            ]

            if (
                similarity
                < MATCH_THRESHOLD
            ):
                continue

            next_score, next_pairs = (
                solve(
                    on_index + 1,
                    used_mask
                    | (1 << off_index),
                )
            )

            candidate_score = (
                similarity
                + next_score
            )

            if (
                candidate_score
                > best_score
            ):
                best_score = (
                    candidate_score
                )

                best_pairs = (
                    (
                        on_index,
                        off_index,
                        similarity,
                    ),
                ) + next_pairs

        return (
            best_score,
            best_pairs,
        )

    _, pairs = solve(
        0,
        0,
    )

    return list(pairs)


# ============================================================
# 对比逻辑
# ============================================================

def compare_issue_pair(
    candidate_id,
    on_index,
    off_index,
    on_issue,
    off_issue,
    similarity,
):

    row = {
        "source_candidate_id":
            candidate_id,

        "on_issue_index":
            on_index + 1
            if on_index is not None
            else None,

        "off_issue_index":
            off_index + 1
            if off_index is not None
            else None,

        "match_similarity":
            similarity,

        "match_status":
            "matched"
            if (
                on_issue is not None
                and off_issue is not None
            )
            else (
                "on_only"
                if on_issue is not None
                else "off_only"
            ),
    }

    # --------------------------------------------------------
    # ON / OFF 原始字段
    # --------------------------------------------------------

    for field in ISSUE_FIELDS:

        row[f"on_{field}"] = (
            on_issue.get(field)
            if on_issue
            else None
        )

        row[f"off_{field}"] = (
            off_issue.get(field)
            if off_issue
            else None
        )

    # --------------------------------------------------------
    # 没有匹配时
    # --------------------------------------------------------

    if (
        on_issue is None
        or off_issue is None
    ):
        row[
            "question_semantic_similarity"
        ] = 0.0

        row[
            "answer_semantic_similarity"
        ] = 0.0

        row["problem_type_same"] = False
        row["crm_module_same"] = False
        row["crm_feature_same"] = False
        row["resolution_same"] = False
        row["knowledge_value_same"] = False
        row["needs_followup_same"] = False
        row["temporal_status_same"] = False

        row["confidence_diff"] = None

        row["critical_same_count"] = 0
        row["critical_same_ratio"] = 0.0

        row["risk_level"] = "high"

        return row

    # --------------------------------------------------------
    # 文本语义相似度
    # --------------------------------------------------------

    row[
        "question_semantic_similarity"
    ] = text_similarity(
        on_issue.get(
            "question_normalized"
        ),
        off_issue.get(
            "question_normalized"
        ),
    )

    row[
        "answer_semantic_similarity"
    ] = text_similarity(
        on_issue.get("answer"),
        off_issue.get("answer"),
    )

    # --------------------------------------------------------
    # 分类字段
    # --------------------------------------------------------

    compare_fields = [
        "problem_type",
        "crm_module",
        "crm_feature",
        "resolution",
        "knowledge_value",
        "needs_followup",
        "temporal_status",
    ]

    for field in compare_fields:

        row[
            f"{field}_same"
        ] = same_value(
            on_issue.get(field),
            off_issue.get(field),
        )

    # --------------------------------------------------------
    # confidence
    # --------------------------------------------------------

    try:
        on_conf = float(
            on_issue.get(
                "confidence"
            )
        )

        off_conf = float(
            off_issue.get(
                "confidence"
            )
        )

        row[
            "confidence_diff"
        ] = (
            off_conf - on_conf
        )

    except (
        TypeError,
        ValueError,
    ):
        row[
            "confidence_diff"
        ] = None

    # --------------------------------------------------------
    # 核心字段一致率
    # --------------------------------------------------------

    critical_fields = [
        "crm_module_same",
        "resolution_same",
        "knowledge_value_same",
        "needs_followup_same",
        "temporal_status_same",
    ]

    same_count = sum(
        1
        for field in critical_fields
        if row[field]
    )

    row[
        "critical_same_count"
    ] = same_count

    row[
        "critical_same_ratio"
    ] = (
        same_count
        / len(critical_fields)
    )

    # --------------------------------------------------------
    # 风险等级
    # --------------------------------------------------------

    # 以下几项直接影响是否能进入最终 RAG，
    # 所以权重比文字是否完全一致更高。
    severe_difference = any(
        not row[field]
        for field in [
            "resolution_same",
            "knowledge_value_same",
            "needs_followup_same",
            "temporal_status_same",
        ]
    )

    if (
        row[
            "question_semantic_similarity"
        ] < 0.45
    ):
        row[
            "risk_level"
        ] = "high"

    elif severe_difference:
        row[
            "risk_level"
        ] = "high"

    elif not row[
        "crm_module_same"
    ]:
        row[
            "risk_level"
        ] = "medium"

    elif row[
        "answer_semantic_similarity"
    ] < 0.45:
        row[
            "risk_level"
        ] = "medium"

    else:
        row[
            "risk_level"
        ] = "low"

    return row


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print(
        "Thinking ON / OFF Comparison V2"
    )
    print("=" * 72)

    print()
    print(
        f"ON : {ON_FILE}"
    )
    print(
        f"OFF: {OFF_FILE}"
    )

    on_data = load_workbook(
        ON_FILE
    )

    off_data = load_workbook(
        OFF_FILE
    )

    # 两边 candidate 取并集
    all_candidate_ids = sorted(
        on_data[
            "candidate_ids"
        ]
        | off_data[
            "candidate_ids"
        ]
    )

    print()
    print(
        f"Candidate 总数: "
        f"{len(all_candidate_ids)}"
    )

    candidate_rows = []
    issue_rows = []

    # ========================================================
    # Candidate 逐条比较
    # ========================================================

    for candidate_id in (
        all_candidate_ids
    ):

        on_issues = (
            get_candidate_issues(
                on_data["extracted"],
                candidate_id,
            )
        )

        off_issues = (
            get_candidate_issues(
                off_data["extracted"],
                candidate_id,
            )
        )

        on_count = len(
            on_issues
        )

        off_count = len(
            off_issues
        )

        on_useful = (
            on_count > 0
        )

        off_useful = (
            off_count > 0
        )

        # ----------------------------------------------------
        # Issue 智能匹配
        # ----------------------------------------------------

        matches = (
            best_issue_matching(
                on_issues,
                off_issues,
            )
        )

        matched_on = {
            on_idx
            for (
                on_idx,
                _,
                _,
            ) in matches
        }

        matched_off = {
            off_idx
            for (
                _,
                off_idx,
                _,
            ) in matches
        }

        # 匹配成功
        for (
            on_idx,
            off_idx,
            score,
        ) in matches:

            issue_rows.append(
                compare_issue_pair(
                    candidate_id,
                    on_idx,
                    off_idx,
                    on_issues[
                        on_idx
                    ],
                    off_issues[
                        off_idx
                    ],
                    score,
                )
            )

        # ON 独有
        for on_idx, on_issue in (
            enumerate(on_issues)
        ):

            if (
                on_idx
                in matched_on
            ):
                continue

            issue_rows.append(
                compare_issue_pair(
                    candidate_id,
                    on_idx,
                    None,
                    on_issue,
                    None,
                    0.0,
                )
            )

        # OFF 独有
        for off_idx, off_issue in (
            enumerate(off_issues)
        ):

            if (
                off_idx
                in matched_off
            ):
                continue

            issue_rows.append(
                compare_issue_pair(
                    candidate_id,
                    None,
                    off_idx,
                    None,
                    off_issue,
                    0.0,
                )
            )

        candidate_rows.append(
            {
                "source_candidate_id":
                    candidate_id,

                "on_useful":
                    on_useful,

                "off_useful":
                    off_useful,

                "useful_same":
                    (
                        on_useful
                        == off_useful
                    ),

                "on_issue_count":
                    on_count,

                "off_issue_count":
                    off_count,

                "issue_count_same":
                    (
                        on_count
                        == off_count
                    ),

                "matched_issue_count":
                    len(matches),

                "on_unmatched_count":
                    (
                        on_count
                        - len(
                            matched_on
                        )
                    ),

                "off_unmatched_count":
                    (
                        off_count
                        - len(
                            matched_off
                        )
                    ),
            }
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    issue_df = pd.DataFrame(
        issue_rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    candidate_count = len(
        candidate_df
    )

    def add_summary(
        metric,
        value,
    ):
        summary_rows.append(
            {
                "metric": metric,
                "value": value,
            }
        )

    add_summary(
        "candidate_count",
        candidate_count,
    )

    if candidate_count:

        add_summary(
            "useful_same_ratio",
            candidate_df[
                "useful_same"
            ].mean(),
        )

        add_summary(
            "issue_count_same_ratio",
            candidate_df[
                "issue_count_same"
            ].mean(),
        )

        add_summary(
            "on_total_issue_count",
            candidate_df[
                "on_issue_count"
            ].sum(),
        )

        add_summary(
            "off_total_issue_count",
            candidate_df[
                "off_issue_count"
            ].sum(),
        )

    matched_df = issue_df[
        issue_df[
            "match_status"
        ] == "matched"
    ].copy()

    add_summary(
        "matched_issue_count",
        len(matched_df),
    )

    if len(matched_df):

        add_summary(
            "avg_match_similarity",
            matched_df[
                "match_similarity"
            ].mean(),
        )

        add_summary(
            "avg_question_similarity",
            matched_df[
                "question_semantic_similarity"
            ].mean(),
        )

        add_summary(
            "avg_answer_similarity",
            matched_df[
                "answer_semantic_similarity"
            ].mean(),
        )

        for field in [
            "problem_type_same",
            "crm_module_same",
            "resolution_same",
            "knowledge_value_same",
            "needs_followup_same",
            "temporal_status_same",
        ]:

            add_summary(
                f"{field}_ratio",
                matched_df[
                    field
                ].mean(),
            )

        add_summary(
            "avg_critical_same_ratio",
            matched_df[
                "critical_same_ratio"
            ].mean(),
        )

        add_summary(
            "high_risk_issue_count",
            (
                matched_df[
                    "risk_level"
                ]
                == "high"
            ).sum(),
        )

        add_summary(
            "medium_risk_issue_count",
            (
                matched_df[
                    "risk_level"
                ]
                == "medium"
            ).sum(),
        )

        add_summary(
            "low_risk_issue_count",
            (
                matched_df[
                    "risk_level"
                ]
                == "low"
            ).sum(),
        )

    unmatched_count = (
        issue_df[
            "match_status"
        ]
        != "matched"
    ).sum()

    add_summary(
        "unmatched_issue_count",
        unmatched_count,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # 人工 Review
    # ========================================================

    if not issue_df.empty:

        review_df = issue_df[
            (
                issue_df[
                    "risk_level"
                ].isin(
                    [
                        "high",
                        "medium",
                    ]
                )
            )
            |
            (
                issue_df[
                    "match_status"
                ]
                != "matched"
            )
        ].copy()

        risk_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
        }

        review_df[
            "_risk_order"
        ] = (
            review_df[
                "risk_level"
            ]
            .map(
                risk_order
            )
            .fillna(9)
        )

        review_df = (
            review_df.sort_values(
                [
                    "_risk_order",
                    "source_candidate_id",
                    "match_similarity",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
            )
            .drop(
                columns=[
                    "_risk_order"
                ]
            )
        )

    else:
        review_df = pd.DataFrame()

    # ========================================================
    # Candidate 有差异
    # ========================================================

    candidate_diff_df = (
        candidate_df[
            (
                ~candidate_df[
                    "useful_same"
                ]
            )
            |
            (
                ~candidate_df[
                    "issue_count_same"
                ]
            )
            |
            (
                candidate_df[
                    "on_unmatched_count"
                ] > 0
            )
            |
            (
                candidate_df[
                    "off_unmatched_count"
                ] > 0
            )
        ]
        .copy()
    )

    # ========================================================
    # 导出
    # ========================================================

    with pd.ExcelWriter(
        OUT_FILE,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        candidate_df.to_excel(
            writer,
            sheet_name="candidate_compare",
            index=False,
        )

        candidate_diff_df.to_excel(
            writer,
            sheet_name="candidate_diff",
            index=False,
        )

        issue_df.to_excel(
            writer,
            sheet_name="issue_compare",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review_diff",
            index=False,
        )

    # ========================================================
    # 控制台输出
    # ========================================================

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for _, row in (
        summary_df.iterrows()
    ):
        value = row["value"]

        if isinstance(
            value,
            float,
        ):
            print(
                f"{row['metric']:<35}"
                f"{value:.4f}"
            )
        else:
            print(
                f"{row['metric']:<35}"
                f"{value}"
            )

    print()
    print(
        f"输出文件：{OUT_FILE}"
    )


if __name__ == "__main__":
    main()