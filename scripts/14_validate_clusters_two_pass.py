import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "cluster_validations_v3.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "cluster_validations_v3.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema - Pass A
# ============================================================

class IntentValidation(BaseModel):

    cluster_validity: Literal[
        "valid",
        "needs_split",
        "uncertain",
    ]

    confidence: float

    canonical_question: str | None = None

    suspicious_issue_keys: list[str]

    reason: str


# ============================================================
# Schema - Pass B
# ============================================================

class AnswerValidation(BaseModel):

    answer_relation: Literal[
        "duplicate",
        "complementary",
        "multiple_causes",
        "incomplete_vs_complete",
        "temporal_versions",
        "conflict",
        "mixed",
        "unknown",
    ]

    confidence: float

    reason: str

    recommended_action: Literal[
        "merge",
        "merge_with_care",
        "manual_review",
    ]


# ============================================================
# Pass A Prompt
# ============================================================

INTENT_SYSTEM_PROMPT = """
你正在审核 CRM RAG 知识库中的问题 Cluster。

重要：
你现在只能判断“用户问题意图是否一致”。

你不会看到答案，
也不允许推测客服可能给出了什么答案。

判断依据只能是：

- question_normalized
- 问题对象
- 问题动作
- 问题范围
- 用户实际想知道什么


============================================================
cluster_validity
============================================================

只能选择：

valid

所有问题本质上是在问同一个可复用知识问题。

例如：

A：短信为什么发送失败？
B：短信发送不出去是什么原因？
C：短信发送失败如何排查？

如果它们都在表达同一个“短信发送失败排查”意图，
可以判 valid。

注意：
故障排查问题允许存在很多不同原因。
不要因为你猜测原因可能不同而拆分。


例如：

A：CRM无法登录怎么办？
B：CRM登录不上如何处理？

应判：

valid

即使现实中可能分别由密码、网址、
浏览器、系统维护等原因造成，
这些都不是拆问题的理由。


needs_split

只有 question_normalized 本身需要不同知识答案时才能拆分。

例如：

A：预计保养里程是什么意思？
B：预计保养里程为什么出现异常？

一个问定义，一个问异常原因。

needs_split


例如：

A：短信发送失败的原因是什么？
B：短信发送失败是否扣费？

一个问原因，一个问计费。

needs_split


例如：

A：培训需要多长时间？
B：培训什么时候安排？

一个问 duration，
一个问 schedule。

needs_split


例如：

A：如何购买短信套餐？
B：短信套餐在哪里买？

如果前者主要询问完整购买流程，
后者只是询问入口，
需要判断两者是否可以由同一知识条目直接覆盖。

如果一个明显只是另一个知识中的一个子点，
可以 needs_split。


uncertain

仅靠问题本身无法可靠判断。


============================================================
关键规则
============================================================

1. 不要根据可能的答案来推断问题意图。

2. 对“为什么失败 / 为什么异常 / 如何排查”
这类 troubleshooting 问题：

只要故障现象和用户目标相同，
即使可能存在不同根因，
通常仍属于同一个知识意图。

3. 对“忘记密码怎么办”：

“管理员重置”
和
“用户自助重置”

属于可能的答案差异。

除非 question 本身明确限定了：

- 管理员如何操作
- 用户自己如何操作

否则不能因为潜在操作路径不同就拆问题。

4. 对“是否支持”与“如何操作”：

如果一个询问能力，
一个询问具体操作，
通常不是完全相同的问题。

5. 对对象不同的情况：

客户 vs 车辆
账号 vs 员工
短信签名 vs 短信套餐

通常应该拆开。

6. 对范围不同的情况：

整体企业微信收费
vs
企业微信某个具体子功能收费

通常应拆开或 uncertain。


============================================================
canonical_question
============================================================

只有 valid 时填写。

要求：

- 独立
- 简洁
- 可搜索
- 是问句
- 不包含具体客户、门店、人名
- 能覆盖 Cluster 全部成员


============================================================
suspicious_issue_keys
============================================================

needs_split 时填写真正意图不同的成员。

valid 时必须 []。


============================================================
输出要求
============================================================

必须只输出合法 JSON object。

不要输出 Markdown。
不要输出 JSON 之外的文字。

JSON：

{
  "cluster_validity": "valid | needs_split | uncertain",
  "confidence": 0.0,
  "canonical_question": "标准问题或 null",
  "suspicious_issue_keys": [],
  "reason": "简短原因"
}
"""


# ============================================================
# Pass B Prompt
# ============================================================

ANSWER_SYSTEM_PROMPT = """
你正在审核 CRM RAG 知识库中同一个问题 Cluster 的答案。

前一步已经确认：
这些 question 属于同一个知识意图。

所以你现在不要重新判断问题是否应该拆分。

你的任务只有一个：

判断不同客服记录中的答案之间是什么关系。


============================================================
answer_relation
============================================================

duplicate

核心事实和解决方式基本一样，
只是措辞不同。


complementary

答案描述同一规则的不同部分，
合在一起更完整。


multiple_causes

同一个故障、异常、失败问题，
不同记录给出了不同可能原因。

例如：

短信发送失败：

- 敏感词
- 余额不足
- 签名未报备
- 运营商限制

这是 multiple_causes。


incomplete_vs_complete

有的回答不完整，
例如：

- 我看看
- 已反馈
- 正在处理中
- 没有最终结论

而其他来源给出了明确答案。


temporal_versions

同一个问题，
产品规则、操作路径或支持状态随时间发生变化。

例如：

旧：
不支持导出。

新：
已经支持导出。


conflict

同一个问题出现无法同时成立的结论，
并且现有时间信息不足以解释为版本变化。

例如：

A：默认密码是手机号。
B：默认密码是姓名首字母+手机号。


mixed

同时存在多种关系。

例如：

部分答案重复，
部分答案补充，
同时还存在一个疑似旧版本答案。


unknown

证据不足。


============================================================
重要原则
============================================================

1. 不要因为答案不同重新把问题拆开。

2. 多个根因不是 conflict，
应该是 multiple_causes。

3. 两种不同解决方法不一定 conflict。

例如：

忘记密码：
- 联系管理员重置
- APP 自助重置

如果两者有可能适用于不同版本或场景，
应结合 temporal_status、上下文判断：
temporal_versions / complementary / conflict。

4. “不支持” vs “支持”
通常属于 conflict 或 temporal_versions。

5. “正在开发” vs “已上线”
通常属于 temporal_versions。

6. 无答案 vs 有完整答案：
incomplete_vs_complete。


============================================================
recommended_action
============================================================

duplicate
complementary
incomplete_vs_complete

→ merge


multiple_causes

→ merge_with_care


temporal_versions
conflict

→ manual_review


mixed

→ merge_with_care 或 manual_review


unknown

→ manual_review


============================================================
输出要求
============================================================

必须只输出合法 JSON object。

不要输出 Markdown。
不要输出 JSON 之外的文字。

JSON：

{
  "answer_relation": "duplicate | complementary | multiple_causes | incomplete_vs_complete | temporal_versions | conflict | mixed | unknown",
  "confidence": 0.0,
  "reason": "简短原因",
  "recommended_action": "merge | merge_with_care | manual_review"
}
"""


# ============================================================
# Client
# ============================================================


def build_client():

    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
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


def load_processed():

    processed = set()

    if not OUTPUT_JSONL.exists():
        return processed

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

                cluster_id = obj.get(
                    "cluster_id"
                )

                if cluster_id:
                    processed.add(
                        cluster_id
                    )

            except Exception:
                continue

    return processed


# ============================================================
# API Helper
# ============================================================

def call_json(
    system_prompt,
    user_prompt,
    schema_class,
):

    client = build_client()

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = (
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                system_prompt,
                        },
                        {
                            "role": "user",
                            "content":
                                user_prompt,
                        },
                    ],
                    response_format={
                        "type": "json_object"
                    },
                    temperature=0,
                    extra_body={
                        "enable_thinking":
                            False
                    },
                )
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            data = json.loads(
                content
            )

            return (
                schema_class
                .model_validate(
                    data
                )
            )

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# Pass A
# ============================================================

def build_intent_prompt(
    cluster_id,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        blocks.append(
            f"""
Issue {idx}

issue_key:
{clean_text(row.get("issue_key"))}

question_normalized:
{clean_text(row.get("question_normalized"))}
""".strip()
        )

    return f"""
请只根据问题文本判断这个 Cluster。

cluster_id:
{cluster_id}

member_count:
{len(members)}

{chr(10).join(blocks)}
"""


# ============================================================
# Pass B
# ============================================================

def build_answer_prompt(
    cluster_id,
    canonical_question,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        blocks.append(
            f"""
Issue {idx}

issue_key:
{clean_text(row.get("issue_key"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

answer:
{clean_text(row.get("answer"))}

solution:
{clean_text(row.get("solution"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}
""".strip()
        )

    return f"""
下面这个 Cluster 已经确认属于同一个问题意图。

cluster_id:
{cluster_id}

canonical_question:
{canonical_question}

请只判断答案关系。

{chr(10).join(blocks)}
"""


# ============================================================
# 处理一个 Cluster
# ============================================================

def validate_cluster(
    cluster_id,
    members,
):

    started = time.time()

    # --------------------------------------------------------
    # Pass A：Question only
    # --------------------------------------------------------

    intent_prompt = build_intent_prompt(
        cluster_id,
        members,
    )

    intent_result = call_json(
        INTENT_SYSTEM_PROMPT,
        intent_prompt,
        IntentValidation,
    )

    # 强制逻辑
    if (
        intent_result.cluster_validity
        != "valid"
    ):
        intent_result.canonical_question = None

    if (
        intent_result.cluster_validity
        == "valid"
    ):
        intent_result.suspicious_issue_keys = []

    # --------------------------------------------------------
    # needs_split / uncertain
    # 不再进行答案分析
    # --------------------------------------------------------

    if (
        intent_result.cluster_validity
        != "valid"
    ):

        return {
            "cluster_id":
                cluster_id,

            "member_count":
                len(members),

            "cluster_validity":
                intent_result.cluster_validity,

            "intent_confidence":
                intent_result.confidence,

            "canonical_question":
                None,

            "intent_reason":
                intent_result.reason,

            "suspicious_issue_keys":
                intent_result
                .suspicious_issue_keys,

            "answer_relation":
                None,

            "answer_confidence":
                None,

            "answer_reason":
                None,

            "recommended_action":
                (
                    "split"
                    if (
                        intent_result
                        .cluster_validity
                        == "needs_split"
                    )
                    else "manual_review"
                ),

            "request_seconds":
                time.time() - started,
        }

    # --------------------------------------------------------
    # Pass B：Answers
    # --------------------------------------------------------

    answer_prompt = build_answer_prompt(
        cluster_id,
        intent_result
        .canonical_question,
        members,
    )

    answer_result = call_json(
        ANSWER_SYSTEM_PROMPT,
        answer_prompt,
        AnswerValidation,
    )

    # --------------------------------------------------------
    # 统一 recommended_action
    # --------------------------------------------------------

    relation = (
        answer_result
        .answer_relation
    )

    if relation in {
        "duplicate",
        "complementary",
        "incomplete_vs_complete",
    }:
        action = "merge"

    elif relation == "multiple_causes":
        action = "merge_with_care"

    elif relation in {
        "conflict",
        "temporal_versions",
        "unknown",
    }:
        action = "manual_review"

    elif relation == "mixed":

        action = (
            answer_result
            .recommended_action
        )

        if action not in {
            "merge_with_care",
            "manual_review",
        }:
            action = "merge_with_care"

    else:
        action = "manual_review"

    return {
        "cluster_id":
            cluster_id,

        "member_count":
            len(members),

        "cluster_validity":
            "valid",

        "intent_confidence":
            intent_result.confidence,

        "canonical_question":
            intent_result
            .canonical_question,

        "intent_reason":
            intent_result.reason,

        "suspicious_issue_keys":
            [],

        "answer_relation":
            answer_result
            .answer_relation,

        "answer_confidence":
            answer_result.confidence,

        "answer_reason":
            answer_result.reason,

        "recommended_action":
            action,

        "request_seconds":
            time.time() - started,
    }


# ============================================================
# 导出
# ============================================================

def export_excel(
    member_df,
):

    rows = []

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception:
                pass

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return

    df[
        "suspicious_issue_keys"
    ] = df[
        "suspicious_issue_keys"
    ].apply(
        lambda x:
            " | ".join(x)
            if isinstance(x, list)
            else clean_text(x)
    )

    validity_stats = (
        df[
            "cluster_validity"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "cluster_validity"
        )
        .reset_index(
            name="count"
        )
    )

    answer_stats = (
        df[
            "answer_relation"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "answer_relation"
        )
        .reset_index(
            name="count"
        )
    )

    action_stats = (
        df[
            "recommended_action"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "recommended_action"
        )
        .reset_index(
            name="count"
        )
    )

    review_df = df[
        (
            df[
                "cluster_validity"
            ]
            != "valid"
        )
        |
        (
            df[
                "intent_confidence"
            ]
            < 0.90
        )
        |
        (
            df[
                "answer_relation"
            ].isin(
                [
                    "conflict",
                    "temporal_versions",
                    "mixed",
                    "unknown",
                ]
            )
        )
    ].copy()

    review_ids = set(
        review_df[
            "cluster_id"
        ].tolist()
    )

    review_members = member_df[
        member_df[
            "cluster_id"
        ].isin(
            review_ids
        )
    ].copy()

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "validated_cluster_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "valid_cluster_count",
                "value":
                    int(
                        (
                            df[
                                "cluster_validity"
                            ]
                            == "valid"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "needs_split_count",
                "value":
                    int(
                        (
                            df[
                                "cluster_validity"
                            ]
                            == "needs_split"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "uncertain_count",
                "value":
                    int(
                        (
                            df[
                                "cluster_validity"
                            ]
                            == "uncertain"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "review_cluster_count",
                "value":
                    len(review_df),
            },
        ]
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        validity_stats.to_excel(
            writer,
            sheet_name="validity_stats",
            index=False,
        )

        answer_stats.to_excel(
            writer,
            sheet_name="answer_relation_stats",
            index=False,
        )

        action_stats.to_excel(
            writer,
            sheet_name="action_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review_clusters",
            index=False,
        )

        review_members.to_excel(
            writer,
            sheet_name="review_members",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 8.6 v3 - Two-pass cluster validation"
    )
    print("=" * 70)

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    cluster_ids = (
        member_df[
            "cluster_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Cluster count: "
        f"{len(cluster_ids)}"
    )

    processed = (
        load_processed()
    )

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for cluster_id in cluster_ids:

        if cluster_id in processed:
            continue

        members = (
            member_df[
                member_df[
                    "cluster_id"
                ]
                == cluster_id
            ]
            .to_dict(
                orient="records"
            )
        )

        tasks.append(
            (
                cluster_id,
                members,
            )
        )

    print(
        f"本次待处理: "
        f"{len(tasks)}"
    )

    if not tasks:

        export_excel(
            member_df
        )

        return

    total = len(tasks)
    completed = 0

    lock = threading.Lock()

    started_at = time.time()

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    validate_cluster,
                    cluster_id,
                    members,
                ):
                cluster_id

                for (
                    cluster_id,
                    members,
                ) in tasks
            }

            for future in as_completed(
                future_map
            ):

                cluster_id = (
                    future_map[
                        future
                    ]
                )

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    print()
                    print(
                        f"[FAILED] "
                        f"{cluster_id}"
                    )

                    print(
                        type(exc).__name__,
                        exc,
                    )

                    continue

                with lock:

                    fout.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    fout.flush()

                    completed += 1

                elapsed = (
                    time.time()
                    - started_at
                )

                avg = (
                    elapsed
                    / completed
                )

                eta = (
                    avg
                    * (
                        total
                        - completed
                    )
                )

                if (
                    completed % 10 == 0
                    or completed == total
                ):

                    print(
                        f"[{completed}/"
                        f"{total}] "
                        f"{completed/total*100:.1f}% "
                        f"| ETA "
                        f"{eta/60:.1f} min"
                    )

    export_excel(
        member_df
    )

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        f"JSONL: "
        f"{OUTPUT_JSONL}"
    )

    print(
        f"Excel: "
        f"{OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()