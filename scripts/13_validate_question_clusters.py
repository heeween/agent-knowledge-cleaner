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
    / "cluster_validations.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "cluster_validations.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class ClusterValidation(BaseModel):

    cluster_validity: Literal[
        "valid",
        "needs_split",
        "uncertain",
    ]

    confidence: float

    canonical_question: str | None = None

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

    reason: str

    suspicious_issue_keys: list[str]

    recommended_action: Literal[
        "merge",
        "merge_with_care",
        "split",
        "manual_review",
    ]


# ============================================================
# Prompt
# ============================================================
SYSTEM_PROMPT = """
你正在审核 CRM RAG 知识库中的候选问题 Cluster。

每个 Cluster 包含多个从真实 CRM 客服聊天中抽取出的 issue。

之前已经通过：
- question_normalized embedding
- pair-level same_intent 判断
- 保守 Clique 聚类

将这些 issue 放入同一个候选 Cluster。

你的任务有两个彼此独立的部分：

A. 判断这些“问题”是否属于同一个知识意图。
B. 在问题属于同一知识意图的前提下，判断“答案”之间是什么关系。

非常重要：

============================================================
最高优先级规则
============================================================

cluster_validity 必须主要根据 question_normalized 判断。

不要因为：

- 答案不同
- 原因不同
- 解决方案不同
- 不同客服给出的处理方式不同
- resolution 不同
- temporal_status 不同

就自动把 Cluster 判成 needs_split。

这些差异应该由 answer_relation 描述，
而不是用 cluster_validity 拆问题。


例如：

问题都是：

“短信发送失败的原因是什么？”

不同答案分别是：

- 余额不足
- 短信签名未报备
- 内容有敏感词
- 运营商限制

这是：

cluster_validity = valid
answer_relation = multiple_causes

绝对不能因为“原因不同”判 needs_split。


再例如：

问题都是：

“车辆保养后为什么仍然触发保养提醒？”

不同来源给出的原因包括：

- 里程录入错误
- 工单录入错误
- 数据同步异常
- 系统判断规则

如果 question_normalized 都确实在问
“保养后为什么仍触发提醒”，

则应该：

cluster_validity = valid
answer_relation = multiple_causes

而不是 needs_split。


============================================================
一、cluster_validity
============================================================

cluster_validity 只能是：

1. valid

所有 question_normalized 本质上都在询问
同一个可复用 CRM 知识问题。

判断核心：

“如果不看现有 answer，
只看用户提出的问题，
这些问题是否在寻找同一类知识答案？”

允许：

- 不同措辞
- 同义表达
- 一个问题更完整
- 不同来源使用不同业务描述
- 同一个问题存在不同原因
- 同一个问题存在不同解决办法
- 同一个问题存在旧答案和新答案
- 同一个问题有的来源未回答、有的来源回答完整


2. needs_split

只有当 question_normalized 本身询问的内容不同，
才应该使用 needs_split。

例如：

A：CRM 培训需要多长时间？
B：CRM 培训时间如何安排？

一个问“培训时长”，
一个问“时间安排”。

即使都属于培训模块，
也不是同一个知识问题。

应该 needs_split。


例如：

A：预计保养里程是什么意思？
B：预计保养里程为什么出现负值？

一个问字段定义，
一个问异常原因。

应该 needs_split。


例如：

A：短信发送失败的原因是什么？
B：短信发送失败是否会扣费？

一个问失败原因，
一个问计费规则。

应该 needs_split。


例如：

A：企业微信功能收费标准是多少？
B：企业微信拉群功能收费标准是多少？

如果一个问整个企业微信功能收费，
另一个只问某具体子功能收费，
不能仅因为关键词相似就认定 valid。

应根据问题范围判断 needs_split 或 uncertain。


3. uncertain

仅凭现有 question_normalized，
无法可靠判断它们是不是同一个知识意图。


============================================================
二、answer_relation
============================================================

answer_relation 与 cluster_validity 分开判断。

即使 answer_relation 是 conflict，
cluster_validity 仍然完全可能是 valid。

即使 answer_relation 是 temporal_versions，
cluster_validity 仍然完全可能是 valid。


可选类型：


duplicate

不同来源答案核心事实基本相同，
只是措辞、表达顺序不同。


complementary

答案描述的是同一个知识规则的不同部分，
可以组合成更完整答案。

例如：

一个回答入口在哪里，
另一个补充操作步骤和注意事项。


multiple_causes

同一个“为什么 / 为什么失败 / 为什么异常”
问题存在多个有效原因。

例如：

“短信发送失败的原因是什么？”

答案分别是：

- 余额不足
- 敏感词
- 签名问题
- 运营商限制

必须判断为：

cluster_validity = valid
answer_relation = multiple_causes


incomplete_vs_complete

有的记录：

- 无明确答案
- 正在排查
- 只回答一部分

其他记录有更完整、明确的答案。

这些仍然可以属于同一个问题 Cluster。


temporal_versions

问题是同一个，
但不同时间产品规则、支持状态、操作路径发生变化。

例如：

旧版本：
“不支持导出。”

新版本：
“现在已经支持导出。”

这通常应该：

cluster_validity = valid
answer_relation = temporal_versions

不要因为新旧规则不同直接 needs_split。


conflict

问题明确是同一个，
但答案出现不能同时成立的结论，
并且现有证据无法确认是版本变化造成的。

例如：

问题都是：
“客户数据是否支持导出？”

答案 A：
支持。

答案 B：
不支持。

如果没有时间信息解释差异：

cluster_validity = valid
answer_relation = conflict

注意：
答案冲突不等于问题应该拆分。


mixed

在一个有效 Cluster 中，
答案同时存在多种关系。

例如：

- 部分答案重复
- 部分补充
- 同时存在一个疑似旧版本答案

则可以使用 mixed。


unknown

问题看起来属于同一意图，
但答案信息不足，
无法判断答案之间的关系。


============================================================
三、cluster_validity 与 answer_relation 的关系
============================================================

请严格遵守：

不同答案 ≠ 不同问题

不同原因 ≠ 不同问题

不同解决方案 ≠ 不同问题

答案冲突 ≠ 必须拆问题

时间版本变化 ≠ 必须拆问题


只有 question_normalized 的用户意图不同，
才能作为 needs_split 的主要理由。


============================================================
四、suspicious_issue_keys
============================================================

只有当：

cluster_validity = needs_split

时，才填写真正“问题意图不同”的 issue_key。

不要因为：

- 这个 issue 的答案不同
- 这个 issue 是 unresolved
- 这个 issue 没有答案
- 这个 issue 的答案疑似旧版本

就把它列为 suspicious_issue_keys。

如果 cluster_validity = valid：

suspicious_issue_keys 通常必须为 []。


============================================================
五、canonical_question
============================================================

只有 cluster_validity = valid 时填写。

canonical_question 必须：

- 独立成句
- 保持问句
- 简洁
- 可以直接用于 embedding / RAG 检索
- 不包含具体客户
- 不包含门店名称
- 不包含人员姓名
- 不包含来源聊天中的偶然信息
- 能覆盖整个 Cluster 的共同问题意图


============================================================
六、recommended_action
============================================================

根据以下规则判断：


如果：

cluster_validity = needs_split

则：

recommended_action = split


如果：

cluster_validity = uncertain

则：

recommended_action = manual_review


如果：

cluster_validity = valid

且 answer_relation 属于：

- duplicate
- complementary
- incomplete_vs_complete

通常：

recommended_action = merge


如果：

cluster_validity = valid

且 answer_relation = multiple_causes

则：

recommended_action = merge_with_care


如果：

cluster_validity = valid

且 answer_relation 属于：

- conflict
- temporal_versions

则：

recommended_action = manual_review


如果：

cluster_validity = valid
且 answer_relation = mixed

根据风险判断：

merge_with_care
或
manual_review


============================================================
七、特别容易犯错的例子
============================================================

例 1：

Q1：
短信发送失败的原因是什么？

Q2：
为什么短信发送不出去？

答案 A：
短信余额不足。

答案 B：
运营商要求完成签名报备。

正确：

cluster_validity = valid
answer_relation = multiple_causes


例 2：

Q1：
保养后为什么还会收到保养提醒？

Q2：
车辆刚保养完为什么还在保养邀约列表？

答案 A：
本次工单里程录错。

答案 B：
维修项目没有被识别为保养。

正确：

cluster_validity = valid
answer_relation = multiple_causes


例 3：

Q1：
客户数据能否导出？

Q2：
CRM是否支持导出客户资料？

答案 A：
不支持。

答案 B：
支持导出部分字段。

正确：

如果两个问题问的是同一个导出能力：

cluster_validity = valid

answer_relation 根据证据判断：

conflict
或 temporal_versions

不要仅因为答案不同拆 Cluster。


例 4：

Q1：
CRM账号密码忘记后如何重置？

Q2：
为什么重置密码后仍提示密码错误？

正确：

cluster_validity = needs_split

因为：

一个询问标准密码重置流程，
另一个询问重置后的登录异常。


例 5：

Q1：
CRM培训预计需要多长时间？

Q2：
CRM培训安排在哪个时间段？

正确：

cluster_validity = needs_split

因为：

一个问培训时长，
一个问培训安排。


============================================================
输出要求
============================================================

你必须只输出一个合法 JSON object。

不要输出 Markdown。
不要输出 ```json 代码块。
不要输出 JSON 之外的任何文字。

JSON 格式：

{
  "cluster_validity": "valid | needs_split | uncertain",
  "confidence": 0.0,
  "canonical_question": "标准问题或 null",
  "answer_relation": "duplicate | complementary | multiple_causes | incomplete_vs_complete | temporal_versions | conflict | mixed | unknown",
  "reason": "简短说明判断依据",
  "suspicious_issue_keys": [],
  "recommended_action": "merge | merge_with_care | split | manual_review"
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
# 构造 Cluster Prompt
# ============================================================

def build_cluster_text(
    cluster_id,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        block = f"""
------------------------------
Issue {idx}

issue_key:
{clean_text(row.get("issue_key"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

answer:
{clean_text(row.get("answer"))}

solution:
{clean_text(row.get("solution"))}

crm_module:
{clean_text(row.get("crm_module"))}

crm_feature:
{clean_text(row.get("crm_feature"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}
"""

        blocks.append(
            block.strip()
        )

    return f"""
请审核下面这个 CRM 问题 Cluster。

cluster_id:
{cluster_id}

member_count:
{len(members)}

{chr(10).join(blocks)}

请重点判断：

1. 所有 question_normalized 是否真的是同一个知识意图。
2. 不同 answer / solution 是重复、互补、多原因、
   完整度差异、时间版本还是冲突。
3. 如果 Cluster 中混入了不应该合并的问题，
   必须指出 suspicious_issue_keys。
"""


# ============================================================
# 请求
# ============================================================

def validate_cluster(
    cluster_id,
    members,
):

    client = build_client()

    user_prompt = build_cluster_text(
        cluster_id,
        members,
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            started = time.time()

            response = (
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                SYSTEM_PROMPT,
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

            elapsed = (
                time.time()
                - started
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

            result = (
                ClusterValidation
                .model_validate(
                    data
                )
            )

# --------------------------------------------------------
# 强制修正字段之间的逻辑一致性
# --------------------------------------------------------

            if result.cluster_validity != "valid":
                result.canonical_question = None

            if result.cluster_validity == "needs_split":
                result.recommended_action = "split"

            elif result.cluster_validity == "uncertain":
                result.recommended_action = "manual_review"

            elif result.cluster_validity == "valid":

                # valid cluster 不允许因为答案差异被标记 split
                if result.answer_relation in {
                    "duplicate",
                    "complementary",
                    "incomplete_vs_complete",
                }:
                    result.recommended_action = "merge"

                elif result.answer_relation == "multiple_causes":
                    result.recommended_action = "merge_with_care"

                elif result.answer_relation in {
                    "conflict",
                    "temporal_versions",
                }:
                    result.recommended_action = "manual_review"

                elif result.answer_relation == "mixed":

                    if result.recommended_action == "split":
                        result.recommended_action = "merge_with_care"

                # valid cluster 不应该有“疑似错误成员”
                result.suspicious_issue_keys = []

            return {
                "cluster_id":
                    cluster_id,

                "member_count":
                    len(members),

                "cluster_validity":
                    result.cluster_validity,

                "confidence":
                    result.confidence,

                "canonical_question":
                    result.canonical_question,

                "answer_relation":
                    result.answer_relation,

                "reason":
                    result.reason,

                "suspicious_issue_keys":
                    result.suspicious_issue_keys,

                "recommended_action":
                    result.recommended_action,

                "request_seconds":
                    elapsed,
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait_seconds = (
                    2 ** attempt
                )

                print(
                    f"[retry {attempt}] "
                    f"{cluster_id}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                time.sleep(
                    wait_seconds
                )

    raise last_error


# ============================================================
# 导出
# ============================================================

def export_excel(
    member_df,
):

    rows = []

    if OUTPUT_JSONL.exists():

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

    result_df = pd.DataFrame(
        rows
    )

    if result_df.empty:
        return

    # list 转成 Excel 可读文本
    result_df[
        "suspicious_issue_keys"
    ] = result_df[
        "suspicious_issue_keys"
    ].apply(
        lambda x:
            " | ".join(x)
            if isinstance(x, list)
            else clean_text(x)
    )

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    validity_stats = (
        result_df[
            "cluster_validity"
        ]
        .value_counts()
        .rename_axis(
            "cluster_validity"
        )
        .reset_index(
            name="count"
        )
    )

    answer_stats = (
        result_df[
            "answer_relation"
        ]
        .value_counts()
        .rename_axis(
            "answer_relation"
        )
        .reset_index(
            name="count"
        )
    )

    action_stats = (
        result_df[
            "recommended_action"
        ]
        .value_counts()
        .rename_axis(
            "recommended_action"
        )
        .reset_index(
            name="count"
        )
    )

    # --------------------------------------------------------
    # review
    # --------------------------------------------------------

    review_df = result_df[
        (
            result_df[
                "cluster_validity"
            ]
            != "valid"
        )
        |
        (
            result_df[
                "confidence"
            ]
            < 0.90
        )
        |
        (
            result_df[
                "answer_relation"
            ].isin(
                [
                    "temporal_versions",
                    "conflict",
                    "mixed",
                    "unknown",
                ]
            )
        )
        |
        (
            result_df[
                "recommended_action"
            ].isin(
                [
                    "split",
                    "manual_review",
                ]
            )
        )
    ].copy()

    # --------------------------------------------------------
    # 把 review cluster 成员带出来
    # --------------------------------------------------------

    review_cluster_ids = set(
        review_df[
            "cluster_id"
        ].tolist()
    )

    review_members_df = member_df[
        member_df[
            "cluster_id"
        ].isin(
            review_cluster_ids
        )
    ].copy()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "validated_cluster_count",
                "value":
                    len(result_df),
            },
            {
                "metric":
                    "valid_cluster_count",
                "value":
                    int(
                        (
                            result_df[
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
                            result_df[
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
                            result_df[
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

    # --------------------------------------------------------
    # export
    # --------------------------------------------------------

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

        result_df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review_clusters",
            index=False,
        )

        review_members_df.to_excel(
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
        "Step 8.6 - Validate question clusters"
    )
    print("=" * 70)

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    cluster_ids = (
        member_df["cluster_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Cluster count: "
        f"{len(cluster_ids)}"
    )

    processed = load_processed()

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

        print(
            "没有待处理 Cluster。"
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