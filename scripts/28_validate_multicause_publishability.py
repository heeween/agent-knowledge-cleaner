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

CLEAN_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause_structurally_clean.xlsx"
)

QUESTION_CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "multicause_publishability.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "multicause_publishability.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class PublishabilityResult(BaseModel):

    verdict: Literal[
        "publish",
        "manual_review",
        "reject",
    ]

    reason_category: Literal[
        "stable_reusable",
        "temporal_risk",
        "insufficient_evidence",
        "scope_mismatch",
        "mixed_intent",
        "case_only",
        "conflict_risk",
        "over_specific_action",
        "source_quality_risk",
        "other",
    ]

    confidence: float

    question_scope_valid: bool

    all_causes_answer_question: bool

    evidence_sufficient: bool

    temporally_safe: bool

    actions_safe: bool

    reusable: bool

    risky_cause_indexes: list[int]

    reason: str

    publish_notes: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在执行 CRM RAG 知识的最终 Publishability Gate。

输入是一条已经完成以下审核的 multiple-causes 知识：

1. 问题聚类审核
2. Cause source grounding
3. Cause 类型审核
4. 重复 Cause 合并
5. 错误合并拆分
6. Cause / Action 结构清洗

因此：

你绝对不能：

- 新增 Cause
- 删除 Cause
- 改写 Cause
- 改写 Action
- 补充产品规则

你只负责判断：

“这条知识整体是否适合作为正式 RAG 知识发布？”


============================================================
最终 verdict
============================================================

只有三个：

publish
manual_review
reject


============================================================
publish
============================================================

要求整体满足：

1. canonical_question 范围清晰
2. 所有 Cause 都真正回答这个问题
3. Cause 有足够 source 支持
4. 内容具备可复用性
5. 没有明显冲突
6. 没有把单次事故写成通用规则
7. 没有明显过时风险
8. Action 没有危险的无依据时间/菜单/流程
9. 不依赖某次临时技术修复才能成立


注意：

resolution = unresolved

不等于不能 publish。

例如：

“系统目前不支持某功能”

如果来源明确且属于稳定产品能力，
仍可能 publish。


============================================================
manual_review
============================================================

当知识可能有价值，
但存在必须人工确认的风险：

- 强时效性
- 运营商政策
- 产品版本变化
- 接口许可变化
- 明确存在临时规则
- 不确定是否仍适用
- 问题范围略宽
- 某个 Cause 可能只适用于特定场景
- Action 包含具体等待时间
- 缺少正式文档确认
- 不同 source 可能属于不同版本

则：

manual_review

不要为了提高发布率强行 publish。


============================================================
reject
============================================================

当整体不应该进入正式 KB：

- 问题与 Cause 明显不匹配
- 多个 Cause 实际回答不同问题
- 大部分只是单次事故
- 证据明显不足
- 内容严重依赖临时事件
- 无法形成可复用知识
- 核心结论存在冲突
- Question 范围明显大于 source 能支持的范围


============================================================
问题范围检查
============================================================

canonical_question 不能比 sources 支持的范围更宽。

例如：

Question：

“企业微信消息为什么发送失败？”

Cause source 实际只说明：

“删除好友后查询不到客户信息”

如果没有明确证明：

“查询不到客户 → 发送失败”

则该 Cause 可能属于：

scope mismatch。

如果只是个别 Cause 有问题：

manual_review

如果大量 Cause 都这样：

reject。


============================================================
Temporal 风险
============================================================

不要因为 source 中出现：

temporary

就自动 reject。

你需要区分：

A. 临时事件中确认了稳定机制

例如：

“没有切换耳机模式导致电话没声音”

即使该聊天是 temporary，
这个机制仍可复用。


B. 原因本身就是临时政策/故障

例如：

“今天运营商通道异常”

“当前正在升级”

“今年监管政策导致失败率高”

这种知识不能直接当长期稳定规则。

→ manual_review 或 reject。


============================================================
Action 安全性
============================================================

重点检查：

- 30分钟
- 10分钟
- 3个工作日
- 每几分钟
- 明天
- 次日
- 特定菜单路径
- “必须”
- “一定”

如果 source 只是当时案例，
这些具体时效可能不适合正式 KB。

通常：

actions_safe = false

并倾向：

manual_review。


============================================================
来源质量
============================================================

以下不会自动否定知识：

knowledge_value = medium
resolution = partial
temporary

但如果整条 KB 主要依赖：

low
temporary
unresolved
future_plan

且没有稳定来源支撑，

应 manual_review 或 reject。


============================================================
conflict
============================================================

如果不同 Cause 本身只是不同原因：

不是 conflict。

例如：

短信失败：

- 余额不足
- 敏感词
- 空号

这是正常 multiple_causes。


真正 conflict 是：

同一条件下：

A source 说“支持”
B source 说“不支持”

或：

A 说必须做
B 说不需要做。


============================================================
risky_cause_indexes
============================================================

列出存在以下风险的 Cause index：

- scope mismatch
- temporal
- weak evidence
- action over-specific
- case-only
- conflict

如果没有：

[]


============================================================
publish_notes
============================================================

只记录发布层风险。

例如：

[
  "运营商审核时效可能变化",
  "建议正式发布前由产品确认当前规则"
]

不要生成新的产品答案。


============================================================
输出
============================================================

只输出合法 JSON：

{
  "verdict": "publish | manual_review | reject",
  "reason_category": "stable_reusable | temporal_risk | insufficient_evidence | scope_mismatch | mixed_intent | case_only | conflict_risk | over_specific_action | source_quality_risk | other",
  "confidence": 0.0,
  "question_scope_valid": true,
  "all_causes_answer_question": true,
  "evidence_sufficient": true,
  "temporally_safe": true,
  "actions_safe": true,
  "reusable": true,
  "risky_cause_indexes": [],
  "reason": "简短说明",
  "publish_notes": []
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


def parse_keys(value):

    text = clean_text(value)

    if not text:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def load_processed():

    result = set()

    if not OUTPUT_JSONL.exists():
        return result

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
                    result.add(
                        cluster_id
                    )

            except Exception:
                pass

    return result


# ============================================================
# 构建 Prompt
# ============================================================

def build_prompt(
    cluster_id,
    canonical_question,
    cause_rows,
    source_rows,
):

    cause_blocks = []

    for _, row in (
        cause_rows.iterrows()
    ):

        cause_blocks.append(
            f"""
----------------------------------------
CAUSE {int(row["cause_index"])}

cause:
{clean_text(row["cause"])}

check_or_action:
{clean_text(row["check_or_action"]) or "null"}

source_issue_keys:
{clean_text(row["source_issue_keys"])}
""".strip()
        )

    source_blocks = []

    for _, row in (
        source_rows.iterrows()
    ):

        source_blocks.append(
            f"""
----------------------------------------
SOURCE ISSUE

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

confidence:
{clean_text(row.get("confidence"))}
""".strip()
        )

    return f"""
请审核以下 multiple-causes KB 是否适合正式发布。

============================================================
QUESTION
============================================================

cluster_id:
{cluster_id}

canonical_question:
{canonical_question}


============================================================
FINAL STRUCTURALLY CLEAN CAUSES
============================================================

{chr(10).join(cause_blocks)}


============================================================
SOURCE ISSUES
============================================================

{chr(10).join(source_blocks)}
"""


# ============================================================
# Audit
# ============================================================

def audit_cluster(
    cluster_id,
    canonical_question,
    cause_rows,
    source_rows,
):

    client = build_client()

    prompt = build_prompt(
        cluster_id,
        canonical_question,
        cause_rows,
        source_rows,
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
                                prompt,
                        },
                    ],
                    response_format={
                        "type":
                            "json_object"
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

            data = json.loads(
                response
                .choices[0]
                .message
                .content
            )

            result = (
                PublishabilityResult
                .model_validate(
                    data
                )
            )

            # --------------------------------------------
            # 轻量硬规则
            # --------------------------------------------

            if (
                not result.question_scope_valid
                or
                not result.all_causes_answer_question
            ):

                if (
                    result.verdict
                    == "publish"
                ):
                    result.verdict = (
                        "manual_review"
                    )

            if (
                not result.evidence_sufficient
                and
                result.verdict
                == "publish"
            ):

                result.verdict = (
                    "manual_review"
                )

            if (
                not result.temporally_safe
                and
                result.verdict
                == "publish"
            ):

                result.verdict = (
                    "manual_review"
                )

            if (
                not result.actions_safe
                and
                result.verdict
                == "publish"
            ):

                result.verdict = (
                    "manual_review"
                )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    canonical_question,

                "cause_count":
                    len(cause_rows),

                "verdict":
                    result.verdict,

                "reason_category":
                    result.reason_category,

                "audit_confidence":
                    result.confidence,

                "question_scope_valid":
                    result.question_scope_valid,

                "all_causes_answer_question":
                    result.all_causes_answer_question,

                "evidence_sufficient":
                    result.evidence_sufficient,

                "temporally_safe":
                    result.temporally_safe,

                "actions_safe":
                    result.actions_safe,

                "reusable":
                    result.reusable,

                "risky_cause_indexes":
                    result.risky_cause_indexes,

                "reason":
                    result.reason,

                "publish_notes":
                    result.publish_notes,

                "request_seconds":
                    elapsed,
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# Export
# ============================================================

def export_excel():

    rows = []

    if not OUTPUT_JSONL.exists():
        return

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
        "risky_cause_indexes"
    ] = df[
        "risky_cause_indexes"
    ].apply(
        lambda x:
            " | ".join(
                str(i)
                for i in x
            )
            if isinstance(x, list)
            else clean_text(x)
    )

    df[
        "publish_notes"
    ] = df[
        "publish_notes"
    ].apply(
        lambda x:
            " | ".join(x)
            if isinstance(x, list)
            else clean_text(x)
    )

    verdict_stats = (
        df["verdict"]
        .value_counts()
        .rename_axis(
            "verdict"
        )
        .reset_index(
            name="count"
        )
    )

    reason_stats = (
        df[
            "reason_category"
        ]
        .value_counts()
        .rename_axis(
            "reason_category"
        )
        .reset_index(
            name="count"
        )
    )

    publish_df = df[
        df["verdict"]
        == "publish"
    ].copy()

    review_df = df[
        df["verdict"]
        == "manual_review"
    ].copy()

    reject_df = df[
        df["verdict"]
        == "reject"
    ].copy()

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "audited_cluster_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "publish_count",
                "value":
                    len(
                        publish_df
                    ),
            },
            {
                "metric":
                    "manual_review_count",
                "value":
                    len(
                        review_df
                    ),
            },
            {
                "metric":
                    "reject_count",
                "value":
                    len(
                        reject_df
                    ),
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

        verdict_stats.to_excel(
            writer,
            sheet_name="verdict_stats",
            index=False,
        )

        reason_stats.to_excel(
            writer,
            sheet_name="reason_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        publish_df.to_excel(
            writer,
            sheet_name="publish",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="manual_review",
            index=False,
        )

        reject_df.to_excel(
            writer,
            sheet_name="reject",
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 10.6 - Multicause final publishability"
    )
    print("=" * 70)

    causes_df = pd.read_excel(
        CLEAN_FILE,
        sheet_name="causes",
    )

    members_df = pd.read_excel(
        QUESTION_CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    processed = load_processed()

    tasks = []

    for cluster_id, cause_rows in (
        causes_df.groupby(
            "cluster_id",
            sort=False,
        )
    ):

        cluster_id = str(
            cluster_id
        )

        if cluster_id in processed:
            continue

        canonical_question = (
            clean_text(
                cause_rows.iloc[0][
                    "canonical_question"
                ]
            )
        )

        source_keys = set()

        for value in (
            cause_rows[
                "source_issue_keys"
            ]
        ):

            source_keys.update(
                parse_keys(
                    value
                )
            )

        source_rows = members_df[
            (
                members_df[
                    "cluster_id"
                ].astype(str)
                == cluster_id
            )
            &
            (
                members_df[
                    "issue_key"
                ].astype(str)
                .isin(
                    source_keys
                )
            )
        ].copy()

        tasks.append(
            (
                cluster_id,
                canonical_question,
                cause_rows.copy(),
                source_rows,
            )
        )

    print(
        f"Cluster count: "
        f"{causes_df['cluster_id'].nunique()}"
    )

    print(
        f"本次待审核: "
        f"{len(tasks)}"
    )

    total = len(tasks)
    completed = 0

    started_at = time.time()
    lock = threading.Lock()

    if tasks:

        with OUTPUT_JSONL.open(
            "a",
            encoding="utf-8",
        ) as fout:

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_map = {

                    executor.submit(
                        audit_cluster,
                        cluster_id,
                        canonical_question,
                        cause_rows,
                        source_rows,
                    ):
                    cluster_id

                    for (
                        cluster_id,
                        canonical_question,
                        cause_rows,
                        source_rows,
                    )
                    in tasks
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

                        print(
                            f"[FAILED] "
                            f"{cluster_id}: "
                            f"{exc}"
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

                    print(
                        f"[{completed}/{total}] "
                        f"| ETA "
                        f"{eta/60:.1f} min"
                    )

    export_excel()

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        f"Excel: "
        f"{OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()