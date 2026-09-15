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

MULTICAUSE_FILE = (
    OUTPUT_DIR
    / "kb_entries_multicause.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "multicause_item_validations_v2.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "multicause_item_validations_v2.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class CauseValidation(BaseModel):

    verdict: Literal[
        "valid",
        "duplicate_cause",
        "not_a_cause",
        "wrong_intent",
        "unsupported",
        "action_unsupported",
        "needs_manual_review",
    ]

    confidence: float

    cause_supported: bool

    action_supported: bool

    answers_question: bool

    is_real_cause: bool

    reusable: bool

    supported_cause: str | None = None

    supported_action: str | None = None

    reason: str

    risk_flags: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审核 CRM RAG multiple_causes 知识中的单个“原因”。

本轮只审核：

1. 这是不是一个真正的原因
2. 是否回答 canonical_question
3. source 是否明确支持
4. check_or_action 是否明确有来源

本轮不要判断：

- 是否适合长期发布
- 是否已经过时
- 是否属于临时案例
- 是否应该进入最终 RAG

这些属于下一层 Publishability 审核。


============================================================
最重要规则：不要误用 temporal_status
============================================================

输入 source 中可能包含：

temporal_status = temporary
resolution = unresolved / partial
knowledge_value = low

这些字段：

只能作为风险提示。

绝对不能仅仅因为这些字段，
就把一个真实原因判成：

- not_a_cause
- unsupported
- wrong_intent

例如：

问题：
CRM拨打电话为什么没有声音？

source 明确说：

“没有切换到耳机模式，
点击右下角切换即可。”

即使：

temporal_status = temporary

该内容仍然是：

- 一个真实 cause
- source 支持
- 回答问题

因此结构审核应：

verdict = valid

至于是否适合长期发布，
留到下一步审核。


============================================================
verdict = valid
============================================================

满足：

- 原因确实解释 canonical_question 中的现象
- source 明确支持
- 是原因，不是处理动作
- 不是单纯诊断状态
- 如果有 action，action 有明确来源


============================================================
duplicate_cause
============================================================

当前原因与同一问题中的另一个原因
本质上是同一机制的不同表达。

例如：

“短信签名正在报备”
“短信签名尚未报备完成”

本质都可以归纳为：

“短信签名未完成报备”。

如果当前 source 的确支持该原因，
不要因为 duplicate 而说 unsupported。


============================================================
not_a_cause
============================================================

当前内容本身不是导致现象发生的原因，
而只是：

- 处理动作
- 排查动作
- 诊断方法
- 待办状态
- 处理结果
- “需要技术修复”
- “等待产品升级”

例如：

问题：
报表为什么不准确？

cause：
需要技术人员更新数据。

这是解决动作，不是原因。

→ not_a_cause


例如：

问题：
为什么电话没有声音？

cause：
无法判断是系统还是电脑的问题。

这是诊断状态。

→ not_a_cause


============================================================
wrong_intent
============================================================

source 讲的确实是一个问题，
但不是 canonical_question 所问的现象。

例如：

问题：
企微消息为什么发送失败？

source：
删除好友后无法查询客户资料。

如果 source 没有明确说明这会导致“消息发送失败”，
则不能自动关联。

→ wrong_intent


============================================================
unsupported
============================================================

生成出的原因无法从引用 source
明确找到支持。

禁止依靠常识、经验或推理补齐。


============================================================
action_unsupported
============================================================

cause 本身明确成立，
但 check_or_action 有额外无来源内容。

例如 source 只说：

“修改后重新同步”

生成：

“进入设置 > 数据管理，
等待30分钟后重新跑批”。

如果菜单和30分钟没有 source：

cause_supported = true
action_supported = false
verdict = action_unsupported


============================================================
cause_supported
============================================================

只判断 source 是否明确支持该原因。

temporary 不影响这个字段。


============================================================
answers_question
============================================================

只判断 cause 是否直接解释
canonical_question 中的问题现象。

temporary 不影响这个字段。


============================================================
is_real_cause
============================================================

非常重要。

只从语义判断它是不是“导致问题的原因”。

以下是真实 cause：

- 未切换耳机模式
- 账号名称不一致导致匹配失败
- 号码为空号
- 数据同步账号被禁用
- 机器人和人工同时登录同一账号
- 短信内容命中敏感词

即使 source 是 temporary，
这些仍然是 real cause。


以下不是 cause：

- 联系技术处理
- 等待修复
- 点击按钮测试
- 无法判断故障来源
- 已经补发
- 已经恢复


============================================================
reusable
============================================================

本轮仅作为辅助字段。

不要因为 reusable=false
修改 verdict。

可以根据 source 判断，
但最终发布与否由下一层决定。


============================================================
supported_cause
============================================================

如果原 cause 表达过宽，
但 source 支持更精确的原因，
输出更精确版本。

例如：

原 cause：
系统规则异常。

source：
同步账号被禁用。

supported_cause：
用于数据同步的账号被禁用。


============================================================
supported_action
============================================================

只保留 source 明确支持的检查或处理动作。

无可靠 action：

null


============================================================
risk_flags
============================================================

temporal / unresolved 等信息仍然要记录，
但只是风险标签。

可以使用：

temporary
unresolved
future_plan
single_case
question_scope_mismatch
cause_is_action
cause_is_diagnostic
action_contains_extra_steps
action_contains_unverified_time
action_contains_unverified_path
source_does_not_support_cause
other


注意：

出现：

temporary

绝不等于：

verdict != valid


============================================================
输出
============================================================

只输出合法 JSON object：

{
  "verdict": "valid | duplicate_cause | not_a_cause | wrong_intent | unsupported | action_unsupported | needs_manual_review",
  "confidence": 0.0,
  "cause_supported": true,
  "action_supported": true,
  "answers_question": true,
  "is_real_cause": true,
  "reusable": true,
  "supported_cause": "更准确的原因或 null",
  "supported_action": "明确支持的操作或 null",
  "reason": "简短原因",
  "risk_flags": []
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

                item_id = obj.get(
                    "item_id"
                )

                if item_id:
                    processed.add(
                        item_id
                    )

            except Exception:
                continue

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    cause_row,
    source_rows,
):

    source_blocks = []

    for idx, row in enumerate(
        source_rows,
        start=1,
    ):

        source_blocks.append(
            f"""
----------------------------------------
SOURCE ISSUE {idx}

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

description:
{clean_text(row.get("description"))}

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

crm_module:
{clean_text(row.get("crm_module"))}

crm_feature:
{clean_text(row.get("crm_feature"))}

problem_type:
{clean_text(row.get("problem_type"))}
""".strip()
        )

    return f"""
请审核以下 multiple_causes 知识中的单个原因。

============================================================
QUESTION
============================================================

cluster_id:
{clean_text(cause_row.get("cluster_id"))}

canonical_question:
{clean_text(cause_row.get("canonical_question"))}


============================================================
GENERATED CAUSE
============================================================

cause_index:
{clean_text(cause_row.get("cause_index"))}

cause:
{clean_text(cause_row.get("cause"))}

check_or_action:
{clean_text(cause_row.get("check_or_action"))}


============================================================
REFERENCED SOURCE ISSUES
============================================================

{chr(10).join(source_blocks)}

请严格判断：
该内容是否真的是 canonical_question 的一个有来源支持的原因。
"""


# ============================================================
# API
# ============================================================

def validate_item(
    cause_row,
    source_rows,
):

    client = build_client()

    cluster_id = clean_text(
        cause_row.get("cluster_id")
    )

    cause_index = int(
        cause_row.get(
            "cause_index"
        )
    )

    item_id = (
        f"{cluster_id}#CAUSE-{cause_index:02d}"
    )

    prompt = build_prompt(
        cause_row,
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
                CauseValidation
                .model_validate(
                    data
                )
            )

            # ------------------------------------------------
            # 硬规则
            # ------------------------------------------------

            if not result.cause_supported:

                result.verdict = (
                    "unsupported"
                )

            elif not result.answers_question:

                result.verdict = (
                    "wrong_intent"
                )

            elif not result.is_real_cause:

                result.verdict = (
                    "not_a_cause"
                )

            elif (
                not result.action_supported
                and clean_text(
                    cause_row.get(
                        "check_or_action"
                    )
                )
            ):

                result.verdict = (
                    "action_unsupported"
                )

            return {
                "item_id":
                    item_id,

                "cluster_id":
                    cluster_id,

                "canonical_question":
                    clean_text(
                        cause_row.get(
                            "canonical_question"
                        )
                    ),

                "cause_index":
                    cause_index,

                "original_cause":
                    clean_text(
                        cause_row.get(
                            "cause"
                        )
                    ),

                "original_action":
                    clean_text(
                        cause_row.get(
                            "check_or_action"
                        )
                    ),

                "source_issue_keys":
                    clean_text(
                        cause_row.get(
                            "source_issue_keys"
                        )
                    ),

                "verdict":
                    result.verdict,

                "audit_confidence":
                    result.confidence,

                "cause_supported":
                    result.cause_supported,

                "action_supported":
                    result.action_supported,

                "answers_question":
                    result.answers_question,

                "is_real_cause":
                    result.is_real_cause,

                "reusable":
                    result.reusable,

                "supported_cause":
                    result.supported_cause,

                "supported_action":
                    result.supported_action,

                "reason":
                    result.reason,

                "risk_flags":
                    result.risk_flags,

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
        "risk_flags"
    ] = df[
        "risk_flags"
    ].apply(
        lambda x:
            " | ".join(
                str(v)
                for v in x
            )
            if isinstance(
                x,
                list,
            )
            else clean_text(x)
    )

    # --------------------------------------------------------
    # 当前可保留
    # --------------------------------------------------------

    valid_df = df[
        (
            df[
                "verdict"
            ]
            == "valid"
        )
        &
        (
            df[
                "cause_supported"
            ]
            == True
        )
        &
        (
            df[
                "answers_question"
            ]
            == True
        )
        &
        (
            df[
                "is_real_cause"
            ]
            == True
        )
    ].copy()

    # cause 本身可留，
    # 只是 action 要删/改
    action_fix_df = df[
        (
            df[
                "verdict"
            ]
            == "action_unsupported"
        )
        &
        (
            df[
                "cause_supported"
            ]
            == True
        )
        &
        (
            df[
                "answers_question"
            ]
            == True
        )
        &
        (
            df[
                "is_real_cause"
            ]
            == True
        )
    ].copy()

    rejected_df = df[
        ~df[
            "item_id"
        ].isin(
            set(
                valid_df[
                    "item_id"
                ]
            )
            |
            set(
                action_fix_df[
                    "item_id"
                ]
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Cluster 级统计
    # --------------------------------------------------------

    cluster_stats = (
        df.groupby(
            [
                "cluster_id",
                "canonical_question",
            ],
            dropna=False,
        )
        .agg(
            total_causes=(
                "item_id",
                "count",
            ),
            valid_causes=(
                "verdict",
                lambda s:
                    int(
                        (
                            s == "valid"
                        ).sum()
                    ),
            ),
            action_fix_causes=(
                "verdict",
                lambda s:
                    int(
                        (
                            s
                            == "action_unsupported"
                        ).sum()
                    ),
            ),
            rejected_causes=(
            "verdict",
            lambda s:
                int(
                    (
                        s.isin(
                            [
                                "not_a_cause",
                                "wrong_intent",
                                "unsupported",
                                "needs_manual_review",
                            ]
                        )
                    ).sum()
                ),
            ),
            duplicate_causes=(
            "verdict",
            lambda s:
                int(
                    (
                        s == "duplicate_cause"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    cluster_stats[
        "usable_cause_count"
    ] = (
        cluster_stats[
            "valid_causes"
        ]
        +
        cluster_stats[
            "action_fix_causes"
        ]
        +
        cluster_stats[
            "duplicate_causes"
        ]
    )

    def cluster_status(row):

        usable = int(
            row[
                "usable_cause_count"
            ]
        )

        rejected = int(
            row[
                "rejected_causes"
            ]
        )

        if usable < 2:

            return "REJECT"

        if rejected > 0:

            return "REBUILD"

        if int(
            row[
                "action_fix_causes"
            ]
        ) > 0:

            return "REBUILD"

        return "CLEAN"

    cluster_stats[
        "cluster_status"
    ] = cluster_stats.apply(
        cluster_status,
        axis=1,
    )

    verdict_stats = (
        df[
            "verdict"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "verdict"
        )
        .reset_index(
            name="count"
        )
    )

    status_stats = (
        cluster_stats[
            "cluster_status"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "cluster_status"
        )
        .reset_index(
            name="count"
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "cause_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "valid_count",
                "value":
                    len(valid_df),
            },
            {
                "metric":
                    "action_fix_count",
                "value":
                    len(action_fix_df),
            },
            {
                "metric":
                    "rejected_count",
                "value":
                    len(rejected_df),
            },
            {
                "metric":
                    "cluster_count",
                "value":
                    len(cluster_stats),
            },
            {
                "metric":
                    "clean_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "cluster_status"
                            ]
                            == "CLEAN"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "rebuild_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "cluster_status"
                            ]
                            == "REBUILD"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "reject_cluster_count",
                "value":
                    int(
                        (
                            cluster_stats[
                                "cluster_status"
                            ]
                            == "REJECT"
                        ).sum()
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

        status_stats.to_excel(
            writer,
            sheet_name="cluster_status_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_causes",
            index=False,
        )

        valid_df.to_excel(
            writer,
            sheet_name="valid_causes",
            index=False,
        )

        action_fix_df.to_excel(
            writer,
            sheet_name="action_fix",
            index=False,
        )

        rejected_df.to_excel(
            writer,
            sheet_name="rejected_causes",
            index=False,
        )

        cluster_stats.to_excel(
            writer,
            sheet_name="cluster_stats",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 10.2 - Validate multicause items"
    )
    print("=" * 70)

    cause_df = pd.read_excel(
        MULTICAUSE_FILE,
        sheet_name="causes",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    print(
        f"Cause count: "
        f"{len(cause_df)}"
    )

    processed = load_processed()

    print(
        f"Already processed: "
        f"{len(processed)}"
    )

    tasks = []

    for _, cause_row in (
        cause_df.iterrows()
    ):

        cluster_id = clean_text(
            cause_row.get(
                "cluster_id"
            )
        )

        cause_index = int(
            cause_row.get(
                "cause_index"
            )
        )

        item_id = (
            f"{cluster_id}"
            f"#CAUSE-{cause_index:02d}"
        )

        if item_id in processed:
            continue

        source_keys = set(
            parse_keys(
                cause_row.get(
                    "source_issue_keys"
                )
            )
        )

        source_rows = (
            member_df[
                (
                    member_df[
                        "cluster_id"
                    ]
                    .astype(str)
                    == cluster_id
                )
                &
                (
                    member_df[
                        "issue_key"
                    ]
                    .astype(str)
                    .isin(
                        source_keys
                    )
                )
            ]
            .to_dict(
                orient="records"
            )
        )

        tasks.append(
            (
                cause_row,
                source_rows,
            )
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
                        validate_item,
                        cause_row,
                        source_rows,
                    ):
                    (
                        clean_text(
                            cause_row.get(
                                "cluster_id"
                            )
                        ),
                        int(
                            cause_row.get(
                                "cause_index"
                            )
                        ),
                    )

                    for (
                        cause_row,
                        source_rows,
                    )
                    in tasks
                }

                for future in as_completed(
                    future_map
                ):

                    (
                        cluster_id,
                        cause_index,
                    ) = future_map[
                        future
                    ]

                    try:

                        result = (
                            future.result()
                        )

                    except Exception as exc:

                        print()
                        print(
                            f"[FAILED] "
                            f"{cluster_id}"
                            f"#CAUSE-{cause_index:02d}"
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
                            f"| ETA "
                            f"{eta/60:.1f} min"
                        )

    export_excel()

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