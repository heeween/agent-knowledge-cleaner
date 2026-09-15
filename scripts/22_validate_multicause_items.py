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
    / "multicause_item_validations.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "multicause_item_validations.xlsx"
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
        "temporary_case",
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
你正在审核 CRM RAG 中的一条“故障原因”。

输入包括：

1. canonical_question
2. 当前生成的 cause
3. 当前生成的 check_or_action
4. 该 cause 引用的原始 source issue

你的任务不是重新回答整个问题。

只判断这一条 cause 是否应该保留。


============================================================
核心原则
============================================================

一个合格的 cause 必须同时满足：

1. 真正在解释 canonical_question 中的问题现象
2. 原始 source 明确支持
3. 不是处理动作
4. 不是排查状态
5. 不是单纯“需要技术处理”
6. 不是回答另一个问题
7. 如果有 check_or_action，操作也必须有来源


============================================================
verdict
============================================================

valid

该内容确实是问题的一个独立原因，
并且 source 明确支持。

如果有 check_or_action，
操作也有明确来源。


------------------------------------------------------------

duplicate_cause

该内容本身可能有依据，
但只是同一原因的另一种说法或状态描述。

例如：

原因 A：
短信签名正在报备中。

原因 B：
短信签名尚未报备成功。

如果二者本质都是：

“短信签名未完成报备”

则应视为可合并的重复原因。


本次单条审核无法看到其他 cause 时，
只有当当前内容自身明显只是一个状态描述，
而 supported_cause 应归纳成更稳定的同一原因时，
才使用 duplicate_cause。


------------------------------------------------------------

not_a_cause

内容不是原因，而是：

- 处理动作
- 排查步骤
- 诊断状态
- “需要技术处理”
- “等待系统同步”
- “暂时无法判断”
- 结果描述

例如：

“需要技术团队修复数据”

不是“报表不准确”的原因。

→ not_a_cause


例如：

“无法判断是系统问题还是电脑问题”

不是故障原因。

→ not_a_cause


------------------------------------------------------------

wrong_intent

source 实际回答的是另一个问题。

例如：

canonical_question：
为什么企业微信消息发送失败？

source：
客户删除好友后，在企微里查不到客户资料。

如果来源没有证明这会导致消息发送失败：

→ wrong_intent


------------------------------------------------------------

unsupported

当前 cause 中的核心原因，
无法从所引用的 source 找到明确支持。

禁止使用常识补全。


------------------------------------------------------------

action_unsupported

cause 本身有依据，
但 check_or_action 中包含 source 没有明确支持的步骤、时间、路径、规则。

例如：

source 只说：
“修改后重新同步。”

生成操作：
“进入设置 > 数据管理，等待30分钟重新跑批。”

如果菜单和30分钟没有来源：

→ action_unsupported


此时：

cause_supported = true
action_supported = false


------------------------------------------------------------

temporary_case

该原因只来自一次：

- 临时故障
- temporary
- 单次事故
- 一次性技术修复
- future_plan
- 未确认异常

并没有证据表明它是可长期复用的故障原因。


例如：

“某次系统升级后出现同步 bug，
随后已经修复。”

如果只是该次事故：

→ temporary_case


------------------------------------------------------------

needs_manual_review

来源复杂或语义不足，
无法安全判断。


============================================================
cause_supported
============================================================

source 是否明确支持 cause。


============================================================
action_supported
============================================================

如果 check_or_action 为空：

true

如果不为空：

只有所有实质性操作内容均有来源，
才为 true。


============================================================
answers_question
============================================================

当前 cause 是否真的与 canonical_question
的问题现象直接相关。


============================================================
is_real_cause
============================================================

是真正的原因：

true

只是：

- 操作
- 状态
- 检查方法
- 处理结果
- 技术排查

则 false。


============================================================
reusable
============================================================

该原因是否适合未来相似 CRM 问题复用。

temporary 单次事件通常 false。


============================================================
supported_cause
============================================================

如果当前 cause 表达过度、混乱，
但 source 能支持一个更准确、更窄的原因，
填写该准确版本。

例如：

当前：
“系统配置逻辑异常导致所有车辆计算错误”

source 只支持：
“该车辆保养开始里程填写为0”

则：

supported_cause：
“车辆保养开始里程填写为0或异常值”


如果无需修正，可填写原 cause。


============================================================
supported_action
============================================================

只填写 source 明确支持的操作。

如果没有可靠操作：

null


============================================================
risk_flags
============================================================

可使用：

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

没有则 []。


============================================================
特别注意
============================================================

1. “联系技术处理”
不是原因。

2. “等待升级”
不是原因。

3. “测试自己的手机号判断故障位置”
是检查方法，不是原因。

4. “系统同步延迟”
只有来源明确确认延迟是原因时才能保留。

5. 不要因为 source_issue_key 存在，
就自动认为 source 支持生成内容。

必须核对 source 的 answer / solution。

6. 不要使用外部 CRM 知识。


============================================================
输出
============================================================

只输出合法 JSON object。

{
  "verdict": "valid | duplicate_cause | not_a_cause | wrong_intent | unsupported | action_unsupported | temporary_case | needs_manual_review",
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
        &
        (
            df[
                "reusable"
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
                            ~s.isin(
                                [
                                    "valid",
                                    "action_unsupported",
                                ]
                            )
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