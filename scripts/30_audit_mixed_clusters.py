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

VALIDATION_FILE = (
    OUTPUT_DIR
    / "cluster_validations_v3.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "mixed_cluster_audits.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_cluster_audits.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class MemberDecision(BaseModel):

    issue_key: str

    disposition: Literal[
        "keep",
        "noise",
        "separate_sub_intent",
        "uncertain",
    ]

    sub_intent_id: str | None = None

    reason: str


class SubIntent(BaseModel):

    sub_intent_id: str

    canonical_question: str

    issue_keys: list[str]

    description: str


class MixedClusterAudit(BaseModel):

    classification: Literal[
        "single_intent_structurable",
        "sub_intents",
        "mixed_with_noise",
        "unsafe_mixed",
    ]

    confidence: float

    core_question: str

    members: list[MemberDecision]

    sub_intents: list[SubIntent]

    answer_structure_types: list[
        Literal[
            "direct_answer",
            "cause",
            "condition",
            "procedure",
            "limitation",
            "diagnostic",
            "status",
            "temporary_workaround",
            "other",
        ]
    ]

    has_conflict: bool

    has_temporal_risk: bool

    safe_for_next_stage: bool

    reason: str

    risk_flags: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审核 CRM RAG 知识中的 mixed question cluster。

这些 Cluster 已经经过 question-only 聚类审核，
并被判断为问题意图基本相关，
但它们的答案关系属于 mixed。

本轮绝对不要生成 KB。

你的任务只是判断：

1. 这些成员是否仍然属于同一个核心问题
2. 是否存在真正不同的子意图
3. 是否有少量 noise / 错位成员
4. mixed 是否只是因为答案结构不同
5. 这个 Cluster 是否安全进入下一阶段


============================================================
最重要原则
============================================================

“答案不同”不等于“问题意图不同”。

例如同一个问题：

“为什么短信发送失败？”

成员 A：
因为余额不足。

成员 B：
因为包含敏感词。

成员 C：
检查运营商报备状态。

答案内容不同，
但问题意图仍然可能完全相同。

不能仅因为答案不同，
拆成多个 sub_intent。


============================================================
single_intent_structurable
============================================================

所有或几乎所有成员：

都在回答同一个核心问题。

mixed 只是因为答案分别包含：

- 原因
- 操作步骤
- 条件
- 限制
- 状态说明
- 诊断方法
- 补充说明

例如：

问题都在问：

“如何修改客户手机号？”

不同答案分别说明：

- 修改入口
- 修改条件
- 手机号已存在时的限制

这仍然可能是一条可结构化知识。

classification =
single_intent_structurable


============================================================
sub_intents
============================================================

只有当成员的问题本身存在
两个或以上可以独立检索、独立回答的用户意图时，
才使用。

例如：

A：
如何修改登录密码？

B：
忘记密码怎么办？

C：
默认密码是什么？

虽然都属于“密码”，
但这是三个独立用户意图。

→ sub_intents


每个 sub_intent：

必须提供：

sub_intent_id
canonical_question
issue_keys
description


============================================================
mixed_with_noise
============================================================

Cluster 主体确实属于一个核心问题，
但有少量成员：

- Q/A 错位
- 实际回答别的问题
- 内容无法支撑该问题
- 抽取边界污染

这些成员应该 disposition=noise。

例如：

5 个成员都回答“如何导出客户”，
其中 1 个 answer 实际在讲“修改密码”。

→ mixed_with_noise

不要因为一个 noise
把整个 Cluster 判成 sub_intents。


============================================================
unsafe_mixed
============================================================

使用场景：

- 意图严重混杂，无法可靠拆分
- 多个成员信息不足
- Q/A 错位严重
- 明显存在冲突但无法判断
- 无法形成可靠的下一阶段输入

→ unsafe_mixed

safe_for_next_stage = false


============================================================
Member disposition
============================================================

keep

该成员属于 Cluster 的核心问题。


noise

问题看似相关，
但 answer / solution 实际不回答核心问题，
或明显存在抽取污染。


separate_sub_intent

该成员属于一个真正独立的子意图。

必须填写：

sub_intent_id


uncertain

无法可靠判断。


============================================================
Sub-intent 判断规则
============================================================

必须从“用户要解决什么问题”判断。

不要从：

- answer 长短
- answer 使用不同方案
- answer 来源不同
- resolution 不同
- temporal_status 不同

来拆意图。


例如：

“为什么登录失败？”

A：
账号停用。

B：
密码错误。

这是不同原因，
不是不同 sub-intent。


但是：

A：
登录失败怎么办？

B：
如何重置密码？

虽然有关联，
但用户请求目标不同。

可以拆 sub-intent。


============================================================
answer_structure_types
============================================================

只描述这个 Cluster 的答案形态。

允许：

direct_answer
cause
condition
procedure
limitation
diagnostic
status
temporary_workaround
other

可以同时有多个。


============================================================
Conflict
============================================================

不同原因、不同操作路径，
不自动算 conflict。

只有针对相同条件/相同问题：

A 说支持
B 说不支持

或：

A 说必须做 X
B 说明确不能做 X

才是 conflict。


============================================================
Temporal risk
============================================================

temporary / future_plan
只作为风险。

不要仅因为 temporary
把成员判 noise。

如果知识本体依赖：

- 当前版本
- 当时故障
- 临时政策
- 临时 workaround

则：

has_temporal_risk = true


============================================================
safe_for_next_stage
============================================================

true：

结构已经足够清楚，
下一阶段可以：

- 对 single intent 做结构化
- 对明确 sub-intents 分别处理
- 对 mixed_with_noise 去除 noise 后处理


false：

unsafe_mixed
或仍有关键 uncertain。


============================================================
issue_key
============================================================

输出中的 issue_key
必须严格来自输入。

禁止生成不存在的 issue_key。


============================================================
输出
============================================================

只输出合法 JSON：

{
  "classification": "single_intent_structurable | sub_intents | mixed_with_noise | unsafe_mixed",
  "confidence": 0.0,
  "core_question": "核心问题",
  "members": [
    {
      "issue_key": "原 issue_key",
      "disposition": "keep | noise | separate_sub_intent | uncertain",
      "sub_intent_id": null,
      "reason": "简短原因"
    }
  ],
  "sub_intents": [],
  "answer_structure_types": [],
  "has_conflict": false,
  "has_temporal_risk": false,
  "safe_for_next_stage": true,
  "reason": "整体判断",
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
                pass

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    cluster_id,
    canonical_question,
    rows,
):

    blocks = []

    for idx, row in enumerate(
        rows,
        start=1,
    ):

        blocks.append(
            f"""
============================================================
MEMBER {idx}
============================================================

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

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
请审核以下 mixed Cluster。

cluster_id:
{cluster_id}

question-only validation canonical_question:
{canonical_question}

member_count:
{len(rows)}


{chr(10).join(blocks)}
"""


# ============================================================
# Audit
# ============================================================

def audit_cluster(
    cluster_id,
    canonical_question,
    rows,
):

    client = build_client()

    prompt = build_prompt(
        cluster_id,
        canonical_question,
        rows,
    )

    allowed_keys = {
        clean_text(
            row.get("issue_key")
        )
        for row in rows
    }

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
                MixedClusterAudit
                .model_validate(
                    data
                )
            )

            # ------------------------------------------------
            # issue_key 安全检查
            # ------------------------------------------------

            output_keys = {
                item.issue_key
                for item in result.members
            }

            if not output_keys.issubset(
                allowed_keys
            ):
                raise RuntimeError(
                    "模型生成了不存在的 issue_key"
                )

            # 必须覆盖所有成员
            if output_keys != allowed_keys:

                missing = (
                    allowed_keys
                    - output_keys
                )

                raise RuntimeError(
                    "模型未覆盖全部 issue_key: "
                    + str(
                        sorted(missing)
                    )
                )

            # ------------------------------------------------
            # sub_intent key 安全检查
            # ------------------------------------------------

            for sub in result.sub_intents:

                if not set(
                    sub.issue_keys
                ).issubset(
                    allowed_keys
                ):
                    raise RuntimeError(
                        "sub_intent 出现非法 issue_key"
                    )

            # ------------------------------------------------
            # 硬规则
            # ------------------------------------------------

            if any(
                item.disposition
                == "uncertain"
                for item in result.members
            ):

                result.safe_for_next_stage = (
                    False
                )

            if (
                result.classification
                == "unsafe_mixed"
            ):

                result.safe_for_next_stage = (
                    False
                )

            if (
                result.classification
                == "sub_intents"
                and
                len(
                    result.sub_intents
                ) < 2
            ):

                result.safe_for_next_stage = (
                    False
                )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    canonical_question,

                "member_count":
                    len(rows),

                "classification":
                    result.classification,

                "audit_confidence":
                    result.confidence,

                "core_question":
                    result.core_question,

                "members":
                    [
                        item.model_dump()
                        for item
                        in result.members
                    ],

                "sub_intents":
                    [
                        item.model_dump()
                        for item
                        in result.sub_intents
                    ],

                "answer_structure_types":
                    result.answer_structure_types,

                "has_conflict":
                    result.has_conflict,

                "has_temporal_risk":
                    result.has_temporal_risk,

                "safe_for_next_stage":
                    result.safe_for_next_stage,

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
# Excel
# ============================================================

def export_excel():

    objects = []

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
                objects.append(
                    json.loads(line)
                )
            except Exception:
                pass

    cluster_rows = []
    member_rows = []
    sub_rows = []

    for obj in objects:

        cluster_rows.append(
            {
                "cluster_id":
                    obj["cluster_id"],

                "canonical_question":
                    obj[
                        "canonical_question"
                    ],

                "member_count":
                    obj[
                        "member_count"
                    ],

                "classification":
                    obj[
                        "classification"
                    ],

                "audit_confidence":
                    obj[
                        "audit_confidence"
                    ],

                "core_question":
                    obj[
                        "core_question"
                    ],

                "answer_structure_types":
                    " | ".join(
                        obj.get(
                            "answer_structure_types",
                            []
                        )
                    ),

                "has_conflict":
                    obj[
                        "has_conflict"
                    ],

                "has_temporal_risk":
                    obj[
                        "has_temporal_risk"
                    ],

                "safe_for_next_stage":
                    obj[
                        "safe_for_next_stage"
                    ],

                "reason":
                    obj[
                        "reason"
                    ],

                "risk_flags":
                    " | ".join(
                        obj.get(
                            "risk_flags",
                            []
                        )
                    ),
            }
        )

        for item in obj.get(
            "members",
            []
        ):

            member_rows.append(
                {
                    "cluster_id":
                        obj["cluster_id"],

                    "core_question":
                        obj[
                            "core_question"
                        ],

                    "issue_key":
                        item[
                            "issue_key"
                        ],

                    "disposition":
                        item[
                            "disposition"
                        ],

                    "sub_intent_id":
                        item.get(
                            "sub_intent_id"
                        ),

                    "reason":
                        item[
                            "reason"
                        ],
                }
            )

        for sub in obj.get(
            "sub_intents",
            []
        ):

            sub_rows.append(
                {
                    "cluster_id":
                        obj["cluster_id"],

                    "sub_intent_id":
                        sub[
                            "sub_intent_id"
                        ],

                    "canonical_question":
                        sub[
                            "canonical_question"
                        ],

                    "issue_count":
                        len(
                            sub[
                                "issue_keys"
                            ]
                        ),

                    "issue_keys":
                        " | ".join(
                            sub[
                                "issue_keys"
                            ]
                        ),

                    "description":
                        sub[
                            "description"
                        ],
                }
            )

    clusters_df = pd.DataFrame(
        cluster_rows
    )

    members_df = pd.DataFrame(
        member_rows
    )

    sub_df = pd.DataFrame(
        sub_rows
    )

    class_stats = (
        clusters_df[
            "classification"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "classification"
        )
        .reset_index(
            name="count"
        )
    )

    disposition_stats = (
        members_df[
            "disposition"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "disposition"
        )
        .reset_index(
            name="count"
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "mixed_cluster_count",
                "value":
                    len(
                        clusters_df
                    ),
            },
            {
                "metric":
                    "member_count",
                "value":
                    len(
                        members_df
                    ),
            },
            {
                "metric":
                    "safe_for_next_stage",
                "value":
                    int(
                        (
                            clusters_df[
                                "safe_for_next_stage"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "unsafe_for_next_stage",
                "value":
                    int(
                        (
                            clusters_df[
                                "safe_for_next_stage"
                            ]
                            == False
                        ).sum()
                    ),
            },
            {
                "metric":
                    "conflict_cluster_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "has_conflict"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "temporal_risk_cluster_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "has_temporal_risk"
                            ]
                            == True
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

        class_stats.to_excel(
            writer,
            sheet_name="classification_stats",
            index=False,
        )

        disposition_stats.to_excel(
            writer,
            sheet_name="disposition_stats",
            index=False,
        )

        clusters_df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        members_df.to_excel(
            writer,
            sheet_name="member_decisions",
            index=False,
        )

        if not sub_df.empty:

            sub_df.to_excel(
                writer,
                sheet_name="sub_intents",
                index=False,
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 11.1 - Mixed cluster decomposition audit"
    )
    print("=" * 70)

    validations_df = pd.read_excel(
        VALIDATION_FILE,
        sheet_name="all_clusters",
    )

    members_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    # --------------------------------------------------------
    # 只取 v3 已确认 valid + mixed
    # --------------------------------------------------------

    mixed_df = validations_df[
        (
            validations_df[
                "cluster_validity"
            ]
            == "valid"
        )
        &
        (
            validations_df[
                "answer_relation"
            ]
            == "mixed"
        )
    ].copy()

    print(
        f"Mixed cluster count: "
        f"{len(mixed_df)}"
    )

    processed = load_processed()

    tasks = []

    for _, cluster_row in (
        mixed_df.iterrows()
    ):

        cluster_id = clean_text(
            cluster_row[
                "cluster_id"
            ]
        )

        if cluster_id in processed:
            continue

        canonical_question = (
            clean_text(
                cluster_row.get(
                    "canonical_question"
                )
            )
        )

        cluster_members = (
            members_df[
                members_df[
                    "cluster_id"
                ].astype(str)
                == cluster_id
            ]
            .copy()
        )

        rows = (
            cluster_members
            .to_dict(
                orient="records"
            )
        )

        tasks.append(
            (
                cluster_id,
                canonical_question,
                rows,
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
                        audit_cluster,
                        cluster_id,
                        canonical_question,
                        rows,
                    ):
                    cluster_id

                    for (
                        cluster_id,
                        canonical_question,
                        rows,
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
        f"Excel: {OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()