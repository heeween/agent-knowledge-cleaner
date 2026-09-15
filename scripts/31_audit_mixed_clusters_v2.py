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
    / "mixed_cluster_audits_v2.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
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
你正在审核 CRM RAG 中 answer_relation=mixed 的 question cluster。

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
最高优先级原则
============================================================

必须严格区分：

“不同用户意图”

和

“同一个用户意图下的不同原因 / 不同处理路径”。

只有“用户想完成的目标不同”，
才能拆 sub_intents。

不同原因绝对不能作为拆 sub_intent 的依据。


============================================================
非常重要：Cause 不等于 Sub-intent
============================================================

例如：

问题：
“为什么短信发送失败？”

成员 A：
余额不足。

成员 B：
敏感词。

成员 C：
运营商报备异常。

这是：

一个用户意图
+
三个不同原因。

必须判：

single_intent_structurable

禁止拆成三个 sub_intents。


再例如：

问题：
“已回访客户为什么仍在待回访列表？”

成员 A：
车辆匹配错误。

成员 B：
系统未识别最新回访状态。

虽然根因不同，
但用户目标完全相同：

“解释为什么仍显示”。

必须判：

single_intent_structurable。


再例如：

问题：
“回访数据数量为什么不一致？”

成员 A：
系统 Bug。

成员 B：
筛选条件选错。

这是不同原因，
不是不同意图。

必须判：

single_intent_structurable。


再例如：

问题：
“车主画像为什么显示空白？”

成员 A：
客户端加载滞后。

成员 B：
系统 Bug。

仍然只是同一个故障问题下的不同原因。

必须判：

single_intent_structurable。


============================================================
判断 sub_intents 的唯一标准
============================================================

只有当成员对应的“用户请求目标”本身不同，
才允许拆 sub_intents。

用户请求目标包括：

- 想知道原因
- 想完成某个操作
- 想确认是否支持
- 想知道规则
- 想查询状态
- 想修改配置
- 想恢复功能
- 想找到入口
- 想理解业务机制

如果两个成员只是：

“为什么会这样”的不同原因，

它们不是 sub_intents。


============================================================
什么时候才允许 sub_intents
============================================================

例如：

A：
为什么车辆还显示在流失邀约列表？

B：
怎么把车辆从流失邀约列表移除？

A 的目标是：

理解原因 / 机制。

B 的目标是：

执行移除操作。

这是两个可以独立检索、
独立回答的问题。

可以拆：

sub_intents。


再例如：

A：
默认密码是什么？

B：
忘记密码怎么办？

C：
如何修改密码？

虽然都属于“密码”领域，
但用户目标分别是：

- 查询默认规则
- 找回访问能力
- 主动修改密码

这是不同用户意图。

可以拆：

sub_intents。


再例如：

A：
是否支持导出客户数据？

B：
如何导出客户数据？

第一个问题目标是：

确认产品能力。

第二个问题目标是：

执行操作。

这是两个不同意图。

可以拆。


============================================================
禁止因为以下情况拆 sub_intents
============================================================

以下情况都不能单独作为拆分依据：

- 不同原因
- 不同根因
- 不同解决方法
- 不同处理路径
- 不同系统层级
- 一个是 Bug，一个是配置问题
- 一个是用户侧问题，一个是系统侧问题
- 一个是临时故障，一个是稳定机制
- resolution 不同
- temporal_status 不同
- knowledge_value 不同
- answer 长短不同
- source 不同
- 同一个问题下有多种可能情况


============================================================
拆分前必须执行 3 个测试
============================================================

在决定：

classification = sub_intents

之前，必须先执行以下测试。


------------------------------------------------------------
Test 1：Possible Causes Test
------------------------------------------------------------

如果把这些成员的答案写成：

“可能原因包括 A、B、C……”

是否仍然自然回答同一个 canonical question？

如果答案是：

是

则这些成员属于：

同一用户意图下的不同原因。

不得拆 sub_intents。


------------------------------------------------------------
Test 2：Shared Question Test
------------------------------------------------------------

这些成员是否可以共享同一个 canonical question，
只是答案内容不同？

如果可以，

不得拆 sub_intents。


------------------------------------------------------------
Test 3：Diagnostic Branch Test
------------------------------------------------------------

如果用户搜索其中一个成员的问题，

另一个成员的答案是否仍然可以作为：

- 合理诊断分支
- 另一种可能原因
- 另一种解决路径
- 另一种条件说明

如果可以，

不得拆 sub_intents。


只有当这三个测试都无法解释成员差异，
并且用户目标本身确实不同，

才允许 sub_intents。


============================================================
single_intent_structurable
============================================================

当所有或绝大多数成员都围绕同一个用户目标时，

classification =
single_intent_structurable


即使答案包含多个不同结构：

- direct answer
- cause
- condition
- procedure
- limitation
- diagnostic
- status
- temporary workaround

只要用户目标一致，
仍然属于 single intent。


例如：

问题：

“无法登录 CRM 怎么办？”

不同成员分别说：

- 密码错误
- 账号停用
- 登录地址变更
- 浏览器问题

仍然是一条意图：

“解决 CRM 登录失败”。

不能拆成四个 sub-intent。


再例如：

问题：

“为什么企业微信消息推送失败？”

不同成员分别说：

- 客户删除好友
- 接口权限异常
- 群配置异常
- 系统同步失败

仍然可能属于：

同一个故障问题下的不同原因。

不能仅因为原因不同进行拆分。


============================================================
mixed_with_noise
============================================================

当 Cluster 主体确实属于同一个核心问题，
但少量成员明显不属于时：

classification =
mixed_with_noise


以下情况可以判 noise：

- Q/A 错位
- answer 实际回答另一个问题
- solution 与 question 无关
- 抽取边界污染
- 没有有效答案
- 只有“看一下”“处理中”等无知识内容
- question 看似类似，但实际解决目标不同且只是孤立成员


例如：

5 个成员都在回答：

“如何导出客户数据？”

其中 1 个成员 answer 实际在讲：

“如何修改密码”。

这个成员：

disposition = noise

不要因为一个 noise
把整个 Cluster 判为 sub_intents。


============================================================
noise 与 sub_intent 的区别
============================================================

noise：

该成员不应该属于这个 Cluster，
或者其 Q/A 本身失真、错位、无效。

separate_sub_intent：

该成员本身是有效知识，
只是它的用户目标与 Cluster 其他成员不同。


例如：

Cluster：

A：
为什么车辆还在流失列表？

B：
如何把车辆移出流失列表？

B 不是 noise。

B 是一个有效的独立用户意图。

所以：

separate_sub_intent。


而如果：

C：
如何修改客户手机号？

明显与流失列表无关，

则：

noise。


============================================================
unsafe_mixed
============================================================

只有在以下情况下使用：

classification =
unsafe_mixed

例如：

- 意图严重混杂
- 多个成员 Q/A 错位
- 无法可靠识别核心问题
- 关键成员证据不足
- 无法确定哪些成员应该保留
- 存在多个相互矛盾的解释且无法归因
- 无法安全形成下一阶段输入


unsafe_mixed 时：

safe_for_next_stage = false


============================================================
Member disposition
============================================================

每个成员必须给出 disposition。


keep

该成员属于 Cluster 的核心问题。


noise

该成员明显不回答核心问题，
或者存在明显抽取污染。


separate_sub_intent

该成员是有效知识，
但用户目标与核心问题不同。

只有真正不同用户目标时才能使用。


uncertain

无法可靠判断。


============================================================
sub_intent_id
============================================================

只有当：

classification = sub_intents

时，才应该存在正式 sub_intents。


每个 sub_intent 必须满足：

1. 用户目标独立
2. 可以独立检索
3. 可以独立回答
4. canonical_question 与其他 sub_intent 明显不同
5. 不能只是不同原因


不能因为：

“一个是系统问题”

“一个是配置问题”

“一个是 Bug”

“一个是用户操作错误”

就拆成不同 sub_intent。


============================================================
SubIntent canonical_question
============================================================

sub_intent 的 canonical_question
必须描述真实用户问题。

不要写成答案标签。

错误：

“系统 Bug 原因”

“配置问题”

“接口类问题”


正确：

“为什么已完成回访的客户仍显示在待回访列表中？”

“如何把车辆从流失邀约列表中移除？”


============================================================
answer_structure_types
============================================================

只描述这个 Cluster 中出现的答案结构。

允许值：

direct_answer
cause
condition
procedure
limitation
diagnostic
status
temporary_workaround
other


可以同时存在多个。


例如：

一个 Cluster 同时包含：

- 原因解释
- 检查方法
- 操作步骤

则可以输出：

[
  "cause",
  "diagnostic",
  "procedure"
]


answer_structure_types
不能作为拆 sub_intents 的依据。


============================================================
Conflict
============================================================

不同原因不是 conflict。

不同处理路径也不自动是 conflict。

只有在：

相同问题
+
相同条件

下出现互斥结论，
才算 conflict。


例如：

A：
系统支持导出。

B：
系统不支持导出。

这是 conflict。


又例如：

A：
必须打开某开关。

B：
明确说明不需要打开该开关。

这是 conflict。


但是：

A：
短信失败可能因为余额不足。

B：
短信失败可能因为敏感词。

不是 conflict。


============================================================
Temporal Risk
============================================================

temporary
future_plan
historical

只是时效风险信息。

不能作为拆意图依据。


如果答案依赖：

- 当前版本
- 当时系统故障
- 临时政策
- 临时 workaround
- 当时运营商规则

则：

has_temporal_risk = true


但：

temporary 不等于 noise。

例如：

某次故障中确认：

“未切换耳机模式会导致电话没有声音”

即使 source 属于 temporary，

这个原因本身仍然可能是稳定机制。


============================================================
safe_for_next_stage
============================================================

以下通常可以：

safe_for_next_stage = true


single_intent_structurable

结构清楚的 mixed_with_noise

真正明确的 sub_intents


以下必须：

safe_for_next_stage = false


unsafe_mixed

存在关键 uncertain

sub_intents 无法形成清晰分组


============================================================
core_question
============================================================

core_question 应表达：

这个 Cluster 最主要的用户目标。

如果 classification = sub_intents，

core_question 可以是该 Cluster 的上位主题，
但不能掩盖真实子问题。


============================================================
issue_key 安全要求
============================================================

所有输出的 issue_key：

必须严格来自输入。

禁止：

- 修改 issue_key
- 猜测 issue_key
- 新增 issue_key
- 删除输入成员而不在 members 中输出

members 必须覆盖所有输入成员。


============================================================
最终分类优先级
============================================================

建议按以下顺序判断：

第一步：

是否存在严重不可判断问题？

是：
unsafe_mixed


第二步：

主体是否同一个用户目标，
只是夹杂少量错误成员？

是：
mixed_with_noise


第三步：

是否真正存在两个或以上独立用户目标？

只有通过 3 个 sub-intent 测试后才能判：

sub_intents


第四步：

否则：

single_intent_structurable


============================================================
输出要求
============================================================

只输出合法 JSON。

不要输出 Markdown。
不要解释 JSON 外的内容。

格式：

{
  "classification": "single_intent_structurable | sub_intents | mixed_with_noise | unsafe_mixed",
  "confidence": 0.0,
  "core_question": "核心用户问题",
  "members": [
    {
      "issue_key": "必须来自输入",
      "disposition": "keep | noise | separate_sub_intent | uncertain",
      "sub_intent_id": null,
      "reason": "简短说明"
    }
  ],
  "sub_intents": [
    {
      "sub_intent_id": "SUB-01",
      "canonical_question": "真实用户问题",
      "issue_keys": [
        "必须来自输入"
      ],
      "description": "说明该子意图的用户目标"
    }
  ],
  "answer_structure_types": [
    "cause",
    "procedure"
  ],
  "has_conflict": false,
  "has_temporal_risk": false,
  "safe_for_next_stage": true,
  "reason": "整体判断依据",
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