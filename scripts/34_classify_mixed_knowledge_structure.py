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

MIXED_AUDIT_FILE = (
    OUTPUT_DIR
    / "mixed_cluster_audits_v2.xlsx"
)

GROUNDING_FILE = (
    OUTPUT_DIR
    / "mixed_source_qa_alignments_v2.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "mixed_structure_classifications.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "mixed_structure_classifications.xlsx"
)

MODEL = "qwen3.8-max"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class EvidenceRole(BaseModel):

    issue_key: str

    role: Literal[
        "core_answer",
        "cause",
        "procedure",
        "condition",
        "scenario",
        "limitation",
        "diagnostic",
        "status",
        "temporary_workaround",
        "supporting_detail",
        "weak_evidence",
    ]

    scenario_label: str | None = None

    reason: str


class StructureClassification(BaseModel):

    knowledge_structure: Literal[
        "direct_synthesis",
        "structured_howto",
        "multiple_causes",
        "scenario_branches",
        "troubleshooting",
        "conflict_temporal",
        "insufficient_evidence",
    ]

    confidence: float

    canonical_question: str

    evidence_roles: list[EvidenceRole]

    independent_cause_count: int

    scenario_count: int

    has_true_conflict: bool

    temporal_dependency: Literal[
        "none",
        "minor",
        "material",
        "unknown",
    ]

    generation_ready: bool

    recommended_generation_mode: Literal[
        "single_answer",
        "howto_sections",
        "cause_items",
        "scenario_sections",
        "troubleshooting_steps",
        "do_not_generate",
    ]

    reason: str

    risk_flags: list[str]


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在执行 CRM RAG Mixed Knowledge Structure Gate。

这些 Cluster 已经过：

1. question intent cluster validation
2. mixed decomposition audit
3. source Q/A grounding audit

本轮仍然不要生成最终 KB。

你的任务只是决定：

对于每个 Cluster 中已经通过 grounding 的 source evidence，
未来应该采用什么知识结构来表达。


============================================================
最重要原则
============================================================

你看到的 evidence 已经被筛选为：

usable_for_kb = true

但是：

usable_for_kb 只代表 Q/A 对齐，

不代表：

- 一定稳定
- 一定当前有效
- 一定没有冲突
- 一定可以直接发布


本轮只做：

Knowledge Structure Classification。


============================================================
只能使用 Grounded Evidence
============================================================

输入中：

supported_answer

是最主要证据。

supported_solution

只有在 Grounding Gate 保留下来时才能使用。


禁止恢复已经被 Grounding Gate 删除的信息。

禁止：

- 使用原 Solution 中被删除的信息
- 补充 CRM 常识
- 补菜单路径
- 补 URL
- 补时间
- 补规则
- 补操作


============================================================
knowledge_structure
============================================================


------------------------------------------------------------
1. direct_synthesis
------------------------------------------------------------

多个 source 实质上都在回答：

同一个问题
+
同一个答案核心。

差异只是：

- 措辞不同
- 信息完整度不同
- 相互补充少量细节

不需要：

- 多原因列表
- 场景分支
- troubleshooting 流程

未来可生成一条普通答案。


例如：

问题：
“在哪里查看回访统计？”

A：
在回访统计页面查看。

B：
进入回访管理后查看统计结果。

→ direct_synthesis


------------------------------------------------------------
2. structured_howto
------------------------------------------------------------

Question 主要是：

“如何做？”
“怎么操作？”
“流程是什么？”

Evidence 可以组成一个相对明确的：

- 前置条件
- 操作步骤
- 注意事项
- 限制

它们是同一个操作目标，

不是不同故障原因。


例如：

“如何创建 CRM 账号？”

Evidence：

A：
进入权限中心创建。

B：
需要手机号。

C：
首次登录需要修改密码。

如果这些信息能够共同形成一个 How-to 结构：

→ structured_howto


注意：

如果不同 source 实际是在不同条件下使用完全不同流程，
应考虑 scenario_branches。


------------------------------------------------------------
3. multiple_causes
------------------------------------------------------------

Question 主要是在问：

- 为什么
- 什么原因
- 为什么会出现某异常

并且 evidence 中存在至少 2 个：

相互独立的原因 / 机制。

例如：

“为什么回访数量对不上？”

A：
筛选时间字段错误。

B：
财务修改过结算时间。

C：
部分工单没有配置为需要回访。

→ multiple_causes


非常重要：

不同原因不是不同 scenario。

如果它们可以自然写成：

“可能原因包括……”

优先 multiple_causes。


------------------------------------------------------------
4. scenario_branches
------------------------------------------------------------

同一个用户问题，

但不同答案只在某些明确条件下成立。

必须按照条件分支表达。

结构类似：

如果 A：
    做 X

如果 B：
    做 Y

如果 C：
    做 Z


例如：

问题：

“如何修改客户手机号？”

Evidence：

A：
手机号未被占用时，直接修改。

B：
手机号已存在其他客户时，需要先处理重复客户。

这种属于：

scenario_branches


Scenario 必须有 Grounded 条件。

禁止自行发明条件。


------------------------------------------------------------
5. troubleshooting
------------------------------------------------------------

Question 是：

“功能异常怎么办？”

Evidence 不是简单的多个原因，

而是形成一个合理排查过程：

1. 先检查 A
2. 再尝试 B
3. 不行再处理 C

或：

- 检查配置
- 刷新
- 重登
- 远程排查

这种以诊断 / 排查流程为主：

→ troubleshooting


与 multiple_causes 区别：

multiple_causes：
回答“有哪些原因”。

troubleshooting：
回答“出现问题后按什么顺序检查/处理”。


============================================================
6. conflict_temporal
============================================================

如果 evidence 在相同问题和相同条件下出现：

互相不兼容的结论，

例如：

A：
默认密码是手机号。

B：
默认密码是 123456。

C：
默认密码是 6 个 1。

而 Source 没有明确条件能够解释差异，

则不能自动合并。

→ conflict_temporal


又例如：

A：
入口在旧菜单。

B：
入口已改到新菜单。

如果可能属于版本变化：

→ conflict_temporal


以下不属于 conflict：

- 不同原因
- 不同 troubleshooting 方法
- 不同条件下不同方案
- 一个答案更详细
- 一个答案只回答部分内容


============================================================
7. insufficient_evidence
============================================================

虽然 source 通过 Q/A Grounding，

但整体证据仍不足以安全形成可复用知识。

例如：

- 只有单次具体案例
- 答案只是非常模糊的 workaround
- 关键规则缺失
- 无法形成清晰结构
- evidence 全部高度临时且没有稳定知识核心

→ insufficient_evidence


============================================================
Cause 与 Scenario 必须严格区分
============================================================

例如：

问题：

“为什么系统无法登录？”

A：
密码错误。

B：
账号停用。

C：
访问地址错误。

这是：

multiple_causes

不是：

scenario_branches。


因为这些是在回答：

不同“原因”。


而：

问题：

“如何登录系统？”

A：
管理员账号 → 使用入口 A。

B：
普通账号 → 使用入口 B。

这是：

scenario_branches

因为方案依赖明确条件。


============================================================
Cause 与 Troubleshooting
============================================================

如果 evidence 是：

“可能因为 A、B、C”

→ multiple_causes


如果 evidence 是：

“先检查 A；
如果没有问题再检查 B；
最后尝试 C”

→ troubleshooting


如果 source 没有明确顺序，

不要自动创造排查顺序。

这种情况下应优先：

multiple_causes
或 scenario_branches。


============================================================
Temporal Risk
============================================================

输入会包含：

temporal_status

以及 Step 11.1 的：

has_temporal_risk


不要只因为出现 temporary
就判 conflict_temporal。

temporary 可能只是：

某个 case 当时发生故障。


只有当知识内容本身依赖：

- 当前版本
- 临时产品 Bug
- 当时政策
- 当时入口
- 当时默认值
- 临时 workaround

并且会影响最终答案有效性时，

temporal_dependency 才应：

material


如果只是个别 source temporary，

但核心知识明显稳定：

temporal_dependency 可以：

minor


============================================================
generation_ready
============================================================

generation_ready = true

意味着：

下一阶段可以根据现有 grounded evidence
生成“候选 KB”。

不是正式发布。


以下通常可以 true：

direct_synthesis
structured_howto
multiple_causes
scenario_branches
troubleshooting


以下必须 false：

conflict_temporal
insufficient_evidence


如果存在 material temporal dependency，
且无法从 evidence 判断当前有效答案：

必须 false。


============================================================
Evidence Role
============================================================

每个 issue_key 必须输出一次。

允许：

core_answer
cause
procedure
condition
scenario
limitation
diagnostic
status
temporary_workaround
supporting_detail
weak_evidence


例如：

Question：
“为什么登录失败？”

Answer：
“密码错误。”

role =
cause


Question：
“如何创建账号？”

Answer：
“进入权限中心创建。”

role =
procedure


Question：
“如果手机号已经存在怎么办？”

Answer：
“需先处理重复客户。”

role 可以：

scenario
或 condition


============================================================
independent_cause_count
============================================================

只有真正独立原因才计数。

不同措辞重复同一原因：

只算 1。


如果不是 multiple_causes，
也可以是 0 或 1。


如果：

knowledge_structure = multiple_causes

原则上必须：

independent_cause_count >= 2


否则 generation_ready=false。


============================================================
scenario_count
============================================================

只有真实条件分支才计数。

如果：

knowledge_structure = scenario_branches

原则上必须：

scenario_count >= 2


否则 generation_ready=false。


============================================================
canonical_question
============================================================

尽量沿用输入 Cluster 的 core_question /
canonical_question。

只允许轻微规范化。

不能改变用户意图。


============================================================
严格禁止
============================================================

本轮禁止：

- 生成最终 Answer
- 合并成完整 KB 文案
- 补充 source 不存在的事实
- 判断正式发布
- 自动选择“当前最新版”
- 从常识推断 CRM 产品逻辑

本轮只：

分类结构。


============================================================
输出
============================================================

只输出合法 JSON：

{
  "knowledge_structure": "direct_synthesis | structured_howto | multiple_causes | scenario_branches | troubleshooting | conflict_temporal | insufficient_evidence",
  "confidence": 0.0,
  "canonical_question": "规范问题",
  "evidence_roles": [
    {
      "issue_key": "输入中的 issue_key",
      "role": "core_answer | cause | procedure | condition | scenario | limitation | diagnostic | status | temporary_workaround | supporting_detail | weak_evidence",
      "scenario_label": null,
      "reason": "简短原因"
    }
  ],
  "independent_cause_count": 0,
  "scenario_count": 0,
  "has_true_conflict": false,
  "temporal_dependency": "none | minor | material | unknown",
  "generation_ready": true,
  "recommended_generation_mode": "single_answer | howto_sections | cause_items | scenario_sections | troubleshooting_steps | do_not_generate",
  "reason": "为什么选择该结构",
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
# Utils
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

                if obj.get("cluster_id"):

                    processed.add(
                        obj["cluster_id"]
                    )

            except Exception:
                pass

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    cluster_row,
    source_rows,
):

    blocks = []

    for idx, row in enumerate(
        source_rows,
        start=1,
    ):

        blocks.append(
            f"""
============================================================
GROUNDED EVIDENCE {idx}
============================================================

issue_key:
{clean_text(row.get("issue_key"))}

question:
{clean_text(row.get("question"))}

question_normalized:
{clean_text(row.get("question_normalized"))}

supported_answer:
{clean_text(row.get("supported_answer"))}

supported_solution:
{clean_text(row.get("supported_solution"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}

source_confidence:
{clean_text(row.get("source_confidence"))}

alignment:
{clean_text(row.get("alignment"))}

alignment_confidence:
{clean_text(row.get("alignment_confidence"))}
""".strip()
        )

    return f"""
请对下面的 Mixed Cluster 做 Knowledge Structure Classification。

============================================================
CLUSTER
============================================================

cluster_id:
{clean_text(cluster_row.get("cluster_id"))}

canonical_question:
{clean_text(cluster_row.get("canonical_question"))}

core_question:
{clean_text(cluster_row.get("core_question"))}

Step 11.1 classification:
{clean_text(cluster_row.get("classification"))}

Step 11.1 answer_structure_types:
{clean_text(cluster_row.get("answer_structure_types"))}

Step 11.1 has_conflict:
{clean_text(cluster_row.get("has_conflict"))}

Step 11.1 has_temporal_risk:
{clean_text(cluster_row.get("has_temporal_risk"))}

grounded_source_count:
{len(source_rows)}


{chr(10).join(blocks)}
"""


# ============================================================
# Classify
# ============================================================

def classify_cluster(
    cluster_row,
    source_rows,
):

    client = build_client()

    cluster_id = clean_text(
        cluster_row.get(
            "cluster_id"
        )
    )

    allowed_keys = {
        clean_text(
            row.get(
                "issue_key"
            )
        )
        for row in source_rows
    }

    prompt = build_prompt(
        cluster_row,
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
                client
                .chat
                .completions
                .create(
                    model=MODEL,
                    messages=[
                        {
                            "role":
                                "system",
                            "content":
                                SYSTEM_PROMPT,
                        },
                        {
                            "role":
                                "user",
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
                StructureClassification
                .model_validate(
                    data
                )
            )

            output_keys = {
                item.issue_key
                for item
                in result.evidence_roles
            }

            if output_keys != allowed_keys:

                raise RuntimeError(
                    "evidence_roles 未完整覆盖 source issue_key。"
                    f" expected={sorted(allowed_keys)}"
                    f" actual={sorted(output_keys)}"
                )

            # =================================================
            # Hard Rules
            # =================================================

            if (
                result.knowledge_structure
                == "multiple_causes"
                and
                result.independent_cause_count
                < 2
            ):

                result.generation_ready = (
                    False
                )

            if (
                result.knowledge_structure
                == "scenario_branches"
                and
                result.scenario_count
                < 2
            ):

                result.generation_ready = (
                    False
                )

            if result.knowledge_structure in {
                "conflict_temporal",
                "insufficient_evidence",
            }:

                result.generation_ready = (
                    False
                )

                result.recommended_generation_mode = (
                    "do_not_generate"
                )

            if (
                result.temporal_dependency
                == "material"
                and
                result.has_true_conflict
            ):

                result.generation_ready = (
                    False
                )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    result.canonical_question,

                "source_count":
                    len(source_rows),

                "knowledge_structure":
                    result.knowledge_structure,

                "structure_confidence":
                    result.confidence,

                "independent_cause_count":
                    result.independent_cause_count,

                "scenario_count":
                    result.scenario_count,

                "has_true_conflict":
                    result.has_true_conflict,

                "temporal_dependency":
                    result.temporal_dependency,

                "generation_ready":
                    result.generation_ready,

                "recommended_generation_mode":
                    result.recommended_generation_mode,

                "reason":
                    result.reason,

                "risk_flags":
                    result.risk_flags,

                "evidence_roles":
                    [
                        item.model_dump()
                        for item
                        in result.evidence_roles
                    ],

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
# Excel Export
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
    role_rows = []

    for obj in objects:

        cluster_rows.append(
            {
                "cluster_id":
                    obj["cluster_id"],

                "canonical_question":
                    obj[
                        "canonical_question"
                    ],

                "source_count":
                    obj[
                        "source_count"
                    ],

                "knowledge_structure":
                    obj[
                        "knowledge_structure"
                    ],

                "structure_confidence":
                    obj[
                        "structure_confidence"
                    ],

                "independent_cause_count":
                    obj[
                        "independent_cause_count"
                    ],

                "scenario_count":
                    obj[
                        "scenario_count"
                    ],

                "has_true_conflict":
                    obj[
                        "has_true_conflict"
                    ],

                "temporal_dependency":
                    obj[
                        "temporal_dependency"
                    ],

                "generation_ready":
                    obj[
                        "generation_ready"
                    ],

                "recommended_generation_mode":
                    obj[
                        "recommended_generation_mode"
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
            "evidence_roles",
            []
        ):

            role_rows.append(
                {
                    "cluster_id":
                        obj["cluster_id"],

                    "issue_key":
                        item[
                            "issue_key"
                        ],

                    "role":
                        item[
                            "role"
                        ],

                    "scenario_label":
                        item.get(
                            "scenario_label"
                        ),

                    "reason":
                        item[
                            "reason"
                        ],
                }
            )

    clusters_df = pd.DataFrame(
        cluster_rows
    )

    roles_df = pd.DataFrame(
        role_rows
    )

    structure_stats = (
        clusters_df[
            "knowledge_structure"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "knowledge_structure"
        )
        .reset_index(
            name="count"
        )
    )

    temporal_stats = (
        clusters_df[
            "temporal_dependency"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "temporal_dependency"
        )
        .reset_index(
            name="count"
        )
    )

    generation_ready_df = (
        clusters_df[
            clusters_df[
                "generation_ready"
            ]
            == True
        ]
        .copy()
    )

    blocked_df = (
        clusters_df[
            clusters_df[
                "generation_ready"
            ]
            != True
        ]
        .copy()
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "cluster_count",
                "value":
                    len(
                        clusters_df
                    ),
            },
            {
                "metric":
                    "generation_ready_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "generation_ready"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "blocked_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "generation_ready"
                            ]
                            != True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "true_conflict_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "has_true_conflict"
                            ]
                            == True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "material_temporal_count",
                "value":
                    int(
                        (
                            clusters_df[
                                "temporal_dependency"
                            ]
                            == "material"
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

        structure_stats.to_excel(
            writer,
            sheet_name="structure_stats",
            index=False,
        )

        temporal_stats.to_excel(
            writer,
            sheet_name="temporal_stats",
            index=False,
        )

        clusters_df.to_excel(
            writer,
            sheet_name="all_clusters",
            index=False,
        )

        roles_df.to_excel(
            writer,
            sheet_name="evidence_roles",
            index=False,
        )

        generation_ready_df.to_excel(
            writer,
            sheet_name="generation_ready",
            index=False,
        )

        blocked_df.to_excel(
            writer,
            sheet_name="blocked",
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 11.3 - Mixed Knowledge Structure Gate"
    )
    print("=" * 70)

    mixed_df = pd.read_excel(
        MIXED_AUDIT_FILE,
        sheet_name="all_clusters",
    )

    grounding_df = pd.read_excel(
        GROUNDING_FILE,
        sheet_name="all_sources",
    )

    # --------------------------------------------------------
    # 只使用 Grounding Gate 通过的 evidence
    # --------------------------------------------------------

    usable_df = (
        grounding_df[
            grounding_df[
                "usable_for_kb"
            ]
            == True
        ]
        .copy()
    )

    print(
        f"Mixed cluster count: "
        f"{len(mixed_df)}"
    )

    print(
        f"Usable grounded sources: "
        f"{len(usable_df)}"
    )

    processed = load_processed()

    tasks = []

    for _, cluster_row in (
        mixed_df.iterrows()
    ):

        cluster_id = clean_text(
            cluster_row.get(
                "cluster_id"
            )
        )

        if cluster_id in processed:
            continue

        source_rows = (
            usable_df[
                usable_df[
                    "cluster_id"
                ]
                .astype(str)
                == cluster_id
            ]
            .to_dict(
                orient="records"
            )
        )

        if not source_rows:

            print(
                f"[SKIP] {cluster_id}: "
                f"NO_VALID_SOURCE"
            )

            continue

        tasks.append(
            (
                cluster_row.to_dict(),
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
                        classify_cluster,
                        cluster_row,
                        source_rows,
                    ):
                    clean_text(
                        cluster_row.get(
                            "cluster_id"
                        )
                    )

                    for (
                        cluster_row,
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
                        f"{eta / 60:.1f} min"
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