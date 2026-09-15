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

KB_FILE = (
    OUTPUT_DIR
    / "kb_entries_safe.xlsx"
)

CLUSTER_FILE = (
    OUTPUT_DIR
    / "question_clusters.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "kb_entry_validations.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entry_validations.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class KBValidation(BaseModel):

    verdict: Literal[
        "pass",
        "partial",
        "unsupported",
        "wrong_intent",
        "contradiction",
        "needs_manual_review",
    ]

    confidence: float

    question_alignment: Literal[
        "aligned",
        "partially_aligned",
        "misaligned",
    ]

    grounding: Literal[
        "fully_supported",
        "mostly_supported",
        "partially_supported",
        "unsupported",
    ]

    completeness: Literal[
        "complete",
        "acceptable",
        "missing_key_information",
    ]

    temporal_risk: Literal[
        "none",
        "low",
        "high",
    ]

    unsupported_claims: list[str]

    missing_key_information: list[str]

    contradictions: list[str]

    reason: str


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在审计一条准备进入 CRM RAG 知识库的最终知识条目。

你会看到：

1. 生成后的 canonical_question
2. 生成后的 answer
3. solution_steps
4. notes
5. 原始 Cluster 中所有 issue 的 question / answer / solution
6. 每个 issue 的 resolution / knowledge_value / temporal_status

你的任务不是重新回答问题。

你的唯一任务是：

验证生成知识是否忠实于输入材料。


============================================================
最高优先级原则
============================================================

生成知识中的每一个事实、规则、限制、步骤，
都必须能够从至少一个输入 issue 中得到明确支持。

禁止使用常识补全。

禁止因为某个结论“听起来合理”就认为有依据。

如果输入没有明确说，
就视为 unsupported。


============================================================
一、verdict
============================================================

pass

满足：

- 问题意图正确
- 核心答案有来源支持
- 没有重要虚构内容
- 没有明显矛盾
- 没有把其他问题的答案串进来

允许轻微语言归纳。


partial

生成答案整体方向正确，
但存在：

- 小部分无依据扩展
- 遗漏重要条件
- 某个步骤没有足够依据
- 将有限场景表达得过于普遍

删除或修正少量内容后可以使用。


unsupported

核心答案或大量关键事实，
无法从输入 issue 中找到依据。

例如：

输入只说“需要技术处理”，
生成答案却给出了完整后台菜单路径。


wrong_intent

生成答案实际上回答了另一个问题。

例如：

问题：

车辆年审提醒未显示怎么办？

生成答案：

修改账号手机号、重置密码。

如果原始材料不能证明账号问题就是年审提醒问题的处理方式，
应判 wrong_intent。


contradiction

生成知识内部，
或生成知识与可靠来源之间，
出现不能同时成立的规则。

例如：

生成答案同时说：

“默认密码就是手机号”

以及：

“密码由管理员创建账号时自行设置”

如果没有上下文说明二者适用不同场景，
属于 contradiction。


needs_manual_review

现有来源存在明显冲突、历史版本、
临时状态或证据复杂，
模型无法安全决定。


============================================================
二、question_alignment
============================================================

aligned

生成 answer 明确回答 canonical_question。


partially_aligned

回答只覆盖问题一部分，
或者混入较多相关但非直接内容。


misaligned

答案主要回答了另一个问题。


============================================================
三、grounding
============================================================

fully_supported

所有实质性知识都有明确来源支持。


mostly_supported

绝大部分有依据，
只有很轻微的归纳扩展。


partially_supported

有较明显内容无法回溯到来源。


unsupported

核心内容基本没有来源依据。


============================================================
四、completeness
============================================================

complete

来源中重要且可靠的信息基本得到保留。


acceptable

虽然不是最完整版本，
但足以正确、安全地回答问题。


missing_key_information

遗漏了会明显影响正确使用的重要条件或限制。


============================================================
五、temporal_risk
============================================================

none

内容属于稳定规则，
没有明显时间依赖。


low

来源中存在：
- temporary
- unresolved
- 等待处理
- 可能变化的规则

但生成答案已经谨慎表达。


high

生成答案把明显的：

- temporary
- future_plan
- unresolved
- 临时 workaround
- 历史规则

写成了稳定、长期有效的事实。


============================================================
六、unsupported_claims
============================================================

列出生成知识中无法从来源找到明确支持的具体内容。

不要泛泛写：

“有部分内容缺乏依据”。

应该写成：

[
  "默认密码固定为手机号",
  "必须在PC端操作",
  "修改后系统会自动重新计算"
]

如果没有：

[]


============================================================
七、missing_key_information
============================================================

列出来源中存在，
但生成知识遗漏且会影响正确性的关键内容。

没有则 []。


============================================================
八、contradictions
============================================================

列出具体矛盾。

例如：

[
  "答案同时称默认密码为手机号，又称创建账号时由管理员自行设置密码"
]

没有则 []。


============================================================
特别注意
============================================================

1. 客服来源中的 URL、价格、收费标准、版本状态，
如果被写入最终知识，
必须确认输入 issue 中明确存在。

2. temporary / future_plan 信息不得被包装成稳定规则。

3. “可能”“一般”“通常”也属于知识声明，
不能用这些词来掩盖无来源推测。

例如：

输入没有提到手机号，
生成：

“系统可能默认使用手机号作为密码”

仍然属于 unsupported。


4. 不要因为知识答案写得漂亮就判 pass。

判断的唯一依据是：
是否忠实于来源。


============================================================
输出要求
============================================================

必须只输出合法 JSON object。

不要输出 Markdown。
不要输出 ```json。
不要输出 JSON 以外的内容。

JSON：

{
  "verdict": "pass | partial | unsupported | wrong_intent | contradiction | needs_manual_review",
  "confidence": 0.0,
  "question_alignment": "aligned | partially_aligned | misaligned",
  "grounding": "fully_supported | mostly_supported | partially_supported | unsupported",
  "completeness": "complete | acceptable | missing_key_information",
  "temporal_risk": "none | low | high",
  "unsupported_claims": [],
  "missing_key_information": [],
  "contradictions": [],
  "reason": "简短审计结论"
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
# Prompt builder
# ============================================================

def build_prompt(
    kb_row,
    members,
):

    source_blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        source_blocks.append(
            f"""
------------------------------
SOURCE ISSUE {idx}

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
""".strip()
        )

    return f"""
请审计下面这条生成后的 CRM 知识。

============================================================
GENERATED KB
============================================================

cluster_id:
{clean_text(kb_row.get("cluster_id"))}

canonical_question:
{clean_text(kb_row.get("canonical_question"))}

answer:
{clean_text(kb_row.get("answer"))}

solution_steps:
{clean_text(kb_row.get("solution_steps"))}

notes:
{clean_text(kb_row.get("notes"))}

crm_module:
{clean_text(kb_row.get("crm_module"))}

crm_feature:
{clean_text(kb_row.get("crm_feature"))}

problem_type:
{clean_text(kb_row.get("problem_type"))}


============================================================
ORIGINAL SOURCE ISSUES
============================================================

{chr(10).join(source_blocks)}

请逐条核对生成知识中的事实是否真的有来源依据。
"""


# ============================================================
# API
# ============================================================

def validate_entry(
    kb_row,
    members,
):

    client = build_client()

    prompt = build_prompt(
        kb_row,
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
                                prompt,
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
                KBValidation
                .model_validate(
                    data
                )
            )

            return {
                "cluster_id":
                    clean_text(
                        kb_row.get(
                            "cluster_id"
                        )
                    ),

                "canonical_question":
                    clean_text(
                        kb_row.get(
                            "canonical_question"
                        )
                    ),

                "kb_confidence":
                    kb_row.get(
                        "knowledge_confidence"
                    ),

                "verdict":
                    result.verdict,

                "audit_confidence":
                    result.confidence,

                "question_alignment":
                    result.question_alignment,

                "grounding":
                    result.grounding,

                "completeness":
                    result.completeness,

                "temporal_risk":
                    result.temporal_risk,

                "unsupported_claims":
                    result.unsupported_claims,

                "missing_key_information":
                    result.missing_key_information,

                "contradictions":
                    result.contradictions,

                "reason":
                    result.reason,

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

def export_excel(
    kb_df,
    member_df,
):

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

    audit_df = pd.DataFrame(
        rows
    )

    if audit_df.empty:
        return

    for col in [
        "unsupported_claims",
        "missing_key_information",
        "contradictions",
    ]:

        audit_df[col] = (
            audit_df[col]
            .apply(
                lambda x:
                    " | ".join(
                        str(v)
                        for v in x
                    )
                    if isinstance(x, list)
                    else clean_text(x)
            )
        )

    # --------------------------------------------------------
    # 合并原 KB
    # --------------------------------------------------------

    merged_df = kb_df.merge(
        audit_df,
        on=[
            "cluster_id",
            "canonical_question",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # 允许自动进入下一阶段
    # --------------------------------------------------------

    approved_df = merged_df[
        (
            merged_df["verdict"]
            == "pass"
        )
        &
        (
            merged_df["question_alignment"]
            == "aligned"
        )
        &
        (
            merged_df["grounding"]
            .isin(
                [
                    "fully_supported",
                    "mostly_supported",
                ]
            )
        )
        &
        (
            merged_df["temporal_risk"]
            != "high"
        )
    ].copy()

    review_df = merged_df[
        ~merged_df["cluster_id"].isin(
            approved_df[
                "cluster_id"
            ]
        )
    ].copy()

    review_ids = set(
        review_df[
            "cluster_id"
        ]
        .dropna()
        .tolist()
    )

    review_members = member_df[
        member_df[
            "cluster_id"
        ].isin(
            review_ids
        )
    ].copy()

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    verdict_stats = (
        audit_df[
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

    grounding_stats = (
        audit_df[
            "grounding"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "grounding"
        )
        .reset_index(
            name="count"
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "audited_entry_count",
                "value":
                    len(audit_df),
            },
            {
                "metric":
                    "approved_entry_count",
                "value":
                    len(approved_df),
            },
            {
                "metric":
                    "review_entry_count",
                "value":
                    len(review_df),
            },
            {
                "metric":
                    "pass_count",
                "value":
                    int(
                        (
                            audit_df[
                                "verdict"
                            ]
                            == "pass"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "wrong_intent_count",
                "value":
                    int(
                        (
                            audit_df[
                                "verdict"
                            ]
                            == "wrong_intent"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "contradiction_count",
                "value":
                    int(
                        (
                            audit_df[
                                "verdict"
                            ]
                            == "contradiction"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "high_temporal_risk_count",
                "value":
                    int(
                        (
                            audit_df[
                                "temporal_risk"
                            ]
                            == "high"
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

        grounding_stats.to_excel(
            writer,
            sheet_name="grounding_stats",
            index=False,
        )

        merged_df.to_excel(
            writer,
            sheet_name="all_entries",
            index=False,
        )

        approved_df.to_excel(
            writer,
            sheet_name="approved",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review",
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
        "Step 9.2 - Validate generated KB"
    )
    print("=" * 70)

    kb_df = pd.read_excel(
        KB_FILE,
        sheet_name="kb_entries",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    print(
        f"KB entry count: "
        f"{len(kb_df)}"
    )

    processed = load_processed()

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for _, kb_row in kb_df.iterrows():

        cluster_id = clean_text(
            kb_row.get("cluster_id")
        )

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
                kb_row,
                members,
            )
        )

    print(
        f"本次待处理: "
        f"{len(tasks)}"
    )

    if not tasks:

        export_excel(
            kb_df,
            member_df,
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
                    validate_entry,
                    kb_row,
                    members,
                ):
                clean_text(
                    kb_row.get(
                        "cluster_id"
                    )
                )

                for (
                    kb_row,
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
        kb_df,
        member_df,
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