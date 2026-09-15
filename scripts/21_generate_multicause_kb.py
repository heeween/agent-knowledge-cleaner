import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Optional

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
    / "kb_entries_multicause.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_multicause.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class CauseItem(BaseModel):

    cause: str

    check_or_action: Optional[str] = None

    source_issue_keys: list[str]


class MultiCauseEntry(BaseModel):

    canonical_question: str

    summary_answer: str

    causes: list[CauseItem]

    notes: list[str]

    crm_module: str

    crm_feature: str

    problem_type: str

    knowledge_confidence: float


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在构建 CRM RAG 知识库中的“多原因故障知识”。

输入是一组被验证为：

answer_relation = multiple_causes

的 issue。

这些 issue 的问题意图相同，
但不同来源可能给出了不同原因或不同排查方式。

你的任务不是把答案简单拼接。

你的任务是：

整理出一个结构化的“可能原因列表”。


============================================================
最高优先级原则
============================================================

所有原因、检查方法、解决方法：

必须来自输入 issue 的明确内容。

禁止使用：

- CRM 常识
- 经验猜测
- 推理出来的新原因
- 输入中不存在的菜单路径
- 输入中不存在的技术原理


============================================================
一、canonical_question
============================================================

优先使用输入中的 canonical_question。

可以轻微润色。

不得扩大问题范围。


============================================================
二、summary_answer
============================================================

只做简短总述。

例如：

“该问题可能由多种原因导致，可按以下已知原因逐项检查。”

不要在 summary 中新增原因。


============================================================
三、causes
============================================================

每个 cause 表示一个独立、明确的原因。

格式：

{
  "cause": "原因",
  "check_or_action": "对应检查或处理方式",
  "source_issue_keys": []
}


============================================================
原因去重
============================================================

语义相同的原因必须合并。

例如：

“短信签名没有报备”

和：

“当前短信签名尚未完成报备”

应该合成一个原因。


但是：

“短信签名未报备”

和：

“短信内容存在敏感词”

是两个不同原因。

不得合并。


============================================================
不要制造“冲突”
============================================================

不同 source 提供不同原因，
不表示它们互相冲突。

例如：

A：
发送失败是因为签名未报备。

B：
发送失败是因为敏感词。

正确：

原因1：签名未报备
原因2：敏感词

错误：

“来源存在冲突，需要确认哪个说法正确。”


============================================================
check_or_action
============================================================

只有 source 中明确存在检查方法或处理方式，
才填写。

例如：

cause：
短信余额不足

source 明确说：
需要充值短信余额

则：

check_or_action：
“检查短信余额，不足时充值。”


如果 source 只描述原因，
没有说明怎么检查或处理：

null


禁止自己发明：

- 排查顺序
- 后台入口
- 菜单路径
- 自动修复方法


============================================================
source_issue_keys
============================================================

每一个 cause 都必须保留真正支持该原因的 issue_key。

只能填写输入中存在的 issue_key。

一个原因可以有多个 source。


============================================================
四、notes
============================================================

只记录必要且有来源支持的限制。

例如：

- 某原因仅出现在特定场景
- 来源标记 temporary
- 仍需技术确认
- 某操作有权限限制

没有则 []。


============================================================
五、稳定性
============================================================

如果某个 source：

temporal_status = temporary

或者：

resolution = unresolved

不要把其内容写成稳定确定事实。

应该使用：

“已有案例显示可能……”
“该原因仍需进一步确认……”

并在 notes 中说明。


============================================================
六、问题范围
============================================================

特别注意：

如果问题是：

“短信为什么发送失败？”

可以整理多个失败原因。


但如果某个 source 实际回答的是：

“短信发送失败是否扣费？”

不要把“失败不扣费”当成失败原因。


只允许与 canonical_question
直接相关的内容进入 causes。


============================================================
七、知识价值
============================================================

客服语言：

“我看看”
“反馈技术”
“稍等”
“已经处理”

不能作为 cause。

除非其中包含明确故障原因。


============================================================
八、knowledge_confidence
============================================================

根据：

- 原因是否有明确来源
- 多来源是否一致
- 是否 stable
- 是否 resolved
- 是否存在推测

给 0~1。

不要因为原因数量多就自动给高分。


============================================================
输出要求
============================================================

只输出合法 JSON object。

不要 Markdown。

格式：

{
  "canonical_question": "问题",
  "summary_answer": "简短总述",
  "causes": [
    {
      "cause": "原因",
      "check_or_action": "处理方法或 null",
      "source_issue_keys": [
        "ISSUE-..."
      ]
    }
  ],
  "notes": [],
  "crm_module": "模块",
  "crm_feature": "功能",
  "problem_type": "类型",
  "knowledge_confidence": 0.0
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
    cluster_row,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        blocks.append(
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
请将下面这个 multiple_causes Cluster
整理成结构化多原因知识。

cluster_id:
{clean_text(cluster_row.get("cluster_id"))}

canonical_question:
{clean_text(cluster_row.get("canonical_question"))}

answer_relation:
{clean_text(cluster_row.get("answer_relation"))}

validation_confidence:
{clean_text(cluster_row.get("validation_confidence"))}


============================================================
SOURCE ISSUES
============================================================

{chr(10).join(blocks)}
"""


# ============================================================
# API
# ============================================================

def generate_entry(
    cluster_row,
    members,
):

    client = build_client()

    cluster_id = clean_text(
        cluster_row.get(
            "cluster_id"
        )
    )

    valid_keys = {
        clean_text(
            row.get("issue_key")
        )
        for row in members
    }

    prompt = build_prompt(
        cluster_row,
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
                MultiCauseEntry
                .model_validate(
                    data
                )
            )

            # ------------------------------------------------
            # 强制清理 source_issue_keys
            # ------------------------------------------------

            cleaned_causes = []

            for cause in result.causes:

                keys = [
                    key
                    for key
                    in cause.source_issue_keys
                    if key in valid_keys
                ]

                # 没有来源的原因直接丢弃
                if not keys:
                    continue

                cleaned_causes.append(
                    {
                        "cause":
                            cause.cause,

                        "check_or_action":
                            cause.check_or_action,

                        "source_issue_keys":
                            keys,
                    }
                )

            return {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    result.canonical_question,

                "summary_answer":
                    result.summary_answer,

                "causes":
                    cleaned_causes,

                "notes":
                    result.notes,

                "crm_module":
                    result.crm_module,

                "crm_feature":
                    result.crm_feature,

                "problem_type":
                    result.problem_type,

                "knowledge_confidence":
                    result.knowledge_confidence,

                "cause_count":
                    len(
                        cleaned_causes
                    ),

                "member_count":
                    len(members),

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

    if not rows:
        return

    entry_rows = []
    cause_rows = []

    for obj in rows:

        cluster_id = obj[
            "cluster_id"
        ]

        causes = obj.get(
            "causes",
            []
        )

        entry_rows.append(
            {
                "cluster_id":
                    cluster_id,

                "canonical_question":
                    obj.get(
                        "canonical_question"
                    ),

                "summary_answer":
                    obj.get(
                        "summary_answer"
                    ),

                "notes":
                    " | ".join(
                        obj.get(
                            "notes",
                            []
                        )
                    ),

                "crm_module":
                    obj.get(
                        "crm_module"
                    ),

                "crm_feature":
                    obj.get(
                        "crm_feature"
                    ),

                "problem_type":
                    obj.get(
                        "problem_type"
                    ),

                "knowledge_confidence":
                    obj.get(
                        "knowledge_confidence"
                    ),

                "cause_count":
                    len(causes),

                "member_count":
                    obj.get(
                        "member_count"
                    ),

                "request_seconds":
                    obj.get(
                        "request_seconds"
                    ),
            }
        )

        for idx, cause in enumerate(
            causes,
            start=1,
        ):

            cause_rows.append(
                {
                    "cluster_id":
                        cluster_id,

                    "canonical_question":
                        obj.get(
                            "canonical_question"
                        ),

                    "cause_index":
                        idx,

                    "cause":
                        cause.get(
                            "cause"
                        ),

                    "check_or_action":
                        cause.get(
                            "check_or_action"
                        ),

                    "source_issue_keys":
                        " | ".join(
                            cause.get(
                                "source_issue_keys",
                                []
                            )
                        ),
                }
            )

    entries_df = pd.DataFrame(
        entry_rows
    )

    causes_df = pd.DataFrame(
        cause_rows
    )

    summary_df = pd.DataFrame(
        [
            {
                "metric":
                    "entry_count",
                "value":
                    len(entries_df),
            },
            {
                "metric":
                    "total_cause_count",
                "value":
                    len(causes_df),
            },
            {
                "metric":
                    "avg_causes_per_entry",
                "value":
                    (
                        len(causes_df)
                        / len(entries_df)
                        if len(entries_df)
                        else 0
                    ),
            },
            {
                "metric":
                    "confidence_ge_090",
                "value":
                    int(
                        (
                            entries_df[
                                "knowledge_confidence"
                            ]
                            >= 0.9
                        ).sum()
                    ),
            },
            {
                "metric":
                    "confidence_lt_080",
                "value":
                    int(
                        (
                            entries_df[
                                "knowledge_confidence"
                            ]
                            < 0.8
                        ).sum()
                    ),
            },
        ]
    )

    review_df = entries_df[
        (
            entries_df[
                "knowledge_confidence"
            ]
            < 0.85
        )
        |
        (
            entries_df[
                "cause_count"
            ]
            < 2
        )
    ].copy()

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        entries_df.to_excel(
            writer,
            sheet_name="entries",
            index=False,
        )

        causes_df.to_excel(
            writer,
            sheet_name="causes",
            index=False,
        )

        review_df.to_excel(
            writer,
            sheet_name="review",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 10.1 - Generate multiple-causes KB"
    )
    print("=" * 70)

    validation_df = pd.read_excel(
        VALIDATION_FILE,
        sheet_name="all_clusters",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    target_df = validation_df[
        (
            validation_df[
                "cluster_validity"
            ]
            == "valid"
        )
        &
        (
            validation_df[
                "answer_relation"
            ]
            == "multiple_causes"
        )
    ].copy()

    print(
        f"multiple_causes cluster count: "
        f"{len(target_df)}"
    )

    processed = load_processed()

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for _, cluster_row in (
        target_df.iterrows()
    ):

        cluster_id = clean_text(
            cluster_row.get(
                "cluster_id"
            )
        )

        if cluster_id in processed:
            continue

        members = (
            member_df[
                member_df[
                    "cluster_id"
                ]
                .astype(str)
                == cluster_id
            ]
            .to_dict(
                orient="records"
            )
        )

        if not members:
            continue

        tasks.append(
            (
                cluster_row,
                members,
            )
        )

    print(
        f"本次待处理: "
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
                        generate_entry,
                        cluster_row,
                        members,
                    ):
                    clean_text(
                        cluster_row.get(
                            "cluster_id"
                        )
                    )

                    for (
                        cluster_row,
                        members,
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