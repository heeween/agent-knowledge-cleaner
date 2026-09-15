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
    / "kb_entries_safe.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "kb_entries_safe.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


SAFE_RELATIONS = {
    "duplicate",
    "complementary",
    "incomplete_vs_complete",
}


# ============================================================
# Schema
# ============================================================

class KnowledgeEntry(BaseModel):

    canonical_question: str

    answer: str

    solution_steps: list[str]

    notes: list[str]

    crm_module: str

    crm_feature: str

    problem_type: str

    knowledge_confidence: float

    source_issue_keys: list[str]


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在把真实 CRM 客服聊天整理成最终可用于 RAG 的知识条目。

当前 Cluster 已经经过：
- 问题抽取
- question_normalized
- embedding
- same_intent 判断
- Cluster 验证

所以这些 issue 已确认属于同一个知识问题。

你的任务是：

根据 Cluster 中已有的 question、answer、solution，
整理出一条清晰、可靠、可复用的 CRM 知识条目。


============================================================
最重要规则
============================================================

1. 只能使用输入材料中明确存在的信息。

禁止：
- 自己补充产品规则
- 根据常识猜测
- 发明操作路径
- 发明菜单名称
- 发明限制条件
- 发明版本状态

如果材料没有明确支持某个结论，就不要写。


2. 不要简单复制某一条客服回复。

你要做的是“知识归纳”。

例如输入：

A：
在账号管理里重置。

B：
管理员进入员工账号页面后可以重置密码。

最终可以整理为：

管理员可在员工账号管理页面中重置账号密码。


3. 如果部分来源答案不完整：

例如：
- 我看看
- 已反馈
- 等技术处理
- 稍后回复

而其他来源有明确完整答案，

优先使用明确、完整的答案。

不要把“正在排查”写进最终知识，
除非它是理解当前产品状态所必需的信息。


4. duplicate

如果多条回答本质相同：

去重并生成最清晰的一版。


5. complementary

如果多条回答分别提供不同部分：

可以融合。

例如：

来源 A：
入口在客户管理。

来源 B：
点击客户详情后选择修改。

可以融合为：

进入客户管理，打开客户详情后选择修改。


6. incomplete_vs_complete

优先保留：
- 明确
- 已解决
- 可执行
- 信息完整

的答案。

不要因为某条记录没有结论而降低完整答案的质量。


============================================================
canonical_question
============================================================

使用 Cluster 验证阶段给出的 canonical_question 为主要依据。

可以轻微润色，但不要改变问题意图。

要求：

- 独立成句
- 简洁
- 可搜索
- 不包含客户名、门店、人名
- 适合作为 RAG 检索问题


============================================================
answer
============================================================

answer 是最终面向用户的知识回答。

要求：

- 直接回答问题
- 简洁
- 不写客服口吻
- 不出现“这边”“亲”“我帮您看一下”
- 不出现聊天过程
- 不出现无法验证的推测
- 不机械堆砌多条来源


============================================================
solution_steps
============================================================

只有材料中存在明确操作步骤时才填写。

例如：

[
  "进入客户管理",
  "打开目标客户详情",
  "点击修改",
  "保存"
]

如果没有明确的步骤：

[]

不要把解释性文字硬拆成操作步骤。


============================================================
notes
============================================================

用于保留必要但不适合放进主答案的注意事项。

例如：

- 权限限制
- 特定前置条件
- 某字段不能直接修改
- 需要管理员操作
- 明确存在的产品限制

如果没有：

[]


============================================================
crm_module / crm_feature / problem_type
============================================================

优先根据输入 issue 的已有分类整理。

不要创建输入材料完全不存在的新业务模块。


============================================================
knowledge_confidence
============================================================

0~1。

综合考虑：

- 多个来源是否一致
- 答案是否清晰
- 是否已解决
- 是否可复用
- 是否存在不完整来源

duplicate 且多个来源一致：
通常可以较高。

complementary：
如果可以无冲突融合，也可以较高。

如果来源证据仍然薄弱：
应降低。


============================================================
source_issue_keys
============================================================

必须保留所有被实际用于最终答案的 issue_key。

如果某条来源：
- 完全没有答案
- 只是“我看看”
- 没有贡献任何知识

可以不放进 source_issue_keys。


============================================================
输出要求
============================================================

只输出合法 JSON object。

不要 Markdown。
不要 ```json。
不要输出 JSON 之外的文字。

JSON 格式：

{
  "canonical_question": "问题",
  "answer": "最终知识答案",
  "solution_steps": [],
  "notes": [],
  "crm_module": "模块",
  "crm_feature": "功能",
  "problem_type": "类型",
  "knowledge_confidence": 0.0,
  "source_issue_keys": []
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

def build_user_prompt(
    cluster_id,
    validation_row,
    members,
):

    blocks = []

    for idx, row in enumerate(
        members,
        start=1,
    ):

        blocks.append(
            f"""
------------------------------
Issue {idx}

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

crm_module:
{clean_text(row.get("crm_module"))}

crm_feature:
{clean_text(row.get("crm_feature"))}

problem_type:
{clean_text(row.get("problem_type"))}

resolution:
{clean_text(row.get("resolution"))}

knowledge_value:
{clean_text(row.get("knowledge_value"))}

temporal_status:
{clean_text(row.get("temporal_status"))}
""".strip()
        )

    return f"""
请生成下面 Cluster 的最终 CRM 知识条目。

cluster_id:
{cluster_id}

已确认 canonical_question:
{clean_text(validation_row.get("canonical_question"))}

answer_relation:
{clean_text(validation_row.get("answer_relation"))}

Cluster members:

{chr(10).join(blocks)}
"""


# ============================================================
# API
# ============================================================

def generate_entry(
    cluster_id,
    validation_row,
    members,
):

    client = build_client()

    prompt = build_user_prompt(
        cluster_id,
        validation_row,
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
                KnowledgeEntry
                .model_validate(
                    data
                )
            )

            member_keys = {
                clean_text(
                    x.get("issue_key")
                )
                for x in members
            }

            # 防止模型乱造 source key
            result.source_issue_keys = [
                x
                for x in result.source_issue_keys
                if x in member_keys
            ]

            return {
                "cluster_id":
                    cluster_id,

                "answer_relation":
                    clean_text(
                        validation_row.get(
                            "answer_relation"
                        )
                    ),

                "canonical_question":
                    result.canonical_question,

                "answer":
                    result.answer,

                "solution_steps":
                    result.solution_steps,

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

                "source_issue_keys":
                    result.source_issue_keys,

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
# Excel Export
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

    for col in [
        "solution_steps",
        "notes",
        "source_issue_keys",
    ]:

        df[col] = df[col].apply(
            lambda x:
                " | ".join(
                    str(v)
                    for v in x
                )
                if isinstance(x, list)
                else clean_text(x)
        )

    relation_stats = (
        df[
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

    confidence_stats = pd.DataFrame(
        [
            {
                "metric":
                    "entry_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "avg_knowledge_confidence",
                "value":
                    df[
                        "knowledge_confidence"
                    ].mean(),
            },
            {
                "metric":
                    "confidence_ge_090",
                "value":
                    int(
                        (
                            df[
                                "knowledge_confidence"
                            ]
                            >= 0.90
                        ).sum()
                    ),
            },
            {
                "metric":
                    "confidence_lt_080",
                "value":
                    int(
                        (
                            df[
                                "knowledge_confidence"
                            ]
                            < 0.80
                        ).sum()
                    ),
            },
        ]
    )

    review_df = df[
        (
            df[
                "knowledge_confidence"
            ]
            < 0.85
        )
        |
        (
            df[
                "answer"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )
    ].copy()

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        confidence_stats.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        relation_stats.to_excel(
            writer,
            sheet_name="relation_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="kb_entries",
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
        "Step 9.1 - Generate safe KB entries"
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

    safe_df = validation_df[
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
            ].isin(
                SAFE_RELATIONS
            )
        )
    ].copy()

    print(
        f"Safe cluster count: "
        f"{len(safe_df)}"
    )

    processed = load_processed()

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for _, validation_row in (
        safe_df.iterrows()
    ):

        cluster_id = clean_text(
            validation_row[
                "cluster_id"
            ]
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
                cluster_id,
                validation_row,
                members,
            )
        )

    print(
        f"本次待处理: "
        f"{len(tasks)}"
    )

    if not tasks:

        export_excel()
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
                    generate_entry,
                    cluster_id,
                    validation_row,
                    members,
                ):
                cluster_id

                for (
                    cluster_id,
                    validation_row,
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