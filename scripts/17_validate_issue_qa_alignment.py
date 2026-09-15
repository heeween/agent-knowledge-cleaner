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
    / "issue_qa_alignments.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "issue_qa_alignments.xlsx"
)

MODEL = "qwen3.5-flash"

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class AlignmentResult(BaseModel):

    alignment: Literal[
        "aligned",
        "partially_aligned",
        "misaligned",
        "no_answer",
        "uncertain",
    ]

    confidence: float

    answer_target: str | None = None

    mismatch_reason: str | None = None

    usable_for_kb: bool


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在检查 CRM 客服聊天中已经抽取出来的单个 issue。

每个 issue 包含：

- question
- question_normalized
- description
- answer
- solution

你的任务非常单一：

判断 answer / solution 是否真的在回答该 issue 的 question。

你不是判断答案是否正确，
也不是判断产品规则是否最新。

只判断：

“问的这个问题，和给出的回答是不是同一件事？”


============================================================
alignment
============================================================

aligned

answer / solution 明确在回答 question。

允许：
- 回答不够完整
- 回答比较简短
- 只有一个解决办法
- 只解释部分原因

只要核心回答对象和问题一致即可。


例如：

问题：
短信发送失败是否扣费？

答案：
失败短信不扣费。

→ aligned


例如：

问题：
CRM 登录不上怎么办？

答案：
先重置密码，再重新登录。

→ aligned


============================================================

partially_aligned

answer 与 question 有明确关系，
但只回答了其中一部分，
或者混入了另一个问题的内容。

例如：

问题：
短信发送失败的原因及处理方法是什么？

答案：
短信签名没有报备。

只回答了“原因”，
没有回答处理方法。

→ partially_aligned


============================================================

misaligned

answer 实际在回答另一个问题。

例如：

问题：
车辆年审提醒为什么没显示？

答案：
如果忘记账号密码，
联系管理员重置密码。

→ misaligned


例如：

问题：
如何购买短信套餐？

答案：
发送失败的短信不会扣费。

→ misaligned


这种情况通常说明：

- issue 边界切错
- 相邻聊天串线
- question 与 answer 来自不同事件
- 抽取模型错误关联


============================================================

no_answer

问题存在，
但没有实质回答。

例如：

答案只有：

- 我看一下
- 稍等
- 已反馈
- 技术处理中
- 晚点回复

且没有实际知识结论。

→ no_answer


============================================================

uncertain

仅凭现有文本无法可靠判断。


============================================================
usable_for_kb
============================================================

aligned：

通常 true。


partially_aligned：

如果 answer 中确实包含直接回答 question 的有效知识，
可以 true。

如果混入内容过多，
则 false。


misaligned：

必须 false。


no_answer：

必须 false。


uncertain：

默认 false。


============================================================
answer_target
============================================================

用一句非常短的话描述：

“这个 answer 实际回答的是什么问题？”

例如：

问题：
车辆年审提醒未显示怎么办？

答案：
管理员重置账号密码。

answer_target：

“账号忘记密码后的重置方法”


如果 aligned：

可以直接概括 question 的目标。


============================================================
mismatch_reason
============================================================

aligned 时：

null

其他情况简短说明问题。

例如：

“问题询问车辆年审提醒，答案实际讨论账号密码重置。”


============================================================
重要规则
============================================================

1. 不要使用 CRM 常识补全。

2. 不要因为 question 和 answer
   都属于同一个 crm_module 就判 aligned。

3. 判断必须具体到：

- 对象
- 动作
- 故障现象
- 用户真正想解决的问题

4. “账号”
和
“车辆年审”

明显不是同一问题。

5. “短信失败原因”
和
“短信失败是否扣费”

也不是同一问题。

6. 如果 answer 只说“我看看”
但 solution 字段有明确解决办法，
仍然要结合 solution 一起判断。


============================================================
输出要求
============================================================

只输出合法 JSON object。

不要 Markdown。
不要 JSON 代码块。
不要输出额外文字。

格式：

{
  "alignment": "aligned | partially_aligned | misaligned | no_answer | uncertain",
  "confidence": 0.0,
  "answer_target": "答案实际回答的问题或 null",
  "mismatch_reason": "原因或 null",
  "usable_for_kb": true
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

                issue_key = obj.get(
                    "issue_key"
                )

                if issue_key:
                    processed.add(
                        issue_key
                    )

            except Exception:
                continue

    return processed


# ============================================================
# Prompt
# ============================================================

def build_prompt(row):

    return f"""
请检查下面这个抽取 issue 内部的问答对应关系。

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


# ============================================================
# API
# ============================================================

def validate_issue(row):

    client = build_client()

    prompt = build_prompt(
        row
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
                AlignmentResult
                .model_validate(
                    data
                )
            )

            # 强制规则
            if result.alignment in {
                "misaligned",
                "no_answer",
                "uncertain",
            }:
                result.usable_for_kb = False

            if (
                result.alignment
                == "aligned"
            ):
                result.mismatch_reason = None

            return {
                "issue_key":
                    clean_text(
                        row.get(
                            "issue_key"
                        )
                    ),

                "cluster_id":
                    clean_text(
                        row.get(
                            "cluster_id"
                        )
                    ),

                "question":
                    clean_text(
                        row.get(
                            "question"
                        )
                    ),

                "question_normalized":
                    clean_text(
                        row.get(
                            "question_normalized"
                        )
                    ),

                "answer":
                    clean_text(
                        row.get(
                            "answer"
                        )
                    ),

                "solution":
                    clean_text(
                        row.get(
                            "solution"
                        )
                    ),

                "resolution":
                    clean_text(
                        row.get(
                            "resolution"
                        )
                    ),

                "knowledge_value":
                    clean_text(
                        row.get(
                            "knowledge_value"
                        )
                    ),

                "temporal_status":
                    clean_text(
                        row.get(
                            "temporal_status"
                        )
                    ),

                "alignment":
                    result.alignment,

                "alignment_confidence":
                    result.confidence,

                "answer_target":
                    result.answer_target,

                "mismatch_reason":
                    result.mismatch_reason,

                "usable_for_kb":
                    result.usable_for_kb,

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

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return

    alignment_stats = (
        df[
            "alignment"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "alignment"
        )
        .reset_index(
            name="count"
        )
    )

    unusable_df = df[
        (
            df[
                "usable_for_kb"
            ]
            != True
        )
        |
        (
            df[
                "alignment"
            ]
            .isin(
                [
                    "misaligned",
                    "no_answer",
                    "uncertain",
                ]
            )
        )
        |
        (
            df[
                "alignment_confidence"
            ]
            < 0.90
        )
    ].copy()

    # 哪些 cluster 受到污染
    bad_cluster_ids = set(
        unusable_df[
            "cluster_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    affected_df = df[
        df[
            "cluster_id"
        ].isin(
            bad_cluster_ids
        )
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "validated_issue_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "aligned_count",
                "value":
                    int(
                        (
                            df[
                                "alignment"
                            ]
                            == "aligned"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "partially_aligned_count",
                "value":
                    int(
                        (
                            df[
                                "alignment"
                            ]
                            == "partially_aligned"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "misaligned_count",
                "value":
                    int(
                        (
                            df[
                                "alignment"
                            ]
                            == "misaligned"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "no_answer_count",
                "value":
                    int(
                        (
                            df[
                                "alignment"
                            ]
                            == "no_answer"
                        ).sum()
                    ),
            },
            {
                "metric":
                    "unusable_issue_count",
                "value":
                    int(
                        (
                            df[
                                "usable_for_kb"
                            ]
                            != True
                        ).sum()
                    ),
            },
            {
                "metric":
                    "affected_cluster_count",
                "value":
                    len(
                        bad_cluster_ids
                    ),
            },
        ]
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        alignment_stats.to_excel(
            writer,
            sheet_name="alignment_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_issues",
            index=False,
        )

        unusable_df.to_excel(
            writer,
            sheet_name="review_issues",
            index=False,
        )

        affected_df.to_excel(
            writer,
            sheet_name="affected_clusters",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 9.3 - Validate issue QA alignment"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 只审核 Step 9.1 实际使用到的 Cluster
    # --------------------------------------------------------

    kb_df = pd.read_excel(
        KB_FILE,
        sheet_name="kb_entries",
    )

    member_df = pd.read_excel(
        CLUSTER_FILE,
        sheet_name="cluster_members",
    )

    cluster_ids = set(
        kb_df[
            "cluster_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    target_df = member_df[
        member_df[
            "cluster_id"
        ]
        .astype(str)
        .isin(
            cluster_ids
        )
    ].copy()

    # 防止重复 issue
    target_df = (
        target_df
        .drop_duplicates(
            subset=[
                "issue_key"
            ]
        )
    )

    print(
        f"Target cluster count: "
        f"{len(cluster_ids)}"
    )

    print(
        f"Target issue count: "
        f"{len(target_df)}"
    )

    processed = (
        load_processed()
    )

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for _, row in (
        target_df.iterrows()
    ):

        issue_key = clean_text(
            row.get(
                "issue_key"
            )
        )

        if issue_key in processed:
            continue

        tasks.append(
            row
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

    started_at = time.time()

    lock = threading.Lock()

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    validate_issue,
                    row,
                ):
                clean_text(
                    row.get(
                        "issue_key"
                    )
                )

                for row in tasks
            }

            for future in as_completed(
                future_map
            ):

                issue_key = (
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
                        f"{issue_key}"
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
                    completed % 20 == 0
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